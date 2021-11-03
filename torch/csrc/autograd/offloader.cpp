
#include <THC/THCCachingHostAllocator.h>
#include <c10/cuda/CUDACachingAllocator.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/csrc/autograd/command_dispatcher.h>

#include <map>

namespace torch {
namespace autograd {
namespace profiler {

void deleteCallback(void* the_) {
  DeleteTensorInfo* the = static_cast<DeleteTensorInfo*>(the_);
  std::cerr << "Delete " << the->old_ptr.device() << " " << the->old_ptr.get()
            << "\n";
  // BUG: possible call cudaFree in cudaLaunchKernel cause error
  if (the->old_ptr) {
    auto to_remove = std::move(the->old_ptr);
  } else {
    std::cerr << "Try to delete nullptr \n";
  }
  delete the;
}

std::map<const void*, CUDAEventStub> offloadFinishEvent;
std::set<c10::StorageImpl*> offloadStorages;

void copyBytesCallBack(
    size_t nbytes,
    const void* src,
    c10::Device src_device,
    void* dst,
    c10::Device dst_device) {
  if (src_device.type() == c10::DeviceType::CPU &&
      dst_device.type() == c10::DeviceType::CUDA) {
    auto it = offloadFinishEvent.find(src);
    if (it == offloadFinishEvent.end()) {
      std::cerr << "Cannot find this tensor in offload tensors.\n";
    } else {
      cudaStubs()->streamWaitEvent(it->second, prefetch_stream);
      std::cerr << "Prefetch " << dst << " event " << it->second.get() << "\n";
      cudaStubs()->memcpyAsync(
          dst, src, nbytes, cudaMemcpyHostToDevice, prefetch_stream);
      offloadFinishEvent.erase(it);
    }

  } else if (
      src_device.type() == c10::DeviceType::CUDA &&
      dst_device.type() == c10::DeviceType::CPU) {
    cudaStubs()->memcpyAsync(
        dst, src, nbytes, cudaMemcpyDeviceToHost, offload_stream);
    auto event = cudaStubs()->registerStreamEvent(offload_stream);
    std::cerr << "Offload " << dst << " event " << event.get() << "\n";
    offloadFinishEvent.insert(make_pair(dst, event));
  }
}

void Offloader::prefetch(const at::RecordFunction& fn, int kidx) {
  int swap_in_tensor_count = 0;

  auto swap_in = [&](at::Tensor& tensor) {
    auto original_data_ptr = tensor.data_ptr();
    auto storage_impl_ = tensor.storage().unsafeGetStorageImpl();

    if (storage_impl_->is_swapped_out_) {
      swap_in_tensor_count += 1;

      DeleteTensorInfo* old = new DeleteTensorInfo();
      old->old_ptr = std::move(storage_impl_->swap_in());
      cudaStubs()->insertHostFunction(
          deleteCallback, (void*)old, prefetch_stream);

      offloadStorages.erase(storage_impl_);

      // std::cerr << "Prefetch " << original_data_ptr << " to "
      //           << tensor.data_ptr() << " " << tensor.storage().device()
      //           << " size=" << tensor.storage().nbytes() << "\n";
    }
  };

  auto inputs = fn.inputs();
  for (int i = 0; i < inputs.size(); i++) {
    // std::cerr << "Inputs " << inputs[i].type()->annotation_str() << "\n";
    if (inputs[i].isTensor()) {
      at::Tensor& tensor = inputs[i].toTensor();
      if (tensor.defined() && tensor.has_storage()) {
        swap_in(tensor);
      }
    } else if (inputs[i].isTensorList()) {
      c10::List<at::Tensor>&& tensorList = inputs[i].toTensorList();
      for (at::Tensor tensor : tensorList) {
        if (tensor.defined() && tensor.has_storage()) {
          swap_in(tensor);
        }
      }
    }
  }

  auto outputs = fn.outputs();
  for (int i = 0; i < outputs.size(); i++) {
    if (!outputs[i].isTensor())
      continue;

    at::Tensor& tensor = outputs[i].toTensor();
    if (tensor.defined() && tensor.has_storage()) {
      swap_in(tensor);
    }
  }

  if (swap_in_tensor_count > 0) {
    std::cerr << "Compute Wait for prefetch stream " << fn.name() << "\n";
    auto prefetch_finish_event =
        cudaStubs()->registerStreamEvent(prefetch_stream);
    cudaStubs()->streamWaitEvent(prefetch_finish_event, compute_stream);
  }
}

} // namespace profiler
} // namespace autograd
} // namespace torch
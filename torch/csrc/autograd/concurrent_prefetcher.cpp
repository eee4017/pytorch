#include <c10/cuda/CUDACachingAllocator.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/csrc/autograd/command_dispatcher.h>

#include <thread>

namespace torch {
namespace autograd {
namespace profiler {

void Offloader::prefetchThread() {
  while (true) {
    auto dependency = prefetch_command_queue.pop();

    std::cerr << "[Thread] Pop from prefetch_command_queue " << dependency.kidx
              << "\n";
    if (dependency.kidx < 0)
      break;

    std::vector<c10::DataPtr> old_ptrs(dependency.prefetch_blocks.size()); 
    for (int i = 0;i < dependency.prefetch_blocks.size(); i++) {
      auto block = dependency.prefetch_blocks[i];
      auto original_data_ptr = block->data();

      // DeleteTensorInfo* old = new DeleteTensorInfo();
      old_ptrs[i] = std::move(block->swap_in(prefetch_stream.get()));

      std::cerr << "[thread] Prefetch " << original_data_ptr << " to " << block->data()
                << " size=" << block->nbytes() << "\n";
    }

    auto prefetch_finish_event =
        cudaStubs()->registerStreamEvent(prefetch_stream);
    cudaStubs()->streamWaitEvent(prefetch_finish_event, compute_stream);
    cudaStubs()->eventSynchronize(prefetch_finish_event);
  }
}

#ifdef CONCURRENT_PREFETECHER
Offloader::Offloader() {
  prefetch_thread = std::thread(&Offloader::prefetchThread, this);
}

Offloader::~Offloader() {
  std::cerr << "offloadFinishEvent.size() = " << offloadFinishEvent.size()
            << "\n";
  offloadFinishEvent.clear();
  prefetch_command_queue.push(KernelDependencies(-1));
  prefetch_thread.join();
  std::cerr << "comandDispatcherFinalizer joined\n";
}

void Offloader::prefetch(const at::RecordFunction& fn, int kidx) {
  int swap_in_tensor_count = 0;
  KernelDependencies dependency(kidx);

  auto swap_in = [&](at::Tensor& tensor) {
    auto original_data_ptr = tensor.data_ptr();
    auto storage_impl_ = tensor.storage().unsafeGetStorageImpl();

    if (storage_impl_->is_swapped_out_) {
      swap_in_tensor_count += 1;
      dependency.prefetch_blocks.push_back(storage_impl_);

      offloadStorages.erase(storage_impl_);
    }
  };

  auto inputs = fn.inputs();
  for (int i = 0; i < inputs.size(); i++) {
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
    std::cerr << "[" << kidx << "]"
              << " push to prefetch_command_queue"
              << "\n";
    prefetch_command_queue.push(dependency);
  }
}
#endif

} // namespace profiler
} // namespace autograd
} // namespace torch
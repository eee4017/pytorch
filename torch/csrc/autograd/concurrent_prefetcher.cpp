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

    std::cerr << "[Thread]recv from prefetch_command_queue " << dependency.kidx << "\n";
    if (dependency.kidx < 0)
      break;

    for (auto block : dependency.prefetch_blocks) {
      auto original_data_ptr = block->data();

      DeleteTensorInfo* old = new DeleteTensorInfo();
      old->old_ptr = std::move(block->swap_in());
      cudaStubs()->insertHostFunction(
          deleteCallback, (void*)old, prefetch_stream);

      std::cerr << "[Thread]Prefetch " << original_data_ptr << " to "
                << block->data() << " " << block->device()
                << " size=" << block->nbytes() << "\n";
    }

    auto prefetch_finish_event =
        cudaStubs()->registerStreamEvent(prefetch_stream);
    cudaStubs()->streamWaitEvent(prefetch_finish_event, compute_stream);
    cudaStubs()->eventSynchronize(prefetch_finish_event);
  }
}
/*
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
*/

} // namespace profiler
} // namespace autograd
} // namespace torch
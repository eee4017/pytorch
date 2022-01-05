
#include <torch/csrc/autograd/command_dispatcher.h>

namespace torch {
namespace autograd {
namespace profiler {


void NaiveOffloader::offload(const at::RecordFunction& fn, int kidx) {
  int swap_out_tensor_count = 0;

  auto swap_out = [&](at::Tensor& tensor) {
    auto original_data_ptr = tensor.data_ptr();

    auto storage_impl_ = tensor.storage().unsafeGetStorageImpl();
    storage_impl_->lms_unpin();
    // if (!storage_impl_->is_swapped_out_) {
    //   swap_out_tensor_count += 1;

    //   if (swap_out_tensor_count == 1) {
    //     auto compute_stream_event = cudaStubs()->registerComputeStreamEvent();
    //     cudaStubs()->streamWaitEvent(compute_stream_event, offload_stream);
    //   }

    //   DeleteTensorInfo* old = new DeleteTensorInfo();
    //   old->old_ptr = std::move(storage_impl_->swap_out(
    //       c10::Device(c10::DeviceType::CPU, 0), copyBytesCallBack, offload_stream.get()));
    //   // cudaStubs()->insertHostFunction(
    //   //     deleteCallback, (void*)old, offload_stream);
    //   offloadStorages.insert(storage_impl_);

    //   std::cerr << "Offload " << original_data_ptr << " to "
    //             << tensor.data_ptr() << " " << tensor.storage().device()
    //             << " size=" << tensor.storage().nbytes() << "\n";
    // };
  };

  auto inputs = fn.inputs();
  for (int i = 0; i < inputs.size(); i++) {
    if (!inputs[i].isTensor())
      continue;
    at::Tensor& tensor = inputs[i].toTensor();
    if (tensor.defined() && tensor.has_storage() &&
        tensor.storage().nbytes() > 1e8) {
      swap_out(tensor);
    }
  }
}


}
} // namespace autograd
} // namespace torch
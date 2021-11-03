
#include <torch/csrc/autograd/command_dispatcher.h>

namespace torch {
namespace autograd {
namespace profiler {

void ScheduledOffloader::parseSchedulerResults() {
  std::ifstream ifs("/root/share/scheduling_results.txt");

  std::cerr << "start to parse scheduler results from /root/share/scheduling_results.txt\n";
  std::string line;
  while (getline(ifs, line)) {
    std::stringstream ss;
    ss << line;
    std::string kidx_s, name;
    ss >> kidx_s;
    int kidx = std::stoi(kidx_s);
    ss >> name;
    scheduling_name[kidx] = name;
    std::string sch;
    while (ss >> sch) {
      scheduling_results[kidx].push_back(sch[0] - '0');
      scheduling_results_size[kidx].push_back(
          std::stoi(sch.substr(2, sch.length())));
    }
  }
}

ScheduledOffloader::ScheduledOffloader() {
  parseSchedulerResults();
}

void ScheduledOffloader::offload(const at::RecordFunction& fn, int kidx) {
  int swap_out_tensor_count = 0;

  auto swap_out = [&](at::Tensor& tensor, int tidx) {
    auto original_data_ptr = tensor.data_ptr();

    auto storage_impl_ = tensor.storage().unsafeGetStorageImpl();
    if (!storage_impl_->is_swapped_out_ &&
        tidx < scheduling_results[kidx].size() &&
        scheduling_results[kidx][tidx] == 1) {
      swap_out_tensor_count += 1;

      if (swap_out_tensor_count == 1) {
        auto compute_stream_event = cudaStubs()->registerComputeStreamEvent();
        cudaStubs()->streamWaitEvent(compute_stream_event, offload_stream);
      }

      DeleteTensorInfo* old = new DeleteTensorInfo();
      old->old_ptr = std::move(storage_impl_->swap_out(
          c10::Device(c10::DeviceType::CPU, 0), copyBytesCallBack));
      cudaStubs()->insertHostFunction(
          deleteCallback, (void*)old, offload_stream);
      offloadStorages.insert(storage_impl_);

      // std::cerr << "Offload " << original_data_ptr << " to "
      //           << tensor.data_ptr() << " " << tensor.storage().device()
      //           << " size=" << tensor.storage().nbytes()
      //           << " scheduling_results_size="
      //           << scheduling_results_size[kidx][tidx] << "\n";
    };
  };

  auto inputs = fn.inputs();
  for (int i = 0; i < inputs.size(); i++) {
    if (!inputs[i].isTensor())
      continue;
    at::Tensor& tensor = inputs[i].toTensor();
    if (tensor.defined() && tensor.has_storage() &&
        tensor.storage().nbytes() > MEMORY_BLOCK_SIZE_THRESHOLD) {
      swap_out(tensor, i);
    }
  }

}

} // namespace profiler
} // namespace autograd
} // namespace torch
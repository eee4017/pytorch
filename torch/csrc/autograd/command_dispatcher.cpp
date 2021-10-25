#include <c10/cuda/CUDAGuard.h>

#include <THC/THCCachingHostAllocator.h>

#include <torch/csrc/autograd/command_dispatcher.h>
#include <torch/csrc/jit/frontend/tracer.h>
#include <torch/csrc/jit/runtime/operator.h>

#include <chrono>
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>

#define MEMORY_BLOCK_SIZE_THRESHOLD 1e4

namespace torch {
namespace autograd {
namespace profiler {

CUDAStreamStub compute_stream;
CUDAStreamStub offload_stream;
CUDAStreamStub prefetch_stream;

std::map<const void*, CUDAEventStub> offloadTensors;

void copyBytesCallBack(
    size_t nbytes,
    const void* src,
    c10::Device src_device,
    void* dst,
    c10::Device dst_device) {
  if (src_device.type() == c10::DeviceType::CPU &&
      dst_device.type() == c10::DeviceType::CUDA) {
    auto it = offloadTensors.find(src);
    if (it == offloadTensors.end()) {
      std::cerr << "Cannot find this tensor in offload tensors.\n";
    } else {
      cudaStubs()->streamWaitEvent(it->second, prefetch_stream);
      cudaStubs()->memcpyAsync(
          dst, src, nbytes, cudaMemcpyHostToDevice, prefetch_stream);
      offloadTensors.erase(it);
    }

  } else if (
      src_device.type() == c10::DeviceType::CUDA &&
      dst_device.type() == c10::DeviceType::CPU) {
    cudaStubs()->memcpyAsync(
        dst, src, nbytes, cudaMemcpyDeviceToHost, offload_stream);
    auto event = cudaStubs()->registerStreamEvent(offload_stream);
    offloadTensors.insert(make_pair(dst, event));
  }
}

struct DeleteTensorInfo {
  at::DataPtr old_ptr;
};

void deleteCallback(void* the_) {
  DeleteTensorInfo* the = static_cast<DeleteTensorInfo*>(the_);

  if (the->old_ptr) {
    auto target = std::move(the->old_ptr);
  } else {
    std::cerr << "Try to delete nullptr \n";
  }
  delete the;
}

void prefetch(const at::RecordFunction& fn) {
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

      std::cerr << "Prefetch " << original_data_ptr << " to "
                << tensor.data_ptr() << " " << tensor.storage().device()
                << " size=" << tensor.storage().nbytes() << "\n";
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
  // std::cerr << "Outputs " << outputs.size() << "\n";
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

std::vector<std::string> scheduling_name;
std::vector<std::vector<int>> scheduling_results;
std::vector<std::vector<int>> scheduling_results_size;

void parseSchedulerResults() {
  std::ifstream ifs("/root/share/scheduling_results.txt");

  std::string line;
  while (getline(ifs, line)) {
    scheduling_results.push_back(std::vector<int>());
    scheduling_results_size.push_back(std::vector<int>());
    std::stringstream ss;
    ss << line;
    std::string name;
    ss >> name;
    scheduling_name.push_back(name);
    std::string sch;
    while (ss >> sch) {
      scheduling_results.back().push_back(sch[0] - '0');
      scheduling_results_size.back().push_back(
          std::stoi(sch.substr(2, sch.length())));
    }
  }
}

std::ofstream dependencies_ofs;
void exportDependencies(const at::RecordFunction& fn) {
  dependencies_ofs << fn.name();
  for (auto i : fn.inputs()) {
    if (!i.isTensor())
      continue;
    at::Tensor& tensor = i.toTensor();
    if (tensor.defined() && tensor.has_storage() &&
        tensor.storage().nbytes() > 0) {
      dependencies_ofs << "|" << tensor.data_ptr() << " "
                       << tensor.storage().nbytes();
    }
  }
  for (auto i : fn.outputs()) {
    if (!i.isTensor())
      continue;
    at::Tensor& tensor = i.toTensor();
    if (tensor.defined() && tensor.has_storage() &&
        tensor.storage().nbytes() > 0) {
      dependencies_ofs << "|" << tensor.data_ptr() << " "
                       << tensor.storage().nbytes();
    }
  }
  dependencies_ofs << "\n";
}

int kidx = 0;

void offload(const at::RecordFunction& fn) {
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

      std::cerr << "Offload " << original_data_ptr << " to "
                << tensor.data_ptr() << " " << tensor.storage().device()
                << " size=" << tensor.storage().nbytes()
                << " scheduling_results_size="
                << scheduling_results_size[kidx][tidx] << "\n";
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

  std::cerr << "Kernel " << scheduling_name[kidx] << " == " << fn.name()
            << "\n";
  kidx += 1;
}

int step = 0;
int depth = 0;
int in_backward = 0;
int in_optimizer = 0;

void comandDispatcherFinalizer() {
  std::cerr << "comandDispatcherInitializer\n";
  std::cerr << "offloadTensors.size() = " << offloadTensors.size() << "\n";
  offloadTensors.clear();

  std::cerr << "comandDispatcherFinalizer joined\n";
  return;
}

void comandDispatcherInitializer() {  
  compute_stream = cudaStubs()->getComputeStream();
  offload_stream = cudaStubs()->streamCreate();
  prefetch_stream = cudaStubs()->streamCreate();

  std::cerr << "comandDispatcherInitializer\n";
  parseSchedulerResults();
  dependencies_ofs.open("/root/share/dependencies.txt");

  at::Allocator *pinned_memory_allocator = getTHCCachingHostAllocator();
  c10::DataPtr swap_space = pinned_memory_allocator->allocate(1e9);
  swap_space.clear();


  auto handle = at::addGlobalCallback(
      at::RecordFunctionCallback(
          [](const at::RecordFunction& fn)
              -> std::unique_ptr<at::ObserverContext> {
            at::RecordScope scope = fn.scope();
            depth += 1;
            if (fn.scope() == at::RecordScope::BACKWARD_FUNCTION)
              in_backward = 1;
            auto fn_name = std::string(fn.name().str());
            if (fn_name.find("Optimizer") != std::string::npos) {
              in_optimizer = 1;
            }
            int need_depth = (in_backward || in_optimizer);

            auto ctx_ptr = std::make_unique<CommandDispatcherObserverContext>();
            if (depth == 2 + need_depth && step == 1)
              prefetch(fn);
            return ctx_ptr;
          },
          [](const at::RecordFunction& fn, at::ObserverContext* ctx_ptr) {
            int need_depth = (in_backward || in_optimizer);

            if (depth == 2 + need_depth) {
              if (step == 0)
                exportDependencies(fn);

              if (step == 1)
                offload(fn);
            }
            depth -= 1;

            if (fn.scope() == at::RecordScope::BACKWARD_FUNCTION)
              in_backward = 0;
            auto fn_name = std::string(fn.name().str());
            if (fn_name.find("Optimizer") != std::string::npos) {
              in_optimizer = 0;
            }

            if (fn_name.find("ProfilerStep") != std::string::npos) {
              step += 1;
              kidx = 0;
            }
          })
          .needsInputs(true)
          .needsOutputs(true)
          .needsIds(true));
}
} // namespace profiler
} // namespace autograd
} // namespace torch
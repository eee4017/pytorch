#include <c10/cuda/CUDAGuard.h>

#include <torch/csrc/autograd/command_dispatcher.h>
#include <torch/csrc/jit/frontend/tracer.h>
#include <torch/csrc/jit/runtime/operator.h>

#include <iostream>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>

namespace torch {
namespace autograd {
namespace profiler {

const std::map<at::RecordScope, std::string> RecordScopeName = {
    {at::RecordScope::FUNCTION, "FUNCTION"},
    {at::RecordScope::BACKWARD_FUNCTION, "BACKWARD_FUNCTION"},
    {at::RecordScope::TORCHSCRIPT_FUNCTION, "TORCHSCRIPT_FUNCTION"},
    {at::RecordScope::KERNEL_FUNCTION_DTYPE, "KERNEL_FUNCTION_DTYPE"},
    {at::RecordScope::USER_SCOPE, "USER_SCOPE"}};

std::map<void*, c10::Device> offloadTensorsDevice;
std::set<c10::intrusive_ptr<c10::TensorImpl>> offloadTensors;

CUDAStreamStub d2h_stream;
CUDAStreamStub h2d_stream;

struct CommandDispatcherData {
  std::vector<c10::intrusive_ptr<c10::TensorImpl, c10::UndefinedTensorImpl>>
      tensors;
  at::StringView name;

  CommandDispatcherData(const at::StringView& name_) : name(name_){};
};

void copyBytesCallBack(
    size_t nbytes,
    const void* src,
    c10::Device src_device,
    void* dst,
    c10::Device dst_device) {
  std::cerr << "copyBytesCallBack " << src_device.type() << " "
            << dst_device.type() << "\n";
  if (src_device.type() == c10::DeviceType::CPU &&
      dst_device.type() == c10::DeviceType::CUDA) {
    std::cerr << "H2D Memory Copy\n";
    offloadTensorsDevice.erase((void*)src);
  } else if (
      src_device.type() == c10::DeviceType::CUDA &&
      dst_device.type() == c10::DeviceType::CPU) {
    std::cerr << "D2H Memory Copy\n";
    offloadTensorsDevice.insert(std::make_pair(dst, src_device));
  }
}

void returnAllTensors() {
  std::cerr << "returnAllTensors " << offloadTensors.size() << "\n";
  for (auto ptr : offloadTensors) {
    auto device = offloadTensorsDevice.find(ptr->data())->second;
    ptr->scheduleMemcopyAsync(
        c10::Device(c10::DeviceType::CUDA, 0), copyBytesCallBack);
  }
  offloadTensorsDevice.clear();
  offloadTensors.clear();
}

void prefetch(const at::RecordFunction& fn) {
  if (fn.name() == at::StringView("aten::to") ||
      fn.name() == at::StringView("aten::copy_"))
    return;

  // if (std::string(fn.name().str()).find("TBackward") != std::string::npos) {
  //   std::cerr << fn.name() << "\n";
  //   returnAllTensors();
  //   return;
  // }

  auto inputs = fn.inputs();
  auto data = new CommandDispatcherData(fn.name());
  for (int i = 0; i < inputs.size(); i++) {
    if (!inputs[i].isTensor())
      continue;

    at::Tensor& tensor = inputs[i].toTensor();
    if (tensor.defined() && tensor.has_storage() &&
        tensor.storage().nbytes() > 1e6) {
      std::cerr << "Input " << tensor.data_ptr() << " "
                << tensor.device().type() << " " << tensor.storage().nbytes()
                << "\n";
      auto it = offloadTensorsDevice.find(tensor.data_ptr());
      if (it != offloadTensorsDevice.end()) {
        auto ptr = tensor.getIntrusivePtr();
        ptr->scheduleMemcopyAsync(it->second, copyBytesCallBack);
        offloadTensors.erase(ptr);
      }
    }
  }
}

void offload(const at::RecordFunction& fn) {
  if (fn.name() == at::StringView("aten::to") ||
      fn.name() == at::StringView("aten::copy_"))
    return;

  auto outputs = fn.outputs();
  auto data = new CommandDispatcherData(fn.name());

  if (outputs.size() > 0) {
    // auto event = cudaStubs()->registerComputeStreamEvent();
    for (int i = 0; i < outputs.size(); i++) {
      if (!outputs[i].isTensor())
        continue;

      at::Tensor& tensor = outputs[i].toTensor();
      if (tensor.defined() && tensor.device().is_cuda() &&
          tensor.has_storage() && tensor.storage().nbytes() > 1e6) {
        std::cerr << "Output " << tensor.data_ptr() << " "
                  << tensor.device().type() << " " << tensor.storage().nbytes()
                  << "\n";

        // cudaStubs()->issueStreamWaitEvent(event, d2h_stream);
        auto ptr = tensor.getIntrusivePtr();
        ptr->scheduleMemcopyAsync(
            c10::Device(c10::DeviceType::CPU, 0), copyBytesCallBack);
        offloadTensors.insert(ptr);

        // The space is allocated at this point, but the tranfer will happened
        // when the kernel is end.
        std::cerr << "OffloadEnd " << tensor.data_ptr() << " "
                  << tensor.device().type() << " " << tensor.get_device()
                  << "\n";
      }
    }
  }
}

void comandDispatcherFinalizer() {
  returnAllTensors();
  return;
}

void comandDispatcherInitializer() {
  // d2h_stream = cudaStubs()->createStream();
  // h2d_stream = cudaStubs()->createStream();

  auto handle = at::addGlobalCallback(
      at::RecordFunctionCallback(
          [](const at::RecordFunction& fn)
              -> std::unique_ptr<at::ObserverContext> {
            at::RecordScope scope = fn.scope();
            std::cerr << "Before " << fn.name() << " "
                      << RecordScopeName.at(scope) << "\n";
            auto ctx_ptr = std::make_unique<CommandDispatcherObserverContext>();

            prefetch(fn);
            return ctx_ptr;
          },
          [](const at::RecordFunction& fn, at::ObserverContext* ctx_ptr) {
            std::cerr << "After " << fn.name() << "\n";
            offload(fn);
          })
          .needsInputs(true)
          .needsOutputs(true)
          .needsIds(true));
}
} // namespace profiler
} // namespace autograd
} // namespace torch
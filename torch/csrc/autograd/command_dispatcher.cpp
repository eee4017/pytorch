#include <THC/THCCachingHostAllocator.h>
#include <c10/cuda/CUDACachingAllocator.h>
#include <c10/cuda/CUDAGuard.h>

#include <torch/csrc/autograd/command_dispatcher.h>
#include <torch/csrc/jit/frontend/tracer.h>
#include <torch/csrc/jit/runtime/operator.h>

#include <cuda.h>
#include <dlfcn.h>

#include <chrono>
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>

namespace torch {
namespace autograd {
namespace profiler {

std::ofstream dependencies_ofs;
void exportDependencies(const at::RecordFunction& fn, int kidx) {
  dependencies_ofs << kidx << "|" << fn.name();
  for (auto i : fn.inputs()) {
    if (!i.isTensor())
      continue;
    at::Tensor& tensor = i.toTensor();
    if (tensor.defined() && tensor.has_storage() &&
        tensor.storage().nbytes() > 0) {
      dependencies_ofs << "|" << tensor.storage().unsafeGetStorageImpl() << " "
                       << tensor.storage().nbytes();
    }
  }
  for (auto i : fn.outputs()) {
    if (!i.isTensor())
      continue;
    at::Tensor& tensor = i.toTensor();
    if (tensor.defined() && tensor.has_storage() &&
        tensor.storage().nbytes() > 0) {
      dependencies_ofs << "|" << tensor.storage().unsafeGetStorageImpl() << " "
                       << tensor.storage().nbytes();
    }
  }
  dependencies_ofs << std::endl;
}

int kidx = 0;
int step = 0;
int depth = 0;
int in_backward = 0;
int in_optimizer = 0;

enum CommandDispatcherFunctionScope { Forward, Backward, Optimization };
CommandDispatcherFunctionScope scope;

enum CommandDispatcherStatus {
  CommandDispatcherNaiveOffloader,
  CommandDispatcherScheduledOffloader,
};
Offloader* offloader;
Offloader* scheduled_offloader;
Offloader* naive_offloader;

const int SCHEDULER_MAX_STEP = 10;

size_t preallocate_swap_space_size = 0;
int scheduler_enable_step[SCHEDULER_MAX_STEP] = {0};
int dependency_enable_step[SCHEDULER_MAX_STEP] = {0};
CommandDispatcherStatus status;
void getConfigFromEnviron() {
  auto parse_comma_split = [](const char* tmp, int* table) {
    if (tmp) {
      std::stringstream ss(tmp);
      while (ss.good()) {
        std::string substr;
        getline(ss, substr, ',');

        int index = stoi(substr);
        if (0 <= index && index < SCHEDULER_MAX_STEP) {
          table[index] = 1;
        }
      }
    }
  };

  char* tmp;
  parse_comma_split(getenv("SCHEDULER_ENABLE_STEP"), scheduler_enable_step);
  parse_comma_split(getenv("DEPENDECY_ENABLE_STEP"), dependency_enable_step);

  tmp = getenv("SCHEDULER");
  std::string scheduler_str = "scheduled";
  if (tmp) {
    scheduler_str = std::string(tmp);
    if (scheduler_str.compare("naive") == 0) {
      status = CommandDispatcherStatus::CommandDispatcherNaiveOffloader;
    } else {
      status = CommandDispatcherStatus::CommandDispatcherScheduledOffloader;
    }
  }

  tmp = getenv("PREALLOCATE_SWAP_SPACE");
  if (tmp) {
    preallocate_swap_space_size = atoll(tmp);
  } else {
    preallocate_swap_space_size = 3 * 1e9;
  }

  std::cerr << "======= Command Dispatcher =======\nSCHEDULER = "
            << scheduler_str << "\n";
  std::cerr << "PREALLOCATE_SWAP_SPACE = " << preallocate_swap_space_size
            << "\n";
  std::cerr << "SCHEDULER_ENABLE_STEP = ";
  for (int i = 0; i < SCHEDULER_MAX_STEP; i++) {
    std::cerr << scheduler_enable_step[i] << " ";
  }
  std::cerr << "\n";

  std::cerr << "DEPENDENCY_ENABLE_STEP = ";
  for (int i = 0; i < SCHEDULER_MAX_STEP; i++) {
    std::cerr << dependency_enable_step[i] << " ";
  }
  std::cerr << "\n";
  std::cerr << "======= Command Dispatcher =======\n";
}

#define checkCudaErrors(err) __checkCudaErrors(err, __FILE__, __LINE__)

inline void __checkCudaErrors(CUresult err, const char* file, const int line) {
  if (CUDA_SUCCESS != err) {
    fprintf(
        stderr,
        "CUDA Driver API error = %04d from file <%s>, line %i.\n",
        err,
        file,
        line);
    exit(-1);
  }
}

void* libcudaHandle;
cudaMemPool_t default_memory_pool;
typedef decltype(cuMemPoolGetAttribute)* cuMemPoolGetAttributeType;
cuMemPoolGetAttributeType _cuMemPoolGetAttribute;

std::vector<std::pair<uint64_t, int64_t>> memory_usage_log;
void recordMemoryPoolUsage() {
  cuuint64_t usage;
  checkCudaErrors(_cuMemPoolGetAttribute(
      default_memory_pool, CU_MEMPOOL_ATTR_USED_MEM_CURRENT, &usage));
  time_t curr_time =
      std::chrono::system_clock::now().time_since_epoch().count();
  memory_usage_log.push_back(std::make_pair(usage, curr_time));
}

void comandDispatcherFinalizer() {
  dependencies_ofs.close();
  std::ofstream memory_ofs(
      "/root/share/memory_usage.txt", std::ofstream::out | std::ofstream::app);
  for (auto p : memory_usage_log) {
    memory_ofs << p.first << " " << p.second << "\n";
  }
  memory_ofs.close();

  // if(naive_offloader)
  //   delete naive_offloader;
  // if(scheduled_offloader)
  //   delete scheduled_offloader;
  offloader = nullptr;
  cudaStubs()->synchronize();
  return;
}

CUDAStreamStub compute_stream;
CUDAStreamStub offload_stream;
CUDAStreamStub prefetch_stream;

void comandDispatcherInitializer() {
  // int leastPriority;
  // int greatestPriority;
  // cudaDeviceGetStreamPriorityRange(&leastPriority, &greatestPriority);
  // std::cerr << "Stream Priority " << leastPriority << " " << greatestPriority
  //           << "\n";

  compute_stream = cudaStubs()->getComputeStream();
  offload_stream = cudaStubs()->streamCreate(0);
  prefetch_stream = cudaStubs()->streamCreate(0);
  getConfigFromEnviron();

  std::cerr << "comandDispatcherInitializer " << step << "\n";
  libcudaHandle = dlopen("libcuda.so", RTLD_LAZY);
  _cuMemPoolGetAttribute =
      (cuMemPoolGetAttributeType)dlsym(libcudaHandle, "cuMemPoolGetAttribute");
  cudaDeviceGetDefaultMemPool(&default_memory_pool, 0);

  dependencies_ofs.open(
      "/root/share/dependencies.txt", std::ofstream::out | std::ofstream::app);
  if (status == CommandDispatcherStatus::CommandDispatcherScheduledOffloader
  && step > 1) {
    std::cerr << "use ScheduledOffloader\n";
    offloader = new ScheduledOffloader();
  } else {
    std::cerr << "use NaiveOffloader\n";
    offloader = new NaiveOffloader();
  }

  c10::cuda::CUDACachingAllocator::setUserEnabledLMS(true);
  // scheduled_offloader = new ScheduledOffloader();
  // naive_offloader = new NaiveOffloader();

  at::Allocator* pinned_memory_allocator = getTHCCachingHostAllocator();
  c10::DataPtr swap_space =
      pinned_memory_allocator->allocate(preallocate_swap_space_size);
  swap_space.clear();

  scope = CommandDispatcherFunctionScope::Forward;
  kidx = 0;
  auto handle = at::addGlobalCallback(
      at::RecordFunctionCallback(
          [](const at::RecordFunction& fn)
              -> std::unique_ptr<at::ObserverContext> {
            recordMemoryPoolUsage();
            kidx += 1;
            depth += 1;
            if (fn.scope() == at::RecordScope::BACKWARD_FUNCTION) {
              scope = CommandDispatcherFunctionScope::Backward;
            }
            auto name = std::string(fn.name().str());
            if (name.find("Optimizer") != std::string::npos) {
              scope = CommandDispatcherFunctionScope::Optimization;
            }
            if (name.find("ProfilerStep") != std::string::npos) {
              std::cerr << "ProfilerStep " << step << "\n";
              scope = CommandDispatcherFunctionScope::Forward;
              if (step <= 1) {
                std::cerr << "use NaiveOffloader\n";
                offloader = new NaiveOffloader();
              } else {
                std::cerr << "use ScheduledOffloader\n";
                offloader = new ScheduledOffloader();
              }
              kidx = 0;
            }

            auto ctx_ptr = std::make_unique<CommandDispatcherObserverContext>();
            ctx_ptr->need_offload = false;

            // function filter logic
            if (scope == CommandDispatcherFunctionScope::Backward ||
                scope == CommandDispatcherFunctionScope::Optimization) {
              if (depth == 3)
                ctx_ptr->need_offload = true;
            } else {
              if (depth == 2)
                ctx_ptr->need_offload = true;
            }

            if (ctx_ptr->need_offload) {
              offloader->prefetch(fn, kidx);
            }

            return ctx_ptr;
          },
          [](const at::RecordFunction& fn, at::ObserverContext* ctx_ptr_) {
            auto ctx_ptr =
                dynamic_cast<CommandDispatcherObserverContext*>(ctx_ptr_);
            // std::cerr << "after " << kidx << " " << depth << " " << scope <<
            // " "
            //           << step << " " << ctx_ptr->need_offload << " "
            //           << fn.name() << "\n";
            if (ctx_ptr->need_offload && dependency_enable_step[step]) {
              exportDependencies(fn, kidx);
            }

            if (ctx_ptr->need_offload && scheduler_enable_step[step] &&
                offloader) {
              offloader->offload(fn, kidx);
            }

            depth -= 1;
            auto name = std::string(fn.name().str());
            if (name.find("ProfilerStep") != std::string::npos) {
              step += 1;
              delete offloader;
              // unsigned int flag;
              // cudaStreamGetFlags(prefetch_stream.get(), &flag);
              // std::cerr << "cudaStreamGetFlags " << flag << "\n";
              // cudaStreamGetFlags(offload_stream.get(), &flag);
              // std::cerr << "cudaStreamGetFlags " << flag << "\n";

              // cudaStreamSynchronize(prefetch_stream.get());
              // cudaStreamSynchronize(offload_stream.get());
            }
          })
          .needsInputs(true)
          .needsOutputs(true)
          .needsIds(true));
}
} // namespace profiler
} // namespace autograd
} // namespace torch
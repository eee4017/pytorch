#include <torch/csrc/autograd/profiler_legacy.h>
#include <torch/csrc/autograd/concurrent_queue.h>

#include <map>
#include <queue>

// #define CONCURRENT_PREFETECHER

namespace torch {
namespace autograd {
namespace profiler {

#define MEMORY_BLOCK_SIZE_THRESHOLD 1e4

extern CUDAStreamStub compute_stream;
extern CUDAStreamStub offload_stream;
extern CUDAStreamStub prefetch_stream;
extern std::map<const void*, CUDAEventStub> offloadFinishEvent;
extern std::set<c10::StorageImpl*> offloadStorages;
struct DeleteTensorInfo {
  at::DataPtr old_ptr;
};


struct KernelDependencies {
  std::vector<c10::StorageImpl*> prefetch_blocks;
  int kidx;

  KernelDependencies(int kidx) : kidx(kidx) {}
};


void comandDispatcherInitializer();
void comandDispatcherFinalizer();

void deleteCallback(void* the_);
void copyBytesCallBack(
    size_t nbytes,
    const void* src,
    c10::Device src_device,
    void* dst,
    c10::Device dst_device);

extern std::mutex offloadMutex;
extern std::map<const void*, CUDAEventStub> offloadFinishEvent;

class Offloader {
 public:
  void prefetch(const at::RecordFunction& fn, int kidx);
  virtual void offload(const at::RecordFunction& fn, int kidx) = 0;
  Offloader();
  ~Offloader();

 protected:
  void prefetchThread();
  std::set<c10::StorageImpl*> offloadStorages;

 private:
  std::thread prefetch_thread;
  ConcurrentQueue<KernelDependencies> prefetch_command_queue;
};

class NaiveOffloader : public Offloader {
 public:
  void offload(const at::RecordFunction& fn, int kidx);

 private:
};

class ScheduledOffloader : public Offloader {
 public:
  void offload(const at::RecordFunction& fn, int kidx);
  ScheduledOffloader();

 private:
  void parseSchedulerResults();
  std::map<int, std::string> scheduling_name;
  std::map<int, std::vector<int>> scheduling_results;
  std::map<int, std::vector<int>> scheduling_results_size;
};

struct CommandDispatcherObserverContext : public at::ObserverContext {
  bool need_offload;
};

} // namespace profiler
} // namespace autograd
} // namespace torch
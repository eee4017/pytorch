#include <torch/csrc/autograd/profiler_legacy.h>

#include <map>
#include <queue>

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

void comandDispatcherInitializer();
void comandDispatcherFinalizer();


void deleteCallback(void* the_);
void copyBytesCallBack(
    size_t nbytes,
    const void* src,
    c10::Device src_device,
    void* dst,
    c10::Device dst_device);
class Offloader {
 public:
  virtual void prefetch(const at::RecordFunction& fn, int kidx);
  virtual void offload(const at::RecordFunction& fn, int kidx) = 0;

 private:
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
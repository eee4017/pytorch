#include <condition_variable>
#include <mutex>
#include <queue>
#include <thread>

namespace torch {
namespace autograd {
namespace profiler {

template <typename T>
class ConcurrentQueue {
 public:
  bool empty() const {
    return queue_.empty();
  }

  T pop() {
    std::unique_lock<std::mutex> mlock(mutex_);
    while (queue_.empty()) {
      cond_.wait(mlock);
    }
    auto val = queue_.front();
    queue_.pop();
    mlock.unlock();
    cond_.notify_one();
    return val;
  }

  void pop(T& item) {
    std::unique_lock<std::mutex> mlock(mutex_);
    while (queue_.empty()) {
      cond_.wait(mlock);
    }
    item = queue_.front();
    queue_.pop();
    mlock.unlock();
    cond_.notify_one();
  }

  void push(const T& item) {
    std::unique_lock<std::mutex> mlock(mutex_);
    queue_.push(item);
    mlock.unlock();
    cond_.notify_one();
  }
  ConcurrentQueue() = default;
  ConcurrentQueue(const ConcurrentQueue&) = delete; // disable copying
  ConcurrentQueue& operator=(const ConcurrentQueue&) =
      delete; // disable assignment

 private:
  std::queue<T> queue_;
  std::mutex mutex_;
  std::condition_variable cond_;
};

} // namespace profiler
} // namespace autograd
} // namespace torch
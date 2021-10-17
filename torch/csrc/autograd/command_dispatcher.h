#include <torch/csrc/autograd/profiler_legacy.h>

namespace torch {
namespace autograd {
namespace profiler {

struct CommandDispatcherObserverContext : public at::ObserverContext {
};

void comandDispatcherInitializer();
void comandDispatcherFinalizer();

} // namespace profiler
} // namespace autograd
} // namespace torch
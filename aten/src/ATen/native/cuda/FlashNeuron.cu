#include <ATen/native/cuda/FlashNeuron.h>
#include <ATen/Context.h>

#define find(n) (32 * (unsigned int)(n / 1024) + (n % 32))
#define mask(n) (0x80000000 >> (unsigned int)((n % 1024) / 32))

namespace at {
namespace native {

using namespace at::cuda;

FN_manager FN_mngt;

FN_manager::FN_manager(): global_tID(-1) {

}

FN_manager::~FN_manager() {

}

}} // at::native

#include <ATen/native/cuda/FlashNeuron.h>
#include <ATen/Context.h>

#define find(n) (32 * (unsigned int)(n / 1024) + (n % 32))
#define mask(n) (0x80000000 >> (unsigned int)((n % 1024) / 32))

namespace at {
namespace native {

using namespace at::cuda;

FN_manager FN_mngt;

FN_manager::FN_manager(): global_tID(-1), global_oID(-1) {

}

FN_manager::~FN_manager() {

}

int FN_manager::getOid() {
  return global_oID;
}

void FN_manager::setOid() {
  global_oID++;
}

void FN_manager::resetOid() {
  global_oID = -1;
}

int FN_manager::getTid() {
  return global_tID;
}

void FN_manager::setTid() {
  global_tID++;
}

void FN_manager::resetTid() {
  global_tID = -1;
}

}} // at::native

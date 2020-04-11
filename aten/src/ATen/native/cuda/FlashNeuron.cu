#include <ATen/native/cuda/FlashNeuron.h>
#include <ATen/Context.h>

#define find(n) (32 * (unsigned int)(n / 1024) + (n % 32))
#define mask(n) (0x80000000 >> (unsigned int)((n % 1024) / 32))

#define FLAG_OFFLOAD  (1U << 0)
#define FLAG_FP16  (1U << 1)
#define FLAG_CSR   (1U << 2)
#define FLAG_SSD   (1U << 3)
#define FLAG_TESLA (1U << 4)
#define FLAG_RAID0 (1U << 5)
#define FLAG_DEBUG (1U << 6)
// 7~11 bit will be used for arc_vm (device) cudamalloc size
#define DEVSIZE_MASK  (0x00000F80)
// 12~16 bit will be used for arc_vm (p2p) cudamalloc size
#define TMPSIZE_MASK  (0x0001F000)
#define FLAG_TIMER (1U << 17)

#define BLK_SIZE ((size_t)1 << 12)

namespace at {
namespace native {

using namespace at::cuda;

FN_manager FN_mngt;

FN_manager::FN_manager():isTimer(false), isOffload(false), isFP16(false), isCSR(false),
                         isUsingSSD(false), isTesla(false), isDebug(false),
                         device_size(0), max_device(0), temporal_size(0), max_temporal(0),
                         global_tID(-1), global_oID(-1) {

}

FN_manager::~FN_manager() {

}

void FN_manager::setting(int flags) {
  uint64_t device_in_gb;
  device_in_gb = (flags & DEVSIZE_MASK) >> 7;
  device_size = device_in_gb << 30;
  max_device = device_size / BLK_SIZE;

  uint64_t temporal_in_gb;
  temporal_in_gb = (flags & TMPSIZE_MASK) >> 12;
  temporal_size = temporal_in_gb << 30;
  max_temporal = temporal_size / BLK_SIZE;

  if (device_in_gb > 0) {
    device_table = new short[max_device];
    memset(device_table, 0, sizeof(short) * max_device);

    device_page_map = new unsigned int[max_device];
    memset(device_page_map, 0, sizeof(unsigned int) * max_device);
  }

  if (temporal_in_gb > 0) {
    temporal_table = new short[max_temporal];
    memset(temporal_table, 0, sizeof(short) * max_temporal);

    temporal_page_map = new unsigned int[max_temporal];
    memset(temporal_page_map, 0, sizeof(unsigned int) * max_temporal);
  }

  if (flags & FLAG_TIMER) {
    printf("Timer profiler set\n");
    isTimer = true;
  }

  if (flags & FLAG_OFFLOAD) {
    printf("Offload flag set\n");
    isOffload = true;
  }

  if (flags & FLAG_FP16) {
    printf("FP16 flag set\n");
    isOffload = true;
    isFP16 = true;
  }

  if (flags & FLAG_CSR) {
    printf("CSR flag set\n");
    isOffload = true;
    isCSR = true;
  }

  if (flags & FLAG_TESLA) {
    printf("Tesla GPU flag set\n");
    isTesla = true;
  }

  if (flags & FLAG_DEBUG) {
    printf("Debug mode on\n");
    isDebug = true;
  }

  if (flags & FLAG_SSD) {
    printf("SSD flag set\n");
    isOffload = true;
    isUsingSSD = true;
  }
}

// Operation ID assignment
int FN_manager::getOid() { return global_oID; }
void FN_manager::setOid() { global_oID++; }
void FN_manager::resetOid() { global_oID = -1; }

// Tensor ID assignment
int FN_manager::getTid() { return global_tID; }
void FN_manager::setTid() { global_tID++; }
void FN_manager::resetTid() { global_tID = -1; }

// Flag check functions
bool FN_manager::is_timer() { return isTimer; }
bool FN_manager::is_offload() { return isOffload; }
bool FN_manager::is_fp16() { return isFP16; }
bool FN_manager::is_csr() { return isCSR; }
bool FN_manager::is_using_ssd() { return isUsingSSD; }
bool FN_manager::is_debug() { return isDebug; }


}} // at::native

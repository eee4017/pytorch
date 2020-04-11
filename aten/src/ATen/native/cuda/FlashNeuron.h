#pragma once

#include <ATen/ATen.h>

#define BLK_SZ ((size_t)1 << 12)
#define NUM_TENSOR 32768
#define NUM_OP 32768

using namespace std;

namespace at {
namespace native {

class FN_manager {
 public:
  FN_manager();
  ~FN_manager();

  void setting(int flags);

  int getOid();
  void setOid();
  void resetOid();

  int getTid();
  void setTid();
  void resetTid();

  bool is_timer();
  bool is_offload();
  bool is_fp16();
  bool is_csr();
  bool is_using_ssd();
  bool is_debug();

  bool liveness_result[NUM_TENSOR] = {false};

 private:
  bool isTimer;
  bool isOffload;
  bool isFP16;
  bool isCSR;
  bool isUsingSSD;
  bool isTesla;
  bool isDebug;

  short* device_table;
  uint64_t device_size;
  uint64_t max_device;
  unsigned int* device_page_map;

  short* temporal_table;
  uint64_t temporal_size;
  uint64_t max_temporal;
  unsigned int* temporal_page_map;

  int global_tID;
  int global_oID;
};

extern FN_manager FN_mngt;

}} // at::native

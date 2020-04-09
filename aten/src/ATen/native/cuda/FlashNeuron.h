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

   int global_tID;
};

extern FN_manager FN_mngt;

}} // at::native

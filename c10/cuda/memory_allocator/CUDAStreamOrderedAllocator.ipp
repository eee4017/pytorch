

#include <c10/cuda/CUDACachingAllocator.h>

#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAFunctions.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/util/UniqueVoidPtr.h>

#include <cuda_runtime_api.h>
#include <algorithm>
#include <bitset>
#include <chrono>
#include <deque>
#include <fstream>
#include <iostream>
#include <iterator>
#include <map>
#include <memory>
#include <mutex>
#include <set>
#include <unordered_map>
#include <unordered_set>
#include <vector>


namespace c10 {
namespace cuda {
namespace CUDACachingAllocator {


class THCCachingAllocator {
 private:
  std::mutex mutex;

  // allocated blocks by device pointer
  std::unordered_map<void*, Block*> allocated_blocks;

  // lock around calls to cudaFree (to prevent deadlocks with NCCL)
  mutable std::mutex cuda_free_mutex;

  void add_allocated_block(Block* block) {
    std::lock_guard<std::mutex> lock(mutex);
    allocated_blocks[block->ptr] = block;
  }

 public:
  std::vector<cudaMemPool_t> device_allocator;

  std::mutex* getCudaFreeMutex() const {
    return &cuda_free_mutex;
  }

  Block* get_allocated_block(void* ptr, bool remove = false) {
    std::lock_guard<std::mutex> lock(mutex);
    auto it = allocated_blocks.find(ptr);
    if (it == allocated_blocks.end()) {
      return nullptr;
    }
    Block* block = it->second;
    if (remove) {
      allocated_blocks.erase(it);
    }
    return block;
  }

  void init(int device_count) {
    int size = device_allocator.size();
    if (size < device_count) {
      device_allocator.resize(device_count);
      for (int i = size; i < device_count; i++) {
        cudaMemPoolProps poolProps = {};
        std::cerr << "creating cuda memory pool on device " << i << "\n";
        // poolProps.location.type = cudaMemLocationTypeDevice;
        // poolProps.location.id = i;
        // C10_CUDA_CHECK(cudaMemPoolCreate(&device_allocator[i], &poolProps));
        C10_CUDA_CHECK(cudaDeviceGetDefaultMemPool(&device_allocator[i], i));
        uint64_t threshold = UINT64_MAX;
        C10_CUDA_CHECK(cudaMemPoolSetAttribute(device_allocator[i], cudaMemPoolAttrReleaseThreshold, &threshold));

        int disable = 0;
        // C10_CUDA_CHECK(cudaMemPoolSetAttribute(device_allocator[i], cudaMemPoolReuseFollowEventDependencies, &disable));
        // C10_CUDA_CHECK(cudaMemPoolSetAttribute(device_allocator[i], cudaMemPoolReuseAllowOpportunistic, &disable));
        C10_CUDA_CHECK(cudaMemPoolSetAttribute(device_allocator[i], cudaMemPoolReuseAllowInternalDependencies, &disable));
      }
    }
  }

  /** allocates a block which is safe to use from the provided stream */
  void malloc(void** devPtr, int device, size_t size, cudaStream_t stream) {
    int activated_device;
    cudaGetDevice(&activated_device);
    if (activated_device != device) {
      cudaSetDevice(device);
    }

    C10_CUDA_CHECK(cudaMallocAsync(devPtr, size, stream));
    Block* block = new Block(device, stream, size, nullptr, *devPtr);
    add_allocated_block(block);
  }

  void free(void* ptr) {
    if (!ptr) {
      return;
    }
    Block* block = get_allocated_block(ptr, true /* remove */);
    if (!block) {
      TORCH_CHECK(false, "invalid device pointer: ", ptr);
    }
    // int activated_device;
    // cudaGetDevice(&activated_device);
    // if (activated_device != device) {
    //   cudaSetDevice(device);
    // }

    C10_CUDA_CHECK(cudaFreeAsync(block->ptr, block->stream));
  }

  void free_async(void* ptr, cudaStream_t stream) {
    if (!ptr) {
      return;
    }

    std::cerr << "free_async " << ptr << "\n";
    Block* block = get_allocated_block(ptr, true /* remove */);
    if (!block) {
      TORCH_CHECK(false, "invalid device pointer: ", ptr);
    }
    // int activated_device;
    // cudaGetDevice(&activated_device);
    // if (activated_device != device) {
    //   cudaSetDevice(device);
    // }

    C10_CUDA_CHECK(cudaFreeAsync(block->ptr, stream));
  }

  void setMemoryFraction(double fraction, int device) {
    TORCH_INTERNAL_ASSERT(
        0 <= device && device < device_allocator.size(),
        "Allocator not initialized for device ",
        device,
        ": did you call init?");
    TORCH_INTERNAL_ASSERT(
        0 <= fraction && fraction <= 1,
        "invalid fraction:",
        fraction,
        ". Please set within (0, 1).");
    int activated_device;
    cudaGetDevice(&activated_device);
    if (activated_device != device) {
      cudaSetDevice(device);
    }
    std::cerr << "Unimplemented " << __FUNCTION__ << "\n";
  }

  void emptyCache() {
    int count = device_allocator.size();
    for (int i = 0; i < count; i++) {
      C10_CUDA_CHECK(cudaMemPoolDestroy(device_allocator[i]));
    }
  }

  void* getBaseAllocation(void* ptr, size_t* outSize) {
    Block* block = get_allocated_block(ptr);
    if (!block) {
      TORCH_CHECK(false, "invalid device pointer: ", ptr);
    }
    return block->ptr;
  }

  void recordStream(const DataPtr& ptr, cuda::CUDAStream stream) {
    // Empty tensor's storage().data() might be a null ptr. As there is no
    // blocks associated with those tensors, it is fine to do nothing here.
    if (!ptr.get()) {
      return;
    }

    // If a tensor is not allocated by this instance, simply skip
    // This usually happens when CUDA tensors are shared across processes,
    // we have implemented reference counting based sharing mechanism to
    // guarantee tensors won't be accidentally freed by one process while
    // they are still being used in another
    if (ptr.get_deleter() != &raw_delete)
      return;

    Block* block = get_allocated_block(ptr.get());
    // block must not be null reaching here
    TORCH_INTERNAL_ASSERT(block != nullptr, "No allocated block can be found");
    std::cerr << "Unimplemented " << __FUNCTION__ << "\n";
  }

  std::vector<SegmentInfo> snapshot() {
    std::vector<SegmentInfo> result;

    std::cerr << "Unimplemented " << __FUNCTION__ << "\n";
    return result;
  }
};

THCCachingAllocator caching_allocator;

bool forceUncachedAllocator() {
  static bool force_uncached =
      getenv("PYTORCH_NO_CUDA_MEMORY_CACHING") != nullptr;
  return force_uncached;
}

static void uncached_delete(void* ptr) {
  int device;
  C10_CUDA_CHECK(cudaGetDevice(&device));
  C10_CUDA_CHECK(cudaFreeAsync(ptr, cuda::getCurrentCUDAStream(device)));
}

struct CudaCachingAllocator : public Allocator {
  DataPtr allocate(size_t size) const override {
    constexpr size_t one_exa_bytes = 1152921504606846976ULL;
    TORCH_CHECK_WITH(
        CUDAOutOfMemoryError,
        size < one_exa_bytes,
        "CUDA out of memory. Tried to allocate more than 1EB memory.");
    int device;
    C10_CUDA_CHECK(cudaGetDevice(&device));
    void* r = nullptr;
    if (forceUncachedAllocator()) {
      // Deliberately don't use cudaMallocMaybeCapturing here, to force an error
      // if someone tries to use forceUncachedAllocator while capturing.
      C10_CUDA_CHECK(cudaMalloc(&r, size));
      return {r, r, &uncached_delete, Device(DeviceType::CUDA, device)};
    }
    if (size != 0) {
      caching_allocator.malloc(
          &r, device, size, cuda::getCurrentCUDAStream(device));
    }
    return {r, r, &raw_delete, Device(DeviceType::CUDA, device)};
  }
  DeleterFnPtr raw_deleter() const override {
    if (forceUncachedAllocator()) {
      return &uncached_delete;
    } else {
      return &raw_delete;
    }
  }
};

CudaCachingAllocator device_allocator;

Allocator* get(void) {
  return &device_allocator;
}

void init(int device_count) {
  caching_allocator.init(device_count);
}

void setMemoryFraction(double fraction, int device) {
  caching_allocator.setMemoryFraction(fraction, device);
}

void emptyCache(void) {
  caching_allocator.emptyCache();
}

void cacheInfo(int dev_id, size_t* cachedAndFree, size_t* largestBlock) {
  std::cerr << "Unimplemented " << __FUNCTION__ << "\n";
}

void* getBaseAllocation(void* ptr, size_t* size) {
  return caching_allocator.getBaseAllocation(ptr, size);
}

void recordStream(const DataPtr& ptr, cuda::CUDAStream stream) {
  caching_allocator.recordStream(ptr, stream);
}

std::mutex* getFreeMutex() {
  return caching_allocator.getCudaFreeMutex();
}

static inline void assertValidDevice(int device) {
  int device_num = caching_allocator.device_allocator.size();
  TORCH_CHECK(0 <= device && device < device_num, "Invalid device argument.");
}

DeviceStats getDeviceStats(int device) {
  assertValidDevice(device);
  std::cerr << "Unimplemented " << __FUNCTION__ << "\n";
  return DeviceStats();
}

void resetAccumulatedStats(int device) {
  assertValidDevice(device);
  std::cerr << "Unimplemented " << __FUNCTION__ << "\n";
}

void resetPeakStats(int device) {
  assertValidDevice(device);
  std::cerr << "Unimplemented " << __FUNCTION__ << "\n";
}

std::vector<SegmentInfo> snapshot() {
  return caching_allocator.snapshot();
}

namespace {
std::mutex IpcMutex;
std::unordered_map<std::string, std::weak_ptr<void>> ipcMemHandle_to_devptr;
} // namespace

std::shared_ptr<void> getIpcDevPtr(std::string handle) {
  std::lock_guard<std::mutex> lock(IpcMutex);

  auto iter = ipcMemHandle_to_devptr.find(handle);
  if (iter != ipcMemHandle_to_devptr.end()) {
    auto devptr = iter->second.lock();
    if (devptr)
      return devptr;
  }
  // This ipcMemHandle hasn't been opened, or already expired, open it to
  // enable IPC access to that mem block.
  void* dev = nullptr;
  auto ipc_handle = reinterpret_cast<const cudaIpcMemHandle_t*>(handle.c_str());
  C10_CUDA_CHECK(
      cudaIpcOpenMemHandle(&dev, *ipc_handle, cudaIpcMemLazyEnablePeerAccess));
  // devPtr has to be deleted in same device when created.
  int curr_device;
  C10_CUDA_CHECK(cudaGetDevice(&curr_device));
  auto sp = std::shared_ptr<void>(dev, [handle, curr_device](void* ptr) {
    cuda::CUDAGuard device_guard(curr_device);
    std::lock_guard<std::mutex> deleter_lock(IpcMutex);
    C10_CUDA_CHECK(cudaIpcCloseMemHandle(ptr));
    ipcMemHandle_to_devptr.erase(handle);
  });
  std::weak_ptr<void> wp = sp;
  // To eliminate an additional search, we can use insert().
  // It doesn't overwrite when key already exists(ptr expired).
  // But in the deleter for sp we erased the entry,
  // this should be safe to do now.
  ipcMemHandle_to_devptr.insert(iter, {handle, wp});

  return sp;
}

// CUDAGraph interactions
void notifyCaptureBegin(
    int device,
    CaptureId_t graph_id,
    MempoolId_t mempool_id) {
  assertValidDevice(device);
  std::cerr << "Unimplemented " << __FUNCTION__ << "\n";
}

void notifyCaptureEnd(int device, CaptureId_t graph_id) {
  assertValidDevice(device);
  std::cerr << "Unimplemented " << __FUNCTION__ << "\n";
}

void notifyCaptureDestroy(int device, MempoolId_t mempool_id) {
  assertValidDevice(device);
  std::cerr << "Unimplemented " << __FUNCTION__ << "\n";
}

void* raw_alloc(size_t nbytes) {
  if (nbytes == 0) {
    return nullptr;
  }
  int device;
  C10_CUDA_CHECK(cudaGetDevice(&device));
  void* r = nullptr;
  caching_allocator.malloc(
      &r, device, nbytes, cuda::getCurrentCUDAStream(device));
  return r;
}

void* raw_alloc_with_stream(size_t nbytes, cudaStream_t stream) {
  if (nbytes == 0) {
    return nullptr;
  }
  int device;
  C10_CUDA_CHECK(cudaGetDevice(&device));
  void* r = nullptr;
  caching_allocator.malloc(&r, device, nbytes, stream);
  return r;
}

void raw_delete(void* ptr) {
  caching_allocator.free(ptr);
}


void raw_delete_with_stream(void* ptr, cudaStream_t stream) {
  caching_allocator.free_async(ptr, stream);
}


} // namespace CUDACachingAllocator
} // namespace cuda
} // namespace c10

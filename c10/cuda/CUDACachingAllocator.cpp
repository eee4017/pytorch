
#include <c10/cuda/CUDACachingAllocator.h>

#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAFunctions.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/util/UniqueVoidPtr.h>
#include <c10/core/LargeModelSupport.h>

#include <cuda_runtime_api.h>
#include <algorithm>
#include <bitset>
#include <deque>
#include <iterator>
#include <map>
#include <memory>
#include <mutex>
#include <set>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <iostream>
#include <chrono>
#include <fstream>


// #define STREAM_ORDERED_ALLOCATOR
#define IBM_LMS_ALLOCATOR
// #define PYTORCH_ALLOCATOR

#ifdef STREAM_ORDERED_ALLOCATOR
#include "memory_allocator/PytorchDeviceCachingAllocator.ipp"
#include "memory_allocator/CUDAStreamOrderedAllocator.ipp"
#endif

#ifdef IBM_LMS_ALLOCATOR
#include "memory_allocator/IBMLMSAllocator.ipp"
#endif

#ifdef PYTORCH_ALLOCATOR
#include "memory_allocator/PytorchDeviceCachingAllocator.ipp"
#include "memory_allocator/PytorchAllocator.ipp"
#endif





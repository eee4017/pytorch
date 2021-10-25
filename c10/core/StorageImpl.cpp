#include <ATen/cuda/PinnedMemoryAllocator.h>
#include <THC/THCCachingHostAllocator.h>
#include <c10/core/StorageImpl.h>
#include <cuda_runtime_api.h>
#include <iostream>
namespace c10 {

at::DataPtr StorageImpl::swap_out(
    Device dst_device,
    CopyBytesFunction copyBytesCallback) {
  if (is_swapped_out_) {
    return DataPtr();
  }
  auto old_data_ptr = std::move(data_ptr());
  // auto allocator = GetAllocator(dst_device.type());
  // auto allocator = GetAllocator(c10::DeviceType::CPU);
  auto allocator = getTHCCachingHostAllocator();
  auto data_ptr = allocator->allocate(size_bytes_);
  data_ptr_ = std::move(data_ptr); // CPU PTR

  copyBytesCallback_ = copyBytesCallback;
  copyBytesCallback_(
      size_bytes_,
      old_data_ptr.get(),
      old_data_ptr.device(),
      data_ptr.get(),
      data_ptr.device());

  is_swapped_out_ = true;
  original_allocator_ = allocator_;
  allocator_ = allocator;

  std::cerr << "StorageImpl::swap_out to " << data_ptr.device() << "\n";
  return (std::move(old_data_ptr));
}

at::DataPtr StorageImpl::swap_in() {
  if (!is_swapped_out_) {
    return DataPtr();
  }

  auto old_data_ptr = std::move(data_ptr());
  auto allocator = original_allocator_;
  auto data_ptr = allocator->allocate(size_bytes_);
  data_ptr_ = std::move(data_ptr); // GPU PTR

  copyBytesCallback_(
      size_bytes_,
      old_data_ptr.get(),
      old_data_ptr.device(),
      data_ptr.get(),
      data_ptr.device());

  is_swapped_out_ = false;
  original_allocator_ = allocator_;
  allocator_ = allocator;

  std::cerr << "StorageImpl::swap_in to " << data_ptr.device() << "\n";
  return (std::move(old_data_ptr));
}

void StorageImpl::release_old_data_ptr() {
  // std::cerr << "Delete data_ptr_queue_.size() = " << data_ptr_queue_.size()
  // << "\n"; if(data_ptr_queue_.size() > 0){
  //   auto top = std::move(data_ptr_queue_.front());
  //   data_ptr_queue_.pop_front();
  // }
}

at::DataPtr StorageImpl::set_data_ptr(at::DataPtr&& data_ptr) {
  if (is_swapped_out_)
    std::cerr << "SET DATAPTR " << __FUNCTION__ << " " << data_ptr.get()
              << "\n";
  std::swap(data_ptr_, data_ptr);
  return std::move(data_ptr);
};

void StorageImpl::set_data_ptr_noswap(at::DataPtr&& data_ptr) {
  if (is_swapped_out_)
    std::cerr << "SET DATAPTR " << __FUNCTION__ << " " << data_ptr.get()
              << "\n";
  data_ptr_ = std::move(data_ptr);
}

// TODO: Return const ptr eventually if possible
void* StorageImpl::data() {
  return data_ptr_.get();
}

void* StorageImpl::data() const {
  return data_ptr_.get();
}

at::DeviceType StorageImpl::device_type() const {
  return data_ptr_.device().type();
}

at::Allocator* StorageImpl::allocator() {
  return allocator_;
}

const at::Allocator* StorageImpl::allocator() const {
  return allocator_;
};

// You generally shouldn't use this method, but it is occasionally
// useful if you want to override how a tensor will be reallocated,
// after it was already allocated (and its initial allocator was
// set)
void StorageImpl::set_allocator(at::Allocator* allocator) {
  allocator_ = allocator;
}

Device StorageImpl::device() const {
  return data_ptr_.device();
}

void StorageImpl::set_resizable(bool resizable) {
  if (resizable) {
    // We need an allocator to be resizable
    AT_ASSERT(allocator_);
  }
  resizable_ = resizable;
}

/**
 * Can only be called when use_count is 1
 */
void StorageImpl::UniqueStorageShareExternalPointer(
    void* src,
    size_t size_bytes,
    DeleterFnPtr d) {
  if (is_swapped_out_)
    std::cerr << "SET DATAPTR " << __FUNCTION__ << " " << src << "\n";
  UniqueStorageShareExternalPointer(
      at::DataPtr(src, src, d, data_ptr_.device()), size_bytes);
}

/**
 * Can only be called when use_count is 1
 */
void StorageImpl::UniqueStorageShareExternalPointer(
    at::DataPtr&& data_ptr,
    size_t size_bytes) {
  if (is_swapped_out_)
    std::cerr << "SET DATAPTR " << __FUNCTION__ << " " << data_ptr.get()
              << "\n";
  data_ptr_ = std::move(data_ptr);
  size_bytes_ = size_bytes;
  allocator_ = nullptr;
  resizable_ = false;
}

} // namespace c10
#pragma once

#include <c10/core/Allocator.h>
#include <c10/core/ScalarType.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/util/intrusive_ptr.h>
#include <c10/core/LargeModelSupport.h>
#include <deque>

namespace c10 {

using CopyBytesFunction = void (*)(
    size_t nbytes,
    const void* src,
    Device src_device,
    void* dst,
    Device dst_device);

// A storage represents the underlying backing data buffer for a
// tensor.  This concept was inherited from the original Torch7
// codebase; we'd kind of like to get rid of the concept
// (see https://github.com/pytorch/pytorch/issues/14797) but
// it's hard work and no one has gotten around to doing it.
//
// NB: storage is supposed to uniquely own a data pointer; e.g.,
// two non-null data pointers alias if and only if they are from
// the same storage.  Technically you can violate this invariant
// (e.g., you can create a non-owning StorageImpl with at::from_blob)
// but a lot of things won't work correctly, including:
//
// - An ordinary deleter on such a storage is wrong, because normal deleters
//   assume unique ownership, but if you have two storages at the same data,
//   that implies there is some sort of shared ownership. So your deleter would
//   have to actually be internally doing some sort of refcount thing
// - Deepcopy in Python side relies on storage equality and not data pointer
//   equality; so if there are two separate storages pointing to the same data,
//   the data will actually get duplicated in that case (one data ptr before,
//   two data ptrs after)
// - Version counts won't work correctly, because we do all VC tracking at the
//   level of storages (unless you explicitly disconnect the VC with detach);
//   mutation because data pointers are the same are totally untracked
struct C10_API StorageImpl final : public c10::intrusive_ptr_target {
 public:
  struct use_byte_size_t {};

  StorageImpl(
      use_byte_size_t use_byte_size,
      size_t size_bytes,
      at::DataPtr data_ptr,
      at::Allocator* allocator,
      bool resizable)
      : data_ptr_(std::move(data_ptr)),
        size_bytes_(size_bytes),
        resizable_(resizable),
        received_cuda_(false),
        allocator_(allocator),
        lms_(allocator ? allocator->AsLmsStorage(this) : nullptr) {
    if (resizable) {
      TORCH_INTERNAL_ASSERT(
          allocator_, "For resizable storage, allocator must be provided");
    }
  }

  StorageImpl(
      use_byte_size_t use_byte_size,
      size_t size_bytes,
      at::Allocator* allocator,
      bool resizable)
      : StorageImpl(
            use_byte_size_t(),
            size_bytes,
            allocator->allocate(size_bytes),
            allocator,
            resizable) {}

  StorageImpl& operator=(StorageImpl&& other) = default;
  StorageImpl& operator=(const StorageImpl&) = delete;
  StorageImpl() = delete;
  StorageImpl(StorageImpl&& other) = default;
  StorageImpl(const StorageImpl&) = delete;
  ~StorageImpl() = default;

  void reset() {
    lms_.reset(nullptr);
    data_ptr_.clear();
    size_bytes_ = 0;
  }

  template <typename T>
  inline T* data() const {
    return unsafe_data<T>();
  }

  template <typename T>
  inline T* unsafe_data() const {
    if (lms_enabled()) lms_->ensure_data();
    return static_cast<T*>(this->data_ptr_.get());
  }

  void release_resources() override {
    if (lms_enabled())
      lms_->release_resources();
    data_ptr_.clear();
  }

  size_t nbytes() const {
    return size_bytes_;
  }

  // TODO: remove later
  void set_nbytes(size_t size_bytes) {
    size_bytes_ = size_bytes;
  }

  bool resizable() const {
    return resizable_;
  };

  at::DataPtr& data_ptr() {
    if (lms_enabled()) lms_->ensure_data();
    return data_ptr_;
  };

  const at::DataPtr& data_ptr() const {
    if (lms_enabled()) lms_->ensure_data();
    return data_ptr_;
  };

  // Returns the previous data_ptr
  at::DataPtr set_data_ptr(at::DataPtr&& data_ptr);

  void set_data_ptr_noswap(at::DataPtr&& data_ptr);

  // TODO: Return const ptr eventually if possible
  void* data();

  void* data() const;

  at::DeviceType device_type() const;

  at::Allocator* allocator();

  const at::Allocator* allocator() const;

  // You generally shouldn't use this method, but it is occasionally
  // useful if you want to override how a tensor will be reallocated,
  // after it was already allocated (and its initial allocator was
  // set)
  void set_allocator(at::Allocator* allocator);

  Device device() const;

  void set_resizable(bool resizable);

  /**
   * Can only be called when use_count is 1
   */
  void UniqueStorageShareExternalPointer(
      void* src,
      size_t size_bytes,
      DeleterFnPtr d = nullptr);

  /**
   * Can only be called when use_count is 1
   */
  void UniqueStorageShareExternalPointer(
      at::DataPtr&& data_ptr,
      size_t size_bytes);

  // This method can be used only after storage construction and cannot be used
  // to modify storage status
  void set_received_cuda(bool received_cuda) {
    received_cuda_ = received_cuda;
  }

  bool received_cuda() {
    return received_cuda_;
  }
  
  DataPtr swap_out(Device to_device, CopyBytesFunction copyBytesCallback, cudaStream_t stream);
  DataPtr swap_in(cudaStream_t stream);
  void swap_to(Device to_device);
  void release_old_data_ptr();

  bool is_swapped_out_ = false;

  // Large Model Support
  bool lms_enabled() const;
  bool lms_pin();
  bool lms_unpin();
  bool lms_reclaimed() const;
  void lms_copy_reclaimed_data(void* dst, size_t size);

 private:
  Allocator* original_allocator_;
  CopyBytesFunction copyBytesCallback_;

  DataPtr data_ptr_;
  // std::deque<DataPtr> data_ptr_queue_;
  size_t size_bytes_;
  bool resizable_;
  // Identifies that Storage was received from another process and doesn't have
  // local to process cuda memory allocation
  bool received_cuda_;
  Allocator* allocator_;

  std::unique_ptr<LmsStorageImpl> lms_;
  friend class LmsStorageImpl;  // data_ptr_ access
};

// StorageImpl accessors for LMS to avoid circular depencencies
inline const Allocator* LmsStorageImpl::allocator() const {
  return storage_->allocator();
}

inline size_t LmsStorageImpl::capacity() const {
  return storage_->nbytes();
}

inline Device LmsStorageImpl::device() const {
  return storage_->device();
}

inline void* LmsStorageImpl::device_ptr() const {
  return storage_->data_ptr_.get();
}

inline at::DataPtr LmsStorageImpl::set_device_ptr(at::DataPtr&& data_ptr) {
  std::swap(storage_->data_ptr_, data_ptr);
  return std::move(data_ptr);
}

} // namespace c10

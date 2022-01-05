#include <mutex>
#include <cstring>
#include <c10/core/LargeModelSupport.h>

/*
namespace c10 {

void LmsStorageImpl::release_resources() {
  if (state_ == State::kZombie)
    return;
  if (transition_ != Transition::kNone) {
    debug_log(0, "pending transition at release_resources");
    transition_wait();
  }
  reclaim_list_remove_internal();
  host_data_ptr_.clear();
  state_ = State::kZombie;
}

bool LmsStorageImpl::reclaimed() const {
  return state_ == State::kReclaimed;
};

bool LmsStorageImpl::pin() {
  std::unique_lock<std::mutex> lock(mutex_);
  bool initial = (++pincount_ == 1);
  if (initial) {
    if (state_ != State::kInit) {
      ensure_data_internal(true); // pin 
    }
    state_ = State::kActive;
  }
  TORCH_INTERNAL_ASSERT(state_ == State::kActive);
  TORCH_INTERNAL_ASSERT(pincount_ > 0);
  return initial;
}

bool LmsStorageImpl::unpin() {
  std::unique_lock<std::mutex> lock(mutex_);
  TORCH_INTERNAL_ASSERT(state_ == State::kActive);
  TORCH_INTERNAL_ASSERT(pincount_ > 0);
  bool final = (--pincount_ == 0);
  if (final) {
    bool pageout = reclaim_list_add();
    if (pageout) {
      // Speculative pageout requested by allocator
      pageout_internal(true); // speculative 
    }
    state_ = State::kInactive;
  }
  return final;
}

void LmsStorageImpl::ensure_data() {
  std::unique_lock<std::mutex> lock(mutex_);
  if (pincount_ == 0 && state_ != State::kInit) {
    ensure_data_internal(false); // !pin 
    state_ = State::kInit;
  }
  if (transition_ != Transition::kNone) {
    transition_wait();
  }
}

bool LmsStorageImpl::reclaim(bool sync) {
  std::unique_lock<std::mutex> lock(mutex_, std::try_to_lock);
  if (!lock.owns_lock()) {
    // Inability to acquire the lock means this is exiting
    // the inactive state and thus not a good candidate to reclaim.
    return false;
  }
  pageout_internal(false); // !speculative
  if (sync) {
    reclaim_wait();
  }
  return true;
}

bool LmsStorageImpl::reclaim_sync() {
  std::unique_lock<std::mutex> lock(mutex_, std::try_to_lock);
  if (!lock.owns_lock()) {
    // See comment in reclaim above.  The contending thread will
    // complete the transition.
    return false;
  }
  TORCH_INTERNAL_ASSERT(transition_ != Transition::kNone);
  reclaim_wait();
  return true;
}

void LmsStorageImpl::copy_reclaimed_data(void* dst, size_t size) const {
  std::unique_lock<std::mutex> lock(mutex_);
  TORCH_INTERNAL_ASSERT(state_ == State::kReclaimed);
  memcpy(dst, host_data_ptr_.get(), size);
}

void LmsStorageImpl::list_add(LmsStorageList* list) {
  list->append(&list_hook_);
}

bool LmsStorageImpl::list_remove() {
  return list_hook_.remove();
}

void LmsStorageImpl::ensure_data_internal(bool pin) {
  switch (state_) {
    case State::kInactive:
      reclaim_list_remove();
      break;
    case State::kReclaimed:
      pagein(pin);
      break;
    case State::kInit:
    case State::kActive:
      // Nothing to do
      break;
    case State::kZombie:
      TORCH_INTERNAL_ASSERT(false, "Unexpected use of LMS zombie");
      break;
  }
}

void LmsStorageImpl::reclaim_list_remove_internal() {
  if (pincount_ == 0 && state_ == State::kInactive) {
    reclaim_list_remove();
  }
}

void LmsStorageImpl::pageout_internal(bool speculative) {
  if (transition_ == Transition::kNone) {
    size_t size = capacity();
    void* dst = host_data_ptr_.get();
    if (!dst) {
      host_data_ptr_ = host_allocator_->allocate(size);
      dst = host_data_ptr_.get();
    }
    do_pageout(dst, device_ptr(), size, speculative);
    transition_ = Transition::kPagingOut;
  }
}

void LmsStorageImpl::pagein(bool pin) {
  TORCH_INTERNAL_ASSERT(!device_ptr());
  size_t size = capacity();
  auto dst = allocator()->allocate(size);
  do_pagein(dst.get(), host_data_ptr_.get(), size, pin);
  set_device_ptr(std::move(dst));
  transition_ = Transition::kPagingIn;
}

void LmsStorageImpl::transition_wait() {
  do_sync(false); // !reclaim

  // Free host allocation
  host_data_ptr_.clear();
  transition_ = Transition::kNone;
}

void LmsStorageImpl::reclaim_wait() {
  do_sync(true); // reclaim

  // Release device allocation (allocator will free it)
  auto old_device_ptr = set_device_ptr(at::DataPtr(nullptr, device()));
  old_device_ptr.release_context();
  state_ = State::kReclaimed;
  transition_ = Transition::kNone;
}

} // namespace c10

*/
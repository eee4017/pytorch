#include <c10/cuda/CUDACachingAllocator.h>
#include <torch/csrc/THP.h>
#include <torch/csrc/autograd/python_alloc_delete_hook.h>
#include <torch/csrc/utils/python_numbers.h>

#include <iostream>

namespace py = pybind11;

namespace torch {
namespace autograd {

void AllocDeleteHook::call_alloc_hook(void* ptr, int device, size_t size) {
  py::gil_scoped_acquire acquire;
  auto alloc_hook = py::reinterpret_borrow<py::function>(alloc_hook_);
  alloc_hook(
      py::reinterpret_steal<py::object>(
          THPUtils_packInt64(reinterpret_cast<int64_t>(ptr))),
      py::reinterpret_steal<py::object>(
          THPUtils_packInt32(static_cast<int32_t>(device))),
      py::reinterpret_steal<py::object>(
          THPUtils_packInt64(static_cast<int64_t>(size))));
}

void AllocDeleteHook::call_delete_hook(void* ptr, int device, size_t size) {
  py::gil_scoped_acquire acquire;
  auto delete_hook = py::reinterpret_borrow<py::function>(delete_hook_);
  delete_hook(
      py::reinterpret_steal<py::object>(
          THPUtils_packInt64(reinterpret_cast<int64_t>(ptr))),
      py::reinterpret_steal<py::object>(
          THPUtils_packInt32(static_cast<int32_t>(device))),
      py::reinterpret_steal<py::object>(
          THPUtils_packInt64(static_cast<int64_t>(size))));
}

void AllocDeleteHook::register_hook(
    py::function& alloc_hook,
    py::function& delete_hook) {
  alloc_hook_ = alloc_hook.release().ptr();
  delete_hook_ = delete_hook.release().ptr();

  c10::cuda::CUDACachingAllocator::register_alloc_delete_hook(
      [&](void* ptr, int device, size_t size) {
        call_alloc_hook(ptr, device, size);
      },
      [&](void* ptr, int device, size_t size) {
        call_delete_hook(ptr, device, size);
      });
}

void AllocDeleteHook::remove_hook() {
  if (Py_IsInitialized()) {
    py::gil_scoped_acquire gil;
    Py_XDECREF(alloc_hook_);
    Py_XDECREF(delete_hook_);
  }
  alloc_hook_ = nullptr;
  delete_hook_ = nullptr;
  c10::cuda::CUDACachingAllocator::reset_alloc_delete_hook();
}

} // namespace autograd
} // namespace torch

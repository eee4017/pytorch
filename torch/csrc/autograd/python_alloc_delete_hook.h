#pragma once

#include <pybind11/pybind11.h>
#include <torch/csrc/autograd/python_variable.h>
#include <torch/csrc/python_headers.h>
#include <torch/csrc/Export.h>
#include <ATen/ATen.h>

namespace py = pybind11;

namespace torch { namespace autograd {

struct AllocDeleteHook {
  void register_hook(py::function &alloc_hook, py::function &delete_hook);
  void remove_hook();
  void call_alloc_hook(void *ptr, int device, size_t size);
  void call_delete_hook(void *ptr, int device, size_t size);

private:
  PyObject* alloc_hook_;
  PyObject* delete_hook_;
};

static AllocDeleteHook py_alloc_delete_hook;

}}

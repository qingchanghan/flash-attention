#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>

#include "ATen/cuda/CUDAContext.h"
#include "kernels.h"

namespace kernels {}  // namespace kernels

std::vector<at::Tensor> matrix_transpose(const at::Tensor &input,
                                         std::string version) {
  auto itype = input.scalar_type();
  auto otype = itype;

  TORCH_CHECK(input.is_cuda());
  TORCH_CHECK(input.is_contiguous());
  TORCH_CHECK(input.dim() == 2);

  const int rows = input.size(0);
  const int cols = input.size(1);

  auto opts = input.options();
  auto output = torch::empty(c10::IntArrayRef{cols, rows}, opts.dtype(otype));

  auto stream = at::cuda::getCurrentCUDAStream().stream();

  // 选择类型
  TYPE_SWITCH(itype == torch::kFloat32, itype == torch::kFloat16, [&] {
    // 定义函数指针类型
    using FuncType = void (*)(const elem_type *, elem_type *, const int,
                              const int, cudaStream_t);
    // 选择函数
    FuncType func;
    if (version == "v0") {
      func = matrix_transpose_cuda_v0;
    } else if (version == "v1") {
      func = matrix_transpose_cuda_v1;
    } else if (version == "v2") {
      func = matrix_transpose_cuda_v2;
    } else if (version == "v3") {
      func = matrix_transpose_cuda_v3;
    } else if (version == "v4") {
      func = matrix_transpose_cuda_v4;
    } else if (version == "v5") {
      func = matrix_transpose_cuda_v5;
    } else if (version == "v6") {
      func = matrix_transpose_cuda_v6;
    } else {
      assert(false);
    }
    func((elem_type *)input.data_ptr(), (elem_type *)output.data_ptr(), rows,
         cols, stream);
  });

  C10_CUDA_CHECK(cudaGetLastError());

  return {output};
}

std::vector<at::Tensor> reduce_sum(const at::Tensor &input,
                                   std::string version) {
  auto itype = input.scalar_type();
  auto otype = itype;

  TORCH_CHECK(input.is_cuda());
  TORCH_CHECK(input.is_contiguous());
  TORCH_CHECK(input.dim() == 2);

  const int rows = input.size(0);
  const int cols = input.size(1);

  auto opts = input.options();
  auto output = torch::empty(c10::IntArrayRef{rows}, opts.dtype(otype));

  auto stream = at::cuda::getCurrentCUDAStream().stream();

  // 选择类型
  TYPE_SWITCH(itype == torch::kFloat32, itype == torch::kFloat16, [&] {
    // 定义函数指针类型
    using FuncType = void (*)(const elem_type *, elem_type *, const int,
                              const int, cudaStream_t);
    // 选择函数
    FuncType func;
    if (version == "v0") {
      func = reduce_sum_cuda_v0;
    } else {
      assert(false);
    }
    func((elem_type *)input.data_ptr(), (elem_type *)output.data_ptr(), rows,
         cols, stream);
  });

  C10_CUDA_CHECK(cudaGetLastError());

  return {output};
}

std::vector<at::Tensor> cumsum(const at::Tensor &input, std::string version) {
  auto itype = input.scalar_type();
  auto otype = itype;

  TORCH_CHECK(input.is_cuda());
  TORCH_CHECK(input.is_contiguous());
  TORCH_CHECK(input.dim() == 2);

  const int rows = input.size(0);
  const int cols = input.size(1);

  auto opts = input.options();
  auto output = torch::empty(c10::IntArrayRef{rows, cols}, opts.dtype(otype));
  auto partial_sums =
      torch::empty(c10::IntArrayRef{rows * cols}, opts.dtype(otype));

  auto stream = at::cuda::getCurrentCUDAStream().stream();

  // 选择类型
  TYPE_SWITCH(itype == torch::kFloat32, itype == torch::kFloat16, [&] {
    // 定义函数指针类型
    using FuncType = void (*)(const elem_type *, elem_type *, elem_type *,
                              const int, const int, cudaStream_t);
    // 选择函数
    FuncType func;
    if (version == "v0") {
      func = scan_cuda_v0;
    } else {
      assert(false);
    }
    func((elem_type *)input.data_ptr(), (elem_type *)output.data_ptr(),
         (elem_type *)partial_sums.data_ptr(), rows, cols, stream);
  });

  C10_CUDA_CHECK(cudaGetLastError());

  return {output};
}

std::vector<at::Tensor> softmax(const at::Tensor &input, std::string version) {
  auto itype = input.scalar_type();
  auto otype = itype;

  TORCH_CHECK(input.is_cuda());
  TORCH_CHECK(input.is_contiguous());
  TORCH_CHECK(input.dim() == 2);

  const int rows = input.size(0);
  const int cols = input.size(1);

  auto opts = input.options();
  auto output = torch::empty(c10::IntArrayRef{rows, cols}, opts.dtype(otype));

  auto stream = at::cuda::getCurrentCUDAStream().stream();

  // 选择类型
  TYPE_SWITCH(itype == torch::kFloat32, itype == torch::kFloat16, [&] {
    // 定义函数指针类型
    using FuncType = void (*)(const elem_type *, elem_type *, const int,
                              const int, cudaStream_t);
    // 选择函数
    FuncType func;
    if (version == "v0") {
      func = softmax_cuda_v0;
    } else if (version == "v1") {
      func = softmax_cuda_v1;
    } else if (version == "v2") {
      func = softmax_cuda_v2;
    } else {
      assert(false);
    }
    func((elem_type *)input.data_ptr(), (elem_type *)output.data_ptr(), rows,
         cols, stream);
  });

  C10_CUDA_CHECK(cudaGetLastError());

  return {output};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.doc() = "CUDA kernels for learn";
  m.def("matrix_transpose", &matrix_transpose, "matrix transpose kernel",
        py::arg("input"), py::arg("version"));
  m.def("reduce_sum", &reduce_sum, "reduce sum kernel", py::arg("input"),
        py::arg("version"));
  m.def("cumsum", &cumsum, "cumsum kernel", py::arg("input"),
        py::arg("version"));
  m.def("softmax", &softmax, "softmax kernel", py::arg("input"),
        py::arg("version"));
}

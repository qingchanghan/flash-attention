#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>

#include "ATen/cuda/CUDAContext.h"
#include "kernels.h"

namespace kernels {}  // namespace kernels

std::vector<at::Tensor> matrix_transpose(
    const at::Tensor &input,  // Input: BxSxhidden_size
    std::string version) {
  auto itype = input.scalar_type();
  auto otype = itype;

  TORCH_CHECK(input.is_cuda());
  TORCH_CHECK(input.is_contiguous());
  TORCH_CHECK(input.dim() == 2);
  // c10::IntArrayRef does not own the storage, so we need to construct a
  // vector. Otherwise just constructing IntArrayRef({blah}) will cause
  // uninitialized memory because blah is then deallocated.
  std::vector<int64_t> sizes_vec{input.size(1), input.size(0)};
  auto sizes = c10::IntArrayRef(sizes_vec);
  TORCH_CHECK(sizes.size() == 2);

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
}

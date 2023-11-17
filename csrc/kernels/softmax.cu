#include <assert.h>
#include <c10/macros/Macros.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <stdint.h>

#include <cfloat>
#include <limits>

#include "common.cuh"
#include "cuda_utils.h"
#include "kernels.h"

namespace {

// v0: 每个block处理cols个数据
// 假设cols <= 256 * 64
template <typename T>
__global__ void softmax_ker_v0(const T* input, T* output, const int rows,
                               const int cols) {
  constexpr int COLS_PER_THREAD = 64;
  const int row_idx = blockIdx.x;
  const T* cur_input = input + row_idx * cols;
  T* cur_output = output + row_idx * cols;

  __shared__ float val_max;
  __shared__ float val_sum;

  float vals[COLS_PER_THREAD];
  float local_max = -1e20f;

#pragma unroll
  for (int i = threadIdx.x, j = 0; i < cols; i += blockDim.x, j++) {
    vals[j] = (float)(cur_input[i]);
    local_max = max(local_max, vals[j]);
  }

  // block reduce max
  local_max = block_reduce_max<float>(local_max);
  if (threadIdx.x == 0) {
    val_max = local_max;
  }
  __syncthreads();

  local_max = val_max;
  float local_sum = 0.f;
#pragma unroll
  for (int i = threadIdx.x, j = 0; i < cols; i += blockDim.x, j++) {
    vals[j] = __expf(vals[j] - local_max);
    local_sum = local_sum + vals[j];
  }

  // block reduce sum
  local_sum = block_reduce_sum<float>(local_sum);
  if (threadIdx.x == 0) {
    val_sum = local_sum;
  }
  __syncthreads();

  local_sum = val_sum;
#pragma unroll
  for (int i = threadIdx.x, j = 0; i < cols; i += blockDim.x, j++) {
    cur_output[i] = (T)(vals[j] / local_sum);
  }
}

// v1: 向量化
template <typename T, int N>
__global__ void softmax_ker_v1(const T* input, T* output, const int rows,
                               const int cols) {
  using VT = typename VecType<T, N>::Type;
  using VFloat = typename VecType<float, N>::Type;
  constexpr int COLS_PER_THREAD = 16;
  const int row_idx = blockIdx.x;
  const T* cur_input = input + row_idx * cols;
  T* cur_output = output + row_idx * cols;

  const int col_start = threadIdx.x * N;
  const int col_stride_for_thread = blockDim.x * N;

  __shared__ float val_max;
  __shared__ float val_sum;

  float vals[COLS_PER_THREAD];
  float local_max = -1e20f;

#pragma unroll
  for (int i = col_start, j = 0; i < cols; i += col_stride_for_thread, j += N) {
    VT tmp = FETCH_CONST_VT(cur_input[i], VT);
    FETCH_VT(vals[j], VFloat) = converter<VT, VFloat>(tmp);
    local_max =
        max(local_max, vec_max<float, VFloat>(FETCH_VT(vals[j], VFloat)));
  }

  // block reduce max
  local_max = block_reduce_max<float>(local_max);
  if (threadIdx.x == 0) {
    val_max = local_max;
  }
  __syncthreads();

  local_max = val_max;
  float local_sum = 0.f;
#pragma unroll
  for (int i = col_start, j = 0; i < cols; i += col_stride_for_thread, j += N) {
#pragma unroll
    for (int k = 0; k < N; k++) {
      vals[j + k] = __expf(vals[j + k] - local_max);
      local_sum = local_sum + vals[j + k];
    }
  }

  // block reduce sum
  local_sum = block_reduce_sum<float>(local_sum);
  if (threadIdx.x == 0) {
    val_sum = local_sum;
  }
  __syncthreads();

  local_sum = val_sum;
#pragma unroll
  for (int i = col_start, j = 0; i < cols; i += col_stride_for_thread, j += N) {
#pragma unroll
    for (int k = 0; k < N; k++) {
      vals[j + k] = vals[j + k] / local_sum;
    }
    FETCH_VT(cur_output[i], VT) =
        converter<VFloat, VT>(FETCH_VT(vals[j], VFloat));
  }
}

// v2: 向量化 + shared mem
template <typename T, int N>
__global__ void softmax_ker_v2(const T* input, T* output, const int rows,
                               const int cols) {
  extern __shared__ __align__(sizeof(double)) unsigned char shared_buf[];
  float* buf = reinterpret_cast<float*>(shared_buf);

  using VT = typename VecType<T, N>::Type;
  const int row_idx = blockIdx.x;
  const T* cur_input = input + row_idx * cols;
  T* cur_output = output + row_idx * cols;

  __shared__ float val_max;
  __shared__ float val_sum;

  float local_max = -1e20f;

  const int num_packs = cols / N;
#pragma unroll
  for (int i = threadIdx.x; i < num_packs; i += blockDim.x) {
    T tmp[N];
    FETCH_VT(tmp, VT) = FETCH_CONST_VT(cur_input[i * N], VT);
#pragma unroll
    for (int j = 0; j < N; j++) {
      buf[i * N + j] = tmp[j];
      local_max = max(local_max, tmp[j]);
    }
  }

  // block reduce max
  local_max = block_reduce_max<float>(local_max);
  if (threadIdx.x == 0) {
    val_max = local_max;
  }
  __syncthreads();

  local_max = val_max;

  float local_sum = 0.f;
#pragma unroll
  for (int i = threadIdx.x; i < cols; i += blockDim.x) {
    const float cur_val = __expf(buf[i] - local_max);
    buf[i] = cur_val;
    local_sum = local_sum + cur_val;
  }

  // block reduce sum
  local_sum = block_reduce_sum<float>(local_sum);
  if (threadIdx.x == 0) {
    val_sum = local_sum;
  }
  __syncthreads();

  local_sum = val_sum;

#pragma unroll
  for (int i = threadIdx.x; i < num_packs; i += blockDim.x) {
    T tmp[N];
#pragma unroll
    for (int j = 0; j < N; j++) {
      tmp[j] = static_cast<T>(buf[i * N + j] / local_sum);
    }
    FETCH_VT(cur_output[i * N], VT) = FETCH_VT(tmp, VT);
  }
}

}  // namespace

template <typename T>
void softmax_cuda_v0(const T* input, T* output, const int rows, const int cols,
                     cudaStream_t stream) {
  int block = min(1024, cols);
  int grid = rows;

  softmax_ker_v0<T><<<grid, block, 0, stream>>>(input, output, rows, cols);
}

#define INSTANTIATE_SOFTMAX_V0(T)                                          \
  template void softmax_cuda_v0(const T* input, T* output, const int rows, \
                                const int cols, cudaStream_t stream);

INSTANTIATE_SOFTMAX_V0(float)
INSTANTIATE_SOFTMAX_V0(half)
INSTANTIATE_SOFTMAX_V0(__nv_bfloat16)

template <typename T>
void softmax_cuda_v1(const T* input, T* output, const int rows, const int cols,
                     cudaStream_t stream) {
  int block = min(1024, cols);
  int grid = rows;

  if (cols % 4 == 0) {
    softmax_ker_v1<T, 4><<<grid, block, 0, stream>>>(input, output, rows, cols);
  } else if (cols % 2 == 0) {
    softmax_ker_v1<T, 2><<<grid, block, 0, stream>>>(input, output, rows, cols);
  } else {
    softmax_ker_v1<T, 1><<<grid, block, 0, stream>>>(input, output, rows, cols);
  }
}

#define INSTANTIATE_SOFTMAX_V1(T)                                          \
  template void softmax_cuda_v1(const T* input, T* output, const int rows, \
                                const int cols, cudaStream_t stream);

INSTANTIATE_SOFTMAX_V1(float)
INSTANTIATE_SOFTMAX_V1(half)
INSTANTIATE_SOFTMAX_V1(__nv_bfloat16)

template <typename T>
void softmax_cuda_v2(const T* input, T* output, const int rows, const int cols,
                     cudaStream_t stream) {
  int block = min(512, cols);
  int grid = rows;

  const size_t smem_size = cols * sizeof(float);

  if (!std::is_same<T, float>::value && cols % 8 == 0) {
    if (std::is_same<T, half>::value) {
      if (smem_size >= (48 << 10)) {
        check_cuda_error(cudaFuncSetAttribute(
            softmax_ker_v2<half, 8>,
            cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
      }
      softmax_ker_v2<half, 8><<<grid, block, smem_size, stream>>>(
          reinterpret_cast<const half*>(input), reinterpret_cast<half*>(output),
          rows, cols);
    } else if (std::is_same<T, __nv_bfloat16>::value) {
      if (smem_size >= (48 << 10)) {
        check_cuda_error(cudaFuncSetAttribute(
            softmax_ker_v2<__nv_bfloat16, 8>,
            cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
      }
      softmax_ker_v2<__nv_bfloat16, 8><<<grid, block, smem_size, stream>>>(
          reinterpret_cast<const __nv_bfloat16*>(input),
          reinterpret_cast<__nv_bfloat16*>(output), rows, cols);
    }
  } else if (cols % 4 == 0) {
    if (smem_size >= (48 << 10)) {
      check_cuda_error(cudaFuncSetAttribute(
          softmax_ker_v2<T, 4>, cudaFuncAttributeMaxDynamicSharedMemorySize,
          smem_size));
    }
    softmax_ker_v2<T, 4>
        <<<grid, block, smem_size, stream>>>(input, output, rows, cols);
  } else if (cols % 2 == 0) {
    if (smem_size >= (48 << 10)) {
      check_cuda_error(cudaFuncSetAttribute(
          softmax_ker_v2<T, 2>, cudaFuncAttributeMaxDynamicSharedMemorySize,
          smem_size));
    }
    softmax_ker_v2<T, 2>
        <<<grid, block, smem_size, stream>>>(input, output, rows, cols);
  } else {
    if (smem_size >= (48 << 10)) {
      check_cuda_error(cudaFuncSetAttribute(
          softmax_ker_v2<T, 1>, cudaFuncAttributeMaxDynamicSharedMemorySize,
          smem_size));
    }
    softmax_ker_v2<T, 1>
        <<<grid, block, smem_size, stream>>>(input, output, rows, cols);
  }
}

#define INSTANTIATE_SOFTMAX_V2(T)                                          \
  template void softmax_cuda_v2(const T* input, T* output, const int rows, \
                                const int cols, cudaStream_t stream);

INSTANTIATE_SOFTMAX_V2(float)
INSTANTIATE_SOFTMAX_V2(half)
INSTANTIATE_SOFTMAX_V2(__nv_bfloat16)


#include <assert.h>
#include <c10/macros/Macros.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <stdint.h>

#include <cfloat>
#include <limits>

#include "cuda_utils.h"
#include "kernels.h"

namespace {

template <typename T>
__device__ T warp_reduce_sum(T val) {
#pragma unroll
  for (int mask = 16; mask > 0; mask >>= 1) {
    val = val + __shfl_xor_sync(0xffffffff, val, mask, 32);
  }
  return val;
}

template <typename T>
__device__ T block_reduce_sum(T val) {
  static __shared__ T shm[32];
  int lane_id = threadIdx.x & 0x1f;
  int warp_id = threadIdx.x >> 5;

  val = warp_reduce_sum<T>(val);
  if (lane_id == 0) {
    shm[warp_id] = val;
  }
  __syncthreads();
  val = shm[lane_id];
  val = warp_reduce_sum<T>(val);
  return val;
}

template <typename T>
__global__ void reduce_sum_ker_v0(const T* input, T* output, const int rows,
                                  const int cols) {
  int row_idx = blockIdx.x;
  if (row_idx >= rows) return;
  const T* cur_input = input + row_idx * cols;
  T local_sum = (T)0.;
  for (int i = threadIdx.x; i < cols; i += blockDim.x) {
    local_sum = local_sum + cur_input[i];
  }

  local_sum = block_reduce_sum<T>(local_sum);
  if (threadIdx.x == 0) {
    output[row_idx] = local_sum;
  }
}

}  // namespace

template <typename T>
void reduce_sum_cuda_v0(const T* input, T* output, const int rows,
                        const int cols, cudaStream_t stream) {
  reduce_sum_ker_v0<T><<<rows, 1024, 0, stream>>>(input, output, rows, cols);
}

#define INSTANTIATE_REDUCE_SUM_V0(T)                                          \
  template void reduce_sum_cuda_v0(const T* input, T* output, const int rows, \
                                   const int cols, cudaStream_t stream);

INSTANTIATE_REDUCE_SUM_V0(float)
INSTANTIATE_REDUCE_SUM_V0(half)
INSTANTIATE_REDUCE_SUM_V0(__nv_bfloat16)

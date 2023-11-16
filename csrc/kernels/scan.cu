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

// grid: [cols / BLOCK_SIZE, rows]
// block: [BLOCK_SIZE]
template <typename T>
__global__ void scan_ker_v0(const T* input, T* output, T* partial_sums,
                            const int rows, const int cols) {
  __shared__ T shm[WARP_SIZE];

  const int row_idx = blockIdx.y;
  const int col_idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int idx = row_idx * cols + col_idx;

  const int lane_id = threadIdx.x % WARP_SIZE;
  const int warp_id = threadIdx.x / WARP_SIZE;

  T val = (T)0.;
  if (col_idx < cols) {
    val = input[idx];
  }

  // warp scan
#pragma unroll
  for (int delta = 1; delta <= WARP_SIZE; delta = delta << 1) {
    T last_sum = __shfl_up_sync(0xffffffff, val, delta, WARP_SIZE);

    if (lane_id >= delta) {
      val = val + last_sum;
    }
  }

  if (lane_id == WARP_SIZE - 1) {
    shm[warp_id] = val;
  }

  __syncthreads();

  // scan for all warp
  if (warp_id == 0) {
    T warp_sum = (lane_id < WARP_NUM) ? shm[lane_id] : (T)0.;
#pragma unroll
    for (int delta = 1; delta <= WARP_SIZE; delta = delta << 1) {
      T last_sum = __shfl_up_sync(0xffffffff, warp_sum, delta, WARP_SIZE);

      if (lane_id >= delta) {
        warp_sum = warp_sum + last_sum;
      }
    }

    if (lane_id < WARP_NUM) {
      shm[lane_id] = warp_sum;
    }
  }

  __syncthreads();

  // add warp sum to all threads
  if (warp_id > 0 && warp_id < WARP_NUM) {
    val = val + shm[warp_id - 1];
  }
  if (col_idx < cols) output[idx] = val;

  if (partial_sums != nullptr && threadIdx.x == blockDim.x - 1) {
    partial_sums[blockIdx.y * gridDim.x + blockIdx.x] = val;
  }
}

template <typename T>
__global__ void uniform_add(T* output, T* partial_sums, const int rows,
                            const int cols) {
  const int row_idx = blockIdx.y;
  const int col_idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int idx = row_idx * cols + col_idx;

  if (blockIdx.x == 0 || col_idx >= cols) return;

  T block_sum = partial_sums[blockIdx.y * gridDim.x + blockIdx.x - 1];
  output[idx] = output[idx] + block_sum;
}
}  // namespace

template <typename T>
void scan_cuda_v0(const T* input, T* output, T* partial_sums, const int rows,
                  const int cols, cudaStream_t stream) {
  int block_size = BLOCK_SIZE;
  int grid_x = (cols + block_size - 1) / block_size;
  dim3 grid(grid_x, rows, 1);
  dim3 block = block_size;

  scan_ker_v0<T>
      <<<grid, block, 0, stream>>>(input, output, partial_sums, rows, cols);
  int partial_grid_x = (grid_x + block_size - 1) / block_size;
  dim3 partial_grid(partial_grid_x, rows, 1);
  scan_ker_v0<T><<<partial_grid, block, 0, stream>>>(partial_sums, partial_sums,
                                                     nullptr, rows, grid_x);

  uniform_add<T><<<grid, block, 0, stream>>>(output, partial_sums, rows, cols);
}

#define INSTANTIATE_SCAN_V0(T)                                           \
  template void scan_cuda_v0(const T* input, T* output, T* partial_sums, \
                             const int rows, const int cols,             \
                             cudaStream_t stream);

INSTANTIATE_SCAN_V0(float)
INSTANTIATE_SCAN_V0(half)
INSTANTIATE_SCAN_V0(__nv_bfloat16)


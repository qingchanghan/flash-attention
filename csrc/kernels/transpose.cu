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

// v0: naive 实现
template <typename T>
__global__ void matrix_transpose_ker_v0(const T* input, T* output,
                                        const int rows, const int cols) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= rows * cols) return;
  const int row_idx = idx / cols;
  const int col_idx = idx % cols;
  output[col_idx * rows + row_idx] = input[row_idx * cols + col_idx];
}

// v1: 采用shared memory缓存
template <typename T>
__global__ void matrix_transpose_ker_v1(const T* input, T* output,
                                        const int rows, const int cols) {
  __shared__ T shm[32][32];
  const int tidx = threadIdx.x;
  const int tidy = threadIdx.y;
  int col_idx = blockIdx.x * 32 + tidx;
  int row_idx = blockIdx.y * 32 + tidy;
  if (col_idx < cols && row_idx < rows) {
    shm[tidx][tidy] = input[row_idx * cols + col_idx];
  }

  __syncthreads();
  col_idx = blockIdx.x * 32 + tidy;
  row_idx = blockIdx.y * 32 + tidx;
  if (col_idx < cols && row_idx < rows) {
    output[col_idx * rows + row_idx] = shm[tidy][tidx];
  }
}

// v2: v1 + 避免bank conflict
template <typename T>
__global__ void matrix_transpose_ker_v2(const T* input, T* output,
                                        const int rows, const int cols) {
  __shared__ T shm[32][33];
  const int tidx = threadIdx.x;
  const int tidy = threadIdx.y;
  int col_idx = blockIdx.x * 32 + tidx;
  int row_idx = blockIdx.y * 32 + tidy;
  if (col_idx < cols && row_idx < rows) {
    shm[tidx][tidy] = input[row_idx * cols + col_idx];
  }

  __syncthreads();
  col_idx = blockIdx.x * 32 + tidy;
  row_idx = blockIdx.y * 32 + tidx;
  if (col_idx < cols && row_idx < rows) {
    output[col_idx * rows + row_idx] = shm[tidy][tidx];
  }
}

// v3: v1 + 向量化访问 + 避免bank conflict
template <typename T, int N>
__global__ void matrix_transpose_ker_v3(const T* input, T* output,
                                        const int rows, const int cols) {
  constexpr int SIZE = 32 * N;
  using VT = typename VecType<T, N>::Type;
  __shared__ T shm[SIZE][SIZE + 4];
  const int tidx = threadIdx.x;
  const int tidy = threadIdx.y;

  // 读取input
  // 行方向一次访问 列方向循环访问
  int col_idx = blockIdx.x * SIZE + tidx * N;
  int row_idx = blockIdx.y * SIZE + tidy;

  if (col_idx < cols) {
#pragma unroll
    for (int tidy_i = tidy; row_idx < (blockIdx.y + 1) * SIZE && row_idx < rows;
         row_idx += blockDim.y, tidy_i += blockDim.y) {
      reinterpret_cast<VT*>(&(shm[tidy_i][tidx * N]))[0] =
          *reinterpret_cast<const VT*>(input + row_idx * cols + col_idx);
    }
  }

  __syncthreads();

  // 局部转置: 目的是写回output时 仍然可以向量化
#pragma unroll
  for (row_idx = tidy; row_idx < SIZE; row_idx += blockDim.y) {
#pragma unroll
    for (col_idx = tidx; col_idx < SIZE; col_idx += blockDim.x) {
      // 只操作下三角即可
      if (row_idx > col_idx) {
        T tmp = shm[row_idx][col_idx];
        shm[row_idx][col_idx] = shm[col_idx][row_idx];
        shm[col_idx][row_idx] = tmp;
      }
    }
  }

  __syncthreads();

  // 写回output
  // 仍是 行方向一次访问 列方向循环访问
  // 但行列对调了
  // 所以实际是 列方向一次访问 行方向循环访问
  col_idx = blockIdx.x * SIZE + tidx;
  row_idx = blockIdx.y * SIZE + tidy * N;
  if (row_idx < rows) {
#pragma unroll
    for (int tidx_i = tidx; col_idx < (blockIdx.x + 1) * SIZE && col_idx < cols;
         col_idx += blockDim.x, tidx_i += blockDim.x) {
      reinterpret_cast<VT*>(output + col_idx * rows + row_idx)[0] =
          *reinterpret_cast<VT*>(&(shm[tidx_i][tidy * N]));
    }
  }
}

// v4: from nvidia
// https://github.com/NVIDIA-developer-blog/code-samples/blob/master/series/cuda-cpp/transpose/transpose.cu
const int TILE_DIM = 32;
const int BLOCK_ROWS = 8;
template <typename T>
__global__ void matrix_transpose_ker_v4(T* odata, const T* idata) {
  __shared__ T tile[TILE_DIM][TILE_DIM + 1];

  int x = blockIdx.x * TILE_DIM + threadIdx.x;
  int y = blockIdx.y * TILE_DIM + threadIdx.y;
  int width = gridDim.x * TILE_DIM;

  for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS)
    tile[threadIdx.y + j][threadIdx.x] = idata[(y + j) * width + x];

  __syncthreads();

  x = blockIdx.y * TILE_DIM + threadIdx.x;  // transpose block offset
  y = blockIdx.x * TILE_DIM + threadIdx.y;

  width = gridDim.y * TILE_DIM;

  for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS)
    odata[(y + j) * width + x] = tile[threadIdx.x][threadIdx.y + j];
}

// v5: v3 + 优化写回时的合并访存
template <typename T, int N>
__global__ void matrix_transpose_ker_v5(const T* input, T* output,
                                        const int rows, const int cols) {
  constexpr int SIZE = 32 * N;
  using VT = typename VecType<T, N>::Type;
  __shared__ T shm[SIZE][SIZE + 4];
  const int tidx = threadIdx.x;
  const int tidy = threadIdx.y;

  // 读取input
  // 行方向一次访问 列方向循环访问
  int col_idx = blockIdx.x * SIZE + tidx * N;
  int row_idx = blockIdx.y * SIZE + tidy;

  if (col_idx < cols) {
#pragma unroll
    for (int tidy_i = tidy; row_idx < (blockIdx.y + 1) * SIZE && row_idx < rows;
         row_idx += blockDim.y, tidy_i += blockDim.y) {
      reinterpret_cast<VT*>(&(shm[tidy_i][tidx * N]))[0] =
          *reinterpret_cast<const VT*>(input + row_idx * cols + col_idx);
    }
  }

  __syncthreads();

  // 局部转置: 目的是写回output时 仍然可以向量化
#pragma unroll
  for (row_idx = tidy; row_idx < SIZE; row_idx += blockDim.y) {
#pragma unroll
    for (col_idx = tidx; col_idx < SIZE; col_idx += blockDim.x) {
      // 只操作下三角即可
      if (row_idx > col_idx) {
        T tmp = shm[row_idx][col_idx];
        shm[row_idx][col_idx] = shm[col_idx][row_idx];
        shm[col_idx][row_idx] = tmp;
      }
    }
  }

  __syncthreads();

  // 写回output
  col_idx = blockIdx.y * SIZE + tidx * N;
  row_idx = blockIdx.x * SIZE + tidy;
  if (row_idx < cols) {
#pragma unroll
    for (int tidx_i = tidy; row_idx < (blockIdx.x + 1) * SIZE && row_idx < cols;
         row_idx += blockDim.x, tidx_i += blockDim.x) {
      reinterpret_cast<VT*>(output + row_idx * rows + col_idx)[0] =
          *reinterpret_cast<VT*>(&(shm[tidx_i][tidx * N]));
    }
  }
}

// v6: v5 +
template <typename T, int N>
__global__ void matrix_transpose_ker_v6(const T* input, T* output,
                                        const int rows, const int cols) {
  constexpr int SIZE = 32 * N;
  using VT = typename VecType<T, N>::Type;
  // __shared__ T* shm;  // [SIZE, SIZE + 4]
  extern __shared__ char smem[];
  T* shm = reinterpret_cast<T*>(smem);
  constexpr int shared_stride = SIZE + N;
  const int tidx = threadIdx.x;
  const int tidy = threadIdx.y;

  // 读取input
  // 行方向一次访问 列方向循环访问
  int col_idx = blockIdx.x * SIZE + tidx * N;
  int row_idx = blockIdx.y * SIZE + tidy;

  if (col_idx < cols) {
#pragma unroll
    for (int tidy_i = tidy; row_idx < (blockIdx.y + 1) * SIZE && row_idx < rows;
         row_idx += blockDim.y, tidy_i += blockDim.y) {
      reinterpret_cast<VT*>(&(shm[tidy_i * shared_stride + tidx * N]))[0] =
          *reinterpret_cast<const VT*>(input + row_idx * cols + col_idx);
    }
  }

  __syncthreads();

  // 局部转置: 目的是写回output时 仍然可以向量化
#pragma unroll
  for (row_idx = tidy; row_idx < SIZE; row_idx += blockDim.y) {
#pragma unroll
    for (col_idx = tidx; col_idx < SIZE; col_idx += blockDim.x) {
      // 只操作下三角即可
      if (row_idx > col_idx) {
        T tmp = shm[row_idx * shared_stride + col_idx];
        shm[row_idx * shared_stride + col_idx] =
            shm[col_idx * shared_stride + row_idx];
        shm[col_idx * shared_stride + row_idx] = tmp;
      }
    }
  }

  __syncthreads();

  // 写回output
  col_idx = blockIdx.y * SIZE + tidx * N;
  row_idx = blockIdx.x * SIZE + tidy;
  if (row_idx < cols) {
#pragma unroll
    for (int tidx_i = tidy; row_idx < (blockIdx.x + 1) * SIZE && row_idx < cols;
         row_idx += blockDim.x, tidx_i += blockDim.x) {
      reinterpret_cast<VT*>(output + row_idx * rows + col_idx)[0] =
          *reinterpret_cast<VT*>(&(shm[tidx_i * shared_stride + tidx * N]));
    }
  }
}

}  // namespace

template <typename T>
void matrix_transpose_cuda_v0(const T* input, T* output, const int rows,
                              const int cols, cudaStream_t stream) {
  int block_dim = min(1024, rows * cols);
  int grid_dim = (rows * cols + block_dim - 1) / block_dim;
  matrix_transpose_ker_v0<<<grid_dim, block_dim, 0, stream>>>(input, output,
                                                              rows, cols);
}

#define INSTANTIATE_MATRIX_TRANSPOSE_V0(T)                               \
  template void matrix_transpose_cuda_v0(const T* input, T* output,      \
                                         const int rows, const int cols, \
                                         cudaStream_t stream);

INSTANTIATE_MATRIX_TRANSPOSE_V0(float)
INSTANTIATE_MATRIX_TRANSPOSE_V0(half)
INSTANTIATE_MATRIX_TRANSPOSE_V0(__nv_bfloat16)

template <typename T>
void matrix_transpose_cuda_v1(const T* input, T* output, const int rows,
                              const int cols, cudaStream_t stream) {
  dim3 grid_dim(cols / 32, rows / 32);
  dim3 block_dim(32, 32);
  matrix_transpose_ker_v1<<<grid_dim, block_dim, 0, stream>>>(input, output,
                                                              rows, cols);
}

#define INSTANTIATE_MATRIX_TRANSPOSE_V1(T)                               \
  template void matrix_transpose_cuda_v1(const T* input, T* output,      \
                                         const int rows, const int cols, \
                                         cudaStream_t stream);

INSTANTIATE_MATRIX_TRANSPOSE_V1(float)
INSTANTIATE_MATRIX_TRANSPOSE_V1(half)
INSTANTIATE_MATRIX_TRANSPOSE_V1(__nv_bfloat16)

template <typename T>
void matrix_transpose_cuda_v2(const T* input, T* output, const int rows,
                              const int cols, cudaStream_t stream) {
  dim3 grid_dim(cols / 32, rows / 32);
  dim3 block_dim(32, 32);
  matrix_transpose_ker_v2<<<grid_dim, block_dim, 0, stream>>>(input, output,
                                                              rows, cols);
}

#define INSTANTIATE_MATRIX_TRANSPOSE_V2(T)                               \
  template void matrix_transpose_cuda_v2(const T* input, T* output,      \
                                         const int rows, const int cols, \
                                         cudaStream_t stream);

INSTANTIATE_MATRIX_TRANSPOSE_V2(float)
INSTANTIATE_MATRIX_TRANSPOSE_V2(half)
INSTANTIATE_MATRIX_TRANSPOSE_V2(__nv_bfloat16)

template <typename T>
void matrix_transpose_cuda_v3(const T* input, T* output, const int rows,
                              const int cols, cudaStream_t stream) {
  if (std::is_same<T, float>::value) {
    constexpr int N = 2;
    dim3 grid_dim((cols + 32 * N - 1) / (32 * N),
                  (rows + 32 * N - 1) / (32 * N));
    dim3 block_dim(32, 32);
    matrix_transpose_ker_v3<float, N><<<grid_dim, block_dim, 0, stream>>>(
        reinterpret_cast<const float*>(input), reinterpret_cast<float*>(output),
        rows, cols);
  } else if (std::is_same<T, half>::value) {
    constexpr int N = 2;
    dim3 grid_dim((cols + 32 * N - 1) / (32 * N),
                  (rows + 32 * N - 1) / (32 * N));
    dim3 block_dim(32, 32);
    matrix_transpose_ker_v3<half, N><<<grid_dim, block_dim, 0, stream>>>(
        reinterpret_cast<const half*>(input), reinterpret_cast<half*>(output),
        rows, cols);
  } else if (std::is_same<T, __nv_bfloat16>::value) {
    constexpr int N = 4;
    dim3 grid_dim((cols + 32 * N - 1) / (32 * N),
                  (rows + 32 * N - 1) / (32 * N));
    dim3 block_dim(32, 32);
    matrix_transpose_ker_v3<__nv_bfloat16, N>
        <<<grid_dim, block_dim, 0, stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(input),
            reinterpret_cast<__nv_bfloat16*>(output), rows, cols);
  }
}

#define INSTANTIATE_MATRIX_TRANSPOSE_V3(T)                               \
  template void matrix_transpose_cuda_v3(const T* input, T* output,      \
                                         const int rows, const int cols, \
                                         cudaStream_t stream);

INSTANTIATE_MATRIX_TRANSPOSE_V3(float)
INSTANTIATE_MATRIX_TRANSPOSE_V3(half)
INSTANTIATE_MATRIX_TRANSPOSE_V3(__nv_bfloat16)

template <typename T>
void matrix_transpose_cuda_v4(const T* input, T* output, const int rows,
                              const int cols, cudaStream_t stream) {
  dim3 dimGrid(cols / TILE_DIM, rows / TILE_DIM, 1);
  dim3 dimBlock(TILE_DIM, BLOCK_ROWS, 1);

  matrix_transpose_ker_v4<<<dimGrid, dimBlock, 0, stream>>>(output, input);
}

#define INSTANTIATE_MATRIX_TRANSPOSE_V4(T)                               \
  template void matrix_transpose_cuda_v4(const T* input, T* output,      \
                                         const int rows, const int cols, \
                                         cudaStream_t stream);

INSTANTIATE_MATRIX_TRANSPOSE_V4(float)
INSTANTIATE_MATRIX_TRANSPOSE_V4(half)
INSTANTIATE_MATRIX_TRANSPOSE_V4(__nv_bfloat16)

template <typename T>
void matrix_transpose_cuda_v5(const T* input, T* output, const int rows,
                              const int cols, cudaStream_t stream) {
  if (std::is_same<T, float>::value) {
    constexpr int N = 2;
    dim3 grid_dim((cols + 32 * N - 1) / (32 * N),
                  (rows + 32 * N - 1) / (32 * N));
    dim3 block_dim(32, 32);
    matrix_transpose_ker_v5<float, N><<<grid_dim, block_dim, 0, stream>>>(
        reinterpret_cast<const float*>(input), reinterpret_cast<float*>(output),
        rows, cols);
  } else if (std::is_same<T, half>::value) {
    constexpr int N = 4;
    dim3 grid_dim((cols + 32 * N - 1) / (32 * N),
                  (rows + 32 * N - 1) / (32 * N));
    dim3 block_dim(32, 32);
    matrix_transpose_ker_v5<half, N><<<grid_dim, block_dim, 0, stream>>>(
        reinterpret_cast<const half*>(input), reinterpret_cast<half*>(output),
        rows, cols);
  } else if (std::is_same<T, __nv_bfloat16>::value) {
    constexpr int N = 4;
    dim3 grid_dim((cols + 32 * N - 1) / (32 * N),
                  (rows + 32 * N - 1) / (32 * N));
    dim3 block_dim(32, 32);
    matrix_transpose_ker_v5<__nv_bfloat16, N>
        <<<grid_dim, block_dim, 0, stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(input),
            reinterpret_cast<__nv_bfloat16*>(output), rows, cols);
  }
}

#define INSTANTIATE_MATRIX_TRANSPOSE_V5(T)                               \
  template void matrix_transpose_cuda_v5(const T* input, T* output,      \
                                         const int rows, const int cols, \
                                         cudaStream_t stream);

INSTANTIATE_MATRIX_TRANSPOSE_V5(float)
INSTANTIATE_MATRIX_TRANSPOSE_V5(half)
INSTANTIATE_MATRIX_TRANSPOSE_V5(__nv_bfloat16)

template <typename T>
void matrix_transpose_cuda_v6(const T* input, T* output, const int rows,
                              const int cols, cudaStream_t stream) {
  if (std::is_same<T, float>::value) {
    constexpr int N = 2;
    dim3 grid_dim((cols + 32 * N - 1) / (32 * N),
                  (rows + 32 * N - 1) / (32 * N));
    dim3 block_dim(32, 32);
    int shared_mem_size = (32 * N) * (32 * N + 4) * sizeof(T);
    if (shared_mem_size >= (48 << 10)) {
      check_cuda_error(cudaFuncSetAttribute(
          matrix_transpose_ker_v6<float, N>,
          cudaFuncAttributeMaxDynamicSharedMemorySize, shared_mem_size));
    }
    matrix_transpose_ker_v6<float, N>
        <<<grid_dim, block_dim, shared_mem_size, stream>>>(
            reinterpret_cast<const float*>(input),
            reinterpret_cast<float*>(output), rows, cols);
  } else if (std::is_same<T, half>::value) {
    constexpr int N = 8;
    dim3 grid_dim((cols + 32 * N - 1) / (32 * N),
                  (rows + 32 * N - 1) / (32 * N));
    dim3 block_dim(32, 32);
    int shared_mem_size = (32 * N) * (32 * N + N) * sizeof(T);
    if (shared_mem_size >= (48 << 10)) {
      check_cuda_error(cudaFuncSetAttribute(
          matrix_transpose_ker_v6<half, N>,
          cudaFuncAttributeMaxDynamicSharedMemorySize, shared_mem_size));
    }
    matrix_transpose_ker_v6<half, N>
        <<<grid_dim, block_dim, shared_mem_size, stream>>>(
            reinterpret_cast<const half*>(input),
            reinterpret_cast<half*>(output), rows, cols);
    check_cuda_error(cudaPeekAtLastError());
  } else if (std::is_same<T, __nv_bfloat16>::value) {
    constexpr int N = 4;
    dim3 grid_dim((cols + 32 * N - 1) / (32 * N),
                  (rows + 32 * N - 1) / (32 * N));
    dim3 block_dim(32, 32);
    int shared_mem_size = (32 * N) * (32 * N + 4) * sizeof(T);
    if (shared_mem_size >= (48 << 10)) {
      check_cuda_error(cudaFuncSetAttribute(
          matrix_transpose_ker_v6<__nv_bfloat16, N>,
          cudaFuncAttributeMaxDynamicSharedMemorySize, shared_mem_size));
    }
    matrix_transpose_ker_v6<__nv_bfloat16, N>
        <<<grid_dim, block_dim, shared_mem_size, stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(input),
            reinterpret_cast<__nv_bfloat16*>(output), rows, cols);
  }
}

#define INSTANTIATE_MATRIX_TRANSPOSE_V6(T)                               \
  template void matrix_transpose_cuda_v6(const T* input, T* output,      \
                                         const int rows, const int cols, \
                                         cudaStream_t stream);

INSTANTIATE_MATRIX_TRANSPOSE_V6(float)
INSTANTIATE_MATRIX_TRANSPOSE_V6(half)
INSTANTIATE_MATRIX_TRANSPOSE_V6(__nv_bfloat16)


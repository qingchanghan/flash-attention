#pragma once

#include <cuda.h>
#include <cuda_runtime.h>

#include <iostream>

#define TYPE_SWITCH(COND1, COND2, ...) \
  [&] {                                \
    if (COND1) {                       \
      using elem_type = float;         \
      return __VA_ARGS__();            \
    } else if (COND2) {                \
      using elem_type = half;          \
      return __VA_ARGS__();            \
    } else {                           \
      using elem_type = __nv_bfloat16; \
      return __VA_ARGS__();            \
    }                                  \
  }()

template <typename T, int N>
struct VecType {};

template <>
struct VecType<float, 1> {
  using Type = float;
};

template <>
struct VecType<float, 2> {
  using Type = float2;
};

template <>
struct VecType<float, 4> {
  using Type = float4;
};

template <>
struct VecType<half, 1> {
  using Type = half;
};

template <>
struct VecType<half, 2> {
  using Type = uint32_t;
};

template <>
struct VecType<half, 4> {
  using Type = float2;
};

template <>
struct VecType<half, 8> {
  using Type = float4;
};

template <>
struct VecType<__nv_bfloat16, 1> {
  using Type = __nv_bfloat16;
};

template <>
struct VecType<__nv_bfloat16, 2> {
  using Type = uint32_t;
};

template <>
struct VecType<__nv_bfloat16, 4> {
  using Type = float2;
};

template <>
struct VecType<__nv_bfloat16, 8> {
  using Type = float4;
};

template <typename T>
void matrix_transpose_cuda_v0(const T* input, T* output, const int rows,
                              const int cols, cudaStream_t stream);

template <typename T>
void matrix_transpose_cuda_v1(const T* input, T* output, const int rows,
                              const int cols, cudaStream_t stream);

template <typename T>
void matrix_transpose_cuda_v2(const T* input, T* output, const int rows,
                              const int cols, cudaStream_t stream);

template <typename T>
void matrix_transpose_cuda_v3(const T* input, T* output, const int rows,
                              const int cols, cudaStream_t stream);

template <typename T>
void matrix_transpose_cuda_v4(const T* input, T* output, const int rows,
                              const int cols, cudaStream_t stream);

template <typename T>
void matrix_transpose_cuda_v5(const T* input, T* output, const int rows,
                              const int cols, cudaStream_t stream);

template <typename T>
void matrix_transpose_cuda_v6(const T* input, T* output, const int rows,
                              const int cols, cudaStream_t stream);

template <typename T>
void reduce_sum_cuda_v0(const T* input, T* output, const int rows,
                        const int cols, cudaStream_t stream);

template <typename T>
void scan_cuda_v0(const T* input, T* output, T* partial_sums, const int rows,
                  const int cols, cudaStream_t stream);

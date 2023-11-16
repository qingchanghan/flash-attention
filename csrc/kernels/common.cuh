#pragma once

#include <assert.h>
#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <stdint.h>

#include <cfloat>
#include <limits>

#include "cuda_utils.h"
#include "kernels.h"

#define WARP_SIZE 32
#define BLOCK_SIZE 256
#define WARP_NUM (BLOCK_SIZE / WARP_SIZE)

#define FETCH_VT(num, VT) (reinterpret_cast<VT*>(&(num))[0])
#define FETCH_CONST_VT(num, VT) (reinterpret_cast<const VT*>(&(num))[0])

// VecType
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
  using Type = half2;
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
  using Type = __nv_bfloat162;
};

template <>
struct VecType<__nv_bfloat16, 4> {
  using Type = uint2;  // 和 half4 区分开
};

template <>
struct VecType<__nv_bfloat16, 8> {
  using Type = uint4;  // 和 half8 区分开
};

// converter
// TODO: 部分转换用 ptx 实现
template <typename T1, typename T2>
__inline__ __device__ T2 converter(T1 src) {
  return (T2)src;
}

template <>
__inline__ __device__ float2 converter(half2 src) {
  float2 dst;
  dst.x = (float)src.x;
  dst.y = (float)src.y;
  return dst;
}
template <>
__inline__ __device__ float4 converter(float2 src) {
  float4 dst;
  half2& src1 = FETCH_VT(src.x, half2);
  half2& src2 = FETCH_VT(src.y, half2);
  dst.x = (float)src1.x;
  dst.y = (float)src1.y;
  dst.z = (float)src2.x;
  dst.w = (float)src2.y;
  return dst;
}
template <>
__inline__ __device__ float2 converter(__nv_bfloat162 src) {
  float2 dst;
  dst.x = (float)src.x;
  dst.y = (float)src.y;
  return dst;
}
template <>
__inline__ __device__ float4 converter(uint2 src) {
  float4 dst;
  __nv_bfloat162& src1 = FETCH_VT(src.x, __nv_bfloat162);
  __nv_bfloat162& src2 = FETCH_VT(src.y, __nv_bfloat162);
  dst.x = (float)src1.x;
  dst.y = (float)src1.y;
  dst.z = (float)src2.x;
  dst.w = (float)src2.y;
  return dst;
}
template <>
__inline__ __device__ half2 converter(float2 src) {
  half2 dst;
  dst.x = (half)src.x;
  dst.y = (half)src.y;
  return dst;
}
template <>
__inline__ __device__ float2 converter(float4 src) {
  float2 dst;
  half2 dst1, dst2;

  dst1.x = (half)src.x;
  dst1.y = (half)src.y;
  dst2.x = (half)src.z;
  dst2.y = (half)src.w;
  dst.x = FETCH_VT(dst1, float);
  dst.y = FETCH_VT(dst2, float);
  return dst;
}
template <>
__inline__ __device__ __nv_bfloat162 converter(float2 src) {
  __nv_bfloat162 dst;
  dst.x = (__nv_bfloat16)src.x;
  dst.y = (__nv_bfloat16)src.y;
  return dst;
}
template <>
__inline__ __device__ uint2 converter(float4 src) {
  uint2 dst;
  __nv_bfloat162 dst1, dst2;

  dst1.x = (__nv_bfloat16)src.x;
  dst1.y = (__nv_bfloat16)src.y;
  dst2.x = (__nv_bfloat16)src.z;
  dst2.y = (__nv_bfloat16)src.w;
  dst.x = FETCH_VT(dst1, float);
  dst.y = FETCH_VT(dst2, float);
  return dst;
}

// vec_max
template <typename T, typename VT>
__inline__ __device__ T vec_max(VT val) {
  return (T)val;
}

template <>
__inline__ __device__ float vec_max(float2 val) {
  return fmax(val.x, val.y);
}
template <>
__inline__ __device__ float vec_max(float4 val) {
  return fmax(fmax(val.x, val.y), fmax(val.z, val.w));
}

// device function
template <typename T>
__inline__ __device__ T warp_reduce_sum(T val) {
#pragma unroll
  for (int mask = 16; mask > 0; mask >>= 1) {
    val = val + __shfl_xor_sync(0xffffffff, val, mask, 32);
  }
  return val;
}

template <typename T>
__inline__ __device__ T block_reduce_sum(T val) {
  static __shared__ T shm[32];
  int lane_id = threadIdx.x & 0x1f;
  int warp_id = threadIdx.x >> 5;

  val = warp_reduce_sum<T>(val);
  if (lane_id == 0) {
    shm[warp_id] = val;
  }
  __syncthreads();

  if (warp_id == 0) {
    val = shm[lane_id];
    val = warp_reduce_sum<T>(val);
  }
  return val;
}

template <typename T>
__inline__ __device__ T warp_reduce_max(T val) {
#pragma unroll
  for (int mask = 16; mask > 0; mask >>= 1) {
    val = fmax(val, __shfl_xor_sync(0xffffffff, val, mask, 32));
  }
  return val;
}

template <typename T>
__inline__ __device__ T block_reduce_max(T val) {
  static __shared__ T shm[32];
  int lane_id = threadIdx.x & 0x1f;
  int warp_id = threadIdx.x >> 5;

  val = warp_reduce_max<T>(val);
  if (lane_id == 0) {
    shm[warp_id] = val;
  }
  __syncthreads();

  if (warp_id == 0) {
    val = shm[lane_id];
    val = warp_reduce_max<T>(val);
  }
  return val;
}


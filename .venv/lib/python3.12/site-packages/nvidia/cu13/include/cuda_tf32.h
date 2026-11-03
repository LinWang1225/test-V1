/*
 * Copyright 2025 NVIDIA Corporation.  All rights reserved.
 *
 * NOTICE TO LICENSEE:
 *
 * This source code and/or documentation ("Licensed Deliverables") are
 * subject to NVIDIA intellectual property rights under U.S. and
 * international Copyright laws.
 *
 * These Licensed Deliverables contained herein is PROPRIETARY and
 * CONFIDENTIAL to NVIDIA and is being provided under the terms and
 * conditions of a form of NVIDIA software license agreement by and
 * between NVIDIA and Licensee ("License Agreement") or electronically
 * accepted by Licensee.  Notwithstanding any terms or conditions to
 * the contrary in the License Agreement, reproduction or disclosure
 * of the Licensed Deliverables to any third party without the express
 * written consent of NVIDIA is prohibited.
 *
 * NOTWITHSTANDING ANY TERMS OR CONDITIONS TO THE CONTRARY IN THE
 * LICENSE AGREEMENT, NVIDIA MAKES NO REPRESENTATION ABOUT THE
 * SUITABILITY OF THESE LICENSED DELIVERABLES FOR ANY PURPOSE.  IT IS
 * PROVIDED "AS IS" WITHOUT EXPRESS OR IMPLIED WARRANTY OF ANY KIND.
 * NVIDIA DISCLAIMS ALL WARRANTIES WITH REGARD TO THESE LICENSED
 * DELIVERABLES, INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY,
 * NONINFRINGEMENT, AND FITNESS FOR A PARTICULAR PURPOSE.
 * NOTWITHSTANDING ANY TERMS OR CONDITIONS TO THE CONTRARY IN THE
 * LICENSE AGREEMENT, IN NO EVENT SHALL NVIDIA BE LIABLE FOR ANY
 * SPECIAL, INDIRECT, INCIDENTAL, OR CONSEQUENTIAL DAMAGES, OR ANY
 * DAMAGES WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS,
 * WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS
 * ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE OR PERFORMANCE
 * OF THESE LICENSED DELIVERABLES.
 *
 * U.S. Government End Users.  These Licensed Deliverables are a
 * "commercial item" as that term is defined at 48 C.F.R. 2.101 (OCT
 * 1995), consisting of "commercial computer software" and "commercial
 * computer software documentation" as such terms are used in 48
 * C.F.R. 12.212 (SEPT 1995) and is provided to the U.S. Government
 * only as a commercial end item.  Consistent with 48 C.F.R.12.212 and
 * 48 C.F.R. 227.7202-1 through 227.7202-4 (JUNE 1995), all
 * U.S. Government End Users acquire the Licensed Deliverables with
 * only those rights set forth herein.
 *
 * Any use of the Licensed Deliverables in individual and commercial
 * software must include, in the user documentation and internal
 * comments to the code, the above Disclaimer and U.S. Government End
 * Users Notice.
 */

#ifndef __CUDA_TF32_H__
#define __CUDA_TF32_H__

#ifndef __NV_TL__
#define __NV_TL__
#endif

#ifndef __NV_TL_BUILTIN__
#define __NV_TL_BUILTIN__
#endif

#if !defined(_MSC_VER) && __cplusplus >= 201103L
#  define __CPP_VERSION_AT_LEAST_11_TF32
#elif _MSC_FULL_VER >= 190024210 && _MSVC_LANG >= 201103L
#  define __CPP_VERSION_AT_LEAST_11_TF32
#endif

#if defined(__CUDACC__)
#  define __CUDA_ALIGN__(align) __align__(align)
#else
#  if defined(__CPP_VERSION_AT_LEAST_11_TF32)
#    define __CUDA_ALIGN__(n) alignas(n)
#  else /* defined(__CPP_VERSION_AT_LEAST_11_TF32)*/
#    if defined(__GNUC__)
#      define __CUDA_ALIGN__(n) __attribute__ ((aligned(n)))
#    elif defined(_MSC_VER)
#      define __CUDA_ALIGN__(n) __declspec(align(n))
#    else
#      define __CUDA_ALIGN__(n)
#    endif /* defined(__GNUC__) */
#  endif /* defined(__CPP_VERSION_AT_LEAST_11_TF32) */
#endif /* defined(__CUDACC__) */

#if defined(__cplusplus)

#define __CUDA_TF32_TYPES_EXIST__
#if (defined(__CUDACC_RTC__) && ((__CUDACC_VER_MAJOR__ > 12) || ((__CUDACC_VER_MAJOR__ == 12) && (__CUDACC_VER_MINOR__ >= 3))))
#  define __CUDA_HOSTDEVICETILE__ __device__ __NV_TL__
#elif defined(__CUDACC__) || defined(_NVHPC_CUDA)
#  define __CUDA_HOSTDEVICETILE__ __host__ __device__ __NV_TL__
#else /* defined(__CUDACC__) || defined(_NVHPC_CUDA) */
#  define __CUDA_HOSTDEVICETILE__
#endif /* (defined(__CUDACC_RTC__) && ((__CUDACC_VER_MAJOR__ > 12) || ((__CUDACC_VER_MAJOR__ == 12) && (__CUDACC_VER_MINOR__ >= 3))))) */

/**
 * \brief __nv_tf32 datatype
 * \details This structure implements the datatype for storing Tensor
 * Float 32 numbers. This format uses at least 19 bits to represent
 * its value: 1 sign bit, 8 bits for the exponent, and 10 or more bits
 * for the mantissa. The number of precision bits is unspecified and
 * depends on the hardware target. The storage size of this type is 32
 * bits in total and it requires a 4-byte alignment. The arrangement
 * of the value bits within the 32 bits of storage is unspecified.
 *
 * This type is implemented in the \p cuda_tf32.h header.
 * Currently, the \p __nv_tf32 type may only be used in tile code and
 * there are no APIs that directly operate on this type.
 * Instead, \p __nv_tf32 should be used as the element of a tile
 * (e.g. \p cuda::tiles::tile<__nv_tf32, cuda::tiles::shape<4>>) and passed to
 * CUDA Tile C++ APIs. The supported CUDA Tile C++ APIs include
 * conversions to other numeric types, memory operations, and matrix multiplication.
 * For details on these operations, please see the CUDA Tile C++ API Reference.
 */
struct __NV_TL_BUILTIN__ __CUDA_ALIGN__(4) __nv_tf32 {

private:
  unsigned int __x;
};

#undef __CUDA_HOSTDEVICETILE__
#endif /* defined(__cplusplus) */

#undef __CUDA_ALIGN__
#undef __CPP_VERSION_AT_LEAST_11_TF32
#endif /* __CUDA_TF32_H__ */

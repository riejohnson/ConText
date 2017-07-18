/* * * * *
 *  AzCuda.cuh
 *  Copyright (C) 2013-2016 Rie Johnson
 *
 *  This program is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation, either version 3 of the License, or
 *  (at your option) any later version.
 *
 *  This program is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with this program.  If not, see <http://www.gnu.org/licenses/>.
 * * * * */

#ifndef _AZ_CUDA_CUH_
#define _AZ_CUDA_CUH_

/* host functions using cuda */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <iostream>
#include <sstream>
#include <ctime>

#include "AzP.h"
#include "AzUtil.hpp"

extern bool __doDebug; 

using namespace std; 

class AzCuda {
public:
  /*---------------------------------------------------------------*/
  static int getAttr(int no, cudaDeviceAttr attr, const char *attr_nm, const char *eyec0, bool do_show=false) {
    const char *eyec1 = "AzCuda::getAttr"; 
    int val; 
    cudaError_t ret = cudaDeviceGetAttribute(&val, attr, no); 
    throwIfError(ret, eyec0, eyec1, attr_nm);
    if (do_show) {
      AzBytArr s(attr_nm); s << "=" << val; s.nl(); 
      s.print(log_out); 
    }
    return val; 
  }

  /*---------------------------------------------------------------*/
  static void showAttributes(int no) {
    const char *eyec = "AzCuda::showInfo"; 
    bool do_show = true; 
    getAttr(no, cudaDevAttrMaxThreadsPerBlock, "MaxThreadsPerBlock", eyec, do_show); 
    getAttr(no, cudaDevAttrMaxBlockDimX, "MaxBlockDimX", eyec, do_show);     
    getAttr(no, cudaDevAttrMaxGridDimX, "MaxGridDimX", eyec, do_show); 
    getAttr(no, cudaDevAttrMaxSharedMemoryPerBlock, "MaxSharedMemoryPerBlock", eyec, do_show);     
  }
 
  /*---------------------------------------------------------------*/  
  inline static
  void memcpy(void *dst, const void *src, size_t count, cudaMemcpyKind kind, const char *msg) { 
    const char *eyec = "AzCuda::memcpy"; 
    cudaError_t ret = cudaMemcpy(dst, src, count, kind); 
    if (ret != cudaSuccess) {
      printf("AzCuda::memcpy, found an error.  count=%f\n", (double)count); 
    }
    AzCuda::throwIfError(ret, eyec, msg);  
  }

  /*---------------------------------------------------------------*/
  inline static
  void throwIfError(cudaError_t ret, 
                                  const char *eyec, 
                                  const char *str1="", 
                                  const char *str2="") {
    if (ret == cudaSuccess) return;     
    AzBytArr s("AzCuda::throwIfError: "); s << str1 << " " << str2; 
    s << " cudaGetErrorString returned " << cudaGetErrorString(ret); 
    AzX::pthrow(AzCudaErr, eyec, s.c_str());   
  }

  /*---------------------------------------------------------------*/
  inline static
  void throwIfError(cudaError_t ret, 
                    const char *eyec, const char *str1, const char *str2, 
                    double num) {
    if (ret == cudaSuccess) return;     
    AzBytArr s("AzCuda::throwIfError: "); s << str1 << " " << str2 << " " << num; 
    s << " cudaGetErrorString returned " << cudaGetErrorString(ret);     
    AzX::pthrow(AzCudaErr, eyec, s.c_str()); 
  }  
  
  /*---------------------------------------------------------------*/
  inline static
  void throwIfError(cudaError_t ret, 
                    const char *eyec, 
                    int bb, int tt) {
    if (ret == cudaSuccess) return;     
    AzBytArr s("AzCuda::throwIfError: (#blocks="); s << bb << ", #threads=" << tt << ")"; 
    s << " cudaGetErrorString returned " << cudaGetErrorString(ret);         
    AzX::pthrow(AzCudaErr, eyec, s.c_str()); 
  }  
  
  /*---------------------------------------------------------------*/
  static
  void throwIfblasError(cublasStatus_t ret, 
                                  const char *eyec, 
                                  const char *str1="") {
    if (ret == CUBLAS_STATUS_SUCCESS) return;     
    const char *errmsg = ""; 
    if (ret == CUBLAS_STATUS_NOT_INITIALIZED) errmsg = "CUBLAS_STATUS_NOT_INITIALIZED"; 
    else if (ret == CUBLAS_STATUS_ALLOC_FAILED) errmsg = "CUBLAS_STATUS_ALLOC_FAILED"; 
    else if (ret == CUBLAS_STATUS_INVALID_VALUE) errmsg = "CUBLAS_STATUS_INVALID_VALUE"; 
    else if (ret == CUBLAS_STATUS_ARCH_MISMATCH) errmsg = "CUBLAS_STATUS_ARCH_MISMATCH"; 
    else if (ret == CUBLAS_STATUS_MAPPING_ERROR) errmsg = "CUBLAS_STATUS_MAPPING_ERROR"; 
    else if (ret == CUBLAS_STATUS_EXECUTION_FAILED) errmsg = "CUBLAS_STATUS_EXECUTION_FAILED"; 
    else if (ret == CUBLAS_STATUS_INTERNAL_ERROR) errmsg = "CUBLAS_STATUS_INTERNAL_ERROR"; 
    else errmsg = "Unknown error"; 
    
    AzBytArr s("throwIfblasError: "); s << str1 << " " << errmsg;  
    AzX::pthrow(AzCudaErr, eyec, s.c_str()); 
  }
  
  /*---------------------------------------------------------------*/
  static void throwIfsparseError(cusparseStatus_t ret, const char *eyec, const char *msg) {
    if (ret == CUSPARSE_STATUS_SUCCESS) return; 
    AzBytArr s; 
    if      (ret == CUSPARSE_STATUS_NOT_INITIALIZED) s << "NOT INITIALIZED"; 
    else if (ret == CUSPARSE_STATUS_ALLOC_FAILED) s << "ALLOC_FAILED: the resources could not be allocated."; 
    else if (ret == CUSPARSE_STATUS_ARCH_MISMATCH) s << "ARCH_MISMATCH: the device compute capability(CC) is less than 1.1. The CC of at least 1.1 is required. or the device does not support double precision."; 
    else if (ret == CUSPARSE_STATUS_INVALID_VALUE) s << "INVALID_VALUE: invalid parameters were passed"; 
    else if (ret == CUSPARSE_STATUS_EXECUTION_FAILED) s << "EXECUTION_FAILED: the function failed to launch on the GPU."; 
    else if (ret == CUSPARSE_STATUS_INTERNAL_ERROR) s << "INTERNAL_ERROR: an internal operation failed."; 
    else if (ret == CUSPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED) s << "MATRIX_TYPE_NOT_SUPPORTED: the matrix type is not suppoerted."; 
    else s << "... unknown cause ..."; 

    AzBytArr s1("throwIfsparseError: "); s1 << msg << " " << s.c_str(); 
    AzX::pthrow(AzCudaErr, eyec, s1.c_str());     
  }  

  /*---------------------------------------------------------------*/
  static void throwIfcurandError(curandStatus_t ret, const char *eyec, 
                          const char *str1="", const char *str2="") {
    if (ret == CURAND_STATUS_SUCCESS) return;     
    AzBytArr s(eyec); s << " AzCuda::throwIfcurandError " << str1 << " " << str2 << " curand returned " << (int)ret; 
    AzX::pthrow(AzCudaErr, eyec, s.c_str()); 
  }
  
  /*---------------------------------------------------------------*/
  static
  void sync(const char *eyec) {
    cudaError_t ret = cudaDeviceSynchronize(); 
    throwIfError(ret, eyec, "sync failed"); 
  }

  /*---------------------------------------------------------------*/
  static
  clock_t sync_clock(const char *eyec) {
    sync(eyec); 
    return clock(); 
  }  
  
  /*---------------------------------------------------------------*/
  static
  void check_error(const char *eyec) {
    cudaError_t ret = cudaGetLastError(); 
    throwIfError(ret, eyec); 
  }

  /*---------------------------------------------------------------*/
  inline static
  void check_error(const char *eyec, int bb, int tt) {
    cudaError_t ret = cudaGetLastError(); 
    throwIfError(ret, eyec, bb, tt); 
  }  
  
  /*---------------------------------------------------------------*/
  inline static
  void check_error_if_debug(const char *eyec, int bb, int tt) {
    if (!__doDebug) return; 
    cudaError_t ret = cudaGetLastError(); 
    throwIfError(ret, eyec, bb, tt); 
  }

  /*---------------------------------------------------------------*/
  inline static
  void sync_check_error_if_debug(const char *eyec, int bb, int tt) {
    if (!__doDebug) return; 
    sync(eyec); 
    cudaError_t ret = cudaGetLastError(); 
    throwIfError(ret, eyec, bb, tt); 
  }
  
  /*---------------------------------------------------------------*/  
  static
  void sync_check_error(bool do_sync, const char *eyec) {
    if (do_sync) {
      sync(eyec); 
    }
    cudaError_t ret = cudaGetLastError(); 
    throwIfError(ret, eyec); 
  }
}; 
#endif 
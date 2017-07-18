/* * * * *
 *  AzPmat_gpu.cuh
 *  Copyright (C) 2013-2015,2017 Rie Johnson
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

#ifndef _AZ_PMAT_GPU_HPP_
#define _AZ_PMAT_GPU_HPP_

#include "AzP.h"
#include "AzCuda.cuh"
#include "AzCuda_Pmat.cuh"
#include "AzUtil.hpp"
#include "AzParam.hpp"
#include "AzPrint.hpp"
#include "AzHelp.hpp"
#include "AzPmem.cuh"

extern int max_threads, max_blocks; 
extern bool __doDebug; 

/* pointers are all device pointers unless specified otherwise */

/***************************************************************/
template <class T>
class _AzParr {  
protected:
  T *elm; 
  int num; 
  int no;  /* for AzPmem */
  
public:  
  _AzParr() : elm(NULL), num(0), no(-1) {}
  ~_AzParr() {
    free(); 
  }  
  
  _AzParr(const _AzParr<T> &inp) {
    reset(&inp); 
  }
  _AzParr<T> & operator =(const _AzParr<T> &inp) {
    if (this == &inp) return *this; 
    reset(&inp); 
    return *this; 
  }  
  
  void free(); 
  void free_alloc(int inp_num, const char *str1="", const char *str2=""); 

  int size() const {
    return num; 
  }
  void reset(const _AzParr<T> *inp) {
    const char *eyec = "_AzParr::reset(_AzParr)"; 
    if (num != inp->num) {
      free_alloc(inp->num, eyec); 
    }
    AzCuda::memcpy(elm, inp->elm, sizeof(T)*num, cudaMemcpyDeviceToDevice, "_AzParr::reset device to device"); 
  }
  void copy_from_host(const T *h_ptr, int len) { /* host pointer */
    AzX::throw_if((len != size()), "_AzParr::copy_from_host", "length conflict");
    if (len == 0) return; 
    AzCuda::memcpy(elm, h_ptr, sizeof(T)*num, cudaMemcpyHostToDevice, "_AzParr::copy_from_host host to device"); 
  } 
  void reset_from_host(const T *h_ptr, int num, const char *str1="", const char *str2="") {
    free_alloc(num, str1, str2); 
    copy_from_host(h_ptr, num); 
  }
  void reset_with_zero(int num, const char *str1="", const char *str2="") {
    free_alloc(num, str1, str2); 
    azccall_setval(elm, (T)0, num); 
  }
 
  /*---  copy the entire array to host  ---*/
  void copy_to_host(T *h_ptr, int len) const {
    AzX::throw_if((len != size()), "_AzParr::copy_to_host", "length conflict"); 
    if (len == 0) return; 
    AzCuda::memcpy(h_ptr, elm, sizeof(T)*num, cudaMemcpyDeviceToHost, "_AzParr::copy_to_host device to host"); 
  }
  void copy_to_host(AzBaseArr<T> *harr) const {
    AzX::throw_if((harr == NULL), "_AzParr::copy_to_host(AzBaseArr)", "null input"); 
    harr->free(); 
    harr->alloc(size()); 
    copy_to_host(harr->point_u(), harr->size()); 
  }
  
  /*---  copy part of the array to device  ---*/
  void copy_to(_AzParr &dst, int dst_pos, int src_pos, int len) const { /* dst[dst_pos::len] <- this[src_pos::len] */
    AzX::throw_if((src_pos < 0 || len < 0 || src_pos+len > size()), "_AzParr::copy_to", "source position or length is wrong"); 
    AzX::throw_if((dst_pos < 0 || dst_pos+len > dst.size()), "_AzParr::copy_to", "destination position is wrong"); 
    if (len == 0) return; 
    AzCuda::memcpy(dst.elm+dst_pos, elm+src_pos, sizeof(T)*len, cudaMemcpyDeviceToDevice, "_AzParr::copy_to device to device"); 
  }
  
  /*---  copy one component to a host variable ---*/
  T get(int pos) const {
    AzX::throw_if((pos < 0 || pos >= size()), "_AzParr::get", "invalid position"); 
    T hostvar; 
    AzCuda::memcpy(&hostvar, elm+pos, sizeof(T), cudaMemcpyDeviceToHost, "_AzParr::get device to host"); 
    return hostvar; 
  }

  /*---  write without swapping byte order  ---*/
  void write(AzFile *file) const {
    file->writeInt(num); 
    if (num > 0) {
      T *h_ptr = NULL; 
      AzBaseArray<T> _a_host; 
      _a_host.alloc(&h_ptr, num, "_AzParr::write"); 
      AzCuda::memcpy(h_ptr, elm, sizeof(T)*num, cudaMemcpyDeviceToHost, "_AzParr::write device to host"); 
      file->writeItems(h_ptr, num);     
    }  
  }
  /*---  read  ---*/
  void read(AzFile *file) {
    int sz = file->readInt(); 
    free_alloc(sz, "_AzParr::read"); 
    if (num > 0) {
      T *h_ptr = NULL; 
      AzBaseArray<T> _a_host; 
      _a_host.alloc(&h_ptr, num, "_AzParr::read"); 
      
      file->readItems(h_ptr, num);       
      AzCuda::memcpy(elm, h_ptr, sizeof(T)*num, cudaMemcpyHostToDevice, "_AzParr::read host to device"); 
    }
  }
  
  const T *_dptr() const {
    return elm; 
  }
  T *_dptr_u() {
    return elm; 
  }
};

/***************************************************************/  
class AzPdevice {
public:
  int maxThreadsPerBlock, maxBlockDimX, maxGridDimX, maxSharedMemoryPerBlock; 
  cublasHandle_t h; 
  cusparseHandle_t cusparse_handle; 
  cusparseMatDescr_t cusparse_desc;   
  bool is_h_set; 
  AzPmem pmem; 
  
  AzPdevice() : is_h_set(false), maxThreadsPerBlock(-1), maxBlockDimX(-1), maxGridDimX(-1), maxSharedMemoryPerBlock(-1) {}
  ~AzPdevice() { closeDevice(); }
  
  void closeDevice() {
    pmem.term(); 
    free_handles("closeDevice()"); 
  }
  static int getDevice() {
    int dev; 
    cudaError_t ret = cudaGetDevice(&dev); 
    AzCuda::throwIfError(ret, "AzPdevice_gpu::getDevice"); 
    return dev; 
  }
  int setDevice(const char *str) {
    int no = atol(str); 
    double gb = 0; 
    const char *ptr = strchr(str, ':'); 
    if (ptr != NULL) gb = atof(ptr+1); 
    return setDevice(no, gb); 
  } 
  int setDevice(int no, double gb=0) {
    const char *eyec = "AzPdevice::setDevice"; 
    closeDevice(); 
    
    cudaError_t ret; 
    ret = cudaDeviceReset(); 
    AzCuda::throwIfError(ret, eyec, "cudaDeviceReset");         
    if (no >= 0) {
      ret = cudaSetDevice(no); 
      AzCuda::throwIfError(ret, eyec, "cudaSetDevice"); 
    }
    
    int dev_no = getDevice(); 
    
    bool do_show = true; 
    maxThreadsPerBlock = AzCuda::getAttr(dev_no, cudaDevAttrMaxThreadsPerBlock, "MaxThreadsPerBlock", eyec, do_show); 
    maxBlockDimX = AzCuda::getAttr(dev_no, cudaDevAttrMaxBlockDimX, "MaxBlockDimX", eyec, do_show); 
    maxGridDimX = AzCuda::getAttr(dev_no, cudaDevAttrMaxGridDimX, "MaxGridDimX", eyec, do_show); 
    maxSharedMemoryPerBlock = AzCuda::getAttr(dev_no, cudaDevAttrMaxSharedMemoryPerBlock, "MaxSharedMemoryPerBlock", eyec, do_show); 
    max_threads = maxBlockDimX;  /* default */
    max_blocks = maxGridDimX;    /* default */
    int unit = sizeof(int)+sizeof(AzFloat); 
    if (azc_numShared*unit > maxSharedMemoryPerBlock) {
      AzBytArr s("Maximum shared memory per block is unexpectedly small ("); 
      s << maxSharedMemoryPerBlock << ").  " << "Please reduce azc_numShared in AzP.h to "; 
      s << maxSharedMemoryPerBlock/unit << " or smaller and recompile."; 
      AzX::pthrow_if(true, eyec, s.c_str()); 
    }
    
    /*---  create handles  ---*/
    cublasStatus_t blasret = cublasCreate(&h); 
    AzCuda::throwIfblasError(blasret, eyec, "cublasCreate failed"); 
    cusparseStatus_t 
    sret = cusparseCreate(&cusparse_handle); 
    AzCuda::throwIfsparseError(sret, eyec, "cusparseCreate"); 
    sret = cusparseCreateMatDescr(&cusparse_desc);     
    AzCuda::throwIfsparseError(sret, eyec, "cusparseCreateMatDescr"); 
  
    is_h_set = true; 

    pmem.init(gb);     
    
    return dev_no; 
  }
  static int getDeviceCount() {
    int num; 
    cudaError_t ret = cudaGetDeviceCount(&num); 
    AzCuda::throwIfError(ret, "AzPdevice_gpu::getDeviceCount"); 
    return num; 
  }
  static double sync_clock() {
    return (double)AzCuda::sync_clock("AzPdevice_gpu::sync_clock");   
  }

  /*------------------------------------------------------------*/ 
  #define kw_max_threads "gpu_max_threads="
  #define kw_max_blocks "gpu_max_blocks="
  /*------------------------------------------------------------*/   
  void resetParam(AzParam &azp) {
    const char *eyec = "AzPdevice::resetParam"; 
    azp.vInt(kw_max_threads, &max_threads); 
    azp.vInt(kw_max_blocks, &max_blocks); 
#if 0     
    AzX::pthrow_if((max_threads<=0), AzInputError, eyec, kw_max_threads, "must be positive."); 
    AzX::pthrow_if((max_blocks<=0), AzInputError, eyec, kw_max_blocks, "must be positive."); 
    AzX::pthrow_if((max_threads>maxBlockDimX), eyec, kw_max_threads, "too large"); 
    AzX::pthrow_if((max_blocks>maxGridDimX), eyec, kw_max_blocks, "too large"); 
#else
    AzXi::throw_if_nonpositive(max_threads, eyec, kw_max_threads); 
    AzXi::throw_if_nonpositive(max_blocks, eyec, kw_max_threads);             
    AzXi::invalid_input(max_threads>maxBlockDimX, eyec, kw_max_threads); 
    AzXi::invalid_input(max_blocks>maxGridDimX, eyec, kw_max_blocks); 
#endif     
  }
  void printParam(AzPrint &o) const {
    o.printV(kw_max_threads, max_threads); 
    o.printV(kw_max_blocks, max_blocks);   
  } 
  void printHelp(AzHelp &h) const {
    h.item(kw_max_threads, "Maximum number of threads per block to be generated on GPU.", "Maximum of the GPU"); 
    h.item(kw_max_blocks, "Maximum number of blocks to be generated on on GPU.", "Maximu of the GPU");   
  }   
  
protected: 
  void free_handles(const char *eyec) {  
    if (is_h_set) {
      cublasStatus_t ret = cublasDestroy(h); 
      AzCuda::throwIfblasError(ret, eyec, "cublasDestroy failed");         
      cusparseStatus_t 
      ret2 = cusparseDestroy(cusparse_handle); 
      AzCuda::throwIfsparseError(ret2, eyec, "cusparseDestroy"); 
      ret2 = cusparseDestroyMatDescr(cusparse_desc);      
      AzCuda::throwIfsparseError(ret2, eyec, "cusparseDestroyMatDescr");       
   
      is_h_set = false ;
    }
  }
};

/***************************************************************/
class AzPstreams {
protected:
  AzBaseArray<cudaStream_t> a; 
  cudaStream_t *streams; 
  int id; 
  
public:
  AzPstreams() : streams(NULL), id(-1) {}
  ~AzPstreams() {
    release(); 
  }
  void reset(int num) {
    a.free(&streams); 
    a.alloc(&streams, num, "AzPstreams::reset", "streams"); 
    int ix; 
    for (ix = 0; ix < num; ++ix) {
      cudaError_t ret = cudaStreamCreate(&streams[ix]);         
      AzCuda::throwIfError(ret, "cudaStreamCreate in AzPstreams::reset"); 
    }
    id = -1; 
  }  
  void sync() const {
    int num = a.size(); 
    int ix; 
    for (ix = 0; ix < num; ++ix) {
      cudaError_t ret = cudaStreamSynchronize(streams[ix]); 
      AzCuda::throwIfError(ret, "cudaStreamSynchronize in AzPstreams::sync");       
    }
  }
  void setStreamId(int index) {
    int num = a.size(); 
    if (index < 0 || num <= 0) id = -1; 
    else {
      id = index%num; 
    }
  }
  void setStream(cublasHandle_t h) const { 
    cublasStatus_t ret; 
    if (id < 0) {
      ret = cublasSetStream(h, NULL); 
    }
    else {
      ret = cublasSetStream(h, streams[id]); 
    }
    AzCuda::throwIfblasError(ret, "cublasSetStream in AzPstreams::setStream");    
  }
  void release() {
    int num = a.size(); 
    int ix; 
    for (ix = 0; ix < num; ++ix) {
      cudaStreamDestroy(streams[ix]); 
    }
    id = -1; 
  }
}; 

/***************************************************************/  
class _AzPmat {
protected: 
  bool do_print; 

public: 
  inline void _resetDoPrint(bool inp) { 
    do_print = inp; 
  }
  inline static void _add_cols_d2s(AzFloat *dst, const AzFloat *src, int row_num, 
                         const int *cols, int cnum, AzFloat coeff) {         
    azccall_add_cols_d2s(dst, src, row_num, cols, cnum, coeff); 
  }
  inline static void _add_cols_s2d(AzFloat *dst, const AzFloat *src, int row_num, 
                         const int *cols, int cnum, AzFloat coeff, bool do_z) {         
    if (do_z) azccall_add_cols_s2dz(dst, src, row_num, cols, cnum, coeff); 
    else      azccall_add_cols_s2d(dst, src, row_num, cols, cnum, coeff); 
  }
  inline static void _add_rows_s2d(AzFloat *dst, int dst_r_num, const AzFloat *src, int src_r_num, 
                         int c_num, const int *rows_s2d, AzFloat coeff) {         
    azccall_add_rows_s2d(dst, dst_r_num, src, src_r_num, c_num, rows_s2d, coeff); 
  }
  inline static void _copy_cols(AzFloat *dst, const AzFloat *src, int row_num, 
                         const int *cols, int cnum, bool do_zero_negaindex, AzFloat coeff) {         
    azccall_copy_cols(dst, src, row_num, cols, cnum, do_zero_negaindex, coeff); 
  }
  static void _copy(AzFloat *dst, const AzFloat *src, int num, AzFloat coeff=1); 
  static void _copy_cols2cols(AzFloat *dst, const AzFloat *src, int row_num, const int *cols, int cnum) {
    azccall_copy_cols2cols(dst, src, row_num, cols, cnum); 
  }
  static void _copy_scol2dcol(AzFloat *dst, const AzFloat *src, int row_num, 
                              const int *src_cols, const int *dst_cols, int cnum) {
    azccall_copy_scol2dcol(dst, src, row_num, src_cols, dst_cols, cnum); 
  }
  
  /* dst[dst_r0::r_num] <- src[src_r0::r_num] */
  static void _copy_rowwise(AzFloat *dst, int dst_r_num, int col_num, int dst_r0, 
                            const AzFloat *src, int src_r_num, int src_r0, 
                            int r_num) {
    azccall_copy_rowwise(dst, dst_r_num, col_num, dst_r0, src, src_r_num, src_r0, r_num); 
  }                            
   
  template <class MyFloat>
  static void _copy01(MyFloat *dst, const MyFloat *src_host, int num) {
    if (num <= 0) return; 
    AzCuda::memcpy(dst, src_host, sizeof(dst[0])*num, cudaMemcpyHostToDevice, "_AzPmat::_copy01 host to device"); 
  }
  template <class MyFloat>
  static void _copy10(MyFloat *dst_host, const MyFloat *src, int num) {
    if (num <= 0) return; 
    AzCuda::memcpy(dst_host, src, sizeof(src[0])*num, cudaMemcpyDeviceToHost, "_AzPmat::_copy10 device to host"); 
  }
    
  static void _setval(AzFloat *dst, AzFloat val, int num) {
    azccall_setval(dst, val, num); 
  }
  static void _add(AzFloat *dst, const AzFloat *src, int num, AzFloat coeff=1) {
    azccall_add(dst, src, num, coeff); 
  }
  static void _add_axpy(AzFloat *dst, const AzFloat *src, int num, AzFloat coeff=1);   
  static void _add_geam(AzFloat *dst, const AzFloat *src, int num, AzFloat coeff=1); /* very slow */
  static void _add1(AzFloat *dst, AzFloat dst_coeff, const AzFloat *src, AzFloat src_coeff, int num) {
    azccall_add1(dst, dst_coeff, src, src_coeff, num); 
  }
  static void _add2(AzFloat *dst, AzFloat dst_coeff, const AzFloat *src1, AzFloat src_coeff1, const AzFloat *src2, AzFloat src_coeff2, int num) {
    azccall_add2(dst, dst_coeff, src1, src_coeff1, src2, src_coeff2, num); 
  }
  static void _add_sq1(AzFloat *dst, AzFloat dst_coeff, const AzFloat *src, AzFloat src_coeff, int num) {
    azccall_add_sq1(dst, dst_coeff, src, src_coeff, num); 
  }  
  static void _addval(AzFloat *dst, AzFloat val, int num) {
    azccall_addval(dst, val, num); 
  }
  static void _add_eachrow(AzFloat *dst, int r_num, int c_num, const AzFloat *src, AzFloat coeff) {
    azccall_add_eachrow(dst, r_num, c_num, src, coeff); 
  }
  
  static void _setRow(AzFloat *dst, int row_num, int col_num, int row, AzFloat val) {
    azccall_setRow(dst, row_num, col_num, row, val); 
  }
  
  /*---  sum, absSum, squareSum  ---*/
  static AzFloat _sum_cublas(const AzFloat *src, int num);     
  static AzFloat _sum_noblas(const AzFloat *src, int num) {
    return _get_sum(azc_Op_Sum, src, num); 
  }
  static AzFloat _absSum_cublas(const AzFloat *src, int num);   
  static AzFloat _absSum_noblas(const AzFloat *src, int num) {
    return _get_sum(azc_Op_AbsSum, src, num); 
  }
  static AzFloat _norm2_cublas(const AzFloat *src, int num); 
  static AzFloat _squareSum_noblas(const AzFloat *src, int num) {
    return _get_sum(azc_Op_SquareSum, src, num); 
  }
  
  /*---  count nonzero  ---*/
  static int _nz(const AzFloat *src, int num); 

  /*---  min max  ---*/
  static double _min(const AzFloat *src, int num, int *out_index); 
  static double _max(const AzFloat *src, int num, int *out_index); 

  inline static void _max_eachCol(const AzFloat *src, int r_num, int c_num, 
                           int *out_ind,      /* array of size c_num */
                           AzFloat *out_val) {  /* array of size c_num */
    azccall_max_eachCol(src, r_num, c_num, out_ind, out_val);                            
  }
  inline static void _min_eachCol(const AzFloat *src, int r_num, int c_num, 
                           int *out_ind,      /* array of size c_num */
                           AzFloat *out_val) {  /* array of size c_num */
    azccall_min_eachCol(src, r_num, c_num, out_ind, out_val);                            
  }
    
  /*---  column-wise sum, absSum, squareSum  ---*/
  static void _add_colAbsSum(const AzFloat *src, int row_num, int col_num, AzFloat *sum) {
    return _add_colSum(azc_Op_AbsSum, src, row_num, col_num, sum); 
  }
  static void _add_colSquareSum(const AzFloat *src, int row_num, int col_num, AzFloat *sum) {
    return _add_colSum(azc_Op_SquareSum, src, row_num, col_num, sum); 
  }  

  /*---  element-wise multiplication  ---*/
  static void _elm_multi(AzFloat *dst, const AzFloat *src, int num, bool do_inv) {
    azccall_elm_multi(dst, src, num, do_inv); 
  }
  
  /*---  ---*/
  static void _divide(AzFloat *dst, AzFloat val, int num) { azccall_divide(dst, val, num); }
  static void _multiply(AzFloat *dst, AzFloat val, int num) { azccall_multiply(dst, val, num); }
  static void _multiply_scal(AzFloat *dst, AzFloat val, int num); 
  static void _multiply_eachcol(AzFloat *dst, int r_num, int c_num, const AzFloat *src, bool do_inv) {
    azccall_multiply_eachcol(dst, r_num, c_num, src, do_inv); 
  }  
  static void _multiply_eachrow(AzFloat *dst, int r_num, int c_num, const AzFloat *src, bool do_inv) {
    azccall_multiply_eachrow(dst, r_num, c_num, src, do_inv); 
  } 
  static void _trun(AzFloat *dst, int num, AzFloat minval, AzFloat maxval) {
    azccall_trun(dst, num, minval, maxval); 
  }

  /*---  scale by RMS (for AdaDelta)  ---*/
  inline static void _scale_by_sqrt(AzFloat *dst, int num, const AzFloat *src, AzFloat epsilon, bool do_inv=false) {  
    azccall_scale_by_sqrt(dst, num, src, epsilon, do_inv); 
  }  
  
  /*---  update for Adam  ---*/
  static void _adam_delta(int num, AzFloat *g1, const AzFloat *g2, AzFloat b1t, AzFloat b2t, AzFloat eps) {
    azccall_adam_delta(num, g1, g2, b1t, b2t, eps); 
  }

  /*---  ---*/  
  static void _add_repmat(const AzFloat *src, int row_num, int col_num, 
                      AzFloat *dst, int num_r, int num_c) {
    azccall_add_repmat(src, row_num, col_num, dst, num_r, num_c); 
  }                      

  static void _transpose_noblas(const AzFloat *src, int r_num, int c_num, AzFloat *dst) {
    azccall_transpose(src, r_num, c_num, dst); 
  }
  static void _transpose_cublas(const AzFloat *src, int r_num, int c_num, AzFloat *dst); 
  static void _binarize(AzFloat *dst, int num) { azccall_binarize(dst, num); }
  static void _binarize1(AzFloat *dst, int num) { azccall_binarize1(dst, num); }
  static void _mark_eq(AzFloat *dst, int num, AzFloat value) { azccall_mark_eq(dst, num, value); }
  static void _mark_gt(AzFloat *dst, int num, AzFloat value) { azccall_mark_gt(dst, num, value); }
  static void _mark_lt(AzFloat *dst, int num, AzFloat value) { azccall_mark_lt(dst, num, value); }
  static void _mark_ge(AzFloat *dst, int num, AzFloat value) { azccall_mark_ge(dst, num, value); }
  static void _mark_le(AzFloat *dst, int num, AzFloat value) { azccall_mark_le(dst, num, value); }
  static void _mark_le_rowth(AzFloat *dst, int r_num, int c_num, const AzFloat *row_th, AzFloat coeff) {
    azccall_mark_le_rowth(dst, r_num, c_num, row_th, coeff); 
  }
  static void _mark_gt_colth(AzFloat *dst, int r_num, int c_num, const AzFloat *col_th, AzFloat coeff) { 
    azccall_mark_gt_colth(dst, r_num, c_num, col_th, coeff); 
  }
  static void _get_eachCol(const AzFloat *src, int r_num, int c_num, const int *rows, AzFloat *out_vals) {
    azccall_get_eachCol(src, r_num, c_num, rows, out_vals); 
  }
  static void _exp(AzFloat *dst, int num, AzFloat *mask) { azccall_exp(dst, num, mask); }
  static void _log(AzFloat *dst, int num) { azccall_log(dst, num); }
  static void _sqrt(AzFloat *dst, int num) { azccall_sqrt(dst, num); }
  static void _square(AzFloat *dst, int num) { azccall_square(dst, num); }
  static void _pow(AzFloat *dst, int num, AzFloat val) { azccall_pow(dst, num, val); }  
  static void _inverse(AzFloat *dst, int num) { azccall_inverse(dst, num); }
  
  /*---  matrix product  ---*/             
  void _prod10(AzFloat *elm, int r_num, int c_num, 
                      const AzFloat *elm1, int row_num1,  
                      const AzFloat *elm2, int row_num2, 
                      int num,
                      const AzPstreams *streams, 
                      AzFloat alpha, AzFloat beta) const; 
  void _prod01(AzFloat *elm, int r_num, int c_num, 
                     const AzFloat *elm1, int row_num1, 
                     const AzFloat *elm2, int row_num2,
                     int num,
                     const AzPstreams *streams, 
                     AzFloat alpha, AzFloat beta) const; 
  void _prod00(AzFloat *elm, int r_num, int c_num, 
                     const AzFloat *elm1, int row_num1, 
                     const AzFloat *elm2, int row_num2, 
                     int num,
                     const AzPstreams *streams, 
                     AzFloat alpha, AzFloat beta) const; 

  inline void _copy_vardata(int rnum, 
                            const int *dcolind, /* source column index: (begin1,end1),(begin2,end2),...*/
                            const int *dxs, int dxs_num, /* array of data#'s to copy */
                            int max_cnum, 
                            const AzFloat *data,  /* source data */
                            const int *dst_dcolind, /* destination column index */
                            /*---  output  ---*/
                            AzFloat *dst_data) { /* destination data */    
    azccall_copy_vardata(rnum, dcolind, dxs, dxs_num, max_cnum, data, dst_dcolind, dst_data); 
  }             
  
  double _absmax(const AzFloat *elm, int num, int *out_index=NULL) const; 
  double _absmin(const AzFloat *elm, int num, int *out_index=NULL) const;           
  
  static void sh_config(int num, int &bb, int &tt, const char *msg); 
  
protected: 
  static AzFloat _get_sum(int op, const AzFloat *src, int num);                     
  static void _add_colSum(int op, const AzFloat *src, int row_num, int col_num, 
                          AzFloat *col_sum);  
};

/***********************************************************/
class _AzPint {
public: 
  inline static void _setval(int *dst, int val, int num) {
    azccall_setval(dst, val, num); 
  }
  inline static void _add(int *dst, int val, int num) {
    azccall_add(dst, val, num); 
  }
  inline static void _multiply(int *dst, int val, int num) {
    azccall_multiply(dst, val, num); 
  }
  inline static void _divide(int *dst, int val, int num) {
    azccall_divide(dst, val, num); 
  }  
}; 

/***********************************************************/
/* random number generator on gpu */
#include <curand.h>
class _AzPrng {
protected: 
  bool is_rg_set; 
  curandGenerator_t rg; 
public:
  _AzPrng() : is_rg_set(false) { reset(); }
  ~_AzPrng() { _destroy(); }
  void reset() {
    _destroy(); 
    curandStatus_t ret = curandCreateGenerator(&rg, CURAND_RNG_PSEUDO_DEFAULT); 
    AzCuda::throwIfcurandError(ret, "_AzPrng::reset", "curandCreateGenerator failed."); 
    is_rg_set = true; 
  }
  void reset_seed(long long int seed) {
    curandStatus_t ret = curandSetPseudoRandomGeneratorSeed(rg, seed); 
    AzCuda::throwIfcurandError(ret, "_AzPrng::reset_seed", "curandSetPseudoRandomGeneratorSeed failed."); 
  }
  void uniform_01(AzFloat *dev_data, size_t sz); 
  void normal(AzFloat *dev_data, int sz, AzFloat mean, AzFloat sdev); 

protected:
  void _destroy() {  
    if (!is_rg_set) return; 
    curandStatus_t ret = curandDestroyGenerator(rg); 
    AzCuda::throwIfcurandError(ret, "_AzPrng::_destroy", "curandDestroyGenerator failed"); 
    is_rg_set = false; 
  }
}; 
#endif 

/* * * * *
 *  AzPmat_gpu.cu
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
 
#include "AzDmat.hpp" 
#include "AzPmat_gpu.cuh"
#include "AzMemTempl.hpp"
#include "AzCuda.cuh"

extern AzPdevice dev; 
extern int max_threads, max_blocks; 
#define cublas_handle dev.h

#ifdef __AZ_DOUBLE__  
/*---  double-precision  ---*/
#define cublasXgemm cublasDgemm
#define cublasXaxpy cublasDaxpy
#define cublasXamax cublasIdamax
#define cublasXamin cublasIdamin
#define curandGenUniform curandGenerateUniformDouble
#define curandGenNormal  curandGenerateNormalDouble
#define cublasXgeam cublasDgeam
#define cublasXasum cublasDasum
#define cublasXnrm2 cublasDnrm2
#define cublasXdot cublasDdot
#define cublasXscal cublasDscal
#else
/*---  single-precision  ---*/
#define cublasXgemm cublasSgemm
#define cublasXaxpy cublasSaxpy
#define cublasXamax cublasIsamax
#define cublasXamin cublasIsamin
#define curandGenUniform curandGenerateUniform
#define curandGenNormal  curandGenerateNormal 
#define cublasXgeam cublasSgeam
#define cublasXasum cublasSasum
#define cublasXnrm2 cublasSnrm2
#define cublasXdot cublasSdot
#define cublasXscal cublasSscal
#endif

/* pointers are all device pointers unless specified otherwise */

/*-------------------------------------------------------------*/
template <class T>
void _AzParr<T>::free() {
  if (elm != NULL) {  
    dev.pmem.free(no, elm, sizeof(T)*num);      
    elm = NULL; 
  }
  num = 0; 
}  
template void _AzParr<int>::free(); 
template void _AzParr<AzFloat>::free(); 
template void _AzParr<AzByte>::free(); 

/*-------------------------------------------------------------*/  
template <class T>
void _AzParr<T>::free_alloc(int inp_num, const char *str1, const char *str2) {
  free();  
  if (inp_num > 0) {
    size_t sz = sizeof(T)*inp_num; 
    elm = (T *)dev.pmem.alloc(no, sz, str1, str2);   
    num = inp_num; 
  }
  else if (inp_num < 0) {
    AzBytArr s(str1); s << " " << str2; 
    AzX::throw_if(true, "_AzParr::free_alloc", "negative area size -- possibly overflowing", s.c_str()); 
  }
}
template void _AzParr<int>::free_alloc(int, const char *, const char *); 
template void _AzParr<AzFloat>::free_alloc(int, const char *, const char *); 
template void _AzParr<AzByte>::free_alloc(int, const char *, const char *);

/*-------------------------------------------------------------*/  
void _AzPmat::_copy(AzFloat *dst, const AzFloat *src, int num, AzFloat coeff) 
{
  if (coeff != 1) azccall_copy(dst, src, num, coeff); 
  else            AzCuda::memcpy(dst, src, num*sizeof(src[0]), cudaMemcpyDeviceToDevice, "_AzPmat::_copy"); 
}
  
/*-------------------------------------------------------------*/
void _AzPmat::_add_axpy(AzFloat *dst, const AzFloat *src, int num, AzFloat coeff) 
{
  if (coeff == 0 || num <= 0) return; 
  cublasStatus_t ret = cublasXaxpy(cublas_handle, num, &coeff, src, 1, dst, 1); 
  AzCuda::throwIfblasError(ret, "_AzPmat::_add_axpy", "cublasXaxpy (addition) failed"); 
}

/*-------------------------------------------------------------*/
void _AzPmat::_multiply_scal(AzFloat *dst, AzFloat coeff, int num) 
{
  if (coeff == 1 || num <= 0) return; 
  cublasStatus_t ret = cublasXscal(cublas_handle, num, &coeff, dst, 1); 
  AzCuda::throwIfblasError(ret, "_AzPmat::_multiply_scal", "cublasXscal (scaling) failed"); 
}
  
/*---  matrix product  ---*/             
/*-------------------------------------------------------------*/
void _AzPmat::_prod10(AzFloat *elm, int r_num, int c_num, 
                      const AzFloat *elm1, int row_num1,  
                      const AzFloat *elm2, int row_num2, 
                      int num,
                      const AzPstreams *streams,                      
                      AzFloat alpha, AzFloat beta) const
{
  if (r_num <= 0 || c_num <= 0) return; 

  if (streams != NULL) {
    streams->setStream(cublas_handle); 
  }
  /* C = alpha op(A) op(B) + beta C */
  cublasStatus_t ret = cublasXgemm(cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N, 
                    r_num, c_num, num, 
                    &alpha, 
                    elm1, row_num1, elm2, row_num2, &beta, 
                    elm, r_num); 
                    
  AzCuda::throwIfblasError(ret, "_AzPmat::_prod10", "cublasSgemm (matrix multiplication) failed"); 
}                     

/*-------------------------------------------------------------*/
void _AzPmat::_prod01(AzFloat *elm, int r_num, int c_num, 
                     const AzFloat *elm1, int row_num1, 
                     const AzFloat *elm2, int row_num2,
                     int num,
                     const AzPstreams *streams, 
                     AzFloat alpha, AzFloat beta) const
{
  if (r_num <= 0 || c_num <= 0) return; 

  if (streams != NULL) {
    streams->setStream(cublas_handle); 
  }  
  cublasStatus_t ret = cublasXgemm
                    (cublas_handle, CUBLAS_OP_N, CUBLAS_OP_T, 
                    r_num, c_num, num, 
                    &alpha, 
                    elm1, row_num1, elm2, row_num2, &beta, 
                    elm, r_num);                 
  AzCuda::throwIfblasError(ret, "_AzPmat::_prod01", "cublasSgemm (matrix multiplication) failed");          
}

/*-------------------------------------------------------------*/
void _AzPmat::_prod00(AzFloat *elm, int r_num, int c_num, 
                     const AzFloat *elm1, int row_num1, 
                     const AzFloat *elm2, int row_num2, 
                     int num,
                     const AzPstreams *streams, 
                     AzFloat alpha, AzFloat beta) const
{
  if (r_num <= 0 || c_num <= 0) return; 

  if (streams != NULL) {
    streams->setStream(cublas_handle); 
  }  
  cublasStatus_t ret = cublasXgemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, 
                    r_num, c_num, num, 
                    &alpha, 
                    elm1, row_num1, elm2, row_num2, &beta, 
                    elm, r_num); 
                   
  AzCuda::throwIfblasError(ret, "_AzPmat::_prod00", "cublasSgemm (matrix multiplication) failed");     
}                     

/*-------------------------------------------------------------*/
double _AzPmat::_absmax(const AzFloat *elm, int num, int *out_index) const 
{
  const char *eyec = "_AzPmat::_absmax"; 
  int index; 
  cublasStatus_t ret = cublasXamax(cublas_handle, num, elm, 1, &index);         
  AzCuda::throwIfblasError(ret, eyec, "cublasXamax failed");     
  
  AzFloat h_val; 
  cudaError_t ret2 = cudaMemcpy(&h_val, elm+index-1, sizeof(AzFloat), cudaMemcpyDeviceToHost); 
  AzCuda::throwIfError(ret2, eyec); 
  if (out_index != NULL) *out_index = index; 
  return (double)h_val; 
} 

/*-------------------------------------------------------------*/
double _AzPmat::_absmin(const AzFloat *elm, int num, int *out_index) const 
{
  const char *eyec = "_AzPmat::_absmin"; 
  int index; 
  cublasStatus_t ret = cublasXamin(cublas_handle, num, elm, 1, &index);         
  AzCuda::throwIfblasError(ret, eyec, "cublasXamin failed");     
  
  AzFloat h_val; 
  cudaError_t ret2 = cudaMemcpy(&h_val, elm+index-1, sizeof(AzFloat), cudaMemcpyDeviceToHost); 
  AzCuda::throwIfError(ret2, eyec); 
  if (out_index != NULL) *out_index = index; 
  return (double)h_val; 
} 

/*-------------------------------------------------------------*/
/* faster: 3.18 vs. 121.5 */
AzFloat _AzPmat::_absSum_cublas(const AzFloat *elm, int num)
{
  const char *eyec = "_AzPmat::_abssum_cublas"; 
  AzFloat asum = 0; 
  cublasStatus_t ret = cublasXasum(cublas_handle, num, elm, 1, &asum);         
  AzCuda::throwIfblasError(ret, eyec, "cublasXasum failed"); 
  return asum; 
}

/*-------------------------------------------------------------*/
/* faster : 3.43 vs. 121.58 */
AzFloat _AzPmat::_norm2_cublas(const AzFloat *elm, int num)
{
  const char *eyec = "_AzPmat::_norm2_cublas"; 
  AzFloat nrm2 = 0; 
  cublasStatus_t ret = cublasXnrm2(cublas_handle, num, elm, 1, &nrm2);         
  AzCuda::throwIfblasError(ret, eyec, "cublasXnrm2 failed"); 
  return nrm2; 
}

/*-------------------------------------------------------------*/
AzFloat _AzPmat::_sum_cublas(const AzFloat *elm, int num)
{
  _AzParr<AzFloat> val; val.free_alloc(1); 
  AzFloat one = 1; 
  val.copy_from_host(&one, 1); 
  AzFloat sum = 0; 
  cublasStatus_t ret = cublasXdot(cublas_handle, num, elm, 1, val._dptr(), 0, &sum); 
  AzCuda::throwIfblasError(ret, "_AzPmat::_sum_dot", "cublasXdot failed"); 
  return sum; 
} 
 
/*---------------------------------------------------------------*/  
void _AzPmat::sh_config(int num, int &bb, int &tt, const char *msg) {
  AzX::throw_if((num <= 0), msg, "_AzPmat::sh_cofig, num must be positive"); 
  tt = MIN(num, MIN(max_threads, azc_numShared)); 
  bb = MIN((num+tt-1)/tt, max_blocks);    
} 
 
/*-------------------------------------------------------------*/
AzFloat _AzPmat::_get_sum(int op, const AzFloat *src, int num) 
{
  if (num <= 0) return 0; 
  int bb, tt; 
  sh_config(num, bb, tt, "_get_sum");
  _AzParr<AzFloat> o; 
  o.free_alloc(bb, "_AzPmat::_get_sum");  /* device memory */
  azc_kernel(azcsh_sum,bb,tt)(op, src, num, o._dptr_u()); 
  AzCuda::check_error("_AzPmat::_get_sum", bb, tt); 
  
  AzFloat *arr = NULL;     
  AzBaseArray<AzFloat> a_arr; 
  a_arr.alloc(&arr, bb, "_AzPmat::_get_sum", "arr");  /* host memory */
  o.copy_to_host(arr, bb); 
  AzFloat sum = 0; 
  for (int ix = 0; ix < bb; ++ix) {
    sum += arr[ix]; 
  }
  return sum; 
}

/*-------------------------------------------------------------*/
int _AzPmat::_nz(const AzFloat *src, int num) 
{
  if (num <= 0) return 0; 
  int bb, tt; 
  sh_config(num, bb, tt, "_nz");
  _AzParr<int> o; 
  o.free_alloc(bb, "_AzPmat::_nz");  /* device memory */
  azc_kernel(azcsh_nz,bb,tt)(src, num, o._dptr_u()); 
  AzCuda::check_error("_AzPmat::_nz", bb, tt); 
  
  AzIntArr ia; 
  ia.reset(bb, 0); 
  o.copy_to_host(ia.point_u(), bb); 
  int sum = ia.sum(); 
  return sum; 
}

/*-------------------------------------------------------------*/
double _AzPmat::_min(const AzFloat *src, int num, int *out_index) 
{
  if (num <= 0) {
    if (out_index != NULL) *out_index = -1; 
    return 0; 
  }
  int bb, tt; 
  sh_config(num, bb, tt, "_min");
  _AzParr<int> _ind; 
  _ind.free_alloc(bb, "_AzPmat::_min,_ind");  /* device memory */
  _AzParr<double> _val; 
  _val.free_alloc(bb, "_AzPmat::_min,_val");  /* device memory */
  azc_kernel(azcsh_min,bb,tt)(src, num, _ind._dptr_u(), _val._dptr_u()); 
  AzCuda::check_error("_AzPmat::_min", bb, tt); 

  AzDvect v_val(bb); 
  _val.copy_to_host(v_val.point_u(), bb); 
  
  int index; 
  double val = v_val.min(&index); 
  if (out_index != NULL) {
    cudaError_t ret = cudaMemcpy(out_index, _ind._dptr() + index, sizeof(int), cudaMemcpyDeviceToHost); 
    AzCuda::throwIfError(ret, "_AzPmat::_min,copying index");   
  }
  return val; 
}  

/*-------------------------------------------------------------*/
double _AzPmat::_max(const AzFloat *src, int num, int *out_index) 
{
  if (num <= 0) {
    if (out_index != NULL) *out_index = -1; 
    return 0; 
  }
  int bb, tt; 
  sh_config(num, bb, tt, "_max");
  _AzParr<int> _ind; 
  _ind.free_alloc(bb, "_AzPmat::_max,_ind");  /* device memory */
  _AzParr<double> _val; 
  _val.free_alloc(bb, "_AzPmat::_max,_val");  /* device memory */
  azc_kernel(azcsh_max,bb,tt)(src, num, _ind._dptr_u(), _val._dptr_u()); 
  AzCuda::check_error("_AzPmat::_max", bb, tt); 

  AzDvect v_val(bb); 
  _val.copy_to_host(v_val.point_u(), bb); 
  
  int index; 
  double val = v_val.max(&index); 
  if (out_index != NULL) {
    cudaError_t ret = cudaMemcpy(out_index, _ind._dptr() + index, sizeof(int), cudaMemcpyDeviceToHost); 
    AzCuda::throwIfError(ret, "_AzPmat::_max,copying index");   
  }
  return val; 
}  

/*-------------------------------------------------------------*/
void _AzPmat::_add_colSum(int op, const AzFloat *src, int row_num, int col_num, 
                             AzFloat *col_sum) 
{
  if (row_num <= 0 || col_num <= 0) return; 
  int tt = MIN(row_num, MIN(max_threads, azc_numShared));   
  for (int cx = 0; cx < col_num; cx += max_blocks) {
    int bb = MIN(max_blocks, col_num-cx); 
    azc_kernel(azcsh_add_colSum,bb,tt)(op, src+cx*row_num, row_num, col_num-cx, col_sum+cx); 
    AzCuda::check_error("_AzPmat::_get_colSum", bb, tt); 
  }
}

/*-------------------------------------------------------------*/
void _AzPrng::uniform_01(AzFloat *dev_data, size_t sz) 
{
  AzX::throw_if((!is_rg_set), "_AzPrng::uniform_01", "not ready"); 
  AzX::throw_if((dev_data == NULL), "_AzPrng::uniform", "null pointer"); 
  if (sz <= 0) return; 
  curandStatus_t ret = curandGenUniform(rg, dev_data, sz); 
}

/*-------------------------------------------------------------*/
void _AzPrng::normal(AzFloat *dev_data, int sz, AzFloat mean, AzFloat sdev) {
  AzX::throw_if((!is_rg_set), "_AzPrng::normal", "not ready"); 
  AzX::throw_if((dev_data == NULL), "_AzPrng::normal", "null pointer"); 
  if (sz <= 0) return; 
  curandStatus_t ret = curandGenNormal(rg, dev_data, (size_t)sz, mean, sdev); 
}

/*-------------------------------------------------------------*/
void _AzPmat::_transpose_cublas(const AzFloat *src, int r_num, int c_num, AzFloat *dst) 
{
  AzFloat alpha = 1, beta = 0; 
  /* C = alpha op(A) + beta op(B) */
  cublasStatus_t ret = 
  cublasXgeam(cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N, 
              c_num, r_num,  /* dimension of the result */
              &alpha, src, r_num, 
              &beta, src, c_num,  /* dummy ptr */
              dst, c_num); 
  AzCuda::throwIfblasError(ret, "_AzPmat::_transpose_cublas", "cublasXgeam (for transpose) failed");               
}

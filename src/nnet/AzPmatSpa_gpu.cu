/* * * * *
 *  AzPmatSpa_gpu.cuh
 *  Copyright (C) 2015 Rie Johnson
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

#include "AzPmatSpa_gpu.cuh"

#ifdef __AZ_DOUBLE__  
/*---  double-precision  ---*/
#define cusparseXcsrmm cusparseDcsrmm
#define cusparseXcsrmm2 cusparseDcsrmm2
#define cusparseXcsr2csc cusparseDcsr2csc
#else
/*---  single-precision  ---*/
#define cusparseXcsrmm cusparseScsrmm
#define cusparseXcsrmm2 cusparseScsrmm2
#define cusparseXcsr2csc cusparseScsr2csc
#endif

extern AzPdevice dev; 
#define cusparse_handle dev.cusparse_handle
#define cusparse_desc dev.cusparse_desc
                           
/*------------------------------------------------*/
void _AzPmatSpa::_csc2csr(int r_num, int c_num, 
                           const AzFloat *csc_vals, const int *csc_ptrs, const int *csc_rows, int vals_num, /* input */
                           AzFloat *csr_vals, int *csr_ptrs, int *csr_cols) /* output */
const
{
  cusparseStatus_t ret = 
  cusparseXcsr2csc(cusparse_handle, c_num, r_num,  /* b/c using csr2csc to convert csc into csr */
                   vals_num, 
                   csc_vals, csc_ptrs, csc_rows, 
                   csr_vals, csr_cols, csr_ptrs,  /* NOTE: order! */
                   CUSPARSE_ACTION_NUMERIC, CUSPARSE_INDEX_BASE_ZERO); 
                 
  AzCuda::sync_check_error_if_debug("_AzPmatSpa::_csc2csr", 0, 0);                    
  AzCuda::throwIfsparseError(ret, "_AzPmatSpa::_csc2csr", "cuspasreXcsr2csc"); 
}                           
  
/*------------------------------------------------*/  
/* cusparse: C = alpha * op(A) * B + beta * C   (A is csr, B is dense) */
void _AzPmatSpa::_prod_csr_dense(
                  AzFloat *dst, int r_num, int c_num, 
                  const AzFloat *csr_vals, const int *csr_ptrs, const int *csr_cols, int vals_num, /* sparse */
                  const AzFloat *src1, int r_num1, int c_num1, /* dense */
                  AzFloat alpha, AzFloat beta) 
const                   
{ 
  cusparseOperation_t op = CUSPARSE_OPERATION_NON_TRANSPOSE; 
  int m = r_num, n = c_num, k = r_num1; 
  cusparseStatus_t ret = cusparseXcsrmm(cusparse_handle, op, 
                 m, n, k, vals_num, &alpha, cusparse_desc, 
                 csr_vals, csr_ptrs, csr_cols, 
                 src1, r_num1, &beta, 
                 dst, r_num); 
  AzCuda::sync_check_error_if_debug("_AzPmatSpa::_prod_csr_dense", 0, 0);                  
  AzCuda::throwIfsparseError(ret, "_AzPmatSpa::_prod_csr_dense0", "cusparseXcsrmm"); 
}                  

#ifdef __AZ_CSRMM2__
/*------------------------------------------------*/  
/* cusparse: C = alpha * op(A) * op(B) + beta * C   (A is csr, B is dense) */
void _AzPmatSpa::_prod_csr_dense_mm2(
                  bool do_tran0, bool do_tran1, 
                  AzFloat *dst, int r_num, int c_num, 
                  const AzFloat *csr_vals, const int *csr_ptrs, const int *csr_cols, int vals_num, /* sparse */
                  const AzFloat *src1, int r_num1, int c_num1, /* dense */
                  AzFloat alpha, AzFloat beta) 
const                   
{ 
  cusparseOperation_t op0 = (do_tran0) ? CUSPARSE_OPERATION_TRANSPOSE : CUSPARSE_OPERATION_NON_TRANSPOSE; 
  cusparseOperation_t op1 = (do_tran1) ? CUSPARSE_OPERATION_TRANSPOSE : CUSPARSE_OPERATION_NON_TRANSPOSE;  
  int m = r_num, n = c_num, k = r_num1; 
  cusparseStatus_t ret = cusparseXcsrmm2(cusparse_handle, op0, op1, 
                 m, n, k, vals_num, &alpha, cusparse_desc, 
                 csr_vals, csr_ptrs, csr_cols, 
                 src1, r_num1, &beta, 
                 dst, r_num); 
  AzCuda::sync_check_error_if_debug("_AzPmatSpa::_prod_csr_dense_mm2", 0, 0);                  
  AzCuda::throwIfsparseError(ret, "_AzPmatSpa::_prod_csr_dense0", "cusparseXcsrmm2"); 
} 
#endif 

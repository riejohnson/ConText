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
 
#ifndef _AZ_PMAT_SPA_GPU_CUH_
#define _AZ_PMAT_SPA_GPU_CUH_

#include <cuda_runtime.h>
#include <cusparse_v2.h>
#include "AzCuda.cuh"
#include "AzCuda_PmatSpa.cuh"

class _AzPmatSpa {
public: 
  /*---  cusparse  ---*/
  void _prod_csr_dense(
                  AzFloat *dst, int r_num, int c_num, 
                  const AzFloat *csr_vals, const int *csr_ptrs, const int *csr_cols, int vals_num, /* sparse */
                  const AzFloat *src1, int r_num1, int c_num1, /* dense */
                  AzFloat alpha, AzFloat beta) const; 
  void _csc2csr(int r_num, int c_num, 
                const AzFloat *csc_vals, const int *csc_ptrs, const int *csc_rows, int vals_num, /* input */
                AzFloat *csr_vals, int *csr_ptrs, int *csr_cols) const; /* output */

#ifdef __AZ_CSRMM2__
  void _prod_csr_dense_mm2(bool do_tran0, bool do_tran1, 
                  AzFloat *dst, int r_num, int c_num, 
                  const AzFloat *csr_vals, const int *csr_ptrs, const int *csr_cols, int vals_num, /* sparse */
                  const AzFloat *src1, int r_num1, int c_num1, /* dense */
                  AzFloat alpha, AzFloat beta) const;                 
#endif                 
                
  /*---  not cusparse ---*/
  void _prod_dense1_sparse0(
        AzFloat *dst, int r_num, int c_num, 
        const AzFloat *src1, int r_num1, int c_num1, /* dense */
        const AzFloat *csc_vals, const int *csc_ptrs, const int *csc_rows, bool do_add) const {
    azc2call_prod_dense1_sparse0(dst, r_num, c_num, src1, r_num1, c_num1, csc_vals, csc_ptrs, csc_rows, do_add); 
  }                  
  void _prod_sparse0_dense1(
        AzFloat *dst, int r_num, int c_num, 
        const AzFloat *csr_vals, const int *nzrow_ptrs, const int *nzrow_rows, int nzrow_num, const int *csr_cols,                    
        const AzFloat *src2, int r_num2, int c_num2, bool do_add) const { /* dense */                  
    azc2call_prod_sparse0_dense1(dst, r_num, c_num, csr_vals, nzrow_ptrs, nzrow_rows, nzrow_num, 
                                 csr_cols, src2, r_num2, c_num2, do_add); 
  }                  
  void _prod_sparse0_dense1_a(
        AzFloat *dst, int r_num, int c_num, 
        const AzFloat *csr_vals, const int *nzrow_ptrs, const int *nzrow_rows, int nzrow_num, const int *csr_cols,                    
        const AzFloat *src2, int r_num2, int c_num2, AzFloat alpha, bool do_add) const { /* dense */                  
    azc2call_prod_sparse0_dense1_a(dst, r_num, c_num, csr_vals, nzrow_ptrs, nzrow_rows, nzrow_num, 
                                 csr_cols, src2, r_num2, c_num2, alpha, do_add); 
  }   
  void _add_sparse(AzFloat *dst, int r_num, int c_num, 
                   const AzFloat *csc_vals, const int *csc_rows, const int *csc_cols, int vals_num, 
                   AzFloat coeff) const {
    azc2call_add_sparse(dst, r_num, c_num, csc_vals, csc_rows, csc_cols, vals_num, coeff); 
  }                   
}; 
#endif   
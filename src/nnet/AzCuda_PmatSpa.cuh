/* * * * *
 *  AzCuda_PmatSpa.cuh
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

#ifndef _AZ_CUDA_PMAT_SPA_CUH_
#define _AZ_CUDA_PMAT_SPA_CUH_

#include "AzUtil.hpp"
#include "AzP.h"

  void azc2call_prod_dense1_sparse0(
                  AzFloat *dst, int r_num, int c_num, 
                  const AzFloat *src1, int r_num1, int c_num1, /* dense */
                  const AzFloat *csc_vals, const int *csc_ptrs, const int *csc_rows, bool do_add); 
  void azc2call_prod_sparse0_dense1(
                  AzFloat *dst, int r_num, int c_num, 
                  const AzFloat *csr_vals, 
                  const int *nzrow_ptrs, const int *nzrow_rows, int nzrow_num, /* only for nonzero rows */
                  const int *csr_cols,                    
                  const AzFloat *src2, int r_num2, int c_num2, /* dense */                  
                  bool do_add);                        
  void azc2call_prod_sparse0_dense1_a(
                  AzFloat *dst, int r_num, int c_num, 
                  const AzFloat *csr_vals, 
                  const int *nzrow_ptrs, const int *nzrow_rows, int nzrow_num, /* only for nonzero rows */
                  const int *csr_cols,                    
                  const AzFloat *src2, int r_num2, int c_num2, /* dense */                  
                  AzFloat alpha, bool do_add);                  
  void azc2call_add_sparse(
                  AzFloat *dst, int r_num, int c_num, 
                  const AzFloat *csc_vals, const int *csc_rows, const int *csc_cols, int vals_num, 
                  AzFloat coeff); 
#endif 
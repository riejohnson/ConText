/* * * * *
 *  AzCuda_Pmat.cuh
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

#ifndef _AZ_CUDA_PMAT_CUH_
#define _AZ_CUDA_PMAT_CUH_

/* device functions */

#include "AzP.h"
  
  /* c: column, e: ptr to elements, r: the number of rows */
  #define _column(c,e,r) ((e)+(c)*(r))
  /* r: row, c: column, p: ptr to elements, rn: the number of rows */
  #define _entry(r,c,p,rn) (*((p)+(c)*(rn)+(r))) 
 
  /* pointers are all device pointers. */

  void azc_config(int num, int &bb, int &tt, const char *msg=""); 
  
  void azccall_add_cols_d2s(AzFloat *dst, const AzFloat *src, int row_num, 
                            const int *cols_d2s, int cnum, AzFloat coeff);    
  void azccall_add_cols_s2d(AzFloat *dst, const AzFloat *src, int row_num, 
                            const int *cols_s2d, int cnum, AzFloat coeff);  
  void azccall_add_cols_s2dz(AzFloat *dst, const AzFloat *src, int row_num, 
                             const int *cols_s2d, int cnum, AzFloat coeff);  
  void azccall_add_rows_s2d(AzFloat *dst, int dst_r_num, 
                            const AzFloat *src, int src_r_num, 
                            int c_num, 
                            const int *rows_s2d, AzFloat coeff); 
                            
  /*---  copy  ---*/
  void azccall_copy_cols(AzFloat *dst, const AzFloat *src, int row_num, 
                         const int *cols, int cnum, bool do_zero_negaindex, AzFloat coeff);
  void azccall_copy(AzFloat *dst, const AzFloat *src, int num, AzFloat coeff=1); 
  void azccall_copy_cols2cols(AzFloat *dst, const AzFloat *src, int row_num, const int *cols, int cnum); 
  void azccall_copy_scol2dcol(AzFloat *dst, const AzFloat *src, int row_num, const int *src_cols, const int *dst_cols, int cnum); 
  
  /*---  dst[dst_r0::r_num] <- src[src_r0::src_r_num]  ---*/
  void azccall_copy_rowwise(AzFloat *dst, int dst_r_num, int col_num, 
                                   int dst_r0, 
                                   const AzFloat *src, int src_r_num, 
                                   int src_r0, 
                                   int r_num); 
  
  /*---  setval  ---*/
  void azccall_setval(AzFloat *dst, AzFloat val, int num); 

  /*---  add  ---*/
  void azccall_add(AzFloat *dst, const AzFloat *src, int num, AzFloat coeff=1); 

  /*---  add1: A = s*A + t*B  ---*/
  void azccall_add1(AzFloat *dst, AzFloat dst_coeff, const AzFloat *src, AzFloat src_coeff, int num); 

  /*---  add2: A = s*A + t*B + u*C ---*/
  void azccall_add2(AzFloat *dst, AzFloat dst_coeff, const AzFloat *src1, AzFloat src_coeff1, const AzFloat *src2, const AzFloat src_coeff2, int num); 

  /*---  add_sq1: A[i,j] = s*A[i,j] + t*B[i,j]^2  ---*/
  void azccall_add_sq1(AzFloat *dst, AzFloat dst_coeff, const AzFloat *src, AzFloat src_coeff, int num);
  
  /*---  addval  ---*/
  void azccall_addval(AzFloat *dst, AzFloat val, int num); 
  
  /*---  add a column to every column  ---*/
  void azccall_add_eachrow(AzFloat *dst, int r_num, int c_num, const AzFloat *src, AzFloat coeff); 

  /*---  element-wise multiplication  ---*/
  void azccall_elm_multi(AzFloat *dst, const AzFloat *src, int num, bool do_inv=false); 
  
  /*---  divide  ---*/
  void azccall_divide(AzFloat *dst, AzFloat val, int num); 

  /*---  multiply  ---*/
  void azccall_multiply(AzFloat *dst, AzFloat val, int num); 
  void azccall_multiply_eachcol(AzFloat *dst, int r_num, int c_num, const AzFloat *src, bool do_inv=false); 
  void azccall_multiply_eachrow(AzFloat *dst, int r_num, int c_num, const AzFloat *src, bool do_inv=false); 
  
  /*---  add/multiply/divide integers  ---*/
  void azccall_setval(int *dst, int val, int num); 
  void azccall_add(int *dst, int val, int num); 
  void azccall_multiply(int *dst, int val, int num); 
  void azccall_divide(int *dst, int val, int num); 
  
  /*---  truncate into [minval, maxval]  ---*/
  void azccall_trun(AzFloat *dst, int num, AzFloat minval, AzFloat maxval); 
  
  /*---  scale by RMS (for AdaDelta)  ---*/
  void azccall_scale_by_sqrt(AzFloat *dst, int num, const AzFloat *src, AzFloat epsilon, bool do_inv); 

  /*---  adam_delta  ---*/
  void azccall_adam_delta(int num, AzFloat *g1, const AzFloat *g2, AzFloat b1t, AzFloat b2t, AzFloat eps); 
  
  /*---  repmat: add num_r x num_c tiles of row_num x col_num src  ---*/
  void azccall_add_repmat(const AzFloat *src, int row_num, int col_num, 
                          AzFloat *dst, int num_r, int num_c); 

  void azccall_transpose(const AzFloat *src, int r_num, int c_num, AzFloat *dst); 
                             
  /*---  binarize  ---*/
  void azccall_binarize(AzFloat *dst, int num); 
  void azccall_binarize1(AzFloat *dst, int num); 
  void azccall_mark_eq(AzFloat *dst, int num, AzFloat val);   
  void azccall_mark_gt(AzFloat *dst, int num, AzFloat val);   
  void azccall_mark_lt(AzFloat *dst, int num, AzFloat val); 
  void azccall_mark_ge(AzFloat *dst, int num, AzFloat val);   
  void azccall_mark_le(AzFloat *dst, int num, AzFloat val); 
  void azccall_mark_le_rowth(AzFloat *dst, int r_num, int c_num, const AzFloat *row_th, AzFloat coeff);  
  void azccall_mark_gt_colth(AzFloat *dst, int r_num, int c_num, const AzFloat *col_th, AzFloat coeff);
  
  void azccall_get_eachCol(const AzFloat *src, int r_num, int c_num, 
                           const int *rows, /* input: array of size c_num */
                           AzFloat *out_vals); /* output: array of size c_num */
  
  /*---  exp, log, sqrt, square, inverse  ---*/
  void azccall_exp(AzFloat *dst, int num, AzFloat *mask); 
  void azccall_log(AzFloat *dst, int num); 
  void azccall_sqrt(AzFloat *dst, int num);
  void azccall_square(AzFloat *dst, int num);
  void azccall_pow(AzFloat *dst, int num, AzFloat val);  
  void azccall_inverse(AzFloat *dst, int num);
  
  /*---  sum, absSum, squareSum  ---*/
  #define azc_Op_Sum 1
  #define azc_Op_AbsSum 2
  #define azc_Op_SquareSum 3
  __global__ void azcsh_sum(int op, const AzFloat *src, int num, AzFloat *output); 
  
  __global__ void azcsh_nz(const AzFloat *src, int num, int *output); 
  __global__ void azcsh_add_colSum(int op, const AzFloat *src, int row_num, int col_num, AzFloat *output); 

  void azccall_min_eachCol(const AzFloat *src, int r_num, int c_num, 
                           int *out_ind,      /* array of size c_num */
                           AzFloat *out_val);  /* array of size c_num */ 
  void azccall_max_eachCol(const AzFloat *src, int r_num, int c_num, 
                           int *out_ind,      /* array of size c_num */
                           AzFloat *out_val);  /* array of size c_num */ 
                           
  __global__ void azcsh_min(const AzFloat *src, int num, int *out_ind, double *out_val); 
  __global__ void azcsh_max(const AzFloat *src, int num, int *out_ind, double *out_val); 

  /*---  set a value to a row  ---*/ 
  void azccall_setRow(AzFloat *elm, int row_num, int col_num, int row, AzFloat val); 

  /*---  copy scattered data  ---*/                    
  void azccall_copy_vardata(int rnum, /* #rows */
                           const int *dcolind, /* [data point#] pointing columns */
                           const int *dxs, int dxs_num, /* data point# */
                           int max_cnum, 
                           const AzFloat *data, 
                           const int *dst_dcolind, /* [data point#] point columns */           
                           /*---  output  ---*/
                           AzFloat *dst_data);                  
#endif 

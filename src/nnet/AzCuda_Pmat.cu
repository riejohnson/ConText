/* * * * *
 *  AzCuda_Pmat.cu
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

#include <stdio.h> 
#include "AzUtil.hpp"
#include "AzPrint.hpp"

  extern int max_threads, max_blocks; 
  extern bool __doDebug; 

#ifdef __AZ_GPU__
  #include "AzCuda.cuh"
  #include "AzCuda_Pmat.cuh"  /* azc_config */
  static void chk_err(const char *eyec, int bb, int tt) {  
    AzCuda::check_error(eyec, bb,tt);      
  }   
  void azc_config(int num, int &bb, int &tt, const char *msg) {
    AzX::throw_if((num <= 0), msg, "azc_config, num must be positive"); 
    tt = MIN(num, max_threads); 
    bb = MIN((num+tt-1)/tt, max_blocks); 
  }  
#else
  static void chk_err(const char *eyec, int bb, int tt) {}
  void azc_config(int num, int &bb, int &tt, const char *msg) {} 
#endif   
  
  /*---  copy  ---*/
  __global__ void azc_copy(AzFloat *dst, const AzFloat *src, int num, AzFloat coeff) {
    for (int ex = azc_thno; ex < num; ex += azc_thnum) dst[ex] = src[ex]*coeff;   
  } 
  void azccall_copy(AzFloat *dst, const AzFloat *src, int num, AzFloat coeff) {
    if (num <= 0) return; int bb,tt; azc_config(num, bb,tt, "_copy"); 
    azc_kernel(azc_copy,bb,tt)(dst, src, num, coeff); chk_err("_copy",bb,tt); 
  }

  /*---  copy column_i to column_i for selected i's  ---*/
  /*---  NOTE: cols (column#'s) must be unique  ---*/
  __global__ void azc_copy_cols2cols(AzFloat *dst, const AzFloat *src, int row_num, 
                                     const int *cols, int cnum) {
    int num = row_num * cnum; 
    for (int ex = azc_thno; ex < num; ex += azc_thnum) {
      int row = ex % row_num, index = ex / row_num; 
      int col = cols[index]; 
      if (col >= 0) {
        _entry(row, col, dst, row_num) = _entry(row, col, src, row_num); 
      }
    }
  }
  void azccall_copy_cols2cols(AzFloat *dst, const AzFloat *src, int row_num, 
                              const int *cols, int cnum) {
    int num = row_num * cnum; 
    if (num <= 0) return; 
    int bb,tt; azc_config(num, bb,tt, "_copy_cols_to_cols"); 
    azc_kernel(azc_copy_cols2cols,bb,tt)(dst, src, row_num, cols, cnum); 
    chk_err("_copy_cols2cols",bb,tt);                         
  }  
 
  /*---  copy column_i to column_j for selected (i,j)'s  ---*/
  /*---  NOTE: dst_cols (destination column#'s) must be unique  ---*/
  __global__ void azc_copy_scol2dcol(AzFloat *dst, const AzFloat *src, int row_num, 
                                     const int *src_cols, const int *dst_cols, int cnum) {
    int num = row_num * cnum; 
    for (int ex = azc_thno; ex < num; ex += azc_thnum) {
      int row = ex % row_num, index = ex / row_num; 
      int src_col = (src_cols == NULL) ? index : src_cols[index]; 
      int dst_col = (dst_cols == NULL) ? index : dst_cols[index]; 
      if (src_col >= 0 && dst_col >= 0) _entry(row, dst_col, dst, row_num) = _entry(row, src_col, src, row_num); 
    }
  }
  void azccall_copy_scol2dcol(AzFloat *dst, const AzFloat *src, int row_num, 
                              const int *src_cols, const int *dst_cols, int cnum) {
    int num = row_num * cnum; 
    if (num <= 0) return; 
    int bb,tt; azc_config(num, bb,tt, "_copy_scol2dcol(s,d)"); 
    azc_kernel(azc_copy_scol2dcol,bb,tt)(dst, src, row_num, src_cols, dst_cols, cnum); 
    chk_err("_copy_scol2dcol",bb,tt);                         
  }  

  /*---  copy column_cols[i] to column_i ---*/
  __global__ void azc_copy_cols(AzFloat *dst, const AzFloat *src, int row_num, 
                                const int *cols, int cnum, 
                                bool do_zero_negaindex, /* true: set zero if index is negative */
                                                        /* false: don't touch if index is negative */
                                AzFloat coeff) {
    int num = row_num * cnum; 
    if (num == 0) return; 
    for (int ex = azc_thno; ex < num; ex += azc_thnum) {
      int row = ex % row_num; 
      int dst_col = ex / row_num; 
      int src_col = cols[dst_col]; 
      AzFloat *my_dst = _column(dst_col, dst, row_num); 
      if (src_col >= 0) {
        const AzFloat *my_src = _column(src_col, src, row_num);   
        my_dst[row] = my_src[row]*coeff; 
      }
      else if (do_zero_negaindex) {
        my_dst[row] = 0; 
      }
    }
  }
  void azccall_copy_cols(AzFloat *dst, const AzFloat *src, int row_num, 
                         const int *cols, int cnum, bool do_zero_negaindex, AzFloat coeff) {
    int num = row_num * cnum; int bb,tt; azc_config(num, bb,tt, "_copy_cols"); 
    azc_kernel(azc_copy_cols,bb,tt)(dst, src, row_num, cols, cnum, do_zero_negaindex, coeff); 
    chk_err("_copy_cols",bb,tt);                         
  }

  /*---  add specified columns  ---*/
  __global__ void azc_add_cols_d2s(AzFloat *dst, const AzFloat *src, int row_num, 
                                const int *cols_d2s, int cnum, /* dst2src */
                                AzFloat coeff) {
    int num = row_num * cnum; 
    if (num == 0) return; 
    for (int ex = azc_thno; ex < num; ex += azc_thnum) {
      int row = ex % row_num; 
      int dst_col = ex / row_num; 
      int src_col = cols_d2s[dst_col]; 
      AzFloat *my_dst = _column(dst_col, dst, row_num); 
      if (src_col >= 0) {
        const AzFloat *my_src = _column(src_col, src, row_num);   
        my_dst[row] += my_src[row]*coeff; 
      }
    }
  }
  void azccall_add_cols_d2s(AzFloat *dst, const AzFloat *src, int row_num, 
                        const int *cols_d2s, int cnum, AzFloat coeff) {
    int num = row_num * cnum; 
    int bb,tt; azc_config(num, bb,tt, "_add_cols_d2s"); 
    azc_kernel(azc_add_cols_d2s,bb,tt)(dst, src, row_num, cols_d2s, cnum, coeff); 
    chk_err("_add_cols_d2s",bb,tt);                         
  }

  /*---  add specified columns  ---*/
  __global__ void azc_add_cols_s2d(AzFloat *dst, const AzFloat *src, int row_num, 
                                const int *cols_s2d, int cnum, /* src2dst */
                                AzFloat coeff) {
    int num = row_num * cnum; 
    if (num == 0) return; 
    for (int ex = azc_thno; ex < num; ex += azc_thnum) {
      int row = ex % row_num;     
      int src_col = ex / row_num; 
      int dst_col = cols_s2d[src_col];      
      if (dst_col >= 0) {
        _entry(row, dst_col, dst, row_num) += _entry(row, src_col, src, row_num)*coeff; 
      }
    }
  }
  void azccall_add_cols_s2d(AzFloat *dst, const AzFloat *src, int row_num, 
                        const int *cols_s2d, int cnum, AzFloat coeff) {
    int num = row_num * cnum; 
    int bb,tt; azc_config(num, bb,tt, "_cols_s2d"); 
    azc_kernel(azc_add_cols_s2d,bb,tt)(dst, src, row_num, cols_s2d, cnum, coeff); 
    chk_err("_add_cols_s2d",bb,tt);                         
  }
  __global__ void azc_add_cols_s2dz(AzFloat *dst, const AzFloat *src, int row_num, 
                                const int *cols_s2d, int cnum, /* src2dst */
                                AzFloat coeff) {
    int num = row_num * cnum; 
    if (num == 0) return; 
    for (int ex = azc_thno; ex < num; ex += azc_thnum) {
      int row = ex%row_num, src_col = ex/row_num, dst_col = cols_s2d[src_col];      
      if (dst_col >= 0) {
        AzFloat val = _entry(row, src_col, src, row_num); 
        if (val != 0) _entry(row, dst_col, dst, row_num) += val*coeff; 
      }
    }
  }   
  void azccall_add_cols_s2dz(AzFloat *dst, const AzFloat *src, int row_num, 
                        const int *cols_s2d, int cnum, AzFloat coeff) {
    int num = row_num * cnum; 
    int bb,tt; azc_config(num, bb,tt, "_cols_s2dz"); 
    azc_kernel(azc_add_cols_s2dz,bb,tt)(dst, src, row_num, cols_s2d, cnum, coeff); 
    chk_err("_add_cols_s2dz",bb,tt);                         
  }
  
  /*---  add specified rows  ---*/
  __global__ void azc_add_rows_s2d(AzFloat *dst, int dst_r_num, 
                                   const AzFloat *src, int src_r_num, 
                                   int c_num, 
                                   const int *rows_s2d, /* src2dst */
                                   AzFloat coeff) {
    int num = src_r_num * c_num; 
    if (num == 0) return; 
    for (int ex = azc_thno; ex < num; ex += azc_thnum) {
      int src_row = ex % src_r_num;     
      int col = ex / src_r_num; 
      int dst_row = rows_s2d[src_row];      
      if (dst_row >= 0) {
        _entry(dst_row, col, dst, dst_r_num) += _entry(src_row, col, src, src_r_num)*coeff; 
      }
    }
  }
  void azccall_add_rows_s2d(AzFloat *dst, int dst_r_num, 
                            const AzFloat *src, int src_r_num, 
                            int c_num, 
                            const int *rows_s2d, AzFloat coeff) {
    int num = src_r_num * c_num; int bb,tt; azc_config(num, bb,tt, "_add_rows_s2d"); 
    azc_kernel(azc_add_rows_s2d,bb,tt)(dst, dst_r_num, src, src_r_num, c_num, rows_s2d, coeff); 
    chk_err("_add_rows_s2d",bb,tt);                         
  }  
  
  /*---  setval  ---*/
  __global__ void azc_setval(AzFloat *dst, AzFloat val, int num) {
    for (int ex = azc_thno; ex < num; ex += azc_thnum) dst[ex] = val;  
  }
  void azccall_setval(AzFloat *dst, AzFloat val, int num) {
    if (num <= 0) return; int bb,tt; azc_config(num, bb,tt, "_setval"); 
    azc_kernel(azc_setval,bb,tt)(dst, val, num); chk_err("_setval",bb,tt); 
  }   
  
  /*---  add (as fast as axpy)  ---*/
  __global__ void azc_add(AzFloat *dst, const AzFloat *src, int num, AzFloat coeff) {
    for (int ex = azc_thno; ex < num; ex += azc_thnum) dst[ex] += (src[ex]*coeff);   
  }
  void azccall_add(AzFloat *dst, const AzFloat *src, int num, AzFloat coeff) {
    if (coeff == 0 || num <= 0) return; int bb,tt; azc_config(num, bb,tt, "_add"); 
    azc_kernel(azc_add,bb,tt)(dst, src, num, coeff); chk_err("_add",bb,tt);   
  }  

  /*---  addval  ---*/
  __global__ void azc_addval(AzFloat *dst, AzFloat val, int num) {  
    for (int ex = azc_thno; ex < num; ex += azc_thnum) dst[ex] += val; 
  }
  void azccall_addval(AzFloat *dst, AzFloat val, int num) {
    if (val == 0 || num <= 0) return;  int bb,tt; azc_config(num, bb,tt, "_addval"); 
    azc_kernel(azc_addval,bb,tt)(dst, val, num); chk_err("_addval",bb,tt); 
  }  
  
  /*---  add a column vector to every column  ---*/
  __global__ void azc_add_eachrow(AzFloat *dst, int r_num, int c_num, const AzFloat *src, AzFloat coeff) {  
    int num = r_num*c_num; 
    for (int ex = azc_thno; ex < num; ex += azc_thnum) dst[ex] += coeff*src[ex%r_num]; 
  }
  void azccall_add_eachrow(AzFloat *dst, int r_num, int c_num, const AzFloat *src, AzFloat coeff) {
    int num = r_num*c_num; if (num <= 0) return; int bb,tt; azc_config(num, bb,tt, "_add_eachrow"); 
    azc_kernel(azc_add_eachrow,bb,tt)(dst, r_num, c_num, src, coeff); chk_err("_add_eachrow",bb,tt); 
  }
  
  /*---  element-wise multiplication  ---*/
  __global__ void azc_elm_multi(AzFloat *dst, const AzFloat *src, int num, bool do_inv) {
    for (int ex = azc_thno; ex < num; ex += azc_thnum) {       
      if (do_inv) {
        if (src[ex] == 0) dst[ex] = 0; 
        else              dst[ex] /= src[ex]; 
      }
      else {
        dst[ex] *= src[ex]; 
      }
    }  
  }  
  void azccall_elm_multi(AzFloat *dst, const AzFloat *src, int num, bool do_inv) {
    if (num <= 0) return; int bb,tt; azc_config(num, bb,tt, "_elm_multi(element-wise)"); 
    azc_kernel(azc_elm_multi,bb,tt)(dst, src, num, do_inv); chk_err("_elm_multi",bb,tt); 
  }  

  /*---  multiply src[i] (or 1/src[i]) to the i-th column  ---*/
  __global__ void azc_multiply_eachcol(AzFloat *dst, int r_num, int c_num, const AzFloat *src, bool do_inv) {
    int num = r_num * c_num; 
    for (int ex = azc_thno; ex < num; ex += azc_thnum) {       
      int col = ex / r_num; 
      AzFloat val = src[col]; 
      if (do_inv) {
        if (val == 0) dst[ex] = 0; 
        else          dst[ex] /= val; 
      }
      else {
        dst[ex] *= val; 
      }
    }  
  }  
  void azccall_multiply_eachcol(AzFloat *dst, int r_num, int c_num, const AzFloat *src, bool do_inv) {
    int num = r_num * c_num; 
    if (num <= 0) return; int bb,tt; azc_config(num, bb,tt, "_multiply_eachcol"); 
    azc_kernel(azc_multiply_eachcol,bb,tt)(dst, r_num, c_num, src, do_inv); chk_err("_multiply_eachcol",bb,tt); 
  }
  
  /*---  multiply src[i] (or 1/src[i]) to the i-th row  ---*/
  __global__ void azc_multiply_eachrow(AzFloat *dst, int r_num, int c_num, const AzFloat *src, bool do_inv) {  
    int num = r_num*c_num; 
    for (int ex = azc_thno; ex < num; ex += azc_thnum) {
      AzFloat val = src[ex%r_num];  /* src[row] */
      if (do_inv) {
        if (val == 0) dst[ex] = 0; 
        else          dst[ex] /= val; 
      }
      else dst[ex] *= val; 
    }
  }
  void azccall_multiply_eachrow(AzFloat *dst, int r_num, int c_num, const AzFloat *src, bool do_inv) {
    int num = r_num*c_num; if (num <= 0) return; int bb,tt; azc_config(num, bb,tt, "_multiply_eachrow"); 
    azc_kernel(azc_multiply_eachrow,bb,tt)(dst, r_num, c_num, src, do_inv); chk_err("_multiply_eachrow",bb,tt); 
  }
    
  /*---  divide  ---*/
  __global__ void azc_divide(AzFloat *dst, AzFloat val, int num) {
    if (val == 1) return; 
    for (int ex = azc_thno; ex < num; ex += azc_thnum) dst[ex] /= val;
  }
  void azccall_divide(AzFloat *dst, AzFloat val, int num) {
    if (val == 1 || num <= 0) return; int bb,tt; azc_config(num, bb,tt, "_divide"); 
    azc_kernel(azc_divide,bb,tt)(dst, val, num); chk_err("_divide",bb,tt); 
  }   
  
  /*---  multiply  ---*/
  __global__ void azc_multiply(AzFloat *dst, AzFloat val, int num) {
    for (int ex = azc_thno; ex < num; ex += azc_thnum) dst[ex] *= val;   
  }
  void azccall_multiply(AzFloat *dst, AzFloat val, int num) {
    if (val == 1 || num <= 0) return; int bb,tt; azc_config(num, bb,tt, "_multiply"); 
    azc_kernel(azc_multiply,bb,tt)(dst, val, num); chk_err("_multiply",bb,tt); 
  }  

  /*---  set constant ---*/
  __global__ void azc_setval(int *dst, int val, int num) {
    for (int ex = azc_thno; ex < num; ex += azc_thnum) dst[ex] = val;   
  }
  void azccall_setval(int *dst, int val, int num) {
    if (num <= 0) return; int bb,tt; azc_config(num, bb,tt, "_setval(int)"); 
    azc_kernel(azc_setval,bb,tt)(dst, val, num); chk_err("_setval(int)",bb,tt);   
  }
   
  /*---  add1: A = s*A + t*B  ---*/
  __global__ void azc_add1(AzFloat *dst, AzFloat dst_coeff, const AzFloat *src, AzFloat src_coeff, int num) {
    for (int ex = azc_thno; ex < num; ex += azc_thnum) {
      dst[ex] = dst[ex]*dst_coeff + src[ex]*src_coeff;       
    }
  }
  void azccall_add1(AzFloat *dst, AzFloat dst_coeff, const AzFloat *src, AzFloat src_coeff, int num) {
    if (num <= 0) return; int bb,tt; azc_config(num, bb,tt, "_add1"); 
    azc_kernel(azc_add1,bb,tt)(dst, dst_coeff, src, src_coeff, num); 
    chk_err("_add1",bb,tt);   
  }  
  /*---  add2: A = s*A + t*B + u*C ---*/
  __global__ void azc_add2(AzFloat *dst, AzFloat dst_coeff, const AzFloat *src1, AzFloat src_coeff1, const AzFloat *src2, AzFloat src_coeff2, int num) {
    for (int ex = azc_thno; ex < num; ex += azc_thnum) {  
      dst[ex] = dst[ex]*dst_coeff + src1[ex]*src_coeff1 + src2[ex]*src_coeff2;     
    }
  }
  void azccall_add2(AzFloat *dst, AzFloat dst_coeff, const AzFloat *src1, AzFloat src_coeff1, const AzFloat *src2, AzFloat src_coeff2, int num) {
    if (num <= 0) return; int bb,tt; azc_config(num, bb,tt, "_add2"); 
    azc_kernel(azc_add2,bb,tt)(dst, dst_coeff, src1, src_coeff1, src2, src_coeff2, num); 
    chk_err("_add2",bb,tt);   
  } 

  /*---  add_sq1: A[i,j] = s*A[i,j] + t*B[i,j]^2  ---*/
  __global__ void azc_add_sq1(AzFloat *dst, AzFloat dst_coeff, const AzFloat *src, AzFloat src_coeff, int num) {
    for (int ex = azc_thno; ex < num; ex += azc_thnum) {
      dst[ex] = dst[ex]*dst_coeff + src[ex]*src[ex]*src_coeff;       
    }
  }
  void azccall_add_sq1(AzFloat *dst, AzFloat dst_coeff, const AzFloat *src, AzFloat src_coeff, int num) {
    if (num <= 0) return; int bb,tt; azc_config(num, bb,tt, "_add_sq1"); 
    azc_kernel(azc_add_sq1,bb,tt)(dst, dst_coeff, src, src_coeff, num); 
    chk_err("_add_sq1",bb,tt);   
  }    
  
  /*---  add  ---*/
  __global__ void azc_add(int *dst, int val, int num) {
    for (int ex = azc_thno; ex < num; ex += azc_thnum) dst[ex] += val;   
  }
  void azccall_add(int *dst, int val, int num) {
    if (num <= 0) return; int bb,tt; azc_config(num, bb,tt, "_add(int)"); 
    azc_kernel(azc_add,bb,tt)(dst, val, num); chk_err("_add(int)",bb,tt);   
  }  
  /*---  multiply  ---*/
  __global__ void azc_multiply(int *dst, int val, int num) {
    for (int ex = azc_thno; ex < num; ex += azc_thnum) dst[ex] *= val;   
  }
  void azccall_multiply(int *dst, int val, int num) {
    if (num <= 0) return; int bb,tt; azc_config(num, bb,tt, "_multiply(int)"); 
    azc_kernel(azc_multiply,bb,tt)(dst, val, num); 
    chk_err("_multiply(int)",bb,tt);   
  }
  /*---  divide  ---*/
  __global__ void azc_divide(int *dst, int val, int num) {
    for (int ex = azc_thno; ex < num; ex += azc_thnum) dst[ex] /= val;   
  }
  void azccall_divide(int *dst, int val, int num) {
    if (num <= 0) return; 
    AzX::pthrow_if((val == 0), "_divide(int)", "division by zero?!"); 
    int bb,tt; azc_config(num, bb,tt, "_divide(int)"); 
    azc_kernel(azc_divide,bb,tt)(dst, val, num); 
    chk_err("_divide(int)",bb,tt);   
  }

  /*---  scale by RMS (for AdaDelta)  ---*/  
  __global__ void azc_multiply_sqrt(AzFloat *dst, int num, const AzFloat *sq, AzFloat epsilon) {
    for (int i = azc_thno; i < num; i +=azc_thnum) dst[i] *= sqrt(sq[i] + epsilon); 
  }
  __global__ void azc_divide_by_sqrt(AzFloat *dst, int num, const AzFloat *sq, AzFloat epsilon) {
    for (int i = azc_thno; i < num; i +=azc_thnum) dst[i] /= sqrt(sq[i] + epsilon); 
  }  
  void azccall_scale_by_sqrt(AzFloat *dst, int num, const AzFloat *sq, AzFloat epsilon, bool do_inv) {
    int bb, tt; azc_config(num, bb, tt, "_scale_by_sqrt"); 
    if (!do_inv) azc_kernel(azc_multiply_sqrt,bb,tt)(dst, num, sq, epsilon); 
    else         azc_kernel(azc_divide_by_sqrt,bb,tt)(dst, num, sq, epsilon); 
    chk_err("_scale_by_sqrt", bb, tt); 
  } 
 
  /*---  Adam update  ---*/  
  __global__ void azc_adam_delta(int num, AzFloat *g1, const AzFloat *g2, 
                  AzFloat b1t, AzFloat b2t, AzFloat eps) {
    for (int i = azc_thno; i < num; i +=azc_thnum) g1[i] /= ((1-b1t)*(sqrt(g2[i]/(1-b2t))+eps)); 
  }
  void azccall_adam_delta(int num, AzFloat *g1, const AzFloat *g2, 
                           AzFloat b1t, AzFloat b2t, AzFloat eps) {
    int bb, tt; azc_config(num, bb, tt, "_adam_delta"); 
    azc_kernel(azc_adam_delta,bb,tt)(num, g1, g2, b1t, b2t, eps); 
    chk_err("_adam_delta", bb, tt); 
  } 
  
  /*---  truncate  ---*/
  __global__ void azc_trun(AzFloat *dst, int num, AzFloat minval, AzFloat maxval) {
    for (int ex = azc_thno; ex < num; ex += azc_thnum) dst[ex] = MIN(maxval, MAX(minval, dst[ex]));   
  }
  void azccall_trun(AzFloat *dst, int num, AzFloat minval, AzFloat maxval) {
    if (num <= 0) return; int bb,tt; azc_config(num, bb,tt, "_trun"); 
    azc_kernel(azc_trun,bb,tt)(dst, num, minval, maxval);
    chk_err("_AzPmat::_trun",bb,tt); 
  }  
  
  /*---  sum, absSum, squareSum  ---*/
  __global__ void azcsh_sum(int op, const AzFloat *src, int num, AzFloat *output) {  
    __shared__ AzFloat temp[azc_numShared]; 
    int ex; 
    temp[threadIdx.x] = 0; 
    if      (op == azc_Op_Sum)       for (ex = azc_thno; ex < num; ex += azc_thnum) temp[threadIdx.x] += src[ex]; 
    else if (op == azc_Op_AbsSum)    for (ex = azc_thno; ex < num; ex += azc_thnum) temp[threadIdx.x] += fabs(src[ex]); 
    else if (op == azc_Op_SquareSum) for (ex = azc_thno; ex < num; ex += azc_thnum) temp[threadIdx.x] += (src[ex]*src[ex]); 
    __syncthreads(); 
    
    if (threadIdx.x == 0) {
      AzFloat mysum = 0; 
      for (int ix = 0; ix < blockDim.x; ++ix) {
        mysum += temp[ix]; 
      }
      output[blockIdx.x] = mysum; 
    }
  }
  
  /*---  count nonzero components  ---*/
  __global__ void azcsh_nz(const AzFloat *src, int num, int *output) {  
    __shared__ int temp[azc_numShared]; 
    int ex; 
    temp[threadIdx.x] = 0; 
    for (ex = azc_thno; ex < num; ex += azc_thnum) if (src[ex] != 0) ++temp[threadIdx.x]; 
    __syncthreads(); 
    
    if (threadIdx.x == 0) {
      AzFloat mysum = 0; 
      for (int ix = 0; ix < blockDim.x; ++ix) {
        mysum += temp[ix]; 
      }
      output[blockIdx.x] = (int)mysum; 
    }
  }
  
  /*---  out_vals[c] <- src[rows[c],c] (for large-cat evaluation)  ---*/
  __global__ void azc_get_eachCol(const AzFloat *src, int r_num, int c_num, 
                                  const int *rows, /* input: array of size c_num */
                                  AzFloat *out_vals) { /* output: array of size c_num */
    for (int col = azc_thno; col < c_num; col += azc_thnum) out_vals[col] = _entry(rows[col], col, src, r_num); 
  } 
  void azccall_get_eachCol(const AzFloat *src, int r_num, int c_num, 
                           const int *rows, /* input: array of size c_num */
                           AzFloat *out_vals) { /* output: array of size c_num */
    int bb,tt; azc_config(c_num, bb,tt, "_get_eachCol"); 
    azc_kernel(azc_get_eachCol,bb,tt)(src, r_num, c_num, rows, out_vals); chk_err("_get_eachCol",bb,tt); 
  }   
  
  /*---  find maximum of each column  ---*/
  __global__ void azc_max_eachCol(const AzFloat *src, int r_num, int c_num, 
                                  int *out_ind,      /* array of size c_num */
                                  AzFloat *out_val) { /* array of size c_num */
    for (int col = azc_thno; col < c_num; col += azc_thnum) {
      const AzFloat *data = _column(col, src, r_num);
      AzFloat best_val = data[0]; 
      if (out_ind != NULL) out_ind[col] = 0; 
      for (int row = 1; row < r_num; ++row) {
        if (data[row] > best_val) {
          best_val = data[row]; 
          if (out_ind != NULL) out_ind[col] = row; 
        }
      }
      if (out_val != NULL) out_val[col] = best_val; 
    }
  } 
  void azccall_max_eachCol(const AzFloat *src, int r_num, int c_num, 
                           int *out_ind,      /* array of size c_num */
                           AzFloat *out_val) { /* array of size c_num */                      
    int bb,tt; 
    azc_config(c_num, bb,tt, "_max_eachCol"); 
    azc_kernel(azc_max_eachCol,bb,tt)(src, r_num, c_num, out_ind, out_val); 
    chk_err("_max_eachCol",bb,tt); 
  }  
  
  /*---  find minimum of each column  ---*/
  __global__ void azc_min_eachCol(const AzFloat *src, int r_num, int c_num, 
                                  int *out_ind, AzFloat *out_val) {  
    for (int col = azc_thno; col < c_num; col += azc_thnum) {
      const AzFloat *data = _column(col, src, r_num);
      AzFloat best_val = data[0]; 
      if (out_ind != NULL) out_ind[col] = 0; 
      for (int row = 1; row < r_num; ++row) {
        if (data[row] < best_val) {
          best_val = data[row]; 
          if (out_ind != NULL) out_ind[col] = row; 
        }
      }
      if (out_val != NULL) out_val[col] = best_val; 
    }
  }   
  void azccall_min_eachCol(const AzFloat *src, int r_num, int c_num, 
                           int *out_ind,      /* array of size c_num */
                           AzFloat *out_val) { /* array of size c_num */                      
    int bb,tt; azc_config(c_num, bb,tt, "_min_eachCol"); 
    azc_kernel(azc_min_eachCol,bb,tt)(src, r_num, c_num, out_ind, out_val); 
    chk_err("_min_eachCol",bb,tt); 
  }    
  
  /*---  minimum  ---*/
  __global__ void azcsh_min(const AzFloat *src, int num, 
                          int *out_ind, double *out_val) {  
    __shared__ int sh_ind[azc_numShared]; 
    __shared__ AzFloat sh_val[azc_numShared]; 
    sh_ind[threadIdx.x] = -1; 
    sh_val[threadIdx.x] = 0; 
    for (int ex = azc_thno; ex < num; ex += azc_thnum) {
      if (sh_ind[threadIdx.x] < 0 || src[ex] < sh_val[threadIdx.x]) {
        sh_ind[threadIdx.x] = ex; 
        sh_val[threadIdx.x] = src[ex]; 
      }
    }
    __syncthreads(); 
    
    if (threadIdx.x == 0) {
      out_ind[blockIdx.x] = -1; 
      out_val[blockIdx.x] = 0; 
      for (int ix = 0; ix < blockDim.x; ++ix) {
        if (sh_ind[ix] >= 0) {
          if (out_ind[blockIdx.x] < 0 || sh_val[ix] < out_val[blockIdx.x]) {
            out_ind[blockIdx.x] = sh_ind[ix]; 
            out_val[blockIdx.x] = sh_val[ix]; 
          }
        }
      }
    }
  }
  
  /*---  maximum  ---*/
  __global__ void azcsh_max(const AzFloat *src, int num, 
                          int *out_ind, double *out_val) {  
    __shared__ int sh_ind[azc_numShared]; 
    __shared__ AzFloat sh_val[azc_numShared]; 
    sh_ind[threadIdx.x] = -1; 
    sh_val[threadIdx.x] = 0; 
    for (int ex = azc_thno; ex < num; ex += azc_thnum) {
      if (sh_ind[threadIdx.x] < 0 || src[ex] > sh_val[threadIdx.x]) {
        sh_ind[threadIdx.x] = ex; 
        sh_val[threadIdx.x] = src[ex]; 
      }
    }
    __syncthreads(); 
    
    if (threadIdx.x == 0) {
      out_ind[blockIdx.x] = -1; 
      out_val[blockIdx.x] = 0; 
      for (int ix = 0; ix < blockDim.x; ++ix) {
        if (sh_ind[ix] >= 0) {
          if (out_ind[blockIdx.x] < 0 || sh_val[ix] > out_val[blockIdx.x]) {
            out_ind[blockIdx.x] = sh_ind[ix]; 
            out_val[blockIdx.x] = sh_val[ix]; 
          }
        }
      }
    }
  }  
  
  /*---  column-wise sum, absSum, squareSum  ---*/
  __global__ void azcsh_add_colSum(int op, const AzFloat *src, int row_num, int col_num, AzFloat *output) {  
    int col = blockIdx.x;
    if (col >= col_num) return; 
 
    int ex0 = col*row_num; 
    int ex1 = ex0 + row_num; 
    __shared__ AzFloat temp[azc_numShared]; 
    int ex; 
    temp[threadIdx.x] = 0; 
    if      (op == azc_Op_Sum)       for (ex=ex0+threadIdx.x; ex<ex1; ex+=blockDim.x) temp[threadIdx.x] += src[ex];  
    else if (op == azc_Op_AbsSum)    for (ex=ex0+threadIdx.x; ex<ex1; ex+=blockDim.x) temp[threadIdx.x] += fabs(src[ex]);  
    else if (op == azc_Op_SquareSum) for (ex=ex0+threadIdx.x; ex<ex1; ex+=blockDim.x) temp[threadIdx.x] += (src[ex]*src[ex]); 
    __syncthreads(); 
    
    if (threadIdx.x == 0) {
      AzFloat mysum = 0;    
      for (int ix = 0; ix < blockDim.x; ++ix) {
        mysum += temp[ix]; 
      }
      output[blockIdx.x] += mysum; 
    }
  } 
  
  /*---  repmat: add num_r x num_c tiles of row_num x col_num src  ---*/
  __global__ void azc_add_repmat(const AzFloat *src, int row_num, int col_num, 
                             AzFloat *dst, 
                             int num_r, int num_c) {
    int dst_row_num = num_r * row_num; 
    int num = dst_row_num * num_c * col_num; 
    for (int ix = azc_thno; ix < num; ix += azc_thnum) {
      int dst_row = ix % dst_row_num; 
      int dst_col = ix / dst_row_num; 
      int src_row = dst_row % row_num; 
      int src_col = dst_col % col_num;
      dst[ix] += (_column(src_col, src, row_num))[src_row]; 
    }
  }
  void azccall_add_repmat(const AzFloat *src, int row_num, int col_num, 
                          AzFloat *dst, int num_r, int num_c) {
    int num = num_r * row_num * num_c * col_num; 
    if (num <= 0) return; int bb,tt; azc_config(num, bb,tt, "_add_repmat"); 
    azc_kernel(azc_add_repmat,bb,tt)(src, row_num, col_num, dst, num_r, num_c); 
    chk_err("_add_repmat",bb,tt); 
  }  
 
  /*---  transpose  ---*/
  __global__ void azc_transpose(const AzFloat *src, int r_num, int c_num, AzFloat *dst) {
    int num = r_num*c_num; 
    for (int ex = azc_thno; ex < num; ex += azc_thnum) {
      int row = ex % r_num; 
      int col = ex / r_num; 
      dst[c_num*row + col] = src[ex]; 
    }
  }
  void azccall_transpose(const AzFloat *src, int r_num, int c_num, AzFloat *dst) {
    int num = r_num*c_num; 
    if (num <= 0) return; 
    int bb,tt; azc_config(num, bb,tt, "_transpose"); 
    azc_kernel(azc_transpose,bb,tt)(src, r_num, c_num, dst); chk_err("_transpose",bb,tt); 
  }

  /*---  binarize: 1 if x>0; -1 if x<0; 0 otherwise  ---*/
  __global__ void azc_binarize(AzFloat *dst, int num) {  
    for (int ix = azc_thno; ix < num; ix += azc_thnum) {
      if      (dst[ix] > 0) dst[ix] = 1; 
      else if (dst[ix] < 0) dst[ix] = -1; 
      else                  dst[ix] = 0; 
    }
  }  
  void azccall_binarize(AzFloat *dst, int num) {
    if (num <= 0) return; int bb,tt; azc_config(num, bb,tt, "_binarize"); 
    azc_kernel(azc_binarize,bb,tt)(dst, num); chk_err("_binarize",bb,tt); 
  }  
  
  /*---  binarize1: (x!=0)?1:0  ---*/
  __global__ void azc_binarize1(AzFloat *dst, int num) {  
    for (int ix = azc_thno; ix < num; ix += azc_thnum) {
      if (dst[ix] != 0) dst[ix] = 1; 
      else              dst[ix] = 0; 
    }
  }
  void azccall_binarize1(AzFloat *dst, int num) {
    if (num <= 0) return; int bb,tt; azc_config(num, bb,tt, "_binarize1"); 
    azc_kernel(azc_binarize1,bb,tt)(dst, num); chk_err("_binarize1",bb,tt); 
  }  

  /*---  mark_eq: (x==v)?1:0  ---*/
  __global__ void azc_mark_eq(AzFloat *dst, int num, AzFloat val) {  
    for (int ix = azc_thno; ix < num; ix += azc_thnum) dst[ix] = (dst[ix] == val) ? 1 : 0; 
  }
  void azccall_mark_eq(AzFloat *dst, int num, AzFloat value) {
    if (num <= 0) return; 
    int bb,tt; azc_config(num, bb,tt, "_mark_eq"); 
    azc_kernel(azc_mark_eq,bb,tt)(dst, num, value); chk_err("_mark_eq",bb,tt); 
  }  

  /*---  mark_gt: (x>v)?1:0  ---*/
  __global__ void azc_mark_gt(AzFloat *dst, int num, AzFloat val) {  
    for (int ix = azc_thno; ix < num; ix += azc_thnum) dst[ix] = (dst[ix] > val) ? 1 : 0; 
  }
  void azccall_mark_gt(AzFloat *dst, int num, AzFloat val) {
    if (num <= 0) return; 
    int bb,tt; azc_config(num, bb,tt, "_mark_gt"); 
    azc_kernel(azc_mark_gt,bb,tt)(dst, num, val); chk_err("_mark_gt",bb,tt); 
  }  

  /*---  mark_lt: (x<v)?1:0  ---*/
  __global__ void azc_mark_lt(AzFloat *dst, int num, AzFloat val) {  
    for (int ix = azc_thno; ix < num; ix += azc_thnum) dst[ix] = (dst[ix] < val) ? 1 : 0; 
  }
  void azccall_mark_lt(AzFloat *dst, int num, AzFloat val) {
    if (num <= 0) return; 
    int bb,tt; azc_config(num, bb,tt, "_mark_lt"); 
    azc_kernel(azc_mark_lt,bb,tt)(dst, num, val); chk_err("_mark_lt",bb,tt); 
  }  

  /*---  mark_ge: (x>=v)?1:0  ---*/
  __global__ void azc_mark_ge(AzFloat *dst, int num, AzFloat val) {  
    for (int ix = azc_thno; ix < num; ix += azc_thnum) dst[ix] = (dst[ix] >= val) ? 1 : 0; 
  }
  void azccall_mark_ge(AzFloat *dst, int num, AzFloat val) {
    if (num <= 0) return; 
    int bb,tt; azc_config(num, bb,tt, "_mark_ge"); 
    azc_kernel(azc_mark_ge,bb,tt)(dst, num, val); chk_err("_mark_ge",bb,tt); 
  }  

  /*---  mark_le: (x<=v)?1:0  ---*/
  __global__ void azc_mark_le(AzFloat *dst, int num, AzFloat val) {  
    for (int ix = azc_thno; ix < num; ix += azc_thnum) dst[ix] = (dst[ix] <= val) ? 1 : 0; 
  }
  void azccall_mark_le(AzFloat *dst, int num, AzFloat val) {
    if (num <= 0) return; 
    int bb,tt; azc_config(num, bb,tt, "_mark_le"); 
    azc_kernel(azc_mark_le,bb,tt)(dst, num, val); chk_err("_mark_le",bb,tt); 
  }  
 
  /*---  mark_le: (x[row,]<=v[row])?coeff:0  ---*/
  __global__ void azc_mark_le_rowth(AzFloat *dst, int r_num, int c_num, const AzFloat *row_th, AzFloat coeff) {  
    int num = r_num*c_num; 
    for (int ix = azc_thno; ix < num; ix += azc_thnum) {
      int row = ix%r_num; 
      dst[ix] = (dst[ix] <= row_th[row]) ? coeff : 0; 
    }
  }
  void azccall_mark_le_rowth(AzFloat *dst, int r_num, int c_num, const AzFloat *row_th, AzFloat coeff) {
    if (r_num*c_num <= 0) return; 
    int bb,tt; azc_config(r_num*c_num, bb,tt, "_mark_le_rowth"); 
    azc_kernel(azc_mark_le_rowth,bb,tt)(dst, r_num, c_num, row_th, coeff); chk_err("_mark_le_rowth",bb,tt); 
  } 
  
  /*---  mark_gt_colth: (x[,col]>v[col])?coeff:0  ---*/
  __global__ void azc_mark_gt_colth(AzFloat *dst, int r_num, int c_num, const AzFloat *col_th, AzFloat coeff) {
    int num = r_num*c_num;      
    for (int ix = azc_thno; ix < num; ix += azc_thnum) {
      int col = ix/r_num; 
      dst[ix] = (dst[ix] > col_th[col]) ? coeff : 0; 
    }
  }   
  void azccall_mark_gt_colth(AzFloat *dst, int r_num, int c_num, const AzFloat *col_th, AzFloat coeff) {
    if (r_num*c_num <= 0) return; 
    int bb,tt; azc_config(r_num*c_num, bb,tt, "_mark_gt_colth"); 
    azc_kernel(azc_mark_gt_colth,bb,tt)(dst, r_num, c_num, col_th, coeff); chk_err("_mark_gt_colth",bb,tt); 
  }   
  
  /*---  exp  ---*/
  __global__ void azc_exp(AzFloat *dst, int num, AzFloat *mask) {  
    for (int ix = azc_thno; ix < num; ix += azc_thnum) {
      if (mask != NULL) mask[ix] = azc_exp_mask(dst[ix]); 
      dst[ix] = myexp(dst[ix]); 
    }
  }
  void azccall_exp(AzFloat *dst, int num, AzFloat *mask) {
    if (num <= 0) return; 
    int bb,tt; azc_config(num, bb,tt, "_exp"); 
    azc_kernel(azc_exp,bb,tt)(dst, num, mask); chk_err("_exp",bb,tt); 
  }  

  /*---  log  ---*/
  __global__ void azc_log(AzFloat *dst, int num) {  
    for (int ix = azc_thno; ix < num; ix += azc_thnum) {
      dst[ix] = log(dst[ix]); 
    }
  }
  void azccall_log(AzFloat *dst, int num) {
    if (num <= 0) return; 
    int bb,tt; azc_config(num, bb,tt, "_log"); 
    azc_kernel(azc_log,bb,tt)(dst, num); chk_err("_log",bb,tt); 
  }  

  /*---  power  ---*/
  __global__ void azc_pow(AzFloat *dst, int num, AzFloat val) {  
    for (int ix = azc_thno; ix < num; ix += azc_thnum) dst[ix] = pow(dst[ix], val); 
  }
  void azccall_pow(AzFloat *dst, int num, AzFloat val) { 
    if (num <= 0) return; int bb,tt; azc_config(num, bb,tt, "_pow"); 
    azc_kernel(azc_pow,bb,tt)(dst, num, val); chk_err("_pow",bb,tt); 
  }
  
  /*---  square root  ---*/
  __global__ void azc_sqrt(AzFloat *dst, int num) {  
    for (int ix = azc_thno; ix < num; ix += azc_thnum) {
      dst[ix] = sqrt(dst[ix]); 
    }
  }
  void azccall_sqrt(AzFloat *dst, int num) { 
    if (num <= 0) return; int bb,tt; azc_config(num, bb,tt, "_sqrt"); 
    azc_kernel(azc_sqrt,bb,tt)(dst, num); chk_err("_sqrt",bb,tt); 
  }
  
  /*---  square  ---*/
  __global__ void azc_square(AzFloat *dst, int num) {  
    for (int ix = azc_thno; ix < num; ix += azc_thnum) {
      dst[ix] *= dst[ix]; 
    }
  }
  void azccall_square(AzFloat *dst, int num) {
    if (num <= 0) return; int bb,tt; azc_config(num, bb,tt, "_square"); 
    azc_kernel(azc_square,bb,tt)(dst, num); chk_err("_square",bb,tt); 
  }  

  /*---  inverse  ---*/
  __global__ void azc_inverse(AzFloat *dst, int num) {  
    for (int ix = azc_thno; ix < num; ix += azc_thnum) {
      if (dst[ix] != 0) dst[ix] = 1/dst[ix]; 
    }
  }
  void azccall_inverse(AzFloat *dst, int num) {
    if (num <= 0) return; int bb,tt; azc_config(num, bb,tt, "_inverse"); 
    azc_kernel(azc_inverse,bb,tt)(dst, num); chk_err("_inverse",bb,tt); 
  }  
  
  /*--------------------------------------------------*/  
  __global__ void azc_setRow(AzFloat *elm, int row_num, int col_num, 
                             int row, AzFloat val) 
  {                          
    for (int col = azc_thno; col < col_num; col += azc_thnum) {
      *(_column(col, elm, row_num) + row) = val; 
    }
  }
  void azccall_setRow(AzFloat *dst, int row_num, int col_num, int row, AzFloat val) {
    if (row_num <= 0 || col_num <= 0) return; int bb,tt; azc_config(col_num, bb,tt, "_setRow"); 
    azc_kernel(azc_setRow,bb,tt)(dst, row_num, col_num, row, val); chk_err("_setRow",bb,tt); 
  }  
  
  /*--------------------------------------------------*/  
  /* dst[dst_r0::r_num] <- src[src_r0::r_num] */
  __global__ void azc_copy_rowwise(AzFloat *dst, int dst_r_num, int col_num, 
                                   int dst_r0, 
                                   const AzFloat *src, int src_r_num, 
                                   int src_r0, 
                                   int r_num)
  {                          
    int num = r_num*col_num;   
    for (int ix = azc_thno; ix < num; ix += azc_thnum) {
      int col = ix / r_num; 
      int row = ix % r_num; 
      (_column(col, dst, dst_r_num))[dst_r0+row] = (_column(col, src, src_r_num))[src_r0+row]; 
    }
  }
  void azccall_copy_rowwise(AzFloat *dst, int dst_r_num, int col_num, int dst_r0, 
                               const AzFloat *src, int src_r_num, int src_r0, 
                               int r_num) {
    int num = col_num * r_num; 
    if (num <= 0) return; int bb,tt; azc_config(num, bb,tt, "_copy_rowwise"); 
    azc_kernel(azc_copy_rowwise,bb,tt)(dst, dst_r_num, col_num, dst_r0, src, src_r_num, src_r0, r_num); 
    chk_err("_copy_rowwise",bb,tt);   
  }   
    
  /*------------------------------------------------*/
  __global__ void azc_copy_vardata(int rnum, 
                                   const int *dcolind, /* source column index: (begin1,end1),(begin2,end2),...*/
                                   const int *dxs, int dxs_num, /* array of data#'s to copy */
                                   int max_cnum, 
                                   const AzFloat *data,  /* source data */
                                   const int *dst_dcolind, /* destination column index */
                                   /*---  output  ---*/
                                   AzFloat *dst_data) /* destination data */
  {
    int max_len = max_cnum*rnum; 
    int num = dxs_num * max_len; 
    for (int idx = azc_thno; idx < num; idx += azc_thnum) {
      int dst_dx = idx / max_len; 
      int offs = idx % max_len; 
      int dx = dxs[dst_dx]; 
      int pos = dcolind[dx*2]*rnum + offs; 
      if (pos < dcolind[dx*2+1]*rnum) {
        int dst_pos = dst_dcolind[dst_dx*2]*rnum + offs;  
        dst_data[dst_pos] = data[pos]; 
      }
    }
  }
  void azccall_copy_vardata(int rnum, 
                           const int *dcolind, 
                           const int *dxs, int dxs_num, 
                           int max_cnum, 
                           const AzFloat *data, 
                           const int *dst_dcolind,                                  
                           /*---  output  ---*/
                           AzFloat *dst_data) {                                  
    int bb,tt; 
    if (dxs_num <= 0 || max_cnum <= 0) return; 
    azc_config(dxs_num*max_cnum*rnum, bb,tt, "_copy_vardata"); 
    azc_kernel(azc_copy_vardata,bb,tt)(rnum, dcolind, dxs, dxs_num, max_cnum, data, dst_dcolind, dst_data); 
    chk_err("_copy_vardata",bb,tt); 
  }   

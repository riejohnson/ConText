/* * * * *
 *  AzCuda_PmatApp.cu
 *  Copyright (C) 2014,2015,2017 Rie Johnson
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

#include "AzCuda_PmatApp.cuh"

#ifdef __AZ_GPU__
  extern int max_threads, max_blocks; 
  #include "AzCuda.cuh"
  #include "AzCuda_Pmat.cuh"  /* azc_config */
  static void chk_err(const char *eyec, int bb, int tt) {
    AzCuda::check_error(eyec, bb, tt);   
  }   
#else
  static bool azc_config(int num, int &bb, int &tt, const char *msg="") { return true; }
  static void chk_err(const char *eyec, int bb, int tt) {}
#endif   
  
  /*****  PmatApp  *****/
   /*---  L1L2: sqrt(x^2+d^2)-d  ---*/
  __global__ void azc_l1l2(const AzFloat *src, AzFloat *dst, int num, AzFloat del) {  
    double delsq = del*del; 
    int ix; 
    for (ix = azc_thno; ix < num; ix += azc_thnum) {
      dst[ix] = sqrt(src[ix]*src[ix] + delsq) - del; 
    }
  }

  /*---  L1L2-deriv: x/sqrt(x^2+d^2)  ---*/
  __global__ void azc_add_l1l2deriv(const AzFloat *src, AzFloat *dst, int num, AzFloat del, AzFloat coeff) {  
    double delsq = del*del; 
    int ix; 
    for (ix = azc_thno; ix < num; ix += azc_thnum) {
      dst[ix] += (src[ix]*coeff) / sqrt(src[ix]*src[ix] + delsq); 
    }
  } 
  void azccall_l1l2(const AzFloat *src, AzFloat *dst, int num, AzFloat del) {
    int bb, tt; 
    azc_config(num, bb, tt, "azccall_l1l2"); 
    azc_kernel(azc_l1l2,bb,tt)(src, dst, num, del); 
    chk_err("azccall_l1l2", bb, tt);     
  }  
  void azccall_add_l1l2deriv(const AzFloat *src, AzFloat *dst, int num, AzFloat del, AzFloat coeff) {
    int bb, tt; 
    azc_config(num, bb, tt, "azccall_add_l1l2deriv"); 
    azc_kernel(azc_add_l1l2deriv,bb,tt)(src, dst, num, del, coeff); 
    chk_err("azccall_add_l1l2deriv", bb, tt); 
  } 
  
  /*--------------------------------------------------*/  
  /*---        min(th,max(0,x)) activation         ---*/
  /*--------------------------------------------------*/  
  __global__ void azc_activate_th(int num, AzFloat *elm, AzFloat th) {
    for (int ex = azc_thno; ex < num; ex += azc_thnum) {
      if      (elm[ex] <= 0)           elm[ex] = 0; 
      else if (th > 0 && elm[ex] > th) elm[ex] = th;     
    }
  }                    
  __global__ void azc_activate_th_deriv(int num, AzFloat *elm, AzFloat th,
                             AzFloat *deriv_elm) /* must not be NULL */ {
    for (int ex = azc_thno; ex < num; ex += azc_thnum) {                             
      if (elm[ex] <= 0) {
        elm[ex] = 0; 
        deriv_elm[ex] = 0; 
      } else if (th > 0 && elm[ex] > th) {
        elm[ex] = th; 
        deriv_elm[ex] = 0; 
      } else {
        deriv_elm[ex] = 1; 
      }
    }
  }
  __global__ void azc_activate_rect(int num, AzFloat *elm) {
    for (int ex = azc_thno; ex < num; ex += azc_thnum) if (elm[ex] <= 0) elm[ex] = 0;  
  }                    
  __global__ void azc_activate_rect_deriv(int num, AzFloat *elm,
                             AzFloat *deriv_elm) /* must not be NULL */ {
    for (int ex = azc_thno; ex < num; ex += azc_thnum) {                             
      if (elm[ex] <= 0) {
        elm[ex] = 0; 
        deriv_elm[ex] = 0; 
      } else {
        deriv_elm[ex] = 1; 
      }
    }
  }
  /*--------------------------------------------------*/   
  void azccall_activate_th(AzFloat *elm, int num, AzFloat th, 
                           AzFloat *deriv_elm) /* may be NULL */ {
    /* note: single vs multi didn't matter */
    int bb, tt; 
    azc_config(num, bb, tt, "azccall_activate_th"); 
    if (th < 0) {
      if (deriv_elm == NULL) azc_kernel(azc_activate_rect,bb,tt)(num, elm); 
      else                   azc_kernel(azc_activate_rect_deriv,bb,tt)(num, elm, deriv_elm);     
    }
    else {
      if (deriv_elm == NULL) azc_kernel(azc_activate_th,bb,tt)(num, elm, th); 
      else                   azc_kernel(azc_activate_th_deriv,bb,tt)(num, elm, th, deriv_elm); 
    }
    chk_err("azccall_activate_th", bb, tt); 
  }
  
  /*--------------------------------------------------*/  
  /*---        max(0,x) if x>0; ax otherwise       ---*/
  /*--------------------------------------------------*/  
  __global__ void azc_activate_leaky_rect(int num, AzFloat *elm, AzFloat aa) {
    for (int ex = azc_thno; ex < num; ex += azc_thnum) if (elm[ex] <= 0) elm[ex] *= aa;     
  }                    
  __global__ void azc_activate_leaky_rect_deriv(int num, AzFloat *elm, AzFloat aa, 
                             AzFloat *deriv_elm) /* must not be NULL */ {
    for (int ex = azc_thno; ex < num; ex += azc_thnum) {                             
      if (elm[ex] <= 0) { elm[ex] *= aa; deriv_elm[ex] = aa; } 
      else                deriv_elm[ex] = 1; 
    }
  }
  /*--------------------------------------------------*/   
  void azccall_activate_leaky_rect(AzFloat *elm, int num, AzFloat aa, 
                           AzFloat *deriv_elm) /* may be NULL */ {
    int bb, tt; azc_config(num, bb, tt, "azccall_activate_leaky_rect"); 
    if (deriv_elm == NULL) azc_kernel(azc_activate_leaky_rect,bb,tt)(num, elm, aa); 
    else                   azc_kernel(azc_activate_leaky_rect_deriv,bb,tt)(num, elm, aa, deriv_elm); 
    chk_err("azccall_activate_leaky_rect", bb, tt); 
  }  

  /*--------------------------------------------------*/
  /*---             sigmoid activation             ---*/
  /*--------------------------------------------------*/  
  __global__ void azc_activate_log(AzFloat *elm, int num) {
    int ex; 
    for (ex = azc_thno; ex < num; ex += azc_thnum) {  
      elm[ex] = (AzFloat)1/((AzFloat)1+myexp(-elm[ex])); 
    }
  } 
  /*--------------------------------------------------*/  
  __global__ void azc_activate_log_deriv(AzFloat *elm, int num, 
                                    AzFloat *deriv_elm) {
    int ex; 
    for (ex = azc_thno; ex < num; ex += azc_thnum) {  
      AzFloat ss = (AzFloat)1/((AzFloat)1+myexp(-elm[ex])); 
      elm[ex] = ss; 
      deriv_elm[ex] = ss*(1-ss); 
    }
  } 
  /*--------------------------------------------------*/  
  void azccall_activate_log(AzFloat *elm, int num,
                           AzFloat *deriv_elm) /* may be NULL */ {
    int bb, tt; 
    azc_config(num, bb, tt, "azccall_activate_log"); 
    if (deriv_elm == NULL) azc_kernel(azc_activate_log,bb,tt)(elm, num); 
    else                   azc_kernel(azc_activate_log_deriv,bb,tt)(elm, num, deriv_elm); 
    chk_err("azccall_activate_log", bb, tt); 
  } 
  
  /*------------------------------------------------*/
  /*---             tanh activation              ---*/
  /*------------------------------------------------*/
  __global__ void azc_activate_tanh(AzFloat *elm, int num) {
    int ex; 
    for (ex = azc_thno; ex < num; ex += azc_thnum) {
      double e2 = myexp(2*elm[ex]); 
      elm[ex] = (AzFloat)((e2-1)/(e2+1));   
    }
  } 
  /*--------------------------------------------------*/  
  __global__ void azc_activate_tanh_deriv(AzFloat *elm, int num, 
                                  AzFloat *deriv_elm) {
    int ex; 
    for (ex = azc_thno; ex < num; ex += azc_thnum) {
      double e2 = myexp(2*elm[ex]); 
      elm[ex] = (AzFloat)((e2-1)/(e2+1));
      deriv_elm[ex] = 4*e2/(e2+1)/(e2+1); 
    }
  }
  /*--------------------------------------------------*/  
  void azccall_activate_tanh(AzFloat *elm, int num,
                             AzFloat *deriv_elm) /* may be NULL */ {
    int bb, tt; 
    azc_config(num, bb, tt, "azccall_activate_tanh"); 
    if (deriv_elm == NULL) azc_kernel(azc_activate_tanh,bb,tt)(elm, num); 
    else                   azc_kernel(azc_activate_tanh_deriv,bb,tt)(elm, num, deriv_elm); 
    chk_err("azccall_activate_tanh", bb, tt); 
  } 
  
  /*--------------------------------------------------*/
  /*---            softplus activation             ---*/
  /*--------------------------------------------------*/  
  __global__ void azc_activate_softplus(AzFloat *elm, int num) {
    int ex; 
    for (ex = azc_thno; ex < num; ex += azc_thnum) {
      double e2 = myexp(elm[ex]); 
      elm[ex] = (AzFloat)log(1+e2);   
    }
  } 
  /*--------------------------------------------------*/  
  __global__ void azc_activate_softplus_deriv(AzFloat *elm, int num, 
                                    AzFloat *deriv_elm) {
    int ex; 
    for (ex = azc_thno; ex < num; ex += azc_thnum) {
      double e2 = myexp(elm[ex]); 
      elm[ex] = (AzFloat)log(1+e2);  
      deriv_elm[ex] = (AzFloat)(e2/(e2+1)); 
    }
  }
  /*--------------------------------------------------*/  
  void azccall_activate_softplus(AzFloat *elm, int num, 
                                 AzFloat *deriv_elm) /* may be NULL */ {
    int bb, tt; 
    azc_config(num, bb, tt, "azccall_activate_softplus"); 
    if (deriv_elm == NULL) azc_kernel(azc_activate_softplus,bb,tt)(elm, num); 
    else                   azc_kernel(azc_activate_softplus_deriv,bb,tt)(elm, num, deriv_elm); 
    chk_err("azccall_activate_softplus", bb, tt); 
  }   

  /*--------------------------------------------------*/    
  /*------------------------------------------------*/
  __global__ void azc_truncate(AzFloat *elm, int num, AzFloat border) {
    int i; 
    for (i = azc_thno; i < num; i +=azc_thnum) {
      elm[i] = MAX(-border, MIN(border, elm[i])); 
    }
  }
  __global__ void azc_truncate_deriv(AzFloat *elm, int num, AzFloat border, 
                                     AzFloat *deriv_elm) /* must not be NULL */ {
    int i; 
    for (i = azc_thno; i < num; i +=azc_thnum) {
      if (elm[i] < -border) {
        elm[i] = -border; 
        deriv_elm[i] = 0; 
      } 
      else if (elm[i] > border) {
        elm[i] = border; 
        deriv_elm[i] = 0; 
      } 
    }
  }  
  void azccall_truncate(AzFloat *elm, int num, AzFloat border, 
                        AzFloat *deriv_elm) /* may be NULL */ {
    int bb, tt; 
    azc_config(num, bb, tt, "azccall_truncate"); 
    if (deriv_elm == NULL) azc_kernel(azc_truncate,bb,tt)(elm, num, border); 
    else                   azc_kernel(azc_truncate_deriv,bb,tt)(elm, num, border, deriv_elm); 
    chk_err("azccall_truncate", bb, tt); 
  }  

  /*******           For convolutional layers             *******/
  /*------------------------------------------------------------*/
  /*---              filtering/unfiltering                   ---*/
  /*------------------------------------------------------------*/
  __global__ void azc_add_with_map(int num, const azcparam_add_with_map p) 
  {
    int ex;   
    for (ex = azc_thno; ex < num; ex += azc_thnum) {
      int index = ex;  
      int row = index % p.row_num; 
      index = index / p.row_num; 
      int dx = index / p.width2; 
      int col2 = index % p.width2; 
      if (dx >= p.data_num) continue;  
    
      int base1 = dx*p.width1; 
      int base2 = dx*p.width2; 
  
      const int *col1 = _column(col2, p.a2to1, p.nummax); 
      AzFloat *e2 = _column(base2+col2, p.elm2, p.row_num); 
      int ix; 
      for (ix = 0; ix < p.nummax; ++ix) {
        if (col1[ix] == p.stopper) break; 
        const AzFloat *e1 = _column(base1+col1[ix], p.elm1, p.row_num); 
        e2[row] += e1[row];     
      }
    }
  }

  /*------------------------------------------------------------*/
  void azccall_add_with_map(const azcparam_add_with_map p) {
    int num = p.data_num * p.width2 * p.row_num; 
    int bb, tt; 
    azc_config(num, bb, tt, "azccall_add_with_map"); 
    azc_kernel(azc_add_with_map,bb,tt)(num, p); 
    chk_err("azccall_add_with_map", bb, tt); 
  }
 
  /*----------------------------------------------------------------------
   *   add_with_map_var: Added on 2/14/2017 for speed-up. Old "fastVar=2".
   *   - a2to1 gives a template for the largest data.
   *     Watch out for col1's pointing outside.  
   *   - This is almost as fast as and more flexible than 
   *     the "no-template" version. 
   *---------------------------------------------------------------------*/   
  __global__ void azc_add_with_map_var(int num, const azcparam_var p) {
    for (int ex = azc_thno; ex < num; ex += azc_thnum) {
      int row = ex % p.row_num; 
      int c2 = ex / p.row_num; 

      int dx = p.c2dx[c2/p.c2dx_denomi]; 
      
      int base1 = p.pos1[dx], num1 = p.pos1[dx+1] - base1;     
      int col2 = c2 - p.pos2[dx]; 

      const int *col1_arr = _column(col2, p.col2to1_ptr, p.nummax); 
      AzFloat *e2 = _column(c2, p.elm2, p.row_num); 

      for (int ix = 0; ix < p.nummax; ++ix) {
        int col1 = col1_arr[ix]; 
        if (col1 == p.stopper) break; 
        if (col1 >= num1) continue; 
        const AzFloat *e1 = _column(base1+col1, p.elm1, p.row_num); 
        e2[row] += e1[row];     
      }
    }
  }  
  /*------------------------------------------------------------*/
  void azccall_add_with_map_var(const azcparam_var p, int cnum2) {
    AzX::throw_if((cnum2 <= 0), "azccall_add_with_map_var", "cnum2 must be positive"); 
    int num = cnum2 * p.row_num; 
    int bb, tt; azc_config(num, bb, tt, "azccall_add_with_map_var");    
    azc_kernel(azc_add_with_map_var,bb,tt)(num, p);       
    chk_err("azccall_add_with_map_var", bb, tt); 
  }   

  /*******                   Pooling                      *******/  
  /*------------------------------------------------------------*/
  /*---                  average pooling                     ---*/
  /*------------------------------------------------------------*/
  __global__ void azc_pooling_avg(int num, const azcparam_pooling_avg p) 
  {
    int ex;   
    for (ex = azc_thno; ex < num; ex += azc_thnum) {
      int index = ex; 
      int row = index % p.row_num; 
      index = index / p.row_num;   
      int dx = index / p.width2; 
      int col2 = index % p.width2; 
      if (dx >= p.data_num) continue; 

      int base1 = dx*p.width1; 
      int base2 = dx*p.width2; 

      const int *col1 = _column(col2, p.col1_ptr, p.col1_nummax); 
      AzFloat *e2 = _column(base2+col2, p.elm2, p.row_num); 
      int ix; 
      for (ix = 0; ix < p.col1_nummax; ++ix) {
        if (col1[ix] == p.stopper) break; 
        const AzFloat *e1 = _column(base1+col1[ix], p.elm1, p.row_num); 
        e2[row] += e1[row];  
      }
      int col1_num = ix; 
      if (col1_num != 0) e2[row] /= (AzFloat)col1_num; 
    }
  }
  /*------------------------------------------------------------*/
  void azccall_pooling_avg(const azcparam_pooling_avg p) {
    int num = p.data_num * p.width2 * p.row_num; 
    int bb, tt; 
    azc_config(num, bb, tt, "azccall_pooling_avg"); 
    azc_kernel(azc_pooling_avg,bb,tt)(num, p); 
    chk_err("azccall_pooling_avg", bb, tt);         
  }              
  
  /*------------------------------------------------------------*/
  /*---                  average unpooling                   ---*/
  /*------------------------------------------------------------*/
  __global__ void azc_unpooling_avg(int num, const azcparam_unpooling_avg p) {
    int ex;   
    for (ex = azc_thno; ex < num; ex += azc_thnum) {
      int index = ex;     
      int row = index % p.row_num; 
      index = index / p.row_num;         
      int dx = index / p.width1; 
      int col1 = index % p.width1; 
      if (dx >= p.data_num) continue; 

      int base1 = dx*p.width1; 
      int base2 = dx*p.width2;   

      const int *col2 = _column(col1, p.col2_ptr, p.col2_nummax); 
      int ix; 
      for (ix = 0; ix < p.col2_nummax; ++ix) {
        if (col2[ix] == p.stopper) break;       
        AzFloat *e1 = _column(base1+col1, p.elm1, p.row_num); 
        const AzFloat *e2 = _column(base2+col2[ix], p.elm2, p.row_num); 
        AzFloat denomi = (AzFloat)p.col2_to_num[col2[ix]]; 
        if (denomi != 0) e1[row] += e2[row]/denomi; 
      }
    }
  }

  /*------------------------------------------------------------*/
  void azccall_unpooling_avg(const azcparam_unpooling_avg p) {
    int num = p.data_num * p.width1 * p.row_num; 
    int bb, tt; 
    azc_config(num, bb, tt, "azccall_unpooling_avg"); 
    azc_kernel(azc_unpooling_avg,bb,tt)(num, p); 
    chk_err("azccall_unpooling_avg", bb, tt);         
  }

  /*------------------------------------------------------------*/
  /*---                     l2 pooling                       ---*/
  /*------------------------------------------------------------*/
  __global__ void azc_pooling_l2(int num, const azcparam_pooling_l2 p) {
    int ex;   
    for (ex = azc_thno; ex < num; ex += azc_thnum) {
      int index = ex;   
      int row = index % p.row_num; 
      index = index / p.row_num; 
      int dx = index / p.width2; 
      int col2 = index % p.width2; 
      if (dx >= p.data_num) continue; 

      int base1 = dx*p.width1; 
      int base2 = dx*p.width2; 

      const int *col1 = _column(col2, p.col1_ptr, p.col1_nummax); 
      AzFloat *e2 = _column(base2+col2, p.elm2, p.row_num); 
      int ix; 
      for (ix = 0; ix < p.col1_nummax; ++ix) {
        if (col1[ix] == p.stopper) break; 
        const AzFloat *e1 = _column(base1+col1[ix], p.elm1, p.row_num); 
        e2[row] += e1[row]*e1[row]; 
      }
      e2[row] = sqrt(e2[row]); 
    }
  }

  /*------------------------------------------------------------*/
  void azccall_pooling_l2(const azcparam_pooling_l2 p) {
    int num = p.data_num * p.width2 * p.row_num; 
    int bb, tt; 
    azc_config(num, bb, tt, "azccall_pooling_l2");  
    azc_kernel(azc_pooling_l2,bb,tt)(num, p); 
    chk_err("azccall_pooling_l2", bb, tt);         
  }
 
  /*------------------------------------------------------------*/
  /*---                    l2 unpooling                      ---*/
  /*------------------------------------------------------------*/
  __global__ void azc_unpooling_l2(int num, const azcparam_unpooling_l2 p) {
    int ex;   
    for (ex = azc_thno; ex < num; ex += azc_thnum) {
      int index = ex; 
      int row = index % p.row_num; 
      index = index / p.row_num;    
      int dx = index / p.width1; 
      int col1 = index % p.width1; 
      if (dx >= p.data_num) continue; 

      int base1 = dx*p.width1; 
      int base2 = dx*p.width2;   

      AzFloat *e1 = _column(base1+col1, p.elm1, p.row_num); 
      const AzFloat *org_e1 = _column(base1+col1, p.org_elm1, p.row_num); 
      const int *col2 = _column(col1, p.col2_ptr, p.col2_nummax); 
      int ix; 
      for (ix = 0; ix < p.col2_nummax; ++ix) {
        if (col2[ix] == p.stopper) break;       
        const AzFloat *e2 = _column(base2+col2[ix], p.elm2, p.row_num); 
        const AzFloat *org_e2 = _column(base2+col2[ix], p.org_elm2, p.row_num); 
        if (org_e2[row] != 0) e1[row] += (e2[row] * org_e1[row] / org_e2[row]); 
      }
    }
  }

  /*------------------------------------------------------------*/
  void azccall_unpooling_l2(const azcparam_unpooling_l2 p) {
    int num = p.data_num * p.width1 * p.row_num; 
    int bb, tt; 
    azc_config(num, bb, tt, "azccall_unpooling_l2"); 
    azc_kernel(azc_unpooling_l2,bb,tt)(num, p); 
    chk_err("azccall_unpooling_l2", bb, tt);         
  }
  
  /*------------------------------------------------------------*/
  /*---                    max pooling                       ---*/
  /*------------------------------------------------------------*/
  __global__ void azc_pooling_max(int num, const azcparam_pooling_max p) {
    int ex;   
    for (ex = azc_thno; ex < num; ex += azc_thnum) {
      int index = ex; 
      int row = index % p.row_num; 
      index = index / p.row_num;         
      int dx = index / p.width2; 
      int col2 = index % p.width2; 
      if (dx >= p.data_num) continue; 
  
      int base1 = dx*p.width1; 
      int base2 = dx*p.width2; 

      const int *col1 = _column(col2, p.col1_ptr, p.col1_nummax); 
      int *chosen = (p.chosen_ptr != NULL) ? _column(col2+base2, p.chosen_ptr, p.row_num) : NULL; 
      AzFloat *max_elm = _column(col2+base2, p.elm2, p.row_num); 
      int ix; 
      for (ix = 0; ix < p.col1_nummax; ++ix) {
        if (col1[ix] == p.stopper) break; 
        const AzFloat *e1 = _column(col1[ix]+base1, p.elm1, p.row_num); 
        if (ix == 0 || e1[row] > max_elm[row]) {
          max_elm[row] = e1[row]; 
          if (chosen != NULL) chosen[row] = col1[ix]; 
        }
      }    
    }                         
  }
  /*------------------------------------------------------------*/
  void azccall_pooling_max(const azcparam_pooling_max p) {
    int num = p.data_num * p.width2 * p.row_num; 
    int bb, tt; 
    azc_config(num, bb, tt, "azccall_pooling_max"); 
    azc_kernel(azc_pooling_max,bb,tt)(num, p); 
    chk_err("azccall_pooling_max", bb, tt);         
  }

  /*------------------------------------------------------------*/  
  /*---                   max unpooling                      ---*/
  /*------------------------------------------------------------*/
  /* Note: assume overlapping pooling */
  /* thread: portions of rows of one data point  */
  __global__ void azc_unpooling_max(int num, const azcparam_unpooling_max p) {
    int ex;   
    for (ex = azc_thno; ex < num; ex += azc_thnum) {
      int index = ex;   
      int row = index % p.row_num; 
      index = index / p.row_num;    
      int dx = index % p.data_num; 
      if (dx >= p.data_num) continue; 
  
      int base1 = dx*p.width1; 
      int base2 = dx*p.width2; 
    
      int col2; 
      for (col2 = 0; col2 < p.width2; ++col2) {
        const AzFloat *e2 = _column(col2+base2, p.elm2, p.row_num); 
        const int *chosen = _column(col2+base2, p.ptr_chosen, p.row_num); 
        int col1 = chosen[row]; 
        if (col1 >= 0) { /* 3/11/2014: for variable-length pooling */
          (_column(col1+base1, p.elm1, p.row_num))[row] += e2[row]; 
        }
      }
    }
  }
  /*------------------------------------------------------------*/
  void azccall_unpooling_max(const azcparam_unpooling_max p) {
    int num = p.data_num * p.row_num; 
    int bb, tt; 
    azc_config(num, bb, tt, "azccall_unpooling_max"); 
    azc_kernel(azc_unpooling_max,bb,tt)(num, p); 
    chk_err("azccall_unpooling_max", bb, tt);         
  }
  
  /*------------------------------------------------------------*/  
  __global__ void azc_unpooling_max2(int num, const azcparam_unpooling_max p) {
    for (int ex = azc_thno; ex < num; ex += azc_thnum) {
      int index = ex; 
      int row = index % p.row_num; 
      index = index / p.row_num;         
      int dx = index / p.width1; 
      int col1 = index % p.width1; 
      if (dx >= p.data_num) return; 

      int base1 = dx*p.width1; int base2 = dx*p.width2;   

      AzFloat *e1 = _column(base1+col1, p.elm1, p.row_num); 
      const int *col2 = _column(col1, p.col1to2_ptr, p.nummax); 
      for (int ix = 0; ix < p.nummax; ++ix) {
        if (col2[ix] == p.stopper) break;       
        const int *chosen = _column(base2+col2[ix], p.ptr_chosen, p.row_num); 
        if (chosen[row] == col1) {
          const AzFloat *e2 = _column(base2+col2[ix], p.elm2, p.row_num); 
          e1[row] += e2[row]; 
        }
      }  
    }    
  }  
  /*------------------------------------------------------------*/
  void azccall_unpooling_max2(const azcparam_unpooling_max p) {
    int num = p.data_num * p.width1 * p.row_num; 
    int bb, tt; azc_config(num, bb, tt, "azccall_unpooling_max2"); 
    azc_kernel(azc_unpooling_max2,bb,tt)(num, p); 
    chk_err("azccall_unpooling_max2", bb, tt);         
  }   


  /***  To speed-up pooling with variable-sized input/output  ***/
   /*------------------------------------------------------------*/
  __global__ void azc_pooling_max_var(int num, const azcparam_var p, int *_chosen) {
    for (int ex = azc_thno; ex < num; ex += azc_thnum) {
      int row = ex % p.row_num, c2 = ex / p.row_num, dx = p.c2dx[c2/p.c2dx_denomi]; 
      int base1 = p.pos1[dx], num1 = p.pos1[dx+1] - base1; 
      int col2 = c2 - p.pos2[dx];     
      
      const int *col1_arr = _column(col2, p.col2to1_ptr, p.nummax); 
      
      int *chosen = (_chosen != NULL) ? _column(c2, _chosen, p.row_num) : NULL; 
      AzFloat *max_elm = _column(c2, p.elm2, p.row_num); 
      for (int ix = 0; ix < p.nummax; ++ix) {
        int col1 = col1_arr[ix]; 
        if (col1 == p.stopper) break; 
        if (col1 >= num1) continue;                
        const AzFloat *e1 = _column(col1+base1, p.elm1, p.row_num); 
        if (ix == 0 || e1[row] > max_elm[row]) {
          max_elm[row] = e1[row]; 
          if (chosen != NULL) chosen[row] = col1; 
        }
      }    
    }                         
  }
  /*------------------------------------------------------------*/
  void azccall_pooling_max_var(const azcparam_var p, int cnum2, int *_chosen) {
    int num = cnum2 * p.row_num; 
    int bb, tt; azc_config(num, bb, tt, "azccall_pooling_max_var"); 
    azc_kernel(azc_pooling_max_var,bb,tt)(num, p, _chosen); 
    chk_err("azccall_pooling_max_var", bb, tt);         
  }

  /*------------------------------------------------------------*/
  __global__ void azc_pooling_avg_var(int num, const azcparam_var p, int sz) {
    for (int ex = azc_thno; ex < num; ex += azc_thnum) {
      int row = ex % p.row_num, c2 = ex / p.row_num, dx = p.c2dx[c2/p.c2dx_denomi]; 
      int base1 = p.pos1[dx], num1 = p.pos1[dx+1] - base1; 
      int col2 = c2 - p.pos2[dx]; 

      const int *col1_arr = _column(col2, p.col2to1_ptr, p.nummax); 
 
      AzFloat *ptr2 = _column(c2, p.elm2, p.row_num) + row; 
      for (int ix = 0; ix < p.nummax; ++ix) {
        int col1 = col1_arr[ix]; 
        if (col1 == p.stopper) break;         
        if (col1 >= num1) continue; 
        *ptr2 += _entry(row, col1+base1, p.elm1, p.row_num);  
      }    
      *ptr2 /= (AzFloat)sz; 
    }                         
  }
  /*------------------------------------------------------------*/
  void azccall_pooling_avg_var(const azcparam_var p, int cnum2, int sz) {
    int num = cnum2 * p.row_num; 
    int bb, tt; azc_config(num, bb, tt, "azccall_pooling_avg_var"); 
    azc_kernel(azc_pooling_avg_var,bb,tt)(num, p, sz); 
    chk_err("azccall_pooling_avg_var", bb, tt);         
  }
   
  /*------------------------------------------------------------*/
  __global__ void azc_unpooling_max_var(int num, const azcparam_var p, const int *_chosen) {
    for (int ex = azc_thno; ex < num; ex += azc_thnum) { 
      int row = ex%p.row_num, c2 = ex/p.row_num, dx = p.c2dx[c2/p.c2dx_denomi];       
      int col2 = c2 - p.pos2[dx]; 
      int base1 = p.pos1[dx], num1 = p.pos1[dx+1] - base1;
      
      AzFloat *e2 = _column(c2, p.elm2, p.row_num); 
      const int *col1_arr = _column(col2, p.col2to1_ptr, p.nummax); 
      for (int ix = 0; ix < p.nummax; ++ix) {
        int col1 = col1_arr[ix]; 
        if (col1 == p.stopper) break;            
        if (col1 >= num1) continue; 
     
        if (col1 < 0) {
          if (row == 0) 
          printf("azc_unpooling_max_var col1 < 0, col1=%d, dx=%d, num1=%d, azc_thno=%d, ex=%d, row=%d, c2=%d, col2=%d\n", 
                 col1, dx, num1, azc_thno, ex, row, c2, col2); 
          continue; 
        }
       
        const int *chosen = _column(base1+col1, _chosen, p.row_num); 
        if (chosen[row] == col2) {
          const AzFloat *e1 = _column(base1+col1, p.elm1, p.row_num);            
          e2[row] += e1[row]; 
        }
      }  
    }    
  }
  /*------------------------------------------------------------*/
  void azccall_unpooling_max_var(const azcparam_var p, int cnum2, const int *_chosen) {
    int num = cnum2 * p.row_num; 
    int bb, tt; azc_config(num, bb, tt, "azccall_unpooling_max_var");  
    azc_kernel(azc_unpooling_max_var,bb,tt)(num, p, _chosen); 
    chk_err("azccall_unpooling_max_var", bb, tt);         
  }   
   
  /*------------------------------------------------------------*/
  __global__ void azc_unpooling_avg_var(int num, const azcparam_var p, int sz) {
    for (int ex = azc_thno; ex < num; ex += azc_thnum) { 
      int row = ex % p.row_num, c2 = ex / p.row_num, dx = p.c2dx[c2/p.c2dx_denomi];       
      int col2 = c2 - p.pos2[dx]; 
      int base1 = p.pos1[dx], num1 = p.pos1[dx+1] - base1;
      
      AzFloat *ptr2 = _column(c2, p.elm2, p.row_num) + row; 
      const int *col1_arr = _column(col2, p.col2to1_ptr, p.nummax); 
      for (int ix = 0; ix < p.nummax; ++ix) {
        int col1 = col1_arr[ix]; 
        if (col1 == p.stopper) break;       
        if (col1 >= num1) continue; 
        *ptr2 += (_entry(row, base1+col1, p.elm1, p.row_num)/(AzFloat)sz); 
      }  
    }    
  }
  /*------------------------------------------------------------*/
  void azccall_unpooling_avg_var(const azcparam_var p, int cnum2, int sz) {
    int num = cnum2 * p.row_num; 
    int bb, tt; azc_config(num, bb, tt, "azccall_unpooling_avg_var"); 
    azc_kernel(azc_unpooling_avg_var,bb,tt)(num, p, sz); 
    chk_err("azccall_unpooling_avg_var", bb, tt);         
  }   
  
  /*******                    Misc                        *******/    
  /*------------------------------------------------------------*/    
  /*---      rearrange (for locally-connected weights)       ---*/
  /*------------------------------------------------------------*/    
  __global__ void azc_rearrange(int num, const azcparam_rearrange p) {
    int ex;   
    for (ex = azc_thno; ex < num; ex += azc_thnum) {
      int index = ex;   
      int row = index % p.rnum; 
      index = index / p.rnum;         
      int dx = index / p.loc_num; 
      int loc = index % p.loc_num; 
      if (dx >= p.d_num) continue; 

      const AzFloat *e1 = _column(p.loc_num*dx + loc, p.elm1, p.rnum); 
      AzFloat *e2 = _column(p.d_num*loc + dx, p.elm2, p.rnum); 
      e2[row] = e1[row];    
    }
  }

  /*------------------------------------------------------------*/
  void azccall_rearrange(const azcparam_rearrange p) {
    int num = p.d_num * p.loc_num * p.rnum; 
    int bb, tt; 
    azc_config(num, bb, tt, "azccall_rearrange"); 
    azc_kernel(azc_rearrange,bb,tt)(num, p); 
    chk_err("azccall_rearrange", bb, tt);         
  }    

  /*------------------------------------------------------------*/  
  /*------------------------------------------------------------*/      
  __global__ void azc_undo_rearrange(int num, const azcparam_undo_rearrange p) {
    int ex;   
    for (ex = azc_thno; ex < num; ex += azc_thnum) {
      int index = ex;   
      int row = index % p.rnum; 
      index = index / p.rnum;         
      int dx = index / p.loc_num; 
      int loc = index % p.loc_num; 
      if (dx >= p.d_num) continue; 

      AzFloat *e1 = _column(p.loc_num*dx + loc, p.elm1, p.rnum); 
      const AzFloat *e2 = _column(p.d_num*loc + dx, p.elm2, p.rnum);     
      e1[row] = e2[row]; 
    }
  }

  /*------------------------------------------------------------*/
  void azccall_undo_rearrange(const azcparam_undo_rearrange p) {
    int num = p.d_num * p.loc_num * p.rnum; 
    int bb, tt; 
    azc_config(num, bb, tt, "azccall_undo_rearrange"); 
    azc_kernel(azc_undo_rearrange,bb,tt)(num, p); 
    chk_err("azccall_undo_rearrange", bb, tt);         
  }   

  /*------------------------------------------------------------*/
  /*-- local response normalization across neurons  (cmrnorm) --*/
  /*------------------------------------------------------------*/
  __global__ void azc_resnorm_crossmap(int num, const azcparam_resnorm_crossmap p) {
    int ex;   
    for (ex = azc_thno; ex < num; ex += azc_thnum) {
      int index = ex;   
      int row = index % p.rnum; 
      int col = index / p.rnum; 
      if (col >= p.cnum) continue; 
  
      int sz, rr0, rr1; 
      if (p.size < p.rnum) {
        int halfsize = p.size / 2; 
        sz = halfsize*2 + 1; 
        sz = 1; /* to match with cuda-convnet */
        rr0 = row - halfsize;
        rr1 = row + halfsize; 
      }
      else {
        sz = 1; 
        rr0 = 0; 
        rr1 = p.rnum-1; 
      }
    
      int base = col*p.rnum; 
      const AzFloat *input = p.elm + base; 
      AzFloat *oneplussqavg = p.elm_oneplussqavg + base; 
      AzFloat *normalized = p.elm_normalized + base;   

      AzFloat sqsum = 0; 
      int rr; 
      for (rr = rr0; rr <= rr1; ++rr) {
        int myrr = (rr+p.rnum) % p.rnum; 
        sqsum += input[myrr]*input[myrr]; 
      }
      oneplussqavg[row] = p.one+p.alpha*sqsum/(double)sz; 
      normalized[row] *= pow(oneplussqavg[row], -p.beta); 
    }
  }
  
  /*------------------------------------------------------------*/
  void azccall_resnorm_crossmap(const azcparam_resnorm_crossmap p) {
    int num = p.cnum * p.rnum; 
    int bb, tt; 
    azc_config(num, bb, tt, "azccall_resnorm_crossmap");  
    azc_kernel(azc_resnorm_crossmap,bb,tt)(num, p); 
    chk_err("azccall_resnorm_crossmap", bb, tt);         
  }
  
  /*------------------------------------------------------------*/
  /*-- local response normalization across channels (cmrnorm) --*/
  /*------------------------------------------------------------*/
  __global__ void azc_resnorm_crossmap_all(int num, const azcparam_resnorm_crossmap p, 
                  const AzFloat *col_sqsum) {
    for (int ex = azc_thno; ex < num; ex += azc_thnum) {
      int index = ex;   
      int row = index % p.rnum; 
      int col = index / p.rnum; 
      if (col >= p.cnum) continue; 

      int base = col*p.rnum; 
      AzFloat *oneplussqavg = p.elm_oneplussqavg + base; 
      AzFloat *normalized = p.elm_normalized + base;   
      
      oneplussqavg[row] = p.one+p.alpha*col_sqsum[col]; 
      normalized[row] *= pow(oneplussqavg[row], -p.beta); 
    }
  }
  
  /*------------------------------------------------------------*/
  void azccall_resnorm_crossmap_all(const azcparam_resnorm_crossmap p, const AzFloat *col_sqsum) {
    int num = p.cnum * p.rnum; 
    int bb, tt; 
    azc_config(num, bb, tt, "azccall_resnorm_crossmap_all");  
    azc_kernel(azc_resnorm_crossmap_all,bb,tt)(num, p, col_sqsum); 
    chk_err("azccall_resnorm_crossmap_all", bb, tt);         
  }  
  
  /*-------------------------------------------------------------*/  
  /*---  prep for undoing response normalization (cross map)  ---*/
  /*-------------------------------------------------------------*/
  /* tmp <- g_k * (-2 alpha beta)/N_k * v_k(1 + alpha/N sum_i v_i^2)^{-beta-1} */
  /*     =  (-2 alpha beta f_k g_k)/(N_k d_k) */
  /*  d_k := 1 + alpha/N_k sum_i v_i^2 */
  /*  f_k := v_k d_k^{-beta} */
  __global__ void azc_prep_unresnorm_crossmap(int num, const azcparam_prep_unresnorm_crossmap p) {
    int ex;   
    for (ex = azc_thno; ex < num; ex += azc_thnum) {
      int index = ex;   
      int row = index % p.rnum; 
      int col = index / p.rnum; 
      if (col >= p.cnum) continue; 

      int sz = 1; 
      int base = p.rnum*col; 
      const AzFloat *grad = p.elm_grad + base; 
      const AzFloat *aft = p.elm_aft + base; 
      const AzFloat *oneplussqavg = p.elm_oneplussqavg + base; 
      AzFloat *tmp = p.elm_tmp + base; 

      tmp[row] = (-2*p.alpha*p.beta * aft[row]*grad[row]) / (oneplussqavg[row] * (double)sz); 
    }
  }

  /*------------------------------------------------------------*/
  void azccall_prep_unresnorm_crossmap(const azcparam_prep_unresnorm_crossmap p) {
    int num = p.cnum * p.rnum; 
    int bb, tt; 
    azc_config(num, bb, tt, "azccall_prep_unresnorm_crossmap"); 
    azc_kernel(azc_prep_unresnorm_crossmap,bb,tt)(num, p); 
    chk_err("azccall_prep_unresnorm_crossmap", bb, tt);         
  }  

  /*-------------------------------------------------------------*/  
  /*---      undo response normalization (cross map)        ---*/
  /*-------------------------------------------------------------*/
  /* v_j sum_k (-2 alpha beta f_k g_k)/(N_k d_k)  +  (f_j g_j) / v_j */
  /*------------------------------------------------------------*/
  __global__ void azc_unresnorm_crossmap(int num, const azcparam_unresnorm_crossmap p) {
    int ex;   
    for (ex = azc_thno; ex < num; ex += azc_thnum) {
      int index = ex; 
      int row = index % p.rnum; 
      int col = index / p.rnum; 
      if (col >= p.cnum) continue; 

      int rr0, rr1; 
      if (p.size < p.rnum) {
        int halfsize = p.size / 2; 
        rr0 = row - halfsize;
        rr1 = row + halfsize; 
      }
      else {
        rr0 = 0; 
        rr1 = p.rnum-1; 
      }
    
      int base = col*p.rnum; 
      const AzFloat *tmp = p.elm_tmp + base;  
      const AzFloat *bef = p.elm_bef + base;  
      const AzFloat *oneplussqavg = p.elm_oneplussqavg + base;       
      const AzFloat *grad = p.elm_grad + base;  
      AzFloat *out = p.elm_out + base; 

      AzFloat val = 0; 
      if (bef[row] != 0) {
        int rr; 
        for (rr = rr0; rr <= rr1; ++rr) { /* neighbor relationship is mutual */
          int myrr = (rr+p.rnum)%p.rnum; 
          val += tmp[myrr]; 
        }
      }
      out[row] = bef[row]*val + grad[row]*pow(oneplussqavg[row],-p.beta); 
    }
  }

  /*------------------------------------------------------------*/
  void azccall_unresnorm_crossmap(const azcparam_unresnorm_crossmap p) {
    int num = p.cnum * p.rnum; 
    int bb, tt; 
    azc_config(num, bb, tt, "azccall_unresnorm_crossmap");  
    azc_kernel(azc_unresnorm_crossmap,bb,tt)(num, p); 
    chk_err("azccall_unresnorm_crossmap", bb, tt);         
  } 
  
  /*------------------------------------------------------------*/
  __global__ void azc_unresnorm_crossmap_all(int num, const azcparam_unresnorm_crossmap p, const AzFloat *tmp_colSum) {
    int ex;   
    for (ex = azc_thno; ex < num; ex += azc_thnum) {
      int index = ex; 
      int row = index % p.rnum; 
      int col = index / p.rnum; 
      if (col >= p.cnum) continue; 

      int base = col*p.rnum; 
      const AzFloat *bef = p.elm_bef + base;  
      const AzFloat *oneplussqavg = p.elm_oneplussqavg + base;       
      const AzFloat *grad = p.elm_grad + base;  
      AzFloat *out = p.elm_out + base; 

      out[row] = bef[row]*tmp_colSum[col] + grad[row]*pow(oneplussqavg[row],-p.beta); 
    }
  }

  /*------------------------------------------------------------*/
  void azccall_unresnorm_crossmap_all(const azcparam_unresnorm_crossmap p, const AzFloat *tmp_colSum) {
    int num = p.cnum * p.rnum; 
    int bb, tt; 
    azc_config(num, bb, tt, "azccall_unresnorm_crossmap_all");  
    azc_kernel(azc_unresnorm_crossmap_all,bb,tt)(num, p, tmp_colSum); 
    chk_err("azccall_unresnorm_crossmap_all", bb, tt);         
  } 
  
  /*------------------------------------------------------------*/
  /*---            response normalization (local)            ---*/
  /*------------------------------------------------------------*/  
  __global__ void azc_resnorm_local(int num, const azcparam_resnorm_local p) {
    int ex;   
    for (ex = azc_thno; ex < num; ex += azc_thnum) {
      int index = ex;  
      int row = index % p.rnum; 
      index = index / p.rnum;    
      int dx = index / p.cnum; 
      int col = index % p.cnum; 
      if (dx >= p.data_num) continue;  

      int base = dx*p.cnum; 
    
      /*---  compute  1 + alpha avg_i v_i^2  ---*/
      const int *neigh = _column(col, p.neighbors, p.nummax); 
      AzFloat *oneplussqavg = _column(base+col, p.elm_oneplussqavg, p.rnum); 
      AzFloat *normalized = _column(base+col, p.elm_normalized, p.rnum); 
      int ix; 
      for (ix = 0; ix < p.nummax; ++ix) {
        if (neigh[ix] == p.stopper) break; 
        const AzFloat *neigh_val = _column(base+neigh[ix], p.elm, p.rnum); 
        oneplussqavg[row] += neigh_val[row]*neigh_val[row]; 
      }
      int sz = p.neigh_sz[col]; 
  
      sz = 1; /* to match with cuda-convnet */
  
      AzFloat coeff = 1; 
      if (sz != 0) {
        coeff = p.alpha / (AzFloat)sz; 
      }
      oneplussqavg[row] *= coeff; 
      oneplussqavg[row] += 1; 
      /*---  multiply (1 + alpha avg_i v_i^2)^{-beta}  ---*/
      normalized[row] *= pow(oneplussqavg[row], -p.beta); 
    }
  }

  /*------------------------------------------------------------*/
  void azccall_resnorm_local(const azcparam_resnorm_local p) {
    int num = p.data_num * p.cnum * p.rnum; 
    int bb, tt; 
    azc_config(num, bb, tt, "azccall_resnorm_local"); 
    azc_kernel(azc_resnorm_local,bb,tt)(num, p); 
    chk_err("azccall_resnorm_local", bb, tt);         
  } 

  /*------------------------------------------------------------*/  
  /*---   prep for undoing response normalization (local)    ---*/
  /*------------------------------------------------------------*/
  /* tmp <- g_k * (-2 alpha beta)/N_k * v_k(1 + alpha/N sum_i v_i^2)^{-beta-1} */
  /*     =  (-2 alpha beta f_k g_k)/(N_k d_k)                   */
  /*  d_k := 1 + alpha/N_k sum_i v_i^2                          */
  /*  f_k := v_k d_k^{-beta}                                    */
  /*------------------------------------------------------------*/
  __global__ void azc_prep_unresnorm_local(int num, const azcparam_prep_unresnorm_local p) { 
    int ex;   
    for (ex = azc_thno; ex < num; ex += azc_thnum) {
      int index = ex; 
      int row = index % p.rnum; 
      index = index / p.rnum;    
      int dx = index / p.cnum; 
      int col = index % p.cnum; 
      if (dx >= p.data_num) continue; 

      int base0 = dx*p.rnum*p.cnum; 
      int base = base0 + p.rnum*col; 
      const AzFloat *grad = p.elm_grad + base; 
      const AzFloat *aft = p.elm_aft + base; 
      const AzFloat *oneplussqavg = p.elm_oneplussqavg + base; 
      AzFloat *tmp = p.elm_tmp + base; 
      int sz = p.neigh_sz[col]; 

      sz = 1; /* to match with cuda-convnet */

      tmp[row] = (-2*p.alpha*p.beta * aft[row]*grad[row]) / (oneplussqavg[row] * (AzFloat)sz); 
    }
  }

  /*------------------------------------------------------------*/
  void azccall_prep_unresnorm_local(const azcparam_prep_unresnorm_local p) {
    int num = p.data_num * p.cnum * p.rnum; 
    int bb, tt; 
    azc_config(num, bb, tt, "azccall_prep_unresnorm_local"); 
    azc_kernel(azc_prep_unresnorm_local,bb,tt)(num, p);   
    chk_err("azccall_prep_unresnorm_local", bb, tt);         
  }    

  /*------------------------------------------------------------*/  
  /*---          undo response normalization (local)         ---*/  
  /* v_j sum_k (-2 alpha beta f_k g_k)/(N_k d_k)  +  (f_j g_j) / v_j */
  /*------------------------------------------------------------*/
  __global__ void azc_unresnorm_local(int num, const azcparam_unresnorm_local p) { 
    int ex;   
    for (ex = azc_thno; ex < num; ex += azc_thnum) {
      int index = ex; 
      int row = index % p.rnum; 
      index = index / p.rnum;    
      int dx = index / p.cnum; 
      int col = index % p.cnum; 
      if (dx >= p.data_num) continue; 

      int base0 = dx*p.rnum*p.cnum; 
      const AzFloat *tmp = p.elm_tmp + base0;  

      int base = base0 + col*p.rnum; 
      AzFloat *out = p.elm_out + base; 
      const AzFloat *bef = p.elm_bef + base;  
      const AzFloat *oneplussqavg = p.elm_oneplussqavg + base;       
      const AzFloat *grad = p.elm_grad + base;
    
      const int *whose_neigh = _column(col, p.whose_neighbor, p.nummax); 

      AzFloat val = 0; 
      if (bef[row] != 0) {
        int ix; 
        for (ix = 0; ix < p.nummax; ++ix) {    
          int kx = whose_neigh[ix]; 
          if (kx == p.stopper) break;         
          val += _entry(row, kx, tmp, p.rnum); 
        }       
      }
      out[row] = bef[row]*val + grad[row]*pow(oneplussqavg[row], -p.beta);  
    }
  }

  /*------------------------------------------------------------*/
  void azccall_unresnorm_local(const azcparam_unresnorm_local p) {
    int num = p.data_num * p.cnum * p.rnum; 
    int bb, tt; 
    azc_config(num, bb, tt, "azccall_unresnorm_local"); 
    azc_kernel(azc_unresnorm_local,bb,tt)(num, p); 
    chk_err("azccall_unresnorm_local", bb, tt);         
  } 

  /*------------------------------------------------*/
  /* loss: f(p)=log(1+exp(-zp)) for y in {0,1}      */
  /* derivative: f'(p)=-z/(exp(zp)+1)               */
  /* z=2y-1 for y in {0,1}; z=y for y in {-1,1}     */                          
  /*------------------------------------------------*/
  __global__ void azc_binlogi_deriv(bool is_01, 
             const AzFloat *p, const AzFloat *y, int num, 
             AzFloat *ld, AzFloat *loss) {
    for (int ex = azc_thno; ex < num; ex += azc_thnum) {    
      AzFloat pp = p[ex]; 
      AzFloat yy = (is_01) ? 2*y[ex]-1 : y[ex]; 
      AzFloat ee = exp(yy*pp); 
      ld[ex] = -yy/(ee+1); /* -z/(exp(zp)+1) */
      if (loss != NULL) {
        loss[ex] = log(1+1/ee);  /* log(1+exp(-zp)) */
      }
    }
  }
  /*------------------------------------------------------------*/
  void azccall_binlogi_deriv(bool is_01, 
             const AzFloat *p, const AzFloat *y, int num, 
             AzFloat *ld, AzFloat *loss) {
    int bb, tt; 
    azc_config(num, bb, tt, "azccall_binlogi_deriv"); 
    azc_kernel(azc_binlogi_deriv,bb,tt)(is_01, p, y, num, ld, loss); 
    chk_err("azccall_binlogi_deriv", bb, tt);         
  } 
  /*------------------------------------------------------------*/  
  __global__ void azc_binlogi_loss(bool is_01, 
             const AzFloat *p, const AzFloat *y, int num, 
             AzFloat *loss) {
    for (int ex = azc_thno; ex < num; ex += azc_thnum) {    
      AzFloat pp = p[ex]; 
      AzFloat yy = (is_01) ? 2*y[ex]-1 : y[ex]; 
      loss[ex] = log(1+exp(-yy*pp));  /* log(1+exp(-zp)) */
    }
  }  
  /*------------------------------------------------------------*/
  void azccall_binlogi_loss(bool is_01, 
             const AzFloat *p, const AzFloat *y, int num, 
             AzFloat *loss) {
    int bb, tt; 
    azc_config(num, bb, tt, "azccall_binlogi_loss"); 
    azc_kernel(azc_binlogi_loss,bb,tt)(is_01, p, y, num, loss); 
    chk_err("azccall_binlogi_loss", bb, tt);         
  } 
  
  /*------------------------------------------------------------*/  
  __global__ void azc_for_log_loss(const AzFloat *p, const int *y_row,
                                   int rnum, int cnum, AzFloat *out) {
    for (int col = azc_thno; col < cnum; col += azc_thnum) out[col] = _entry(y_row[col], col, p, rnum); 
  }  
  /*------------------------------------------------------------*/
  void azccall_for_log_loss(const AzFloat *p, const int *y_row, 
                            int rnum, int cnum, AzFloat *out) {                     
    int bb, tt; azc_config(cnum, bb, tt, "azccall_for_log_loss"); 
    azc_kernel(azc_for_log_loss,bb,tt)(p, y_row, rnum, cnum, out); 
    chk_err("azccall_for_log_loss", bb, tt);         
  } 
  /*------------------------------------------------------------*/  
  __global__ void azc_for_log_grad(AzFloat *p, const int *y_row, 
                                   int rnum, int cnum) {
    for (int col = azc_thno; col < cnum; col += azc_thnum) _entry(y_row[col], col, p, rnum) -= 1; 
  }  
  /*------------------------------------------------------------*/
  void azccall_for_log_grad(AzFloat *p, const int *y_row, 
                            int rnum, int cnum) {                     
    int bb, tt; azc_config(cnum, bb, tt, "azccall_for_log_grad"); 
    azc_kernel(azc_for_log_grad,bb,tt)(p, y_row, rnum, cnum); 
    chk_err("azccall_for_log_grad", bb, tt);         
  }   

  /*------------------------------------------------------------*/  
  /* normalize over columns so that the sum becomes one.  */
  __global__ void azc_sumone(AzFloat *inout, int rnum, const int *beg_end, int dnum, bool do_scale) {
    int num = rnum*dnum;             
    for (int ex = azc_thno; ex < num; ex += azc_thnum) {    
      int row = ex%rnum, dx = ex/rnum; 
      AzFloat sum = 0; 
      int bb = beg_end[dx*2], ee = beg_end[dx*2+1]; 
      int pos = bb*rnum+row; 
      for (int ix = bb; ix < ee; ++ix, pos+=rnum) sum += inout[pos]; 
      if (sum == 0) return; 
      if (do_scale) sum /= (AzFloat)(ee-bb); 
      pos = bb*rnum+row; 
      for (int ix = bb; ix < ee; ++ix, pos+=rnum) inout[pos]/= sum; 
    }
  }  
  /*------------------------------------------------------------*/
  void azccall_sumone(AzFloat *inout, int rnum, const int *beg_end, int dnum, bool do_scale) {
    int bb, tt; 
    azc_config(rnum*dnum, bb, tt, "azccall_sumone"); 
    azc_kernel(azc_sumone,bb,tt)(inout, rnum, beg_end, dnum, do_scale); 
    chk_err("azccall_sumone", bb, tt);         
  }

  /*------------------------------------------------------------*/  
  __global__ void azc_unsumone(AzFloat *grad, const AzFloat *inp, int rnum, const int *beg_end, int dnum, bool do_scale) {
    int num = rnum*dnum;             
    for (int ex = azc_thno; ex < num; ex += azc_thnum) {    
      int row = ex%rnum, dx = ex/rnum; 
      AzFloat sum = 0, zg = 0; 
      int bb = beg_end[dx*2], ee = beg_end[dx*2+1]; 
      int pos = bb*rnum+row; 
      for (int ix = bb; ix < ee; ++ix, pos+=rnum) {
        sum += inp[pos]; 
        zg += inp[pos]*grad[pos]; 
      }
      pos = bb*rnum+row;      
      AzFloat scale = (do_scale) ? ee-bb : 1; 
      if (sum != 0) { 
        AzFloat val0 = zg/sum/sum; 
        for (int ix = bb; ix < ee; ++ix, pos+=rnum) grad[pos] = scale*(grad[pos]/sum - val0); 
      }
    }
  }  
  /*------------------------------------------------------------*/
  void azccall_unsumone(AzFloat *grad, const AzFloat *inp, int rnum, const int *beg_end, int dnum, bool do_scale) {
    int bb, tt; 
    azc_config(rnum*dnum, bb, tt, "azccall_unsumone"); 
    azc_kernel(azc_unsumone,bb,tt)(grad, inp, rnum, beg_end, dnum, do_scale); 
    chk_err("azccall_unsumone", bb, tt);         
  }  

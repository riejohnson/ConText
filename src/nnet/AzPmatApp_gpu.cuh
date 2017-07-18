/* * * * *
 *  AzpPmatApp_gpu.cuh
 *  Copyright (C) 2014-2017 Rie Johnson
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
 
#ifndef _AZ_PMAT_APP_GPU_CUH_
#define _AZ_PMAT_APP_GPU_CUH_

#include "AzCuda_PmatApp.cuh"

/* 
 *  At the moment, this is unnecessary indirection, but it may be useful later when 
 *  the code is optimized also for CPU.  
 */
class _AzPmatApp {
public:
  inline static void _l1l2(const AzFloat *src, AzFloat *dst, int num, AzFloat del) {
    azccall_l1l2(src, dst, num, del); 
  }
  inline static void _add_l1l2deriv(const AzFloat *src, AzFloat *dst, int num, AzFloat del, AzFloat coeff) {
    azccall_add_l1l2deriv(src, dst, num, del, coeff); 
  }
  
  /*---  activation  ---*/
  inline static void _activate_leaky_rect(AzFloat *elm, int num, AzFloat aa, AzFloat *deriv_elm=NULL) {
    azccall_activate_leaky_rect(elm, num, aa, deriv_elm); 
  }  
  inline static void _activate_th(AzFloat *elm, int num, AzFloat th, AzFloat *deriv_elm=NULL) {
    azccall_activate_th(elm, num, th, deriv_elm); 
  }
  inline static void _activate_log(AzFloat *elm, int num, AzFloat *deriv_elm=NULL) {
    azccall_activate_log(elm, num, deriv_elm); 
  }
  inline static void _activate_tanh(AzFloat *elm, int num, AzFloat *deriv_elm=NULL) {
    azccall_activate_tanh(elm, num, deriv_elm); 
  }
  inline static void _activate_softplus(AzFloat *elm, int num, AzFloat *deriv_elm=NULL) {
    azccall_activate_softplus(elm, num, deriv_elm); 
  }
  inline static void _truncate(AzFloat *elm, int num, AzFloat border, AzFloat *deriv_elm=NULL) {
    azccall_truncate(elm, num, border, deriv_elm); 
  }                  
  
  /*---  filtering/unfiltering  ---*/
  inline static void _add_with_map(int data_num, 
                             const AzFloat *elm1, int width1, 
                             AzFloat *elm2, int width2, 
                             int row_num, 
                             const int *a2to1,  /* [col# in m2] col#_1 in m1, col#_2 in m1, ... */
                             int nummax, 
                             int stopper) {
    azcparam_add_with_map p(data_num, elm1, width1, elm2, width2, row_num, a2to1, nummax, stopper);
    azccall_add_with_map(p); 
  } 
  inline static void _add_with_map_var(const azcparam_var &p, int cnum2) {
    azccall_add_with_map_var(p, cnum2); 
  }  
                
  /*---  pooling  ---*/
  inline static void _pooling_avg(int data_num, 
                             const AzFloat *elm1, int width1, 
                             AzFloat *elm2, int width2, 
                             int row_num, 
                             const int *col1_ptr, 
                             int col1_nummax, 
                             int stopper) { 
    azcparam_pooling_avg p(data_num, elm1, width1, elm2, width2, row_num, col1_ptr, col1_nummax, stopper); 
    azccall_pooling_avg(p); 
  }                           
                             
  inline static void _unpooling_avg(int data_num, 
                             AzFloat *elm1, int width1, 
                             const AzFloat *elm2, int width2, 
                             int row_num, 
                             const int *col2_ptr, 
                             int col2_nummax, 
                             int stopper,
                             const int *col2_to_num) {
    azcparam_unpooling_avg p(data_num, elm1, width1, elm2, width2, row_num, col2_ptr, col2_nummax, stopper, col2_to_num);
    azccall_unpooling_avg(p); 
  }
  
  inline static void _pooling_l2(int data_num, 
                             const AzFloat *elm1, int width1, 
                             AzFloat *elm2, int width2, 
                             int row_num, 
                             const int *col1_ptr, 
                             int col1_nummax, 
                             int stopper) {
    azcparam_pooling_l2 p(data_num, elm1, width1, elm2, width2, row_num, col1_ptr, col1_nummax, stopper); 
    azccall_pooling_l2(p);     
  }
  
  inline static void _unpooling_l2(int data_num, 
                             AzFloat *elm1, int width1, 
                             const AzFloat *elm2, int width2, 
                             int row_num, 
                             const int *col2_ptr, 
                             int col2_nummax, 
                             int stopper,
                             const AzFloat *org_elm1, const AzFloat *org_elm2) {
    azcparam_unpooling_l2 p(data_num, elm1, width1, elm2, width2, row_num, col2_ptr, col2_nummax, stopper, org_elm1, org_elm2); 
    azccall_unpooling_l2(p); 
  }      
  
  inline static void _pooling_max(int data_num, 
                         const AzFloat *elm1, int width1, 
                         AzFloat *elm2, int width2, 
                         int row_num, 
                         const int *col1_ptr, 
                         int col1_nummax, 
                         int stopper, 
                         int *chosen_ptr) { /* may be NULL */
    azcparam_pooling_max p(data_num, elm1, width1, elm2, width2, row_num, col1_ptr, col1_nummax, stopper, chosen_ptr); 
    azccall_pooling_max(p);     
  }                       
  
  inline static void _unpooling_max(int data_num, AzFloat *elm1, int width1, 
                            const AzFloat *elm2, int width2, int row_num, 
                            const int *ptr_chosen,
                            const int *col1to2_ptr, int nummax, int stopper) {
    azcparam_unpooling_max p(data_num, elm1, width1, elm2, width2, row_num, ptr_chosen, col1to2_ptr, nummax, stopper); 
    azccall_unpooling_max(p);     
  }
  inline static void _unpooling_max2(int data_num, AzFloat *elm1, int width1, 
                            const AzFloat *elm2, int width2, int row_num, 
                            const int *ptr_chosen,
                            const int *col1to2_ptr, int nummax, int stopper) {
    azcparam_unpooling_max p(data_num, elm1, width1, elm2, width2, row_num, ptr_chosen, col1to2_ptr, nummax, stopper); 
    azccall_unpooling_max2(p);     
  }  

  /*---  pooling with variable-sized input and output  ---*/  
  inline static void _pooling_max_var(const azcparam_var &p, int cnum2, int *_chosen) { 
    azccall_pooling_max_var(p, cnum2, _chosen); 
  }
  inline static void _pooling_avg_var(const azcparam_var &p, int cnum2, int sz) { 
    azccall_pooling_avg_var(p, cnum2, sz); 
  }   
  inline static void _unpooling_max_var(const azcparam_var &p, int cnum2, const int *_chosen) { 
    azccall_unpooling_max_var(p, cnum2, _chosen); 
  }
  inline static void _unpooling_avg_var(const azcparam_var &p, int cnum2, int sz) { 
    azccall_unpooling_avg_var(p, cnum2, sz); 
  }
  
  inline static void _maxout(const azcparam_maxout &p) { azccall_maxout(p); }
  inline static void _unmaxout(const azcparam_unmaxout &p) { azccall_unmaxout(p); }  

  /*---  for local weights  ---*/
  inline static void _rearrange(int loc_num, 
                              const AzFloat *elm1, 
                              int d_num, 
                              AzFloat *elm2,
                              int rnum) {
    azcparam_rearrange p(loc_num, elm1, d_num, elm2, rnum); 
    azccall_rearrange(p); 
  }                              
  inline static void _undo_rearrange(int loc_num, 
                              AzFloat *elm1, 
                              int d_num, 
                              const AzFloat *elm2,
                              int rnum) {
    azcparam_undo_rearrange p(loc_num, elm1, d_num, elm2, rnum); 
    azccall_undo_rearrange(p); 
  }                              
                               
  /*---  response normalization  ---*/                               
  inline static void _resnorm_cross(const AzFloat *elm, 
                         int rnum, int cnum, 
                         int size, AzFloat alpha, AzFloat beta, AzFloat one, 
                         AzFloat *elm_normalized, 
                         AzFloat *elm_oneplussqavg) { /* must be initialized by zero */
    azcparam_resnorm_crossmap p(elm, rnum, cnum, size, alpha, beta, one, elm_normalized, elm_oneplussqavg); 
    azccall_resnorm_crossmap(p);     
  }
  inline static void _resnorm_cross_all(const AzFloat *elm, 
                         int rnum, int cnum, 
                         AzFloat alpha, AzFloat beta, AzFloat one, 
                         AzFloat *elm_normalized, 
                         AzFloat *elm_oneplussqavg, /* must be initialized by zero */
                         const AzFloat *col_sqsum) {
    azcparam_resnorm_crossmap p(elm, rnum, cnum, rnum, alpha, beta, one, elm_normalized, elm_oneplussqavg); 
    azccall_resnorm_crossmap_all(p, col_sqsum);     
  }  
  inline static void _prep_unresnorm_cross(
                         const AzFloat *elm_grad, 
                         const AzFloat *elm_aft, 
                         const AzFloat *elm_oneplussqavg, 
                         int rnum, int cnum, 
                         int size, AzFloat alpha, AzFloat beta, 
                         AzFloat *elm_tmp) {
    azcparam_prep_unresnorm_crossmap p(elm_grad, elm_aft, elm_oneplussqavg, rnum, cnum, size, alpha, beta, elm_tmp); 
    azccall_prep_unresnorm_crossmap(p); 
  }                         
  inline static void _unresnorm_cross(
                         const AzFloat *elm_tmp, const AzFloat *elm_grad, const AzFloat *elm_bef, const AzFloat *elm_oneplussqavg, 
                         int rnum, int cnum, int size, AzFloat beta, AzFloat *elm_out) {
    azcparam_unresnorm_crossmap p(elm_tmp, elm_grad, elm_bef, elm_oneplussqavg, rnum, cnum, size, beta, elm_out); 
    azccall_unresnorm_crossmap(p); 
  }
  inline static void _unresnorm_cross_all(
                         const AzFloat *elm_grad, const AzFloat *elm_bef, const AzFloat *elm_oneplussqavg, 
                         int rnum, int cnum, AzFloat beta, AzFloat *elm_out, const AzFloat *tmp_col_sum) {
    azcparam_unresnorm_crossmap p(NULL, elm_grad, elm_bef, elm_oneplussqavg, rnum, cnum, rnum, beta, elm_out); 
    azccall_unresnorm_crossmap_all(p, tmp_col_sum); 
  }
  
  inline static void _resnorm_local(int data_num, 
                             const AzFloat *elm,
                             int rnum, 
                             int cnum, 
                             const int *neighbors, 
                             int nummax, 
                             int stopper, 
                             const int *neigh_sz, 
                             AzFloat alpha, AzFloat beta, 
                             AzFloat *elm_normalized, 
                             AzFloat *elm_oneplussqavg) { /* must be initialized by zero */
    azcparam_resnorm_local p(data_num, elm, rnum, cnum, neighbors, nummax, stopper, neigh_sz, alpha, beta, elm_normalized, elm_oneplussqavg); 
    azccall_resnorm_local(p); 
  }
  inline static void _prep_unresnorm_local(int data_num, 
                         const AzFloat *elm_grad, 
                         const AzFloat *elm_aft, 
                         const AzFloat *elm_oneplussqavg, 
                         int rnum, int cnum, 
                         const int *neigh_sz, 
                         AzFloat alpha, AzFloat beta, 
                         AzFloat *elm_tmp) {
    azcparam_prep_unresnorm_local p(data_num, elm_grad, elm_aft, elm_oneplussqavg, rnum, cnum, neigh_sz, alpha, beta, elm_tmp); 
    azccall_prep_unresnorm_local(p); 
  }
  inline static void _unresnorm_local(int data_num,  
                         const AzFloat *elm_tmp,  /* (-2 alpha beta f_k g_k)/(N_k d_k) */
                         const AzFloat *elm_grad, /* g_j */
                         const AzFloat *elm_bef, /* v_j */
                         const AzFloat *elm_oneplussqavg, /* d_j */
                         int rnum, int cnum, 
                         /*---  whose neighbor am I?  ---*/
                         const int *whose_neighbor, 
                         int nummax, 
                         int stopper,
                         /*---  ---*/
                         AzFloat beta, 
                         AzFloat *elm_out) {
    azcparam_unresnorm_local p(data_num, elm_tmp, elm_grad, elm_bef, elm_oneplussqavg, rnum, cnum, whose_neighbor, nummax, stopper, beta, elm_out); 
    azccall_unresnorm_local(p);     
  } 
  
  /*---  loss function  ---*/
  static void _binlogi_deriv(bool is_01, const AzFloat *p, const AzFloat *y, int num, 
                             AzFloat *ld, AzFloat *loss) {
    azccall_binlogi_deriv(is_01, p, y, num, ld, loss); 
  }                            
  static void _binlogi_loss(bool is_01, const AzFloat *p, const AzFloat *y, int num, 
                            AzFloat *loss) {
    azccall_binlogi_loss(is_01, p, y, num, loss); 
  }

  static void _for_log_grad(AzFloat *p, const int *y_row, int rnum, int cnum) {
    azccall_for_log_grad(p, y_row, rnum, cnum); 
  }                            
  static void _for_log_loss(const AzFloat *p, const int *y_row, int rnum, int cnum, AzFloat *out) {
    azccall_for_log_loss(p, y_row, rnum, cnum, out); 
  }
  
  /*---  ---*/
  static void _sumone(AzFloat *inout, int rnum, const int *beg_end, int dnum, bool do_scale) {
    azccall_sumone(inout, rnum, beg_end, dnum, do_scale); 
  }
  static void _unsumone(AzFloat *grad, const AzFloat *inp, int rnum, const int *beg_end, int dnum, bool do_scale) {
    azccall_unsumone(grad, inp, rnum, beg_end, dnum, do_scale); 
  }
};
#endif 
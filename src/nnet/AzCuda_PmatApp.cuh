/* * * * *
 *  AzCuda_PmatApp.cuh
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

#ifndef _AZ_CUDA_PMATAPP_CUH_
#define _AZ_CUDA_PMATAPP_CUH_

#include "AzUtil.hpp"
#include "AzP.h"

  /*****  PmatApp  *****/                                                      

  /*---  l1l2  ---*/  
  void azccall_l1l2(const AzFloat *src, AzFloat *dst, int num, AzFloat del); 
  void azccall_add_l1l2deriv(const AzFloat *src, AzFloat *dst, int num, AzFloat del, AzFloat coeff); 

  void azccall_activate_leaky_rect(AzFloat *elm, int num, AzFloat aa, AzFloat *deriv_elm=NULL); 
  void azccall_activate_th(AzFloat *elm, int num, AzFloat th, AzFloat *deriv_elm=NULL);        
  void azccall_activate_log(AzFloat *elm, int num, AzFloat *deriv_elm=NULL); 
  void azccall_activate_tanh(AzFloat *elm, int num, AzFloat *deriv_elm=NULL); 
  void azccall_activate_softplus(AzFloat *elm, int num, AzFloat *deriv_elm=NULL); 
  void azccall_truncate(AzFloat *elm, int num, AzFloat border, AzFloat *deriv_elm=NULL); 

  /*---  filtering  ---*/                            
  class azcparam_add_with_map {
  public: 
    int data_num, width1, width2, row_num, nummax, stopper; 
    const AzFloat *elm1; 
    AzFloat *elm2; 
    const int *a2to1; /* [col# in m2] col#_1 in m1, col#_2 in m1, ... */
    azcparam_add_with_map(int _data_num, const AzFloat *_elm1, int _width1, AzFloat *_elm2, int _width2, int _row_num, 
                          const int *_a2to1, int _nummax, int _stopper) {
      data_num=_data_num; elm1=_elm1; width1=_width1; elm2=_elm2; width2=_width2; row_num=_row_num;     
      a2to1=_a2to1; nummax=_nummax; stopper=_stopper; 
    }
  }; 
  void azccall_add_with_map(const azcparam_add_with_map p); 
                  
  /*---  avg pooling  ---*/
  class azcparam_pooling_avg {  
  public: 
    int data_num, width1, width2, row_num, col1_nummax, stopper; 
    const AzFloat *elm1; 
    AzFloat *elm2; 
    const int *col1_ptr; 
    azcparam_pooling_avg(int _data_num, const AzFloat *_elm1, int _width1, AzFloat *_elm2, int _width2, 
                         int _row_num, const int *_col1_ptr, int _col1_nummax, int _stopper) {
      data_num=_data_num; elm1=_elm1; width1=_width1; elm2=_elm2; width2=_width2; 
      row_num=_row_num; col1_ptr=_col1_ptr; col1_nummax=_col1_nummax; stopper=_stopper; 
    }
  };   
  void azccall_pooling_avg(const azcparam_pooling_avg p);                     

  /*---  avg unpooling  ---*/
  class azcparam_unpooling_avg {
  public: 
    int data_num, width1, width2, row_num, col2_nummax, stopper; 
    AzFloat *elm1; 
    const AzFloat *elm2; 
    const int *col2_ptr, *col2_to_num; 
    azcparam_unpooling_avg(int _data_num, AzFloat *_elm1, int _width1, const AzFloat *_elm2, int _width2, int _row_num, 
                           const int *_col2_ptr, int _col2_nummax, int _stopper, const int *_col2_to_num) {
      data_num=_data_num; elm1=_elm1; width1=_width1; elm2=_elm2; width2=_width2; row_num=_row_num; 
      col2_ptr=_col2_ptr; col2_nummax=_col2_nummax; stopper=_stopper; col2_to_num=_col2_to_num;       
    }
  }; 
  void azccall_unpooling_avg(const azcparam_unpooling_avg p); 

  /*---  l2 pooling  ---*/
  class azcparam_pooling_l2 {
  public: 
    int data_num, width1, width2, row_num, col1_nummax, stopper; 
    const AzFloat *elm1; 
    AzFloat *elm2; 
    const int *col1_ptr; 
    azcparam_pooling_l2(int _data_num, const AzFloat *_elm1, int _width1, AzFloat *_elm2, int _width2, int _row_num, 
                        const int *_col1_ptr, int _col1_nummax, int _stopper) {
      data_num=_data_num; elm1=_elm1; width1=_width1; elm2=_elm2; width2=_width2; row_num=_row_num; 
      col1_ptr=_col1_ptr; col1_nummax=_col1_nummax; stopper=_stopper; 
    }
  }; 
  void azccall_pooling_l2(const azcparam_pooling_l2 p);   
  
  /*---  l2 unpooling  ---*/
  class azcparam_unpooling_l2 {
  public:
    int data_num, width1, width2, row_num, col2_nummax, stopper; 
    AzFloat *elm1; 
    const AzFloat *elm2, *org_elm1, *org_elm2; 
    const int *col2_ptr; 
    azcparam_unpooling_l2(int _data_num, AzFloat *_elm1, int _width1, const AzFloat *_elm2, int _width2, int _row_num, 
                          const int *_col2_ptr, int _col2_nummax, int _stopper, 
                          const AzFloat *_org_elm1, /* input */ 
                          const AzFloat *_org_elm2) /* pooled */ {
      data_num=_data_num; elm1=_elm1; width1=_width1; elm2=_elm2; width2=_width2; row_num=_row_num; 
      col2_ptr=_col2_ptr; col2_nummax=_col2_nummax; stopper=_stopper; org_elm1=_org_elm1; org_elm2=_org_elm2; 
    }
  };  
  void azccall_unpooling_l2(const azcparam_unpooling_l2 p);                         
                             
  /*---  max pooling  ---*/
  class azcparam_pooling_max {
  public:
    int data_num, width1, width2, row_num, col1_nummax, stopper; 
    const AzFloat *elm1; 
    AzFloat *elm2; 
    const int *col1_ptr; 
    int *chosen_ptr; 
    azcparam_pooling_max(int _data_num, const AzFloat *_elm1, int _width1, AzFloat *_elm2, int _width2, int _row_num, 
                         const int *_col1_ptr, int _col1_nummax, int _stopper, int *_chosen_ptr) {
      data_num=_data_num; elm1=_elm1; width1=_width1; elm2=_elm2; width2=_width2; row_num=_row_num; 
      col1_ptr=_col1_ptr; col1_nummax=_col1_nummax; stopper=_stopper; chosen_ptr=_chosen_ptr; 
    }
  }; 
  void azccall_pooling_max(const azcparam_pooling_max p);   
  
  /*---  max unpooling  ---*/
  class azcparam_unpooling_max {
  public:
    int data_num, width1, width2, row_num, nummax, stopper; 
    AzFloat *elm1; 
    const AzFloat *elm2; 
    const int *ptr_chosen; 
    const int *col1to2_ptr; 
    azcparam_unpooling_max(int _data_num, AzFloat *_elm1, int _width1, const AzFloat *_elm2, int _width2, int _row_num, const int *_ptr_chosen,
                            const int *_col1to2_ptr, int _nummax, int _stopper) {
      data_num=_data_num; elm1=_elm1; width1=_width1; elm2=_elm2; width2=_width2; row_num=_row_num; ptr_chosen=_ptr_chosen; 
      col1to2_ptr=_col1to2_ptr; nummax=_nummax; stopper=_stopper; 
    }
  };
  void azccall_unpooling_max(const azcparam_unpooling_max p); 
  void azccall_unpooling_max2(const azcparam_unpooling_max p);   

  /****  Pooling with variazble sized input/output  ---*/
  class azcparam_var { /* 1: input, 2: output */
  public:
    int row_num, nummax, stopper; 
    const AzFloat *elm1; AzFloat *elm2; 
    const int *col2to1_ptr, *pos1, *pos2, *c2dx;
    int c2dx_denomi; 
    azcparam_var(const AzFloat *_elm1, AzFloat *_elm2, int _row_num, 
                 const int *_col2to1_ptr, int _nummax, int _stopper,
                 const int *_pos1, const int *_pos2, const int *_c2dx, 
                 int _c2dx_denomi=1) {
      elm1=_elm1; elm2=_elm2; row_num=_row_num; 
      col2to1_ptr=_col2to1_ptr; nummax=_nummax; stopper=_stopper;
      pos1=_pos1; pos2=_pos2; c2dx=_c2dx; c2dx_denomi = _c2dx_denomi; 
    }
  };  
  void azccall_add_with_map_var(const azcparam_var p, int cnum2); 

  class azcparam_pooling_var { /* for fprop */
  public:
    int row_num, nummax, stopper; 
    const AzFloat *elm1; AzFloat *elm2; 
    const int *col2to1_ptr, *pos1, *pos2, *c2dx; int *chosen; 
    azcparam_pooling_var(const AzFloat *_elm1, AzFloat *_elm2, int _row_num, 
                         const int *_col2to1_ptr, int _nummax, int _stopper, int *_chosen, 
                         const int *_pos1, const int *_pos2, const int *_c2dx) {
      elm1=_elm1; elm2=_elm2; row_num=_row_num; 
      col2to1_ptr=_col2to1_ptr; nummax=_nummax; stopper=_stopper; chosen=_chosen; 
      pos1=_pos1; pos2=_pos2; c2dx=_c2dx; 
    }
  };                 
  void azccall_pooling_max_var(const azcparam_var p, int cnum2, int *_chosen); 
  void azccall_pooling_avg_var(const azcparam_var p, int cnum2, int sz); 
  void azccall_unpooling_max_var(const azcparam_var p, int cnum2, const int *_chosen); 
  void azccall_unpooling_avg_var(const azcparam_var p, int cnum2, int sz); 
  
  class azcparam_unpooling_var { /* for bprop */
  public:
    int row_num, nummax, stopper; 
    AzFloat *elm1; const AzFloat *elm2; 
    const int *col1to2_ptr, *pos1, *pos2, *c1dx, *chosen; 
    azcparam_unpooling_var(AzFloat *_elm1, const AzFloat *_elm2, int _row_num, 
                         const int *_col1to2_ptr, int _nummax, int _stopper, const int *_chosen, 
                         const int *_pos1, const int *_pos2, const int *_c1dx) {
      elm1=_elm1; elm2=_elm2; row_num=_row_num; 
      col1to2_ptr=_col1to2_ptr; nummax=_nummax; stopper=_stopper; chosen=_chosen; 
      pos1=_pos1; pos2=_pos2; c1dx=_c1dx; 
    }
  };                
  
  /*---  ---*/
  class azcparam_maxout {
  public:
    const AzFloat *elm1; AzFloat *elm2; int *chosen; 
    int rnum1, rnum2, cnum, piece, stride;
    azcparam_maxout(const AzFloat *e1, AzFloat *e2, int *ch, int rn1, int rn2, int cn, int pi, int st) {
      elm1=e1; elm2=e2; chosen=ch; rnum1=rn1; rnum2=rn2; cnum=cn; piece=pi; stride=st; 
    }
  }; 
  void azccall_maxout(const azcparam_maxout p); /* not tested */
  
  class azcparam_unmaxout {
  public:
    AzFloat *elm1; const AzFloat *elm2; const int *chosen; 
    int rnum1, rnum2, cnum, piece, stride;
    azcparam_unmaxout(AzFloat *e1, const AzFloat *e2, const int *ch, int rn1, int rn2, int cn, int pi, int st) {
      elm1=e1; elm2=e2; chosen=ch; rnum1=rn1; rnum2=rn2; cnum=cn; piece=pi; stride=st; 
    }    
  }; 
  void azccall_unmaxout(const azcparam_unmaxout p); /* not tested */
  
  /*---  for local weights  ---*/
  class azcparam_rearrange {
  public:
    int loc_num, d_num, rnum; 
    const AzFloat *elm1; 
    AzFloat *elm2; 
    azcparam_rearrange(int _loc_num, const AzFloat *_elm1, int _d_num, AzFloat *_elm2, int _rnum) {
      loc_num=_loc_num; elm1=_elm1; d_num=_d_num; elm2=_elm2; rnum=_rnum; 
    }
  }; 
  void azccall_rearrange(const azcparam_rearrange p); 

  class azcparam_undo_rearrange {
  public: 
    int loc_num, d_num, rnum; 
    AzFloat *elm1; 
    const AzFloat *elm2; 
    azcparam_undo_rearrange(int _loc_num, AzFloat *_elm1, int _d_num, const AzFloat *_elm2, int _rnum) {
      loc_num=_loc_num; elm1=_elm1; d_num=_d_num; elm2=_elm2; rnum=_rnum; 
    }
  }; 
  void azccall_undo_rearrange(const azcparam_undo_rearrange p); 

  /*---  response normalization (cross map)  ---*/
  class azcparam_resnorm_crossmap {
  public:
    const AzFloat *elm; 
    int rnum, cnum, size;
    AzFloat alpha, beta, one; 
    AzFloat *elm_normalized, /* inout */ *elm_oneplussqavg /* must be initialized by zero */; 
    azcparam_resnorm_crossmap(const AzFloat *_elm, int _rnum, int _cnum, int _size, AzFloat _alpha, AzFloat _beta, AzFloat _one, 
                                 AzFloat *_elm_normalized, /* inout */ AzFloat *_elm_oneplussqavg) { /* must be initialized by zero */ 
      elm=_elm; rnum=_rnum; cnum=_cnum; size=_size; alpha=_alpha; beta=_beta; elm_normalized=_elm_normalized; elm_oneplussqavg=_elm_oneplussqavg; one=_one; 
    }                                 
  }; 
  void azccall_resnorm_crossmap(const azcparam_resnorm_crossmap p); 
  void azccall_resnorm_crossmap_all(const azcparam_resnorm_crossmap p, const AzFloat *col_sqsum); 
  
  /*---  prep for undoing response normalization (cross map)  ---*/
  class azcparam_prep_unresnorm_crossmap {  
  public: 
    const AzFloat *elm_grad, *elm_aft, *elm_oneplussqavg; 
    int rnum, cnum, size; 
    AzFloat alpha, beta, *elm_tmp; 
    azcparam_prep_unresnorm_crossmap(const AzFloat *_elm_grad, const AzFloat *_elm_aft, const AzFloat *_elm_oneplussqavg, 
                         int _rnum, int _cnum, int _size, AzFloat _alpha, AzFloat _beta, AzFloat *_elm_tmp) {
      elm_grad=_elm_grad; elm_aft=_elm_aft; elm_oneplussqavg=_elm_oneplussqavg; 
      rnum=_rnum; cnum=_cnum; size=_size; alpha=_alpha; beta=_beta; elm_tmp=_elm_tmp;                     
    } 
  };   
  void azccall_prep_unresnorm_crossmap(const azcparam_prep_unresnorm_crossmap p); 

  /*---  undo response normalization (cross map)  ---*/
  class azcparam_unresnorm_crossmap {
  public: 
    const AzFloat *elm_tmp, *elm_grad, *elm_bef, *elm_oneplussqavg; 
    int rnum, cnum, size; 
    AzFloat beta, *elm_out; 
    azcparam_unresnorm_crossmap(const AzFloat *_elm_tmp, const AzFloat *_elm_grad, const AzFloat *_elm_bef, const AzFloat *_elm_oneplussqavg, 
                                   int _rnum, int _cnum, int _size, AzFloat _beta, AzFloat *_elm_out) {
      elm_tmp=_elm_tmp; elm_grad=_elm_grad; elm_bef=_elm_bef; elm_oneplussqavg=_elm_oneplussqavg; rnum=_rnum; cnum=_cnum; size=_size; beta=_beta; elm_out=_elm_out; 
    }
  }; 
  void azccall_unresnorm_crossmap(const azcparam_unresnorm_crossmap p); 
  void azccall_unresnorm_crossmap_all(const azcparam_unresnorm_crossmap p, const AzFloat *tmp_col_sum); 
  
  /*---  response normalization (local)  ---*/
  class azcparam_resnorm_local {
  public: 
    int data_num, rnum, cnum, nummax, stopper; 
    const AzFloat *elm; 
    const int *neighbors, *neigh_sz; 
    AzFloat alpha, beta; 
    AzFloat *elm_normalized /* inout */, *elm_oneplussqavg /* must be initialized by zero */; 
    azcparam_resnorm_local(int _data_num, const AzFloat *_elm, int _rnum, int _cnum, const int *_neighbors, int _nummax, int _stopper, 
                           const int *_neigh_sz, AzFloat _alpha, AzFloat _beta, 
                           AzFloat *_elm_normalized, /* inout */ AzFloat *_elm_oneplussqavg) /* must be initialized by zero */ {
      data_num=_data_num; elm=_elm; rnum=_rnum; cnum=_cnum; neighbors=_neighbors; nummax=_nummax; stopper=_stopper; 
      neigh_sz=_neigh_sz; alpha=_alpha; beta=_beta; elm_normalized=_elm_normalized; elm_oneplussqavg=_elm_oneplussqavg; 
    }
  }; 
  void azccall_resnorm_local(const azcparam_resnorm_local p); 
  
  /*---  prep for undoing response normalization (local)  ---*/
  class azcparam_prep_unresnorm_local {
  public: 
    int data_num, rnum, cnum; 
    const AzFloat *elm_grad, *elm_aft, *elm_oneplussqavg; 
    const int *neigh_sz; 
    AzFloat *elm_tmp; 
    AzFloat alpha, beta; 
    azcparam_prep_unresnorm_local(int _data_num, const AzFloat *_elm_grad, const AzFloat *_elm_aft, const AzFloat *_elm_oneplussqavg, 
                                  int _rnum, int _cnum, const int *_neigh_sz, AzFloat _alpha, AzFloat _beta, AzFloat *_elm_tmp) {
      data_num=_data_num; elm_grad=_elm_grad; elm_aft=_elm_aft; elm_oneplussqavg=_elm_oneplussqavg; 
      rnum=_rnum; cnum=_cnum; neigh_sz=_neigh_sz; alpha=_alpha; beta=_beta; elm_tmp=_elm_tmp; 
    }  
  };     
  void azccall_prep_unresnorm_local(const azcparam_prep_unresnorm_local p);   
    
  /*---  undo response normalization (local)  ---*/
  class azcparam_unresnorm_local {
  public:
    int data_num, rnum, cnum, nummax, stopper; 
    const AzFloat *elm_tmp, *elm_grad, *elm_bef, *elm_oneplussqavg; 
    const int *whose_neighbor; 
    AzFloat beta; 
    AzFloat *elm_out; 
    azcparam_unresnorm_local(int _data_num, const AzFloat *_elm_tmp,  /* (-2 alpha beta f_k g_k)/(N_k d_k) */
                             const AzFloat *_elm_grad, /* g_j */ const AzFloat *_elm_bef, /* v_j */ const AzFloat *_elm_oneplussqavg, /* f_j */
                             int _rnum, int _cnum, const int *_whose_neighbor, int _nummax, int _stopper,
                             AzFloat _beta, AzFloat *_elm_out) {
      data_num=_data_num; elm_tmp=_elm_tmp; elm_grad=_elm_grad; elm_bef=_elm_bef; elm_oneplussqavg=_elm_oneplussqavg; 
      rnum=_rnum; cnum=_cnum; whose_neighbor=_whose_neighbor; nummax=_nummax; stopper=_stopper; beta=_beta; elm_out=_elm_out; 
    }
  };
  void azccall_unresnorm_local(const azcparam_unresnorm_local p);          
  void azccall_binlogi_deriv(bool is_01, const AzFloat *p, const AzFloat *y, int num, 
                             AzFloat *ld, AzFloat *loss); 
  void azccall_binlogi_loss(bool is_01, const AzFloat *p, const AzFloat *y, int num, 
                            AzFloat *loss);
  void azccall_for_log_loss(const AzFloat *p, const int *y_row, int rnum, int cnum, AzFloat *out); 
  void azccall_for_log_grad(AzFloat *p, const int *y_row, int rnum, int cnum);                 
                            
  void azccall_sumone(AzFloat *inout, int rnum, const int *beg_end, int dnum, bool do_scale); 
  void azccall_unsumone(AzFloat *grad, const AzFloat *inp, int rnum, const int *beg_end, int dnum, bool do_scale); 
#endif 

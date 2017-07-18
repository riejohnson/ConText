/* * * * *
 *  AzpPmatApp.hpp
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

#ifndef _AZ_PMAT_APP_HPP_
#define _AZ_PMAT_APP_HPP_

#include "AzUtil.hpp"
#include "AzPmat.hpp"
#include "AzPmatApp_gpu.cuh"

class AzPmatApp {
protected:
  _AzPmatApp a; 
  
public:
  inline double l1l2sum(const AzPmat *m_src, AzFloat del) const {
    AzPmat m_dst; 
    l1l2(m_src, &m_dst, del); 
    return m_dst.sum(); 
  }
  void l1l2(const AzPmat *m_src, AzPmat *m_dst, AzFloat del) const; 
  void l1l2deriv(const AzPmat *m_src, AzPmat *m_dst, AzFloat del, AzFloat coeff=1) const; 
  void add_l1l2deriv(const AzPmat *m_src, AzPmat *m_dst, AzFloat del, AzFloat coeff=1) const; 
  
  /*---  activation  ---*/
  void activate_leaky_rect(AzPmat *mm, double aa, AzPmat *m_deriv) const;   
  void activate_th(AzPmat *mm, double th, AzPmat *m_deriv) const; 
  void activate_log(AzPmat *mm, AzPmat *m_deriv) const; /* sigmoid */
  void activate_tanh(AzPmat *mm, AzPmat *m_deriv) const; 
  void activate_softplus(AzPmat *mm, AzPmat *m_deriv) const; 

  void truncate(AzPmat *mm, double border, AzPmat *m_deriv) const;  

  /*---  filtering  ---*/   
  void add_with_map(int data_num, const AzPmat *m1, AzPmat *m2, int row_num, 
                    const AzPintArr2 *pia2_2to1) const;            
       
  /*---  pooling  ---*/
  void pooling_avg(const AzPmat *m1, int width1, 
                   AzPmat *m2, 
                   const AzPintArr2 *aIa_col1) const; 
  void unpooling_avg(AzPmat *m1, 
                     const AzPmat *m2, int width2, 
                     const AzPintArr2 *pia2_col2,
                     const AzPintArr *pia_col2_to_num) const; 

  void pooling_l2(const AzPmat *m1, int width1, 
                  AzPmat *m2, 
                  const AzPintArr2 *aIa_col1) const; 
  void unpooling_l2(AzPmat *m1, 
                    const AzPmat *m2, int width2, 
                    const AzPintArr2 *pia2_col2,
                    const AzPmat *org_m1, const AzPmat *org_m2) const; 
                             
  void pooling_max(const AzPmat *m1, int width1, 
                   AzPmat *m2, 
                   const AzPintArr2 *aIa_col1, 
                   AzPintArr *m_chosen) const;        

  void unpooling_max(AzPmat *m1,
                     const AzPmat *m2, 
                     const AzPintArr *m_chosen,
                     const AzPintArr2 *pia2_col1to2) const;
  void unpooling_max2(AzPmat *m1,
                     const AzPmat *m2, 
                     const AzPintArr *m_chosen,
                     const AzPintArr2 *pia2_col1to2) const;                     

  /*---  for speed-up with variable-sized input/output  ---*/
  void add_with_map_var(       const AzIntArr &ia_pos1, const AzIntArr &ia_pos2, /* beg0, beg1, ..., end */
       int denomi, const AzPmat &m1, AzPmat &m2, int row_num, 
       const AzPintArr2 &pia2_2to1) /* template */ const;   
  void pooling_max_var(const AzPmat &m1, AzPmat &m2, 
                       const AzPintArr2 &pia2_2to1, /* template for one with the larget size */
                       const AzIntArr &ia_pos1, const AzIntArr &ia_pos2,                          
                       AzPintArr *pia_chosen) /* may be NULL */ const; 
  void unpooling_max_var(const AzPmat &m1, AzPmat &m2, 
                         const AzPintArr2 &pia2_2to1, /* template for one with the larget size */
                         const AzIntArr &ia_pos1, const AzIntArr &ia_pos2,                          
                         const AzPintArr &pia_chosen) const; 
  void pooling_avg_var(const AzPmat &m1, AzPmat &m2, 
                       const AzPintArr2 &pia2_2to1, /* template for one with the larget size */
                       const AzIntArr &ia_pos1, const AzIntArr &ia_pos2, int sz) const; 
  void unpooling_avg_var(const AzPmat &m1, AzPmat &m2, 
                         const AzPintArr2 &pia2_2to1, /* template for one with the larget size */
                         const AzIntArr &ia_pos1, const AzIntArr &ia_pos2, int sz) const; 
                         
  /*---  for local weights  ---*/
  void rearrange(int loc_num, 
                 const AzPmat *m1, 
                 int d_num, 
                 AzPmat *m2) const; 
  void undo_rearrange(int loc_num, 
                      AzPmat *m1, 
                      int d_num, 
                      const AzPmat *m2) const; 
                     
  /*---  response normalization  ---*/                         
  void resnorm_cross(const AzPmat *m_input,
                         int size, double alpha, double beta, double one, 
                         AzPmat *m_inout, 
                         AzPmat *m_oneplussqavg, 
                         bool do_force_old) const; 
  void unresnorm_cross(const AzPmat *m_grad, 
                             const AzPmat *m_bef, 
                             const AzPmat *m_aft, 
                             const AzPmat *m_oneplussqavg, 
                             AzPmat *m_tmp, 
                             AzPmat *m_out, 
                             int size, double alpha, double beta,
                             bool do_force_old) const; 
  void resnorm_local(const AzPmat *m_input, 
                     const AzPintArr2 *pia2_neighbors, 
                     const AzPintArr *pia_neigh_sz, 
                     double alpha, double beta, 
                     AzPmat *m_inout, 
                     AzPmat *m_oneplussqavg) const; 
  void unresnorm_local(const AzPmat *m_grad, 
                       const AzPmat *m_bef, 
                       const AzPmat *m_aft, 
                       const AzPmat *m_oneplussqavg, 
                       AzPmat *m_tmp, 
                       AzPmat *m_out, 
                       const AzPintArr2 *pia2_whose_neighbor, 
                       const AzPintArr *pia_neigh_sz,                                 
                       double alpha, double beta) const;
                       
  /*---  loss function  ---*/                       
  void binlogi_deriv(bool is_01, const AzPmat *m_p, const AzPmat *m_y, 
                     AzPmat *m_ld, AzPmat *m_loss) const;    
  void binlogi_loss(bool is_01, const AzPmat *m_p, const AzPmat *m_y, 
                    AzPmat *m_loss) const;                     
  void for_log_grad(AzPmat &m_p, const AzPintArr &pia_y_row) const;  
  void for_log_loss(const AzPmat &m_p, const AzPintArr &pia_y_row, AzPmat &m_out) const; 
                    
  /*---  ---*/                   
  void sumone(AzPmatVar *mv, bool do_scale) const; 
  void unsumone(AzPmatVar *mv_grad, const AzPmat *m_inp, bool do_scale) const; 
  
protected: 
  /*---  column index to data index  ---*/
  void gen_col2dx(const AzIntArr &ia_pos, AzPintArr &pia_c2d, const char *eyec, int denomi=1) const {  
    AzIntArr ia_c2d; gen_col2dx(ia_pos, ia_c2d, eyec, denomi); 
    pia_c2d.reset(&ia_c2d); 
  }
  void gen_col2dx(const AzIntArr &ia_pos, AzIntArr &ia_c2d, const char *eyec, int denomi=1) const { 
    ia_c2d.reset(); 
    for (int dx = 0; dx < ia_pos.size()-1; ++dx) {
      int sz = ia_pos[dx+1] - ia_pos[dx]; 
      AzX::throw_if(sz%denomi != 0, eyec, "a wrong denomi"); 
      for (int ix = 0; ix < sz/denomi; ++ix) ia_c2d.put(dx); 
    } 
  }  
  int prep_for_var(const char *eyec, const AzPmat &m1, AzPmat &m2, 
                   const AzIntArr &ia_pos1, const AzIntArr &ia_pos2,
                   int &maxsz2) const; 
}; 
#endif 

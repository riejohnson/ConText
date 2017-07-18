/* * * * *
 *  AzPmatApp.cpp
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

#include "AzPmat.hpp"
#include "AzPmatApp.hpp"

/*---  L1L2: sqrt(x^2+d^2)-d  ---*/
void AzPmatApp::l1l2(const AzPmat *m_src, AzPmat *m_dst, AzFloat del) const {  
  m_dst->reform_tmpl(m_src); 
  a._l1l2(m_src->_dptr(), m_dst->_dptr_u(), m_src->size(), del); 
}

/*---  L1L2-deriv: x/sqrt(x^2+d^2)*coeff  ---*/
void AzPmatApp::l1l2deriv(const AzPmat *m_src, AzPmat *m_dst, AzFloat del, AzFloat coeff) const {
  m_dst->reform_tmpl(m_src); 
  a._add_l1l2deriv(m_src->_dptr(), m_dst->_dptr_u(), m_src->size(), del, coeff); 
}
void AzPmatApp::add_l1l2deriv(const AzPmat *m_src, AzPmat *m_dst, AzFloat del, AzFloat coeff) const {
  m_dst->shape_chk_tmpl(m_src, "AzPmatApp::add_l1l2deriv", "m_dst"); 
  a._add_l1l2deriv(m_src->_dptr(), m_dst->_dptr_u(), m_src->size(), del, coeff); 
}

/*------------------------------------------------*/
void AzPmatApp::activate_leaky_rect(AzPmat *mm, double aa, AzPmat *m_deriv) const {                            
  if (m_deriv == NULL) a._activate_leaky_rect(mm->_dptr_u(), mm->size(), (AzFloat)aa);  /* corrected on 4/27/2017 */
  else {
    m_deriv->shape_chk_tmpl(mm, "AzPmatApp::activate_leaky_rect", "m_deriv"); 
    a._activate_leaky_rect(mm->_dptr_u(), mm->size(), (AzFloat)aa, m_deriv->_dptr_u()); 
  }
}

/*------------------------------------------------*/
void AzPmatApp::activate_th(AzPmat *mm, double th, AzPmat *m_deriv) const {                            
  if (m_deriv == NULL) a._activate_th(mm->_dptr_u(), mm->size(), (AzFloat)th);   
  else {
    m_deriv->shape_chk_tmpl(mm, "AzPmatApp::activate_th", "m_deriv"); 
    a._activate_th(mm->_dptr_u(), mm->size(), (AzFloat)th, m_deriv->_dptr_u());     
  }
}

/*------------------------------------------------*/
void AzPmatApp::activate_log(AzPmat *mm, AzPmat *m_deriv) const {
  if (m_deriv == NULL) a._activate_log(mm->_dptr_u(), mm->size());
  else {
    m_deriv->shape_chk_tmpl(mm, "AzPmatApp::activate_log", "m_deriv"); 
    a._activate_log(mm->_dptr_u(), mm->size(), m_deriv->_dptr_u()); 
  }
} 

/*------------------------------------------------*/
void AzPmatApp::activate_tanh(AzPmat *mm, AzPmat *m_deriv) const {
  if (m_deriv == NULL) a._activate_tanh(mm->_dptr_u(), mm->size()); 
  else {
    m_deriv->shape_chk_tmpl(mm, "AzPmatApp::activate_tanh", "m_deriv"); 
    a._activate_tanh(mm->_dptr_u(), mm->size(), m_deriv->_dptr_u()); 
  }
} 

/*------------------------------------------------*/
void AzPmatApp::activate_softplus(AzPmat *mm, AzPmat *m_deriv) const {
  if (m_deriv == NULL) a._activate_softplus(mm->_dptr_u(), mm->size()); 
  else {
    m_deriv->shape_chk_tmpl(mm, "AzPmatApp::activate_softplus", "m_deriv"); 
    a._activate_softplus(mm->_dptr_u(), mm->size(), m_deriv->_dptr_u()); 
  }
} 

/*------------------------------------------------*/
void AzPmatApp::truncate(AzPmat *mm, double border, AzPmat *m_deriv) const {
  AzX::throw_if((border <= 0), "AzPmatApp::truncate", "border must be positive"); 
  if (m_deriv == NULL) a._truncate(mm->_dptr_u(), mm->size(), (AzFloat)border); 
  else {
    m_deriv->shape_chk_tmpl(mm, "AzPmatApp::truncate", "m_deriv"); 
    a._truncate(mm->_dptr_u(), mm->size(), (AzFloat)border, m_deriv->_dptr_u()); 
  }
}

/*------------------------------------------------*/
/* ignore #row and #col.  check only size() */
/*------------------------------------------------------------*/
void AzPmatApp::add_with_map(int data_num, 
                             const AzPmat *m1, 
                             AzPmat *m2, 
                             int row_num, 
                             const AzPintArr2 *pia2_2to1) const {
  const char *eyec = "AzPmatApp::add_with_map"; 
  int cnum1 = m1->size() / row_num; 
  int cnum2 = m2->size() / row_num; 
    
  AzX::throw_if((cnum1 % data_num != 0), eyec, "conflict in input #col"); 
  int width1 = cnum1 / data_num; 

  int width2 = pia2_2to1->size(); 
  AzX::throw_if((cnum2 != width2*data_num), eyec, "conflict in #col"); 

  int nummax = pia2_2to1->maxNum(); 
  int stopper = pia2_2to1->stopper(); 

  a._add_with_map(data_num, m1->_dptr(), width1, m2->_dptr_u(), width2, row_num, 
                  pia2_2to1->_dptr(), nummax, stopper); 
}

/*------------------------------------------------------------*/
void AzPmatApp::add_with_map_var(
       const AzIntArr &ia_pos1, const AzIntArr &ia_pos2, /* beg0, beg1, ..., end */
       int denomi, 
       const AzPmat &m1, AzPmat &m2, int row_num, 
       const AzPintArr2 &pia2_2to1) /* template */ const {
  const char *eyec = "AzPmatApp::add_with_map_var"; 
  int data_num = ia_pos1.size()-1; 
  AzX::throw_if(ia_pos2.size()-1 != data_num, eyec, "Wrong output index size");   
  int cnum1 = ia_pos1[data_num], cnum2 = ia_pos2[data_num]; 
  m2.reform(row_num, cnum2); 
  AzX::throw_if(m1.size() != row_num*cnum1, eyec, "Wrong input dimensions"); 

  int maxsz2 = 0; for (int dx = 0; dx < data_num; ++dx) maxsz2 = MAX(maxsz2, ia_pos2[dx+1]-ia_pos2[dx]); 
  AzX::throw_if(pia2_2to1.size() < maxsz2, eyec, "pia2_2to1.size()<maxsz2"); 
  
  AzPintArr pia_pos1(ia_pos1), pia_pos2(ia_pos2); 
  AzPintArr pia_c2dx; gen_col2dx(ia_pos2, pia_c2dx, eyec, denomi); 
  azcparam_var p(m1._dptr(), m2._dptr_u(), row_num, 
                 pia2_2to1._dptr(), pia2_2to1.maxNum(), pia2_2to1.stopper(), 
                 pia_pos1._dptr(), pia_pos2._dptr(), pia_c2dx._dptr(), denomi);   
  a._add_with_map_var(p, cnum2); 
}

/*------------------------------------------------------------*/
/*------------------------------------------------------------*/
void AzPmatApp::pooling_avg(const AzPmat *m1, 
                            int width1, 
                            AzPmat *m2, 
                            const AzPintArr2 *aIa_col1) const {
  const char *eyec = "AzPmatApp::pooling_avg"; 
  AzX::throw_if((m1->col_num % width1 != 0), eyec, "conflict in input #col"); 
  int data_num = m1->col_num / width1; 

  int width2 = aIa_col1->size(); 
  AzX::throw_if((m2->col_num != width2*data_num), eyec, "conflict in #col"); 
  AzX::throw_if((m2->row_num != m1->row_num), eyec, "conflict in #row"); 
  
  int col1_nummax = aIa_col1->maxNum(); 
  int stopper = aIa_col1->stopper(); 
  
  a._pooling_avg(data_num, m1->_dptr(), width1, m2->_dptr_u(), width2, m1->row_num, 
               aIa_col1->_dptr(), col1_nummax, stopper); 
}

/*------------------------------------------------------------*/
void AzPmatApp::unpooling_avg(AzPmat *m1, 
                             const AzPmat *m2,
                             int width2, 
                             const AzPintArr2 *pia2_col2,
                             const AzPintArr *pia_col2_to_num) const {
  const char *eyec = "AzPmatApp::unpooling_avg"; 
  int width1 = pia2_col2->size(); 
  AzX::throw_if((m2->col_num % width2 != 0), eyec, "conflict in #col of m2"); 
  int data_num = m2->col_num / width2; 
  AzX::throw_if((m1->col_num != width1*data_num), eyec, "conflict in #col of m1"); 

  AzX::throw_if((m2->row_num != m1->row_num), eyec, "conflict in #row"); 
  AzX::throw_if((pia_col2_to_num->size() != width2), eyec, "conflict in the length of pia_col2_to_num"); 
  
  int col2_nummax = pia2_col2->maxNum(); 
  int stopper = pia2_col2->stopper();
  a._unpooling_avg(data_num, m1->_dptr_u(), width1, m2->_dptr(), width2, m1->row_num, 
                 pia2_col2->_dptr(), col2_nummax, stopper, pia_col2_to_num->_dptr()); 
}

/*------------------------------------------------------------*/
void AzPmatApp::pooling_l2(const AzPmat *m1, 
                           int width1, 
                           AzPmat *m2, 
                           const AzPintArr2 *aIa_col1) const {
  const char *eyec = "AzPmatApp::pooling_l2"; 
  AzX::throw_if((m1->col_num % width1 != 0), eyec, "conflict in input #col"); 
  int data_num = m1->col_num / width1; 

  int width2 = aIa_col1->size(); 
  AzX::throw_if((m2->col_num != width2*data_num), eyec, "conflict in #col"); 
  AzX::throw_if((m2->row_num != m1->row_num), eyec, "conflict in #row"); 
  
  int col1_nummax = aIa_col1->maxNum(); 
  int stopper = aIa_col1->stopper(); 
  
  a._pooling_l2(data_num, m1->_dptr(), width1, m2->_dptr_u(), width2, m1->row_num, 
                aIa_col1->_dptr(), col1_nummax, stopper); 
}

/*------------------------------------------------------------*/
void AzPmatApp::unpooling_l2(AzPmat *m1, 
                             const AzPmat *m2,
                             int width2, 
                             const AzPintArr2 *pia2_col2,
                             const AzPmat *org_m1, 
                             const AzPmat *org_m2) const {
  const char *eyec = "AzPmatApp::unpooling_l2"; 
  int width1 = pia2_col2->size(); 
  AzX::throw_if((m2->col_num % width2 != 0), eyec, "conflict in #col of m2"); 
  int data_num = m2->col_num / width2; 
  AzX::throw_if((m1->col_num != width1*data_num), eyec, "conflict in #col of m1"); 
  AzX::throw_if((m2->row_num != m1->row_num), eyec, "conflict in #row"); 

  int col2_nummax = pia2_col2->maxNum(); 
  int stopper = pia2_col2->stopper();
  a._unpooling_l2(data_num, m1->_dptr_u(), width1, m2->_dptr(), width2, m1->row_num, 
                  pia2_col2->_dptr(), col2_nummax, stopper, 
                  org_m1->_dptr(), org_m2->_dptr()); 
}

/*------------------------------------------------------------*/
void AzPmatApp::pooling_max(const AzPmat *m1, 
                         int width1, 
                         AzPmat *m2, 
                         const AzPintArr2 *aIa_col1, 
                         AzPintArr *m_chosen) const /* may be NULL */ {
  const char *eyec = "AzPmatApp::pooling_max"; 
  
  AzX::throw_if((m1->col_num % width1 != 0), eyec, "m1 has wrong #col"); 
  int data_num = m1->col_num / width1; 
  
  int width2 = aIa_col1->size(); 
  AzX::throw_if((m2->col_num != width2*data_num), eyec, "#col mismatch"); 
  if (m_chosen != NULL) {
    AzX::throw_if((m2->size() != m_chosen->size()), eyec, "size check failed: m_chosen and m2"); 
  }
  AzX::throw_if((m1->row_num != m2->row_num), eyec, "#row is wrong"); 

  int col1_nummax = aIa_col1->maxNum(); 
  int stopper = aIa_col1->stopper();   
  int *ptr_chosen = (m_chosen != NULL) ? m_chosen->_dptr_u() : NULL; 
  a._pooling_max(data_num, m1->_dptr(), width1, m2->_dptr_u(), width2, m1->row_num, 
               aIa_col1->_dptr(), col1_nummax, stopper, 
               ptr_chosen); 
}                          

/*------------------------------------------------------------*/
void AzPmatApp::unpooling_max(AzPmat *m1, 
                              const AzPmat *m2, 
                              const AzPintArr *m_chosen,
                              const AzPintArr2 *pia2_col1to2) /* [col1] col2_1, col2_2, ... */ const {
  const char *eyec = "AzPmatApp::unpooling_max"; 
  int width1 = pia2_col1to2->size(); 
  AzX::throw_if((m2->size() != m_chosen->size()), eyec, "size check failed: m2 and m_chosen"); 
  AzX::throw_if((m2->row_num != m1->row_num), eyec, "#row mismatch"); 
  
  AzX::throw_if((m1->col_num % width1 != 0), eyec, "m1 has wrong #col"); 
  int data_num = m1->col_num / width1; 

  int width2 = m2->col_num / data_num; 
  AzX::throw_if((m2->col_num != width2*data_num), eyec, "#col mismatch"); 

  a._unpooling_max(data_num, m1->_dptr_u(), width1, m2->_dptr(), width2, m1->row_num, m_chosen->_dptr(), 
                    pia2_col1to2->_dptr(), pia2_col1to2->maxNum(), pia2_col1to2->stopper()); 
}

/*------------------------------------------------------------*/
void AzPmatApp::unpooling_max2(AzPmat *m1, 
                              const AzPmat *m2, 
                              const AzPintArr *m_chosen,
                              const AzPintArr2 *pia2_col1to2) /* [col1] col2_1, col2_2, ... */ const {
  const char *eyec = "AzPmatApp::unpooling_max2"; 
  int width1 = pia2_col1to2->size(); 
  AzX::throw_if((m2->size() != m_chosen->size()), eyec, "size check failed: m2 and m_chosen"); 
  AzX::throw_if((m2->row_num != m1->row_num), eyec, "#row mismatch"); 
  
  AzX::throw_if((m1->col_num % width1 != 0), eyec, "m1 has wrong #col"); 
  int data_num = m1->col_num / width1; 

  int width2 = m2->col_num / data_num; 
  AzX::throw_if((m2->col_num != width2*data_num), eyec, "#col mismatch"); 

  a._unpooling_max2(data_num, m1->_dptr_u(), width1, m2->_dptr(), width2, m1->row_num, m_chosen->_dptr(), 
                    pia2_col1to2->_dptr(), pia2_col1to2->maxNum(), pia2_col1to2->stopper()); 
}

/*------------------------------------------------------------*/
int AzPmatApp::prep_for_var(const char *eyec, const AzPmat &m1, AzPmat &m2, 
                     const AzIntArr &ia_pos1, const AzIntArr &ia_pos2,
                     int &maxsz2) const {
  int data_num = ia_pos1.size()-1; 
  AzX::throw_if(data_num != ia_pos2.size()-1, eyec, "#data-pos1-pos2 conflict"); 
  int cnum1 = ia_pos1[data_num], cnum2 = ia_pos2[data_num]; 
  AzX::throw_if(m1.colNum() != cnum1, eyec, "m1 dimension is wrong"); 
  m2.reform(m1.rowNum(), cnum2);  
  maxsz2 = 0; 
  for (int dx = 0; dx < data_num; ++dx) maxsz2 = MAX(maxsz2, ia_pos2[dx+1]-ia_pos2[dx]); 
  return cnum2; 
}
  
/*------------------------------------------------------------*/
void AzPmatApp::pooling_max_var(const AzPmat &m1, AzPmat &m2, 
                         const AzPintArr2 &pia2_2to1, /* template for one with the larget size */
                         const AzIntArr &ia_pos1, const AzIntArr &ia_pos2,                          
                         AzPintArr *pia_chosen) /* may be NULL */ const {
  const char *eyec = "AzPmatApp::pooling_max_var"; 
  int maxsz2, cnum2 = prep_for_var(eyec, m1, m2, ia_pos1, ia_pos2, maxsz2); 
  AzX::throw_if(pia2_2to1.size() < maxsz2, eyec, "pia2_2to1.size()<maxsz2"); 
  AzPintArr pia_c2dx; gen_col2dx(ia_pos2, pia_c2dx, eyec); 

  int *_d_chosen = NULL; 
  if (pia_chosen != NULL) {
    pia_chosen->alloc(m2.size()); 
    pia_chosen->set(-1); 
    _d_chosen = pia_chosen->_dptr_u(); 
  }
  AzPintArr pia_pos1(&ia_pos1), pia_pos2(&ia_pos2); 
  azcparam_var p(m1._dptr(), m2._dptr_u(), m1.rowNum(), 
           pia2_2to1._dptr(), pia2_2to1.maxNum(), pia2_2to1.stopper(),
           pia_pos1._dptr(), pia_pos2._dptr(), pia_c2dx._dptr()); 

  a._pooling_max_var(p, cnum2, _d_chosen); 
}

/*------------------------------------------------------------*/
void AzPmatApp::pooling_avg_var(const AzPmat &m1, AzPmat &m2, 
                         const AzPintArr2 &pia2_2to1, /* template for one with the larget size */
                         const AzIntArr &ia_pos1, const AzIntArr &ia_pos2, int sz) const {
  const char *eyec = "AzPmatApp::pooling_avg_var"; 
  
  AzX::throw_if(sz <= 0, eyec, "sz must be positive"); 
  int maxsz2, cnum2 = prep_for_var(eyec, m1, m2, ia_pos1, ia_pos2, maxsz2); 
  AzX::throw_if(pia2_2to1.size() < maxsz2, eyec, "pia2_2to1.size()<maxsz2"); 
  AzPintArr pia_c2dx; gen_col2dx(ia_pos2, pia_c2dx, eyec); 
  AzPintArr pia_pos1(&ia_pos1), pia_pos2(&ia_pos2);   
  azcparam_var p(m1._dptr(), m2._dptr_u(), m1.rowNum(), 
           pia2_2to1._dptr(), pia2_2to1.maxNum(), pia2_2to1.stopper(), 
           pia_pos1._dptr(), pia_pos2._dptr(), pia_c2dx._dptr()); 
  a._pooling_avg_var(p, cnum2, sz); 
}                    

/*------------------------------------------------------------*/
void AzPmatApp::unpooling_max_var(const AzPmat &m1, AzPmat &m2, 
                         const AzPintArr2 &pia2_2to1, /* template for one with the larget size */
                         const AzIntArr &ia_pos1, const AzIntArr &ia_pos2,                          
                         const AzPintArr &pia_chosen) const {
  const char *eyec = "AzPmatApp::unpooling_max_var"; 
  int maxsz2, cnum2 = prep_for_var(eyec, m1, m2, ia_pos1, ia_pos2, maxsz2);  
  AzX::throw_if(pia2_2to1.size() < maxsz2, eyec, "pia2_2to1.size()<maxsz2");   
  AzX::throw_if(m1.size() != pia_chosen.size(), eyec, "Dim mismatch btw m1 and chosen"); 

  AzPintArr pia_c2dx; gen_col2dx(ia_pos2, pia_c2dx, eyec);  
  AzPintArr pia_pos1(&ia_pos1), pia_pos2(&ia_pos2);   
  azcparam_var p(m1._dptr(), m2._dptr_u(), m1.rowNum(), 
           pia2_2to1._dptr(), pia2_2to1.maxNum(), pia2_2to1.stopper(), 
           pia_pos1._dptr(), pia_pos2._dptr(), pia_c2dx._dptr()); 
  a._unpooling_max_var(p, cnum2, pia_chosen._dptr()); 
}

/*------------------------------------------------------------*/
void AzPmatApp::unpooling_avg_var(const AzPmat &m1, AzPmat &m2, 
                         const AzPintArr2 &pia2_2to1, /* template for one with the larget size */
                         const AzIntArr &ia_pos1, const AzIntArr &ia_pos2, int sz) const {
  const char *eyec = "AzPmatApp::unpooling_avg_var"; 
  
  int maxsz2, cnum2 = prep_for_var(eyec, m1, m2, ia_pos1, ia_pos2, maxsz2);   
  AzX::throw_if(pia2_2to1.size() < maxsz2, eyec, "pia2_2to1.size()<maxsz2");   
  AzPintArr pia_c2dx; gen_col2dx(ia_pos2, pia_c2dx, eyec); 
  AzPintArr pia_pos1(&ia_pos1), pia_pos2(&ia_pos2);   
  azcparam_var p(m1._dptr(), m2._dptr_u(), m1.rowNum(), 
           pia2_2to1._dptr(), pia2_2to1.maxNum(), pia2_2to1.stopper(), 
           pia_pos1._dptr(), pia_pos2._dptr(), pia_c2dx._dptr()); 
  a._unpooling_avg_var(p, cnum2, sz); 
}

/* response normalization */
/*------------------------------------------------------------*/
/* "cmrnorm" */
/*------------------------------------------------------------*/
void AzPmatApp::resnorm_cross(const AzPmat *m_input, 
                         int size, double alpha, double beta, 
                         double one, 
                         AzPmat *m_inout, 
                         AzPmat *m_oneplussqavg, 
                         bool do_force_old) const {
  const char *eyec = "AzPmatApp::resnorm_cross"; 
  
  int rnum = m_input->rowNum(); 
  int cnum = m_input->colNum(); 
  m_inout->shape_chk(rnum, cnum, eyec, "m_inout"); 
  m_oneplussqavg->reform(rnum, cnum);  

  if (!do_force_old && size >= rnum) {
    AzPmat m_col_sqsum; 
    m_col_sqsum.colSquareSum(m_input); 
    a._resnorm_cross_all(m_input->_dptr(), rnum, cnum, (AzFloat)alpha, (AzFloat)beta, (AzFloat)one, 
                         m_inout->_dptr_u(), m_oneplussqavg->_dptr_u(), m_col_sqsum._dptr());     
  }
  else {
    a._resnorm_cross(m_input->_dptr(), rnum, cnum, size, (AzFloat)alpha, (AzFloat)beta, (AzFloat)one, m_inout->_dptr_u(), m_oneplussqavg->_dptr_u()); 
  }
}                         

/*------------------------------------------------------------*/
void AzPmatApp::unresnorm_cross(const AzPmat *m_grad, 
                                      const AzPmat *m_bef, 
                                      const AzPmat *m_aft, 
                                      const AzPmat *m_oneplussqavg, 
                                      AzPmat *m_tmp, 
                                      AzPmat *m_out, 
                                      int size, double alpha, double beta, 
                                      bool do_force_old) const {                                      
  const char *eyec = "AzPmatApp::unresnorm_cross"; 
  int rnum = m_grad->rowNum(); 
  int cnum = m_grad->colNum();
  m_bef->shape_chk(rnum, cnum, eyec, "m_bef"); 
  m_aft->shape_chk(rnum, cnum, eyec, "m_aft"); 
  m_oneplussqavg->shape_chk(rnum, cnum, eyec, "m_oneplussqavg"); 
  m_out->shape_chk(rnum, cnum, eyec, "m_out"); 
  m_tmp->reform(rnum, cnum); 

  a._prep_unresnorm_cross(m_grad->_dptr(), m_aft->_dptr(), m_oneplussqavg->_dptr(), rnum, cnum, 
                          size, (AzFloat)alpha, (AzFloat)beta, m_tmp->_dptr_u()); 
  
  if (!do_force_old && size >= rnum) {
    AzPmat m_tmp_col_sum; 
    m_tmp_col_sum.colSum(m_tmp); 
    a._unresnorm_cross_all(m_grad->_dptr(), m_bef->_dptr(), m_oneplussqavg->_dptr(), rnum, cnum, 
                           (AzFloat)beta, m_out->_dptr_u(), m_tmp_col_sum._dptr());   
  }
  else {
    a._unresnorm_cross(m_tmp->_dptr(), m_grad->_dptr(), m_bef->_dptr(), m_oneplussqavg->_dptr(), rnum, cnum, 
                       size, (AzFloat)beta, m_out->_dptr_u());    
  }                           
}
 
/*------------------------------------------------------------*/   
void AzPmatApp::resnorm_local(const AzPmat *m_input, 
                              const AzPintArr2 *pia2_neighbors, 
                              const AzPintArr *pia_neigh_sz, 
                              double alpha, double beta, 
                              AzPmat *m_inout, 
                              AzPmat *m_oneplussqavg) const {
  const char *eyec = "AzPmatApp::resnorm_local"; 
  
  int rnum = m_input->rowNum(); 
  int cnum = pia2_neighbors->size(); 
  int col_num = m_input->colNum(); 
  AzX::throw_if((col_num % cnum != 0), eyec, "#column is wrong");   
  m_inout->shape_chk(rnum, col_num, eyec, "m_inout"); 
  int data_num = col_num / cnum;  
  m_oneplussqavg->reform(rnum, col_num); 

  AzX::throw_if((pia_neigh_sz->size() != cnum), eyec, "size of pia_neigh_sz is wrong"); 
  int nummax = pia2_neighbors->maxNum(); 
  int stopper = pia2_neighbors->stopper(); 
  a._resnorm_local(data_num, m_input->_dptr(), rnum, cnum, pia2_neighbors->_dptr(), nummax, stopper, 
                   pia_neigh_sz->_dptr(), (AzFloat)alpha, (AzFloat)beta, 
                   m_inout->_dptr_u(), 
                   m_oneplussqavg->_dptr_u()); 
}                              

/*------------------------------------------------------------*/ 
void AzPmatApp::unresnorm_local(const AzPmat *m_grad, 
                                const AzPmat *m_bef, 
                                const AzPmat *m_aft, 
                                const AzPmat *m_oneplussqavg, 
                                AzPmat *m_tmp, 
                                AzPmat *m_out, 
                                const AzPintArr2 *pia2_whose_neighbor, 
                                const AzPintArr *pia_neigh_sz,                                 
                                double alpha, double beta) const {                                
  const char *eyec = "AzPmatApp::unresnorm_local"; 
  int rnum = m_grad->rowNum(); 
  int col_num = m_grad->colNum();
  m_bef->shape_chk(rnum, col_num, eyec, "m_bef"); 
  m_aft->shape_chk(rnum, col_num, eyec, "m_aft"); 
  m_oneplussqavg->shape_chk(rnum, col_num, eyec, "m_oneplussqavg"); 
  m_out->shape_chk(rnum, col_num, eyec, "m_out"); 
  m_tmp->reform(rnum, col_num); 

  int cnum = pia_neigh_sz->size(); 
  AzX::throw_if((col_num % cnum != 0), eyec, "#column is wrong"); 
  AzX::throw_if((pia2_whose_neighbor->size() != cnum), eyec, "size of pia2_whose_neighbors is wrong"); 
  int data_num = col_num / cnum; 
  
  a._prep_unresnorm_local(data_num, m_grad->_dptr(), m_aft->_dptr(), m_oneplussqavg->_dptr(), rnum, cnum, 
                          pia_neigh_sz->_dptr(), (AzFloat)alpha, (AzFloat)beta, m_tmp->_dptr_u()); 

  int nummax = pia2_whose_neighbor->maxNum(); 
  int stopper = pia2_whose_neighbor->stopper();                           
  a._unresnorm_local(data_num, m_tmp->_dptr(), m_grad->_dptr(), m_bef->_dptr(), m_oneplussqavg->_dptr(), rnum, cnum, 
                     pia2_whose_neighbor->_dptr(), nummax, stopper, (AzFloat)beta, 
                     m_out->_dptr_u()); 
}                         

/*------------------------------------------------------------*/    
/*------------------------------------------------------------*/    
void AzPmatApp::rearrange(int loc_num, 
                          const AzPmat *m1, 
                          int d_num, 
                          AzPmat *m2) const {
  int rnum = m1->rowNum(); 
  AzX::throw_if((m2->rowNum() != rnum), "AzPmatApp::rearrange", "#row is wrong"); 
  int cnum = loc_num*d_num; 
  AzX::throw_if((m1->colNum() != cnum || m2->colNum() != cnum), "AzPmatApp::rearrange", "#column is wrong"); 
  a._rearrange(loc_num, m1->_dptr(), d_num, m2->_dptr_u(), rnum);       
}  

/*------------------------------------------------------------*/    
void AzPmatApp::undo_rearrange(int loc_num, 
                               AzPmat *m1, 
                               int d_num, 
                               const AzPmat *m2) const {
  int rnum = m1->rowNum(); 
  AzX::throw_if((m2->rowNum() != rnum), "AzPmatApp::rearrange", "#row is wrong"); 
  int cnum = loc_num*d_num; 
  AzX::throw_if((m1->colNum() != cnum || m2->colNum() != cnum), "AzPmatApp::rearrange", "#column is wrong"); 
  a._undo_rearrange(loc_num, m1->_dptr_u(), d_num, m2->_dptr(), rnum);
}

/*------------------------------------------------------------*/   
void AzPmatApp::binlogi_deriv(bool is_01, 
                const AzPmat *m_p, const AzPmat *m_y, 
                AzPmat *m_ld, AzPmat *m_loss) const {  
  const char *eyec = "AzPmatApp::bin_logi_deriv";  
  AzX::throw_if((m_y == NULL || m_p == NULL || m_ld == NULL), eyec, "null input");  
  m_y->shape_chk_tmpl(m_p, eyec, "Y and P"); 
  m_ld->reform_tmpl(m_p); 
  if (m_loss != NULL) m_loss->reform_tmpl(m_p); 
  AzFloat *loss = (m_loss == NULL) ? NULL : m_loss->_dptr_u(); 
  a._binlogi_deriv(is_01, m_p->_dptr(), m_y->_dptr(), m_y->size(), 
                   m_ld->_dptr_u(), loss); 
}                            

/*------------------------------------------------------------*/   
void AzPmatApp::binlogi_loss(bool is_01, 
                const AzPmat *m_p, const AzPmat *m_y, 
                AzPmat *m_loss) const {  
  const char *eyec = "AzPmatApp::bin_logi_loss";  
  AzX::throw_if((m_p == NULL || m_y == NULL || m_loss == NULL), eyec, "null input");  
  m_y->shape_chk_tmpl(m_p, eyec, "Y and P"); 
  m_loss->reform_tmpl(m_p); 
  a._binlogi_loss(is_01, m_p->_dptr(), m_y->_dptr(), m_y->size(), m_loss->_dptr_u()); 
} 

/*------------------------------------------------------------*/   
void AzPmatApp::for_log_grad(AzPmat &m_p, const AzPintArr &pia_y_row) const {
  const char *eyec = "AzPmatApp::for_log_grad";  
  AzX::throw_if(m_p.colNum() != pia_y_row.size(), eyec, "shape mismatch"); 
  a._for_log_grad(m_p._dptr_u(), pia_y_row._dptr(), m_p.rowNum(), m_p.colNum()); 
}                            

/*------------------------------------------------------------*/   
void AzPmatApp::for_log_loss(const AzPmat &m_p, const AzPintArr &pia_y_row, AzPmat &m_out) const {
  const char *eyec = "AzPmatApp::log_loss";  
  AzX::throw_if(m_p.colNum() != pia_y_row.size(), eyec, "shape mismatch");  
  m_out.reform(1, m_p.colNum()); 
  a._for_log_loss(m_p._dptr(), pia_y_row._dptr(), m_p.rowNum(), m_p.colNum(), m_out._dptr_u()); 
}  

/*------------------------------------------------------------*/  
void AzPmatApp::sumone(AzPmatVar *mv, bool do_scale) const {
  AzX::throw_if_null(mv, "AzPmatApp::sumone", "input"); 
  a._sumone(mv->data_u()->_dptr_u(), mv->rowNum(), mv->d_index()->_dptr(), mv->dataNum(), do_scale); 
}
  
/*------------------------------------------------------------*/ 
void AzPmatApp::unsumone(AzPmatVar *mv_grad, const AzPmat *m_inp, bool do_scale) const {
  const char *eyec = "AzPmatApp::unsumone"; 
  AzX::throw_if_null(mv_grad, eyec, "mv_grad"); 
  AzX::throw_if_null(m_inp, eyec, "m_inp"); 
  mv_grad->data()->shape_chk_tmpl(m_inp, eyec, "mv_grad and m_inp"); 
  a._unsumone(mv_grad->data_u()->_dptr_u(), m_inp->_dptr(), mv_grad->rowNum(), 
              mv_grad->d_index()->_dptr(), mv_grad->dataNum(), do_scale); 
}    

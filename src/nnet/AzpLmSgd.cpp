/* * * * *
 *  AzpLmSgd.cpp 
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

#include "AzpLmSgd.hpp" 
#include "AzPmatApp.hpp"

/*--------------------------------------------------------*/
template <class M>
void AzpLmSgd::_updateDelta(int d_num, 
                         const AzpLmParam &p, 
                         const M *m_x,
                         const AzPmat *m_deriv) 
{
  if (p.dont_update()) return;  
  AzX::no_support((grad_num != 0), "AzpLmSgd::_updateDelta", "Multiple updateDelta before flushing"); 
    
  if (doing_partial()) {
    if (do_gradpart) AzPs::prod(&m_w_grad, m_x, m_deriv, false, true); 
    else {
      AzPmat m; 
      AzPs::prod(&m, m_x, m_deriv, false, true); 
      m_w_grad.zeroOut(); 
      m_w_grad.add_s2d(&m, ia_p2w.point(), ia_p2w.size());  /* d2s -> s2d: 04/22/2015 */
    }
  }
  else {
    AzPs::prod(&m_w_grad, m_x, m_deriv, false, true); 
  }
  
  if (!p.do_no_intercept) {
    gen_one(m_x->colNum(), &m_one); 
    if (doing_partial()) { 
      if (do_gradpart) v_i_grad.prod(&m_one, m_deriv, false, true); 
      else {
        AzPmat m; m.prod(&m_one, m_deriv, false, true);    
        v_i_grad.zeroOut(); 
        v_i_grad.add_s2d(&m, ia_p2w.point(), ia_p2w.size());  /* d2s -> s2d: 04/22/2015 */
      }
    }
    else v_i_grad.prod(&m_one, m_deriv, false, true); 

  }
  if      (p.do_nodiv)        grad_num = 1; 
  else if (p.do_count_regions) grad_num = m_x->colNum(); 
  else                         grad_num = d_num; 
} 
template void AzpLmSgd::_updateDelta<AzPmat>(int, const AzpLmParam &, const AzPmat *, const AzPmat *); 
template void AzpLmSgd::_updateDelta<AzPmatSpa>(int, const AzpLmParam &, const AzPmatSpa *, const AzPmat *); 

/*--------------------------------------------------------*/
void AzpLmSgd::_updateDelta(int d_num, const AzpLmParam &p, const AzPmat *m_grad) {
  if (p.dont_update()) return;  
  const char *eyec = "AzpLmSgd::_updateDelta(grad)"; 
  AzX::no_support(doing_partial(), eyec, "Partial update"); 
  if (grad_num <= 0) {
    m_w_grad.set(m_grad); 
    if (p.do_nodiv) grad_num = 1; 
    else            grad_num = d_num;     
  }
  else {
    m_w_grad.add(m_grad); 
    if (p.do_nodiv) grad_num = 1; 
    else            grad_num += d_num;     
  }
} 

/*--------------------------------------------------------*/
template <class M>
void AzpLmSgd::_updateDelta2(int d_num, 
                         const AzpLmParam &p, 
                         const M *m_x,
                         const AzPmat *m_deriv) 
{
  if (p.dont_update()) return;  
  AzX::throw_if((grad_num <= 0), "AzpLmSgd::_updateDelta2", "grad_num=0"); 
  AzX::no_support(doing_partial(), "AzpLmSgd::_updateDelta2", "Partial with multiple updateDelta before flushing."); 
  AzPs::add_prod(&m_w_grad, m_x, m_deriv, false, true); 
  if (!p.do_no_intercept) {
    gen_one(m_x->colNum(), &m_one); 
    v_i_grad.add_prod(&m_one, m_deriv, false, true); 
  }
  if      (p.do_nodiv)         grad_num = 1; 
  else if (p.do_count_regions) grad_num += m_x->colNum(); 
  else                         grad_num += d_num; 
}
template void AzpLmSgd::_updateDelta2<AzPmat>(int, const AzpLmParam &, const AzPmat *, const AzPmat *); 
template void AzpLmSgd::_updateDelta2<AzPmatSpa>(int, const AzpLmParam &, const AzPmatSpa *, const AzPmat *); 

/*------------------------------------------------------------*/  
void AzpLmSgd::flushDelta(const AzpLmParam &p, const AzpLmSgd_Param &ps)
{
  if (p.dont_update()) return; 
  if (grad_num <= 0) return; 

  if (p.do_fixw) m_w_grad.zeroOut(); /* for analysis purposes only */
  if (p.do_fixi) v_i_grad.zeroOut(); /* for analysis purposes only */
  
  if (p.grad_clip > 0) { /* added 1/11/2016 */
    double val = p.grad_clip * (double)grad_num; 
    m_w_grad.truncate(-val, val); 
    v_i_grad.truncate(-val, val); 
  }
  
  double etab = (ps.etab_coeff == 1) ? ps.eta : ps.eta*ps.etab_coeff; 
  if (ps.do_fast_flush && ps.momentum > 0) {  
    check_ws("flushDelta with momentum (fast_flush)"); 

    double mm = MAX(0,ps.momentum);  
    if (p.reg_L2 > 0 && p.reg_L2init <= 0) {
      m_w_dlt.add(mm, &m_w_grad, -ps.eta/(double)grad_num, &m_w, -ps.eta*p.reg_L2);      
    }
    else {     
      m_w_dlt.add(mm, &m_w_grad, -ps.eta/(double)grad_num); 
      if (!p.no_regadd()) {
        add_reg_grad(p, ps.eta, &m_w, &m_w_dlt, &m_w_init); 
      }   
    } 
    m_w.add(&m_w_dlt); 

    v_i_dlt.add(mm, &v_i_grad, -etab/(double)grad_num);  
    if (p.do_reg_intercept && !p.no_regadd()) {
      add_reg_grad(p, etab, &v_i, &v_i_dlt, &v_i_init);  /* regularization */      
    }    
    v_i.add(&v_i_dlt); 
    do_gradpart = false; 
  }
  else if (ps.momentum > 0 || p.reg_L2init > 0) { /* use momentum; slower; keeping this for compatibility */
    check_ws("flushDelta with momentum"); 
    double mm = MAX(0,ps.momentum); 
    m_w_grad.multiply(-ps.eta/(double)grad_num); 
    add_reg_grad(p, ps.eta, &m_w, &m_w_grad, &m_w_init);     /* regularization */
    m_w_grad.add(&m_w_dlt, mm); 
    m_w.add(&m_w_grad); 
    m_w_dlt.set_chk(&m_w_grad);  
    
    v_i_grad.multiply(-etab/(double)grad_num); 
    if (p.do_reg_intercept) {
      add_reg_grad(p, etab, &v_i, &v_i_grad, &v_i_init);  /* regularization */      
    }
    v_i_grad.add(&v_i_dlt, mm); 
    v_i.add(&v_i_grad); 
    v_i_dlt.set_chk(&v_i_grad); 
    do_gradpart = false; 
  }
  else { /* don't use momentum */
    regularize(p, ps.eta, etab); 
    if (doing_partial() && do_gradpart) {
      m_w.add_s2d(&m_w_grad, ia_p2w.point(), ia_p2w.size(), -ps.eta/(double)grad_num/ws); 
      v_i.add_s2d(&v_i_grad, ia_p2w.point(), ia_p2w.size(), -etab/(double)grad_num);      
    }
    else {
      m_w.add(&m_w_grad, -ps.eta/(double)grad_num/ws); 
      v_i.add(&v_i_grad, -etab/(double)grad_num); 
    }
    do_gradpart = doing_partial(); 
  }
  if (p.weight_clip > 0) {
    m_w.truncate(-p.weight_clip/ws, p.weight_clip/ws); 
    v_i.truncate(-p.weight_clip/ws, p.weight_clip/ws);     
  }
  
  if (p.reg_L2const > 0) do_l2const(p); 
  grad_num = 0; 
  if (ws < 1e-4) flush_ws(); 
}

/*------------------------------------------------------------*/ 
void AzpLmSgd::regularize(const AzpLmParam &p, double eta, double etab) 
{
  AzX::no_support((p.reg_L2init > 0), "AzpLmSgd::regularize", "reg_L2init in this configuration"); 

  if (p.reg_L2 == 0) {}
  else if (p.reg_L2 > 0) {
    ws *= (1 - p.reg_L2 * eta); 
    if (p.do_reg_intercept) {
      v_i.multiply(1 - p.reg_L2*etab); 
    }
  }
  else if (p.reg_L1L2 > 0) {
    check_ws("AzpLmSgd::regularize(L1L2)"); 
    AzPmatApp app; 
    AzPmat m; 
    app.l1l2deriv(&m_w, &m, (AzFloat)p.reg_L1L2_delta); 
    m_w.add(&m, -p.reg_L1L2*eta); 
    if (p.do_reg_intercept) {
      app.l1l2deriv(&v_i, &m, (AzFloat)p.reg_L1L2_delta); 
      v_i.add(&m, -p.reg_L1L2*etab); 
    }   
  }
}

/*------------------------------------------------------------*/ 
void AzpLmSgd::add_reg_grad(const AzpLmParam &p, 
                           double eta, 
                           const AzPmat *m, 
                           /*---  output  ---*/
                           AzPmat *m_delta, 
                           /*--- optinal input  ---*/
                           const AzPmat *m_init) const
{
  const char *eyec = "AzpLmSgd::add_reg_grad";  
  if (p.reg_L2init > 0) {
    AzX::throw_if((m_init == NULL || m_init->rowNum() != m->rowNum()), eyec, "No m_init?!");  
    int cnum0 = m_init->colNum(); 
    double coeff0 = eta*p.reg_L2init; 
    if (cnum0 == m_delta->colNum()) { /* penalty: (a/2)(x-init)^2; derivative: a(x-init)  */
      m_delta->add(1, m, -coeff0, m_init, coeff0); /* negative gradient */
    }
    else {
      m_delta->add(0, cnum0, m, 0, -coeff0); 
      m_delta->add(0, cnum0, m_init, 0, coeff0); 
    }
  }
  if (p.reg_L2 == 0){}
  else if (p.reg_L2 > 0) {
    m_delta->add(m, -p.reg_L2*eta); 
  }
  else if (p.reg_L1L2 > 0) {
    double coeff = -p.reg_L1L2*eta; 
    AzPmatApp app; 
    app.add_l1l2deriv(m, m_delta, (AzFloat)p.reg_L1L2_delta, (AzFloat)coeff);       
  }
}

/*------------------------------------------------------------*/ 
void AzpLmSgd::do_l2const(const AzpLmParam &p) 
{
  if (p.reg_L2const <= 0) return; 
  double c2 = p.reg_L2const*p.reg_L2const; 
  if (p.do_l2const_each) c2 *= m_w.rowNum(); 

  AzPmat m_sqsum; 
  m_sqsum.colSquareSum(&m_w);   
  if (p.do_reg_intercept) {
    m_sqsum.add_colSquareSum(&v_i); 
  }
  AzDvect v_sqsum; m_sqsum.get(&v_sqsum); 
  const double *sqsum = v_sqsum.point(); 

  AzDvect v(v_sqsum.rowNum()); 
  bool do_multi=false; 
  v.set(1); 
  for (int col = 0; col < m_w.colNum(); ++col) {
    if (sqsum[col] > c2) {
      double n2 = sqrt(sqsum[col]); 
      double eps = 1e-8; 
      v.set(col, p.reg_L2const/(n2+eps)); 
      do_multi=true; 
    }
  }
  if (do_multi) {
    AzPmat m(&v); m.change_dim(1, v.rowNum()); 
    m_w.multiply_eachcol(&m); 
    if (p.do_reg_intercept) v_i.multiply_eachcol(&m); 
  }
}
/* * * * *
 *  AzpLmRmsp.hpp 
 *  Copyright (C) 2016,2017 Rie Johnson
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

#ifndef _AZP_LM_RMSP_HPP_
#define _AZP_LM_RMSP_HPP_

#include "AzpLmSgd.hpp"

/* 
 * RmsProp by (Tieleman & Hinton, 2012)
 */

class AzpLmRmsp_Param {
public:
  double decay, eta, coeff, eps, grad_clip_after; 
  bool do_shrink; 
  AzpLmRmsp_Param() : eta(-1), decay(0.95), coeff(1), eps(-1), grad_clip_after(-1), do_shrink(false) {}

  /*------------------------------------------------------------*/ 
  #define kw_decay  "rmsprop_decay="
  #define kw_rmsprop_eps "rmsprop_eps="
  #define kw_grad_clip_after "grad_clip_after="
  #define kw_do_shrink "Shrink"
  void resetParam(AzParam &azp, const char *pfx, bool is_warmstart=false) {
    azp.reset_prefix(pfx); 
    azp.vFloat(kw_decay, &decay); 
    azp.vFloat(kw_eta, &eta); 
    azp.vFloat(kw_rmsprop_eps, &eps); 
    eps = MAX(eps, azc_epsilon); 
    azp.vFloat(kw_grad_clip_after, &grad_clip_after); 
    azp.swOn(&do_shrink, kw_do_shrink); 
    azp.reset_prefix(); 
  }  

  void checkParam(const char *pfx) {  
    const char *eyec = "AzpLmRmsp_Param::checkParam";   
    AzXi::throw_if_nonpositive(decay, eyec, kw_decay, pfx); 
    AzXi::throw_if_nonpositive(eta, eyec, kw_eta, pfx); 
    AzX::throw_if((decay >= 1), AzInputError, eyec, kw_decay, "must be no greater than 1."); 
  }

  void printParam(const AzOut &out, const char *pfx) const {
    if (out.isNull()) return; 
    AzPrint o(out, pfx); 
    o.printV(kw_eta, eta); 
    o.printV(kw_decay, decay); 
    o.printV(kw_rmsprop_eps, eps); 
    o.printV(kw_grad_clip_after, grad_clip_after); 
    o.printSw(kw_do_shrink, do_shrink); 
    o.printEnd(); 
  } 
}; 

class AzpLmRmsp : public virtual AzpLmSgd {
protected:
  AzPmat m_w_g2avg, v_i_g2avg; 

public:
  AzpLmRmsp() {}
  virtual const char *description() const { return "Rmsp"; }

  virtual void resetWork() {
    AzpLmSgd::resetWork(); 
    m_w_g2avg.zeroOut();  v_i_g2avg.zeroOut(); 
  }
  virtual void reformWork() {
    AzpLmSgd::reformWork(); 
    m_w_g2avg.reform_tmpl(&m_w); v_i_g2avg.reform_tmpl(&v_i);   
  }  
  
  void reset(const AzpLmRmsp *inp) { /* copy */
    AzpLmSgd::reset(inp); 
    m_w_g2avg.set(&inp->m_w_g2avg); v_i_g2avg.set(&inp->v_i_g2avg);   
  }                          

  void flushDelta(const AzpLmParam &p, const AzpLmRmsp_Param &pa) {
    if (p.dont_update()) return; 
    if (grad_num <= 0) return; 
    check_ws("AzpLmRmsp::flushDelta"); 
    
    bool do_reg = true; 
    rmsp_update(&m_w, &m_w_grad, &m_w_g2avg, &m_w_init, p, pa, do_reg); 
    do_reg = p.do_reg_intercept; 
    rmsp_update(&v_i, &v_i_grad, &v_i_g2avg, &v_i_init, p, pa, do_reg); 

    if (p.reg_L2const > 0) do_l2const(p); 
    grad_num = 0; 
  }

protected:   
  void rmsp_update(AzPmat *m_weight, 
                       AzPmat *m_grad, /* input: grad */
                       AzPmat *m_g2avg, 
                       AzPmat *m_init, 
                       const AzpLmParam &p, 
                       const AzpLmRmsp_Param &pa, 
                       bool do_reg) {
    double rho = pa.decay, eta = pa.eta*pa.coeff; 

    m_grad->divide(-grad_num);  /* negative gradient: -g_t */
    if (do_reg && !pa.do_shrink) add_reg_grad(p, 1, m_weight, m_grad, m_init); /* regularization */
    if (p.grad_clip > 0) m_grad->truncate(-p.grad_clip, p.grad_clip);     
    /*
     *  r_t = (1-decay) g^2_t + decay r_{t-1}
     *  delta_t = -eta / sqrt(r_t + epsilon) g   # epsilon for stability 
     *  x_{t+1} = x_t + delta_t
     */
     
    m_g2avg->add_square(rho, m_grad, 1-rho);
    m_grad->scale_by_sqrt(m_g2avg, pa.eps, true); 
    if (do_reg && pa.do_shrink) add_reg_grad(p, 1, m_weight, m_grad, m_init); /* shrink weights */    
    if (pa.grad_clip_after > 0) m_grad->truncate(-pa.grad_clip_after, pa.grad_clip_after); 
    m_weight->add(m_grad, eta); 
    if (p.weight_clip > 0) m_weight->truncate(-p.weight_clip, p.weight_clip); 
  }   
};
#endif 
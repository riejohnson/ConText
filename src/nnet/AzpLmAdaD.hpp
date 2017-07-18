/* * * * *
 *  AzpLmAdaD.hpp 
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

#ifndef _AZP_LM_ADAD_HPP_
#define _AZP_LM_ADAD_HPP_

#include "AzpLmSgd.hpp"

/* 
 * "ADADELTA: An Adaptive Learning Rate Method", Matthew D. Zeiler, arXuvL1212,5701
 */

class AzpLmAdaD_Param {
public:
  double rho, eps; 
  double coeff; 
  AzpLmAdaD_Param() : rho(-1), eps(-1), coeff(1) {}

  /*------------------------------------------------------------*/ 
  #define kw_rho    "adad_rho="
  #define kw_eps    "adad_epsilon="
  void resetParam(AzParam &azp, const char *pfx, bool is_warmstart=false) {
    azp.reset_prefix(pfx); 
    azp.vFloat(kw_rho, &rho); 
    azp.vFloat(kw_eps, &eps); 
    coeff = 1; 
    azp.reset_prefix(); 
  }  

  void checkParam(const char *pfx) {  
    const char *eyec = "AzpLmParam_adad::checkParam";   
    AzXi::throw_if_nonpositive(rho, eyec, kw_rho, pfx); 
    AzXi::throw_if_nonpositive(eps, eyec, kw_eps, pfx); 
    AzX::throw_if((rho >= 1), AzInputError, eyec, kw_rho, "must be no greater than 1."); 
  }

  void printParam(const AzOut &out, const char *pfx) const {
    if (out.isNull()) return; 
    AzPrint o(out, pfx); 
    o.printV(kw_rho, rho); 
    o.printV(kw_eps, eps); 
    o.printEnd(); 
  } 
}; 

/*---  AdaDelta  ---*/
class AzpLmAdaD : public virtual AzpLmSgd {
protected:
  AzPmat m_w_g2avg, v_i_g2avg; 
  /* use dlt as d2avg */

public:
  AzpLmAdaD() {}
  virtual const char *description() const { return "AdaDelta"; }

  virtual void resetWork() {
    AzpLmSgd::resetWork(); 
    m_w_g2avg.zeroOut();  v_i_g2avg.zeroOut(); 
  }
  virtual void reformWork() {
    AzpLmSgd::reformWork(); 
    m_w_g2avg.reform_tmpl(&m_w); v_i_g2avg.reform_tmpl(&v_i);   
  }  
  
  void reset(const AzpLmAdaD *inp) { /* copy */
    AzpLmSgd::reset(inp); 
    m_w_g2avg.set(&inp->m_w_g2avg); v_i_g2avg.set(&inp->v_i_g2avg);   
  }                          

  void flushDelta(const AzpLmParam &p, const AzpLmAdaD_Param &pa) {
    if (p.dont_update()) return; 
    if (grad_num <= 0) return; 
    check_ws("AzpLmAdaD::flushDelta"); 

    bool do_reg = true; 
    adad_update(&m_w, &m_w_grad, &m_w_g2avg, &m_w_dlt, &m_w_init, 
                p, pa, do_reg); 
    do_reg = p.do_reg_intercept; 
    adad_update(&v_i, &v_i_grad, &v_i_g2avg, &v_i_dlt, &v_i_init, 
                p, pa, do_reg); 

    if (p.reg_L2const > 0) do_l2const(p); 
    grad_num = 0; 
  }

protected:   
  void adad_update(AzPmat *m_weight, 
                       AzPmat *m_grad, /* input: grad */
                       AzPmat *m_g2avg, 
                       AzPmat *m_d2avg, 
                       AzPmat *m_init, 
                       const AzpLmParam &p, 
                       const AzpLmAdaD_Param &pa, 
                       bool do_reg) {
    double rho = pa.rho, eps = pa.eps; 

    m_grad->divide(-grad_num);  /* negative gradient: -g_t */
    if (p.grad_clip > 0) { /* added 1/11/2016 */
      m_grad->truncate(-p.grad_clip, p.grad_clip);  
    }
    
    if (do_reg) {
      double dummy_eta = 1; 
      add_reg_grad(p, dummy_eta, m_weight, m_grad, m_init);     /* regularization */
    }  
    /*
     *  E[g^2]_t = rho E[g^2]_{t-1} + (1 - rho) g_t^2
     *  delta_t = - (RMS[delta_{t-1}] / RMS[g]_t) g_t;    RMS[g]_t = sqrt(E[g^2]_t + epsilon)
     *  E[delta^2]_t = rho E[delta^2]_{t-1} + (1-rho) delta_t^2
     *  x_{t+1} = x_t + delta_t 
     */
    m_g2avg->add_square(rho, m_grad, 1-rho);   /* E[g^2]_t = rho E[g^2]_{t-1} + (1 - rho)g_t^2 */
    m_grad->scale_by_sqrt(m_d2avg, eps);       /* - RMS[delta_{t-1}] g_t */
    m_grad->scale_by_sqrt(m_g2avg, eps, true); /* delta_t = -(RMS[delta_{t-1}]/RMS[g]_t) g_t */
    m_d2avg->add_square(rho, m_grad, 1-rho);   /* E[delta^2]_t = rho E[delta^2]_{t-1} + (1-rho) delta_t^2 */

    m_weight->add(m_grad, pa.coeff); 
    if (p.weight_clip > 0) m_weight->truncate(-p.weight_clip, p.weight_clip); 
  }   
};
#endif 
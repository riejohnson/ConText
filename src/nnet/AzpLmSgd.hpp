/* * * * *
 *  AzpLmSgd.hpp 
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

#ifndef _AZP_LM_SGD_HPP_
#define _AZP_LM_SGD_HPP_

#include "AzUtil.hpp"
#include "AzPmat.hpp"
#include "AzpLm.hpp"

class AzpLmSgd_Param {
public:
  double eta, etab_coeff, momentum;
  bool do_fast_flush; 
  AzpLmSgd_Param() : eta(-1), etab_coeff(1), momentum(-1), do_fast_flush(true) {}

  /*------------------------------------------------------------*/ 
  #define kw_momentum    "momentum="
  #define kw_eta         "step_size="
  #define kw_etab_coeff    "step_sizeb_coeff="
  #define kw_do_fast_flush "FastFlush"  
  #define kw_no_fast_flush "NoFastFlush"
  void resetParam(AzParam &azp, const char *pfx, bool is_warmstart=false) {
    azp.reset_prefix(pfx); 
    azp.vFloat(kw_eta, &eta); 
    azp.vFloat(kw_etab_coeff, &etab_coeff);
    azp.vFloat(kw_momentum, &momentum); 
    if (do_fast_flush) azp.swOff(&do_fast_flush, kw_no_fast_flush); 
    else               azp.swOn(&do_fast_flush, kw_do_fast_flush); 
    azp.reset_prefix(); 
  }  

  void checkParam(const char *pfx) {  
    const char *eyec = "AzpLmSgd_Param::checkParam";   
    AzXi::throw_if_nonpositive(eta, eyec, kw_eta, pfx); 
    AzXi::throw_if_nonpositive(etab_coeff, eyec, kw_etab_coeff, pfx);     
  }

  void printParam(const AzOut &out, const char *pfx) const {
    if (out.isNull()) return; 
    AzPrint o(out, pfx); 
    o.printV(kw_eta, eta); 
    o.printV(kw_etab_coeff, etab_coeff);     
    o.printV(kw_momentum, momentum); 
    o.printSw(kw_do_fast_flush, do_fast_flush); 
    o.printEnd(); 
  } 
  void printHelp(AzHelp &h) const {
    h.item_required(kw_eta, "Step-size (learning rate) for SGD."); 
    h.item(kw_momentum, "Momentum for SGD."); 
  }
}; 

class AzpLmSgd : public virtual AzpLm {
protected:
  bool do_gradpart; 
  AzPmat m_w_grad, m_w_dlt; 
  AzPmat v_i_grad, v_i_dlt; 

  int grad_num; 
 
public:
  AzpLmSgd() : grad_num(0), do_gradpart(false) {}
  virtual const char *description() const { return "SGD"; }
  virtual void resetWork() {
    m_w_grad.zeroOut(); m_w_dlt.zeroOut();
    v_i_grad.zeroOut(); v_i_dlt.zeroOut(); 
    grad_num = 0; 
  }
  virtual void reformWork() {
    m_w_grad.reform_tmpl(&m_w); m_w_dlt.reform_tmpl(&m_w); 
    v_i_grad.reform_tmpl(&v_i); v_i_dlt.reform_tmpl(&v_i);     
    grad_num = 0; 
  }
  virtual void clearTemp() { clearGrad(); }

  void reset(const AzpLmSgd *inp) { /* copy */
    AzpLm::reset(inp); 
    m_w_grad.set(&inp->m_w_grad); m_w_dlt.set(&inp->m_w_dlt); 
    v_i_grad.set(&inp->v_i_grad); v_i_dlt.set(&inp->v_i_dlt);   
    grad_num = inp->grad_num; 
  }

  virtual void updateDelta(int d_num, const AzpLmParam &p, 
                           const AzPmatSpa *m_x, const AzPmat *m_deriv, const void *pp=NULL) {                           
    AzX::throw_if_null(m_x, m_deriv, "AzpLmSgd::updateDelta(spa)"); 
    if (grad_num <= 0) _updateDelta(d_num, p, m_x, m_deriv); 
    else               _updateDelta2(d_num, p, m_x, m_deriv); 
  }                           
  virtual void updateDelta(int d_num, const AzpLmParam &p, 
                           const AzPmat *m_x, const AzPmat *m_deriv, const void *pp=NULL) {    
    AzX::throw_if_null(m_deriv, "AzpLmSgd::updateDelta(dense)"); 
    if (m_x == NULL)        _updateDelta(d_num, p, m_deriv); /* 12/04/2016 for bn */
    else if (grad_num <= 0) _updateDelta(d_num, p, m_x, m_deriv);
    else                    _updateDelta2(d_num, p, m_x, m_deriv); 
  }
  virtual void flushDelta(const AzpLmParam &p, const AzpLmSgd_Param &ps);  

protected:  
  virtual void clearGrad() { m_w_grad.zeroOut(); v_i_grad.zeroOut(); grad_num=0; }
  template <class M>
  void _updateDelta(int d_num, const AzpLmParam &p, const M *m_x, const AzPmat *m_deriv);  
  template <class M>
  void _updateDelta2(int d_num, const AzpLmParam &p, const M *m_x, const AzPmat *m_deriv);  
  void _updateDelta(int d_num, const AzpLmParam &p, const AzPmat *m_grad); 
  virtual void regularize(const AzpLmParam &p, double eta, double etab);   
  virtual void add_reg_grad(const AzpLmParam &p, double eta, const AzPmat *m, 
                    AzPmat *m_delta, /* output */
                    const AzPmat *m_init=NULL) /* optional input */ const; 
  virtual void do_l2const(const AzpLmParam &p); 
};
#endif 
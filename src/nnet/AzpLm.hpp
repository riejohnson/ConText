/* * * * *
 *  AzpLm.hpp
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

#ifndef _AZP_LM_HPP_
#define _AZP_LM_HPP_

#include "AzUtil.hpp"
#include "AzPmat.hpp"
#include "AzPmatApp.hpp"
#include "AzpLmParam.hpp"
#include "AzRandGen.hpp"

/*---  "Lm" for linear model  ---*/
class AzpLm {
protected:
  double ws; /* scale of weights; only for m_w */
  AzPmat m_w;  /* [feat#, class#] */
  AzPmat v_i;  /* [1, class#] */

  AzPmat m_one; 
  
  /*---  for reg_L2init  ---*/  
  AzPmat m_w_init, v_i_init;   

  /*---  for speed up on high dimensionality  ---*/
  AzIntArr ia_p2w; /* part to whole */
  AzIntArr ia_w2p; /* whole to part */
  AzPmat m_w_part, v_i_part; 

  bool doing_partial() const {
    return (ia_p2w.size() > 0); 
  }

  static const int version = 0; 
  static const int reserved_len = 64;   
public:
  AzpLm() : ws(1) {}
  void reset_partial(const AzIntArr *_ia_p2w, const AzIntArr *_ia_w2p) {
    ia_p2w.reset(_ia_p2w); 
    ia_w2p.reset(_ia_w2p); 
    m_w_part.set(&m_w, ia_p2w.point(), ia_p2w.size()); 
    v_i_part.set(&v_i, ia_p2w.point(), ia_p2w.size()); 
  }
  void reset_partial() {
    ia_p2w.reset(); 
    ia_w2p.reset(); 
    m_w_part.destroy(); 
    v_i_part.destroy(); 
  }

  const AzPmat *weights() const {
    check_ws("AzpLm::weights()"); 
    return &m_w; 
  }
  const AzPmat *intercepts() const {
    return &v_i; 
  }

  /*--------------------------------*/

  /*------------------------------------------------------------*/    
  virtual void reset(const AzpLm *inp) { /* copy */
    ws = inp->ws; 
    m_w.set(&inp->m_w); 
    v_i.set(&inp->v_i); 
    m_w_init.set(&inp->m_w_init); 
    v_i_init.set(&inp->v_i_init);    
  }
 
  /*------------------------------------------------------------*/    
  virtual void reset(const AzpLmParam &p, const AzpLm *inp, double coeff) {
    const char *eyec = "AzpLm::reset(p,inp,coeff)";   
    int inp_cnum = inp->m_w.colNum(); 
    if (m_w.rowNum() != inp->m_w.rowNum()) {
      AzBytArr s("nn:");s << m_w.rowNum() << " input:" << inp->m_w.rowNum(); 
      AzX::throw_if(true, eyec, "#rows mismatch", s.c_str()); 
    }
    if (inp_cnum > m_w.colNum()) {
      AzBytArr s("nn:");s << m_w.colNum() << " input:" << inp_cnum; 
      AzX::throw_if(true, eyec, "Invalid #column", s.c_str()); 
    }
    inp->check_ws("input to reset"); 
    check_ws(eyec); 

    if (inp_cnum < m_w.colNum()) {
      initWeights(p); 
    }
    m_w.set(0, inp_cnum, &inp->m_w, 0, coeff);
    v_i.set(0, inp_cnum, &inp->v_i, 0, coeff);    
  }
  
  /*================================================================*/      
  virtual const char *description() const = 0; 
  virtual void updateDelta(int d_num, 
                           const AzpLmParam &p, 
                           const AzPmat *m_x,      /* [dim,data] */
                           const AzPmat *m_deriv,  /* [class,data] */
                           const void *pp=NULL) = 0; 
  virtual void updateDelta(int d_num, 
                           const AzpLmParam &p, 
                           const AzPmatSpa *m_x,        /* [dim,data] */
                           const AzPmat *m_deriv,       /* [class,data] */                        
                           const void *pp=NULL) = 0; 
                          
  virtual void reformWork() = 0; 
  virtual void resetWork() = 0; 
  virtual void clearTemp() = 0; 
 
  /*================================================================*/    
  
  /*------------------------------------------------------------*/      
  virtual void copyWeights_from(const AzpLm *inp) {
    ws = inp->ws; 
    m_w.set(&inp->m_w); 
    v_i.set(&inp->v_i); 
    resetWork(); 
  }
  
  /*-----                read/write                        -----*/  
  /*------------------------------------------------------------*/      
  virtual void write(AzFile *file) const {
    check_ws("write"); 
    AzTools::write_header(file, version, reserved_len); 
    m_w.write(file); 
    v_i.write(file); 
  }
  /*------------------------------------------------------------*/      
  virtual void read(AzFile *file) {
    reset_ws(); 
    AzTools::read_header(file, reserved_len);     
    m_w.read(file); 
    v_i.read(file); 
    reformWork(); 
  }
  /*------------------------------------------------------------*/      
  virtual void write(const char *fn) const { AzFile::write(fn, this); }  
  virtual void read(const char *fn) { AzFile::read(fn, this); }  

  /*-----                initialization                    -----*/  
  /*------------------------------------------------------------*/      
  virtual void reset() { 
    reset_ws(); 
    m_w.destroy(); 
    v_i.destroy(); 
    reset_partial(); 
  }  
  /*------------------------------------------------------------*/    
  virtual void reset(int class_num, const AzpLmParam &p, int inp_dim) {
    reset_ws(); 
    m_w.reform(inp_dim, class_num); 
    v_i.reform(1, class_num); 
    reset_partial();  
    reformWork();     
  }

  /*-----                                                  -----*/  
  /*------------------------------------------------------------*/      
  virtual void end_of_epoch(const AzpLmParam &p) { 
    flush_ws(); 
  }
  
  virtual void setup_for_reg_L2init(const AzpLmParam &p) {
    if (p.reg_L2init > 0 && p.reg_L2init < AzpLmParam_reg_infinity) {
      /*---  keep initial weights for regularization  ---*/
      m_w_init.set(&m_w); 
      v_i_init.set(&v_i); 
    }  
  }
  virtual void check_for_reg_L2init(const AzpLmParam &p) const {
    const char *eyec = "AzpLm::check_for_reg_L2init"; 
    if (p.reg_L2init > 0 && p.reg_L2init < AzpLmParam_reg_infinity) {
      m_w_init.shape_chk_tmpl(&m_w, eyec, "m_w_init"); 
      v_i_init.shape_chk_tmpl(&v_i, eyec, "v_i_init"); 
    }
  }  
  
  /*------------------------------------------------------------*/      
  virtual double regloss(const AzpLmParam &p, double *iniloss=NULL) const {
    if (p.dont_update()) return 0; 
    if (p.reg_L2init > 0) {
      int cnum0 = m_w_init.colNum();  
      AzPmat m(m_w_init.rowNum(), m_w_init.colNum()); 
      m.set(0, cnum0, &m_w, 0, ws); 
      m.add(&m_w_init, -1); 
      double ilossi = 0, ilossw = m.squareSum()*0.5*p.reg_L2init; 
      double loss = ilossw; 
      if (p.reg_L2 > 0) loss += m_w.squareSum()*0.5*p.reg_L2; 
      if (p.do_reg_intercept) {
        m.set(0, cnum0, &v_i, 0); 
        m.add(&v_i_init, -1); 
        ilossi = m.squareSum()*0.5*p.reg_L2init; 
        loss += ilossi; 
        if (p.reg_L2 > 0) loss += v_i.squareSum()*0.5*p.reg_L2; 
      }
      if (iniloss != NULL) *iniloss += (ilossw + ilossi); 
      return loss; 
    }  
    if (p.reg_L2const > 0) {
      return 0; 
    }
    if (p.reg_L1L2 > 0) {
      AzPmatApp app; 
      double loss = app.l1l2sum(&m_w, (AzFloat)p.reg_L1L2_delta) + app.l1l2sum(&v_i, (AzFloat)p.reg_L1L2_delta); 
      return loss * p.reg_L1L2;    
    }
    if (p.reg_L2 > 0) {
      double loss = m_w.squareSum() *ws*ws; 
      if (p.do_reg_intercept) {
        loss += v_i.squareSum(); 
      }
      loss *= 0.5 * p.reg_L2; 
      return loss; 
    }
    return 0; 
  }
  
  /*------------------------------------------------------------*/      
  inline virtual void apply(const AzPmat *m_x, AzPmat *m_out) /*one const one*/ {
    const AzPmat *mw = (doing_partial()) ? &m_w_part : &m_w; 
    const AzPmat *vi = (doing_partial()) ? &v_i_part : &v_i; 
    m_out->prod(mw, m_x, true, false); 
    if (ws != 1) m_out->multiply(ws); 
    gen_one(m_x, &m_one);     
    m_out->add_prod(vi, &m_one, true, false);    
  }

  /*------------------------------------------------------------*/      
  /* sparse */
  inline virtual void apply(const AzPmatSpa *m_x, AzPmat *m_out) /*one const one*/ {
    const AzPmat *mw = (doing_partial()) ? &m_w_part : &m_w; 
    const AzPmat *vi = (doing_partial()) ? &v_i_part : &v_i; 
    AzPs::prod(m_out, mw, m_x, true, false); 
    if (ws != 1) m_out->multiply(ws);
    gen_one(m_x->colNum(), &m_one); 
    m_out->add_prod(vi, &m_one, true, false);      
  }
  
  /*------------------------------------------------------------*/   
  inline virtual void unapply(AzPmat *m_d, const AzPmat *m_lossd) const {
    const AzPmat *mw = (doing_partial()) ? &m_w_part : &m_w; 
    m_d->prod(mw, m_lossd, false, false);     
    if (ws != 1) m_d->multiply(ws);    
  }
  
  /*------------------------------------------------------------*/      
  virtual int get_dim() const { return m_w.rowNum(); }
  virtual int nodeNum() const { return m_w.colNum(); }

  /*------------------------------------------------------------*/  
  virtual void initWeights(const AzpLmParam &p) {  
    reset_ws(); 
    if (p.initw_max == 0) {
      /*---  initialize by zero  ---*/
      m_w.zeroOut(); 
      v_i.zeroOut(); 
      return; 
    }
    AzDmat md_w(m_w.rowNum(), m_w.colNum()); 
    AzRandGen rg; 
    if (p.initw_max == AzpLmParam_iw_auto) { /* [He et al.,15] */
      double val = sqrt(2/(double)m_w.rowNum()); 
      rg.gaussian(val, &md_w); 
    }
    else if (p.do_iw_uniform) rg.uniform(p.initw_max, &md_w); 
    else                      rg.gaussian(p.initw_max, &md_w); 
    if (p.do_initw_nonega) {
      md_w.abs(); 
    }
    if (p.initw_rownorm > 0) {
      AzDmat md_tran; md_w.transpose_to(&md_tran); 
      md_tran.normalize(); 
      md_tran.transpose_to(&md_w);    
      if (p.initw_rownorm != 1) md_w.multiply(p.initw_rownorm); 
    }  
    if (p.initw_nzmax > 0 && p.initw_nzmax < m_w.rowNum()) {
      AzTimeLog::print("#nz = ", p.initw_nzmax, log_out); 
      AzDmat md_w2(md_w.rowNum(), md_w.colNum()); 
      int col; 
      for (col = 0; col < md_w.colNum(); ++col) {
        AzIntArr ia; 
        AzTools::sample(md_w.rowNum(), p.initw_nzmax, &ia); 
        int ix; 
        for (ix = 0; ix < ia.size(); ++ix) {
          int row = ia.get(ix); 
          md_w2.set(row, col, md_w.get(row, col)); 
        }
      }
      m_w.set(&md_w2);  
    }
    else {
      m_w.set(&md_w); 
    }
    if (p.initint > 0) v_i.set(p.initint);    
    if (p.do_no_intercept) v_i.zeroOut(); 
  }
  /*------------------------------------------------------------*/
  virtual void initWeights(const AzPmat &_m_w, 
                           const AzPmat &_v_i) {
    reset_ws(); 
    m_w.set(&_m_w); 
    v_i.set(&_v_i);     
  }  

  /*---  for tied weights of autoencoder  ---*/
  /*------------------------------------------------------------*/  
  virtual void transposeWeights_from(const AzpLm *inp) {
    ws = inp->ws; 
    m_w.transpose_from(&inp->m_w); 
  }  
  
  /*------------------------------------------------------------*/      
  inline virtual int classNum() const { 
    return (doing_partial()) ? m_w_part.colNum() : m_w.colNum(); 
  }

  /*------------------------------------------------------------*/      
  inline void flush_ws() {
    if (ws != 1) {
      m_w.multiply(ws); 
      ws = 1; 
    }
  }

  /*------------------------------------------------------------*/    
  virtual void show_stat(AzBytArr &s) const {
    double n22 = m_w.squareSum()*ws*ws / (double)m_w.colNum(); 
    double ww = m_w.absSum() * ws / (double)m_w.size(); 
    double wavg = m_w.sum() * ws / (double)m_w.size(); 
    double ii = v_i.absSum() / (double)v_i.size(); 
    double iavg = v_i.sum() / (double)v_i.size(); 
    int nz = m_w.nz(); 
    s << "dim=" << m_w.rowNum() << "x" << m_w.colNum() << ","; 
    s << "n22," << n22 << ","; 
    s << "nz," << nz << "," << (double)nz/(double)m_w.size() << ","; 
    s << "absavgw," << ww << ","; 
    s << "avgw," << wavg << ","; 
    s << "absavgi," << ii << ","; 
    s << "avgi," << iavg << ",";
    double wavg2 = n22/(double)m_w.rowNum(), wsdev = sqrt(wavg2 - wavg*wavg); 
    s << "sdevw," << wsdev << ","; 
  }
  
protected:  
  void checkIndex(int idx, const char *msg) const {
    AzX::throw_if((idx < 0 || idx >= classNum()), "AzpLm::checkIndex", msg, "index is out of range"); 
  }      
  inline void reset_ws() { ws = 1; }   
  inline void check_ws(const char *msg) const {
    AzX::throw_if((ws != 1), "AzpLm::check_ws", "weight scale must have been flushed before calling:", msg); 
  }
  inline void gen_one(const AzPmat *m_x, AzPmat *m_one) const {
    gen_one(m_x->colNum(), m_one); 
  }
  inline void gen_one(int cnum, AzPmat *m_one) const {
    if (m_one->rowNum() == 1 && m_one->colNum() == cnum) return; 
    m_one->reform_noinit(1, cnum); 
    m_one->set(1); 
  }  
}; 

/********************************************************************/
class AzpLm_Untrainable : public virtual /* implements */ AzpLm {
public:
  AzpLm_Untrainable() {}
  AzpLm_Untrainable(const AzDmat *_m_w, const AzDmat *_v_i) {
    reset(_m_w, _v_i); 
  }
  AzpLm_Untrainable(const AzPmat *_m_w, const AzPmat *_v_i) {
    reset(_m_w, _v_i); 
  }
  template <class M>
  void reset(const M *m_init_w, const M *m_init_i) {
    AzX::throw_if((m_init_w->colNum() != m_init_i->colNum() || m_init_i->rowNum() != 1), 
                  "AzpLm_Untrainable::reset(w,i)", "Invalid input"); 
    reset_ws(); 
    m_w.set(m_init_w); 
    v_i.set(m_init_i); 
  }    
  virtual const char *description() const {
    return "Untrainable"; 
  }
  virtual void updateDelta(int d_num, const AzpLmParam &p, const AzPmatSpa *m_x, const AzPmat *m_deriv, const void *pp=NULL) {
    untrainable("updateDelta(spa)"); 
  }
  virtual void updateDelta(int d_num, const AzpLmParam &p, const AzPmat *m_x, const AzPmat *m_deriv, const void *pp=NULL) {
    untrainable("updateDelta(dense)"); 
  }                           
  virtual void setup_for_reg_L2init(const AzpLmParam &p) { untrainable("setup_for_reg_L2init"); }
  virtual void check_for_reg_L2init(const AzpLmParam &p) const { untrainable("check_for_reg_L2init"); }   
  virtual void resetWork() {}
  virtual void reformWork() {}  
  virtual void clearTemp() {}
  virtual void initWeights(const AzpLmParam &p) {
    untrainable("initWeights"); 
  }
protected: 
  inline virtual void untrainable(const char *msg) const {
    AzX::throw_if(true, "AzLmod_Untrainable", msg, "This class doesn't support tarining."); 
  }
}; 
#endif 
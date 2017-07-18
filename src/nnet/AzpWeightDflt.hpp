/* * * * *
 *  AzpWeightDflt.hpp
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

#ifndef _AZP_WEIGHT_DFLT_HPP_
#define _AZP_WEIGHT_DFLT_HPP_

#include "AzpWeight_.hpp"
#include "AzpLmSgd.hpp"

/**********************************************************/  
class AzpWeightDflt : public virtual AzpWeight_ {
protected: 
  /* Use of arrays was originally for local weights, which is no longer supported. */
  /* Arrays need to be kept for file compatibility.                                */
  AzDataArr<AzpLmSgd> lmods_sgd; 
  AzBaseArr<AzpLm *> lms; 
  AzBaseArr<const AzpLm *> lms_const;   
  AzpLm_Untrainable dummy; 
  AzpLmParam p; 
  AzpLmSgd_Param ps, org_ps; 
  bool do_thru; /* Don't apply weights.  Set input to output as it is. */
  int thru_dim; 
  
  virtual void reset_lms() {
    lms_const.realloc(lmods_sgd.size(), "AzpWeightDflt::reset_lms", "lms_const"); 
    lms.realloc(lmods_sgd.size(), "AzpWeightDflt::reset_lms", "lms");    
    for (int lx = 0; lx < lmods_sgd.size(); ++lx) {
      lms_const(lx, lmods_sgd[lx]); 
      lms(lx, lmods_sgd(lx)); 
    }
  }

  virtual AzpLm *lm() { return lms[0]; }  
  virtual const AzpLm *lm_const() const { return lms_const[0]; }
  virtual const AzpLm *templ() const {
/*    AzX::throw_if((lms_const.size() <= 0), "AzpWeightDflt::templ", "No tempalte"); */
    if (lms_const.size() <= 0) return &dummy;  /* 11/18/2015: for renet side */
    return lms_const[0]; 
  }
  static const int version = 0; 
/*  static const int reserved_len = 64;  */
  static const int reserved_len = 59;  /* 04/13/2015: do_thru(1) and thru_dim(4) */
  virtual void read_common(AzFile *file) {  
    AzTools::read_header(file, reserved_len);  
    do_thru = file->readBool(); /* 04/13/2015 */
    thru_dim = file->readInt(); /* 04/13/2015 */
    bool dummy = file->readBool();  /* for compatibility 6/9/2017 */
  }    
  virtual void write_common(AzFile *file) const {
    AzTools::write_header(file, version, reserved_len); 
    file->writeBool(do_thru); /* 04/13/2015 */
    file->writeInt(thru_dim); /* 04/13/2015 */
    file->writeBool(false);  /* for compatibility 6/9/2017 */
  }  

public: 
  AzpWeightDflt() : do_thru(false), thru_dim(-1) {}  
  
  virtual bool are_weights_fixed() const {
    return p.dont_update(); 
  }
protected: 
  virtual int _reset_common(int loc_num, /* not used */
             int w_num, int inp_dim, bool is_spa, bool is_var) {
    const char *eyec = "AzpWeightDflt::reset_common"; 
    if (do_thru) {
      AzX::no_support(is_spa, eyec, "The thru option with sparse input"); 
      force_thru(inp_dim); 
      AzX::throw_if((w_num != thru_dim), eyec, "With the thru option, the dimensionality of input and output must be the same"); 
      return -1; 
    }
    return 1; 
  }
public:  
  virtual void reset(int loc_num, /* not used */
             int w_num, int inp_dim, bool is_spa, bool is_var) {
    int sz = _reset_common(loc_num, w_num, inp_dim, is_spa, is_var); 
    if (do_thru) return; 
    lmods_sgd.reset(sz); 
    for (int lx = 0; lx < lmods_sgd.size(); ++lx) ((AzpLm *)lmods_sgd(lx))->reset(w_num, p, inp_dim);
    reset_lms();    
  }
  void destroy() { lmods_sgd.reset(); lms.free(); lms_const.free(); }  
  virtual void force_thru(int inp_dim) {
    do_thru = true; 
    lmods_sgd.reset(); 
    reset_lms(); 
    thru_dim = inp_dim; 
  } 
  virtual void setup_for_reg_L2init() {
    for (int lx = 0; lx < lms.size(); ++lx) lms[lx]->setup_for_reg_L2init(p); 
  }
  virtual void check_for_reg_L2init() const {
    for (int lx = 0; lx < lms.size(); ++lx) lms[lx]->check_for_reg_L2init(p); 
  }    
  virtual AzpWeight_ *clone() const {
    AzpWeightDflt *o = new AzpWeightDflt();    
    o->lmods_sgd.reset(&lmods_sgd); 
    o->reset_lms(); 
    o->p = p; 
    o->ps = ps; 
    o->org_ps = org_ps; 
    o->do_thru = do_thru; 
    o->thru_dim = thru_dim; 
    return o; 
  }   
  virtual void read(AzFile *file) {  
    read_common(file); 
    lmods_sgd.read(file); reset_lms(); 
  }
  virtual void write(AzFile *file) const {
    write_common(file); 
    lmods_sgd.write(file); 
  }      
  
  /*------------------------------------------------------------*/   
  #define kw_do_thru "ThruWeights"
  /*------------------------------------------------------------*/ 
  virtual void _resetParam_common(AzParam &azp, const char *pfx, bool is_warmstart) {
    if (!is_warmstart) {
      azp.reset_prefix(pfx); 
      azp.swOn(&do_thru,  kw_do_thru); 
      azp.reset_prefix(""); 
    }    
    p.resetParam(azp, pfx, is_warmstart); 
  }  
  virtual void _resetParam(AzParam &azp, const char *pfx, bool is_warmstart=false) {
    _resetParam_common(azp, pfx, is_warmstart); 
    ps.resetParam(azp, pfx, is_warmstart); 
    org_ps = ps; /* save the original parameters */
  }
  virtual void _printParam_common(const AzOut &out, const char *pfx) const {
    AzPrint o(out, pfx); 
    o.printSw(kw_do_thru, do_thru); 
    if (!do_thru) {
      p.printParam(out, pfx); 
    }
  }  
  virtual void printParam(const AzOut &out, const AzPfx &pfx) const {
    _printParam_common(out, pfx.pfx());   
    if (!do_thru) ps.printParam(out, pfx.pfx()); 
  }
  virtual void printHelp(AzHelp &h) const { 
    p.printHelp(h); 
    ps.printHelp(h); 
  }
  virtual void resetParam(AzParam &azp, const AzPfx &pfx, bool is_warmstart=false) {
    for (int px=0; px<pfx.size(); ++px) _resetParam(azp, pfx[px], is_warmstart); 
    if (!do_thru) {
      p.checkParam(pfx.pfx()); 
      ps.checkParam(pfx.pfx()); 
    }
  }  

  virtual void reset_do_no_intercept(bool flag) {
    p.do_no_intercept = flag; 
  }  
  virtual void multiply_to_stepsize(double factor, const AzOut *out=NULL) {
    ps.eta = org_ps.eta * factor; 
    if (out != NULL) {
      AzBytArr s("eta="); s << ps.eta; 
      AzPrint::writeln(*out, s); 
    }
  }
  virtual void set_momentum(double newmom, const AzOut *out=NULL) {
    ps.momentum = newmom; 
    if (out != NULL) {
      AzBytArr s("momentum="); s << ps.momentum; 
      AzPrint::writeln(*out, s); 
    }
  }
  
  virtual void force_no_intercept() {
    p.do_no_intercept = true; 
    p.do_reg_intercept = false; 
  }
  virtual void force_no_reg() { p.force_no_reg(); }
 
  virtual void initWeights() {
    for (int lx = 0; lx < lms.size(); ++lx) lms[lx]->initWeights(p); 
  }
  virtual void initWeights(const AzpLm *inp, double coeff) {
    if (do_thru) return; 
    check_localw("initWeights(lmod,coeff)");    
    lm()->reset(p, inp, coeff); 
  } 
  virtual void upward(bool is_test, const AzPmat *m_x, AzPmat *m_out) {
    if (do_thru) {
      m_out->set(m_x); 
      return;      
    }
    m_out->reform_noinit(classNum(), m_x->colNum()); 
    lm()->apply(m_x, m_out); 
  }
  virtual void upward(bool is_test, const AzPmatSpa *m_x, AzPmat *m_out) {
    AzX::no_support(do_thru, "AzpWeightDflt::upward(spa)", "The thru option with sparse input"); 
    check_localw("upward with sparse input"); 
    lm()->apply(m_x, m_out);
  }
  virtual void downward(const AzPmat *m_lossd, AzPmat *m_d) const {
    if (do_thru) {
      m_d->set(m_lossd); 
      return;       
    }
    m_d->reform_noinit(get_dim(), m_lossd->colNum()); 
    lm_const()->unapply(m_d, m_lossd); 
  }  
  virtual void updateDelta(int d_num, const AzPmat *m_x, const AzPmat *m_lossd) {             
    if (do_thru) return; 
    if (lmods_sgd.size() == 1) lmods_sgd(0)->updateDelta(d_num, p, m_x, m_lossd, &ps); 
    else                       lm()->updateDelta(d_num, p, m_x, m_lossd);           
  }
  virtual void updateDelta(int d_num, const AzPmatSpa *m_x, const AzPmat *m_lossd) {             
    if (do_thru) return; 
    check_localw("updateDelta with sparse input");    
    if (lmods_sgd.size() == 1) lmods_sgd(0)->updateDelta(d_num, p, m_x, m_lossd, &ps); 
    else                       lm()->updateDelta(d_num, p, m_x, m_lossd); 
  }
  virtual void flushDelta() {
    for (int lx = 0; lx < lmods_sgd.size(); ++lx) lmods_sgd(lx)->flushDelta(p, ps); 
  }
  virtual void clearTemp() {
    for (int lx = 0; lx < lms.size(); ++lx) lms[lx]->clearTemp(); 
  }
  virtual void end_of_epoch() {
    for (int lx = 0; lx < lms.size(); ++lx) lms[lx]->end_of_epoch(p); 
  }

  /*------------------------------------------------------------*/    
  virtual double regloss(double *iniloss) const {
    double loss = 0; 
    for (int lx = 0; lx < lms_const.size(); ++lx) loss += lms_const[lx]->regloss(p, iniloss); 
    return loss; 
  }
  virtual int num_weights() const {
    int num = 0; 
    for (int lx = 0; lx < lms_const.size(); ++lx) {   
      num += lms_const[lx]->get_dim() * lms_const[lx]->classNum(); 
    }
    return num; 
  }  
  virtual void show_stat(AzBytArr &s) const {
    if (lms_const.size() <= 0) return; 
    s.c("cnv/fc:"); 
    lm_const()->show_stat(s); 
    if (p.do_showwi) {
      lm_const()->weights()->dump(log_out, "w"); 
      lm_const()->intercepts()->dump(log_out, "i"); 
    }  
  }
  virtual int get_dim() const { 
    if (do_thru) return thru_dim; 
    return templ()->get_dim(); 
  }
  virtual int classNum() const { 
    if (do_thru) return thru_dim; 
    return templ()->classNum(); 
  }

  virtual const AzpLm *linmod() const {
    check_thru("linmod"); 
    check_localw("linmod"); 
    return lm_const(); 
  }
  virtual AzpLm *linmod_u() {
    check_thru("linmod_u");    
    check_localw("linmod_u"); 
    return lm(); 
  }  
  
protected:    
  virtual void check_localw(const char *msg) const {}
  virtual void check_thru(const char *msg) const {
    AzX::throw_if(do_thru, "AzpWeightDflt::check_thru", msg, "cannot be used with the thru option"); 
  }
}; 

#endif 
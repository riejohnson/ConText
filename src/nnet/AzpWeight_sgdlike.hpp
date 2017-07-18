/* * * * *
 *  AzpWeight_sgdlike.hpp
 *  Copyright (C) 2015-2017 Rie Johnson
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

#ifndef _AZP_WEIGHT_SGDLIKE_HPP_
#define _AZP_WEIGHT_SGDLIKE_HPP_

#include "AzpWeightDflt.hpp"

/**********************************************************/  
template <class Opt, class OptParam>
class AzpWeight_sgdlike : public virtual AzpWeightDflt {
protected: 
  AzDataArr<Opt> lmods_opt; 
  OptParam pa; 
  
  virtual void reset_lms() {
    lms_const.realloc(lmods_opt.size(), "AzpWeight_sgdlike::reset_lms", "lms_const"); 
    lms.realloc(lmods_opt.size(), "AzpWeight_sgdlike::reset_lms", "lms");    
    for (int lx = 0; lx < lmods_opt.size(); ++lx) {
      lms_const(lx, lmods_opt[lx]); 
      lms(lx, lmods_opt(lx)); 
    }
  }

public: 
  AzpWeight_sgdlike() {}
  virtual void reset(int loc_num, /* not used */
             int w_num, int inp_dim, bool is_spa, bool is_var) {
    int sz = _reset_common(loc_num, w_num, inp_dim, is_spa, is_var); 
    if (do_thru) return; 
    lmods_sgd.reset(); 
    lmods_opt.reset(sz); 
    for (int lx = 0; lx < lmods_opt.size(); ++lx) ((AzpLm *)lmods_opt(lx))->reset(w_num, p, inp_dim);
    reset_lms();    
  }    
  virtual AzpWeight_ *clone() const {
    AzpWeight_sgdlike *o = new AzpWeight_sgdlike();    
    o->lmods_opt.reset(&lmods_opt); 
    o->reset_lms(); 
    o->p = p; 
    o->pa = pa; 
    o->do_thru = do_thru; 
    o->thru_dim = thru_dim;     
    return o; 
  }  
  virtual void read(AzFile *file) {  
    read_common(file); 
    lmods_opt.read(file); /* AzpLm::read */ reset_lms(); 
  }  
  virtual void write(AzFile *file) const {
    write_common(file); 
    lmods_opt.write(file);  /* AzpLm::write */
  }   
  
  /*------------------------------------------------------------*/ 
  virtual void _resetParam(AzParam &azp, const char *pfx, bool is_warmstart=false) {
    _resetParam_common(azp, pfx, is_warmstart); 
    pa.resetParam(azp, pfx, is_warmstart); 
  }   
  virtual void printParam(const AzOut &out, const AzPfx &pfx) const {
    _printParam_common(out, pfx.pfx()); 
    pa.printParam(out, pfx.pfx()); 
  } 
  virtual void resetParam(AzParam &azp, const AzPfx &pfx, bool is_warmstart=false) {
    for (int px=0; px<pfx.size(); ++px) _resetParam(azp, pfx[px], is_warmstart); 
    if (!do_thru) {
      p.checkParam(pfx.pfx()); 
      pa.checkParam(pfx.pfx()); 
    }
  }  
  
  /*---  do something about parameters  ---*/
  virtual void multiply_to_stepsize(double factor, const AzOut *out=NULL) {
    pa.coeff = factor; 
  }  
  virtual void set_momentum(double newmom, const AzOut *out=NULL) {}

  virtual void flushDelta() {
    for (int lx = 0; lx < lmods_opt.size(); ++lx) lmods_opt(lx)->flushDelta(p, pa); 
  }
}; 

#endif 
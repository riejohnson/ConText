/* * * * *
 *  AzpDropoutDflt.hpp
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

#ifndef _AZP_DROPOUT_DFLT_HPP_
#define _AZP_DROPOUT_DFLT_HPP_

#include "AzpDropout_.hpp"
#include "AzPmatApp.hpp"

/*------------------------------------------------------------*/
class AzpDropoutDflt : public virtual /* implements */ AzpDropout_ {
protected: 
  class AzpDropoutDflt_Param {
  protected: 
    static const int version = 0; 
    static const int reserved_len = 64;  
  public: 
    double dropout; 
    AzpDropoutDflt_Param() : dropout(-1) {}

    #define kw_dropout "dropout="
    void resetParam(AzParam &azp, const char *pfx, bool for_testonly) {
      azp.reset_prefix(pfx); 
      if (!for_testonly) azp.vFloat(kw_dropout, &dropout); 
      azp.reset_prefix(); 
    }
    void checkParam(const char *pfx) {
      AzXi::invalid_input((dropout >= 1), "AzpDropoutDflt::resetParam", kw_dropout, pfx);  
    }
    virtual void printParam(const AzOut &out, const char *pfx) const {
      AzPrint o(out, pfx); 
      o.printV(kw_dropout, dropout); 
      o.printEnd(); 
    }    
    virtual void printHelp(AzHelp &h) const {
      h.item(kw_dropout, "Dropout value.", "No dropout"); 
    }
    void write(AzFile *file) const {
      AzTools::write_header(file, version, reserved_len); 
      file->writeDouble(dropout); 
    }
    void read(AzFile *file) {
      AzTools::read_header(file, reserved_len); 
      dropout = file->readDouble();  
    }    
  }; 
protected: 
  AzPrng rng; 
  AzpDropoutDflt_Param p; 
  AzPmat m_mask; 
  static const int version = 0; 
  static const int reserved_len = 64;   
public:
  AzpDropoutDflt() {}
  virtual void resetParam(AzParam &azp, const AzPfx &pfx, bool is_warmstart) {
    for (int px=0; px<pfx.size(); ++px) p.resetParam(azp, pfx[px], is_warmstart); 
    p.checkParam(pfx.pfx()); 
  }  
  virtual void printParam(const AzOut &out, const AzPfx &pfx) const { p.printParam(out, pfx.pfx()); }
  virtual void printHelp(AzHelp &h) const { p.printHelp(h); }  
  virtual void read(AzFile *file) { 
    AzTools::read_header(file, reserved_len); 
    p.read(file); 
  }
  virtual void write(AzFile *file) const { 
    AzTools::write_header(file, version, reserved_len); 
    p.write(file); 
  }
      
  virtual void reset(const AzOut &out) {
    if (p.dropout <= 0) return; 
    int seed = rand();  
    rng.reset_seed(seed); 
  }
  virtual AzpDropout_ *clone() const {
    AzpDropoutDflt *o = new AzpDropoutDflt(); 
    o->p = p; 
    o->m_mask.set(&m_mask);
    return o; 
  }
  virtual bool is_active() const { return (p.dropout > 0); }
  virtual void upward(bool is_test, AzPmat *m) {
    if (p.dropout <= 0) return; 
    if (is_test) {
      m->multiply(1-p.dropout); 
    }
    else {
      m_mask.reform_noinit(m->rowNum(), m->colNum()); 
      rng.uniform_01(&m_mask);  /* [0,1] */
      m_mask.mark_gt((AzFloat)p.dropout);  /* ([i,j] > dropout) ? 1 : 0 */
      m->elm_multi(&m_mask); 
    }
  }
  virtual void downward(AzPmat *m) {
    if (p.dropout <= 0) return; 
    m_mask.shape_chk_tmpl(m, "AzpDropoutDflt::downward", "m_mask"); 
    m->elm_multi(&m_mask); 
  }
  virtual AzPrng &ref_rng() { return rng; }
}; 
#endif 
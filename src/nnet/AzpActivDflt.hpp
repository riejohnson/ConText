/* * * * *
 *  AzpActivDflt.hpp
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

#ifndef _AZP_ACTIV_DFLT_HPP_
#define _AZP_ACTIV_DFLT_HPP_

#include "AzpActiv_.hpp"
#include "AzPmatApp.hpp"
#include "AzTools.hpp"

#define AzpActivDflt_None 'N'
#define AzpActivDflt_Log 'L'
#define AzpActivDflt_Rect 'R'
#define AzpActivDflt_0cut '0'
#define AzpActivDflt_Softplus 'S'
#define AzpActivDflt_Tanh 'T'

/*------------------------------------------------------------*/
class AzpActivDflt : public virtual /* implements */ AzpActiv_ {
protected: 
  class AzpActivDflt_Param {
  protected: 
    static const int version = 0; 
/*    static const int reserved_len = 64;  */
    static const int reserved_len = 56; /* 2/21/2017: for slope */
  public: 
    AzBytArr s_activ_typ; 
    AzByte typ; 
    double trunc, slope;
    bool do_stat; 
  
    AzpActivDflt_Param() : s_activ_typ("None"), typ(AzpActivDflt_None), do_stat(false), trunc(-1), slope(-1) {}

    void set_default_type(const char *type) {
      s_activ_typ.reset(type);  
    }
    void setup_no_activation() {
      s_activ_typ.reset("None"); 
      typ = 'N'; 
      trunc = -1; 
    }
  
    #define kw_activ_typ "activ_type="
    #define kw_do_stat "ActivStat"
    #define kw_trunc "truncate="
    #define kw_slope "activ_slope="
    
    virtual void resetParam(AzParam &azp, const char *pfx, bool is_warmstart) {
      azp.reset_prefix(pfx); 
      if (!is_warmstart) {
        azp.vStr(kw_activ_typ, &s_activ_typ); 
        if (s_activ_typ.length() > 0) typ = *s_activ_typ.point(); 
        azp.vFloat(kw_trunc, &trunc); 
        azp.vFloat(kw_slope, &slope);         
      }
      azp.swOn(&do_stat, kw_do_stat);       
      azp.reset_prefix(); 
    }
    virtual void checkParam(const char *pfx) const {
      const char *eyec = "AzpActivDflt_Param::checkParam"; 
      AzXi::throw_if_empty(s_activ_typ, eyec, kw_activ_typ, pfx); 
      AzXi::invalid_input((typ != AzpActivDflt_None && typ != AzpActivDflt_Log && 
        typ != AzpActivDflt_Rect && typ != AzpActivDflt_0cut && 
        typ != AzpActivDflt_Softplus && typ != AzpActivDflt_Tanh), 
        eyec, kw_activ_typ, pfx); 
    }
    virtual void printParam(const AzOut &out, const char *pfx) const {
      AzPrint o(out); 
      o.reset_prefix(pfx); 
      o.printV(kw_activ_typ, s_activ_typ);   
      o.printSw(kw_do_stat, do_stat);    
      o.printV(kw_trunc, trunc);
      if (typ == AzpActivDflt_Rect) o.printV(kw_slope, slope); 
      o.printEnd(); 
    } 
    virtual void printHelp(AzHelp &h) const {
      h.item(kw_activ_typ, "Non-linear activation type.  \"None\" | \"Log\" (sigmoid) | \"Rect\" (rectifier) | \"Softplus\" | \"Tanh\""); 
      /* kw_do_stat kw_trunc */
    }
    virtual void write(AzFile *file) const {
      AzTools::write_header(file, version, reserved_len);    
      file->writeDouble(slope); 
      s_activ_typ.write(file); 
      file->writeByte(typ); 
      file->writeDouble(trunc); 
    }
    virtual void read(AzFile *file) {
      AzTools::read_header(file, reserved_len);
      slope = file->readDouble(); 
      s_activ_typ.read(file); 
      typ = file->readByte(); 
      trunc = file->readDouble(); 
    }
  }; 

protected:
  AzpActivDflt_Param p;   
  AzPmatApp app; 
  AzPmat m_drv;  /* derivative */
  
  /*---  to generate histogram  ---*/
  AzDvect v_border, v_pop, v_pop_last; 

  virtual void copy_from(const AzpActivDflt *i) {
    p = i->p; 
    m_drv.set(&i->m_drv); 
    v_border.set(&i->v_border); 
    v_pop.set(&i->v_pop); 
    v_pop_last.set(&i->v_pop_last); 
  }   
  
  static const int version = 0; 
  static const int reserved_len = 64;  
public:  
  virtual void resetParam(AzParam &azp, const AzPfx &pfx, bool is_warmstart=false) {
    for (int px=0; px<pfx.size(); ++px) p.resetParam(azp, pfx[px], is_warmstart); 
    p.checkParam(pfx.pfx()); 
  }
  virtual void printParam(const AzOut &out, const AzPfx &pfx) const { p.printParam(out, pfx.pfx()); }
  virtual void printHelp(AzHelp &h) const { p.printHelp(h); }  
  virtual void write(AzFile *file) const { 
    p.write(file); 
    AzTools::write_header(file, version, reserved_len);      
  }
  virtual void read(AzFile *file) { 
    p.read(file); 
    AzTools::read_header(file, reserved_len);    
  }
  
  virtual AzpActiv_ *clone() const {
    AzpActivDflt *o = new AzpActivDflt(); 
    o->copy_from(this); 
    return o; 
  }
  virtual void reset(const AzOut &out) {}
  virtual void setup_no_activation() {
    p.setup_no_activation(); 
  }
  virtual void set_default_type(const char *type) {
    p.set_default_type(type);  
  }
  virtual void upward(bool is_test, AzPmat *m) {
    if (p.typ == AzpActivDflt_None) return; 
    if (p.do_stat) count(m);   
    if (is_test) {
      activate(m); 
      if (p.trunc > 0) app.truncate(m, p.trunc, NULL);     
    }
    else {
      m_drv.reform_noinit(m->rowNum(), m->colNum()); 
      activate(m, &m_drv); 
      if (p.trunc > 0) app.truncate(m, p.trunc, &m_drv); 
    }
  }
  virtual void upward2(AzPmat *m) {
    /* regard m_drv (derivatives) as a constant mask */
    if (p.typ == AzpActivDflt_None) return;     
    m_drv.shape_chk_tmpl(m, "AzpActivDflt::upward2", "m_drv"); 
    m->elm_multi(&m_drv); 
  }
  virtual void downward(AzPmat *m) {
    if (p.typ == AzpActivDflt_None) return;     
    m_drv.shape_chk_tmpl(m, "AzpActivDflt::downward", "m_drv"); 
    m->elm_multi(&m_drv); 
  }
  virtual void release_ld() { 
    if (p.typ == AzpActivDflt_None) return; 
    m_drv.destroy(); 
  }
  
  virtual void show_stat(AzBytArr &s) const {
    if (p.typ == AzpActivDflt_None) return;     
    if (!p.do_stat) return; 
    AzDvect my_v_pop(&v_pop_last); 
    my_v_pop.normalize1(); 
    s.nl(); 
    double accum = 0; 
    int ix; 
    for (ix = 0; ix < my_v_pop.rowNum(); ++ix) {
      accum += my_v_pop.get(ix); 
      s.c("b"); s.cn(v_border.get(ix)); s.c("="); s.cn(my_v_pop.get(ix),3); s.c("("); s.cn(accum,3); s.c("),"); 
    }
  }

  void end_of_epoch() {
    if (p.typ == AzpActivDflt_None) return; 
    v_pop_last.set(&v_pop); 
    v_pop.zeroOut(); 
  }
  
protected:
  virtual void activate(AzPmat *m, AzPmat *m_deriv=NULL) {
    if (p.typ == AzpActivDflt_None) {
      if (m_deriv != NULL) m_deriv->set(1); 
    }    
    else if (p.typ == AzpActivDflt_Rect) {
      if (p.slope > 0) app.activate_leaky_rect(m, p.slope, m_deriv); 
      else             app.activate_th(m, -1, m_deriv); 
    }
    else if (p.typ == AzpActivDflt_0cut) app.activate_th(m, -1, m_deriv); 
    else if (p.typ == AzpActivDflt_Log)  app.activate_log(m, m_deriv); 
    else if (p.typ == AzpActivDflt_Tanh) app.activate_tanh(m, m_deriv); 
    else if (p.typ == AzpActivDflt_Softplus) app.activate_softplus(m, m_deriv); 
  }
 
  /*---  to keep count  ---*/
  void init_count() {
//    const double border[] = {0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 2, 3, 9999999999, -1};  
    #define AzpActivDfltCountStop  7777
    const double border[] = {-1e+10, -10, -5, -1, -0.5, -0.2, -0.1, 0, 0.1, 0.2, 0.5, 1, 5, 10, 1e+10, AzpActivDfltCountStop}; 
    int num = 0; 
    int ix; 
    for (ix = 0; ; ++ix, ++num) if (border[ix] == AzpActivDfltCountStop) break;
    v_border.set(border, num); 
    v_pop.reform(num);   
  }
  void count(const AzPmat *m) {
    if (v_border.rowNum() == 0) init_count(); 
    AzDmat md; m->get(&md); 
    const double *border = v_border.point(); 
    int row, col; 
    for (col = 0; col < md.colNum(); ++col) {
      for (row = 0; row < md.rowNum(); ++row) {
        double val = md.get(row, col); 
        int bx; 
        for (bx = 0; bx < v_border.rowNum(); ++bx) {
          if (val <= border[bx]) {
            v_pop.add(bx, 1); 
            break; 
          }
        }
      }
    }    
  }
}; 
#endif 
/* * * * *
 *  AzpResNormDflt.hpp
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

#ifndef _AZP_RESNORM_DFLT_HPP_
#define _AZP_RESNORM_DFLT_HPP_

#include "AzpResNorm_.hpp"
#include "AzPmatApp.hpp"
#include "AzpPatchDflt.hpp"
#include "AzTools.hpp"

extern bool __doDebug; 

#define AzpResNormDflt_None 'N'
#define AzpResNormDflt_Local 'L'
#define AzpResNormDflt_Cross 'C'
#define AzpResNormDflt_Text 'T'

/**********************************************************************************/
class AzpResNormDflt : public virtual AzpResNorm_ {
protected:
  class AzpResNormDflt_Param {
  protected: 
    static const int version = 0; 
    static const int reserved_len = 64;  
  public:
    double alpha, beta; 
    double one; 
    int width; 
    AzBytArr s_resnorm_type; 
    AzByte resnorm_type; 
    bool do_force_old; 
    
    AzpResNormDflt_Param() : alpha(1), beta(0.5), width(0), do_force_old(false), 
                             s_resnorm_type("None"), resnorm_type(AzpResNormDflt_None), one(1) {}
    void write(AzFile *file) const {
      AzTools::write_header(file, version, reserved_len);    
      s_resnorm_type.write(file);  
      file->writeByte(resnorm_type);         
      file->writeDouble(alpha); 
      file->writeDouble(beta); 
      file->writeInt(width); 
      file->writeDouble(one); 
    }
    void read(AzFile *file) {
      AzTools::read_header(file, reserved_len); 
      s_resnorm_type.read(file); 
      resnorm_type = file->readByte();     
      alpha = file->readDouble(); 
      beta = file->readDouble(); 
      width = file->readInt(); 
      one = file->readDouble(); 
    }
  
    /*---------------------------------------------------------------*/
    #define kw_resnorm_type "resnorm_type="
    #define kw_resnorm_alpha "resnorm_alpha="
    #define kw_resnorm_beta "resnorm_beta="
    #define kw_resnorm_width "resnorm_width="
    #define kw_resnorm_one "resnorm_one="
    #define kw_do_force_old "OldResnorm"
    #define kw_do_disable_resnorm "DisableResnorm"
    
    void resetParam(AzParam &azp, const char *pfx, bool is_warmstart) {
      const char *eyec = "AzpResNorm::resetParam"; 
      azp.reset_prefix(pfx); 
      if (is_warmstart) {
        bool do_disable = false; 
        azp.swOn(&do_disable, kw_do_disable_resnorm); 
        if (do_disable) {
          s_resnorm_type.reset("None"); 
          resnorm_type = AzpResNormDflt_None;  
        }
      }
      else {
        azp.vStr(kw_resnorm_type, &s_resnorm_type); 
        resnorm_type = (s_resnorm_type.length() > 0) ? *s_resnorm_type.point() : AzpResNormDflt_None; 
        if (resnorm_type == AzpResNormDflt_Text) {
          alpha=1; beta=0.5; width=-1; one=1; 
        }
        else {
          azp.vFloat(kw_resnorm_alpha, &alpha); 
          azp.vFloat(kw_resnorm_beta, &beta); 
          azp.vInt(kw_resnorm_width, &width); 
          azp.vFloat(kw_resnorm_one, &one); 
        }
      }
      azp.swOn(&do_force_old, kw_do_force_old); 
      azp.reset_prefix(""); 
    }
    bool is_cross() const { return (resnorm_type == AzpResNormDflt_Cross || resnorm_type == AzpResNormDflt_Text); }
    void checkParam(const char *pfx) {
      const char *eyec = "AzpResNromDflt_Param::checkParam"; 
      AzStrPool sp("None", "Text", "Local", "Cross"); 
      AzXi::check_input(s_resnorm_type.c_str(), &sp, eyec, kw_resnorm_type); 
      if (resnorm_type != AzpResNormDflt_None) {                       
        if (!is_cross()) AzXi::throw_if_nonpositive(width, eyec, kw_resnorm_width, pfx); 
        AzXi::throw_if_nonpositive(alpha, eyec, kw_resnorm_alpha, pfx);         
        AzXi::throw_if_nonpositive(beta, eyec, kw_resnorm_beta, pfx);          
      }        
    }
    /*---------------------------------------------------------------*/    
    void printParam(const AzOut &out, const char *pfx) const {
      if (out.isNull()) return; 
      AzPrint o(out, pfx); 
      o.ppBegin("AzpResNorm", ""); 
      if (resnorm_type != AzpResNormDflt_None) {
        o.printV(kw_resnorm_type, s_resnorm_type); 
        o.printV(kw_resnorm_width, width);          
        if (resnorm_type != AzpResNormDflt_Text) {
          o.printV(kw_resnorm_alpha, alpha); 
          o.printV(kw_resnorm_beta, beta); 
          o.printV(kw_resnorm_one, one);     
        }
        o.printSw(kw_do_force_old, do_force_old); 
      }
      o.ppEnd(); 
    }
    void printHelp(AzHelp &h) const {
      h.item(kw_resnorm_type, "Response normalization type.  \"None\" | \"Text\"", "None"); 
    }
  };

  AzpResNormDflt_Param p; 
  AzPmat m_bef; /* before response normalization */
  AzPmat m_aft; /* after response normalization */
  AzPmat m_oneplussqavg; /* 1 + sum_i (v_i^2) */
  AzPmat m_tmp; 
  
  AzPintArr2 pia2_neighbors, pia2_whose_neighbor; 
  AzPintArr pia_neigh_sz; 
   
  AzPmatApp app; 
  
  static const int version = 0; 
  static const int reserved_len = 64;
public:
  AzpResNormDflt() {}
  virtual void resetParam(AzParam &azp, const AzPfx &pfx, bool is_warmstart=false) {
    for (int px=0; px<pfx.size(); ++px) p.resetParam(azp, pfx[px], is_warmstart); 
    p.checkParam(pfx.pfx()); 
  }
  virtual void printParam(const AzOut &out, const AzPfx &pfx) const { p.printParam(out, pfx.pfx()); }
  virtual void printHelp(AzHelp &h) const { p.printHelp(h); }  
  virtual void write(AzFile *file) const { 
    p.write(file); 
    AzTools::write_header(file, version, reserved_len);     
    if (p.resnorm_type == AzpResNormDflt_Local) { /* 6/25/2017 */
      pia2_neighbors.write(file);  
      pia2_whose_neighbor.write(file); 
      pia_neigh_sz.write(file); 
    }
  }
  virtual void read(AzFile *file) { 
    p.read(file); 
    AzTools::read_header(file, reserved_len); 
    if (p.resnorm_type == AzpResNormDflt_Local) { /* 6/25/2017 */
      pia2_neighbors.read(file); 
      pia2_whose_neighbor.read(file); 
      pia_neigh_sz.read(file); 
    }    
  }
  
  void reset(const AzOut &out, const AzxD *input, int cc) {
    if (p.is_cross()) {
      if (p.width <= 0) {
        p.width = cc; 
        AzBytArr s("   Setting ", kw_resnorm_width); s << p.width; 
        AzPrint::writeln(out, s); 
      }
    }
    else if (p.resnorm_type == AzpResNormDflt_Local) {
      AzX::no_support(input == NULL || !input->is_valid(), "AzpResNormDflt::reset", 
                          "local response normalization with variable-sized input"); 
      AzpPatchDflt::mapping_for_resnorm(input, p.width, &pia2_neighbors, &pia2_whose_neighbor, &pia_neigh_sz); 
    }
  }
  
  /*----------------------------------------------------------------------------------------*/
  AzpResNorm_ *clone() const {
    AzpResNormDflt *o = new AzpResNormDflt(); 
    o->p = p; 
    o->pia2_neighbors.reset(&pia2_neighbors); 
    o->pia2_whose_neighbor.reset(&pia2_whose_neighbor); 
    o->pia_neigh_sz.reset(&pia_neigh_sz); 
    
    o->m_bef.set(&m_bef); 
    o->m_aft.set(&m_aft); 
    o->m_oneplussqavg.set(&m_oneplussqavg); 
    o->m_tmp.set(&m_tmp); 
    return o; 
  }

  /*----------------------------------------------------------------------------------------*/  
  void upward(bool is_test, AzPmat *m) {
    if (is_test) _apply(m); 
    else         _upward(m); 
  }

  /*----------------------------------------------------------------------------------------*/  
  void downward(AzPmat *m) {
    if (p.resnorm_type == AzpResNormDflt_None) return; 
    AzPmat m_grad(m); 
    if (p.is_cross()) {
      app.unresnorm_cross(&m_grad, &m_bef, &m_aft, &m_oneplussqavg, 
                                &m_tmp, m, p.width, p.alpha, p.beta, p.do_force_old); 
    }
    else if (p.resnorm_type == AzpResNormDflt_Local) {
      app.unresnorm_local(&m_grad, &m_bef, &m_aft, &m_oneplussqavg, 
                          &m_tmp, m, &pia2_whose_neighbor, &pia_neigh_sz, 
                          p.alpha, p.beta); 
    }    
  }   

protected:  
  void _upward(AzPmat *m) /* row: banks, col: positions */
  {
    if (p.resnorm_type == AzpResNormDflt_None) return; 
    m_bef.set(m); 
    if (p.is_cross()) {
      app.resnorm_cross(&m_bef, p.width, p.alpha, p.beta, p.one, m, &m_oneplussqavg, p.do_force_old); 
    }
    else if (p.resnorm_type == AzpResNormDflt_Local) {
      app.resnorm_local(&m_bef, &pia2_neighbors, &pia_neigh_sz, p.alpha, p.beta, m, &m_oneplussqavg); 
    }
    m_aft.set(m);   
  }

  /*----------------------------------------------------------------------------------------*/  
  void _apply(AzPmat *m) const
  {
    if (p.resnorm_type == AzpResNormDflt_None) return; 
    AzPmat m_before(m); 
    AzPmat m_o; 
    if (p.is_cross()) {
      app.resnorm_cross(&m_before, p.width, p.alpha, p.beta, p.one, m, &m_o, p.do_force_old); 
    }
    else if (p.resnorm_type == AzpResNormDflt_Local) {
      app.resnorm_local(&m_before, &pia2_neighbors, &pia_neigh_sz, p.alpha, p.beta, m, &m_o); 
    }
  }
}; 
#endif 
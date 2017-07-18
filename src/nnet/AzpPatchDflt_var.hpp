/* * * * *
 *  AzpPatchDflt_var.hpp
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

#ifndef _AZP_PATCH_DFLT_VAR_HPP_
#define _AZP_PATCH_DFLT_VAR_HPP_

#include "AzpPatch_var_.hpp"
#include "AzPmat.hpp"
#include "AzPmatApp.hpp"
#include "AzpPatchDflt.hpp"

/*!!!!! for 1D data only !!!!!*/
class AzpPatchDflt_var : public virtual AzpPatch_var_ {
protected: 
  AzpPatchDflt_Param p; 
  
  bool do_force_shape, do_old_patch; 
  int patch_skip;
  #define kw_patch_skip "patch_skip="  
  #define kw_do_force_shape "ForceSameShape"
  #define kw_do_old_patch "OldPatch" /* do it in the old and slower way for debugging purposes */
  virtual void _resetParam(AzParam &azp, const char *pfx, bool is_warmstart) {
    azp.reset_prefix(pfx); 
    if (!is_warmstart) {
      azp.swOn(&do_force_shape, kw_do_force_shape); 
      azp.vInt(kw_patch_skip, &patch_skip); 
    }
    azp.swOn(&do_old_patch, kw_do_old_patch); 
    azp.reset_prefix(); 
  }
  virtual void _printParam(const AzOut &out, const char *pfx) const {
    AzPrint o(out); 
    o.reset_prefix(pfx); 
    o.printV(kw_patch_skip, patch_skip); 
    o.printSw(kw_do_force_shape, do_force_shape); 
    o.printSw(kw_do_old_patch, do_old_patch);  
  }  
  
  int cc; 
  AzPintArr pia_inp_dcolind; 
  AzPmatApp app; 
  AzPintArr2 pia2_inp2out; 

  /*---  for speed-up  ---*/
  AzIntArr ia_ipos, ia_opos; 
  AzPintArr2 pia2_out2inp; 
  int curr_iszmax; 
  
  static const int version = 0; 
/*  static const int reserved_len = 64;  */
/*  static const int reserved_len = 63; /* 10/26/2015: for do_force_shape */
  static const int reserved_len = 59; /* 11/07/2015: for patch_skip */
public:  
  AzpPatchDflt_var() : cc(0), do_force_shape(false), patch_skip(-1), curr_iszmax(0), do_old_patch(false) {}
  virtual void resetParam(AzParam &azp, const AzPfx &pfx, bool is_warmstart) {
    for (int px=0; px<pfx.size(); ++px) { p.resetParam(azp, pfx[px], is_warmstart); _resetParam(azp, pfx[px], is_warmstart); }
    p.checkParam(pfx.pfx()); 
  }  
  virtual void printParam(const AzOut &out, const AzPfx &pfx) const { p.printParam(out, pfx.pfx()); _printParam(out, pfx.pfx()); }
  virtual void printHelp(AzHelp &h) const {}  
  virtual void read(AzFile *file) { 
    p.read(file); 
    AzTools::read_header(file, reserved_len); 
    patch_skip = file->readInt();     
    do_force_shape = file->readBool();     
  }
  virtual void write(AzFile *file) const { 
    p.write(file); 
    AzTools::write_header(file, version, reserved_len); 
    file->writeInt(patch_skip);     
    file->writeBool(do_force_shape);        
  }

  /*------------------------------------------------------------*/      
  virtual void reset(const AzOut &out, int _cc, bool is_spa) {
    const char *eyec = "AzpPatchDflt_var::reset"; 
    AzX::no_support((is_spa && p.pch_sz > 0), eyec, "patch generation with sparse input"); 
    if (is_spa) return; /* no support for sparse input */

    AzX::no_support((p.pch_sz <= 0), eyec, "all-in-one with variable-sized input"); 
    AzX::no_support(do_old_patch && p.do_transpose, eyec, "OldPatch and Transpose"); 
    
    reset(); 
    cc = _cc; 
    AzXi::throw_if_nonpositive(cc, eyec, "#channels");     
    AzX::throw_if(p.do_transpose && cc%p.pch_sz!=0, AzInputError, eyec, 
                  "To transpose, #rows must be a multiple of patch size."); 
  }

  /*------------------------------------------------------------*/     
  virtual AzpPatch_var_ *clone() const {
    AzpPatchDflt_var *o = new AzpPatchDflt_var(); 
    o->p = p; 
    o->cc = cc; 
    o->pia_inp_dcolind.reset(&pia_inp_dcolind); 
    o->pia2_inp2out.reset(&pia2_inp2out); 
    o->ia_ipos.reset(&ia_ipos); 
    o->ia_opos.reset(&ia_opos); 
    o->pia2_out2inp.reset(&pia2_out2inp); 
    o->curr_iszmax = curr_iszmax; 
    return o; 
  } 
  
  virtual int patch_length() const { return (p.do_transpose) ? cc/p.pch_sz : cc*p.pch_sz; }
 
  /*------------------------------------------------------------*/ 
  virtual void upward(bool is_test, 
                      const AzPmatVar *m_inp, /* col: pixel */
                      AzPmatVar *m_out) { /* col: patch */
    if (do_asis()) {
      m_out->set(m_inp); 
      return;       
    }
    if (!do_old_patch) { /* do it in the new faster way */
      if (p.do_transpose) _upwardT_var(is_test, m_inp, m_out); 
      else                _upward_var(is_test, m_inp, m_out);        
    }
    else { /* keeping this path only for debugging purposes */
      AzX::throw_if((m_inp->rowNum() != cc), "AzpPatchDflt_var::upward", "#channel mismatch"); 
      if (!is_test) pia_inp_dcolind.reset(m_inp->d_index()); 

      AzPintArr2 pia2_out2inp; /* column to column */ 
      AzIntArr ia_out_dataind; 
      if (is_test) map(m_inp, &ia_out_dataind, NULL, &pia2_out2inp);
      else         map(m_inp, &ia_out_dataind, &pia2_inp2out, &pia2_out2inp);     
      AzPmat m_out_pch(cc, pia2_out2inp.size()); 
      app.add_with_map(1, m_inp->data(), &m_out_pch, cc, &pia2_out2inp);
      m_out_pch.change_dim(p.pch_sz*cc, m_out_pch.size()/p.pch_sz/cc); 
      m_out->set(&m_out_pch, &ia_out_dataind); 
    }
  }

  /*------------------------------------------------------------*/ 
  virtual void downward(const AzPmatVar *m_out, AzPmatVar *m_inp) {    
    if (do_asis()) {
      m_inp->set(m_out); 
      return;       
    }  
    if (!do_old_patch) { /* do it in the new faster way */
      if (p.do_transpose) _downwardT_var(m_out, m_inp); 
      else                _downward_var(m_out, m_inp); 
    }
    else { /* keeping this path only for debugging purposes */
      m_inp->reform(cc, &pia_inp_dcolind); 
      AzPmat m_inp_pch(m_inp->rowNum(), m_inp->colNum()); 
      app.add_with_map(1, m_out->data(), &m_inp_pch, cc, &pia2_inp2out);
      m_inp->update(&m_inp_pch); 
    }
  }  
  
protected:   
  bool do_asis() const {
    return (p.pch_sz == 1 && p.pch_step == 1 && p.padding == 0); 
  }
  int get_pskip() const {
    return (patch_skip > 0) ? patch_skip : 1; 
  }
  
  /*------------------------------------------------------------*/ 
  int how_many(int whole) const {
    if (do_force_shape) return whole; 
    return DIVUP(p.padding*2+whole-p.pch_sz, p.pch_step) + 1;  
  }
  int how_manyT(int pch_num) const {
    if (do_force_shape) return pch_num; 
    return (pch_num-1)*p.pch_step +p.pch_sz-p.padding*2;     
  }
  void map(const AzPmatVar *m_inp, 
           AzIntArr *ia_out_dataind,  /* unit: patch */
           AzPintArr2 *pia2_inp2out, /* unit: column (pixel) */
           AzPintArr2 *pia2_out2inp) /* unit: column (pixel) */ const {
    const char *eyec = "AzpPatchDflt_var::map"; 
    int pskip = get_pskip(); 
    int dx, pch_num; 
    AzDataArr<AzIntArr> aia_inp2out, aia_out2inp; 
    if (pia2_inp2out != NULL) aia_inp2out.reset(m_inp->colNum()); 
    if (pia2_out2inp != NULL) {
      pch_num = 0; 
      for (dx = 0; dx < m_inp->dataNum(); ++dx) {
        int col0, col1; m_inp->get_begin_end(dx, col0, col1); 
        pch_num += how_many(col1-col0);  
      }
      aia_out2inp.reset(pch_num*p.pch_sz); 
    }
    ia_out_dataind->reset();    

    pch_num = 0; 
    int ox = 0; 
    for (dx = 0; dx < m_inp->dataNum(); ++dx) {
      ia_out_dataind->put(pch_num); 
      int col0, col1; 
      m_inp->get_begin_end(dx, col0, col1); 
      int sz = col1 - col0; 
      int p_num = how_many(sz); 
      int pch_no; 
      for (pch_no = 0; pch_no < p_num; ++pch_no) {
        int pos0 = p.pch_step*pch_no - p.padding; 
        int pos1 = pos0 + p.pch_sz*pskip; 
        int px; 
        for (px = pos0; px < pos1; px+=pskip, ++ox) { /* position within data */
          if (px >= 0 && px < sz) {
            if (pia2_inp2out != NULL) aia_inp2out.point_u(px+col0)->put(ox); 
            if (pia2_out2inp != NULL) aia_out2inp.point_u(ox)->put(px+col0);             
          }
        }
      }
      
      pch_num += p_num; 
      ia_out_dataind->put(pch_num); /* unit is a patch, not a column/pixel */
    }       
    if (pia2_out2inp != NULL) {
      AzX::throw_if((ox != aia_out2inp.size()), eyec, "conflict in the number of output regions"); 
      AzX::throw_if((pch_num != aia_out2inp.size()/p.pch_sz), eyec, "conflict in the number of patches"); 
    }
    if (pia2_inp2out != NULL) pia2_inp2out->reset(&aia_inp2out); 
    if (pia2_out2inp != NULL) pia2_out2inp->reset(&aia_out2inp); 
  }

  /*------------------------------------------------------------*/     
  virtual void reset() {
    cc = 0; 
    pia_inp_dcolind.reset(); 
    pia2_inp2out.reset(); 
    ia_ipos.reset(); ia_opos.reset(); pia2_out2inp.reset(); 
    curr_iszmax = 0;      
  }        

  /*------------------------------------------------------------*/   
  /*------------------------------------------------------------*/   
  virtual void _upward_var_prep(const AzPmatVar &mv_inp, AzIntArr &ia_out_dind, 
                         AzIntArr &ia_ipos, AzIntArr &ia_opos,
                         int &iszmax, int &oszmax) const {
    AzX::throw_if(!mv_inp.is_consecutive(), "AzpPatchDflt_var::_upward_var_prep", 
                  "input columns of PmatVar must be consecutive");                            
    ia_ipos.reset(); ia_opos.reset(); ia_out_dind.reset(); iszmax = 0; oszmax = 0; 
    if (mv_inp.dataNum() <= 0) return; 
    int ipos = 0, opos = 0; 
    for (int dx = 0; dx < mv_inp.dataNum(); ++dx) {
      ia_out_dind.put(opos/p.pch_sz); 
      int isz = mv_inp.size(dx),  osz = how_many(isz)*p.pch_sz; 
      iszmax = MAX(iszmax, isz);  oszmax = MAX(oszmax, osz); 
      ia_ipos.put(ipos);          ia_opos.put(opos); 
      ipos += isz;                opos += osz; 
      ia_out_dind.put(opos/p.pch_sz);    
    }
    ia_ipos.put(ipos); ia_opos.put(opos); 
  }
  
  /*------------------------------------------------------------*/ 
  virtual void _upward_var(bool is_test, 
                     const AzPmatVar *mv_inp, /* col: pixel */
                     AzPmatVar *mv_out) { /* col: patch */
    const char *eyec = "AzpPatchDflt_var::_upward_var"; 
    AzX::throw_if((mv_inp->rowNum() != cc), eyec, "#channel mismatch"); 
    if (!is_test) pia_inp_dcolind.reset(mv_inp->d_index()); 

    AzIntArr ia_out_dind; int iszmax, oszmax; 
    _upward_var_prep(*mv_inp, /* input */
                     ia_out_dind, ia_ipos, ia_opos, iszmax, oszmax); /* output */
    if (iszmax <= 0) return; 

    if (iszmax > curr_iszmax) {
      _var_map(iszmax, &pia2_inp2out, &pia2_out2inp); /* make a template for the largest */
      curr_iszmax = iszmax; 
      AzX::throw_if(pia2_out2inp.size() != oszmax, eyec, "#output doesn't match (1)");       
    }
    AzX::throw_if(pia2_out2inp.size() < oszmax, eyec, "#output doesn't match (2)");     
    AzPmat m; app.add_with_map_var(ia_ipos, ia_opos, p.pch_sz, *mv_inp->data(), m, cc, pia2_out2inp);
    m.change_dim(p.pch_sz*cc, m.size()/p.pch_sz/cc); /* pixel -> patch */
    mv_out->set(&m, &ia_out_dind); 
  }
  /*------------------------------------------------------------*/ 
  virtual void _downward_var(const AzPmatVar *mv_out, AzPmatVar *mv_inp) {      
    AzPmat m;      
    app.add_with_map_var(ia_opos, ia_ipos, 1, *mv_out->data(), m, cc, pia2_inp2out);
    mv_inp->set(&m, &pia_inp_dcolind); 
  }  
 
  /*------------------------------------------------------------*/ 
  /*------------------------------------------------------------*/   
  virtual void _upwardT_var_prep(const AzPmatVar &mv_inp, AzIntArr &ia_out_dind, 
                         AzIntArr &ia_ipos, AzIntArr &ia_opos,
                         int &iszmax, int &oszmax) const {
    AzX::throw_if(!mv_inp.is_consecutive(), "AzpPatchDflt_var::_upward_var_prepT", 
                  "input columns of PmatVar must be consecutive"); 
    ia_ipos.reset(); ia_opos.reset(); ia_out_dind.reset(); iszmax = 0; oszmax = 0; 
    if (mv_inp.dataNum() <= 0) return; 
    int ipos = 0, opos = 0; 
    for (int dx = 0; dx < mv_inp.dataNum(); ++dx) {
      ia_out_dind.put(opos); 
      int pch_num = mv_inp.size(dx); 
      int isz = pch_num*p.pch_sz, osz = how_manyT(pch_num); 
      iszmax = MAX(iszmax, isz);  oszmax = MAX(oszmax, osz); 
      ia_ipos.put(ipos);          ia_opos.put(opos); 
      ipos += isz;                opos += osz; 
      ia_out_dind.put(opos);    
    }
    ia_ipos.put(ipos); ia_opos.put(opos); 
  }

  /*------------------------------------------------------------*/ 
  virtual void _upwardT_var(bool is_test, 
                     const AzPmatVar *mv_inp, /* col: patch */
                     AzPmatVar *mv_out) { /* col: pixel */
    const char *eyec = "AzpPatchDflt_var::_upwardT_var"; 
    AzX::throw_if((mv_inp->rowNum() != cc), eyec, "Wrong input #rows"); 
    if (!is_test) pia_inp_dcolind.reset(mv_inp->d_index()); 

    AzIntArr ia_out_dind; int iszmax, oszmax; 
    _upwardT_var_prep(*mv_inp, /* input */
          ia_out_dind, ia_ipos, ia_opos, iszmax, oszmax); /* output */
    if (iszmax <= 0) return; 

    if (iszmax > curr_iszmax) {
      _var_map(oszmax, &pia2_out2inp, &pia2_inp2out); /* make a template for the largest */
      curr_iszmax = iszmax; 
      AzX::throw_if(pia2_out2inp.size() != oszmax, eyec, "#output doesn't match (1)");       
    }
    AzX::throw_if(pia2_out2inp.size() < oszmax, eyec, "#output doesn't match (2)");     
    AzPmat m; app.add_with_map_var(ia_ipos, ia_opos, 1, *mv_inp->data(), m, cc/p.pch_sz, pia2_out2inp);
    mv_out->set(&m, &ia_out_dind); 
  }  
  /*------------------------------------------------------------*/ 
  virtual void _downwardT_var(const AzPmatVar *mv_out, AzPmatVar *mv_inp) {      
    AzPmat m;      
    app.add_with_map_var(ia_opos, ia_ipos, p.pch_sz, *mv_out->data(), m, cc/p.pch_sz, pia2_inp2out);
    m.change_dim(cc, m.size()/cc); 
    mv_inp->set(&m, &pia_inp_dcolind); 
  }   

  /*------------------------------------------------------------*/   
  /*------------------------------------------------------------*/   
  void _var_map(int isz, 
                AzPintArr2 *pia2_inp2out, /* unit: column (pixel) */
                AzPintArr2 *pia2_out2inp) /* unit: column (pixel) */ const { 
    const char *eyec = "AzpPatchDflt_var::var_map"; 
    int opch_num = how_many(isz), ipskip = get_pskip(); 
    AzDataArr<AzIntArr> aia_inp2out, aia_out2inp; 
    if (pia2_inp2out != NULL) aia_inp2out.reset(isz); 
    if (pia2_out2inp != NULL) aia_out2inp.reset(opch_num*p.pch_sz);   
    int ox = 0; /* destination pixel */
    for (int opch = 0; opch < opch_num; ++opch) { /* destination patch */
      int ipos0 = p.pch_step*opch - p.padding; 
      int ipos1 = ipos0 + p.pch_sz*ipskip;      
      for (int ipx = ipos0; ipx < ipos1; ipx+=ipskip, ++ox) { /* ipx: position within input data */
        if (ipx < 0 || ipx >= isz) continue; 
        if (pia2_inp2out != NULL) aia_inp2out.point_u(ipx)->put(ox); 
        if (pia2_out2inp != NULL) aia_out2inp.point_u(ox)->put(ipx); 
      }
    }    
    AzX::throw_if(ox != opch_num*p.pch_sz, eyec, "Conflict in #patches"); 
    if (pia2_inp2out != NULL) pia2_inp2out->reset(&aia_inp2out); 
    if (pia2_out2inp != NULL) pia2_out2inp->reset(&aia_out2inp); 
  }    
};      
#endif 
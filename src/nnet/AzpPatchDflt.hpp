/* * * * *
 *  AzpPatchDflt.hpp
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

#ifndef _AZP_PATCH_DFLT_HPP_
#define _AZP_PATCH_DFLT_HPP_

#include "AzpPatch_.hpp"
#include "AzPmatApp.hpp"
#include "AzxD.hpp"
#include "Az2D.hpp"

/* generate patches/filters for convolutional NNs */

class AzpPatchDflt_Param {  
protected: 
  static const int version = 0; 
/*  static const int reserved_len = 64;  */
  static const int reserved_len = 63; /* 2/20/2017: for do_transpose */
public: 
  int pch_sz, pch_step, padding; 
  bool do_transpose; 
  
  AzpPatchDflt_Param() : pch_sz(-1), pch_step(1), padding(0), do_transpose(false) {}    

  #define kw_pch_sz "patch_size="
  #define kw_pch_step "patch_stride="
  #define kw_padding "padding="
  #define kw_do_transpose "Transpose"
  virtual void setup_allinone() {
    pch_sz=-1; pch_step=1; padding=0; do_transpose=false;  
  }
  virtual void setup_asis() {
    pch_sz=1; pch_step=1; padding=0; do_transpose=false; 
  }
  virtual void setup(int _pch_sz, int _pch_step, int _padding) {
    pch_sz=_pch_sz; pch_step=_pch_step; padding=_padding; do_transpose=false; 
  }
  virtual void resetParam(AzParam &azp, const char *pfx, bool is_warmstart) {
    if (is_warmstart) return; 
    azp.reset_prefix(pfx); 
    azp.vInt(kw_pch_sz, &pch_sz); 
    azp.vInt(kw_pch_step, &pch_step); 
    azp.vInt(kw_padding, &padding); 
    azp.swOn(&do_transpose, kw_do_transpose); 
    azp.reset_prefix(); 
  }
  virtual void checkParam(const char *pfx) {
    const char *eyec = "AzpPatchDflt::checkParam"; 
    if (pch_sz > 0) {
      AzXi::throw_if_nonpositive(pch_step, eyec, kw_pch_step, pfx); 
/*      AzXi::throw_if_negative(padding, eyec, kw_padding, pfx); */ /* 4/1/2017: allow negative padding for bw-gen */
    }
    else {
      pch_step = -1; padding = 0;
      AzX::throw_if(do_transpose, eyec, "\"Transpose\" requires patch_size > 0."); 
    }
  }
  virtual void printParam(const AzOut &out, const char *pfx) const {
    AzPrint o(out); 
    o.reset_prefix(pfx); 
    if (pch_sz > 0) {
      o.printV(kw_pch_sz, pch_sz); 
      o.printV(kw_pch_step, pch_step); 
      o.printV(kw_padding, padding, true); /* 4/1/2017: force printing even if it's -1 */
    }
    o.printSw(kw_do_transpose, do_transpose); 
    o.ppEnd(); 
  }
  virtual void printHelp(AzHelp &h) const {
    h.item(kw_pch_sz, "Region size in the convolution layer with dense input.  If not specified, the layer becomes a fully-connected layer."); 
    h.item(kw_pch_step, "Region stride.  For dense input only."); 
    h.item(kw_padding, "Padding size at the edge.  For dense input only."); 
  }
  virtual void write(AzFile *file) const {
    AzTools::write_header(file, version, reserved_len);  
    file->writeBool(do_transpose); /* 2/20/2017 */
    file->writeInt(pch_sz); 
    file->writeInt(pch_step); 
    file->writeInt(padding); 
    bool dummy = true; file->writeBool(dummy); /* 2/20/2017: for compatibility */
  }
  virtual void read(AzFile *file) {
    AzTools::read_header(file, reserved_len);   
    do_transpose = file->readBool(); /* 2/20/2017 */
    pch_sz = file->readInt(); 
    pch_step = file->readInt(); 
    padding = file->readInt(); 
    bool dummy = file->readBool(); /* 2/20/2017: for compatibility */
  }    
}; 

class AzpPatchDflt : public virtual AzpPatch_ {
protected: 
  AzpPatchDflt_Param p; 

  AzxD i_region, o_region; 
  int cc;  /* # of channels */
  bool is_asis, is_allinone; 
  AzPintArr2 pia2_inp2out, pia2_out2inp; 
  Az2D tmpl;  /* filter template */
  AzIntArr ia_x0, ia_y0; /* filter location: upper left corner */ 
 
  AzPmatApp app; 

  virtual void transpose() {
    AzxD tmp(&i_region); i_region.reset(&o_region); o_region.reset(&tmp); 
    AzPintArr2 pia2_tmp(&pia2_inp2out); pia2_inp2out.reset(&pia2_out2inp); pia2_out2inp.reset(&pia2_tmp);     
  }
  
  static const int version = 0; 
  static const int reserved_len = 64;   
public:  
  AzpPatchDflt() : cc(0), is_asis(false), is_allinone(false) {}
  virtual void resetParam(AzParam &azp, const AzPfx &pfx, bool is_warmstart) {
    for (int px=0; px<pfx.size(); ++px) p.resetParam(azp, pfx[px], is_warmstart); 
    p.checkParam(pfx.pfx()); 
  }  
  virtual void printParam(const AzOut &out, const AzPfx &pfx) const { p.printParam(out, pfx.pfx()); }
  virtual void printHelp(AzHelp &h) const { p.printHelp(h); }

  virtual void reset(const AzpPatchDflt_Param &_p, const AzOut &out, const AzxD *input, int _cc, 
                     bool is_spa=false, bool is_var=false) {
    p = _p; reset(out, input, _cc, is_spa, is_var); 
  }
  virtual void reset(const AzOut &out, const AzxD *input, int _cc, bool is_spa, bool is_var) {
    const char *eyec = "AzpPatchDflt::reset"; 
    reset(); 
    int dim = input->get_dim(); 
    AzX::no_support((dim != 1 && dim != 2), eyec, "anything other than 1D or 2D data"); 
    i_region.reset(input); 
    cc = _cc; 
    AzXi::throw_if_nonpositive(cc, eyec, "#channels"); 
    if (p.pch_sz <= 0) def_allinone(); 
    else {
      AzX::no_support(is_spa, eyec, "convolution layer with sparse input."); 
      def_filters(); 
    }
    if (p.do_transpose) {
      AzX::throw_if(cc%patch_size() != 0, eyec, "To transpose, input #row must be a multiple of patch size."); 
      transpose(); 
      AzPrint::writeln(out, "... Transposing ... ");       
    }
  }  
  virtual void setup_allinone(const AzOut &out, const AzxD *input, int _cc) {
    p.setup_allinone(); 
    bool is_spa = false, is_var = false; 
    reset(out, input, _cc, is_spa, is_var); 
  }
  /*---  use this only for variable-sized top layer  ---*/
  virtual void setup_asis(const AzOut &out, const AzxD *_input, int _cc) { 
    p.setup_asis(); 
    AzxD input(_input); 
    if (!input.is_valid()) {
      int sz=2; 
      input.reset(&sz, 1); /* dummy as reset expects 1D or 2D fixed-sized data */
    }
    bool is_spa = false, is_var = false; 
    reset(out, &input, _cc, is_spa, is_var); 
  }
  virtual const AzxD *input_region(AzxD *o=NULL) const { 
    if (o != NULL) o->reset(&i_region); 
    return &i_region; 
  }
  virtual const AzxD *output_region(AzxD *o=NULL) const { 
    if (o != NULL) o->reset(&o_region); 
    return &o_region; 
  }
  
  /* size: # of pixels, length: size times cc */
  virtual int patch_size() const { return tmpl.size(); }  
  virtual int patch_length() const { return (p.do_transpose) ? cc/patch_size() : cc*patch_size(); }
  
  inline virtual bool no_change_in_shape() const {
    if (is_asis) return true; 
    return i_region.isSame(&o_region); 
  }  

  virtual void show_input(const char *header, const AzOut &out) const { i_region.show(header, out); }  
  virtual void show_output(const char *header, const AzOut &out) const { o_region.show(header, out); }  

  virtual int get_channels() const { return cc; }
  
  virtual bool is_convolution() const { return (p.pch_sz > 0); } 
  
  /*------------------------------------------------------------*/ 
  virtual void upward(bool is_test, /* not used */
                      const AzPmat *m_inp, /* each column represents a pixel; more than one data point */
                      AzPmat *m_out) const { /* each column represents a patch */
    const char *eyec = "AzpPatchDflt::upward"; 
    if (is_asis) {
      m_out->set(m_inp); 
      return; 
    }

    AzX::throw_if((m_inp->rowNum() != cc), eyec, "Conflict in # of channels"); 
    int isize = input_region()->size(); 
    int osize = output_region()->size(); 
    AzX::throw_if((m_inp->colNum() % isize != 0), eyec, "Conflict in input length"); 
    int data_num = m_inp->colNum() / isize; 

    int occ = (p.do_transpose) ? cc/patch_size() : cc*patch_size(); 
    m_out->reform(occ, osize*data_num); 
    if (is_allinone) {
      m_out->fill(m_inp); 
      return; 
    }
    int unit = (p.do_transpose) ? cc/patch_size() : cc; 
    app.add_with_map(data_num, m_inp, m_out, unit, &pia2_out2inp);    
  }

  /*------------------------------------------------------------*/ 
  virtual void downward(const AzPmat *m_out, 
                        AzPmat *m_inp) const {
    const char *eyec = "AzpPatchDflt::downward"; 
    if (is_asis) {
      m_inp->set(m_out); 
      return; 
    }

    int occ = (p.do_transpose) ? cc/patch_size() : cc*patch_size(); 
    AzX::throw_if((m_out->rowNum() != occ), eyec, "Conflict in patch length"); 
    int isize = input_region()->size(); 
    int osize = output_region()->size(); 
    AzX::throw_if((m_out->colNum() % osize != 0), eyec, "Conflict in # of patches"); 
    int data_num = m_out->colNum() / osize; 

    m_inp->reform(cc, isize*data_num); 
    if (is_allinone) {
      m_inp->fill(m_out); 
      return; 
    }
    int unit = (p.do_transpose) ? cc/patch_size() : cc; 
    app.add_with_map(data_num, m_out, m_inp, unit, &pia2_inp2out);     
  }  
  
  /*------------------------------------------------------------*/     
  virtual void reset() {
    i_region.reset(); o_region.reset(); 
    is_asis = is_allinone = false; 
    cc = 0; 
    tmpl.reset(0,0); 
    ia_x0.reset(); ia_y0.reset();    
    pia2_inp2out.reset(); pia2_out2inp.reset(); 
  }
  
  /*------------------------------------------------------------*/     
  virtual AzpPatch_ *clone() const {
    AzpPatchDflt *o = new AzpPatchDflt(); 
    o->reset(this); 
    return o; 
  }
  /*------------------------------------------------------------*/     
  virtual void reset(const AzpPatchDflt *i) {
    p = i->p; 
    
    i_region.reset(&i->i_region); 
    o_region.reset(&i->o_region); 
    is_asis = i->is_asis; 
    is_allinone = i->is_allinone; 
    cc = i->cc; 
    pia2_inp2out.reset(&i->pia2_inp2out); 
    pia2_out2inp.reset(&i->pia2_out2inp);     
    
    tmpl = i->tmpl;  
    ia_x0.reset(&i->ia_x0); 
    ia_y0.reset(&i->ia_y0); 
  }
  /*------------------------------------------------------------*/     
  virtual void write(AzFile *file) const {
    p.write(file); 
    AzTools::write_header(file, version, reserved_len);    
    i_region.write(file); 
    o_region.write(file); 
    file->writeBool(is_asis); 
    file->writeBool(is_allinone); 
    file->writeInt(cc); 
    pia2_inp2out.write(file); 
    pia2_out2inp.write(file); 
    
    tmpl.write(file); 
    ia_x0.write(file); 
    ia_y0.write(file);     
  }
  /*------------------------------------------------------------*/   
  virtual void read(AzFile *file) {
    p.read(file); 
    AzTools::read_header(file, reserved_len); 
    i_region.read(file); 
    o_region.read(file); 
    is_asis = file->readBool(); 
    is_allinone = file->readBool(); 
    cc = file->readInt(); 
    pia2_inp2out.read(file); 
    pia2_out2inp.read(file); 
    
    tmpl.read(file); 
    ia_x0.read(file); 
    ia_y0.read(file);    
  }  
  /*------------------------------------------------------------*/                         
  virtual void reset_for_pooling(
                       const AzxD *input,   
                       int pch_sz, int pch_step, int pch_padding, 
                       bool do_transpose, 
                       /*---  output  ---*/
                       AzPintArr2 &pia2_out2inp, 
                       AzPintArr2 &pia2_inp2out,
                       AzPintArr *pia_out2num) {   
    p.setup(pch_sz, pch_step, pch_padding); 
    AzOut dummy_out; int dummy_cc=1; 
    reset(dummy_out, input, dummy_cc, false, false);    

    if (do_transpose) {
      AzIntArr ia_inp2num; AzPintArr2 _pia2_out2inp, _pia2_inp2out; 
      mapping_for_pooling(_pia2_out2inp, _pia2_inp2out, NULL, &ia_inp2num); 
      transpose(); 
      pia2_out2inp.reset(&_pia2_inp2out); pia2_inp2out.reset(&_pia2_out2inp); 
      if (pia_out2num != NULL) pia_out2num->reset(&ia_inp2num); 
    }
    else {
      mapping_for_pooling(pia2_out2inp, pia2_inp2out, pia_out2num); 
    }  
  }                         
  static void mapping_for_resnorm(const AzxD *region, int width, 
                                   AzPintArr2 *pia2_neighbors,
                                   AzPintArr2 *pia2_whose_neighbor, 
                                   AzPintArr *pia_neighsz); 
  
protected:   
  virtual void def_allinone(); 
  virtual void def_filters();  
  void set_dim_back();  
  void def_patches(int pch_xsz, int pch_xstride, int xpadding,
                   int pch_ysz, int pch_ystride, int ypadding);  
  void _set_simple_grids(int sz, int pch_sz, int step, 
                         AzIntArr *ia_p0) const {
    int pch_num = DIVUP(sz-pch_sz, step) + 1; 
    for (int px = 0; px < pch_num; ++px) ia_p0->put(px*step); 
  }                
  virtual void mapping_for_filtering(AzPintArr2 *pia2_out2inp, 
                                     AzPintArr2 *pia2_inp2out) const;                 
  virtual void mapping_for_pooling(AzPintArr2 &pia2_out2inp, AzPintArr2 &pia2_inp2out, 
                   AzPintArr *pia_out2num, AzIntArr *out_ia_inp2num=NULL) const; 
};      
#endif 
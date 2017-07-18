/* * * * *
 *  AzpPoolingDflt_var.hpp
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

#ifndef _AZP_POOLING_DFLT_VAR_HPP_
#define _AZP_POOLING_DFLT_VAR_HPP_

#include "AzPmat.hpp"
#include "AzPmatApp.hpp"   
#include "AzpPooling_var_.hpp"
#include "AzpPoolingDflt.hpp"

/*!!!!!  for 1D data only  !!!!!*/

/*-----   Pooling with variable-sized input   -----*/
class AzpPoolingDflt_var : public virtual AzpPooling_var_ {
protected:
  class AzpPoolingDflt_var_Param : public virtual AzpPoolingDflt_Param {
  protected: 
    static const int version = 0; 
/*    static const int reserved_len = 64; */
    static const int reserved_len = 63; /* 3/10/2017: do_transpose */
  public:    
    /*---  7/2/2015: read/write pl_num again for file compatibility  ---*/
    virtual void write(AzFile *file) const {
      AzpPoolingDflt_Param::write(file); 
      AzTools::write_header(file, version, reserved_len); 
      file->writeBool(do_transpose); 
      file->writeInt(pl_num); 
    }
    virtual void read(AzFile *file) {
      AzpPoolingDflt_Param::read(file); 
      AzTools::read_header(file, reserved_len);      
      do_transpose = file->readBool(); 
      pl_num = file->readInt(); 
    }
 
    bool do_old_pool; /* if true, do it in the older and slower way; debuggin purposes only */  
    #define kw_do_old_pool "OldPool"  
    bool do_old_pool2; 
    #define kw_do_old_pool2 "OldPool2"    
    virtual void resetParam(AzParam &azp, const char *pfx, bool is_warmstart) {
      AzpPoolingDflt_Param::resetParam(azp, pfx, is_warmstart);    
      azp.reset_prefix(pfx); 
      azp.swOn(&do_old_pool, kw_do_old_pool);    
      azp.swOn(&do_old_pool2, kw_do_old_pool2);          
      azp.reset_prefix(); 
    }    
    virtual void printParam(const AzOut &out, const char *pfx) const {
      AzpPoolingDflt_Param::printParam(out, pfx); 
      if (ptyp == AzpPoolingDflt_None) return; 
      AzPrint o(out, pfx);    
      o.printSw(kw_do_old_pool, do_old_pool); 
      o.printSw(kw_do_old_pool2, do_old_pool2);       
      o.printEnd(); 
    }      
    AzpPoolingDflt_var_Param() : do_old_pool(false), do_old_pool2(false) {}
  }; 
  
protected: 
  AzpPoolingDflt_var_Param p; 
  AzPintArr2 pia2_inp2out;   
  AzPintArr pia_out2num; /* used for average pooling */
  AzPintArr pia_inp_ind; 
  AzIntArr ia_out_ind; 
  int outnum; 
  
  int inpsz; 
  
  AzPintArr pia_chosen; /* used only by max pooling */
  AzPmat m_inp_save, m_out_save; /* used only by l2 pooling */
  
  /*---  var  ---*/
  AzIntArr ia_ipos, ia_opos; 
  AzPintArr2 pia2_out2inp; 
  int curr_iszmax;   
  /*-------------*/
  
  AzPmatApp app;   
  static const int version = 0; 
  static const int reserved_len = 64;    
public:
  AzpPoolingDflt_var() : outnum(0), inpsz(0), curr_iszmax(0) {}
  virtual void get_chosen(AzIntArr *ia_chosen) const { pia_chosen.get(ia_chosen); } 
  virtual void resetParam(AzParam &azp, const AzPfx &pfx, bool is_warmstart=false) {
    for (int px=0; px<pfx.size(); ++px) p.resetParam(azp, pfx[px], is_warmstart); 
    p.checkParam(pfx.pfx()); 
  }  
  virtual void printParam(const AzOut &out, const AzPfx &pfx) const { p.printParam(out, pfx.pfx()); }
  virtual void printHelp(AzHelp &h) const { p.printHelp(h); }
  void write(AzFile *file) const { 
    p.write(file); 
    AzTools::write_header(file, version, reserved_len); 
  }
  void read(AzFile *file) { 
    p.read(file); 
    AzTools::read_header(file, reserved_len);    
  }  
  
  void reset(const AzOut &out) {
    outnum = 0;     
    if (do_asis() && p.ptyp != AzpPoolingDflt_None) AzPrint::writeln(out, "AzpPoolingDflt_var::reset, do_asis() is ON"); 
    AzX::no_support(p.do_transpose && !_do_var(), "AzpPoolingDflt_var::reset", "Transpose and fixed-sized pooling"); 
  }
  AzpPooling_var_ *clone() const {
    AzpPoolingDflt_var *o = new AzpPoolingDflt_var(); 
    o->p = p; 
    o->outnum = outnum;    
    o->pia2_inp2out.reset(&pia2_inp2out); 
    o->pia_out2num.reset(&pia_out2num); 
    o->pia_inp_ind.reset(&pia_inp_ind); 
    o->ia_out_ind.reset(&ia_out_ind); 
    o->pia_chosen.reset(&pia_chosen); 
    o->m_inp_save.set(&m_inp_save); 
    o->m_out_save.set(&m_out_save);    
    o->ia_ipos.reset(&ia_ipos); 
    o->ia_opos.reset(&ia_opos); 
    o->pia2_out2inp.reset(&pia2_out2inp); 
    o->curr_iszmax = curr_iszmax;      
    return o; 
  }
  virtual void output(AzxD *_output) const { 
    /* empty region means output is variable-sized */
    if (p.pl_num > 0) _output->reset(&p.pl_num, 1); 
    else              _output->reset(); 
  }
  virtual void upward(bool is_test, const AzPmatVar *m_inp, AzPmatVar *m_out) {
    if (do_asis()) {
      m_out->set(m_inp); 
      ia_out_ind.reset(m_inp->h_index()); /* 5/3/2015: no harm was done w/o this though.  */
      return; 
    }
    if (_upward_var(is_test, *m_inp, *m_out)) {
      return; 
    }
    inpsz = m_inp->colNum(); 
    if (is_test) _apply(m_inp, m_out);  
    else         _upward(m_inp, m_out); 
  }
   
  virtual void upward(bool is_test, AzPmatVar *m_inout) {
    if (do_asis()) {
      ia_out_ind.reset(m_inout->h_index());
      return; 
    }
    AzPmatVar m_out; upward(is_test, m_inout, &m_out); 
    m_inout->set(&m_out); 
  }
  
  virtual void downward(const AzPmatVar *m_lossd_after, AzPmatVar *m_lossd_before) const {
    if (do_asis()) {
      m_lossd_before->set(m_lossd_after); 
      return; 
    }
    if (_downward_var(*m_lossd_after, *m_lossd_before)) {
      return; 
    }    
    const AzPmat *m_after_items = m_lossd_after->data(); 
    int r_num = m_after_items->rowNum(); 
    AzX::throw_if((m_after_items->colNum() != outnum), "AzpPoolingDflt_var::downward", "outnum mismatch"); 
    AzPmat m_before_items(r_num, inpsz); 
    if      (p.ptyp == AzpPoolingDflt_Avg) downward_avg(m_after_items, &m_before_items); 
    else if (p.ptyp == AzpPoolingDflt_Max) downward_max(m_after_items, &m_before_items); 
    else if (p.ptyp == AzpPoolingDflt_L2)  downward_l2(m_after_items, &m_before_items); 
    m_lossd_before->set(&m_before_items, &pia_inp_ind); 
  }

  virtual const AzIntArr *out_ind() const {
    return &ia_out_ind; 
  }

  /*---  variable-size to variable-size  ---*/
  static void map_to_var(const AzPmatVar *m_inp,
                  int pch_sz, int pch_step, 
                  int padding, 
                  bool do_force_shape, 
                  /*---  output  ---*/
                  AzIntArr *ia_out_dataind, 
                  AzPintArr2 *pia2_inp2out, 
                  AzPintArr2 *pia2_out2inp, 
                  AzPintArr *pia_out2num=NULL) {
    const char *eyec = "AzpPoolingDflt_var::map_to_var";                     
    ia_out_dataind->reset(); 
    int item_num = m_inp->colNum();
    int data_num = m_inp->dataNum(); 
    AzDataArr<AzIntArr> aia_inp2out, aia_out2inp; 
    AzIntArr ia_out2num; 
    if (pia2_inp2out != NULL) aia_inp2out.reset(item_num); 
    int o_num = 0; 
    int dx; 
    for (dx = 0; dx < data_num; ++dx) {
      int px_begin, px_end; 
      m_inp->get_begin_end(dx, px_begin, px_end);   
      int num = how_many(px_end-px_begin, pch_sz, pch_step, padding, do_force_shape); 
      o_num += num; 
    }  
    if (pia2_out2inp != NULL) aia_out2inp.reset(o_num); 
    
    int ox = 0; 
    for (dx = 0; dx < data_num; ++dx) {
      ia_out_dataind->put(ox); 
      int px_begin, px_end; 
      m_inp->get_begin_end(dx, px_begin, px_end);     
      int num = how_many(px_end-px_begin, pch_sz, pch_step, padding, do_force_shape); 
      
      for (int ix = 0; ix < num; ++ix, ++ox) {
        int px0 = -padding + px_begin + ix*pch_step; 
        int px1 = MIN(px0+pch_sz, px_end+padding);        
        px0 = MAX(px_begin, px0); 
        px1 = MIN(px_end, px1); 
        if (pia_out2num != NULL) ia_out2num.put(MAX(0,px1-px0));         
        if (px0 < px1) {
          if (pia2_out2inp != NULL) {
            AzIntArr ia; ia.range(px0, px1); 
            aia_out2inp.point_u(ox)->reset(&ia); 
          }
          if (pia2_inp2out != NULL) {
            int px; 
            for (px = px0; px < px1; ++px) aia_inp2out.point_u(px)->put(ox); 
          }
        }
      }
      ia_out_dataind->put(ox); 
    } 
    if (pia2_out2inp != NULL) AzX::throw_if((ox != aia_out2inp.size()), eyec, "Something is wrong ... "); 
    if (pia2_inp2out != NULL) pia2_inp2out->reset(&aia_inp2out); 
    if (pia2_out2inp != NULL) pia2_out2inp->reset(&aia_out2inp); 
    if (pia_out2num != NULL) pia_out2num->reset(&ia_out2num); 
  } 
  bool do_asis() const {
    return (p.ptyp == AzpPoolingDflt_None || (p.pl_num <= 0 && p.pl_sz == 1 && p.pl_step == 1 && p.pl_pad == 0)); 
  }
  
protected:
  virtual void _upward(const AzPmatVar *m_inp, AzPmatVar *m_out) { 
    AzPintArr2 pia2_out2inp; 
    pia_inp_ind.reset(m_inp->d_index()); 
    if (p.ptyp == AzpPoolingDflt_Avg) map(m_inp, p.pl_num, p.pl_sz, p.pl_step, p.pl_pad, p.do_pl_force_shape, 
                                          &ia_out_ind, &pia2_inp2out, &pia2_out2inp, &pia_out2num); 
    else                              map(m_inp, p.pl_num, p.pl_sz, p.pl_step, p.pl_pad, p.do_pl_force_shape, 
                                          &ia_out_ind, &pia2_inp2out, &pia2_out2inp);
    outnum = pia2_out2inp.size(); 
    const AzPmat *m_iitems = m_inp->data(); 
    AzPmat m_oitems(m_iitems->rowNum(), outnum);     
    if      (p.ptyp == AzpPoolingDflt_Avg) upward_avg(m_iitems, pia2_out2inp, &m_oitems); 
    else if (p.ptyp == AzpPoolingDflt_Max) upward_max(m_iitems, pia2_out2inp, &m_oitems); 
    else if (p.ptyp == AzpPoolingDflt_L2)  upward_l2(m_iitems, pia2_out2inp, &m_oitems); 
    m_out->set(&m_oitems, &ia_out_ind); 
  }
  virtual void _apply(const AzPmatVar *m_inp, AzPmatVar *m_out) const { 
    AzPintArr2 pia2_out2inp; 
    AzIntArr ia_out_ind; 
    map(m_inp, p.pl_num, p.pl_sz, p.pl_step, p.pl_pad, p.do_pl_force_shape, &ia_out_ind, NULL, &pia2_out2inp);                     
    const AzPmat *m_iitems = m_inp->data(); 
    AzPmat m_oitems(m_iitems->rowNum(), pia2_out2inp.size());   
    if      (p.ptyp == AzpPoolingDflt_Avg) upward_avg(m_iitems, pia2_out2inp, &m_oitems); 
    else if (p.ptyp == AzpPoolingDflt_Max) apply_max(m_iitems, pia2_out2inp, &m_oitems); 
    else if (p.ptyp == AzpPoolingDflt_L2)  apply_l2(m_iitems, pia2_out2inp, &m_oitems);  
    m_out->set(&m_oitems, &ia_out_ind); 
  }

  static void map(const AzPmatVar *m_inp,
                  int pch_num, 
                  int pch_sz, 
                  int pch_step, 
                  int padding, 
                  bool do_force_shape, 
                  /*---  output  ---*/
                  AzIntArr *ia_out_dataind, 
                  AzPintArr2 *pia2_inp2out, 
                  AzPintArr2 *pia2_out2inp, 
                  AzPintArr *pia_out2num=NULL) {
    if (pch_num > 0) map_to_fixed(m_inp, pch_num, ia_out_dataind, pia2_inp2out, pia2_out2inp, pia_out2num); 
    else             map_to_var(m_inp, pch_sz, pch_step, padding, do_force_shape, ia_out_dataind, pia2_inp2out, pia2_out2inp, pia_out2num); 
  }

  /*---  variable-size to fixed-size  ---*/
  static void map_to_fixed(const AzPmatVar *m_inp, /* data point -> patches */
                  int pch_num, 
                  /*---  output  ---*/
                  AzIntArr *ia_out_dataind, 
                  AzPintArr2 *pia2_inp2out, 
                  AzPintArr2 *pia2_out2inp, 
                  AzPintArr *pia_out2num=NULL) {               
    ia_out_dataind->reset(); 
    int item_num = m_inp->colNum(); 
    int data_num = m_inp->dataNum(); 
    AzDataArr<AzIntArr> aia_inp2out, aia_out2inp; 
    AzIntArr ia_out2num; 
    if (pia2_inp2out != NULL) aia_inp2out.reset(item_num); 
    if (pia2_out2inp != NULL) aia_out2inp.reset(data_num*pch_num); 
    int ox = 0; 
    int dx; 
    for (dx = 0; dx < data_num; ++dx) {
      ia_out_dataind->put(ox); 
      int px_begin, px_end; 
      m_inp->get_begin_end(dx, px_begin, px_end);
      int unit = DIVUP(px_end-px_begin, pch_num); 
      int ix; 
      for (ix = 0; ix < pch_num; ++ix, ++ox) {
        int px0 = px_begin + unit*ix, px1 = MIN(px0+unit, px_end); 
        if (pia_out2num != NULL) ia_out2num.put(MAX(0,px1-px0)); 
        if (px1-px0 > 0) {        
          if (pia2_out2inp != NULL) {
            AzIntArr ia; ia.range(px0, px1); 
            aia_out2inp.point_u(ox)->reset(&ia); 
          }
          if (pia2_inp2out != NULL) {
            int px; 
            for (px = px0; px < px1; ++px) aia_inp2out.point_u(px)->put(ox); 
          }
        }
      }
      ia_out_dataind->put(ox); 
    } 
    if (pia2_out2inp != NULL) AzX::throw_if((ox != aia_out2inp.size()), "AzpPoolingDflt_var::map_to_fixed", 
                                           "Something is wrong ... ");     
    if (pia2_inp2out != NULL) pia2_inp2out->reset(&aia_inp2out); 
    if (pia2_out2inp != NULL) pia2_out2inp->reset(&aia_out2inp); 
    if (pia_out2num != NULL) pia_out2num->reset(&ia_out2num); 
  }
  
  /*---  average pooling  ---*/
  inline void upward_avg(const AzPmat *m_inp, const AzPintArr2 &pia2_out2inp, AzPmat *m_out) const {
    app.pooling_avg(m_inp, m_inp->colNum(), m_out, &pia2_out2inp); 
  }
  inline void downward_avg(const AzPmat *m_out, AzPmat *m_inp) const {
    app.unpooling_avg(m_inp, m_out, m_out->colNum(), &pia2_inp2out, &pia_out2num); 
  }
  
  /*---  l2 pooling  ---*/
  inline void upward_l2(const AzPmat *m_inp, const AzPintArr2 &pia2_out2inp, AzPmat *m_out) {
    app.pooling_l2(m_inp, m_inp->colNum(), m_out, &pia2_out2inp); 
    m_inp_save.set(m_inp); 
    m_out_save.set(m_out); 
  }
  inline void apply_l2(const AzPmat *m_inp, const AzPintArr2 &pia2_out2inp, AzPmat *m_out) const {
    app.pooling_l2(m_inp, m_inp->colNum(), m_out, &pia2_out2inp); 
  }
  inline void downward_l2(const AzPmat *m_out, AzPmat *m_inp) const {
    app.unpooling_l2(m_inp, m_out, m_out->colNum(), &pia2_inp2out, &m_inp_save, &m_out_save); 
  }
  
  /*---  max pooling  ---*/
  inline void upward_max(const AzPmat *m_inp, const AzPintArr2 &pia2_out2inp, AzPmat *m_out) {
    pia_chosen.alloc(m_out->size()); 
    pia_chosen.set(-1); 
    app.pooling_max(m_inp, m_inp->colNum(), m_out, &pia2_out2inp, &pia_chosen); 
  }
  inline void apply_max(const AzPmat *m_inp, const AzPintArr2 &pia2_out2inp, AzPmat *m_out) const {
    app.pooling_max(m_inp, m_inp->colNum(), m_out, &pia2_out2inp, NULL); 
  }
 
  inline void downward_max(const AzPmat *m_out, AzPmat *m_inp) const {
    if (p.do_old_pool2) app.unpooling_max2(m_inp, m_out, &pia_chosen, &pia2_inp2out); 
    else                app.unpooling_max(m_inp, m_out, &pia_chosen, &pia2_inp2out); 
  }
  
  static int how_many(int inp, int pch_sz, int pch_step, int padding, bool do_force_shape, bool do_transpose=false) {
    if (do_force_shape) return inp; 
    if (!do_transpose) return DIVUP(padding*2+inp-pch_sz, pch_step) + 1;  
    else               return (inp-1)*pch_step + pch_sz - padding*2; 
  } 
  virtual int how_many(int whole) const { return how_many(whole, p.pl_sz, p.pl_step, p.pl_pad, p.do_pl_force_shape, p.do_transpose); }

  /*------------------------------------------------------------*/   
  /*------------------------------------------------------------*/    
  virtual void _upward_var_prep(const AzPmatVar &mv_inp, AzIntArr &ia_out_dind, 
                         AzIntArr &ia_ipos, AzIntArr &ia_opos, 
                         int &iszmax, int &oszmax) const {
    AzX::throw_if(!mv_inp.is_consecutive(), "AzpPoolingDflt_var::_upward_var_prep", 
                  "input PmatVar must be consecutive"); 
                           
    ia_ipos.reset(); ia_opos.reset(); ia_out_dind.reset(); iszmax = 0; oszmax = 0; 
    if (mv_inp.dataNum() <= 0) return; 
    int ipos = 0, opos = 0; 
    for (int dx = 0; dx < mv_inp.dataNum(); ++dx) {
      ia_out_dind.put(opos); 
      int isz = mv_inp.size(dx), osz = how_many(isz); 
      iszmax = MAX(iszmax, isz); oszmax = MAX(oszmax, osz); 
      ia_ipos.put(ipos);         ia_opos.put(opos); 
      ipos += isz;               opos += osz; 
      ia_out_dind.put(opos); 
    }
    ia_ipos.put(ipos); ia_opos.put(opos); 
  }

  virtual bool _do_var() const {
    if (p.do_old_pool || p.do_old_pool2 || p.pl_num > 0) return false;   
    if (p.ptyp != AzpPoolingDflt_Max && p.ptyp != AzpPoolingDflt_Avg) return false;
    return true; 
  }   
  
  virtual bool _upward_var(bool is_test, const AzPmatVar &mv1, AzPmatVar &mv2) { 
    if (!_do_var()) return false; 
     
    pia_inp_ind.reset(mv1.d_index());   
    int iszmax, oszmax; 
    _upward_var_prep(mv1, ia_out_ind, ia_ipos, ia_opos, iszmax, oszmax); 
    
    AzPintArr *chosen_ptr = (is_test) ? NULL : &pia_chosen;        
    if (iszmax > curr_iszmax) {
      if (p.do_transpose) map_to_var_tmpl(oszmax, &pia2_out2inp, &pia2_inp2out);
      else                map_to_var_tmpl(iszmax, &pia2_inp2out, &pia2_out2inp);   
      curr_iszmax = iszmax; 
    }     
    AzPmat m2; 
    if      (p.ptyp == AzpPoolingDflt_Max) app.pooling_max_var(*mv1.data(), m2, pia2_out2inp, ia_ipos, ia_opos, chosen_ptr); 
    else if (p.ptyp == AzpPoolingDflt_Avg) app.pooling_avg_var(*mv1.data(), m2, pia2_out2inp, ia_ipos, ia_opos, p.pl_sz); 
    mv2.set(&m2, &ia_out_ind);     
    return true; 
  }

  virtual bool _downward_var(const AzPmatVar &mv_o, AzPmatVar &mv_i) const {
    if (!_do_var()) return false; 
   
    AzPmat m_i; 
    if      (p.ptyp == AzpPoolingDflt_Max) app.unpooling_max_var(*mv_o.data(), m_i, pia2_inp2out, ia_opos, ia_ipos, pia_chosen); 
    else if (p.ptyp == AzpPoolingDflt_Avg) app.unpooling_avg_var(*mv_o.data(), m_i, pia2_inp2out, ia_opos, ia_ipos, p.pl_sz);  
    mv_i.set(&m_i, &pia_inp_ind); 
    return true; 
  }
  
  void for_debug(int iszmax, const AzPintArr2 &pia2, const char *eyec, bool is_down=false) const {
    printf("%s iszmax=%d, curr_iszmax=%d\n", eyec, iszmax, curr_iszmax); fflush(stdout); 
    AzDataArr<AzIntArr> aia; 
    pia2_inp2out.get(aia);
    printf("get of pia2_inp2out succeeded\n"); fflush(stdout);     
    if (!is_down) return; 
    printf("%s pia2.size()=%d\n", eyec, pia2.size()); fflush(stdout); 
    for (int ix = 0; ix < aia.size(); ++ix) {
      printf("%s app, pia2[%d],min,%d,max,%d,num,%d\n", eyec, ix, aia[ix]->min(), aia[ix]->max(), aia[ix]->size()); fflush(stdout); 
    }  
  }
  
  /*---  variable-size to variable-size  ---*/
  void map_to_var_tmpl(int isz, 
          AzPintArr2 *pia2_inp2out, AzPintArr2 *pia2_out2inp) /* output */ const {
    AzDataArr<AzIntArr> aia_inp2out, aia_out2inp; 
    if (pia2_inp2out != NULL) aia_inp2out.reset(isz); 
    int onum = how_many(isz); 
    if (pia2_out2inp != NULL) aia_out2inp.reset(onum); 
 
    for (int ox = 0; ox < onum; ++ox) {
      int ipx0 = ox*p.pl_step - p.pl_pad, ipx1 = ipx0 + p.pl_sz; 
      for (int ipx = MAX(0, ipx0); ipx < MIN(isz, ipx1); ++ipx) {
        if (pia2_out2inp != NULL) aia_out2inp(ox)->put(ipx); 
        if (pia2_inp2out != NULL) aia_inp2out(ipx)->put(ox); 
      }
    } 
    if (pia2_inp2out != NULL) pia2_inp2out->reset(&aia_inp2out); 
    if (pia2_out2inp != NULL) pia2_out2inp->reset(&aia_out2inp); 
  } 
};
#endif 
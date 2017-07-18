/* * * * *
 *  AzpReLayer.cpp    
 *  Copyright (C) 2016,2017 Rie Johnson
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

#include "AzpReLayer.hpp"

#define kw_dflt_pfx ""
/*------------------------------------------------------------*/  
void AzpReLayer_::gen_prefix(int layer_no, AzPfx &pfx) const {                               
  if (pfx.size() > 0) return; 
  pfx.put(kw_dflt_pfx);
  if (is_top()) pfx.put(_ReLay_top_); 
  else {
    AzBytArr s_pfx; s_pfx << layer_no; 
    pfx.put(&s_pfx); 
  }  
}
void AzpReLayer_ComboH_::gen_prefix_combo(int ix, const AzPfx &inp_pfx, AzPfx &pfx) const {
  pfx.reset(&inp_pfx); 
  AzBytArr s_pfx; 
  s_pfx << layer_no << "-" << ix << "_"; /* e.g., 0-0_ */
  pfx.put(s_pfx.c_str()); 
}
void AzpReLayer_Side::gen_prefix_side(int ix, AzPfx &pfx) const {
  pfx.reset(); 
  pfx.put(kw_dflt_pfx); 
  AzBytArr s; s << layer_no << "side" << ix << "_";  /* e.g., 0side0_ */
  pfx.put(s); 
}

/*------------------------------------------------------------*/
void AzpReLayer_::show_weight_dim(const AzpWeight_ *wei, const char *header) const {
  if (header != NULL && strlen(header) > 0) {
    AzPrint::writeln(out, header);  
  }
  AzPrint::writeln(out, "   --------  weights  --------"); 
  AzPrint::writeln(out, "   input dim: ", wei->get_dim()); 
  AzPrint::writeln(out, "   output dim: ", wei->classNum()); 
  AzPrint::writeln(out, "   #weights: ", wei->num_weights()); 
  AzPrint::writeln(out, "   ---------------------------");
}  

/*------------------------------------------------------------*/
void AzpReLayer_Wei_::reset_external_dic(const AzBytArr &s_fn) {
  dicc.reset(); 
  if (s_fn.length() <= 0) return; 
  AzDic dic(s_fn.c_str()); 
  dic.copy_words_only_to(dicc); 
}

/*--- call this only for a bottom layer ---*/
void AzpReLayer_Wei_::check_word_mapping(const AzpData_tmpl_ *data) { 
  AzX::throw_if_null(data, "AzpReLayer_Wei_::check_word_mapping", "data");   
  int dsno = MAX(0, lap.dsno);   
  AzDicc data_dicc(data->get_dic(dsno)); 
  if (dicc.size() > 0) {
    AzX::throw_if(!dicc.is_same(data_dicc), "AzpReLayer_Wei_::check_word_mapping", 
                  "Word mapping conflict: the one given externally vs. the one given with data"); 
  }  
  else {
    dicc.reset(data_dicc);  
  }
}

/*------------------------------------------------------------*/
void AzpReLayer_Wei_::init_x_weights() {
  if (lap.s_iw_fn.length() > 0) {
    const char *w_fn = lap.s_iw_fn.c_str(); 
    reset_external_dic(lap.s_iw_wordmap_fn.c_str());   
    AzPrint::writeln(out, "To set external weights, reading: ", w_fn); 
    AzpLm_Untrainable lm_inp; 
    if (AzBytArr::endsWith(w_fn, ".lm")) {
      lm_inp.read(w_fn); 
    }  
    else if (AzBytArr::endsWith(w_fn, "pmat")) {
      AzPmat m_w(w_fn), m_i(1, m_w.colNum()); 
      lm_inp.reset(&m_w, &m_i); 
    }
    else if (AzBytArr::endsWith(w_fn, "dmatc")) {
      AzDmatc md(w_fn); 
      AzPmat m_w(&md), m_i(1, m_w.colNum());    
      lm_inp.reset(&m_w, &m_i); 
    }
    else {
      AzDmat m_w; 
      if (AzBytArr::endsWith(w_fn, "dmat")) m_w.read(w_fn); 
      else                                  AzTextMat::readMatrix(w_fn, &m_w); 
      AzDmat m_i(1, m_w.colNum()); 
      lm_inp.reset(&m_w, &m_i); 
    }     
    AzBytArr s("  "); s << " (" << lm_inp.nodeNum() << " nodes)";
    AzPrint::writeln(out, s); 
    wei_x->initWeights(&lm_inp, lap.iw_coeff); 
  }
  else {
    wei_x->initWeights(); 
  }
}  

/*------------------------------------------------------------*/      
int AzpReLayer_Wei_::setup(AzParam &azp, const AzpReLayer_Param &pp, const AzPfx &pfx, 
                           bool is_warmstart, bool for_testonly) {
  const char *eyec = "AzpReLayer_Wei_::setup"; 
  check_if_ready(eyec);   

  lap.resetParam(out, azp, pfx, is_top(), is_warmstart); 
  int nodes = (is_top()) ? pp.class_num : lap.nodes; 
  AzX::throw_if((nodes <= 0), eyec, "Negative #node?!"); 
  lap.nodes = nodes;   

  if (!is_re()) {
    cs.dropout->resetParam(out, azp, pfx, for_testonly); 
    check_dropout(pp, eyec); 
    cs.dropout->reset(out);
  }
  if (!is_top() && !is_re())  {
    cs.pool_var->resetParam(out, azp, pfx, is_warmstart); cs.pool_var->reset(out); 
    cs.resnorm->resetParam(out, azp, pfx, is_warmstart);  cs.resnorm->reset(out, NULL, lap.nodes); 
  }
  
  AzpReLayer_Param mypp = pp; 
  if (lap.dsno > 0) { /* using an alternative dataset */
    AzX::throw_if(lap.dsno >= pp.ia_dscc.size(), AzInputError, eyec, "dsno is out of range"); 
    mypp.cc = pp.ia_dscc[lap.dsno]; 
  }
  return wei_setup(azp, mypp, pfx, is_warmstart, for_testonly); 
} 

/*------------------------------------------------------------*/  
void AzpReLayer_Wei_::check_dropout(const AzpReLayer_Param &pp, const char *eyec) const {
  if (!cs.dropout->is_active()) return; 
  AzX::no_support(pp.is_spa, eyec, "Dropout for sparse input"); 
  AzX::no_support(!pp.is_spa && pp.cc2 > 0, eyec, "Dropout for a layer with dense input and side input"); 
}

/*------------------------------------------------------------*/ 
void AzpReLayer_Wei_::upward(bool is_test, const AzDataArr<AzpDataVar_X> &dataarr, AzPmatVar &mv_out, const AzPmatVar *mv2) {
  int dsno = MAX(0, lap.dsno);  /* default: 0 */
  AzX::throw_if(dsno>=dataarr.size(), AzInputError, "AzpReLayer_Wei_::upward", "dsno is out of range."); 
  const AzpDataVar_X *data = dataarr[dsno]; 
  if (data->is_den()) {
    upward(is_test, *data->den(), mv_out, mv2); 
    return; 
  }
  wei_upward_sparse(is_test, *data->spa(), mv_out, mv2);      
  if (!is_re()) {
    cs.pool_var->upward(is_test, &mv_out); 
    cs.resnorm->upward(is_test, mv_out.data_u());     
  }
}          

/*------------------------------------------------------------*/ 
void AzpReLayer_Wei_::upward(bool is_test, const AzPmatVar &mv_below, AzPmatVar &mv_out, const AzPmatVar *mv2) {
  if (cs.dropout->is_active()) {
    AzPmatVar mv_do(&mv_below); 
    cs.dropout->upward(is_test, mv_do.data_u()); 
    wei_upward_dense(is_test, mv_do, mv_out, mv2); 
  }
  else {
    wei_upward_dense(is_test, mv_below, mv_out, mv2);  
  }
  if (!is_re()) {
    cs.pool_var->upward(is_test, &mv_out); 
    cs.resnorm->upward(is_test, mv_out.data_u()); 
  }
}    

/*------------------------------------------------------------*/   
void AzpReLayer_Wei_::downward(const AzPmatVar &mv_loss_deriv, bool dont_update, bool dont_release_sv) {
  if (is_re()) {
    AzPmatVar mv_ld; 
    upper->get_ld(layer_no, mv_ld); 
    wei_downward(mv_ld, dont_update);  
  }
  else {
    if (is_top()) mv_ld_x.set(&mv_loss_deriv);   
    else          upper->get_ld(layer_no, mv_ld_x); 
    cs.resnorm->downward(mv_ld_x.data_u()); 
    cs.pool_var->downward(&mv_ld_x); 
    wei_downward(dont_update); 
  }
  if (!dont_release_sv) release_sv(); 
}      

/*------------------------------------------------------------*/ 
void AzpReLayer_Wei_::get_ld(int id, AzPmatVar &mv_lossd_a, bool do_x2) const {
  mv_lossd_a.reform(1, mv_ld_x.d_index()); 
  if (!do_x2) wei_x->downward(mv_ld_x.data(), mv_lossd_a.data_u());
  else {
    AzX::throw_if(wei_x2 == NULL || wei_x2->classNum() <= 0, "AzpReLayer_Wei_::get_ld", "No x2 weights"); 
    wei_x2->downward(mv_ld_x.data(), mv_lossd_a.data_u());    
  }
  mv_lossd_a.check_colNum("mv_lossd_a in AzpReLayer_Wei_::get_ld");  
  cs.dropout->downward(mv_lossd_a.data_u());
}

/*------------------------------------------------------------*/
int AzpReLayer_Fc::wei_setup(AzParam &azp, const AzpReLayer_Param &pp, const AzPfx &pfx, 
                          bool is_warmstart, bool for_testonly) {
  const char *eyec = "AzpReLayer_Fc::wei_setup"; 
 
  if (!for_testonly) {
    wei_x->resetParam(out, azp, pfx, is_warmstart); 
  }
  if (!is_warmstart) {
    bool is_var = false; 
    wei_x->reset(-1, lap.nodes, pp.cc, pp.is_spa, is_var); 
    init_x_weights();     
  }
  else {
    int nodes = (is_top()) ? pp.class_num : lap.nodes; 
    AzX::throw_if((nodes != lap.nodes), AzInputError, eyec, "#node conflict"); 
    AzX::throw_if((wei_x->classNum() != nodes || wei_x->get_dim() != pp.cc), 
                    AzInputError, eyec, "Conflict in the weight dimensions");    
  }
  if (!is_top()) act_x->resetParam(out, azp, pfx, is_warmstart); /* no activation for top layer */  
  act_x->reset(out);   
  show_weight_dim(wei_x);  
  return lap.nodes; 
}                     

/*------------------------------------------------------------*/
void AzpReLayer_Fc::migrate(AzFile *file, bool is_lay0) {
  const char *eyec = "AzpReLayer_Fc::migrate"; 
  cs.check_if_ready(eyec); 
  /**** read exactly as AzpLayer::read0 does  ****/
  if (is_lay0) dicc.read(file); 

  /*---  read exactly as AzpLayer::read does  ---*/  
  AzTools::read_header(file, 63);     
  bool do_var_top = file->readBool(); 
  bool do_topthru = file->readBool(); 
  bool using_lm2 = file->readBool(); 
  bool is_var = file->readBool(); 
  layer_no = file->readInt(); 
  /*---  read exactly as AzpLayerParam::read does  ---*/
  AzTools::read_header(file, 63);
  bool do_patch_later = file->readBool(); 
  bool do_average_spa = file->readBool();
  bool do_activate_after_pooling = file->readBool();  
  lap.nodes = file->readInt();  
  cs.read(file);  
  init_wei(); 
  init_act();
  cs.weight->destroy(); 
  
  AzX::throw_if(do_var_top || do_topthru || using_lm2, eyec, "Unsupported options (1)"); 
  AzX::throw_if(do_patch_later || do_average_spa || do_activate_after_pooling, eyec, "Unsupported options (2)");   
}

/*------------------------------------------------------------*/ 
void AzpReLayer_Wei_::save_input(bool is_test, const AzPmatVar &mv, const AzPmatVar *mv2) { 
  msv_sv_x.reset(); mv_sv_x.reset(); mv_sv_x2.reset(); 
  if (is_test) return; 
  mv_sv_x.set(&mv); 
  if (mv2 != NULL) mv_sv_x2.set(mv2); 
}
void AzpReLayer_Wei_::save_input(bool is_test, const AzPmatSpaVar &msv, const AzPmatVar *mv2) { 
  msv_sv_x.reset(); mv_sv_x.reset(); mv_sv_x2.reset(); 
  if (is_test) return; 
  msv_sv_x.set(&msv); 
  if (mv2 != NULL) mv_sv_x2.set(mv2);  
}

/*------------------------------------------------------------*/  
template <class M>  /* M: AzPmatVar | AzPmatSpaVar */
/* determine which time steps should be processed concurrently */
void AzpReLayer_Wei_::set_order(const M &mv_below, 
                               int patch, int stride, bool do_backward, 
                               /*---  output  ---*/
                               AzDataArr<AzIntArr> &oaia_inp,
                               AzDataArr<AzIntArr> *oaia_out, 
                               /*---  additional input  ---*/
                               bool do_align_to_end) {
  const char *eyec = "AzpReLayer_Wei_::set_order"; 
  int data_num = mv_below.dataNum(); 
  int last_len = -1, min_len = -1, max_len = -1; 
  bool is_sorted = true; 
  for (int dx = 0; dx < data_num; ++dx) {
    int len = mv_below.get_end(dx) - mv_below.get_begin(dx); 
    AzX::throw_if((len < 0), eyec, "negative length data?!"); 
    if (dx > 0 && len > last_len) is_sorted = false;  
    if (dx == 0) min_len = max_len = len; 
    else       { min_len = MIN(len, min_len); max_len = MAX(len, max_len); }
    last_len = len; 
  }   
  
  AzX::no_support(do_align_to_end && patch > 0, eyec, "AlignToEnd with patches"); 
  
  /*---  use sequences as they are  ---*/
  if (patch <= 0) {  /* no chopping */
    oaia_inp.reset(max_len+1); 
    AzIntArr ia_dxs;  
    const int *dxs = NULL; 
    if (!is_sorted) {
      AzIFarr ifa; 
      for (int dx = 0; dx < data_num; ++dx) ifa.put(dx, mv_below.get_end(dx)-mv_below.get_begin(dx)); 
      ifa.sort_FloatInt(false, true); ifa.int1(&ia_dxs); dxs = ia_dxs.point(); 
    } 
    for (int ix = 0; ix < data_num; ++ix) {
      int dx = (dxs!=NULL)?dxs[ix]:ix, beg=mv_below.get_begin(dx), end=mv_below.get_end(dx), len=end-beg; 
      int offs = (do_align_to_end) ? (max_len-len) : 0; 
      if (do_backward) for (int pos=0; pos<len; ++pos) oaia_inp(offs+pos)->put(end-pos-1);  /* going backward */
      else             for (int pos=0; pos<len; ++pos) oaia_inp(offs+pos)->put(beg+pos);    /* goind forward */          
    }
  }  
  /*---  chop sequences into patches  ---*/
  else if (min_len == max_len && max_len%patch == 0 && stride <= 0) { /* all same length before/after chopping */
    oaia_inp.reset(patch+1); 
    for (int dx = 0; dx < data_num; ++dx) {
      int beg = mv_below.get_begin(dx), end = mv_below.get_end(dx), len = end - beg; 
      if (do_backward) for (int pos=0; pos<len; ++pos) oaia_inp(pos%patch)->put(end-pos-1);  /* going backward */
      else             for (int pos=0; pos<len; ++pos) oaia_inp(pos%patch)->put(beg+pos);    /* goind forward */   
    } 
  }
  else if (stride <= 0) { /* no segment overlap */
    AzDataArr<AzIntArr> aia(DIVUP(max_len,patch)*data_num); 
    int idx = 0; 
    for (int dx = 0; dx < data_num; ++dx) {
      int beg = mv_below.get_begin(dx), end = mv_below.get_end(dx), len = end - beg; 
      if (do_backward) for (int pos=0; pos<len; ++pos) aia(pos/patch+idx)->put(end-pos-1); 
      else             for (int pos=0; pos<len; ++pos) aia(pos/patch+idx)->put(beg+pos); 
      idx += DIVUP(len,patch); 
    }
    AzIFarr ifa; 
    for (int ix = 0; ix < idx; ++ix) ifa.put(ix, aia[ix]->size());  /* segment length */
    ifa.sort_FloatInt(false, true);  /* long segment first */
    oaia_inp.reset((int)ifa.get(0)+1); 
    for (int ix = 0; ix < ifa.size(); ++ix) {
      int xx = ifa.getInt(ix); 
      for (int pos = 0; pos < aia[xx]->size(); ++pos) oaia_inp(pos)->put((*aia[xx])[pos]); 
    }
  }
  else { /* segments overlap with each other */
    AzX::no_support(oaia_out==NULL, eyec, "overlapping regions in this setting ... "); 

    AzDataArr<AzIntArr> aia_inp((DIVUP(max_len-patch,stride)+1)*data_num); 
    AzDataArr<AzIntArr> aia_out(aia_inp.size()); 
    AzIntArr ia_taken(mv_below.colNum(), 0); 
    int idx = 0; 
    for (int dx = 0; dx < data_num; ++dx) {
      int beg = mv_below.get_begin(dx), end = mv_below.get_end(dx), len = end - beg; 
      for (int px=0, offs=0; px < DIVUP(len-patch,stride)+1; ++px, offs += stride) {
        for (int rpos = 0; rpos < patch; ++rpos) {
          int pos = offs + rpos; 
          if (pos >= len) continue; 
          int col = (do_backward) ? end-pos-1 : beg+pos; 
          aia_inp(idx)->put(col); 
          aia_out(idx)->put((ia_taken[col]==0)?col:-1); 
          ia_taken(col, 1);                   
        }
        ++idx; 
      }
    }
    AzX::throw_if(ia_taken.min() == 0, eyec, "Something is wrong ... uncovered position(s)??"); 

    AzIFarr ifa; 
    for (int ix = 0; ix < idx; ++ix) ifa.put(ix, aia_inp[ix]->size()); 
    ifa.sort_FloatInt(false, true);  
    oaia_inp.reset((int)ifa.get(0)+1); oaia_out->reset(oaia_inp.size()); 
    for (int ix = 0; ix < ifa.size(); ++ix) {
      int xx = ifa.getInt(ix); 
      for (int pos = 0; pos < aia_inp[xx]->size(); ++pos) {
        oaia_inp(pos)->put((*aia_inp[xx])[pos]); 
        (*oaia_out)(pos)->put((*aia_out[xx])[pos]); 
      }
    }
  }
} 
template void AzpReLayer_Wei_::set_order<AzPmatVar>(const AzPmatVar &, int, int, bool, AzDataArr<AzIntArr> &, AzDataArr<AzIntArr> *, bool); 
template void AzpReLayer_Wei_::set_order<AzPmatSpaVar>(const AzPmatSpaVar &, int, int, bool, AzDataArr<AzIntArr> &, AzDataArr<AzIntArr> *, bool); 

/*------------------------------------------------------------*/  
/* static */
void AzpReLayer_Wei_::pass_up(const AzPmat &m_curr, AzPmat &m_next, int next_cnum) {
  if (next_cnum <= 0) return; 
  if (next_cnum <= m_curr.colNum()) { /* aligned to the beginning */
    m_next.set(&m_curr, 0, next_cnum); 
  }
  else { /* aligned to the end */
    m_next.reform(m_curr.rowNum(), next_cnum); 
    AzIntArr ia_cols; ia_cols.range(0, m_curr.colNum()); 
    m_next.copy_dcol(&m_curr, ia_cols);     
  }
}                              

/*------------------------------------------------------------*/ 
/* static */
void AzpReLayer_Wei_::pass_down(const AzPmat &m_next, AzPmat &m_curr) {
  int cnum = MIN(m_curr.colNum(), m_next.colNum()); 
  if (cnum <= 0) return; 
  m_curr.add(0, cnum, &m_next); 
}  

/*------------------------------------------------------------*/  
/*------------------------------------------------------------*/  
template <class M> 
void AzpReLayer_Fc::_upward(bool is_test, const M &mv_below, AzPmatVar &mv_out) {
  save_input(is_test, mv_below); 
  mv_out.reform(1, mv_below.d_index());   
  wei_x->upward(is_test, mv_below.data(), mv_out.data_u()); 
  mv_out.check_colNum("mv_out in AzpReLayer_Fc::_upward(dense)");   
  act_x->upward(is_test, mv_out.data_u());   
}
template void AzpReLayer_Fc::_upward<AzPmatVar>(bool, const AzPmatVar &, AzPmatVar &); 
template void AzpReLayer_Fc::_upward<AzPmatSpaVar>(bool, const AzPmatSpaVar &, AzPmatVar &); 

/*------------------------------------------------------------*/  
/* mv_ld_x is set by AzpReLayer_Wei_::downward */
void AzpReLayer_Fc::wei_downward(bool dont_update) {
  act_x->downward(mv_ld_x.data_u());  
  if (!dont_update) {
    if      (mv_sv_x.colNum() > 0)  wei_x->updateDelta(mv_sv_x.dataNum(), mv_sv_x.data(), mv_ld_x.data()); 
    else if (msv_sv_x.colNum() > 0) wei_x->updateDelta(msv_sv_x.dataNum(), msv_sv_x.data(), mv_ld_x.data());  
    else                            AzX::throw_if(true, "AzpReLayer_Fc::wei_downward", "No saved input"); 
  }
}

/*------------------------------------------------------------*/    
template <class X> /* X: AzDataArr<AzpDataVar_X> | AzPmatVar */
void AzpReLayer_ComboH_::_upward(bool is_test, const X &data, AzPmatVar &mv_out, const AzPmatVar *mv2) {
  if (p.do_multi) am_sv.reset(lp.size()); 
  iia_rows.reset(); 
  for (int ix = 0; ix < lp.size(); ++ix) {
    const AzPmatVar *mymv2 = mv2; 
    AzPmatVar mv; 
    if (p.do_split_side && mv2 !=  NULL) {
      int rnum = mv2->rowNum() / lp.size(); 
      mv.reform(rnum, mv2->d_index()); 
      mv.data_u()->set_rowwise(0, rnum, mv2->data(), rnum*ix); 
      mymv2 = &mv; 
    }
    
    int row_begin = mv_out.rowNum(); 
    if (ix == 0) {
      lp[ix]->upward(is_test, data, mv_out, mymv2); 
      if (p.do_multi && !is_test) am_sv(ix)->set(mv_out.data()); 
    }
    else {
      AzPmatVar mv; 
      lp[ix]->upward(is_test, data, mv, mymv2); 
      if (p.do_sum||p.do_avg) { mv_out.add(&mv); row_begin = 0; }
      else if (p.do_multi)    { mv_out.data_u()->elm_multi(mv.data()); am_sv(ix)->set(mv.data()); row_begin = 0; }
      else /* concat */         mv_out.rbind(&mv); 
    }
    int row_end = mv_out.rowNum(); 
    iia_rows.put(row_begin, row_end); 
  }  
  if (p.do_avg) mv_out.data_u()->divide(lp.size()); 
}
template void AzpReLayer_ComboH_::_upward< AzDataArr<AzpDataVar_X> >(bool, const AzDataArr<AzpDataVar_X> &, AzPmatVar &, const AzPmatVar *); 
template void AzpReLayer_ComboH_::_upward<AzPmatVar>(bool, const AzPmatVar &, AzPmatVar &, const AzPmatVar *); 

/*------------------------------------------------------------*/   
void AzpReLayer_ComboH_::downward(const AzPmatVar &mv_loss_deriv, bool dont_update, bool dont_release_sv) {
  if (is_top()) mv_lossd.set(&mv_loss_deriv); 
  else          upper->get_ld(layer_no, mv_lossd);  
  for (int ix = 0; ix < lp.size(); ++ix) lp[ix]->downward(mv_loss_deriv, dont_update, dont_release_sv); 
}

/*------------------------------------------------------------*/   
void AzpReLayer_ComboH_::get_ld(int id, AzPmatVar &mv_out, bool do_x2) const {
 
  int index = -1; 
  for (index = 0; index < lp.size(); ++index) if (lp[index]->id() == id) break; 
  if (index >= 0 && index < lp.size()) {
    /*---  serving as the upper layer of the member layers  ---*/
    int row_begin, row_end; iia_rows.get(index, &row_begin, &row_end); 
    int rnum = row_end - row_begin;   
    mv_out.reform(rnum, mv_lossd.h_index()); 
    mv_out.data_u()->set_rowwise(0, rnum, mv_lossd.data(), row_begin); 
    if      (p.do_avg)     mv_out.data_u()->divide(lp.size()); 
    else if (p.do_multi)   mv_out.data_u()->elm_multi(am_sv[(index+1)%2]); /* combo of two is assumed */
  }
  else {
    AzX::no_support(p.do_split_side && do_x2, "AzpReLayer_ComboH_::get_ld", "UpdateSide with SplitSide"); 
 
    /*---  serving as the upper layer of the lower layer  ---*/
    for (int ix = 0; ix < lp.size(); ++ix) {
      if (ix == 0) lp[ix]->get_ld(layer_no, mv_out, do_x2); 
      else {
        AzPmatVar mv; lp[ix]->get_ld(layer_no, mv, do_x2); 
        mv_out.add(&mv); 
      }     
    }
  }
}  

/*------------------------------------------------------------*/   
int AzpReLayer_ComboH_::setup(AzParam &azp, const AzpReLayer_Param &pp, const AzPfx &pfx, 
                              bool is_warmstart, bool for_testonly) {
  const char *eyec = "AzpReLayer_ComboH_::setup"; 
  p.resetParam(out, azp, pfx, is_warmstart);  
  AzX::no_support(p.do_multi && lp.size()!=2, eyec, "Combination of more (or less) than 2 with the Multi option"); 
  int out_cc = 0; 
  for (int ix = 0; ix < lp.size(); ++ix) {
    AzPfx mypfx; gen_prefix_combo(ix, pfx, mypfx);   
    AzpReLayer_Param my_pp = pp; 
    const AzpReUpperLayer_ *my_upper = this; 
    my_pp.lno = 100*(pp.lno+1)+ix;
    if (p.do_split_side && my_pp.cc2 > 0) {
      AzX::throw_if(my_pp.cc2 % lp.size() != 0, eyec, "With SplitSide on, side #row must be a multiple of the number of member layers."); 
      my_pp.cc2 = pp.cc2 / lp.size(); 
    }
    int occ = 0;     
    if (!is_warmstart) {
      AzBytArr s_layfn; 
      azp.reset_prefix(mypfx.pfx()); 
      #define kw_layfn "layer_fn="
      azp.vStr(kw_layfn, &s_layfn); 
      azp.reset_prefix();    
      occ = lp[ix]->coldstart(&s_layfn, out, my_upper, azp, my_pp, &mypfx); 
    }
    else {
      occ = lp[ix]->warmstart(out, my_upper, azp, my_pp, for_testonly, sv_do_override_lno, &mypfx); 
    }
    if (ix == 0) out_cc = occ; 
    else {
      if (p.do_concat()) out_cc += occ; 
      else AzX::throw_if(occ != out_cc, eyec, "With non-concat option, the output dimensions of combined layers must match."); 
    }
  }
  return out_cc; 
}                  

/*------------------------------------------------------------*/ 
/*------------------------------------------------------------*/ 
#define az_LSTM_num 4
/*------------------------------------------------------------*/ 
void AzpReLayer_LSTM::init_do_ifuo() {
  ado_ifuo.free_alloc(4, true); 
  ado_ifuo(_i_, p.do_i); ado_ifuo(_f_, p.do_f); ado_ifuo(_o_, p.do_o); 
  ifuo_num = 0; 
  for (int ix = 0; ix < ado_ifuo.size(); ++ix) if (ado_ifuo[ix]) ++ifuo_num; 
}
/*------------------------------------------------------------*/ 
int AzpReLayer_LSTM::wei_setup(AzParam &azp, const AzpReLayer_Param &pp, const AzPfx &pfx, bool is_warmstart, bool for_testonly) {
  const char *eyec = "AzpReLayer_LSTM::wei_setup";                           
  check_if_ready(eyec); 

  AzX::throw_if(is_top(), eyec, "A LSTM layer cannot be the top layer");  
  p.resetParam(out, azp, pfx, is_warmstart); 
  init_do_ifuo(); 
  
  for (int ix = 0; ix < act.size(); ++ix) {
    AzPfx mypfx; mypfx.put(pfx, act_pfx[ix]->c_str()); 
    if (!is_warmstart) {
      if (act_pfx[ix]->contains("gate")) act(ix)->set_default_type("Log"); /* gate */
      else                               act(ix)->set_default_type("Tanh");  /* main */
    }
    act(ix)->resetParam(out, azp, mypfx, is_warmstart); 
    act(ix)->reset(out); 
  }
  
  if (!for_testonly) {
    for (int ix = 0; ix < wei.size(); ++ix) {
      if (ix == 0) wei(ix)->resetParam(out, azp, pfx); 
      else         wei(ix)->resetParam(azp, pfx); /* no print to avoid repeating the same thing */  
      AzBytArr s(pfx.pfx(), wei_pfx[ix]->c_str());     
      if (strstr(azp.c_str(), s.c_str())) { AzPfx mypfx(s); wei(ix)->resetParam(out, azp, mypfx); }
      if (ix != 0) wei(ix)->force_no_intercept(); 
    }  
  }
  if (!is_warmstart) {
    bool is_var = false, is_spa = false; 
    wei_x->reset(-1, lap.nodes*ifuo_num, pp.cc, pp.is_spa, is_var); /* i,f,u,o */
    wei_h->reset(-1, lap.nodes*ifuo_num, lap.nodes, is_spa, is_var); /* i,f,u,o */    
    if (pp.cc2 > 0) {
      wei_x2->reset(-1, lap.nodes*ifuo_num, pp.cc2, is_spa, is_var); 
    }
  }
  for (int ix = 0; ix < wei.size(); ++ix) {    
    if (wei[ix]->classNum() <= 0) continue; 
    AzBytArr s(pfx.pfx(), wei_pfx[ix]->c_str()); 
    AzPrint::writeln(out, "   ", s.c_str()); 
    show_weight_dim(wei[ix]);   
  }
  if (!is_warmstart) {
    for (int ix = 0; ix < wei.size(); ++ix) wei(ix)->initWeights();  
    if (lap.s_iw_fn.length() > 0) init_x_weights(); 
  }
  
  return lap.nodes; 
}                          

/*------------------------------------------------------------*/  
template <class M>  /* M: AzPmatVar | AzPmatSpaVar */
int AzpReLayer_LSTM::init_work(const M &mv_below, bool is_test) {
  const char *eyec = "AzpReLayer_LSTM::init_work"; 
  int patch = (is_test && !p.do_test_with_patch) ? -1 : p.patch; /* size of segments */  
  int stride = (patch > 0) ? p.stride : -1;  /* overlap = patch-stride */
  AzDataArr<AzIntArr> aia_icols, aia_ocols; 
  set_order(mv_below, patch, stride, do_backward, aia_icols, &aia_ocols, p.do_align_to_end); 
  aw.reset(aia_icols.size()); 
  int num = 0; 
  for (int pos = 0; pos < aw.size(); ++pos) {
    aw(pos)->set_cols(pos, aia_icols, &aia_ocols); 
    num += aw[pos]->colNum(); 
  }
  io.reset(); 
  if (p.do_less_traffic) {
    io.reset(aia_icols, &aia_ocols);  
  }
    
  for (int pos = 0; pos < aw.size(); ++pos) {
    aw(pos)->act.free_alloc(az_LSTM_num); 
    for (int ix = 0; ix < az_LSTM_num; ++ix) {
      if (ix == _u_) aw(pos)->act.set(ix, act_x->clone()); /* u */
      else           aw(pos)->act.set(ix, act_g->clone()); /* gate */
    }
    aw(pos)->act_ct.free_alloc(1); 
    aw(pos)->act_ct.set(0, act_c->clone()); /* c */
  }
  return num; 
}
template int AzpReLayer_LSTM::init_work<AzPmatVar>(const AzPmatVar &, bool); 
template int AzpReLayer_LSTM::init_work<AzPmatSpaVar>(const AzPmatSpaVar &, bool); 

/*------------------------------------------------------------*/  
template <class M>  /* M: AzPmatVar | AzPmatSpaVar */
void AzpReLayer_LSTM::_upward(bool is_test, const M &mv_below, AzPmatVar &mv_out, const AzPmatVar *mv2) {
  const char *eyec = "AzpReLayer_LSTM::_upward";  

  save_input(is_test, mv_below, mv2);
  int h_num = init_work(mv_below, is_test); 
  int max_len = aw.size() - 1; 
  int num0 = aw[0]->colNum();   
  AzPmat m_vx; wei_x->upward(is_test, mv_below.data(), &m_vx); /* W_x*x + b for i,f,u,o */
  if (mv2 != NULL) { /* input from side layers */
    AzPmat m_vx2; wei_x2->upward(is_test, mv2->data(), &m_vx2);  
    m_vx2.shape_chk_tmpl(&m_vx, eyec, "m_vx2"); 
    m_vx.add(&m_vx2); 
  }

  mv_out.reform(lap.nodes, mv_below.d_index()); 
  aw(0)->m_c.reform(lap.nodes, num0); 
  if (!is_test) m_sv_h.reform(lap.nodes, h_num); 
  AzPmat m_h(lap.nodes, num0); 
  int hx = 0; 
  for (int pos = 0; pos < max_len; ++pos) {  /* time step.  position in segments */
    AzpReLayer_LSTM_Work *cw = aw(pos), *nw = aw(pos+1); 
    
    if (!is_test) {
      m_sv_h.set(hx, m_h.colNum(), &m_h); 
      hx += m_h.colNum(); 
    }
    
    cw->am.reset(az_LSTM_num); 
    AzPmat m_ifuo; wei_h->upward(is_test, &m_h, &m_ifuo); /* W_h*h_{t-1} for i,f,u,o */
    if (io.on()) m_ifuo.add_d2s(&m_vx, io.icols(), pos); /* W_x*x+W_h*h_{t-1}+b for i,f,u,o */   
    else         m_ifuo.add_d2s(&m_vx, cw->icols());     /* W_x*x+W_h*h_{t-1}+b for i,f,u,o */

    int rnum = lap.nodes;     
    for (int ix=0, jx=0; ix<cw->am.size(); ++ix) { /* separate i,f,c,o and activate */
      if (!ado_ifuo[ix]) continue; 
      AzPmat *m = cw->am(ix); 
      m->reform(rnum, m_ifuo.colNum()); 
      m->set_rowwise(0, rnum, &m_ifuo, rnum*jx);    
      cw->act(ix)->upward(is_test, m); 
      ++jx; 
    }
    AzPmat m_u_i(cw->am[_u_]); if (p.do_i) m_u_i.elm_multi(cw->am[_i_]); /* i_t & u_t     */
    cw->m_ct.set(&cw->m_c); if (p.do_f) cw->m_ct.elm_multi(cw->am[_f_]);     /* f_t & c_{t-1} */
    cw->m_ct.add(&m_u_i);                                    /* m_ct = i_t & u_t + f_t & c_{t-1} = c_t */ 
    pass_up(cw->m_ct, nw->m_c, nw->colNum()); 
    
    cw->m_ct_act.set(&cw->m_ct); cw->act_ct(0)->upward(is_test, &cw->m_ct_act);  /* act(c_t) */
    m_h.set(&cw->m_ct_act); if (p.do_o) m_h.elm_multi(cw->am[_o_]); /* act(c_t) & o_t = h_t */  
    if (io.on()) mv_out.data_u()->copy_dcol(&m_h, io.ocols(), pos, true); /* set output */
    else         mv_out.data_u()->copy_dcol(&m_h, cw->ocols(), true);   /* set output */
    m_h.resize(nw->colNum());  
    
    stat_accum(cw); 
    if (is_test) cw->reset(); /* release memory */
  }
  if (!is_test) AzX::throw_if(hx != h_num, eyec, "something is wrong with # of columns of saved h"); 
}
template void AzpReLayer_LSTM::_upward<AzPmatVar>(bool, const AzPmatVar &, AzPmatVar &, const AzPmatVar *); 
template void AzpReLayer_LSTM::_upward<AzPmatSpaVar>(bool, const AzPmatSpaVar &, AzPmatVar &, const AzPmatVar *); 

/*------------------------------------------------------------*/  
int AzpReLayer_LSTM::how_far_bprop() const {
  if (p.bprop_max <= 0) return 0; /* all the way to position 0 */
  return MAX(0, aw.size()-1 - p.bprop_max); 
}

/*------------------------------------------------------------*/  
void AzpReLayer_LSTM::wei_downward(const AzPmatVar &mv_out_lossd, bool dont_update) { 
  const char *eyec = "AzpReLayer_LSTM::_downward"; 
  AzX::throw_if((mv_sv_x.colNum() <= 0 && msv_sv_x.colNum() <= 0), eyec, "No saved input"); 
  AzX::throw_if((wei_x2->classNum()>0 && mv_sv_x2.colNum() <= 0), eyec, "No saved input for side"); 

  int max_len = aw.size()-1; 

  const AzPintArr *pia_ind = (mv_sv_x.colNum() > 0) ? mv_sv_x.d_index() : msv_sv_x.d_index();   
  mv_ld_x.reform(lap.nodes*ifuo_num, pia_ind);  /* w.r.t. W_x*x */
  AzPmat m_ld_h, m_ld_c; /* w.r.t. h, w.r.t. c */
  
  AzPmat m_ld_h_all(lap.nodes*ifuo_num, m_sv_h.colNum());  /* i,f,u,o */
  int hx = m_ld_h_all.colNum(); 

  AzDataArr<AzPmat> amld(az_LSTM_num); 
  AzPmat *m_i=amld(_i_), *m_f=amld(_f_), *m_u=amld(_u_), *m_o=amld(_o_);   
  int last_pos = how_far_bprop(); 
  for (int pos = max_len-1; pos >= last_pos; --pos) {
    AzpReLayer_LSTM_Work *cw = aw(pos); 
    AzPmat m_h; 
    if (io.on()) m_h.set(mv_out_lossd.data(), io.ocols(), pos); /* from the output */
    else         m_h.set(mv_out_lossd.data(), cw->ocols());  /* from the output */ 
    pass_down(m_ld_h, m_h); 
    
    AzPmat m_ct(&m_h); if (p.do_o) m_ct.elm_multi(cw->am[_o_]); 
    cw->act_ct(0)->downward(&m_ct); 
    pass_down(m_ld_c, m_ct); 
   
    if (p.do_o) { m_o->set(&m_h);  m_o->elm_multi(&cw->m_ct_act);}
    if (p.do_i) { m_i->set(&m_ct); m_i->elm_multi(cw->am[_u_]);}
    if (p.do_f) { m_f->set(&m_ct); m_f->elm_multi(&cw->m_c);} 
    m_u->set(&m_ct); if (p.do_i) m_u->elm_multi(cw->am[_i_]);
    int rnum = lap.nodes; 
    AzPmat m_ifuo(rnum*ifuo_num, m_ct.colNum()); 
    for (int ix=0, jx=0; ix<amld.size(); ++ix) {
      if (!ado_ifuo[ix]) continue; 
      cw->act(ix)->downward(amld(ix)); 
      m_ifuo.set_rowwise(rnum*jx, rnum, amld(ix), 0); 
      ++jx; 
    }
    
    m_ld_h_all.set(hx - m_ifuo.colNum(), m_ifuo.colNum(), &m_ifuo); 
    hx -= m_ifuo.colNum(); 

    wei_h->downward(&m_ifuo, &m_ld_h);  /* w.r.t. h_{t-1} */
    if (io.on()) mv_ld_x.data_u()->add_s2d(&m_ifuo, io.icols(), pos); /* w.r.t. W_x*x */
    else         mv_ld_x.data_u()->add_s2d(&m_ifuo, cw->icols());  /* w.r.t. W_x*x */
    m_ld_c.set(&m_ct); if (p.do_f) m_ld_c.elm_multi(cw->am[_f_]);     
  }  

  if (!dont_update) {
    if (mv_sv_x.colNum() > 0) wei_x->updateDelta(mv_ld_x.colNum(), mv_sv_x.data(), mv_ld_x.data()); 
    else                      wei_x->updateDelta(mv_ld_x.colNum(), msv_sv_x.data(), mv_ld_x.data()); 
    if (mv_sv_x2.colNum() > 0) wei_x2->updateDelta(mv_ld_x.colNum(), mv_sv_x2.data(), mv_ld_x.data()); 
    wei_h->updateDelta(m_ld_h_all.colNum(), &m_sv_h, &m_ld_h_all); 
  }
}

/*------------------------------------------------------------*/     
#define kw_do_vertical "VerticalSide"
/*------------------------------------------------------------*/  
int AzpReLayer_Side::side_coldstart(const AzOut &_out, const AzpReUpperLayer_ *_upper, AzParam &azp, const AzpReLayer_Param &pp, 
                                    const AzpCompoSet_ *cset, 
                                    const AzDataArr<AzBytArr> &as_laytype, const AzDataArr<AzBytArr> &as_layfn) {                                   
  layer_no = pp.lno;  /* the layer to which side layers are attached */ 
  out = _out; upper = _upper;  
  azp.swOn(&do_vertical, kw_do_vertical); 
  if (do_vertical) AzPrint::writeln(_out, "   ------  Cold-starting Side (vertical) ------");  
  else             AzPrint::writeln(_out, "   ------  Cold-starting Side  ------");      
  int side_num = as_laytype.size(); 
  lays->reset(side_num); 
  lp.free_alloc(side_num, (AzpReLayer_ *)NULL); 
  int out_cc = 0; 
  for (int lx = 0; lx < side_num; ++lx) {
    AzPfx sidepfx; gen_prefix_side(lx, sidepfx); 
    lays->reset(lx, as_laytype[lx], cset); 
    AzpReLayer_Param my_pp = pp; 
    const AzpReUpperLayer_ *my_upper = this; 
    my_pp.lno = lx;    
    if (!do_vertical) out_cc += (*lays)(lx)->coldstart(as_layfn[lx], _out, my_upper, azp, my_pp, &sidepfx);     
    else {
      if (lx > 0) my_upper = (*lays)[lx-1];
      else        my_pp.cc = out_cc; 
      out_cc = (*lays)(lx)->coldstart(as_layfn[lx], _out, my_upper, azp, my_pp, &sidepfx);
    }
    lp(lx, (*lays)(lx)); 
  }
  return out_cc; 
}

/*------------------------------------------------------------*/     
int AzpReLayer_Side::side_warmstart(const AzOut &_out, const AzpReUpperLayer_ *_upper, AzParam &azp, const AzpReLayer_Param &pp, 
                                    bool for_testonly) {
  const char *eyec = "AzpReLayer_Side::side_warmstart"; 
  AzX::throw_if(layer_no != pp.lno, eyec, "layer# conflict"); 
  out = _out; upper = _upper;  
  if (do_vertical) AzPrint::writeln(_out, "   ------  Warm-starting Side (vertical) ------");  
  else             AzPrint::writeln(_out, "   ------  Warm-starting Side  ------");      
  int side_num = lays->size(); 
  lp.free_alloc(side_num, (AzpReLayer_ *)NULL); 
  int out_cc = 0; 
  for (int lx = 0; lx < side_num; ++lx) {
    AzPfx sidepfx; gen_prefix_side(lx, sidepfx);     
    AzpReLayer_Param my_pp = pp; 
    const AzpReUpperLayer_ *my_upper = this; 
    my_pp.lno = lx; 
    bool do_override_lno = false; 
    if (!do_vertical) out_cc += (*lays)(lx)->warmstart(_out, my_upper, azp, my_pp, for_testonly, do_override_lno, &sidepfx); 
    else {
      if (lx > 0) my_upper = (*lays)[lx-1];
      else        my_pp.cc = out_cc; 
      out_cc = (*lays)(lx)->warmstart(_out, my_upper, azp, my_pp, for_testonly, do_override_lno, &sidepfx); 
    }
    lp(lx, (*lays)(lx)); 
  }
  return out_cc; 
}

/*------------------------------------------------------------*/    
void AzpReLayer_Side::side_upward(bool is_test, const AzDataArr<AzpDataVar_X> &data, AzPmatVar &mv_out) {
  const char *eyec = "AzpReLayer_Side::side_upward"; 
  if (do_vertical) {
    _side_upward_vertical(is_test, data, mv_out); 
    return;     
  }
  iia_rows.reset(); 
  for (int ix = 0; ix < lp.size(); ++ix) {
    int row_begin = mv_out.rowNum(); 
    if (ix == 0) lp[ix]->upward(is_test, data, mv_out); 
    else {
      AzPmatVar mv; 
      lp[ix]->upward(is_test, data, mv); 
      mv_out.rbind(&mv); 
    }
    int row_end = mv_out.rowNum(); 
    iia_rows.put(row_begin, row_end); 
  } 
}

/*------------------------------------------------------------*/    
void AzpReLayer_Side::_side_upward_vertical(bool is_test, const AzDataArr<AzpDataVar_X> &data, AzPmatVar &mv_out) {
  const char *eyec = "AzpReLayer_Side::_side_upward_vertical"; 
  for (int ix = 0; ix < lp.size(); ++ix) {
    if (ix == 0) lp[ix]->upward(is_test, data, mv_out); 
    else {
      AzPmatVar mv(&mv_out); 
      lp[ix]->upward(is_test, mv, mv_out); 
    }   
  } 
}

/*------------------------------------------------------------*/   
void AzpReLayer_Side::downward(const AzPmatVar &mv_loss_deriv, bool dont_update, bool dont_release_sv) {
  bool do_x2 = true; 
  upper->get_ld(layer_no, mv_lossd, do_x2);  
  for (int ix = lp.size()-1; ix >= 0; --ix) lp[ix]->downward(mv_loss_deriv, dont_update, dont_release_sv);  /* vertical needs this order */
}

/*------------------------------------------------------------*/
/* Fc with side */
/*------------------------------------------------------------*/
int AzpReLayer_FcS::wei_setup(AzParam &azp, const AzpReLayer_Param &pp, const AzPfx &pfx, 
                           bool is_warmstart, bool for_testonly) {
  const char *eyec = "AzpReLayer_FcS::wei_setup"; 
  int out_cc = AzpReLayer_Fc::wei_setup(azp, pp, pfx, is_warmstart, for_testonly); 
  AzX::throw_if(out_cc != lap.nodes, eyec, "something is wrong ... "); 
  if (pp.cc2 > 0) {
    if (!for_testonly) wei_x2->resetParam(out, azp, pfx, is_warmstart); 
    wei_x2->force_no_intercept();     
    if (!is_warmstart) {
      wei_x2->reset(-1, lap.nodes, pp.cc2, false, false); 
      AzPrint o(out); bool do_v2 = false; azp.swOn(o, do_v2, "V2"); 
      if (do_v2) x2_needs_init = true; /* delay x2 initialization to exactly match CNet3 */
      else     { x2_needs_init = false; wei_x2->initWeights(); }
    }
    show_weight_dim(wei_x2);  
  }
  return lap.nodes; 
} 
/*------------------------------------------------------------*/  
void AzpReLayer_FcS::v2_setup() {
  if (x2_needs_init) {
    wei_x2->initWeights();  /* initialization of x2 was delayed to exactly match CNet3 */
    x2_needs_init = false;       
  }
}    

/*------------------------------------------------------------*/  
template <class M> 
void AzpReLayer_FcS::_upward(bool is_test, const M &mv_below, AzPmatVar &mv_out, const AzPmatVar *mv2) {
  const char *eyec = "AzpReLayer_FcS::_upward"; 
  save_input(is_test, mv_below, mv2); 
  mv_out.reform(1, mv_below.d_index());   
  wei_x->upward(is_test, mv_below.data(), mv_out.data_u()); 
  if (mv2 != NULL) {
    AzX::throw_if(x2_needs_init, eyec, "x2 weights are not initialized."); 
    AzPmat m_vx2; wei_x2->upward(is_test, mv2->data(), &m_vx2);  
    m_vx2.shape_chk_tmpl(mv_out.data(), eyec, "m_vx2"); 
    mv_out.data_u()->add(&m_vx2); 
  }  
  mv_out.check_colNum("mv_out in AzpReLayer_FcS::_upward(dense)");   
  act_x->upward(is_test, mv_out.data_u());   
}
template void AzpReLayer_FcS::_upward<AzPmatVar>(bool, const AzPmatVar &, AzPmatVar &, const AzPmatVar *); 
template void AzpReLayer_FcS::_upward<AzPmatSpaVar>(bool, const AzPmatSpaVar &, AzPmatVar &, const AzPmatVar *); 

/*------------------------------------------------------------*/  
/* mv_ld_x is set by AzpReLayer_Wei_::downward */
void AzpReLayer_FcS::wei_downward(bool dont_update) {
  act_x->downward(mv_ld_x.data_u());  
  if (!dont_update) {
    if      (mv_sv_x.colNum() > 0)  wei_x->updateDelta(mv_sv_x.dataNum(), mv_sv_x.data(), mv_ld_x.data()); 
    else if (msv_sv_x.colNum() > 0) wei_x->updateDelta(msv_sv_x.dataNum(), msv_sv_x.data(), mv_ld_x.data());  
    else                            AzX::throw_if(true, "AzpReLayer_FcS::_downward", "No saved input"); 
    if      (mv_sv_x2.colNum() > 0) wei_x2->updateDelta(mv_sv_x2.dataNum(), mv_sv_x2.data(), mv_ld_x.data()); 
  }
}

/* * * * *
 *  AzpReNet.cpp
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

#include "AzpReNet.hpp"
#include "AzPrint.hpp"
#include "AzDic.hpp"

extern bool __doDebug; 

/*---  global  ---*/
AzByte param_dlm = ' '; 
/*----------------*/
 
/*------------------------------------------------------------*/ 
/* NOTE: "cc" is # of feature maps for historical reasons ... */
int AzpReNet::setup_nomc(const AzpData_tmpl_ *trn, AzParam &azp, 
                         bool is_warmstart, bool for_testonly) {
  const char *eyec = "AzpReNet::setup_nomc";                       
  if (is_warmstart) {
    AzX::throw_if((lays->size() != hid_num+1), AzInputError, eyec, "#layer doesn't match"); 
  }
  else {
    lays->reset(hid_num+1);  /* +1 for the top layer */   
    for (int lx = 0; lx < hid_num; ++lx) lays->reset(lx, as_laytype[lx], cs); 
    if (do_topthru) lays->reset(hid_num, AzpReLayer_Type_Noop, cs);  
    else            lays->reset(hid_num, (AzBytArr *)NULL, cs); /* top layer */
  }

   /*---  side  ---*/
  int side_cc = -1;
  if (side_num > 0) {  
    int lno = 0;  /* the layer to which side layers are attached */
    AzpReLayer_Param side_pp(trn, trn->xdim(), trn->is_sparse_x(), lno); 
    if (is_warmstart) side_cc = side_lay->side_warmstart(out, (*lays)[0], azp, side_pp, for_testonly); 
    else              side_cc = side_lay->side_coldstart(out, (*lays)[0], azp, side_pp, cs, as_side_laytype, as_side_layfn); 
  }  
  
  /*---  main  ---*/
  AzpReLayer_Param pp(trn); 
  pp.cc = trn->xdim(); pp.is_spa = trn->is_sparse_x(); 
  for (int lx = 0; lx < hid_num; ++lx) {  
    pp.cc2 = (lx == 0) ? side_cc : -1; 
    pp.lno = lx; 
    if (is_warmstart) pp.cc = (*lays)(lx)->warmstart(out, (*lays)[lx+1], azp, pp, for_testonly); 
    else              pp.cc = (*lays)(lx)->coldstart(as_layfn[lx], out, (*lays)[lx+1], azp, pp);  
    pp.is_spa = false; 
  }
  pp.cc2=-1; pp.lno = hid_num; pp.class_num = class_num; 
  int outdim = 0; 
  if (is_warmstart) outdim = (*lays)(hid_num)->warmstart(out, NULL, azp, pp, for_testonly); 
  else              outdim = (*lays)(hid_num)->coldstart(&s_top_layfn, out, NULL, azp, pp); 
  if (hid_num > 0) (*lays)(0)->v2_setup(); /* for compatibility */
  return outdim; 
}

/*------------------------------------------------------------*/
void AzpReNet::_for_bottom(int lno, const AzpData_tmpl_ *trn, AzParam &azp, AzpReLayer_Param &pp) const {
  AzBytArr s_pfx; s_pfx << lno; 
  azp.reset_prefix(s_pfx.c_str()); 
  int dsno = 0; 
  azp.vInt(kw_dsno, &dsno); 
  azp.reset_prefix();   
  pp.cc = trn->xdim(MAX(0,dsno)); 
  pp.is_spa = trn->is_sparse_x(); 
}

/*------------------------------------------------------------*/ 
int AzpReNet::setup_mc(const AzpData_tmpl_ *trn, AzParam &azp, bool is_warmstart, bool for_testonly) {
  const char *eyec = "AzpReNet::setup_mc"; 
  /*---  allocate layers  ---*/
  if (is_warmstart) {
    AzX::throw_if(lays->size() != hid_num+1, AzInputError, eyec, "warmstart: something is wrong with # of layers."); 
  }
  else {
    lays->reset(hid_num+1);  /* +1 for the top layer */
    for (int lx = 0; lx < hid_num; ++lx) lays->reset(lx, as_laytype[lx], cs); 
    if (do_topthru) lays->reset(hid_num, AzpReLayer_Type_Noop, cs);  
    else            lays->reset(hid_num, (AzBytArr *)NULL, cs); /* top layer */
  }
    
  /*---  side  ---*/
  int side_cc = -1; 
  if (side_num > 0) {  
    int lno = 0;  /* the layer to which side layers are attached */
    AzpReLayer_Param side_pp(trn, trn->xdim(), trn->is_sparse_x(), lno); 
    if (is_warmstart) side_cc = side_lay->side_warmstart(out, (*lays)[0], azp, side_pp, for_testonly); 
    else              side_cc = side_lay->side_coldstart(out, (*lays)[0], azp, side_pp, cs, as_side_laytype, as_side_layfn); 
  }  

  /*---  ---*/
  mc.setup(is_warmstart); 
  const AzIntArr &ia_order = mc.order(); 
  int lsz = lays->size(), csz = ia_order.size() - lsz; 
  conns.reset(csz); 
  
  /*---  main  ---*/
  AzIntArr ia_out_cc(ia_order.size(), 0); 
  AzpReLayer_Param pp(trn); 
  pp.is_spa = trn->is_sparse_x(); 
  for (int ix = 0; ix < ia_order.size()-1; ++ix) { /* hidden layers and connectors */
    int lx = ia_order[ix]; 
    int cc = 0; 
    if (lx < lsz) { /* regular layer */
      int below = mc.below(lx), above = mc.above(lx);   
      AzX::throw_if(above < 0, eyec, "No above?!"); 
      const AzpReUpperLayer_ *upper = NULL; 
      if (above < lsz) upper = (*lays)[above]; 
      else             upper = conns[above-lsz]; 
      if (below >= 0) pp.cc = ia_out_cc[below];
      else            _for_bottom(lx, trn, azp, pp); 
      pp.cc2 = (lx == 0) ? side_cc : -1;  pp.lno = lx;   
      if (is_warmstart) cc = (*lays)(lx)->warmstart(out, upper, azp, pp, for_testonly); 
      else              cc = (*lays)(lx)->coldstart(as_layfn[lx], out, upper, azp, pp);  
    }
    else { /* connector */
      cc = conns(lx-lsz)->setup(out, lx, mc.all_below(lx), mc.all_above(lx), ia_out_cc, mc.is_additive());     
    }
    ia_out_cc.update(lx, cc); 
    pp.is_spa = false; 
  }
  
  /*---  top layer  ---*/
  int lx = ia_order[ia_order.size()-1]; 
  int below = mc.below(lx), above = mc.above(lx); 
  AzX::throw_if(lx != hid_num, eyec, "lx != hid_num?!"); 
  AzX::throw_if(below < 0, AzInputError, eyec, "A multi-connection network must have at least one hidden layer."); 
  AzX::throw_if(above >= 0, eyec, "The top layer must not have any layer above"); 
  pp.cc = ia_out_cc[below]; pp.cc2=-1; pp.lno = lx; pp.class_num = class_num; 
  int outdim = 0; 
  if (is_warmstart) outdim = (*lays)(lx)->warmstart(out, NULL, azp, pp, for_testonly); 
  else              outdim = (*lays)(lx)->coldstart(&s_top_layfn, out, NULL, azp, pp); 
  if (hid_num > 0) (*lays)(0)->v2_setup(); /* for compatibility */
  
  /*---  setup ups for use in downward  ---*/ 
  ups.free_alloc(lsz+csz); 
  for (int ix = 0; ix < lsz; ++ix) ups(ix, (const AzpReUpperLayer_ *)(*lays)[ix]); 
  for (int ix = lsz; ix < lsz+csz; ++ix) ups(ix, (const AzpReUpperLayer_ *)conns[ix-lsz]);   

  return outdim; 
}

/*------------------------------------------------------------*/ 
int AzpReNet::init(AzParam &azp, const AzpData_tmpl_ *trn, 
                    const AzpData_tmpl_ *tst, const AzpData_tmpl_ *tst2, 
                    bool is_alone) { /* stand alone */
  const char *eyec = "AzpReNet::init"; 
  
  bool is_warmstart = false;   
  if (lays->size() > 0) {
    is_warmstart = true;    
    check_data_signature(trn, "training data");
    AzTimeLog::print("Warm-start ... ", out); 
  }
  else {
    trn->signature(s_data_sign); 
  }  
  AzTimeLog::print("Data signature: ", s_data_sign.c_str(), out); 
  if (tst != NULL) check_data_signature(tst, "test data");
  if (tst2 != NULL) check_data_signature(tst2, "test data2");

  /*---  check #class (this must be done before resetParam) ---*/
  class_num = trn->ydim(); 
  AzTimeLog::print("#class=", class_num, out); 
  AzX::throw_if((tst != NULL && tst->ydim() != class_num), 
                AzInputError, eyec, "training data and test data have differnt number of classes"); 
  
  /*---  read parameters  ---*/ 
  resetParam(azp, is_warmstart, is_alone); 
  printParam(out, is_alone); 

  if (is_alone) {
    if (rseed > 0) srand(rseed);      /* reset random number generator */
  }
 
  /*---  initialize layers  ---*/
  int outdim = 0; 
  if (is_warmstart) outdim = warmstart(trn, azp); 
  else              outdim = coldstart(trn, azp); 
  azp.check(out); 
  
  /*---  check word-mapping set consistency  ---*/
  check_word_mapping(is_warmstart, trn, tst, tst2);  /* added on 1/16/2016 */
  
  /*---  ---*/
  timer_init(); 
  return outdim; 
}

/*------------------------------------------------------------*/ 
void AzpReNet::training(AzParam &azp, AzpData_ *trn, 
                      const AzpData_ *tst, const AzpData_ *tst2) {
  bool is_alone = true; /* this is a stand-alone version. */
  init(azp, trn, tst, tst2, is_alone); 
  sup_training_loop(trn, tst, tst2); 
}

/*------------------------------------------------------------*/ 
void AzpReNet::sup_training_loop(AzpData_ *trn,  /* not const b/c of first|next_batch */
                   const AzpData_ *tst, const AzpData_ *tst2) {
  AzTimeLog::print("supervised training: #hidden=", lays->size()-1, out);   

  nco.loss->check_target(out, trn);     
  dataseq.init(trn, out);
  reset_stepsize(&sssch);  
  show_layer_stat(); 
  if (do_test_first) {
    AzTimeLog::print("Testing first ... ", out); 
    int myite = 0; 
    double my_tr_loss = -1; 
    _tTr();     
    test_save(trn, tst, tst2, myite, my_tr_loss);     
    timer_show();     
  }
  if (do_save_mem) {
    if (tst != NULL) (const_cast<AzpData_ *>(tst))->release_batch();    
    if (tst2 != NULL) (const_cast<AzpData_ *>(tst2))->release_batch();     
  }
  trn->first_batch(); 
  
  for (int ite = init_ite; ite < ite_num; ++ite) {  /* epochs */
    AzClock clk(do_less_verbose); _tTr(); 
    int data_size = trn->dataNum();  /* size of this data batch */
    int eval_size = trn->is_vg_y() ? trn->colNum() : data_size; 
    const int *dxs = dataseq.gen(trn, data_size); 
    double tr_loss = 0; 
    int ix; 
    for (ix = 0; ix < data_size; ix += minib) { 
      if (max_data_num > 0 && ix >= max_data_num) { /* for debugging */
        AzTimeLog::print("#data hit the threshold ... ", out); 
        break;         
      }      
      int d_num = MIN(minib, data_size - ix);
      AzIntArr ia_dxs(dxs+ix, d_num); /* mini batch */
      up_down(trn, ia_dxs, &tr_loss);      
      show_progress(ix+d_num, dx_inc, data_size, eval_size, tr_loss); 
      if (dx_inc > 0 && (ix+d_num)%(dx_inc*10) == 0 && ix+minib < data_size) {
        show_layer_stat(); 
      }
      if (tr_loss != tr_loss) {
        AzPrint::writeln(out, "Detected nan ... "); 
        break;         
      }
    }
    lays_end_of_epoch();  
    clk.tick(out, "epo=", ite+1, ": ");     

    show_layer_stat(); 
    tr_loss /= (double)eval_size; 
    test_save(trn, tst, tst2, ite, tr_loss); 
    timer_show();  
    if (max_loss > 0 && tr_loss >= max_loss || tr_loss != tr_loss) {
      AzTimeLog::print("Stopping as loss seems to be exploding ... ", out); 
      AzX::throw_if(true, AzNormal, "", ""); 
    }    
    
    change_stepsize_if_required(&sssch, ite);       
    trn->next_batch(); 
  }
  if (ite_num == 0) save(0); 
}

/*------------------------------------------------------------*/ 
void AzpReNet::save(int iteno) {
  if (s_save_fn.length() > 0) {
    AzBytArr s_fn(s_save_fn); s_fn << ".epo" << iteno << ".ReNet";  /* 5/31/2017: ite -> epo, ".net" added  */
    AzTimeLog::print("Saving the model to ", s_fn.c_str(), out); 
    write(s_fn.c_str()); 
  }
  for (int lx = 0; lx < as_save_layfn.size(); ++lx) {
    if (as_save_layfn[lx]->length() <= 0) continue; 
    AzBytArr s_fn(as_save_layfn[lx]); s_fn << ".epo" << iteno << ".ReLayer" << lx; /* 5/31/2017: ite -> epo */
    AzBytArr s("Saving layer#"); s << lx << " to "; 
    AzTimeLog::print(s.c_str(), s_fn.c_str(), out); 
    (*lays)(lx)->write(s_fn.c_str()); 
  }
} 
 
/*------------------------------------------------------------*/ 
void AzpReNet::reset_stepsize(AzStepszSch *sssch) {
  sssch->init(); 
  AzTimeLog::print("Resetting step-sizes to s0 (initial value) ...", out); 
  double coeff = 1; 
  const AzOut *myout = (dmp_out.isNull()) ? NULL : &out;   
  lays_multiply_to_stepsize(coeff, myout); 
}

/*------------------------------------------------------------*/ 
void AzpReNet::change_stepsize_if_required(AzStepszSch *sssch, int ite) {
  double coeff = sssch->get_stepsize_coeff(ite+1); 
  if (coeff > 0) {
    AzBytArr s("Setting step-sizes to s0 times "); s.cn(coeff); 
    AzTimeLog::print(s, out); 
    const AzOut *myout = (dmp_out.isNull()) ? NULL : &out; 
    lays_multiply_to_stepsize(coeff, myout); 
  }
}
  
/*------------------------------------------------------------*/ 
void AzpReNet::test_save(const AzpData_ *trn, 
                       const AzpData_ *tst, 
                       const AzpData_ *tst2, 
                       int ite,
                       double loss) /* no reg */ {
  double iniloss = 0, regloss = -1; 
  double perf=-1, perf2=-1, exact_loss=-1, test_loss = -1, test_loss2 = -1; 
  int myite = ite - init_ite; 
  AzClock clk(do_less_verbose || myite != 0);
  AzBytArr s_pf; 
  bool did_test = false; 
  if (myite == 0 || ite == ite_num-1 || test_interval <= 0 || (myite+1)%test_interval == 0) {
    did_test = true; 
    if (do_save_mem && (tst != NULL || tst2 != NULL)) (const_cast<AzpData_ *>(trn))->release_batch(); 
    if (tst != NULL) perf = test(tst, &test_loss, &s_pf, &clk); 
    if (tst2 != NULL) perf2 = test(tst2, &test_loss2, NULL, &clk);
    if (do_exact_trnloss) { /* show exact training loss */
      regloss = lays_regloss(&iniloss);    /* regularization term */
      exact_loss = get_loss_noreg(trn); /* reg.term is not included */
    }  
  }

  acc_to_err(s_pf, perf, perf2); /* convert it to error rate if it's accuracy */
  
  /*---  show  ...  ---*/
  int width = (do_less_verbose) ? 6 : 15; 
  /*---  loss and performance  ---*/
  AzBytArr s("epoch,"); s << ite+1 << "," << loss; /* loss estimate: reg.term is not included */
  if (exact_loss >= 0) {
    s << ", trn-loss,"; s.cn(exact_loss,width); /* reg.term is not included */
    if (regloss >= 0) {
      s << ","; s.cn(regloss,width); 
    }
  }
  if (did_test) {
    bool is_there_perf = (s_pf.length() > 0 && !s_pf.equals(AzpEvalNoSupport)); 
    if (!do_less_verbose || !is_there_perf) {
      if (tst != NULL)  s << ", test-loss," << test_loss; 
      if (tst2 != NULL) s << "," << test_loss2; 
    }
    if (is_there_perf) {
      if (tst != NULL)  s << ", perf:" << s_pf.c_str() << "," << perf; 
      if (tst2 != NULL) s << "," << perf2; 
    }
  }

  AzTimeLog::print(s, out); 
  if (perf >= 0 || exact_loss > 0) AzPrint::writeln(eval_out, s); 
  
  /*---  save  ---*/  
  if (save_interval>0 && (myite+1)%save_interval==0 && (myite+1)>save_after || 
      ite == ite_num-1) save(ite+1); 
}

/*------------------------------------------------------------*/ 
void AzpReNet::acc_to_err(AzBytArr &s_pf, double &perf, double &perf2) const {
  if (do_show_acc || !s_pf.equals("acc")) return; 
  s_pf.reset("err"); 
  if (perf  >= 0 && perf == perf)   perf =  1 - perf; 
  if (perf2 >= 0 && perf2 == perf2) perf2 = 1 - perf2; 
}  

/*------------------------------------------------------------*/ 
void AzpReNet::show_layer_stat(const char *str) const {
  if (do_less_verbose) return; 
  AzBytArr s(str); 
  lays_show_stat(s); 
  AzPrint::writeln(out, s); 
  AzPrint::writeln(out, ""); 
}

/*------------------------------------------------------------*/ 
void AzpReNet::up_down(const AzpData_ *trn, 
                       const AzIntArr &ia_dxs, 
                       double *out_loss) {
  const char *eyec = "AzpReNet::up_down"; 
  AzX::no_support(!trn->is_sparse_y(), eyec, "No support for dense Y"); 

  _tTr();   
  AzPmatSpa m_spa_y; 
  const AzPmatSpa *mptr_spa_y = for_sparse_y_init(trn, ia_dxs, &m_spa_y);
  
  /*---  upward (fprop)  ---*/  
  AzDataArr<AzpDataVar_X> data; 
  trn->gen_data(ia_dxs.point(), ia_dxs.size(), data);   _tTs(t_DataY); 
  bool is_test = false;   
  AzPmatVar mv_out; 
  up(is_test, data, mv_out);

  /*---  loss  ---*/
  AzPmatSpa ms_y; 
  trn->gen_targets(ia_dxs.point(), ia_dxs.size(), &ms_y); 
  AzX::throw_if((ms_y.colNum() != mv_out.colNum()), AzInputError, eyec, 
                "output data size and target size do not match"); 
  AzPmatVar mv_ld; mv_ld.reform(1, mv_out.d_index()); 
  nco.loss->get_loss_deriv(mv_out.data(), trn, ia_dxs.point(), ia_dxs.size(), mv_ld.data_u(), out_loss, mptr_spa_y); 
  mv_ld.check_colNum("mv_ld in AzpReNet::up_down"); 

  /*---  downward (bprop)  ---*/
  down(mv_ld); 
  flush(); 
  
  for_sparse_y_term();  _tTs(t_DataY); 
}

/*------------------------------------------------------------*/ 
void AzpReNet::up0(bool is_test, const AzDataArr<AzpDataVar_X> &data, AzPmatVar &mv_out) {
  _tLr(); 
  if (side_num > 0) {
    AzPmatVar mv2; 
    side_lay->side_upward(is_test, data, mv2);  
    if (do_zeroout_side) mv2.data_u()->zeroOut(); /* for debugging only */
    (*lays)(0)->upward(is_test, data, mv_out, &mv2); 
  }
  else {
    (*lays)(0)->upward(is_test, data, mv_out);   
  }
  _tLs(l_Upward, 0); 
}

/*------------------------------------------------------------*/ 
void AzpReNet::up_nomc(bool is_test, const AzDataArr<AzpDataVar_X> &data, AzPmatVar &mv_out) {
  up0(is_test, data, mv_out); 
  for (int lx = 1; lx < lays->size(); ++lx) {   
    _tLr(); 
    AzPmatVar mv; 
    (*lays)(lx)->upward(is_test, mv_out, mv); 
    mv_out.set(&mv); 
    _tLs(l_Upward, lx); 
  }
  _tTs(t_Upward); 
}

/*------------------------------------------------------------*/ 
void AzpReNet::up_mc(bool is_test, const AzDataArr<AzpDataVar_X> &data, AzPmatVar &mv_out) {
  up0(is_test, data, mv_out); 

  const AzIntArr &ia_order = mc.order();
  AzX::throw_if(ia_order[0] != 0, "AzpReNet::up_mc", "layer-0 must be the first.");    
  int lsz = lays->size(); 
  AzDataArr<AzPmatVar> amv(lsz+conns.size()); 
  amv(0)->set(&mv_out); /* input to the next layer */
  for (int ix = 1; ix < ia_order.size(); ++ix) {
    int lx = ia_order[ix];      
    if (lx < lsz) {
      int below = mc.below(lx); 
      if (below < 0) (*lays)(lx)->upward(is_test, data, *amv(lx)); 
      else           (*lays)(lx)->upward(is_test, *amv[below], *amv(lx)); 
    }
    else          conns(lx-lsz)->upward(is_test, amv, *amv(lx)); 
    if (ix == ia_order.size()-1) mv_out.set(amv[lx]); /* top */       
      
    mc.release_output(lx, amv); /* to save memory: 11/26/2016 */
  }     
}

/*------------------------------------------------------------*/ 
void AzpReNet::down_nomc(const AzPmatVar &mv_ld, bool dont_update, bool dont_release_sv) {
  for (int lx = lays->size() - 1; lx >= 0; --lx) {
    _tLr(); 
    (*lays)(lx)->downward(mv_ld, dont_update, dont_release_sv);
    _tLs(l_Downward, lx); 
  }  
  if (do_update_side) side_lay->downward(mv_ld, dont_update, dont_release_sv); 
  _tTs(t_Downward); 
}  

/*------------------------------------------------------------*/ 
void AzpReNet::down_mc(const AzPmatVar &mv_ld, bool dont_update, bool dont_release_sv) {
  const AzIntArr &ia_order = mc.order();    
  int lsz = lays->size(); 
  for (int ix = ia_order.size()-1; ix >= 0; --ix) {
    int lx = ia_order[ix];   
    if (lx < lsz) (*lays)(lx)->downward(mv_ld, dont_update, dont_release_sv);
    else          conns(lx-lsz)->downward(ups);
  }
  if (do_update_side) side_lay->downward(mv_ld, dont_update, dont_release_sv);     
} 

/*------------------------------------------------------------*/
/* eval_all2 of AzpCNet3 */
double AzpReNet::test(const AzpData_ *data, double *out_loss, AzBytArr *s_pf, AzClock *clk) {
  int out_num = 0; 
  if (out_loss != NULL) *out_loss = 0; 
  if (clk != NULL) clk->update(); 
  double perf = 0; 
  (const_cast<AzpData_ *>(data))->first_batch();    
  for (int bx = 0; ; ++bx) {
    int data_num = data->dataNum(); 
    for (int dx = 0; dx < data_num; dx += tst_minib) {
      int d_num = MIN(tst_minib, data_num - dx); 
      AzPmatVar mv_p; 
      apply(data, dx, d_num, mv_p); 
      nco.loss->test_eval2(data, dx, d_num, mv_p.data(), perf, out_num, out_loss, s_pf); 
    }
    if (bx+1 >= data->batchNum()) break; 
    (const_cast<AzpData_ *>(data))->next_batch();    
  }
  if (clk != NULL) clk->tick(out, "test: ");  
  if (!do_less_verbose) AzPrint::writeln(out, "#evaled=", out_num); 
  if (out_num != 0) {
    if (out_loss != NULL) *out_loss /= (double)out_num; 
    perf /= (double)out_num; 
  }
  if (do_save_mem) (const_cast<AzpData_ *>(data))->release_batch();    
  return perf; 
}

/*------------------------------------------------------------*/ 
void AzpReNet::apply(const AzpData_ *tst, 
                      const int dx_begin, int d_num, 
                      AzPmatVar &mv_top_out) {                      
  AzDataArr<AzpDataVar_X> data; 
  tst->gen_data(dx_begin, d_num, data); 
  bool is_test = true;  
  up(is_test, data, mv_top_out); 
}

/*------------------------------------------------------------*/ 
double AzpReNet::get_loss_noreg(const AzpData_ *data) {
  int data_num = data->dataNum(); 
  double loss = 0; 
  for (int dx = 0; dx < data_num; dx += tst_minib) {
    int d_num = MIN(tst_minib, data_num - dx); 
  
    AzPmatVar mv_out; 
    apply(data, dx, d_num, mv_out);  
    loss += nco.loss->get_loss(data, dx, d_num, mv_out.data()); 
  }
  int out_num = data->colNum(); 
  if (out_num > 0) {
    loss /= (double)out_num; 
  }
  return loss; 
}

/*------------------------------------------------------------*/ 
void AzpReNet::show_progress(int num, int inc, int data_size, int eval_size, 
                            double loss_sum) const {    
  if (do_less_verbose && (inc <= 0 || inc > data_size)) return; 
  int width = (do_less_verbose) ? 6 : 20; 
           
  if (inc > 0 && num%inc == 0 || num == data_size) {    
    double loss_avg = loss_sum / (double)num; 
    if (eval_size != data_size) loss_avg *= (double)data_size/(double)eval_size; 
    AzBytArr s("... "); s << num << ": "; s.cn(loss_avg,width); /* loss estimate */
    AzTimeLog::print(s, out); 
    AzPrint(out, " "); 
  }
}

/*------------------------------------------------------------*/ 
/*------------------------------------------------------------*/ 
#define kw_save_after "save_after="
#define kw_save_interval "save_interval="
#define kw_save_fn "save_fn="
#define kw_save_layfn "save_layer_fn="
#define kw_test_interval "test_interval="
#define kw_do_show_acc "ShowAccuracy"
#define kw_hid_num "layers="

#define kw_ite_num_old "num_iterations="
#define kw_ite_num     "num_epochs="
#define kw_minib       "mini_batch_size="
#define kw_tst_minib   "test_mini_batch_size="
#define kw_rseed       "random_seed="
#define kw_init_ite    "initial_iteration="
  
#define kw_dx_inc "inc="
#define kw_do_exact_trnloss "ExactTrainingLoss"

#define kw_do_verbose "Verbose"
#define kw_do_less_verbose "LessVerbose"
#define kw_laytype "layer_type="
#define kw_layfn "layer_fn="
#define kw_do_test_first "TestFirst"
#define kw_max_loss "max_loss="
#define kw_side_num "num_sides="
#define kw_do_update_side "UpdateSide"
#define kw_zerotarget_ratio "zero_Y_ratio="
#define kw_do_zeroout_side "ZeroOutSide"
#define kw_do_save_mem "SaveDataMem"
#define kw_do_timer "Timer"
#define kw_max_data_num "max_num_data="
#define kw_do_topthru "TopThru"

/*------------------------------------------------------------*/ 
void AzpReNet::resetParam(AzParam &azp, bool is_warmstart, bool is_alone) {
  const char *eyec = "AzpReNet::resetParam"; 

  if (!is_warmstart) {
    if (!is_alone) azp.swOn(&do_topthru, kw_do_topthru); 
    azp.vInt(kw_hid_num, &hid_num); 
//    AzXi::throw_if_nonpositive(hid_num, eyec, kw_hid_num);         
    resetParam_lays(kw_laytype, azp, hid_num, "", "", as_laytype); 
    resetParam_lays(kw_layfn, azp, hid_num, "", "", as_layfn); 
    AzBytArr s_kw_top_layfn(_ReLay_top_, kw_layfn); 
    azp.vStr(s_kw_top_layfn.c_str(), &s_top_layfn); 
    
    azp.vInt(kw_side_num, &side_num); 
    resetParam_lays(kw_laytype, azp, side_num, "0side", "_", as_side_laytype); /* 0side0_ , 0side1_ ... */
    resetParam_lays(kw_layfn, azp, side_num, "0side", "_", as_side_layfn);    
  }
  if (has_side()) azp.swOn(&do_update_side, kw_do_update_side);   
  azp.swOn(&do_zeroout_side, kw_do_zeroout_side);   
  azp.swOn(&do_less_verbose, kw_do_less_verbose); /* for compatibility */
  azp.swOff(&do_less_verbose, kw_do_verbose, false);   
  azp.swOn(&do_timer, kw_do_timer);   
  sssch.resetParam(azp); /* step-size scheduler */
  mc.resetParam(azp, hid_num, is_warmstart);  /* for multi-connection */
  if (!is_alone) return; 
  
  /*---  parameters used only by stand-alone instances  ---*/
  azp.vInt(kw_max_data_num, &max_data_num);
  azp.swOn(&do_save_mem, kw_do_save_mem); 
   
  AzBytArr s_save_layfn_dflt; 
  azp.vStr(kw_save_layfn, &s_save_layfn_dflt); 
  resetParam_lays(kw_save_layfn, azp, hid_num, "", "", as_save_layfn, &s_save_layfn_dflt); 

  azp.vStr(kw_save_fn, &s_save_fn); 
  azp.vInt(kw_save_interval, &save_interval);  
  azp.vInt(kw_save_after, &save_after); 
  azp.vInt(kw_test_interval, &test_interval); 
  azp.swOn(&do_show_acc, kw_do_show_acc); 
 
  azp.vInt(kw_ite_num, &ite_num, kw_ite_num_old);  
  azp.vInt(kw_minib, &minib); 
  AzXi::throw_if_nonpositive(minib, eyec, kw_minib); 

  azp.vInt(kw_rseed, &rseed); 
  azp.vInt(kw_dx_inc, &dx_inc); 
  azp.swOn(&do_exact_trnloss, kw_do_exact_trnloss);  
  AzXi::throw_if_both(do_save_mem && do_exact_trnloss, eyec, kw_do_save_mem, kw_do_exact_trnloss); 

  azp.swOn(&do_test_first, kw_do_test_first); 
  azp.vFloat(kw_max_loss, &max_loss); 
  
  azp.vInt(kw_zerotarget_ratio, &zerotarget_ratio);  
  do_partial_y = (zerotarget_ratio >= 0); 
  
  _resetParam(azp);  /* read parameters used both train_test and test */

  /*---  loss  ---*/
  nco.loss->resetParam(out, azp, is_warmstart); 
  nco.loss->check_losstype(out, class_num); 

  /*---  data sequence generation  ---*/  
  dataseq.resetParam(azp); 
}

/*------------------------------------------------------------*/ 
/* static */
void AzpReNet::resetParam_lays(const char *kw, AzParam &azp, int num, const char *pfx, const char *dlm, 
                               AzDataArr<AzBytArr> &arr, const AzBytArr *s_dflt) {
  arr.reset(num); 
  for (int lx = 0; lx < num; ++lx) {
    if (s_dflt != NULL) arr(lx)->reset(s_dflt); 
    AzBytArr s_pfx(pfx); s_pfx << lx << dlm; 
    azp.reset_prefix(s_pfx.c_str()); 
    azp.vStr(kw, arr(lx));
    azp.reset_prefix(); 
  }
}                                      
void AzpReNet::printParam_lays(const char *kw, AzPrint &o, const char *pfx, const char *dlm, 
                               const AzDataArr<AzBytArr> &arr) {
  for (int lx = 0; lx < arr.size(); ++lx) {
    AzBytArr s_pfx(pfx); s_pfx << lx << dlm; 
    o.reset_prefix(s_pfx.c_str()); 
    o.printV_if_not_empty(kw, *arr[lx]); 
    o.reset_prefix(); 
  }  
}                                 
void AzpReNet::printParam_lays(const char *kw, AzPrint &o, const char *pfx, const char *dlm, 
                               const AzIntArr &ia) {
  for (int lx = 0; lx < ia.size(); ++lx) {
    AzBytArr s_pfx(pfx); s_pfx << lx << dlm; 
    o.reset_prefix(s_pfx.c_str()); 
    o.printV(kw, ia[lx]); 
    o.reset_prefix(); 
  }  
} 

/*------------------------------------------------------------*/ 
void AzpReNet::printParam(const AzOut &out, bool is_alone) const {
  if (out.isNull()) return; 
  AzPrint o(out); 
  o.printSw(kw_do_topthru, do_topthru); 
  o.printV(kw_side_num, side_num); 
  o.printSw(kw_do_update_side, do_update_side); 
  o.printSw(kw_do_verbose, !do_less_verbose);  
  o.printSw(kw_do_zeroout_side, do_zeroout_side);   
  o.printSw(kw_do_timer, do_timer);   
  sssch.printParam(out);  /* step-size scheduler */  
  mc.printParam(out);  /* for multi-connection */  
  if (!is_alone) return;   
  
  /*---  parameters used only by stand-alone instances  ---*/
  o.printV(kw_max_data_num, max_data_num);   
  o.printSw(kw_do_save_mem, do_save_mem); 
  o.printV(kw_test_interval, test_interval); 
  o.printSw(kw_do_show_acc, do_show_acc); 
  o.printV(kw_save_interval, save_interval); 
  o.printV(kw_save_after, save_after); 
  o.printV_if_not_empty(kw_save_fn, s_save_fn); 
  printParam_lays(kw_save_layfn, o, "", "", as_save_layfn);   
  o.printV(kw_ite_num, ite_num); 
  o.printV(kw_rseed, rseed); 
  o.printV(kw_dx_inc, dx_inc); 

  o.printSw(kw_do_exact_trnloss, do_exact_trnloss); 
  o.printV(kw_minib, minib); 
  o.printSw(kw_do_test_first, do_test_first); 
  o.printV(kw_max_loss, max_loss); 
  o.printV(kw_zerotarget_ratio, zerotarget_ratio); 
  
  /*---  loss  ---*/
  nco.loss->printParam(out); 

  /*---  data sequence generation  ---*/
  dataseq.printParam(out); 
  
  _printParam(o); /* show parameters used both for train_test and test */
  
  o.ppEnd(); 
}
 
/*------------------------------------------------------------*/ 
/* parameters used for both train_test and test */
void AzpReNet::_resetParam(AzParam &azp) {
  const char *eyec = "AzpReNet::_resetParam"; 
  azp.vInt(kw_tst_minib, &tst_minib); 
  AzXi::throw_if_nonpositive(tst_minib, eyec, kw_tst_minib); 
}

/*------------------------------------------------------------*/ 
/* parameters used for both train_test and test */
void AzpReNet::_printParam(AzPrint &o) const {
  o.printV(kw_tst_minib, tst_minib);  
}

/*------------------------------------------------------------*/ 
void AzpReNet::check_data_signature(const AzpData_tmpl_ *data, const char *msg, bool do_loose) const {
  AzBytArr s_sign; 
  data->signature(s_sign);  
  if (do_loose) {
    if (!data->isSignatureCompatible(&s_data_sign, &s_sign)) {
      AzPrint::writeln(out, "model: ", s_data_sign); 
      AzPrint::writeln(out, "data: ", s_sign); 
      AzX::throw_if(true, AzInputError, "AzpReNet::check_data_signature", "data signature loose check failed", msg);       
    }
  }
  else {
    if (s_data_sign.compare(&s_sign) != 0) {
      AzPrint::writeln(out, "model: ", s_data_sign); 
      AzPrint::writeln(out, "data: ", s_sign);
      AzX::throw_if(true, AzInputError, "AzpReNet::check_data_signature", "data signature mismatch", msg);   
    }
  }
}

/*------------------------------------------------------------*/ 
/*------------------------------------------------------------*/ 
int AzpReNet::init_test(AzParam &azp, const AzpData_tmpl_ *tst) {
  const char *eyec = "AzpReNet::init_test";   
  AzX::throw_if((lays->size() <= 0), eyec, "No neural net to test"); 
  AzX::throw_if_null(tst, eyec, "test data"); 
  
  /*---  check the data signature  ---*/
  bool do_loose = true; 
  check_data_signature(tst, eyec, do_loose); 

  /*---  read parameters  ---*/
  resetParam_test(azp); 
  printParam_test(out); 

  /*---  initialize layers  ---*/
  bool for_testonly = true; 
  int outdim = warmstart(tst, azp, for_testonly); 
  azp.check(out); 
   
  bool is_warmstart = true; 
  check_word_mapping(is_warmstart, NULL, tst, NULL); 
  return outdim; 
}  
 
/*------------------------------------------------------------*/ 
void AzpReNet::predict(AzParam &azp, const AzpData_ *tst, AzDmatc &mc_pred, bool do_tok) {
  const char *eyec = "AzpReNet::predict"; 
  
  init_test(azp, tst); 

  AzX::no_support(tst->batchNum() > 1, eyec, "Batches"); 
  
  int data_num = tst->dataNum();  
  int cnum = tst->colNum(); 
  int inc = data_num/50, milestone = inc; 
  int col = 0; 
  for (int dx = 0; dx < data_num; dx += tst_minib) {
    AzTools::check_milestone(milestone, dx, inc); 
    int d_num = MIN(tst_minib, data_num - dx); 
    AzPmatVar mv_out; 
    apply(tst, dx, d_num, mv_out);   
    if (dx == 0) {
      if (do_tok) mc_pred.reform(class_num, cnum); 
      else        mc_pred.reform(class_num, data_num); 
    }
    AzX::throw_if(!do_tok && mv_out.colNum() != d_num, eyec, "Conflict in #output.  Expected one output per data point."); 
    mv_out.data()->copy_to(&mc_pred, col); 
    col += mv_out.colNum(); 
  }
  AzX::throw_if((col != mc_pred.colNum()), eyec, "Unexpected #output");   
  AzTools::finish_milestone(milestone);  
}

/*------------------------------------------------------------*/ 
void AzpReNet::resetParam_test(AzParam &azp) {
  _resetParam(azp); 
}  

/*------------------------------------------------------------*/ 
void AzpReNet::printParam_test(const AzOut &out) const {
  AzPrint o(out); 
  _printParam(o); 
  mc.printParam(out);  /* for multi-connection */ 
  o.ppEnd(); 
}

/*-------------------------------------------------------------------*/ 
/*-------------------------------------------------------------------*/ 
/*  Negative sampling/weighting for unsupervised embeddings training */
/*-------------------------------------------------------------------*/ 
const AzPmatSpa *AzpReNet::for_sparse_y_init(const AzpData_ *trn, const AzIntArr &ia_dxs, AzPmatSpa *m_y) {
  const char *eyec = "AzpReNet::for_sparse_y_init"; 
  if (!do_partial_y) return NULL; 
  AzX::no_support(!trn->is_sparse_y(), eyec, "zero_target_ratio/partial_y with non-sparse targets");  

  AzSmat ms_y; trn->gen_targets(ia_dxs.point(), ia_dxs.size(), &ms_y);  
  AzIntArr ia_p2w, ia_w2p; 
  pick_classes(&ms_y, zerotarget_ratio, ia_p2w, ia_w2p, m_y); 
  if (ia_p2w.size() > 0) {
    (*lays)(top_ind())->linmod_u()->reset_partial(&ia_p2w, &ia_w2p); 
    return m_y;  
  } 
  return NULL; 
}
/*------------------------------------------------------------*/ 
void AzpReNet::for_sparse_y_term() {
  if (do_partial_y) (*lays)(top_ind())->linmod_u()->reset_partial();   
}

/*------------------------------------------------------------*/ 
/* Assume that ms_y is binary and 0 represents "negative"     */
/* returning empty ia_p2w means use ms_y asis                 */
/*------------------------------------------------------------*/ 
void AzpReNet::pick_classes(AzSmat *ms_y, /* inout */
               int nega_count, 
               /*---  output  ---*/
               AzIntArr &ia_p2w, AzIntArr &ia_w2p, AzPmatSpa *m_y) const {
  ia_p2w.reset(); ia_w2p.reset(); 
  if (ms_y->colNum() <= 1 || ms_y->rowNum() <= 0) return;  /* keep target asis */
  AzIntArr ia_nzrows;  
  AzDataArr<AzIFarr> aifa; 
  if (nega_count <= 0) ms_y->nonZeroRowNo(&ia_nzrows); 
  else {
    double nega_val = -1e-10; 
    int cnum = ms_y->colNum(); 
    aifa.reset(cnum); 

    for (int col = 0; col < cnum; ++col) {   
      ms_y->col(col)->nonZero(aifa(col));  /* appended */
      AzIntArr ia_nzr; ms_y->col(col)->nonZeroRowNo(&ia_nzr); 
      ia_nzrows.concat(&ia_nzr); 
      for (int ix = 0; ix < ia_nzr.size(); ++ix) {
        int row = ia_nzr[ix]; 
        for (int jx = 0; jx < nega_count; ) {
          if (jx+1 >= cnum) break; 
          int cx = AzTools::big_rand() % cnum; 
          if (cx != col) {
            aifa(cx)->put(row, nega_val); 
            ++jx; 
          }
        }
      }
    }
    for (int col = 0; col < cnum; ++col) aifa(col)->squeeze_Max();          
    ia_nzrows.unique();  
  }
  
  /*---  only keep non-zero rows  ---*/  
  ia_p2w.reset(&ia_nzrows); 
  if (ia_p2w.size() <= 0) ia_p2w.put(0); /* if all zero, keep one row. */
  ia_w2p.reset(ms_y->rowNum(), -1); 
  for (int px = 0; px < ia_p2w.size(); ++px) ia_w2p(ia_p2w[px], px); 

  bool do_gen_rowindex = false;  
  if (nega_count > 0) m_y->set(aifa, ia_p2w.size(), &ia_w2p, do_gen_rowindex); 
  else {
    ms_y->change_rowno(ia_p2w.size(), &ia_w2p); 
    m_y->set(ms_y, do_gen_rowindex); 
  }
}

/*------------------------------------------------------------*/  
/*------------------------------------------------------------*/ 
void AzpReNet::check_word_mapping(bool is_warmstart, const AzpData_tmpl_ *trn, 
               const AzpData_tmpl_ *tst, const AzpData_tmpl_ *tst2) {
  const char *eyec = "AzpReNet::check_word_mapping"; 
  if (!is_warmstart) {
    AzX::throw_if_null(trn, eyec, "No training data?!"); 
    do_ds_dic = true; 
    ds_dic.reset(trn->datasetNum()); 
    for (int ix = 0; ix < trn->datasetNum(); ++ix) {
      ds_dic(ix)->reset(trn->get_dic(ix)); 
    }
  }  
  if (!do_ds_dic) return; 

  AzTimeLog::print("Checking word-mapping ... ", out); 
  AzX::throw_if((trn == NULL && tst == NULL), eyec, "No data?!"); 
  int ds_num = (trn != NULL) ? trn->datasetNum() : tst->datasetNum(); 
  AzX::throw_if((ds_dic.size() != ds_num), eyec, "# of word-mapping sets does not match the number saved with the model."); 
  AzX::throw_if((tst != NULL && tst->datasetNum() != ds_num), eyec, "# of word-mapping sets of test data is wrong."); 
  AzX::throw_if((tst2 != NULL && tst2->datasetNum() != ds_num), eyec, "# of word-mapping sets of 2nd test data is wrong.");   
  for (int ix = 0; ix < ds_num; ++ix) {
    AzBytArr s_err("Word-mapping check failed."); 
    if (ds_num > 1) s_err << "  data#=" << ix; 
    if (trn != NULL)  AzX::throw_if(!ds_dic[ix]->is_same(trn->get_dic(ix)), eyec, "Training data: ", s_err.c_str()); 
    if (tst != NULL)  AzX::throw_if(!ds_dic[ix]->is_same(tst->get_dic(ix)), eyec, "Test data: ", s_err.c_str()); 
    if (tst2 != NULL) AzX::throw_if(!ds_dic[ix]->is_same(tst2->get_dic(ix)), eyec, "2nd test data: ", s_err.c_str()); 
  } 

  AzTimeLog::print("Checking word-mapping of layers ... ", out); 
  const AzpData_tmpl_ *data = (trn != NULL) ? trn : tst; 
  if (lays->size() > 0) (*lays)(0)->check_word_mapping(data); 
  if (has_side()) side_lay->check_word_mapping(data);   
}

/*------------------------------------------------------------*/ 
void AzpReNet::write_word_mapping_in_lay(const AzBytArr &s_lay_type, const char *lay_fn, const char *dic_fn) const {
  AzpReLayers tmplays; 
  tmplays.reset(1); 
  tmplays.reset(0, &s_lay_type, cs); 
  tmplays(0)->read(lay_fn); 
  tmplays(0)->write_word_mapping(dic_fn); 
}
 
/*------------------------------------------------------------*/ 
void AzpReNet::write_word_mapping(const char *fn, int dsno) const {
  if (!do_ds_dic) {
    AzPrint::writeln(out, "No word-mapping info is saved with this model."); 
    return; 
  }
  AzX::throw_if((dsno < 0 || dsno >= ds_dic.size()), "AzpReNet::write_word_mapping", "data# is out of range"); 
  ds_dic[dsno]->writeText(fn); 
}

/*------------------------------------------------------------*/ 
/*------------------------------------------------------------*/
void AzpReNet::timer_init() { 
  timer = NULL; 
  if (do_timer) timer = &my_timer; 
  if (timer != NULL) timer->reset(hid_num + conns.size());   
}
void AzpReNet::timer_show() const { 
  if (timer != NULL) { timer->show(out); timer->set_zero(); }
}

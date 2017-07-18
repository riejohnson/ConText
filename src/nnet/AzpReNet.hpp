/* * * * *
 *  AzpReNet.hpp
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
 
#ifndef _AZP_RE_NET_HPP_
#define _AZP_RE_NET_HPP_

#include "AzpReLayer.hpp"
#include "AzParam.hpp"
#include "AzHelp.hpp"
#include "AzpData_.hpp"
#include "AzStepszSch.hpp"
#include "AzpDataSeq.hpp"
#include "AzpCompoSet_.hpp"
#include "AzpTimer_CNN.hpp"
#include "AzMultiConn.hpp"
using namespace AzpTimer_CNN_type; 


/*------------------------------------------------------------*/ 
class AzpReConn : public virtual /* implements */ AzpReUpperLayer_ {
protected:  
  int layer_no;
  AzIntArr ia_below, ia_above; 
  AzPmatVar mv_ld; 
  bool do_add; 
  AzIntArr ia_rbeg, ia_rend; /* used only if concat */
public:
  AzpReConn() : layer_no(-1), do_add(false) {}
  virtual ~AzpReConn() {}   
  virtual int setup(const AzOut &out, int lno, const AzIntArr &ia_b, const AzIntArr &ia_a, 
                    const AzIntArr &ia_rnum, bool _do_add) {
    const char *eyec = "AzpReConn::setup"; 
    do_add = _do_add;  
    layer_no = lno; ia_below.reset(&ia_b); ia_above.reset(&ia_a); 
    AzX::throw_if(ia_below.size() <= 0, eyec, "No input"); 
    AzX::throw_if(ia_above.size() <= 0, eyec, "No output"); 
    int offs = 0; 
    if (do_add) {
      for (int ix = 0; ix < ia_below.size(); ++ix) {
        int rnum = ia_rnum[ia_below[ix]]; 
        if (ix == 0) offs = rnum; 
        else {
          AzX::throw_if(rnum != offs, eyec, "Additive connection requires #row of inputs to be all the same.");
        }
      }
    }
    else { /* concat */
      ia_rbeg.reset(); ia_rend.reset(); 
      for (int ix = 0; ix < ia_below.size(); ++ix) {
        ia_rbeg.put(offs);
        offs += ia_rnum[ia_below[ix]]; 
        ia_rend.put(offs); 
      }      
    }
    AzBytArr s("layer#"); s << lno << ": "; 
    AzMultiConn::show_below_above(ia_below, ia_above, s); 
    AzPrint::writeln(out, s); 
    return offs; 
  }  
  virtual void upward(bool is_test, const AzDataArr<AzPmatVar> &amv, AzPmatVar &mv_out) {
    const char *eyec = "AzpReConn::upward"; 
    for (int ix = 0; ix < ia_below.size(); ++ix) {
      const AzPmatVar *m = amv[ia_below[ix]]; 
      if (ix == 0)      mv_out.set(m); 
      else if (do_add)  mv_out.add(m); 
      else { /* concat */
        AzX::throw_if(!mv_out.can_rbind(m), eyec, "Cannot rbind."); 
        mv_out.rbind(m);
        AzX::throw_if(mv_out.rowNum() != ia_rend[ix], eyec, "Something is wrong with #row."); 
      }
    }
  }
  virtual void upward2(const AzDataArr<AzPmatVar> &amv, AzPmatVar &mv_out) { upward(false, amv, mv_out); }
  virtual void downward(const AzBaseArr<const AzpReUpperLayer_ *> &ups) {
    for (int ix = 0; ix < ia_above.size(); ++ix) {
      const AzpReUpperLayer_ * upper = ups[ia_above[ix]]; 
      AzX::throw_if_null(upper, "AzpReConn::downward", "null upper"); 
      if (ix == 0) upper->get_ld(layer_no, mv_ld); 
      else {
        AzPmatVar mv; 
        upper->get_ld(layer_no, mv);         
        mv_ld.data_u()->add(mv.data()); 
      }
    }    
  }
  virtual void get_ld(int id, AzPmatVar &out_mv_ld, bool do_x2=false) const {
    if (do_add) out_mv_ld.set(&mv_ld);  
    else { /* concat */
      int index = ia_below.find(id); 
      AzX::throw_if(index < 0, "AzpReConn::get_ld", "Request from unexpected layer#"); 
      int rbeg = ia_rbeg[index], rend = ia_rend[index]; 
      int rnum = rend - rbeg; 
      out_mv_ld.reform_noinit(rnum, mv_ld.d_index());  
      out_mv_ld.data_u()->set_rowwise(0, rnum, mv_ld.data(), rbeg); 
    }
  }
};

/*------------------------------------------------------------*/
class AzpReNet {
protected:
  const AzpCompoSet_ *cs; 
  AzpNetCompoPtrs nco;  

  AzpReLayers lays_dflt; 
  AzpReLayer_Side side_lay_dflt; 
  AzpReLayers *lays; /* must not be null anytime */
  AzpReLayer_Side *side_lay; /* must not be null anytime */

  /*---  for skip connections  ---*/
  AzMultiConn mc;  
  AzDataArr<AzpReConn> conns;
  AzBaseArr<const AzpReUpperLayer_ *> ups;   
  /*------------------------------*/  

  /*---  for unsupervised embeddings  ---*/  
  int side_num; 
  bool do_update_side; 
  bool has_side() const { return (side_num>0); }
  /*-------------------------------------*/
  
  AzBytArr s_data_sign; 
  int class_num; /* 1: regression */  
  
  AzOut out, eval_out; 

  AzStepszSch sssch;  /* step-size scheduler */
  AzpDataSeq dataseq;  /* data sequencer to make mini-batches */

  bool do_ds_dic; 
  AzDataArr<AzDicc> ds_dic; 

  /*---  parameters  ---*/
  int save_interval, save_after; 
  AzBytArr s_save_fn; 
  AzDataArr<AzBytArr> as_save_layfn; 
  int hid_num; 
  int test_interval, dx_inc; 
  int ite_num, minib, tst_minib, rseed, init_ite; 
  bool do_exact_trnloss, do_show_iniloss, do_show_acc; 
  bool do_less_verbose, do_test_first; 
  AzDataArr<AzBytArr> as_laytype, as_layfn; 
  AzBytArr s_top_layfn; 
  AzDataArr<AzBytArr> as_side_laytype, as_side_layfn; 

  bool do_topthru;   
  bool do_partial_y; 
  int zerotarget_ratio; 
  
  double max_loss; 
    
  bool do_zeroout_side; /* only for debugging */
  bool do_save_mem; /* load test data each time to save cpu memory */
  
  /*---  only for debugging  ---*/
  bool do_timer; 
  int max_data_num;

  /*---  for time measurement  ---*/
  AzpTimer_CNN my_timer, *timer; /* changed from global 3/3/2017 */

  bool do_read_old_ext; 
  
/*  static const int version=0;  */
/*  static const int version=1;   5/28/2017: for mc */
  static const int version=2;  /* 6/1/2017: for mark */  
/*  static const int reserved_len=256; */
/*  static const int reserved_len=255;  2/23/2017: for do_topthru */
  static const int reserved_len=250; /* 6/1/2017: for mark */
  #define AzpReNet_Mark "ReNet"
public:
  #define AzpReNet_VarInit lays(&lays_dflt), side_lay(&side_lay_dflt), do_timer(false), max_data_num(-1), \
             dx_inc(-1), save_interval(-1), max_loss(-1), side_num(0), do_update_side(false), \
             do_partial_y(false), zerotarget_ratio(-1), do_zeroout_side(false), do_show_acc(false), \
             hid_num(0), class_num(1), test_interval(-1), out(log_out), \
             ite_num(0), minib(100), tst_minib(100), rseed(1), init_ite(0), do_test_first(false), do_save_mem(false), \
             do_exact_trnloss(false), do_show_iniloss(false), do_less_verbose(true), do_ds_dic(false), \
             do_topthru(false), timer(NULL), save_after(-1), do_read_old_ext(false)
             
  AzpReNet(const AzpCompoSet_ *_cs) : AzpReNet_VarInit {
    reset(_cs);     
  }
  virtual AzpReNet *clone_nocopy() const { 
    return new AzpReNet(cs); 
  }
 
protected:
  AzpReNet() : AzpReNet_VarInit {} /* for array */
  void reset(const AzpCompoSet_ *_cs) {
    AzX::throw_if(_cs == NULL, "AzpReNet::reset(cs)", "No component set"); 
    cs = _cs; 
    cs->check_if_ready("AzpReNet::constructor"); 
    nco.reset(cs);  
  }
public:    
  virtual ~AzpReNet() {}
  virtual void reset() {
    if (lays != NULL) lays->reset(); 
    s_data_sign.reset(); 
    as_save_layfn.reset();
    as_laytype.reset(); as_layfn.reset(); 
    as_side_laytype.reset(); as_side_layfn.reset(); 
 
    mc.reset(); 
    conns.reset(); 
    ups.free();
  }
 
  inline int classNum() const { return class_num; }
  inline virtual void training(const AzOut *inp_eval_out, AzParam &azp,  
                      AzpData_ *trn, /* not const b/c of first|next_batch */
                      const AzpData_ *tst, const AzpData_ *tst2) {
    if (inp_eval_out != NULL) eval_out = *inp_eval_out; 
    training(azp, trn, tst, tst2); 
  }
  virtual void training(AzParam &azp, AzpData_ *trn, const AzpData_ *tst, const AzpData_ *tst2); 
  virtual void predict(AzParam &azp, const AzpData_ *tst, AzDmatc &mc_pred, bool do_tok); 
  
  /*---  read/write  ---*/
  virtual void write(const char *fn) const { AzFile::write(fn, this); }
  virtual void read(const char *fn)  { AzFile::read(fn, this); }                      
  virtual void write(AzFile *file) const {
    AzTools::write_header(file, version, reserved_len); 
    file->writeBytes(AzpReNet_Mark, 5); 
    file->writeBool(do_topthru); /* 2/23/2017 */
    file->writeBool(do_ds_dic); 
    if (do_ds_dic) ds_dic.write(file);
    s_data_sign.write(file); 
    file->writeInt(class_num);
    nco.write(file);    
    lays->write(file); 
    file->writeInt(side_num); 
    if (has_side()) side_lay->write(file); 
    
    mc.write(file);    
  }
  void read_old_ext(AzFile *file) { do_read_old_ext = true; read(file); do_read_old_ext = false; }
  virtual void read(AzFile *file) {
    cs->check_if_ready("AzpReNet::read"); 
    int my_version = AzTools::read_header(file, reserved_len); 
    AzByte mark[5]; file->readBytes(mark, 5); 
    AzX::throw_if(my_version >= 2 && memcmp(mark, AzpReNet_Mark, 5) != 0, AzInputError, "AzpReNet::read",  
                  "Invalid file: this is not the AzpReNet format."); 
    do_topthru = file->readBool(); 
    do_ds_dic = file->readBool(); 
    if (do_ds_dic) ds_dic.read(file);
    s_data_sign.read(file); 
    class_num = file->readInt(); 
    nco.read(file);    
    lays->read(cs, file);
    side_num = file->readInt(); 
    if (has_side()) side_lay->read(cs, file); 
    hid_num = lays->size() - 1; 
    if (do_read_old_ext && my_version == 0) { /* the old AzpReNet_ext format */
      AzTools::read_header(file, 64); 
      mc.read(file); 
    }
    else if (my_version >= 1) {
      mc.read(file);     
    }
  }                      
  
  virtual void write_word_mapping_in_lay(const AzBytArr &s_lay_type, const char *lay_fn, const char *dic_fn) const; 
  virtual void write_word_mapping(const char *fn, int dsno) const; 

  /*---  ---*/
  virtual int init(AzParam &azp, const AzpData_tmpl_ *trn, const AzpData_tmpl_ *tst=NULL, const AzpData_tmpl_ *tst2=NULL, 
                   bool is_alone=false); 
  void deactivate_out() { out.deactivate(); }
  void activate_out() { out.activate(); }  
  virtual int init_test(AzParam &azp, const AzpData_tmpl_ *tst); 
  virtual void up0(bool is_test, const AzDataArr<AzpDataVar_X> &data, AzPmatVar &mv_out); /* first layer only */  
  virtual void up(bool is_test, const AzDataArr<AzpDataVar_X> &data, AzPmatVar &mv_out) {
    if (mc.is_multi_conn()) up_mc(is_test, data, mv_out); 
    else                    up_nomc(is_test, data, mv_out); 
  }
  virtual void up(bool is_test, const AzPmatVar &mv_inp, AzPmatVar &mv_out) {
    AzDataArr<AzpDataVar_X> darr(1); AzpData_::set_data(mv_inp, *darr(0)); up(is_test, darr, mv_out);  
  }
  virtual void up(bool is_test, const AzPmatSpaVar &msv_inp, AzPmatVar &mv_out) {
    AzDataArr<AzpDataVar_X> darr(1); AzpData_::set_data(msv_inp, *darr(0)); up(is_test, darr, mv_out);  
  }  
  virtual void down(const AzPmatVar &mv_ld, bool dont_update=false, bool dont_release_sv=false) {
    if (mc.is_multi_conn()) down_mc(mv_ld, dont_update, dont_release_sv);  
    else                    down_nomc(mv_ld, dont_update, dont_release_sv); 
  }
  virtual void up2(const AzPmatVar &, AzPmatVar &, bool) { AzX::no_support(true,"AzpReNet::up2","up2"); }   
  virtual void flush() { lays_flushDelta(); _tTs(t_Flush); }
  virtual void release_ld() { lays_release_ld(); } /* optional.  to save memory.  use this with caution.  */
  virtual void multiply_to_stepsize(double coeff) { lays_multiply_to_stepsize(coeff, &out); }
  
  virtual void save(int iteno); 
  virtual void show_layer_stat(const char *str=NULL) const;   
  virtual void reset_stepsize() { reset_stepsize(&sssch); }
  virtual void change_stepsize_if_required(int ite) { change_stepsize_if_required(&sssch, ite); } 
  virtual void end_of_epoch() { lays_end_of_epoch(); }
  virtual const AzpReUpperLayer_ *lay0() const { /* for gan */
    AzX::throw_if(lays->size() <= 0, "AzpReNet::lay0", "No layers"); 
    return (*lays)[0]; 
  }
  
protected:  
  virtual int top_ind() const { return lays->size()-1; }
  virtual int coldstart(const AzpData_tmpl_ *trn, AzParam &azp) { 
    if (mc.is_multi_conn()) return setup_mc(trn, azp, false, false); 
    else                    return setup_nomc(trn, azp, false, false);
  }
  virtual int warmstart(const AzpData_tmpl_ *trn, AzParam &azp, bool for_testonly=false) { 
    if (mc.is_multi_conn()) return setup_mc(trn, azp, true, for_testonly); 
    else                    return setup_nomc(trn, azp, true, for_testonly); 
  }
  virtual void up_down(const AzpData_ *trn, const AzIntArr &ia_dxs, double *out_loss=NULL); 
  
  virtual int setup_mc(const AzpData_tmpl_ *trn, AzParam &azp, bool is_warmstart, bool for_testonly);   
  virtual void _for_bottom(int lno, const AzpData_tmpl_ *trn, AzParam &azp, AzpReLayer_Param &pp) const;   
  virtual int setup_nomc(const AzpData_tmpl_ *trn, AzParam &azp, bool is_warmstart, bool for_testonly);   
  virtual void up_mc(bool is_test, const AzDataArr<AzpDataVar_X> &data, AzPmatVar &mv_out); 
  virtual void down_mc(const AzPmatVar &mv_ld, bool dont_update, bool dont_release_sv); 
  virtual void up_nomc(bool is_test, const AzDataArr<AzpDataVar_X> &data, AzPmatVar &mv_out); 
  virtual void down_nomc(const AzPmatVar &mv_ld, bool dont_update, bool dont_release_sv); 

  virtual void apply(const AzpData_ *tst, const int dx_begin, int d_num, AzPmatVar &mv_top_out); 
  virtual void reset_stepsize(AzStepszSch *sssch);   
  virtual void change_stepsize_if_required(AzStepszSch *sssch, int ite);  
  
  /*-------------------------------------------------------*/
  virtual void resetParam(AzParam &azp, bool is_warmstart, bool is_alone);
  virtual void printParam(const AzOut &out, bool is_alone) const; 
  
  static void resetParam_lays(const char *kw, AzParam &azp, int num, const char *pfx, const char *dlm, 
                               AzDataArr<AzBytArr> &arr, const AzBytArr *s_dflt=NULL);                                 
  static void printParam_lays(const char *kw, AzPrint &o, const char *pfx, const char *dlm, const AzDataArr<AzBytArr> &arr); 
  static void printParam_lays(const char *kw, AzPrint &o, const char *pfx, const char *dlm, const AzIntArr &ia);  
      
  virtual void resetParam_test(AzParam &azp); 
  virtual void printParam_test(const AzOut &out) const; 
  
  virtual void _resetParam(AzParam &azp); 
  virtual void _printParam(AzPrint &o) const; 
  /*-------------------------------------------------------*/

  virtual void sup_training_loop(AzpData_ *trn, const AzpData_ *tst, const AzpData_ *tst2);                         
  virtual void test_save(const AzpData_ *trn, const AzpData_ *tst, const AzpData_ *tst2, 
                         int ite, double tr_loss); 
  virtual void acc_to_err(AzBytArr &s_pf, double &perf, double &perf2) const;
  virtual void check_data_signature(const AzpData_tmpl_ *data, const char *msg, bool do_loose=false) const; 
  virtual double get_loss_noreg(const AzpData_ *data); 
  virtual double test(const AzpData_ *data, double *out_loss, AzBytArr *s_pf=NULL, AzClock *clk=NULL);    
  
  /*---  to display information  ---*/
  virtual void show_progress(int num, int inc, int data_size, int eval_size, double loss_sum) const;  

  /*----------------------------------------------*/ 
  /*  Functions that go over all layers.          */
  /*  For convenience for attching side layers.   */
  /*----------------------------------------------*/   
  virtual void lays_flushDelta() {
    for (int lx = 0; lx < lays->size(); ++lx) { _tLr(); (*lays)(lx)->flushDelta(); _tLs(l_Flush, lx); }
    if (do_update_side) side_lay->flushDelta(); 
  }
  virtual void lays_release_ld() { /* to save memory.  call this after downward. */
    for (int lx=0;lx<lays->size();++lx) { (*lays)(lx)->release_ld(); }
  }
  virtual void lays_end_of_epoch() {
    for (int lx = 0; lx < lays->size(); ++lx) (*lays)(lx)->end_of_epoch(); 
    if (do_update_side) side_lay->end_of_epoch(); 
  }
  virtual void lays_multiply_to_stepsize(double coeff, const AzOut *out) {
    for (int lx = 0; lx < lays->size(); ++lx) (*lays)(lx)->multiply_to_stepsize(coeff, out);
    if (do_update_side) side_lay->multiply_to_stepsize(coeff, out); 
  }  
  virtual double lays_regloss(double *out_iniloss=NULL) const {
    double loss = 0, iniloss = 0; 
    for (int lx = 0; lx < lays->size(); ++lx) loss += (*lays)[lx]->regloss(&iniloss); 
    if (do_update_side) loss += side_lay->regloss(&iniloss); 
    if (out_iniloss != NULL) *out_iniloss = iniloss; 
    return loss;    
  }
  virtual void lays_show_stat(AzBytArr &s) const {
    for (int lx = 0; lx < lays->size(); ++lx) {
      s << "layer#" << lx << ":"; 
      (*lays)[lx]->show_stat(s); 
    }
    if (has_side()) { 
      s << "side:"; side_lay->show_stat(s); 
    }
  }

  /*---  for "do_partial"  ---*/
  virtual const AzPmatSpa *for_sparse_y_init(const AzpData_ *trn, const AzIntArr &ia_dxs, AzPmatSpa *m_y); 
  virtual void for_sparse_y_term(); 
  virtual void pick_classes(AzSmat *ms_y, int nega_count, AzIntArr &ia_p2w, AzIntArr &ia_w2p, AzPmatSpa *m_y) const; 

  /*---  ---*/
  void check_word_mapping(bool is_warmstart, const AzpData_tmpl_ *trn, const AzpData_tmpl_ *tst, const AzpData_tmpl_ *tst2); 

  /*---  for timing  ---*/ 
  void _tTr()                    const { if (timer != NULL) timer->reset_Total(); }
  void _tTs(t_type typ)          const { if (timer != NULL) timer->stamp_Total(typ); }
  void _tLr()                    const { if (timer != NULL) timer->reset_Layer(); }
  void _tLs(l_type typ, int lno) const { if (timer != NULL) timer->stamp_Layer(typ, lno); }
  void timer_init(); 
public: void timer_show() const;    
}; 
#endif

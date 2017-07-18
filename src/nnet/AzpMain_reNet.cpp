/* * * * *
 *  AzpMain_reNet.cpp
 *  Copyright (C) 2017 Rie Johnson
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

#include "AzpMain_reNet.hpp"

/*---  global (declared in AzPmat)  ---*/
extern AzPdevice dev; 
extern bool __doDebug; 
extern bool __doTry; 

/*-------------------------------------*/
extern AzByte param_dlm; 

/*------------------------------------------------------------*/ 
#define kw_do_debug "Debug"
#define kw_do_try "Try"
#define kw_not_doLog "DontLog"
#define kw_doLog "Log"
#define kw_doDump "Dump"
/*------------------------------------------------------------*/ 
void AzpMain_reNet_Param_::_resetParam(AzPrint &o, AzParam &azp) {
  const char *eyec = "AzpMain_reNet_Param::_resetParam"; 
  azp.swOn(o, __doDebug, kw_do_debug);   
  azp.swOn(o, __doTry, kw_do_try);     
  AzPmatSpa_flags::resetParam(azp); AzPmatSpa_flags::printParam(o); 
  dev.resetParam(azp); dev.printParam(o); 
} 
/*------------------------------------------------------------*/ 
void AzpMain_reNet_Param_::setupLogDmp(AzPrint &o, AzParam &azp) {
  azp.swOff(o, doLog, kw_not_doLog, kw_doLog); 
  azp.swOn(o, doDump, kw_doDump); 
  
  log_out.reset(NULL); 
  dmp_out.reset(NULL); 
  if (doLog) log_out.setStdout(); 
  if (doDump) dmp_out.setStderr(); 
}

/*------------------------------------------------------------*/ 
#define kw_eval_fn "evaluation_fn="
#define kw_fn_for_warmstart "fn_for_warmstart="
#define kw_do_no_test "NoTest"
/*------------------------------------------------------------*/ 
class AzpMain_reNet_renet_Param : public virtual AzpMain_reNet_Param_ {
public:
  AzpDataSetDflt dataset; 
  AzBytArr s_eval_fn, s_fn_for_warmstart; 
  bool do_no_test; 
  AzpMain_reNet_renet_Param(AzParam &azp, const AzOut &out, const AzBytArr &s_action) : do_no_test(false) {    
    reset(azp, out, s_action); 
  }
  AzpMain_reNet_renet_Param() : do_no_test(false) {} /* use this for displaying help */
  void resetParam(const AzOut &out, AzParam &p) {
    AzPrint o(out); 
    _resetParam(o, p); 
    p.vStr_prt_if_not_empty(o, kw_fn_for_warmstart, s_fn_for_warmstart); 
    p.swOn(o, do_no_test, kw_do_no_test, false); 
    bool do_train = true, is_there_y = true; 
    dataset.resetParam(out, p, do_train, !do_no_test, is_there_y);        
    p.vStr_prt_if_not_empty(o, kw_eval_fn, s_eval_fn); 
    setupLogDmp(o, p);   
  }  
}; 

/*------------------------------------------------*/
void AzpMain_reNet::renet(int argc, const char *argv[], const AzBytArr &s_action) {
  const char *eyec = "AzpMain_reNet::renet"; 
  AzParam azp(param_dlm, argc, argv); 
  AzpMain_reNet_renet_Param p(azp, log_out, s_action); 

  /*---  read data  ---*/
  p.dataset.reset_data(log_out); 
  
  AzBytArr s; s << "#train=" << p.dataset.trn_data()->dataNum(); 
  if (p.dataset.tst_data() != NULL) {
    s << ", #test=" << p.dataset.tst_data()->dataNum(); 
  }
  AzTimeLog::print("Start ... ", s.c_str(), log_out); 
  p.print_hline(log_out); 

  /*---  set up evaluation file if specified  ---*/
  AzOut eval_out; 
  AzOfs ofs; 
  AzOut *eval_out_ptr = reset_eval(p.s_eval_fn, ofs, eval_out); 

  /*---  set up components  ---*/
  if (p.dataset.trn_data()->isRegression()) AzPrint::writeln(log_out, "... regression ... "); 
  if (p.dataset.trn_data()->isMulticat()) AzPrint::writeln(log_out, "... multi-cat ... "); 

  /*---  training and test  ---*/
  AzObjPtrArr<AzpReNet> opa;  /* so that AzpReNet will be automatically deleted at the end of this function ... */
  AzpReNet *renet = alloc_renet(opa, azp); 
 
   /*---  read NN if specified  ---*/  
  if (p.s_fn_for_warmstart.length() > 0) {
    AzTimeLog::print("Reading for warm-start: ", p.s_fn_for_warmstart.c_str(), log_out); 
    renet->read(p.s_fn_for_warmstart.c_str());   
  } 
  
  AzClock clk;  
  renet->training(eval_out_ptr, azp, p.dataset.trn_data(), p.dataset.tst_data(), p.dataset.tst_data2()); 
  AzTimeLog::print("Done ...", log_out); 
  clk.tick(log_out, "elapsed: ");   
  if (ofs.is_open()) ofs.close(); 
}

/*------------------------------------------------------------*/ 
#define kw_mod_fn "model_fn="
#define kw_pred_fn "prediction_fn="
#define kw_do_text "WriteText"
#define kw_do_tok "PerToken"
/*------------------------------------------------------------*/ 
class AzpMain_reNet_predict_Param : public virtual AzpMain_reNet_Param_ {
public:
  AzpDataSetDflt dataset; 
  AzBytArr s_mod_fn, s_pred_fn; 
  bool do_text, do_tok; 
  /*------------------------------------------------*/
  AzpMain_reNet_predict_Param(AzParam &p, const AzOut &out, const AzBytArr &s_action) : do_text(false), do_tok(false) {
    reset(p, out, s_action); 
  }
  void resetParam(const AzOut &out, AzParam &p) {
    const char *eyec = "AzpMain_reNet_predict_Param::resetParam";   
    AzPrint o(out);     
    _resetParam(o, p);      
    bool do_train = false, do_test = true, is_there_y = false; 
    dataset.resetParam(out, p, do_train, do_test, is_there_y);       
    p.vStr(o, kw_mod_fn, s_mod_fn);  
    p.vStr(o, kw_pred_fn, s_pred_fn);   
    p.swOn(o, do_text, kw_do_text); 
    p.swOn(o, do_tok, kw_do_tok); 
    AzXi::throw_if_empty(&s_mod_fn, eyec, kw_mod_fn); 
    AzXi::throw_if_empty(&s_pred_fn, eyec, kw_pred_fn); 
    setupLogDmp(o, p); 
  }
}; 

/*------------------------------------------------------------*/ 
void AzpMain_reNet::predict(int argc, const char *argv[], const AzBytArr &s_action) {
  const char *eyec = "AzpMain_reNet::predict"; 
  AzX::throw_if((argc < 1), AzInputError, eyec, "No arguments");   
  AzParam azp(param_dlm, argc, argv); 
  AzpMain_reNet_predict_Param p(azp, log_out, s_action); 

  AzObjPtrArr<AzpReNet> opa;  /* so that AzpReNet will be automatically deleted at the end of this function ... */
  AzpReNet *renet = alloc_renet_for_test(opa, azp); 

  AzTimeLog::print("Reading: ", p.s_mod_fn.c_str(), log_out); 
  renet->read(p.s_mod_fn.c_str()); 

  p.dataset.reset_data(log_out, renet->classNum());  

  AzTimeLog::print("Predicting ... ", log_out); 
  AzDmatc m_pred; 
  AzClock clk; 
  renet->predict(azp, p.dataset.tst_data(), m_pred, p.do_tok); 
  clk.tick(log_out, "elapsed: "); 

  if (p.do_text) {
    AzTimeLog::print("Writing (text) ", p.s_pred_fn.c_str(), log_out); 
    int digits = 7; 
    m_pred.writeText(p.s_pred_fn.c_str(), digits); 
  }  
  else {
    AzTimeLog::print("Writing (binary) ", p.s_pred_fn.c_str(), log_out); 
    m_pred.write(p.s_pred_fn.c_str()); 
  }
  AzTimeLog::print("Done ... ", log_out); 
}

/*------------------------------------------------------------*/ 
/*------------------------------------------------------------*/ 
class AzpMain_reNet_write_word_mapping_Param : public virtual AzpMain_reNet_Param_ {
public:
  AzBytArr s_lay_type, s_mod_fn, s_lay0_fn, s_wmap_fn; 
  int dsno; 
    
  /*------------------------------------------------*/
  AzpMain_reNet_write_word_mapping_Param(AzParam &azp, const AzOut &out, const AzBytArr &s_action) : dsno(-1) {  
    reset(azp, out, s_action); 
  }
  #define kw_lay_type "layer_type="
  #define kw_lay0_fn "layer0_fn="
  #define kw_wmap_fn "word_map_fn="
  #define kw_dsno "dsno="
  void resetParam(const AzOut &out, AzParam &p) {
    const char *eyec = "AzpMain_reNet_write_word_mapping::resetParam"; 
    AzPrint o(out); 
    _resetParam(o, p);   
    p.vStr(o, kw_lay_type, s_lay_type); 
    p.vStr_prt_if_not_empty(o, kw_lay0_fn, s_lay0_fn); 
    if (s_lay0_fn.length() <= 0) {
      p.vStr_prt_if_not_empty(o, kw_mod_fn, s_mod_fn);  
      dsno = 0; /* default */
      p.vInt(o, kw_dsno, dsno); 
      AzXi::throw_if_negative(dsno, eyec, kw_dsno); 
    }
    p.vStr(o, kw_wmap_fn, s_wmap_fn); 
    AzBytArr s("No input: either "); s << kw_mod_fn << " or " << kw_lay0_fn << " is required."; 
    AzX::throw_if((s_mod_fn.length() <= 0 && s_lay0_fn.length() <= 0), eyec, s.c_str()); 
    AzXi::throw_if_empty(s_wmap_fn, eyec, kw_wmap_fn); 
  }
}; 

/*------------------------------------------------------------*/ 
void AzpMain_reNet::write_word_mapping(int argc, const char *argv[], const AzBytArr &s_action) {
  const char *eyec = "AzpMain_reNet::write_word_mapping"; 
  AzParam azp(param_dlm, argc, argv); 
  AzpMain_reNet_write_word_mapping_Param p(azp, log_out, s_action); 

  AzObjPtrArr<AzpReNet> opa;  /* so that AzpReNet will be automatically deleted at the end of this function ... */
  AzpReNet *renet = alloc_renet_for_test(opa, azp); 
  if (p.s_lay0_fn.length() > 0) {
    azp.check(log_out); 
    renet->write_word_mapping_in_lay(p.s_lay_type, p.s_lay0_fn.c_str(), p.s_wmap_fn.c_str()); 
  }
  else {     
    AzTimeLog::print("Reading: ", p.s_mod_fn.c_str(), log_out); 
    renet->read(p.s_mod_fn.c_str()); 
    azp.check(log_out);
    renet->write_word_mapping(p.s_wmap_fn.c_str(), p.dsno); 
  }
  AzTimeLog::print("Done ... ", log_out); 
}

/*------------------------------------------------------------*/ 
/*------------------------------------------------------------*/ 
class AzpMain_reNet_write_embedded_Param : public virtual AzpMain_reNet_Param_ {
public:
  AzpDataSetDflt ds; 
  AzBytArr s_mod_fn; 
  AzBytArr s_emb_fn; 
  int top_num, tst_minib; 
  /*------------------------------------------------*/
  AzpMain_reNet_write_embedded_Param() : tst_minib(100), top_num(-1) {} /* use this only for displaying help */
  AzpMain_reNet_write_embedded_Param(AzParam &p, const AzOut &out, const AzBytArr &s_action) {
    reset(p, out, s_action); 
  }
  #define kw_top_num "num_top="
  #define kw_emb_fn "embed_fn="  
  #define kw_tst_minib "test_mini_batch_size="  
  void resetParam(const AzOut &out, AzParam &p) {
    const char *eyec = "AzpMain_reNet_write_embedded_Param::resetParam"; 
    AzPrint o(out); 
    _resetParam(o, p);   
    bool do_train = false, do_test = true, is_there_y = false; 
    ds.resetParam(out, p, do_train, do_test, is_there_y); 
    p.vStr(o, kw_mod_fn, s_mod_fn);  
    p.vStr(o, kw_emb_fn, s_emb_fn); 
    p.vInt(o, kw_tst_minib, tst_minib); 
    p.vInt(o, kw_top_num, top_num);     
    AzXi::throw_if_empty(s_mod_fn, eyec, kw_mod_fn); 
    AzXi::throw_if_empty(s_emb_fn, eyec, kw_emb_fn);        
    AzXi::throw_if_nonpositive(tst_minib, eyec, kw_tst_minib); 
  }  
}; 

/*------------------------------------------------------------*/ 
void AzpMain_reNet::write_embedded(int argc, const char *argv[], const AzBytArr &s_action) {
  const char *eyec = "AzpMain_reNet::write_embedded"; 
  AzParam azp(param_dlm, argc, argv); 
  AzpMain_reNet_write_embedded_Param p(azp, log_out, s_action); 

  AzObjPtrArr<AzpReNet> opa;  /* so that AzpReNet will be automatically deleted at the end of this function ... */
  AzpReNet *renet = alloc_renet_for_test(opa, azp);   
  AzTimeLog::print("Reading: ", p.s_mod_fn.c_str(), log_out); 
  renet->read(p.s_mod_fn.c_str()); 
  p.ds.reset_data(log_out, renet->classNum());    
 
  _write_embedded(renet, azp, p.ds.tst_data(), p.tst_minib, p.top_num, p.s_emb_fn.c_str()); 
}

/*------------------------------------------------------------*/ 
void AzpMain_reNet::_write_embedded(AzpReNet *net, AzParam &azp, 
             const AzpData_ *tst, int tst_minib, 
             int feat_top_num, const char *fn) {
  const char *eyec = "AzpMain_reNet::_write_embedded";   
  AzX::throw_if_null(net, eyec); 
  net->init_test(azp, tst); 
  int data_num = tst->dataNum(); 
  AzTimeLog::print("Processing ... #data=", data_num, log_out); 
  
  AzMats_file<AzSmat> mfile; 
  mfile.reset_for_write(fn, data_num); 
  int inc = MAX(1, data_num/50), milestone = inc; 
  for (int dx = 0; dx < data_num; dx += tst_minib) {
    AzTools::check_milestone(milestone, dx, inc); 
    int d_num = MIN(tst_minib, data_num - dx); 
    
    AzDataArr<AzpDataVar_X> data; tst->gen_data(dx, d_num, data); 
    bool is_test = true; 
    AzPmatVar mv; net->up0(is_test, data, mv);
    
    AzDmatc mdc; mdc.reform(mv.rowNum(), mv.colNum()); 
    mv.data()->copy_to(&mdc, 0); 
    AzIFarr ifa; ifa.prepare(mdc.rowNum());
    for (int ix = 0; ix < d_num; ++ix) {   
      int col0, col1; 
      mv.get_begin_end(ix, col0, col1); 
      AzSmat ms(mv.rowNum(), col1-col0); 
      for (int col = col0; col < col1; ++col) {
        const AZ_MTX_FLOAT *vals = mdc.rawcol(col); 
        ifa.reset_norelease();
        for (int row = 0; row < mv.rowNum(); ++row) {
          if (vals[row] == 0) continue; 
          AzX::no_support(vals[row]<0 && feat_top_num>0, eyec, "num_top with negative activation");           
          ifa.put(row, vals[row]); 
        }
        if (feat_top_num > 0 && ifa.size() > feat_top_num) {
          ifa.sort_Float(false); 
          ifa.cut(feat_top_num);           
        }
        ms.col_u(col-col0)->load(&ifa); 
      }
      mfile.write(&ms); 
    }  
  }
  mfile.done(); 
  
  AzTools::finish_milestone(milestone); 
  AzTimeLog::print("Done ... ", log_out); 
}

/*------------------------------------------------*/
/*------------------------------------------------*/
AzOut *AzpMain_reNet::reset_eval(const AzBytArr &s_eval_fn, 
                                AzOfs &ofs, 
                                AzOut &eval_out) {
  AzOut *eval_out_ptr = NULL;  
  if (s_eval_fn.length() > 0) {
    ofs.open(s_eval_fn.c_str(), ios_base::out); 
    ofs.set_to(eval_out); 
    eval_out_ptr = &eval_out; 
  }
  return eval_out_ptr; 
}

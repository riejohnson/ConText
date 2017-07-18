/* * * * *
 *  AzpEv.hpp 
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
 
#ifndef _AZP_EV_HPP_
#define _AZP_EV_HPP_

#include "AzPmat.hpp"
#include "AzTools.hpp"

class Az_PRF {
public:
  double prec, recall, fval, prf; 
  Az_PRF() : prec(0), recall(0), fval(0) {}
  void add(const Az_PRF &inp) {
    prec += inp.prec; 
    recall += inp.recall; 
    fval += inp.fval; 
  }
  void divide(int num) {
    prec /= (double)num;
    recall /= (double)num; 
    fval /= (double)num; 
  }
}; 

class AzpEv {
public:
  virtual void eval_pred(const char *y_fn, const char *p_fn, const char *typ, 
                        const char *th_fn, const AzOut &out) const; 
  static int eval_classif(const AzPmat *m_p, const AzPmat *m_y); /* 0,0,1,0,...0 */ 
  static int eval_classif(const AzPmat *m_p, const AzSmat *m_y); /* 0,0,1,0,...0 */   
  static int eval_classif_largeCat(const AzPmat *m_p, const AzPmat *m_y); /* 0,0,1,0,...0 */   
  static int eval_classif_largeCat(const AzPmat *m_p, const AzSmat *m_y); /* 0,0,1,0,...0 */  
  
  static Az_PRF eval_micro(const AzPmat *m_p, const AzPmat *m_y, /* 0,0,1,0,...0 */ 
                          const AzDvect *v_th);
  static Az_PRF eval_micro(const AzPmat *m_p, const AzPmat *m_y, double th) {
    AzDvect v_th(m_p->rowNum()); v_th.set(th); 
    return eval_micro(m_p, m_y, &v_th); 
  }                          
  static Az_PRF eval_macro(const AzPmat *m_p, const AzPmat *m_y, /* 0,0,1,0,...0 */ 
                             const AzDvect *v_th); 
  static Az_PRF eval_macro(const AzPmat *m_p, const AzPmat *m_y, /* 0,0,1,0,...0 */ 
                             double th) {
    AzDvect v_th(m_p->rowNum()); v_th.set(th); 
    return eval_macro(m_p, m_y, &v_th); 
  }  

  /*---  optimize thresholds using all data  ---*/                        
  virtual void opt_th_all(int argc, const char *argv[]) const {
    AzX::throw_if((argc != 6), AzInputError, "AzpEv::opt_th_all", "p_fns y_fns Each|FBRn|None th_fn do_print eval_fn"); 
    int argx = 0; 
    const char *p_fns = argv[argx++]; 
    const char *y_fns = argv[argx++]; 
    const char *typ = argv[argx++]; 
    const char *th_fn = argv[argx++]; 
    bool do_print = (atol(argv[argx++]) != 0);     
    const char *eval_fn = argv[argx++]; 
    AzOut eval_out; 
    AzOfs ofs; 
    AzOut *eval_out_ptr = AzTools::reset_out(eval_fn, ofs, eval_out); 
    opt_th_all(p_fns, y_fns, typ, th_fn, do_print, log_out, eval_out_ptr); 
  }   
  virtual void opt_th_all(const char *p_fns, const char *y_fns, 
                  const char *typ, const char *th_fn, bool do_print, const AzOut &out, 
                  const AzOut *eval_out) const; 
                                  
  /*---  optimize thresholds; cross validation on each fold  ---*/                        
  virtual void opt_th_cv(int argc, const char *argv[]) const {
    AzX::throw_if((argc != 6), AzInputError, "AzpEv::opt_th_cv", "p_fns y_fns Global|Each|Loop#lc|Sloop#lc #fold do_print eval_fn"); 
    int argx = 0; 
    const char *p_fns = argv[argx++]; 
    const char *y_fns = argv[argx++]; 
    const char *typ = argv[argx++]; 
    int fold_num = atol(argv[argx++]);
    bool do_print = (atol(argv[argx++]) != 0); 
    const char *eval_fn = argv[argx++]; 
    AzOut eval_out; 
    AzOfs ofs; 
    AzOut *eval_out_ptr = AzTools::reset_out(eval_fn, ofs, eval_out); 
    opt_th_cv(p_fns, y_fns, typ, fold_num, do_print, log_out, eval_out_ptr); 
  }                       
  virtual void opt_th_cv(const char *p_fns, const char *y_fns, 
                 const char *typ, int fold_num, bool do_print, const AzOut &out, 
                 AzOut *eval_out) const; 

  /*---  optimize thresholds; cross validation over the entire data  ---*/                        
  virtual void opt_th_cv2(int argc, const char *argv[]) const {
    AzX::throw_if((argc != 6), AzInputError, "AzpEv::opt_th_cv2", "p_fns y_fns Global|Each|Loop#lc|Sloop#lc #fold do_print eval_fn"); 
    int argx = 0; 
    const char *p_fns = argv[argx++]; 
    const char *y_fns = argv[argx++]; 
    const char *typ = argv[argx++]; 
    int fold_num = atol(argv[argx++]);
    bool do_print = (atol(argv[argx++]) != 0); 
    const char *eval_fn = argv[argx++]; 
    AzOut eval_out; 
    AzOfs ofs; 
    AzOut *eval_out_ptr = AzTools::reset_out(eval_fn, ofs, eval_out); 
    opt_th_cv2(p_fns, y_fns, typ, fold_num, do_print, log_out, eval_out_ptr); 
  }                        
  virtual void opt_th_cv2(const char *p_fns, const char *y_fns, 
                  const char *typ, int fold_num, bool do_print, const AzOut &out, 
                  AzOut *eval_out) const; 
                                     
  /*---  evaluate predictions  ---*/                                        
  virtual void eval_pred(int argc, const char *argv[]) const {
    const char *eyec = "eval_pred"; 
    AzX::throw_if((argc != 4 && argc != 5), AzInputError, "AzpEv::eval_pred", "Arguments: y_fn is_y_text pred_fn Regression|Multicat|_  [th_fn]"); 
    int argx = 0; 
    const char *y_fn = argv[argx++]; 
    const char *p_fn = argv[argx++];     
    const char *typ = argv[argx++]; 
    const char *th_fn = (argx < argc) ? argv[argx++] : ""; 
    eval_pred(y_fn, p_fn, typ, th_fn, log_out); 
  }

protected:
  virtual void opt_th(const char *typ, const AzPmat *m_p, const AzPmat *m_y, 
                      bool do_print, const AzOut &out, AzDvect *v_th) const;
  void th_cv(const AzPmat *m_p, const AzPmat *m_y, int fold_num, const char *typ, 
                    bool do_print, const AzOut &out, 
                    AzOut *eval_out, 
                    Az_PRF &micro_prf_sum, Az_PRF &macro_prf_sum, 
                    int &count, int &dnum_sum) const; 

  static void opt_th_loop(const AzPmat *m_p, const AzPmat *m_y, bool do_sort_by_pop, 
                                       int lc_max, bool do_print, const AzOut &out, AzDvect *v_th); 
  static void opt_th_global(const AzPmat *m_p, const AzPmat *m_y, AzDvect *v_th); 
  static void opt_th_each(const AzPmat *m_p, const AzPmat *m_y, AzDvect *v_th, double fbr=-1); 
                                
  static void show_prf(const AzOut *eval_out, const char *str, const char *typ, int data_num, 
                        const Az_PRF &micro_prf, const Az_PRF &macro_prf, const AzOut &out); 
  static Az_PRF get_f1(int Tp, int P_denomi, int R_denomi);                        
  static double sub_opt_th(const AzDvect *v_p, const AzDvect *v_y, double *out_fval=NULL, 
                           int oth_Tp=0, int oth_P_denomi=0, int oth_R_denomi=0, 
                           int *out_Tp=NULL, int *out_P_denomi=NULL, int *out_R_denomi=NULL); 
};
#endif 
                          
/* * * * *
 *  AzpEv.cpp 
 *  Copyright (C) 2014-2017 Rie Johnson
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

#include "AzpEv.hpp"
#include "AzTextMat.hpp"

/*------------------------------------------------------------*/
void AzpEv::opt_th_all(const char *p_fns, const char *y_fns, 
                         const char *typ, const char *th_fn, 
                         bool do_print, const AzOut &out, 
                         const AzOut *eval_out) const
{
  const char *eyec = "AzpEv::opt_th_all"; 

  AzStrPool sp_p_fn(100,100), sp_y_fn(100,100); 
  if (*p_fns == '@') AzTools::readList(p_fns+1, &sp_p_fn); 
  else               sp_p_fn.put(p_fns); 
  if (*y_fns == '@') AzTools::readList(y_fns+1, &sp_y_fn); 
  else               sp_y_fn.put(y_fns); 
  AzX::throw_if((sp_p_fn.size() != sp_y_fn.size()), AzInputError, eyec, "number mismatch: Y and P"); 
  
  AzPmat m_p, m_y; 
  for (int fx = 0; fx < sp_p_fn.size(); ++fx) {
    AzDmat mp; AzTextMat::readMatrix(sp_p_fn.c_str(fx), &mp); 
    AzDmat my; AzTextMat::readMatrix(sp_y_fn.c_str(fx), &my); 
    if (mp.rowNum() != my.rowNum() || mp.colNum() != my.colNum()) {
      AzBytArr s("Y and P must have the same shape: "); s << sp_p_fn.c_str(fx) << "," << sp_y_fn.c_str(fx); 
      AzX::throw_if(true, AzInputError, eyec, s.c_str()); 
    }
    if (fx == 0) {
      m_p.set(&mp); 
      m_y.set(&my); 
    }
    else {
      AzPmat m_p_tmp(&mp); m_p.cbind(&m_p_tmp); 
      AzPmat m_y_tmp(&my); m_y.cbind(&m_y_tmp); 
    }
  }
  
  /*---  choose thresholds  ---*/
  AzDvect v_th; 
  opt_th(typ, &m_p, &m_y, do_print, out, &v_th); 

  /*---  eval F (optimistic)  ---*/
  Az_PRF micro_prf = eval_micro(&m_p, &m_y, &v_th); 
  Az_PRF macro_prf = eval_macro(&m_p, &m_y, &v_th); 
  show_prf(eval_out, "", typ, m_p.colNum(), micro_prf, macro_prf, out);   
  
  /*---  write threshold  ---*/
  if (AzTools::isSpecified(th_fn)) {
    AzTimeLog::print("Writing ", th_fn, out); 
    int digits = 10; 
    v_th.writeText(th_fn, digits); 
    AzTimeLog::print("Done ... ", out); 
  }
}

/*------------------------------------------------------------*/
void AzpEv::opt_th_cv(const char *p_fns, const char *y_fns, 
                         const char *typ, int fold_num,  
                         bool do_print, const AzOut &out, 
                         AzOut *eval_out) const
{
  const char *eyec = "AzpEv::opt_th_cv"; 
  
  AzStrPool sp_p_fn(100,100), sp_y_fn(100,100); 
  if (*p_fns == '@') AzTools::readList(p_fns+1, &sp_p_fn); 
  else               sp_p_fn.put(p_fns); 
  if (*y_fns == '@') AzTools::readList(y_fns+1, &sp_y_fn); 
  else               sp_y_fn.put(y_fns); 
  AzX::throw_if((sp_p_fn.size() != sp_y_fn.size()), AzInputError, eyec, "number mismatch: Y and P");
  Az_PRF micro_prf_sum, macro_prf_sum; 
  int dnum_sum = 0, count = 0; 
  for (int fx = 0; fx < sp_p_fn.size(); ++fx) {
    AzDmat mp; AzTextMat::readMatrix(sp_p_fn.c_str(fx), &mp); 
    AzDmat my; AzTextMat::readMatrix(sp_y_fn.c_str(fx), &my); 
    if (mp.rowNum() != my.rowNum() || mp.colNum() != my.colNum()) {
      AzBytArr s("Y and P must have the same shape: "); s.c(sp_p_fn.c_str(fx)); s.c(","); s.c(sp_y_fn.c_str(fx)); 
      AzX::throw_if(true, AzInputError, eyec, s.c_str()); 
    }
    AzPmat m_p(&mp), m_y(&my); 
    th_cv(&m_p, &m_y, fold_num, typ, do_print, out, eval_out, 
          micro_prf_sum, macro_prf_sum, count, dnum_sum);                      
  }
  AzBytArr s("avg_of_"); s.cn(count); 
  int dnum_avg = dnum_sum / count; 
  
  Az_PRF micro_prf_avg = micro_prf_sum; micro_prf_avg.divide(count); 
  Az_PRF macro_prf_avg = macro_prf_sum; macro_prf_avg.divide(count); 
  show_prf(eval_out, s.c_str(), typ, dnum_avg, micro_prf_avg, macro_prf_avg, out);   
}     

/*------------------------------------------------------------*/
void AzpEv::opt_th_cv2(const char *p_fns, const char *y_fns, 
                          const char *typ, int fold_num,  
                          bool do_print, 
                          const AzOut &out, 
                          AzOut *eval_out) const
{
  const char *eyec = "AzpEv::opt_th_cv2"; 
  AzTimeLog::print(eyec, out); 
  AzStrPool sp_p_fn(100,100), sp_y_fn(100,100); 
  if (*p_fns == '@') AzTools::readList(p_fns+1, &sp_p_fn); 
  else               sp_p_fn.put(p_fns); 
  if (*y_fns == '@') AzTools::readList(y_fns+1, &sp_y_fn); 
  else               sp_y_fn.put(y_fns); 
  AzX::throw_if((sp_p_fn.size() != sp_y_fn.size()), AzInputError, eyec, "number mismatch: Y and P"); 
  AzPmat m_p, m_y; 
  for (int fx = 0; fx < sp_p_fn.size(); ++fx) {
    AzDmat mp; AzTextMat::readMatrix(sp_p_fn.c_str(fx), &mp); 
    AzDmat my; AzTextMat::readMatrix(sp_y_fn.c_str(fx), &my); 
    if (mp.rowNum() != my.rowNum() || mp.colNum() != my.colNum()) {
      AzBytArr s("Y and P must have the same shape: "); s.c(sp_p_fn.c_str(fx)); s.c(","); s.c(sp_y_fn.c_str(fx)); 
      AzX::throw_if(true, AzInputError, eyec, s.c_str()); 
    }
    if (fx == 0) {
      m_p.set(&mp); 
      m_y.set(&my); 
    }
    else {
      AzPmat m_p_tmp(&mp); m_p.cbind(&m_p_tmp); 
      AzPmat m_y_tmp(&my); m_y.cbind(&m_y_tmp); 
    }
  }  
  
  Az_PRF micro_prf_sum, macro_prf_sum; 
  int dnum_sum = 0, count = 0; 
  th_cv(&m_p, &m_y, fold_num, typ, do_print, out, eval_out, 
        micro_prf_sum, macro_prf_sum, count, dnum_sum);     
  
  int data_num = m_p.colNum(); 
  AzX::throw_if((data_num < fold_num), eyec, "Too little data"); 

  AzBytArr s("avg_of_"); s.cn(count); 
  int dnum_avg = dnum_sum / count; 
  
  Az_PRF micro_prf_avg = micro_prf_sum; micro_prf_avg.divide(count); 
  Az_PRF macro_prf_avg = macro_prf_sum; macro_prf_avg.divide(count);   
  show_prf(eval_out, s.c_str(), typ, dnum_avg, micro_prf_avg, macro_prf_avg, out);   
} 

/*------------------------------------------------------------*/
void AzpEv::th_cv(const AzPmat *m_p, const AzPmat *m_y, 
                     int fold_num, const char *typ, 
                     bool do_print, const AzOut &out, 
                     AzOut *eval_out, 
                     Az_PRF &micro_prf_sum, Az_PRF &macro_prf_sum, 
                     int &count, 
                     int &dnum_sum) const 
{
  const char *eyec = "AzpEv::th_cv"; 
  int data_num = m_p->colNum(); 
  AzX::throw_if((data_num < fold_num), eyec, "Too little data"); 
    
  AzIntArr ia; ia.range(0, data_num); 
  AzTools::shuffle(-1, &ia); 
  int beg = 0; 
  for (int ix = 0; ix < fold_num; ++ix) {
    int end = (ix==fold_num-1) ? data_num : beg + data_num/fold_num; 
    AzIntArr ia_trn, ia_tst; 
    ia_tst.concat(ia.point()+beg, end-beg); 
    if (beg > 0)        ia_trn.concat(ia.point(), beg); 
    if (end < data_num) ia_trn.concat(ia.point()+end, data_num-end); 

    AzPmat m_p_trn; m_p_trn.set(m_p, ia_trn.point(), ia_trn.size()); 
    AzPmat m_y_trn; m_y_trn.set(m_y, ia_trn.point(), ia_trn.size()); 
    AzDvect v_th; 
    opt_th(typ, &m_p_trn, &m_y_trn, do_print, out, &v_th); 

    AzPmat m_p_tst; m_p_tst.set(m_p, ia_tst.point(), ia_tst.size()); 
    AzPmat m_y_tst; m_y_tst.set(m_y, ia_tst.point(), ia_tst.size());  
    Az_PRF micro_prf = eval_micro(&m_p_tst, &m_y_tst, &v_th);     
    micro_prf_sum.add(micro_prf); 
    Az_PRF macro_prf = eval_macro(&m_p_tst, &m_y_tst, &v_th);     
    macro_prf_sum.add(macro_prf); 
    ++count;       
    dnum_sum += m_p_trn.colNum(); 
            
    AzBytArr s("("); s << ix << ")," << typ << ",#data," << m_p_trn.colNum(); 
    s << ",perf," << 1-micro_prf.fval << ",PR," << micro_prf.prec << "," << micro_prf.recall; 
    if (do_print) AzPrint::writeln(out, s);             
            
    beg = end; 
  }
}

/*------------------------------------------------------------*/
void AzpEv::opt_th(const char *typ, 
                      const AzPmat *m_p, const AzPmat *m_y, 
                      bool do_print, const AzOut &out, 
                      AzDvect *v_th) const
{
  const char *eyec = "AzpEv::opt_th"; 
  AzBytArr s_typ(typ); 
  if (s_typ.compare("None") == 0) {
    v_th->reform(m_p->rowNum()); 
    v_th->set(0.5); 
  }
  else if (s_typ.compare("Each") == 0) opt_th_each(m_p, m_y, v_th); 
  else if (s_typ.beginsWith("FBR")) {
    double fbr = atof(s_typ.c_str()+3); 
    AzX::throw_if((fbr <= 0 || fbr >= 1), AzInputError, eyec, "FBR must be followd by a value in (0,1)"); 
    opt_th_each(m_p, m_y, v_th, fbr); 
  }
  else {
    AzX::throw_if(true, AzInputError, eyec, "invalid type: ", typ); 
  }
}

/*------------------------------------------------------------*/
void AzpEv::opt_th_each(const AzPmat *m_p, const AzPmat *m_y, 
                           AzDvect *v_th, double fbr)
{
  const char *eyec = "AzpEv::opt_th_each"; 
  
  m_p->shape_chk_tmpl(m_y, eyec, "Y and P must have the same shape.");   
  AzPmat m_p_tran; m_p_tran.transpose_from(m_p); 
  AzPmat m_y_tran; m_y_tran.transpose_from(m_y); 
  int c_num = m_p_tran.colNum(); 
  v_th->reform(c_num); 
  double P_sum = 0, R_sum = 0, f_sum = 0; 
  for (int cx = 0; cx < c_num; ++cx) {
    AzDvect v_p; m_p_tran.get(cx, &v_p); 
    AzDvect v_y; m_y_tran.get(cx, &v_y);   
    double fval; 
    int Tp, P_denomi, R_denomi;
    double th = sub_opt_th(&v_p, &v_y, &fval, 
                           0, 0, 0, &Tp, &P_denomi, &R_denomi); 
    P_sum += (P_denomi != 0) ? Tp/P_denomi : 0; 
    R_sum += (R_denomi != 0) ? Tp/R_denomi : 0; 
    f_sum += fval; 
    if (fbr > 0 && fval < fbr) {
      th = m_p_tran.max(cx);
    }
    v_th->set(cx, th); 
  }
}

/*------------------------------------------------------------*/
Az_PRF AzpEv::get_f1(int Tp, int P_denomi, int R_denomi)
{                         
  double prec = (Tp <= 0) ? 0 : ((double)Tp/(double)P_denomi); 
  double recall = (Tp <= 0) ? 0 : ((double)Tp/(double)R_denomi); 
  double fval = (prec+recall != 0) ? 2*prec*recall/(prec+recall) : 0; 
  Az_PRF prf; 
  prf.prec = prec; 
  prf.recall = recall; 
  prf.fval = fval; 
  return prf; 
}
    
/*------------------------------------------------------------*/
/* static */
double AzpEv::sub_opt_th(const AzDvect *v_p, const AzDvect *v_y, 
                        double *out_fval, 
                        int oth_Tp, int oth_P_denomi, int oth_R_denomi, 
                        int *out_Tp, int *out_P_denomi, int *out_R_denomi)
{
  const char *eyec = "AzpEv::opt_th"; 
  double eps = 1e-10; 
  int data_num = v_p->rowNum(); 
  AzX::throw_if((data_num <= 0), eyec, "No data"); 
  AzX::throw_if((data_num != v_y->rowNum()), eyec, "Y and P must have the same shape"); 
  const double *p = v_p->point(); 
  const double *y = v_y->point(); 
  AzIFarr ifa_y_pred; 
  int R_denomi = 0; /* true positive + false negative */
  int ix; 
  for (ix = 0; ix < data_num; ++ix) {
    int lab = (int)y[ix]; 
    ifa_y_pred.put(lab, p[ix]); 
    if (lab == 1) ++R_denomi; 
  }
  
  ifa_y_pred.sort_Float(false); /* descending order */
  Az_PRF best_prf = get_f1(oth_Tp, oth_P_denomi, R_denomi+oth_R_denomi); 
  double best_th = MAX(0.5, ifa_y_pred.get(0)+1e-10); 
  int Tp = 0, best_Tp = 0, best_P_denomi = 0; 
  for (ix = 0; ix < data_num; ) {  
    int lab; 
    double pred = ifa_y_pred.get(ix, &lab);    
    if (lab == 1) ++Tp; 
    double pred1 = pred - eps; 
    int ix1; 
    for (ix1 = ix+1; ix1 < data_num; ++ix1) {
      int lab1; 
      pred1 = ifa_y_pred.get(ix1, &lab1); 
      if (pred1 != pred) break; 
      if (lab1 == 1) ++Tp; 
    }

    /*---  cut before [ix1]  ---*/
    int P_denomi = ix1; 
    Az_PRF prf = get_f1(Tp+oth_Tp, P_denomi+oth_P_denomi, R_denomi+oth_R_denomi);   
    if (prf.fval > best_prf.fval) {
      best_th = (pred+pred1)/2; 
      best_prf = prf; 
      best_Tp = Tp; 
      best_P_denomi = P_denomi; 
    }
    ix = ix1; 
  }
  if (out_Tp != NULL) *out_Tp = best_Tp; 
  if (out_P_denomi != NULL) *out_P_denomi = best_P_denomi; 
  if (out_R_denomi != NULL) *out_R_denomi = R_denomi; 
  if (out_fval != NULL) *out_fval = best_prf.fval; 
  return best_th; 
}

/*------------------------------------------------------------*/
/* micro-average */
Az_PRF AzpEv::eval_micro(const AzPmat *m_p, 
                         const AzPmat *m_y, /* 0,0,1,0,...0 */ 
                         const AzDvect *v_th)
{
  AzX::throw_if((v_th == NULL), "AzpEv::eval_micro", "No threshold?!"); 
  AzPmat v(v_th); 
  AzPmat m_th; m_th.repmat_from(&v, 1, m_p->colNum()); 
  AzPmat m(m_p); m.sub(&m_th); m.mark_positive();  /* (p>th)?1:0 */
  int P_denomi = m.nz(); /* true-positive + false-positive */
  int R_denomi = m_y->nz(); /* true-positive + false-negative */  
  m.elm_multi(m_y); /* element-wise multiplication */
  int tp = m.nz(); /* true-positive */  
  return get_f1(tp, P_denomi, R_denomi); 
}

/*------------------------------------------------------------*/
/* macro-average */
Az_PRF AzpEv::eval_macro(const AzPmat *m_p, 
                           const AzPmat *m_y, /* 0,0,1,1,...0 */ 
                           const AzDvect *v_th)
{
  AzX::throw_if((v_th == NULL), "AzpEv::eval_macro", "No threshold?!"); 
  AzPmat v(v_th); 
  AzPmat m_th; m_th.repmat_from(&v, 1, m_p->colNum()); 
  AzPmat m(m_p); m.sub(&m_th); m.mark_positive(); /* (p>?th)?1:0 */
  
  AzPmat m_R_denomi; m_R_denomi.rowSum(m_y); 
  AzPmat m_P_denomi; m_P_denomi.rowSum(&m); 

  m.elm_multi(m_y); /* element-wise multiplication */
  AzPmat m_Tp; m_Tp.rowSum(&m); 
  
  bool do_inv = true; 
  AzPmat m_prec(&m_Tp); m_prec.elm_multi(&m_P_denomi, do_inv); 
  AzPmat m_recall(&m_Tp); m_recall.elm_multi(&m_R_denomi, do_inv); 
  Az_PRF prf; 
  double prec = m_prec.sum() / (double)m_prec.size(); 
  double recall = m_recall.sum() / (double)m_recall.size();  
  AzPmat m_f(&m_prec); m_f.elm_multi(&m_recall); m_f.multiply(2);  /* 2pr */
  AzPmat m_f_denomi(&m_prec); m_f_denomi.add(&m_recall); /* p+r */
  m_f.elm_multi(&m_f_denomi, true);  /* 2pr/(p+r) */
  prf.fval = m_f.sum() / (double)m_f.size();   
  prf.prec = prec; 
  prf.recall = recall; 
  return prf; 
}

/*------------------------------------------------------------*/
/* static */
int AzpEv::eval_classif(const AzPmat *m_p, const AzPmat *m_y) /* ...,0,1,0,... */ 
{                
  int d_num = m_p->colNum(); 
  AzIntArr ia_pred, ia_true; 
  m_p->max_eachCol(&ia_pred); 
  m_y->max_eachCol(&ia_true); 
  int correct = 0; 
  for (int dx = 0; dx < d_num; ++dx) if (ia_pred.get(dx) == ia_true.get(dx)) ++correct; 
  return correct; 
} 

/*------------------------------------------------------------*/
/* static */
int AzpEv::eval_classif(const AzPmat *m_p, 
                           const AzSmat *m_y) /* 0,0,1,0,...0 */ 
{                
  int d_num = m_p->colNum(); 
  AzIntArr ia_pred; 
  m_p->max_eachCol(&ia_pred); 
  int correct = 0; 
  for (int dx = 0; dx < d_num; ++dx) {
    int true_cat; 
    m_y->col(dx)->max(&true_cat); 
    if (ia_pred.get(dx) == true_cat) ++correct; 
  }
  return correct; 
} 

/*------------------------------------------------------------*/
/* static */
int AzpEv::eval_classif_largeCat(const AzPmat *m_p, const AzPmat *m_y) { /* 0,0,1,0,...0 */ 
  AzSmat ms_y; m_y->get(&ms_y); 
  return eval_classif_largeCat(m_p, &ms_y); 
}

/*------------------------------------------------------------*/
/* static */
int AzpEv::eval_classif_largeCat(const AzPmat *m_p, const AzSmat *m_y) { /* 0,0,1,0,...0 */ 
  int d_num = m_p->colNum(); 
  AzIntArr ia; ia.prepare(d_num); 
  for (int dx = 0; dx < d_num; ++dx) {
    int true_cat; m_y->col(dx)->max(&true_cat); 
    ia.put(true_cat); 
  }  
  AzPmat m_val; m_p->get_eachCol(ia.point(), ia.size(), &m_val); 
  AzPmat mp(m_p); mp.mark_gt_colth(&m_val, 1); /* set 1 if greater than the prediction value for the true cat */
  AzPmat m; m.colSum(&mp); m.binarize1(); /* for each data point (column), 1 is set if a wrong cat has the greatest prediction value */
  int correct = d_num - (int)m.sum(); 
  return correct; 
} 

/*-----------------------------------------------------------------*/
void AzpEv::eval_pred(const char *y_fn, const char *p_fn, const char *typ, 
                        const char *th_fn, const AzOut &out) const
{
  const char *eyec = "AzpEv::eval_pred"; 
  AzDmat md_y, md_p; 
  AzTextMat::readMatrix(y_fn, &md_y); 
  AzPmat m_y(&md_y); md_y.destroy(); 
  AzTextMat::readMatrix(p_fn, &md_p); 
  AzPmat m_p(&md_p); md_p.destroy(); 
  int data_num = m_y.colNum(); 
  AzX::throw_if((m_p.colNum() != data_num), AzInputError, eyec, "#data mismatch between truth and prediction"); 
  AzBytArr s("#data,"); s << data_num; 
  if (*typ == 'R') { /* regression */
    AzX::throw_if((AzTools::isSpecified(th_fn)), AzInputError, eyec, "Cannot use thresholds for regression"); 
    double rmse = sqrt(m_p.squareSum()/(double)data_num);   
    s << ",rmse,perf," << rmse; 
  }
  else if (*typ == 'M') { /* multi-label classification */
    Az_PRF micro_prf, macro_prf; 
    if (!AzTools::isSpecified(th_fn)) {
      double th = 0.5; 
      AzBytArr s("!WARNING! Fixing thresholds to "); s << th << ".  Sub-optimal results are expected."; 
      AzPrint::writeln(out, s.c_str()); 
      AzDvect v_th(m_p.rowNum()); v_th.set(th);       
      micro_prf = eval_micro(&m_p, &m_y, &v_th); 
      macro_prf = eval_macro(&m_p, &m_y, &v_th); 
    }
    else {
      AzDvect v_th; 
      AzTimeLog::print("Reading ", th_fn, log_out); 
      AzTextMat::readVector(th_fn, &v_th);
      micro_prf = eval_micro(&m_p, &m_y, &v_th); 
      macro_prf = eval_macro(&m_p, &m_y, &v_th); 
    }
    show_prf(NULL, "", "", m_p.colNum(), micro_prf, macro_prf, out);    
  }
  else { /* single-label classification */
    AzX::throw_if((AzTools::isSpecified(th_fn)), AzInputError, eyec, "Cannot use thresholds for single-cat classification"); 
    int correct = eval_classif(&m_p, &m_y); 
    double acc = (double)correct / (double)data_num;
    s << ",1-acc,perf," << 1-acc; 
  }
  AzPrint::writeln(log_out, s); 
} 

/*------------------------------------------------------------*/
void AzpEv::show_prf(const AzOut *eval_out, 
                         const char *str, 
                         const char *typ, 
                         int data_num, 
                         const Az_PRF &micro_prf, 
                         const Az_PRF &macro_prf, 
                         const AzOut &out) 
{                          
  AzBytArr s(str); s << "," << typ << ",#data," << data_num; 
  s << ",micro," << 1-micro_prf.fval; 
  s << ",PR," << micro_prf.prec << "," << micro_prf.recall; 
  s << ",macro," << 1-macro_prf.fval; 
  s << ",PR," << macro_prf.prec << "," << macro_prf.recall; 
  if (eval_out != NULL) AzPrint::writeln(*eval_out, s); 
  AzPrint::writeln(out, s); 
}

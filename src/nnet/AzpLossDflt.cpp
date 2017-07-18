/* * * * *
 *  AzpLossDflt.cpp
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

#include "AzpLossDflt.hpp"
#include "AzpEv.hpp"
#include "AzpData_.hpp"
#include "AzPmatApp.hpp"

/*------------------------------------------------------------*/ 
/*------------------------------------------------------------*/ 
void AzpLossDflt::get_loss_deriv(const AzPmat *m_p, /* output of the top layer, col: data points */
                             const AzpData_ *data, 
                             const int *dxs, int d_num, 
                             /*---  output  ---*/
                             AzPmat *m_loss_deriv,  /* row: nodes of the output layer, col: data points */
                             double *loss, /* may be NULL: added */
                             const AzPmatSpa *_ms_y, /* may be NULL */
                             const AzPmat *_md_y) 
const  
{
  double my_loss = 0, *lossptr = (loss==NULL) ? NULL : &my_loss; 
  AzPmat _m; 
  const AzPmat *m_dw = gen_dw(data, dxs, d_num, &_m);   
  if (data->is_sparse_y()) { /* sparse target */
    if (_ms_y == NULL && _md_y == NULL) {
      AzPmatSpa m_y_tmp; 
      data->gen_targets(dxs, d_num, &m_y_tmp); 
      _get_loss_deriv(m_p, &m_y_tmp, m_dw, m_loss_deriv, lossptr); 
    }  
    else if (_ms_y != NULL) _get_loss_deriv(m_p, _ms_y, m_dw, m_loss_deriv, lossptr);       
    else                    _get_loss_deriv(m_p, _md_y, m_dw, m_loss_deriv, lossptr); 
  }
  else { /* dense target */
    AzX::no_support((_ms_y != NULL || _md_y != NULL), "AzpLossDflt::get_loss_deriv", "dense target with preparation by cnet3"); 
    AzPmat m_y; 
    data->gen_targets(dxs, d_num, &m_y); 
    _get_loss_deriv(m_p, &m_y, m_dw, m_loss_deriv, lossptr);    
  }
  if (loss != NULL) *loss += *lossptr; 
}

/*------------------------------------------------------------*/ 
template <class M> /* M: AzPmat | AzPmatSpa */
void AzpLossDflt::_get_loss_deriv(const AzPmat *m_p, /* output of the top layer, col: data points */
                              const M *m_y,
                              const AzPmat *m_dw, 
                              AzPmat *m_loss_deriv,  /* row: nodes of the output layer, col: data points */
                              double *loss) /* may be NULL: added */
const                       
{
  AzPmat _m; 
  const AzPmat *m_y_mask = gen_y_mask(m_y, &_m);

  /*---  regression | classification one-vs-others square loss  ---*/
  if (loss_type == AzpLossDflt_Square) {
    m_loss_deriv->set(m_p);
    AzPs::sub(m_loss_deriv, m_y); /* p-y */
    if (loss != NULL) {
      if (m_dw != NULL || m_y_mask != NULL) {
        AzPmat m(m_loss_deriv); m.square(); /* (p-y)^2 */
        if (m_y_mask != NULL) m.elm_multi(m_y_mask); 
        if (m_dw != NULL) m.multiply_eachcol(m_dw); 
        *loss += 0.5*m.sum(); 
      }  
      else {
        *loss += 0.5*m_loss_deriv->squareSum(); /* (1/2)(p-y)^2 */
      }
    }
  }
  /*---  cross entrpy loss for auto encoder  ---*/
  else if (loss_type == AzpLossDflt_CE) {
    bool doInverse = true; 
    AzPmat m0(m_p); m0.add(azc_epsilon); /* p+e */
    AzPsAzPmat(m1,m_y); m1.elm_multi(&m0, doInverse); /* y/(p+e) */
    AzPmat m2(m_p->rowNum(), m_p->colNum()); m2.set(1+azc_epsilon); m2.sub(m_p); /* 1+e-p */
    AzPsAzPmat(m3,m_y); m3.multiply(-1); m3.add(1); /* 1-y */
    m3.elm_multi(&m2, doInverse); /* (1-y)/(1+e-p) */
    m3.sub(&m1); /* -y/(p+e) + (1-y)/((1+e-p) */
    m_loss_deriv->set(&m3);    
  }
  else if (loss_type == AzpLossDflt_BinLogi ||  /* y in {-1,1} */
           loss_type == AzpLossDflt_BinLogi2) { /* y in {0,1} */
    AzPmatApp u; 
    bool is_01 = (loss_type == AzpLossDflt_BinLogi2); 
    AzPsAzPmat(my,m_y); AzPmat m_l, *m_loss = (loss != NULL) ? &m_l : NULL; 
    u.binlogi_deriv(is_01, m_p, &my, m_loss_deriv, m_loss); 
    if (loss != NULL) {
      if (m_y_mask != NULL) m_loss->elm_multi(m_y_mask); 
      if (m_dw != NULL) m_loss->multiply_eachcol(m_dw); 
      *loss += m_loss->sum(); 
    }  
  }    
  /*---  classification: softmax log loss  ---*/
  else { /* assume y is in {0,1} */
    AzPmat m_prob(m_p); to_prob2(&m_prob); /* convert pred_y to probabilities */
    m_loss_deriv->set(&m_prob); 
    AzPs::sub(m_loss_deriv, m_y); /* (in_class) ? (prob-1) : prob */  
    if (loss != NULL) {
      m_prob.add(azc_epsilon); m_prob.log(); AzPs::elm_multi(&m_prob, m_y); 
      if (m_y_mask != NULL) m_prob.elm_multi(m_y_mask); 
      if (m_dw != NULL) m_prob.multiply_eachcol(m_dw); 
      *loss -= m_prob.sum();
    }
  }
  if (m_y_mask != NULL) m_loss_deriv->elm_multi(m_y_mask); 
  if (m_dw != NULL) m_loss_deriv->multiply_eachcol(m_dw); 
}
template void AzpLossDflt::_get_loss_deriv<AzPmat>(const AzPmat *, const AzPmat *, 
                           const AzPmat *, AzPmat *, double *loss) const; 
template void AzpLossDflt::_get_loss_deriv<AzPmatSpa>(const AzPmat *, const AzPmatSpa *, 
                           const AzPmat *, AzPmat *, double *loss) const; 

/*------------------------------------------------------------*/ 
double AzpLossDflt::get_loss(const AzpData_ *data, 
                             int dx_begin, int d_num, 
                             const AzPmat *m_p)
const
{
  double loss = 0;
  AzPmat _m; 
  const AzPmat *m_dw = gen_dw(data, dx_begin, d_num, &_m);   
  if (data->is_sparse_y()) { /* sparse target */
    AzPmatSpa m_y; 
    data->gen_targets(dx_begin, d_num, &m_y); 
    loss += _get_loss(m_p, &m_y, m_dw); 
  }
  else { /* dense target */
    AzPmat m_y; 
    data->gen_targets(dx_begin, d_num, &m_y);
    loss += _get_loss(m_p, &m_y, m_dw); 
  }
  return loss; 
}

/*------------------------------------------------------------*/ 
template <class M> /* M: AzPmat | AzPmatSpa */
double AzpLossDflt::_get_loss(const AzPmat *m_p, /* output values from the top layer */
                              const M *m_y, 
                              const AzPmat *m_dw) /* may be NULL */
const                          
{
  AzPmat m_loss; 
  /*---  regression or squre loss: (p-y)^2/2---*/
  if (loss_type == AzpLossDflt_Square) {
    m_loss.set(m_p); AzPs::sub(&m_loss, m_y); /* p-y */
    m_loss.square();  m_loss.divide(2); /* (p-y)^2/2 */
  }
  /*---  for auto encoder, p must be in (0,1)  ---*/
  else if (loss_type == AzpLossDflt_CE) {
    AzPmat m1(m_p); m1.add(azc_epsilon); m1.log(); AzPs::elm_multi(&m1, m_y); /* y log(p) */
    AzPmat m2(m_p->rowNum(), m_p->colNum()); m2.set(1 + azc_epsilon); m2.sub(m_p);
    m2.log(); /* log(1-p) */ 
    AzPmat m3(m_y->rowNum(), m_y->colNum()); m3.set(1); AzPs::sub(&m3, m_y); /* 1-y */
    m2.elm_multi(&m3); /* (1-y)log(1-p) */    
    m_loss.set(&m1, -1); m_loss.sub(&m2);
  }
  else if (loss_type == AzpLossDflt_BinLogi ||  /* y in {-1,1} */
           loss_type == AzpLossDflt_BinLogi2) { /* y in {0,1} */
    bool is_01 = (loss_type == AzpLossDflt_BinLogi2); 
    AzPsAzPmat(my, m_y); 
    AzPmatApp u;
    u.binlogi_loss(is_01, m_p, &my, &m_loss); 
  }    
  /*---  classification: softmax log loss  ---*/
  else {
    AzPmat m_prob(m_p); to_prob2(&m_prob); /* convert to probabilities */
    m_prob.add(azc_epsilon); 
    m_prob.log();                         /* log(prob) */
    AzPs::elm_multi(&m_prob, m_y);   
    m_loss.set(&m_prob, -1); 
  }

  AzPmat _m; 
  const AzPmat *m_y_mask = gen_y_mask(m_y, &_m);           
  if (m_y_mask != NULL) m_loss.elm_multi(m_y_mask); 
  if (m_dw != NULL) m_loss.multiply_eachcol(m_dw); 
  double loss = m_loss.sum(); 
  return loss; 
}
template double AzpLossDflt::_get_loss<AzPmat>(const AzPmat *, const AzPmat *, const AzPmat *) const; 
template double AzpLossDflt::_get_loss<AzPmatSpa>(const AzPmat *, const AzPmatSpa *, const AzPmat *) const; 

/*------------------------------------------------------------*/
void AzpLossDflt::to_prob2(AzPmat *m_prob) /* for softmax */
const
{
  m_prob->exp();  /* prob <- exp(prob) */
  m_prob->normalize1(); /* divide by colSum */
}

/*------------------------------------------------------------*/ 
double AzpLossDflt::test_eval(const AzpData_ *data, 
                            AzPmat *m_p, 
                            double *out_loss, 
                            const AzOut &out, 
                            AzBytArr *s_pf) const
{
  const char *eyec = "AzpLossDflt::test_eval"; 
/*  int data_size = data->dataNum(); */
  int data_size = m_p->colNum(); /* changed: 10/07/2015 for variable-sized target */
  AzPmat _m; 
  const AzPmat *m_dw = gen_dw(data, 0, data_size, &_m); 
  if (data->is_sparse_y()) {
    AzSmat ms_y; 
    data->gen_targets(0, data->dataNum(), &ms_y); 
    return _test_eval(data_size, m_p, &ms_y, m_dw, out_loss, out, s_pf);     
  }
  else {
    AzPmat m_y; data->gen_targets(0, data->dataNum(), &m_y); 
    return _test_eval(data_size, m_p, &m_y, m_dw, out_loss, out, s_pf); 
  }
}

/*------------------------------------------------------------*/ 
/* dense target */
double AzpLossDflt::_test_eval(
                          int data_size, 
                          AzPmat *m_p, 
                          const AzPmat *m_y, 
                          const AzPmat *m_dw, 
                          double *out_loss,
                          const AzOut &out,
                          AzBytArr *s_pf) const
{
  /*---  regression  ---*/
  if (is_regression) {
    double loss = _get_loss(m_p, m_y, m_dw) / (double)data_size; 
    if (out_loss != NULL) *out_loss = loss; 
    if (s_pf != NULL) s_pf->reset("loss"); 
    return loss; 
  }
  else if (is_multicat) {   
    if (out_loss != NULL) *out_loss = get_loss(m_p, m_y) / (double)data_size; 
    if (do_force_eval) {
      double th = (loss_type == AzpLossDflt_BinLogi) ? 0 : 0.5; /* not good thresholing */    
      Az_PRF prf; AzBytArr s; 
      if (do_firstcat_only) {      
        AzPmat mp(1, m_p->colNum()); mp.set_rowwise(0, 1, m_p, 0); 
        AzPmat my(1, m_y->colNum()); my.set_rowwise(0, 1, m_y, 0);         
        prf = AzpEv::eval_micro(&mp, &my, th); 
        s << "0precision=" << prf.prec << ",0recall=" << prf.recall;       
      }
      else {
        prf = AzpEv::eval_micro(m_p, m_y, th); 
        s << "precision=" << prf.prec << ",recall=" << prf.recall;  
      }
      AzPrint::writeln(out, s); 
      if (s_pf != NULL) s_pf->reset("1-F"); 
      return 1-prf.fval; 
    }
    else {
      if (s_pf != NULL) s_pf->reset(AzpEvalNoSupport); 
      return -1;       
    }
  }
  /*---  classification  ---*/
  else {
    int correct = AzpEv::eval_classif(m_p, m_y); 
    double acc = (double)correct / (double)data_size; 
    if (out_loss != NULL) *out_loss = get_loss(m_p, m_y) / (double)data_size; 
    if (s_pf != NULL) s_pf->reset("err");     
    return 1-acc; 
  }
}
 
/*------------------------------------------------------------*/ 
/* sparse target */
double AzpLossDflt::_test_eval(int data_size, 
                          AzPmat *m_p, 
                          const AzSmat *ms_y, 
                          const AzPmat *m_dw, 
                          double *out_loss,
                          const AzOut &out, 
                          AzBytArr *s_pf) const
{
  const char *eyec = "AzpLossDflt::_test_eval(sparse)"; 

  AzPmatSpa m_y; m_y.set(ms_y, false); /* no row index */
  
  /*---  regression  ---*/
  if (is_regression) {
    double loss = _get_loss(m_p, &m_y, m_dw) / (double)data_size; 
    if (out_loss != NULL) *out_loss = loss; 
    if (s_pf != NULL) s_pf->reset("loss");      
    return loss; 
  }
  /*---  multi-label classification  ---*/
  else if (is_multicat) {
    if (do_force_eval) {
      AzPmat m_y(ms_y);
      return _test_eval(data_size, m_p, &m_y, m_dw, out_loss, out, s_pf);
    }
    if (s_pf != NULL) s_pf->reset(AzpEvalNoSupport);  
    if (out_loss != NULL) *out_loss = _get_loss(m_p, &m_y, NULL) / (double)data_size; 
    return -1;      
  }
  /*---  single-label classification  ---*/
  else {
    AzX::throw_if((ms_y == NULL || ms_y->size() <= 0), eyec, "Expected sparse Y for classification"); 
    int correct = AzpEv::eval_classif(m_p, ms_y); 
    double acc = (double)correct / (double)data_size; 
    if (out_loss != NULL) *out_loss = _get_loss(m_p, &m_y, NULL) / (double)data_size; 
    if (s_pf != NULL) s_pf->reset("err");      
    return 1-acc;    
  }
}

/*------------------------------------------------------------*/ 
void AzpLossDflt::test_eval2(const AzpData_ *data, int dx, int d_num, 
                             const AzPmat *m_p, 
                             double &perf_val, int &num, double *out_loss, /* added */
                             AzBytArr *s_pf) const 
{
  const char *eyec = "AzpLossDflt::test_eval2"; 
  if (data->is_sparse_y()) {
    AzSmat ms_y; data->gen_targets(dx, d_num, &ms_y); 
    _test_eval2(m_p, &ms_y, perf_val, num, out_loss, s_pf);     
  }
  else {
    AzPmat m_y; data->gen_targets(dx, d_num, &m_y); 
    _test_eval2(m_p, &m_y, perf_val, num, out_loss, s_pf); 
  }
}

/*------------------------------------------------------------*/ 
template <class M> /* M : AzSmat | AzPmat */
void AzpLossDflt::_test_eval2(const AzPmat *m_p, const M *m_y, 
                              double &perf_val, int &num, double *out_loss, /* added */
                              AzBytArr *s_pf) const {
  m_p->shape_chk(m_y->rowNum(), m_y->colNum(), "AzpLossDflt::_test_eval2", "targets and predictions");     
  double loss = -1;   
  if (out_loss != NULL) { loss = _get_loss(m_p, m_y, NULL); *out_loss += loss; }
  if (is_regression) { /* regression */
    loss = (out_loss != NULL) ? loss : _get_loss(m_p, m_y, NULL); 
    if (s_pf != NULL) s_pf->reset("loss");      
    perf_val += loss; 
  }
  else if (is_multicat) { /* multi-label classification */
    if (s_pf != NULL) s_pf->reset(AzpEvalNoSupport);  
    perf_val = -1; 
  }
  else { /* single-label classification */
    int correct = (do_largecat_eval) ? AzpEv::eval_classif_largeCat(m_p, m_y) : AzpEv::eval_classif(m_p, m_y); 
    if (s_pf != NULL) s_pf->reset("acc");      
    perf_val += (double)correct; 
  }
  num += m_p->colNum(); 
}
template void AzpLossDflt::_test_eval2<AzSmat>(const AzPmat *, const AzSmat *, double &, int &, double *, AzBytArr *) const; 
template void AzpLossDflt::_test_eval2<AzPmat>(const AzPmat *, const AzPmat *, double &, int &, double *, AzBytArr *) const; 

/*------------------------------------------------------------*/ 
/*------------------------------------------------------------*/ 
#define kw_loss_type "loss="
#define kw_loss_override "loss_override="

/* this must be in sync with AzpCNet3 */
#define kw_zerotarget_ratio "zero_Y_ratio=" 

#define kw_zcoeff "zero_Y_weight="
#define kw_do_dw "UseDataWeights"
#define kw_th_opt "th_opt_type="
#define kw_do_force_eval "ForceEval"
#define kw_do_firstcat_only "FirstOnly"
#define kw_do_largecat_eval "LargeCatEval"

/*------------------------------------------------------------*/ 
void AzpLossDflt::resetParam(const AzOut &out, AzParam &azp, bool is_warmstart)
{
  const char *eyec = "AzpLossDflt::resetParam"; 

  if (!is_warmstart) {
    AzBytArr s(to_lossstr(AzpLossDflt_Log)); 
    azp.vStr(kw_loss_type, &s); 
    loss_type = to_losstype(s); 
    AzX::throw_if((loss_type == AzpLossDflt_Invalid), AzInputError, eyec, "Invalid loss type"); 
  }
  else {
    /*---  override loss type if required  ---*/
    AzBytArr s_lo; 
    azp.vStr(kw_loss_override, &s_lo); 
    if (s_lo.length() > 0) {
      AzpLossDfltType org_loss = loss_type; 
      loss_type = to_losstype(s_lo); 
      AzX::throw_if((loss_type == AzpLossDflt_Invalid), AzInputError, eyec, kw_loss_override, "Invalid loss type"); 
      AzBytArr s; s << "Changing loss from " << to_lossstr(org_loss) << " to " << to_lossstr(loss_type); 
      AzPrint::writeln(out, s); 
    }
  }

  int zerotarget_ratio = -1; 
  azp.vInt(kw_zerotarget_ratio, &zerotarget_ratio); 
  do_art_nega = (zerotarget_ratio > 0); 
  if (do_art_nega) {
    zcoeff = 0; 
    azp.vFloat(kw_zcoeff, &ncoeff);   
  }
  else {
    azp.vFloat(kw_zcoeff, &zcoeff);  
  }
  
  if (zcoeff < 0 && ncoeff < 0); /* no mask is needed */
  else if (zcoeff == 1 && ncoeff == 1) zcoeff = ncoeff = -1; /* no mask is needed */
  else { /* mask is needed */
    if (zcoeff < 0) zcoeff = 1; 
    if (ncoeff < 0) ncoeff = 1; 
  }
  
  azp.swOn(&do_dw, kw_do_dw); 
  azp.swOn(&is_regression, kw_is_regression); /* kw defined in AzpData_ */
  azp.swOn(&is_multicat, kw_is_multicat);     /* kw defined in AzpData_ */  
  if (is_multicat) {
    azp.vStr(kw_th_opt, &s_th_opt);  
    azp.swOn(&do_firstcat_only, kw_do_firstcat_only); 
  }  
  azp.swOn(&do_force_eval, kw_do_force_eval); 
  azp.swOn(&do_largecat_eval, kw_do_largecat_eval); 
}

/*------------------------------------------------------------*/ 
#define kw_do_art_nega "UseArtificialNegativeEntries"
#define kw_art_zcoeff "  internal_zero_Y_weight="
#define kw_art_ncoeff "  internal_nega_Y_weight="
void AzpLossDflt::printParam(const AzOut &out) const
{
  if (out.isNull()) return; 
  AzPrint o(out); 
  o.ppBegin("AzpLossDflt", ""); 
  if (do_art_nega) {
    o.printV(kw_zcoeff, ncoeff); 
    o.printSw(kw_do_art_nega, do_art_nega);     
    o.printV(kw_art_zcoeff, zcoeff); 
    o.printV(kw_art_ncoeff, ncoeff); 
  }
  else             o.printV(kw_zcoeff, zcoeff); 
  o.printV(kw_loss_type, to_lossstr(loss_type)); 
  o.printSw(kw_do_dw, do_dw); 
  o.printSw(kw_is_regression, is_regression); 
  o.printSw(kw_is_multicat, is_multicat); 
  o.printV_if_not_empty(kw_th_opt, s_th_opt);  
  o.printSw(kw_do_force_eval, do_force_eval); 
  o.printSw(kw_do_firstcat_only, do_firstcat_only); 
  o.printSw(kw_do_largecat_eval, do_largecat_eval); 
  
  o.ppEnd(); 
}
void AzpLossDflt::printHelp(AzHelp &h) const {
  h.item_required(kw_loss_type, "Loss function.  \"Square\" ((p-y)^2/2) | \"Log\" (softmax and log loss) | \"BinLogi2\" (binary logistic loss) log(1+exp(-(2y-1)p)) for y in {0,1}"); 
  h.item(kw_is_regression, "Specify this if the task is regression."); 
  h.item(kw_is_multicat, "Specify this if each data point can be assigned multiple labels."); 
}

/*------------------------------------------------------------*/  
void AzpLossDflt::check_target(const AzOut &out, const AzpData_ *data) const 
{
  const char *eyec = "AzpLossDflt::check_target"; 
  double min_tar = data->min_target(), max_tar = data->max_target(); 
  if (loss_type == AzpLossDflt_Log || loss_type == AzpLossDflt_CE || loss_type == AzpLossDflt_BinLogi2) {
    if (min_tar < 0 || max_tar > 1) {
      AzBytArr s("With "); s << loss_dflt_str[loss_type] << ", targets should be in [0,1].  "; 
      s << "min_target=" << min_tar << " max_target=" << max_tar; 
      AzX::throw_if(true, AzInputError, eyec, s.c_str()); 
    }
  }
  else if (loss_type == AzpLossDflt_BinLogi) {
    if (min_tar < -1 || max_tar > 1) {
      AzBytArr s("With "); s << loss_dflt_str[loss_type] << ", targets should be in [-1,1].  "; 
      s << "min_target=" << min_tar << " max_target=" << max_tar; 
      AzX::throw_if(true, AzInputError, eyec, s.c_str()); 
    } 
    if (min_tar != -1 || max_tar != 1) {
      AzBytArr s("!WARNING!: With "); s << loss_dflt_str[loss_type] << ", targets in {-1,-1} are expected.  "; 
      s << "min_target=" << min_tar << " max_target=" << max_tar;         
      AzPrint::writeln(out, s); 
    }
  }
}  

/*------------------------------------------------------------*/  
void AzpLossDflt::check_losstype(const AzOut &out, int class_num) const 
{
  const char *eyec = "AzpLossDflt::check_losstype"; 
  AzX::throw_if((class_num <= 0), eyec, "#class is not set."); 
  if (is_regression) {
    AzX::throw_if(is_multicat, AzInputError, eyec, "regression and multi-cat classification are mutually exclusive"); 
  }
  if (class_num == 1) {
    AzX::throw_if(is_multiclass_log(loss_type), AzInputError, eyec, "There needs to be more than one class to use", 
                  to_lossstr(loss_type)); 
  }
  if (is_multicat) {
    AzX::throw_if((loss_type != AzpLossDflt_Square && loss_type != AzpLossDflt_BinLogi && loss_type != AzpLossDflt_BinLogi2), 
                  AzInputError, eyec, to_lossstr(loss_type), "cannot be used for multi-cat"); 
  }
} 

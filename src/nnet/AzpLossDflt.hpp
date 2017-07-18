/* * * * *
 *  AzpLossDflt.hpp
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

#ifndef _AZP_LOSS_DFLT_HPP_
#define _AZP_LOSS_DFLT_HPP_

#include "AzpLoss_.hpp"
#include "AzParam.hpp"
#include "AzHelp.hpp"

  enum AzpLossDfltType {
    AzpLossDflt_Log = 0, /* softmax + log loss */
    AzpLossDflt_Square = 1, /* (y-p)^2/2 */ 
    AzpLossDflt_CE = 2, /* cross entropy: -y log(p) - (1-y)log(1-p) */
    AzpLossDflt_BinLogi = 3, /* binary logistic: log(1+exp(-yp)) for y in {-1,1} */
    AzpLossDflt_BinLogi2 = 4, /* binary logistic: log(1+exp(-(2y-1)p)) for y in {0,1} */
    AzpLossDflt_Invalid = 5, 
  };
  #define AzpLossDfltType_Num 6
  static const char *loss_dflt_str[AzpLossDfltType_Num] = {
    "Log", "Square", "CE", "BinLogi", "BinLogi2", "???", 
  };
  
/*------------------------------------------------------------*/
class AzpLossDflt : public virtual /* implements */ AzpLoss_ {
protected: 
  bool is_regression, is_multicat, do_force_eval; 
  bool do_firstcat_only; 
  bool do_art_nega; 
  bool do_largecat_eval; 
  
  AzpLossDfltType loss_type; 
  double zcoeff, ncoeff; 
  bool needs_y_mask() const {
    return (zcoeff >= 0 || ncoeff >= 0); 
  }

  bool do_dw; 
  AzBytArr s_th_opt; /* threshold optimization type */    

  static const int version = 0; 
  static const int reserved_len = 64;  
public:
  AzpLossDflt() 
    : is_regression(false), is_multicat(false), do_force_eval(false), 
      do_firstcat_only(false), 
      loss_type(AzpLossDflt_Invalid), zcoeff(-1), ncoeff(-1), 
      do_dw(false), do_art_nega(false), do_largecat_eval(false) {}
  AzpLoss_ *clone() const {
    AzpLossDflt *o = new AzpLossDflt(); 
    *o = *this; 
    return o; 
  }
         
  virtual void check_target(const AzOut &out, const AzpData_ *data) const;        
  virtual void check_losstype(const AzOut &out, int class_num) const; 
  
  virtual void resetParam(const AzOut &out, AzParam &azp, bool is_warmstart=false);
  virtual void printParam(const AzOut &out) const; 
  virtual void printHelp(AzHelp &h) const; 
  virtual void write(AzFile *file) const {
    AzTools::write_header(file, version, reserved_len); 
    file->writeByte(loss_to_byte(loss_type)); 
  }
  virtual void read(AzFile *file) {
    AzTools::read_header(file, reserved_len);
    AzByte val = file->readByte(); 
    loss_type = byte_to_loss(val); 
  }
  
  /*---  ---*/
  virtual void get_loss_deriv(const AzPmat *m_p, /* output of the top layer, col: data points */
                      const AzpData_ *data, 
                      const int *dxs, int d_num, 
                      /*---  output  ---*/
                      AzPmat *m_loss_deriv,  /* row: nodes of the output layer, col: data points */
                      double *loss, /* may be NULL: added */
                      const AzPmatSpa *_ms_y=NULL, /* may be NULL: used for semi-sup */                  
                      const AzPmat *_md_y=NULL) const; /* may be NULL: used for semi-sup */

  virtual double get_loss(const AzpData_ *data, int dx_begin, int d_num, const AzPmat *m_p) const;              
  virtual double test_eval(const AzpData_ *data, 
                         AzPmat *m_p, /* inout */
                         double *out_loss, 
                         const AzOut &out, AzBytArr *s_pf=NULL) const; 
  virtual void test_eval2(const AzpData_ *data, int dx, int d_num, const AzPmat *m_p, 
                          double &perf_val, int &num, double *out_loss, AzBytArr *s_pf) const; /* for token-level application */                                 
  void get_loss_deriv(const AzPmat *m_p, const AzPmat *m_y, 
                      AzPmat *m_loss_deriv, 
                      double *loss) const {/* may be NULL: added */  
    _get_loss_deriv(m_p, m_y, NULL, m_loss_deriv, loss); 
  }  
  double get_loss(const AzPmat *m_p, const AzPmat *m_y) const {
    return _get_loss(m_p, m_y, NULL); 
  }
  virtual bool needs_eval_at_once() const { return (is_multicat && do_force_eval); }
protected:   
  template <class M> /* M: AzPmat | AzPmatSpa */
  void _get_loss_deriv(const AzPmat *m_p, const M *m_y,
                       const AzPmat *m_dw, AzPmat *m_loss_deriv, 
                       double *loss) const; /* may be NULL: added */ 
  template <class M> /* M: AzPmat | AzPmatSpa */
  double _get_loss(const AzPmat *m_p, const M *m_y, const AzPmat *m_dw) const;                     
  double _get_loss(const AzPmat *m_p, const AzSmat *ms_y, const AzPmat *m_dw) const {
    AzPmatSpa m_y; m_y.set(ms_y, false); /* no row index */
    return _get_loss(m_p, &m_y, m_dw);     
  }
  
  void to_prob2(AzPmat *m_prob) const; 

  /*---  loss type  ---*/
  AzpLossDfltType to_losstype(const AzBytArr &s) const {
    for (int ix = 0; ix < AzpLossDfltType_Num; ++ix) {
      if (s.equals(loss_dflt_str[ix])) return (AzpLossDfltType)ix; 
    }
    return AzpLossDflt_Invalid; 
  }
  const char *to_lossstr(AzpLossDfltType losstype) const {
    int intloss = (int)losstype; 
    if (intloss < 0 || intloss >= AzpLossDfltType_Num) return "???"; 
    return loss_dflt_str[intloss]; 
  }  

  template <class T, class U>
  const AzPmat *gen_dw(const AzpData_ *data, T param1, U param2, AzPmat *_m) const {
    if (!do_dw) return NULL; 
    data->gen_dataweights(param1, param2, _m); 
    return _m; 
  }
  
  template <class M> /* M should be AzPmat or AzPmatSpa */
  const AzPmat *gen_y_mask(const M *m_y, AzPmat *m_y_mask) const {  
    if (!needs_y_mask()) return NULL; 
    AzPs::set(m_y_mask, m_y); m_y_mask->mark_positive(); /* (y>0) ? 1 : 0 */
    if (zcoeff > 0) {
      AzPsAzPmat(m_y_mask0,m_y); m_y_mask0.mark_eq(0); /* (y==0) ? 1 : 0 */
      m_y_mask->add(&m_y_mask0, zcoeff);
    }
    if (ncoeff > 0) {
      AzPsAzPmat(m_y_mask_,m_y); m_y_mask_.mark_negative(); /* (y<0) ? 1 : 0 */
      m_y_mask->add(&m_y_mask_, ncoeff);      
    }         
    return m_y_mask; 
  }
 
  bool is_multiclass_log(AzpLossDfltType inp) const {
    return (inp == AzpLossDflt_Log); 
  }

  double _test_eval(int data_size, AzPmat *m_p, const AzPmat *m_y, const AzPmat *m_dw, 
                    double *out_loss, const AzOut &out, AzBytArr *s_pf) const;
  double _test_eval(int data_size, AzPmat *m_p, const AzSmat *ms_y, const AzPmat *m_dw, 
                    double *out_loss, const AzOut &out, AzBytArr *s_pf) const;
  template <class M> /* for token-level application */
  void _test_eval2(const AzPmat *m_p, const M *m_y, double &perf_val, int &num, double *out_loss, AzBytArr *s_pf) const; 
  
  /*---  to save loss in 1 byte in a model file  ---*/
  static AzByte loss_to_byte(AzpLossDfltType inp) { return (AzByte)inp; }
  static AzpLossDfltType byte_to_loss(AzByte loss_byte) { 
    AzX::throw_if((loss_byte >= AzpLossDfltType_Num), AzInputError, "AzpLossDflt::byte_to_loss", "Invalid loss type"); 
    return (AzpLossDfltType)loss_byte; 
  }                             
}; 
#endif

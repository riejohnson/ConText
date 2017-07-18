/* * * * *
 *  AzpLoss_.hpp
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

#ifndef _AZP_LOSS__HPP_
#define _AZP_LOSS__HPP_

#include "AzParam.hpp"
#include "AzHelp.hpp"
#include "AzPmat.hpp"
#include "AzpData_.hpp"

/*------------------------------------------------------------*/
class AzpLoss_ {
public:
  virtual ~AzpLoss_() {}
  virtual AzpLoss_ *clone() const = 0; 
  virtual void resetParam(const AzOut &out, AzParam &azp, bool is_warmstart=false) = 0; 
  virtual void printParam(const AzOut &out) const = 0; 
  virtual void printHelp(AzHelp &h) const = 0; 
  virtual void check_losstype(const AzOut &out, int class_num) const = 0; 
  virtual void check_target(const AzOut &out, const AzpData_ *data) const = 0; 
  
  /*---  ---*/
  virtual void get_loss_deriv(const AzPmat *m_p, /* output of the top layer, col: data points */
                      const AzpData_ *data, 
                      const int *dxs, int d_num, 
                      /*---  output  ---*/
                      AzPmat *m_loss_deriv,  /* row: nodes of the output layer, col: data points */
                      double *loss, /* may be NULL: added */
                      const AzPmatSpa *_ms_y=NULL,
                      const AzPmat *_md_y=NULL) const = 0; 

  virtual double get_loss(const AzpData_ *data, int dx_begin, int d_num, const AzPmat *m_p) const = 0; 

  virtual bool needs_eval_at_once() const = 0; 
  #define AzpEvalNoSupport "NO-SUPPORT"  
  virtual double test_eval(const AzpData_ *data, 
                         AzPmat *m_p, /* inout */
                         double *out_loss, /* output */
                         const AzOut &out, 
                         AzBytArr *s_pf=NULL /* description of the returned performance (e.g., "err") */
                         ) const = 0; 
  virtual void test_eval2(const AzpData_ *data, int dx, int d_num, const AzPmat *m_p, /* input */
                          double &perf_sum, int &num, double *out_loss, AzBytArr *s_pf=NULL) /* added */ 
                          const = 0; 
                         
  virtual void read(AzFile *file) = 0; 
  virtual void write(AzFile *file) const = 0;  
  
  /*---  for auto encoder  ---*/
  virtual void get_loss_deriv(const AzPmat *m_p, const AzPmat *m_y,
                              AzPmat *m_loss_deriv,
                              double *loss) const = 0; /* may be NULL: added */  
  virtual double get_loss(const AzPmat *m_p, const AzPmat *m_y) const = 0; 
}; 
#endif

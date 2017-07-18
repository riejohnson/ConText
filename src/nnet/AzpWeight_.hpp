/* * * * *
 *  AzpWeight_.hpp
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

#ifndef _AZP_WEIGHT__HPP_
#define _AZP_WEIGHT__HPP_

#include "AzpCompo_.hpp"
#include "AzPmat.hpp"
#include "AzpLm.hpp"

/**********************************************************/  
class AzpWeight_ : public virtual AzpCompo_ {
public: 
  virtual AzpWeight_ *clone() const = 0; 
  virtual ~AzpWeight_() {}
  virtual void destroy() = 0; 
  
  virtual bool are_weights_fixed() const = 0; 
  
  /*---  do something with parameters ... ---*/
  virtual void force_no_intercept() = 0; 
  virtual void reset_do_no_intercept(bool flag) = 0; 
  virtual void force_no_reg() = 0;  /* force no regularization */
  virtual void multiply_to_stepsize(double factor, const AzOut *out=NULL) = 0; 
  virtual void set_momentum(double newmom, const AzOut *out=NULL) = 0; 

  virtual void force_thru(int inp_dim) = 0; 
  
  /*---  initialization  ---*/
  virtual void reset(int loc_num, int w_num, int inp_dim, bool is_spa, bool is_var) = 0; 
  virtual void setup_for_reg_L2init() = 0; 
  virtual void check_for_reg_L2init() const = 0;   
  virtual void initWeights() = 0; 
  virtual void initWeights(const AzpLm *inp, double coeff) = 0; 

  /*---  up and down ...  ---*/
  virtual void upward(bool is_test, const AzPmat *m_x, AzPmat *m_out) = 0; 
  virtual void upward(bool is_test, const AzPmatSpa *m_x, AzPmat *m_out) = 0; 
  virtual void downward(const AzPmat *m_lossd, AzPmat *m_d) const = 0; 
  virtual void updateDelta(int d_num, const AzPmat *m_x, const AzPmat *m_lossd) = 0; 
  virtual void updateDelta(int d_num, const AzPmatSpa *m_x, const AzPmat *m_lossd) = 0; 
  virtual void flushDelta() = 0;   /* prev is used in SVRG */
  virtual void clearTemp() = 0; 
  virtual void end_of_epoch() = 0; 

  /*---  seeking information ...  ---*/
  virtual double regloss(double *iniloss) const = 0; 
  virtual int get_dim() const = 0; 
  virtual int classNum() const = 0; 
  virtual const AzpLm *linmod() const = 0; 
  virtual AzpLm *linmod_u() = 0;   

  virtual void reset_monitor() {}
  virtual int num_weights() const = 0; 
  virtual void show_stat(AzBytArr &s) const {}
}; 

#endif 
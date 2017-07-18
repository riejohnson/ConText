/* * * * *
 *  AzpPooling_var_.hpp
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

#ifndef _AZP_POOLING_VAR__HPP_
#define _AZP_POOLING_VAR__HPP_

#include "AzpCompo_.hpp"
#include "AzPmat.hpp"
#include "AzPmatApp.hpp"   

/**************************************************************/
/* from variable size to fixed size */
class AzpPooling_var_ : public virtual AzpCompo_ {
public: 
  virtual ~AzpPooling_var_() {} 
  virtual AzpPooling_var_ *clone() const = 0;   
  virtual void reset(const AzOut &out) = 0; 
  virtual void output(AzxD *out) const = 0; 
  virtual void upward(bool is_test, const AzPmatVar *m_inp, AzPmatVar *m_out) = 0; 
  virtual void downward(const AzPmatVar *m_lossd_after, AzPmatVar *m_lossd_before) const = 0; 
  virtual const AzIntArr *out_ind() const = 0; 
  virtual void get_chosen(AzIntArr *ia_chosen) const = 0; /* for analysis */  
  virtual bool do_asis() const = 0; 
  
  /*---  just for convenience ...  ---*/
  virtual void upward(bool is_test, AzPmatVar *mv_inout) {
    if (do_asis()) return; 
    AzPmatVar mv_out; upward(is_test, mv_inout, &mv_out); 
    mv_inout->set(&mv_out); 
  }
  virtual void downward(AzPmatVar *m_inout) const {
    if (do_asis()) return; 
    AzPmatVar m_inp(m_inout); downward(&m_inp, m_inout); 
  }  
};
#endif 
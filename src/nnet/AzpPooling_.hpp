/* * * * *
 *  AzpPooling_.hpp
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

#ifndef _AZP_POOLING__HPP_
#define _AZP_POOLING__HPP_

#include "AzpCompo_.hpp"
#include "AzParam.hpp"
#include "AzPmat.hpp"
#include "AzxD.hpp"

/*------------------------------------------------------------*/
class AzpPooling_ : public virtual AzpCompo_ { 
public:
  virtual ~AzpPooling_() {}  
  virtual AzpPooling_ *clone() const = 0;  
  virtual void reset(const AzOut &out, const AzxD *input, AzxD *output) = 0; 
  virtual int input_size() const = 0; 
  virtual int output_size() const = 0; 
  virtual void output(AzxD *output) = 0; 
  virtual void upward(bool is_test, const AzPmat *m_inp, AzPmat *m_out) = 0; 
  virtual void downward(const AzPmat *m_lossd_after, AzPmat *m_lossd_before) const = 0;  
  virtual void show(const AzOut &out) const {}; 
  virtual void get_chosen(AzIntArr *ia_chosen) const = 0; /* for analysis */
  virtual bool do_asis() const = 0;   
  
  /*---  just for convenience ...  ---*/  
  virtual void upward(bool is_test, AzPmat *m_inout) {
    if (do_asis()) return; 
    AzPmat m_out; upward(is_test, m_inout, &m_out); 
    m_inout->set(&m_out); 
  }
  virtual void downward(AzPmat *m_inout) const {
    if (do_asis()) return; 
    AzPmat m_inp(m_inout); downward(&m_inp, m_inout); 
  }  
}; 
#endif 


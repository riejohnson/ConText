/* * * * *
 *  AzpPatch_var_.hpp
 *  Copyright (C) 2014-2015 Rie Johnson
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

#ifndef _AZP_PATCH_VAR__HPP_
#define _AZP_PATCH_VAR__HPP_

#include "AzxD.hpp"
#include "AzPmat.hpp"
#include "AzpCompo_.hpp"

/* generate patches/filters for convolutional NNs */
class AzpPatch_var_ : public virtual AzpCompo_ {
public:  
  virtual ~AzpPatch_var_() {}
  virtual AzpPatch_var_ *clone() const = 0; 
  virtual void reset(const AzOut &out, int channels, bool is_spa) = 0; 
  virtual int patch_length() const = 0;  /* size times channels */

  virtual void upward(bool is_test, 
                      const AzPmatVar *m_bef, /* each column represents a pixel; more than one data point */
                      AzPmatVar *m_aft) = 0; /* each column represents a patch */
  virtual void downward(const AzPmatVar *m_aft, AzPmatVar *m_bef) = 0;                 
};      
#endif 
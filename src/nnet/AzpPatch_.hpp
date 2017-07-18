/* * * * *
 *  AzpPatch_.hpp
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

#ifndef _AZP_PATCH__HPP_
#define _AZP_PATCH__HPP_

#include "AzxD.hpp"
#include "AzPmat.hpp"
#include "AzpCompo_.hpp"

/* generate patches/filters for convolutional NNs */
class AzpPatch_ : public virtual AzpCompo_ {
public:  
  virtual ~AzpPatch_() {}
  virtual AzpPatch_ *clone() const = 0; 
  virtual bool is_convolution() const = 0; 
  virtual void reset(const AzOut &out, const AzxD *input, int channels, bool is_spa, bool is_var) = 0; 
  virtual void setup_allinone(const AzOut &out, const AzxD *input, int channels) = 0; 
  virtual void setup_asis(const AzOut &out, const AzxD *input, int channels) = 0;   
  virtual int get_channels() const = 0; 
  virtual bool isSameInput(const AzxD *inp, int channels) {
    return (inp != NULL && inp->isSame(input_region()) && get_channels() && channels); 
  }
  virtual const AzxD *input_region(AzxD *o=NULL) const = 0; 
  virtual const AzxD *output_region(AzxD *o=NULL) const = 0; 
  virtual int patch_length() const = 0;  /* size times channels */

  virtual void show_input(const char *header, const AzOut &out) const {}
  virtual void show_output(const char *header, const AzOut &out) const {}

  virtual void upward(bool is_test, 
                      const AzPmat *m_bef, /* each column represents a pixel; more than one data point */
                      AzPmat *m_aft) const = 0; /* each column represents a patch */
  virtual void downward(const AzPmat *m_aft, AzPmat *m_bef) const = 0;                 
};      
#endif 
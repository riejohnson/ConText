/* * * * *
 *  AzpDropout_.hpp
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

#ifndef _AZP_DROPOUT__HPP_
#define _AZP_DROPOUT__HPP_

#include "AzpCompo_.hpp"
#include "AzPmat.hpp"

/*------------------------------------------------------------*/
class AzpDropout_ : public virtual AzpCompo_ {
public:
  virtual ~AzpDropout_() {}
  virtual bool is_active() const = 0; 
  virtual void reset(const AzOut &out) = 0;  
  virtual AzpDropout_ *clone() const = 0; 
  virtual void upward(bool is_test, AzPmat *m) = 0; 
  virtual void downward(AzPmat *m) = 0; 
  virtual void show_stat(AzBytArr &s) {}
  virtual void read(AzFile *file) = 0; 
  virtual void write(AzFile *file) const = 0; 
  virtual AzPrng &ref_rng() = 0;   
}; 
#endif 
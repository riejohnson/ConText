/* * * * *
 *  AzpResNorm_.hpp
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

#ifndef _AZP_RESNORM__HPP_
#define _AZP_RESNORM__HPP_

#include "AzUtil.hpp"
#include "AzPmat.hpp"
#include "AzParam.hpp"
#include "AzpCompo_.hpp"

/**********************************************************************************/
class AzpResNorm_ : public virtual AzpCompo_ {
public:
  virtual ~AzpResNorm_() {}
  virtual void reset(const AzOut &out, const AzxD *input, int cc) = 0; 
  virtual AzpResNorm_ *clone() const = 0; 
  virtual void upward(bool is_test, AzPmat *m) = 0; 
  virtual void downward(AzPmat *m) = 0;  
}; 
#endif 
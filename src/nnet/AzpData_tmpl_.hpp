/* * * * *
 *  AzpData_tmpl_.hpp
 *  Copyright (C) 2017 Rie Johnson
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

#ifndef _AZP_DATA_TMPL_HPP_
#define _AZP_DATA_TMPL_HPP_

#include "AzUtil.hpp"
#include "AzDic.hpp"
                      
class AzpData_tmpl_ {
public: 
  virtual int datasetNum() const = 0; 
  virtual int xdim(int dsno=0) const  = 0; 
  virtual const AzDicc &get_dic(int dsno=0) const = 0;  
  virtual bool is_vg_x() const = 0; 
  virtual bool is_sparse_x() const = 0;  
  virtual int ydim() const = 0; 
  virtual int dimensionality() const = 0; 
  virtual int size(int index) const = 0; 
  virtual void signature(AzBytArr &s) const {
    s.reset(); 
    s << "dim:" << dimensionality() << ";"; 
    s << "channel:" << xdim() << ";"; 
    for (int ix = 0; ix < dimensionality(); ++ix) {
      s << "size" << ix << ":" << size(ix) << ";"; 
    }
  }
  virtual bool isSignatureCompatible(const AzBytArr &s_nn, const AzBytArr &s_data) const {
    if (s_nn.compare(&s_data) == 0) return true; 
    AzStrPool sp1(10,10), sp2(10,10); 
    AzTools::getStrings(s_nn.c_str(), ';', &sp1); 
    AzTools::getStrings(s_data.c_str(), ';', &sp2); 
    if (sp1.size() != sp2.size()) {
      return false; 
    }
    for (int ix = 0; ix < sp1.size(); ++ix) {
      if (strcmp(sp1.c_str(ix), sp2.c_str(ix)) != 0) {
        AzBytArr s1; sp1.get(ix, &s1); 
        AzBytArr s2; sp2.get(ix, &s2); 
        if (ix == 0 && s1.length() != s2.length()) {
          /*---  sparse_multi with one dataset is equivalent to sparse  ---*/
          if (s1.beginsWith("[0]") && s2.compare(s1.point()+3, s1.length()-3) == 0) continue; 
          if (s2.beginsWith("[0]") && s1.compare(s2.point()+3, s2.length()-3) == 0) continue; 
        }
        else if (s1.beginsWith("size") && s2.beginsWith("size") && s1.endsWith("-1")) {
          /*---  nn is trained for variable-size input, and data is fixed-size ---*/
          continue; 
        }
        else return false; 
      }
    }
    return true; 
  }  
  virtual void get_info(AzxD &data_dim) const {
    AzIntArr ia_sz; 
    int dim = dimensionality(); 
    for (int ix = 0; ix < dim; ++ix) ia_sz.put(size(ix)); 
    data_dim.reset(&ia_sz); 
  }  
};  
#endif 
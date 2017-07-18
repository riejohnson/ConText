/* * * * *
 *  AzpDataSetDflt.hpp
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

#ifndef _AZP_DATASET_DFLT_HPP_
#define _AZP_DATASET_DFLT_HPP_
#include "AzpData_.hpp"
#include "AzpData_img.hpp"
#include "AzpData_imgbin.hpp"
#include "AzpData_sparse.hpp"
#include "AzpData_sparse_multi.hpp"

class AzpDataSetDflt : public virtual AzpDataSet_ {
protected: 
  AzpData_img img; 
  AzpData_imgbin imgbin; 
  AzpData_sparse<AzSmatc,AzSmatc> sparse_no_no; 
  AzpData_sparse<AzSmatc,AzSmatbc> sparse_no_bc; 
  AzpData_sparse<AzSmatbc,AzSmatc> sparse_bc_no; 
  AzpData_sparse<AzSmatbc,AzSmatbc> sparse_bc_bc; 
  
  AzpData_sparse_multi<AzSmatc,AzSmatc> sparse_multi_no_no; 
  AzpData_sparse_multi<AzSmatc,AzSmatbc> sparse_multi_no_bc; 
  AzpData_sparse_multi<AzSmatbc,AzSmatc> sparse_multi_bc_no; 
  AzpData_sparse_multi<AzSmatbc,AzSmatbc> sparse_multi_bc_bc;  
public:   
  virtual void printHelp(AzHelp &h, bool do_train, bool do_test, bool is_there_y) const {
    AzpDataSet_::printHelp(h, do_train, do_test, is_there_y); 
    h.item_required(kw_datatype, "Dataset type.  \"sparse\" | \"img\" | \"imgbin\". "); 
    /* img.printHelp(h, do_train, is_there_y); */
    sparse_no_no.printHelp(h, is_there_y); 
    /* sparse_multi.printHelp(h, do_train, is_there_y); */
  }  
protected:   
  #define kw_dataext "dsno"
  #define kw_dataext_old "data_ext"  
  virtual const AzpData_ *ptr(AzParam &azp, const AzBytArr &s_typ, const AzBytArr &s_x_ext, const AzBytArr &s_y_ext) const { 
    AzBytArr s_kw(kw_dataext, "0"), s_kw_old(kw_dataext_old, "0"), s_dataext; 
    azp.vStr(s_kw.c_str(), &s_dataext, s_kw_old.c_str()); 
    if      (s_typ.compare("image") == 0)    return &img; 
    else if (s_typ.compare("imagebin") == 0) return &imgbin;    
    else if (s_typ.compare("sparse_multi") == 0 || s_dataext.length() > 0) {
      if (s_x_ext.contains("bc")) {
        if (s_y_ext.contains("bc")) return &sparse_multi_bc_bc; 
        else                        return &sparse_multi_bc_no; 
      }       
      else {
        if (s_y_ext.contains("bc")) return &sparse_multi_no_bc; 
        else                        return &sparse_multi_no_no;         
      }
    }     
    else if (s_typ.compare("sparse") == 0) {
      if (s_x_ext.contains("bc")) {
        if (s_y_ext.contains("bc")) return &sparse_bc_bc; 
        else                        return &sparse_bc_no; 
      }       
      else {
        if (s_y_ext.contains("bc")) return &sparse_no_bc; 
        else                        return &sparse_no_no;         
      }
    }     
    else {
      AzX::throw_if(true, AzInputError, "AzpDataSetDflt::ptr", "Unknown data type: ", s_typ.c_str()); return NULL; 
    }
  }
}; 
#endif
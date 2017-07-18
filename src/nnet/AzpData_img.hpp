/* * * * *
 *  AzpData_img.hpp
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

#ifndef _AZP_DATA_IMG_HPP_
#define _AZP_DATA_IMG_HPP_

#include "AzPmat.hpp"
#include "AzpData_.hpp"

class AzpData_img : public virtual AzpData_ {
protected: 
  int sz1, sz2, channels; /* parameters */
  double data_scale; 
  double min_tar, max_tar; 
  
  AzPmat m_x; 
  AzSmatc ms_y; 
  int data_num, rnum, cnum, total_data_num; 
  int dummy_ydim, current_batch, released_batch; 
  
  AzStrPool sp_x_ext, sp_y_ext; 
  
  AzpData_ *clone_nocopy() const { 
    return new AzpData_img(); 
  }  
  
public:
  AzpData_img() : data_num(0), rnum(0), cnum(0), channels(-1), sz1(0), sz2(0), dummy_ydim(-1), 
                  current_batch(0), total_data_num(-1), data_scale(-1), 
                  min_tar(1e+10), max_tar(-1e+10), released_batch(-1) {
    sp_x_ext.put(AzpData_Ext_xpmat, AzpData_Ext_x); 
    sp_y_ext.put(AzpData_Ext_ysmat, AzpData_Ext_y); 
  }               
                  
  virtual bool is_vg_x() const { return false; }
  virtual bool is_sparse_x() const { return false; }  
  virtual bool is_sparse_y() const { return true; }
  
  virtual int ydim() const { return ms_y.rowNum(); }

  #define kw_channels "channels="
  #define kw_sz1 "size1="
  #define kw_sz2 "size2="  
  #define kw_data_scale "data_scale="
  /*------------------------------------------------*/
  virtual void resetParam_data(AzParam &azp) {
    const char *eyec = "AzpData_img::resetParam_data"; 
    azp.vInt(kw_channels, &channels); 
    azp.vInt(kw_sz1, &sz1); 
    azp.vInt(kw_sz2, &sz2); 
    azp.vFloat(kw_data_scale, &data_scale);
    
    
    AzXi::throw_if_nonpositive(channels, eyec, kw_channels); 
    AzXi::throw_if_nonpositive(sz1, eyec, kw_sz1); 
    AzXi::throw_if_nonpositive(sz2, eyec, kw_sz2);
    AzXi::check_input(s_x_ext, &sp_x_ext, eyec, kw_x_ext); 
    if (s_y_ext.length() > 0) AzXi::check_input(s_y_ext, &sp_y_ext, eyec, kw_y_ext);       
  }
  virtual void printParam_data(const AzOut &out, const char *pfx) const {
    if (out.isNull()) return; 
    AzPrint o(out, pfx); 
    o.printV(kw_channels, channels); 
    o.printV(kw_sz1, sz1); 
    o.printV(kw_sz2, sz2);     
    o.printV(kw_data_scale, data_scale); 
    o.printEnd(); 
  }    
  virtual void printHelp_data(AzHelp &h) const {
    h.item(kw_channels, "Use this with \"image\".  Number of channels of the input data."); 
    h.item(kw_sz1, "Use this with \"image\".  Number of horizontal pixels."); 
    h.item(kw_sz2, "Use this with \"image\".  Number of vertical pixels."); 
    h.item(kw_data_scale, "Use this with \"image\".  Multiply this value to the input data."); 
  }
  virtual void reset() {
    destroy(); 
  }
  virtual void destroy() {
    AzpData_::destroy(); 
    m_x.destroy(); 
    ms_y.destroy(); 
    data_num = 0; 
    current_batch = -1; 
  }
  virtual int dataNum() const {
    return data_num; 
  }
  virtual int colNum() const { return cnum; }
  virtual int batchNum() const { return batch_num; }
  virtual int dataNum_total() const { return total_data_num; } 
 
  virtual void _gen_data(const int *dxs, int d_num, AzPmat *m_data, int dsno) const {
    AzX::throw_if((dsno != 0), "AzpData_img::_gen_data(dxs,num)", "dsno<>0?!"); 
    m_data->set(&m_x, dxs, d_num); 
    m_data->change_dim(channels, m_data->size()/channels); 
  }

  virtual void gen_targets(const int *dxs, int d_num, AzPmat *m_out_y) const {
    AzX::throw_if(true, "AzpData_img::gen_targets(dense)(dxs,d_num..)", "target is sparse"); 
  }  
  virtual void gen_targets(const int *dxs, int d_num, AzSmat *m_out_y) const {
    ms_y.copy_to_smat(m_out_y, dxs, d_num); 
  }  
  virtual void gen_targets(const int *dxs, int d_num, AzPmatSpa *m_out_y) const {
    bool gen_rowindex = false;     
    m_out_y->set(&ms_y, dxs, d_num, gen_rowindex); 
  }    
  virtual void gen_membership(AzDataArr<AzIntArr> &aia_cl_dxs) const {
    _gen_membership(ms_y, aia_cl_dxs); 
  }  

  virtual int xdim(int data_no=0) const { 
    if (data_no == 0) return channels; 
    return 0; 
  }
  virtual int dimensionality() const { return 2; }
  virtual int size(int index) const {
    if (index == 0) return sz1; 
    if (index == 1) return sz2; 
    return -1; 
  }
  virtual double min_target() const { return min_tar; }
  virtual double max_target() const { return max_tar; }
      
  /*------------------------------------------*/  
  virtual void reset_data(const AzOut &_out, const char *nm, int dummy_ydim, 
                          AzpData_binfo *bi=NULL) {
    const char *eyec = "AzpData_img::reset_data"; 
    out = _out; 
    s_nm.reset(nm); 

    double min_val = 0, max_val = 0, abssum = 0, abssum_pop = 0; 
    if (bi != NULL) bi->reset(batch_num);     
    total_data_num = 0; 
    int x_row = -1, y_row = -1; 
    int bx; 
    for (bx = batch_num-1; bx >= 0; --bx) {
      AzBytArr s_batchnm; 
      _reset_data(bx); 
      if (bx == batch_num-1) {
        x_row = m_x.rowNum(); 
        y_row = ms_y.rowNum(); 
      }
      else {
        AzX::throw_if((m_x.rowNum() != x_row || ms_y.rowNum() != y_row), 
                      AzInputError, eyec, "Data dimensionality conflict between batches"); 
      }
      if (bi != NULL) bi->update(bx, data_num); 
      total_data_num += m_x.colNum(); 
      current_batch = bx; 
      AzTimeLog::print("#data = ", data_num, out); 
      abssum += m_x.absSum(); abssum_pop += m_x.size(); 
      if (bx == batch_num-1) {
        min_val = m_x.min(); 
        max_val = m_x.max();
        min_tar = ms_y.min(); 
        max_tar = ms_y.max(); 
      }
      else {
        min_val = MIN(min_val, m_x.min()); 
        max_val = MAX(max_val, m_x.max()); 
        min_tar = MIN(min_tar, ms_y.min()); 
        max_tar = MAX(max_tar, ms_y.max()); 
      }
    }  
    AzBytArr s; s << "#total_data=" << total_data_num << ",min,max=" << min_val << "," << max_val; 
    if (abssum_pop > 0) s << ",absavg=" << abssum/abssum_pop; 
    s << ", target-min,max=" << min_tar << "," << max_tar; 
    AzTimeLog::print(s.c_str(), out); 
  }

  /*------------------------------------------*/   
  virtual void first_batch() {
    released_batch = -1;     
    if (current_batch != 0) {
      current_batch = -1; 
      next_batch(); 
    }   
  }
  
  /*------------------------------------------*/    
  virtual void next_batch() {  
    if (batch_num == 1 && current_batch == 0) return; 
    if (released_batch >= 0) current_batch = released_batch; /* resume the sequence */    
    current_batch = (current_batch + 1) % batch_num;    
    _reset_data(current_batch); 
    released_batch = -1;     
  }  
  
  /*------------------------------------------*/   
  virtual void release_batch() {
    released_batch = current_batch; /* suspend the sequence */
    current_batch = -1; 
    m_x.reset(); ms_y.reset(); 
    /* data_num = 0; */ 
  }  
  
protected:   
  virtual void _reset_data(int batch_no) {  
    const char *eyec = "AzpData_img::_reset_data";
    AzTimeLog::print(s_nm.c_str(), " batch#", batch_no, out);    
    m_x.reset();
    ms_y.reset(); 

    /*---  X  ---*/    
    AzBytArr s_x_fn; gen_batch_fn(batch_no, s_x_ext.c_str(), &s_x_fn); 
    AzBytArr s_y_fn; gen_batch_fn(batch_no, s_y_ext.c_str(), &s_y_fn); 
    if (s_x_ext.equals(AzpData_Ext_xpmat)) {
      m_x.read(s_x_fn.c_str()); 
    }
    else if (s_x_ext.equals(AzpData_Ext_x)) {
      AzDmat md_x;
      AzTextMat::readMatrix(s_x_fn.c_str(), &md_x); 
      m_x.set(&md_x); 
    }
    data_num = m_x.colNum(); rnum = m_x.rowNum(); cnum = m_x.colNum(); 
    if (data_scale > 0) {
      AzPrint::writeln(out, " ... multiplying ", data_scale); 
      m_x.multiply(data_scale); 
    }
    
    /*---  Y  ---*/
    if (s_y_ext.equals(AzpData_Ext_ysmat)) {
      AzPrint::writeln(out, "Reading as smatc binary file: ", s_y_fn.c_str()); 
      readY_bin(s_y_fn.c_str(), dummy_ydim, data_num, &ms_y); 
    }
    else {
      AzPrint::writeln(out, "Reading as text file: ", s_y_fn.c_str()); 
      readY_text(s_y_fn.c_str(), dummy_ydim, data_num, &ms_y);
    }
    
    /*---  ---*/
    if (data_num == 0) return; 
    int f_num = sz1*sz2*channels; 
    AzX::throw_if((f_num != m_x.rowNum()), AzInputError, eyec, "feature dimensionality mismatch"); 
  }

  virtual void _gen_data(const int *, int, AzPmatSpa *, int) const { no_support("AzpData_img", "_gen_data(sparse)"); }
  virtual void _gen_data(const int *, int, AzPmatSpaVar *, int) const { no_support("AzpData_img", "_gen_data(sparse and variable-sized)"); }
  virtual void _gen_data(const int *, int, AzPmatVar *, int) const { no_support("AzpData_img", "_gen_data(dense variable-sized)"); }
}; 
#endif 

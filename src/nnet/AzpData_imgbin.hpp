/* * * * *
 *  AzpData_imgbin.hpp
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

#ifndef _AZP_DATA_IMG_BIN_HPP_
#define _AZP_DATA_IMG_BIN_HPP_

#include "AzpData_.hpp"
/*----------------------------------------------------------*/
class AzpImg_bin {
protected:   
  int cc, wid, hei, num, sz; 
  AzBaseArr<AzByte,AZint8> arr;    
public:  
  AzpImg_bin() : cc(3), wid(0), hei(0), num(0), sz(0) {}
  void reset() {
    arr.free(); 
    cc = wid = hei = num = sz = 0; 
  }  
  void destroy() { reset(); }
  int channelNum() const { return cc; }
  int width() const { return wid; }
  int height() const { return hei; }
  void read_header(AzFile *file) {
    cc = file->readInt(); 
    wid = file->readInt(); 
    hei = file->readInt(); 
    num = file->readInt(); 
    sz = wid*hei*cc; 
  }  
  void read(const char *fn) {
    AzFile file(fn); file.open("rb"); 
    read_header(&file); 
    AZint8 sz_total = (AZint8)sz * (AZint8)num; 
    arr.free_alloc(sz_total); 
    file.seekReadBytes(-1, sz_total, arr.point_u()); 
    file.close();
  }
  int dataNum() const { return num; }
  void get_data(int dx, AzBytArr &byt_arr) const {
    check_dx(dx, "AzpImg_bin::get_data(BytArr)");  
    byt_arr.reset(arr.point() + (AZint8)((AZint8)sz*(AZint8)dx), sz); 
  }
  void get_data(const int *dxs, int d_num, AzPmat &m) const { 
    AzDvect v(sz*d_num); 
    double *data = v.point_u(); 
    int mpos = 0; 
    for (int ix = 0; ix < d_num; ++ix) {
      int dx = dxs[ix]; 
      check_dx(dx, "AzpImg_bin::get_data(Pmat)");        
      AZint8 base = (AZint8)sz*(AZint8)dx;   
      for (int ix = 0; ix < sz; ++ix, ++mpos) data[mpos] = arr[(AZint8)(base+(AZint8)ix)];      
    }
    m.set(&v); 
    m.change_dim(cc, wid*hei*d_num); 
  }   
protected:   
  void check_dx(int dx, const char *msg) const {
    AzX::throw_if(dx<0 || dx>=num, msg, "out of range"); 
  }  
}; 

/*----------------------------------------------------------*/
class AzpData_imgbin : public virtual AzpData_ {
protected:
  AzpImg_bin bin_x;  
  AzSmatc ms_y; 
  int total_data_num, current_batch, released_batch, dummy_ydim; 
  double min_tar, max_tar; 
  bool do_01, do_pm1, do_subavg; 
  bool do_onecol; /* true: one column per data point, false: one column per pixel */
  
  AzpData_ *clone_nocopy() const { 
    return new AzpData_imgbin(); 
  }  
  
public:
  AzpData_imgbin() : current_batch(0), total_data_num(-1), dummy_ydim(-1), 
                     do_01(false), do_pm1(false), do_subavg(false), do_onecol(false), 
                     min_tar(1e+10), max_tar(-1e+10), released_batch(-1) {}
            
  virtual bool is_vg_x() const { return true; }
  virtual bool is_sparse_x() const { return false; }  
  virtual bool is_sparse_y() const { return true; }
  
  virtual int ydim() const { return ms_y.rowNum(); }

  /*------------------------------------------------*/
  #define kw_do_01 "Scale01"
  #define kw_do_pm1 "Scale-1+1"
  #define kw_do_subavg "SubAvg"
  virtual void resetParam_data(AzParam &azp) {
    const char *eyec = "AzpData_img::resetParam_data"; 
    AzStrPool sp_x_ext, sp_y_ext; 
    sp_x_ext.put(AzpData_Ext_xbin); sp_y_ext.put(AzpData_Ext_ysmat, AzpData_Ext_y);
    AzXi::check_input(s_x_ext, &sp_x_ext, eyec, kw_x_ext); 
    if (s_y_ext.length() > 0) AzXi::check_input(s_y_ext, &sp_y_ext, eyec, kw_y_ext);
    azp.swOn(&do_01, kw_do_01); 
    if (!do_01) azp.swOn(&do_pm1, kw_do_pm1); 
    azp.swOn(&do_subavg, kw_do_subavg); 
  }
  virtual void printParam_data(const AzOut &out, const char *pfx) const {
    AzPrint o(out, pfx); 
    o.printSw(kw_do_01, do_01); 
    o.printSw(kw_do_pm1, do_pm1); 
    o.printSw(kw_do_subavg, do_subavg); 
  }
  virtual void printHelp_data(AzHelp &h) const {}
  virtual void reset() { destroy(); }
  virtual void destroy() {
    AzpData_::destroy(); 
    bin_x.destroy(); 
    ms_y.destroy(); 
    current_batch = -1; 
  }
  virtual int dataNum() const { return bin_x.dataNum(); }
  virtual int colNum() const { return bin_x.dataNum() * bin_x.width() * bin_x.height(); }
  virtual int batchNum() const { return batch_num; }
  virtual int dataNum_total() const { return total_data_num; } 
 
  virtual void _gen_data(const int *dxs, int d_num, AzPmat *m_data, int dsno) const {
    AzX::throw_if((dsno != 0), "AzpData_imgbin::_gen_data(dxs,num)", "dsno<>0?!"); 
    bin_x.get_data(dxs, d_num, *m_data); 
    if      (do_01)    m_data->divide(256); 
    else if (do_pm1) { m_data->add(-128); m_data->divide(128); }    
    else if (do_subavg) {
      int r_num = m_data->rowNum(), c_num = m_data->colNum(); 
      AzPmat m(m_data); 
      m.change_dim(m.size()/d_num, d_num);  /* one image per one column */
      AzPmat v_avg; v_avg.colSum(&m); v_avg.divide(m.rowNum()); v_avg.change_dim(v_avg.size(), 1); /* average */
      AzPmat mt; mt.transpose_from(&m); /* one image per one row */
      mt.add_col(&v_avg, -1); /* subtract the average */
      m_data->transpose_from(&mt); /* one image per one column */
      m_data->change_dim(r_num, c_num); 
    }    
    if (do_onecol) m_data->change_dim(m_data->size()/d_num, d_num); 
  } 
  virtual void _gen_data(const int *dxs, int d_num, AzPmatVar *mv, int dsno) const { 
    AzPmat m; _gen_data(dxs, d_num, &m, dsno); 
    AzX::throw_if(m.colNum()%d_num != 0, "AzpData_imgbin::_gen_data(pmatvar)", "Something is wrong with #columns."); 
    int sz = m.colNum() / d_num; 
    AzIntArr ia_ind; for (int dx = 0; dx < d_num; ++dx) { ia_ind.put(sz*dx); ia_ind.put(sz*dx+sz); }
    mv->set(&m, &ia_ind); 
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
  virtual void gen_membership(AzDataArr<AzIntArr> &aia_cl_dxs) const { _gen_membership(ms_y, aia_cl_dxs); }  
  virtual int xdim(int data_no=0) const { 
    if (data_no == 0) return bin_x.channelNum(); 
    return 0; 
  }
  virtual int dimensionality() const { return 2; }
  virtual int size(int index) const {
    if (index == 0) return bin_x.width(); 
    if (index == 1) return bin_x.height(); 
    return -1; 
  }
  virtual double min_target() const { return min_tar; }
  virtual double max_target() const { return max_tar; }
      
  /*------------------------------------------*/  
  virtual void reset_data(const AzOut &_out, const char *nm, int _dummy_ydim, 
                          AzpData_binfo *bi=NULL) {
    const char *eyec = "AzpData_imgbin::reset_data"; 
    out = _out; 
    s_nm.reset(nm); 
    dummy_ydim = _dummy_ydim; 

    if (bi != NULL) bi->reset(batch_num);     
    total_data_num = 0; 
    int y_row = -1, cc = -1, width = -1, height = -1; 
    int bx; 
    for (bx = batch_num-1; bx >= 0; --bx) {
      AzBytArr s_batchnm; 
      _reset_data(bx); 
      if (bx == batch_num-1) {
        cc = bin_x.channelNum(); width = bin_x.width(); height = bin_x.height(); 
        y_row = ms_y.rowNum(); 
      }
      else
        AzX::throw_if(bin_x.channelNum() != cc || ms_y.rowNum() != y_row || 
                      bin_x.width() != width || bin_x.height() != height, 
                      AzInputError, eyec, "Data dimension conflict between batches"); 
      if (bi != NULL) bi->update(bx, bin_x.dataNum()); 
      total_data_num += bin_x.dataNum(); 
      current_batch = bx; 
      AzTimeLog::print("#data = ", bin_x.dataNum(), out);    
      if (bx == batch_num-1) { min_tar = ms_y.min();               max_tar = ms_y.max();               }
      else                   { min_tar = MIN(min_tar, ms_y.min()); max_tar = MAX(max_tar, ms_y.max()); }
    }     
    AzBytArr s; s << "#total_data=" << total_data_num; 
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
    bin_x.reset(); ms_y.reset(); 
    /* data_num = 0; */ 
  }  
  
protected:   
  virtual void _reset_data(int batch_no) {  
    const char *eyec = "AzpData_imgbin::_reset_data";
    AzTimeLog::print(s_nm.c_str(), " batch#", batch_no, out);    
    bin_x.reset(); 
    ms_y.reset(); 

    /*---  X  ---*/    
    AzBytArr s_x_fn; gen_batch_fn(batch_no, s_x_ext.c_str(), &s_x_fn); 
    bin_x.read(s_x_fn.c_str());  
    
    /*---  Y  ---*/
    AzBytArr s_y_fn; gen_batch_fn(batch_no, s_y_ext.c_str(), &s_y_fn);     
    if (s_y_ext.equals(AzpData_Ext_ysmat)) readY_bin(s_y_fn.c_str(), dummy_ydim, bin_x.dataNum(), &ms_y); 
    else                                   readY_text(s_y_fn.c_str(), dummy_ydim, bin_x.dataNum(), &ms_y);
  }
  virtual void _gen_data(const int *, int, AzPmatSpa *, int) const { no_support("AzpData_img", "_gen_data(sparse)"); }
  virtual void _gen_data(const int *, int, AzPmatSpaVar *, int) const { no_support("AzpData_img", "_gen_data(sparse and variable-sized)"); }
}; 
#endif 

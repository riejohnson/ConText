/* * * * *
 *  AzpData_sparse_multi.hpp
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

#ifndef _AZP_DATA_SPARSE_MULTI_HPP_
#define _AZP_DATA_SPARSE_MULTI_HPP_

#include "AzpData_sparse.hpp"

template <class Mat, class MatY>
class AzpData_sparse_multi : public virtual AzpData_ {
protected:
  AzDataArr< AzpData_sparse<Mat,MatY> > data; 
  AzStrPool sp_dataext; 
  
  virtual AzpData_ *clone_nocopy() const {
    return new AzpData_sparse_multi(); 
  }
  
public: 
  AzpData_sparse_multi() {}
  
  /*------------------------------------------------*/
  #define kw_dataext "dsno"
  #define kw_dataext_old "data_ext"
  #define kw_do_shape_check ":SameShape"
  /*------------------------------------------------*/
  const char *gen_dataext_kw(int extno, AzBytArr &s_kw, AzBytArr *s_kwo=NULL) const {
    s_kw.reset();     s_kw   << kw_dataext     << extno << "="; 
    if (s_kwo!=NULL) {
      s_kwo->reset(); *s_kwo << kw_dataext_old << extno << "=";  /* for compatibility */
    }
    return s_kw.c_str(); 
  }
  virtual void resetParam(AzParam &azp, AzpDataTrnTst _trn_tst, bool is_there_y) {
    const char *eyec = "AzpData_sparse_multi::resetParam"; 
    sp_dataext.reset(); 
    for (int ix = 0; ; ++ix) {
      AzBytArr s_kw, s_kwo, s_dataext; 
      gen_dataext_kw(ix, s_kw, &s_kwo); 
      azp.vStr(s_kw.c_str(), &s_dataext, s_kwo.c_str()); 
      if (s_dataext.length() <= 0) break; 
      sp_dataext.put(&s_dataext); 
    }
    AzX::throw_if((sp_dataext.size() <= 0), AzInputError, eyec, kw_dataext, "is missing"); 
    data.reset(sp_dataext.size()); 
    for (int ix = 0; ix < data.size(); ++ix) {
      data(ix)->reset_dsno(ix); 
      data(ix)->resetParam(azp, _trn_tst, is_there_y);   
    }
  }
  virtual void printParam(const AzOut &out, const char *pfx="") const {
    if (out.isNull()) return; 
    AzPrint o(out); 
    for (int ix = 0; ix < sp_dataext.size(); ++ix) {
      AzBytArr s_kw; 
      o.printV(gen_dataext_kw(ix, s_kw), sp_dataext.c_str(ix)); 
    }
    o.printEnd(); 
    data[0]->printParam(out, pfx); 
    for (int ix = 0; ix < data.size(); ++ix) data[ix]->printParam_data(out, pfx); /* for gen_regions */
  }
  virtual void resetParam_data(AzParam &azp) {}
  virtual void printParam_data(const AzOut &out, const char *pfx) const {}
  virtual void printHelp_data(AzHelp &h) const {}
  virtual void reset_data(const AzOut &_out, const char *nm, int _dummy_ydim=-1, 
                          AzpData_binfo *_bi=NULL /* not used */) {
    out = _out; 
    AzpData_binfo bi0;  
    for (int ix = 0; ix < data.size(); ++ix) {
      int ydim = (ix == 0) ? _dummy_ydim : 1; 
      AzBytArr s(nm, sp_dataext.c_str(ix));    
      AzpData_binfo bi; 
      data(ix)->reset_data(out, s.c_str(), ydim, &bi); 
      if (ix == 0) bi0.reset(&bi); 
      else         bi0.check_pop(&bi, 0, ix); 
    }
  }

  virtual void first_batch() { for (int ix = 0; ix < data.size(); ++ix) data(ix)->first_batch(); }
  virtual void next_batch() { for (int ix = 0; ix < data.size(); ++ix) data(ix)->next_batch(); }
  virtual void release_batch() { for (int ix = 0; ix < data.size(); ++ix) data(ix)->release_batch(); }
  virtual int dataNum_total() const { return data[0]->dataNum_total(); }
  virtual bool is_vg_x() const { return data[0]->is_vg_x(); }
  virtual bool is_vg_y() const { return data[0]->is_vg_y(); }  
  virtual bool is_sparse_x() const { return data[0]->is_sparse_x(); }
  virtual bool is_sparse_y() const { return data[0]->is_sparse_y(); }

  virtual double min_target() const { 
    if (data.size() <= 0) return 0; 
    double val = data[0]->min_target(); 
    for (int ix = 1; ix < data.size(); ++ix) val = MIN(val, data[ix]->min_target()); 
    return val; 
  }
  virtual double max_target() const { 
    if (data.size() <= 0) return 0; 
    double val = data[0]->max_target(); 
    for (int ix = 1; ix < data.size(); ++ix) val = MAX(val, data[ix]->max_target()); 
    return val; 
  }
  
  virtual const AzIntArr *dataIndex(int dsno=0) const { return data[dsno]->dataIndex(); }
  virtual int xdim(int data_no=0) const { return data[data_no]->xdim(); }
  virtual int datasetNum() const { return data.size(); }

  virtual void reset() { for (int ix = 0; ix < data.size(); ++ix) data(ix)->reset(); }
  virtual void destroy() { for (int ix = 0; ix < data.size(); ++ix) data(ix)->destroy(); }
  virtual int batchNum() const { return data[0]->batchNum(); }
  virtual int ydim() const { return data[0]->ydim(); }
  virtual int dataNum() const { return data[0]->dataNum(); }
  virtual int colNum() const { return data[0]->colNum(); }
  virtual int colNum(int dx) const { return data[0]->colNum(dx); }
  virtual int dimensionality() const { return data[0]->dimensionality(); }
  virtual int size(int index) const { return data[0]->size(index); }

  virtual const AzDicc &get_dic(int dataset_no=0) const { return data[dataset_no]->get_dic(0); }  
  virtual void gen_data(const int *dxs, int d_num, AzpDataVar_X *data_x, int dataset_no=0) const {
    ((AzpData_ *)data[dataset_no])->gen_data(dxs, d_num, data_x, 0);    
  }  
  virtual void gen_data(const int *dxs, int d_num, AzpData_X *data_x, int dataset_no=0) const {
    ((AzpData_ *)data[dataset_no])->gen_data(dxs, d_num, data_x, 0);    
  }

  virtual void gen_targets(const int *dxs, int d_num, AzPmat *m_y) const {
    data[0]->gen_targets(dxs, d_num, m_y); 
  }
  virtual void gen_membership(AzDataArr<AzIntArr> &aia_cl_dxs) const {  data[0]->gen_membership(aia_cl_dxs); }

  virtual void gen_dataweights(const int *dxs, int d_num, AzPmat *m_data) const { 
    data[0]->gen_dataweights(dxs, d_num, m_data); 
  }
  virtual void gen_targets(const int *dxs, int d_num, AzSmat *m_y) const {
    data[0]->gen_targets(dxs, d_num, m_y); 
  } 
  virtual void gen_targets(const int *dxs, int d_num, AzPmatSpa *m_y) const {
    data[0]->gen_targets(dxs, d_num, m_y); 
  }
  
  /*----------------------------------------------------------------------*/
  virtual void signature(AzBytArr &s_sign) const {
    s_sign.reset(); 
    for (int ix = 0; ix < data.size(); ++ix) {
      if (data.size() > 0) s_sign << "[" << ix << "]";  /* to make the same signature as "sparse" in the case of one dataset */
      AzBytArr s; 
      data[ix]->signature(s); 
      s_sign.c(&s); 
    } 
  }
  
protected:   
  virtual void _gen_data(const int *, int, AzPmat *, int) const { no_support("AzpData_sparse_multi", "_gen_data(dense)"); }
  virtual void _gen_data(const int *, int, AzPmatSpa *, int) const { no_support("AzpData_sparse_multi", "_gen_data(sparse)"); }
  virtual void _gen_data(const int *, int, AzPmatSpaVar *, int) const { no_support("AzpData_sparse_multi", "_gen_data(sparse and variable-sized)"); }
  virtual void _gen_data(const int *, int, AzPmatVar *, int) const { no_support("AzpData_sparse_multi", "_gen_data(dense variable-sized)"); } 
};    
#endif 
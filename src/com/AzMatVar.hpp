/* * * * *
 *  AzMatVar.hpp 
 *  Copyright (C) 2015,2017 Rie Johnson
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
#ifndef _AZ_MAT_VAR_HPP_
#define _AZ_MAT_VAR_HPP_

#include "AzUtil.hpp"

/*********************************************************************/
template <class M> /* M: matrix class */
class AzMatVar {
protected: 
  int data_num; 
  AzIntArr ia_dcolind; 
  M m; 
public: 
  AzMatVar() : data_num(0) {}
  AzMatVar(const char *fn) : data_num(0) {
    read(fn); 
  }
 
  int dataNum() const { return data_num; }
  int rowNum() const { return m.rowNum(); }
  int colNum() const { return m.colNum(); }
  void shape_str(AzBytArr &s) const { s << rowNum() << " x " << colNum() << ", #data=" << dataNum(); }
  int get_begin(int idx) const { return col_begin(idx); }
  int col_begin(int idx) const {
    check_idx(idx, "col_begin"); 
    return ia_dcolind.get(idx*2); 
  }
  int get_end(int idx) const { return col_end(idx); }
  int col_end(int idx) const {
    check_idx(idx, "col_end"); 
    return ia_dcolind.get(idx*2+1); 
  }
  void reset(const M *_m, const AzIntArr *_ia_dcolind) {
    m.set(_m); 
    ia_dcolind.reset(_ia_dcolind); 
    data_num = ia_dcolind.size()/2; 
    check_consistency(); 
  }
  void reset(const M *_m) { /* regard one column as one data point */
    AzIntArr ia; 
    for (int col = 0; col < _m->colNum(); ++col) {
      ia.put(col); ia.put(col+1); 
    }
    reset(_m, &ia); 
  }    
  void set(const M *_m, const AzIntArr *_ia_dcolind) { reset(_m, _ia_dcolind); }
  void set(const M *_m) { reset(_m); }
  void destroy() {
    ia_dcolind.reset(); 
    m.reset(); 
    data_num = 0; 
  }
  void reset() { destroy(); }

  const M *data() const { return &m; } 
  M *data_u() { return &m; } /* Use this with a lot of caution */
  const AzIntArr *h_index() const { return &ia_dcolind; }
  const AzIntArr *index() const { return &ia_dcolind; }

  void set(const AzMatVar<M> *mv0, const AzIntArr &ia_dxs) { set(mv0, ia_dxs.point(), ia_dxs.size()); }
  void set(const AzMatVar<M> *mv0, 
           const int *dxs0, /* mv0's data points */
           int dxs0_num) {  /* size of dxs0 */
    int dnum = dxs0_num; 
    data_num = dnum; 

    /*---  generate column index  ---*/
    ia_dcolind.reset(dnum*2, 0); 
  
    int offs = 0; 
    AzIntArr ia_cols; 
    int *h_dcolind = ia_dcolind.point_u(); 
    for (int ix = 0; ix < dxs0_num; ++ix) {
      int dx0 = dxs0[ix]; 
      int col0 = mv0->col_begin(dx0); 
      int col1 = mv0->col_end(dx0); 

      h_dcolind[ix*2] = offs; 
      offs += (col1 - col0);  
      h_dcolind[ix*2+1] = offs; 
      for (int col = col0; col < col1; ++col) {
        ia_cols.put(col); 
      }
    }

    m.set(mv0->data(), ia_cols.point(), ia_cols.size()); 
  }            
           
  void set(const AzMatVar *mv0, int dx0, int dx1) {
    AzIntArr ia; ia.range(dx0, dx1); 
    set(mv0, ia.point(), ia.size()); 
  }
  void set(const AzMatVar *mv) {
    reset(mv->data(), mv->index()); 
  }

  void write(const char *fn) const { AzFile::write(fn, this); }
  void read(const char *fn) { AzFile::read(fn, this); }
  static void write_hdr(AzFile *file, int d_num, const AzIntArr &ia_ind, 
                    bool do_chk=false, int cnum=-1) {
    file->writeInt(d_num); ia_ind.write(file); 
    if (do_chk) check_consistency(d_num, ia_ind, cnum); 
  }
  static void read_hdr(AzFile *file, int &d_num, AzIntArr &ia_ind) {
    d_num = file->readInt(); ia_ind.read(file); 
  }  
  void write(AzFile *file) const { 
    write_hdr(file, data_num, ia_dcolind); 
    m.write(file); 
  }
  void read(AzFile *file) {
    read_hdr(file, data_num, ia_dcolind); 
    m.read(file); 
  }  
  void write_matrix(AzFile *file) const {
    m.write(file);  
  }
  void write_matrix(const char *fn) const {
    AzFile file(fn); file.open("wb"); 
    write_matrix(&file); 
    file.close(true);     
  }
  
  /*---  treat each column as one data point; for gen_feat  ---*/
  void separate_columns() {
    AzIntArr ia; 
    for (int col = 0; col < m.colNum(); ++col) {
      ia.put(col); ia.put(col+1); 
    }
    update_index(&ia); 
  }
  void update_index(const AzIntArr *_ia_dcolind) {
    const char *eyec = "AzMatVar::update_index"; 
    if (_ia_dcolind->size() % 2 != 0) {
      throw new AzException(eyec, "index must be pairs of begin and end"); 
    }
    data_num = _ia_dcolind->size() / 2; 
    ia_dcolind.reset(_ia_dcolind); 
    check_consistency(); 
  }  
  void check_consistency() { check_consistency(data_num, ia_dcolind, m.colNum()); }    
  static void check_consistency(int data_num, const AzIntArr &ia_dcolind, int col_num) {
    const char *eyec = "AzMatVar::check_consistency"; 
    if (ia_dcolind.size() != data_num*2) throw new AzException(eyec, "#data conflict"); 
    const int *dcolind = ia_dcolind.point(); 
    for (int dx = 0; dx < data_num; ++dx) {
      int col0 = dcolind[dx*2], col1 = dcolind[dx*2+1]; 
      if (col1-col0 < 0 || col0 < 0 || col1 > col_num) {
        cout << "col0=" << col0 << " col1=" << col1 << " m.colNum()=" << col_num << endl; 
        AzX::throw_if(true, eyec, "invalid column range"); 
      }
    }
    AzX::throw_if(data_num > 0 && dcolind[data_num*2-1] != col_num, eyec, "size conflict"); 
  }
  
protected: 
  inline void check_idx(int idx, const char *msg="") const {
    AzX::throw_if(idx < 0 || idx >= data_num, "AzMatVar::check_idx", "invalid data index", msg); 
  }
};
#endif 

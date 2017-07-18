/* * * * *
 *  AzxD.hpp
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

#ifndef _AZ_XD_HPP_
#define _AZ_XD_HPP_

#include "AzUtil.hpp"
#include "AzPrint.hpp"

class AzxD {
protected:
  AzIntArr ia_sz; /* size of dim0, dim1, dim2, ... */

public: 
  AzxD() {}
  AzxD(const AzxD *inp) { ia_sz.reset(&inp->ia_sz); } 
  AzxD(int xsz, int ysz) { reset(xsz, ysz); }
  virtual bool is_var() const { return !is_valid(); }
  virtual bool is_valid() const {
    if (ia_sz.size() <= 0) return false; 
    int ix; 
    for (ix = 0; ix < ia_sz.size(); ++ix) if (ia_sz.get(ix) <= 0) return false; 
    return true; 
  }
  virtual void reset(int dim=0) { /* generate a unit size region */
    ia_sz.reset(); 
    if (dim == 0) return; 
    ia_sz.reset(dim, 1); 
  }  
  virtual bool isSame(const AzxD *inp) const {
    if (ia_sz.compare(&inp->ia_sz) != 0) return false; 
    return true; 
  }
  virtual void reset(const AzxD *inp) {
    ia_sz.reset(&inp->ia_sz); 
  }
  virtual void reset(const AzIntArr *ia) {
    ia_sz.reset(ia); 
  }
  virtual void reset(const int *arr, int len) {
    ia_sz.reset(arr, len); 
  }
  virtual void reset(int xsz, int ysz) {
    ia_sz.reset(); ia_sz.put(xsz); ia_sz.put(ysz); 
  }
  
  virtual int get_dim() const { return ia_sz.size(); }

  virtual int sz(int dx) const {
    AzX::throw_if((dx < 0 || dx >= ia_sz.size()), "AzxD::sz", "dim is out of range"); 
    return ia_sz.get(dx); 
  }
    
  virtual int get_min_size() const {
    int dim = get_dim(); 
    int minsz = -1; 
    int dx; 
    for (dx = 0; dx < dim; ++dx) {
      int mysz = sz(dx); 
      if (dx == 0 || mysz < minsz) minsz = mysz; 
    }
    return minsz; 
  }
  virtual int size() const {
    int dim = get_dim(); 
    if (dim == 0) return 0; 
    
    int region_size = 1; 
    int dx; 
    for (dx = 0; dx < dim; ++dx) {
      int mysz = sz(dx); 
      region_size *= mysz; 
    }
    return region_size; 
  }
  void write(AzFile *file) const {
    ia_sz.write(file); 
  }
  void read(AzFile *file) {
    ia_sz.read(file); 
  }
  inline virtual void show(const char *header, const AzOut &out) const {
    AzBytArr s; 
    s.c(header); 
    format(s); 
    AzPrint::writeln(out, s); 
  }
  virtual void format(AzBytArr &s, bool do_reset=false) const {
    if (do_reset) s.reset(); 
//    s.c("("); 
    int ix; 
    for (ix = 0; ix < ia_sz.size(); ++ix) {
      if (ix > 0) s.c(" x "); 
      s.cn(ia_sz.get(ix)); 
    }
//    s.c(")"); 
  }  
}; 
#endif 
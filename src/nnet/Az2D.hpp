/* * * * *
 *  Azp2D.hpp
 *  Copyright (C) 2014-2015,2017 Rie Johnson
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

#ifndef _AZ_2D_HPP_
#define _AZ_2D_HPP_

#include "AzUtil.hpp"
#include "AzPrint.hpp"
#include "AzxD.hpp"

class Az2D { /* rectangle */
public:
  int x0, xsz, y0, ysz; 
  int x1, y1; 

  void copy_from(const AzxD *xd) {
    if      (xd->get_dim() == 1) reset(xd->sz(0), 1); 
    else if (xd->get_dim() == 2) reset(xd->sz(0), xd->sz(1)); 
    else                         AzX::throw_if(true, "Az2D::reset(AzxD)", "dim mismatch"); 
  }
  void copy_to(AzxD *xd) const {
    AzX::throw_if((x0 != 0 || y0 != 0), "Az2D::copy_to(xd)", "must be located at (0,0)"); 
    int arr[2] = {xsz,ysz}; 
    xd->reset(arr, 2); 
  }
  
  /*---  ---*/
  inline bool isSame(const Az2D &inp) const {
    return isSame(&inp); 
  }
  inline bool isSame(const Az2D *inp) const {
    if (x0 != inp->x0 || y0 != inp->y0 || xsz != inp->xsz || ysz != inp->ysz) return false; 
    return true; 
  }
  Az2D() : x0(0), xsz(0), y0(0), ysz(0), x1(0), y1(0) {}
  Az2D(int sz0, int sz1) {
    reset(sz0, sz1); 
  }
  Az2D(const AzxD *xd) {
    copy_from(xd); 
  }
  void reset(int sz0, int sz1) {
    x0 = 0; xsz = sz0; y0 = 0; ysz = sz1; 
    sync();     
  }
  inline int size() const {
    return xsz*ysz; 
  }
  void reset(int inp_x0, int inp_xsz, int inp_y0, int inp_ysz) {
    x0=inp_x0; xsz=inp_xsz; y0=inp_y0; ysz=inp_ysz; 
    sync();  
  }
  void write(AzFile *file) const {
    file->writeInt(x0); 
    file->writeInt(xsz); 
    file->writeInt(y0);
    file->writeInt(ysz); 
  }
  void read(AzFile *file) {
    x0 = file->readInt(); 
    xsz = file->readInt(); 
    y0 = file->readInt(); 
    ysz = file->readInt(); 
    sync(); 
  }
  void show(const char *header, AzOut &out) const {
    AzBytArr s(header); s.c("("); s.cn(x0); s.c(","); s.cn(y0); s.c(") - ("); 
    s.cn(x1); s.c(","); s.cn(y1); s.c(")"); 
    AzPrint::writeln(out, s); 
  }
      
  /*------------------------------------------------------------*/
  void set_contained(const AzDataPool<Az2D> *patches, 
                     AzIntArr *ia_contained) const
  {
    ia_contained->reset(); 
    int px; 
    for (px = 0; px < patches->size(); ++px) {
      const Az2D *pch = patches->point(px); 
      double center_x = pch->x0 + (double)pch->xsz/2; 
      double center_y = pch->y0 + (double)pch->ysz/2; 
      if (contains(center_x, center_y)) {  
        ia_contained->put(px); 
      }
    }    
  }

  inline bool contains(double xx, double yy) const {
    if (xx >= x0 && xx < x1 && 
        yy >= y0 && yy < y1) {
      return true; 
    }
    return false; 
  }
   
protected: 
  void sync() {
    x1=x0+xsz; 
    y1=y0+ysz;
  }
}; 
#endif 
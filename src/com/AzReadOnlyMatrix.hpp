/* * * * *
 *  AzReadOnlyMatrix.hpp 
 *  Copyright (C) 2011,2012,2017 Rie Johnson
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

#ifndef _AZ_READONLY_MATRIX_HPP_
#define _AZ_READONLY_MATRIX_HPP_

#include "AzUtil.hpp"

//! Abstract class: interface for read-only vectors.  
class AzReadOnlyVector {
public:
  virtual int rowNum() const = 0; 
  virtual double get(int row_no) const = 0; 
  virtual int next(AzCursor &cursor, double &out_val) const = 0;  
  virtual int nonZeroRowNum() const = 0; 
  
  /*--------------------------------------*/
  virtual void writeText(const char *fn, int digits) const {
    AzIntArr ia; 
    ia.range(0, rowNum()); 
    writeText(fn, &ia, digits);  
  }

  /*--------------------------------------*/
  virtual void writeText(const char *fn, const AzIntArr &ia, int digits) const {
    AzFile file(fn); file.open("wb"); 
    AzBytArr s;
    for (int ix = 0; ix < ia.size(); ++ix) {
      double val = get(ia[ix]); 
      s.cn(val, digits); s.nl(); 
    }
    s.writeText(&file); 
    file.close(true); 
  }

  /*--------------------------------------*/
  virtual void to_sparse(AzBytArr &s, int digits) const {
    AzCursor cur; 
    for ( ; ; ) {
      double val; 
      int row = next(cur, val); 
      if (row < 0) break; 
      if (s.length() > 0) s << ' '; 
      s << row; 
      if (val != 1) {
        s << ':'; s.cn(val, digits); 
      }
    }    
  }

  /*--------------------------------------*/
  virtual void to_dense(AzBytArr &s, int digits) const {
    for (int row = 0; row < rowNum(); ++row) {
      if (row > 0) s << " "; 
      s.cn(get(row), digits); 
    }
  }
}; 

//! Abstract class: interface for read-only matrices.  
class AzReadOnlyMatrix {
public: 
  virtual int rowNum() const = 0; 
  virtual int colNum() const = 0;  
  virtual double get(int row_no, int col_no) const = 0; 
  virtual const AzReadOnlyVector *col(int col_no) const = 0; 
                     
  /*--------------------------------------*/
  virtual void writeText(const char *fn, int digits, 
                         bool doSparse=false,
                         bool doAppend=false) const {
    AzIntArr ia; ia.range(0, colNum()); 
    writeText(fn, ia, digits, doSparse, doAppend); 
  }

  /*--------------------------------------*/
  virtual void writeText(const char *fn, const AzIntArr &ia, int digits, 
                         bool doSparse=false, 
                         bool doAppend=false) const {
    if (doSparse) {
      AzX::throw_if(doAppend, "AzReadOnlyMatrix::writeText", "sparse matrices cannot be appended"); 
      writeText_sparse(fn, ia, digits); 
      return; 
    }
    AzFile file(fn); 
    if (doAppend) file.open("ab"); 
    else          file.open("wb");
    for (int ix = 0; ix < ia.size(); ++ix) {
      int cx = ia[ix];  
      AzBytArr s; col(cx)->to_dense(s, digits); s.nl(); 
      s.writeText(&file); 
    }
    file.close(true); 
  }
  
  /*--------------------------------------*/
  virtual void writeText_sparse(const char *fn, const AzIntArr &ia, int digits) const {
    AzFile file(fn); 
    file.open("wb"); 
    AzBytArr s_header("sparse "); s_header.cn(rowNum()); s_header.nl(); 
    s_header.writeText(&file);
    for (int ix = 0; ix < ia.size(); ++ix) {
      int cx = ia[ix];  
      AzBytArr s; col(cx)->to_sparse(s, digits); s.nl(); 
      s.writeText(&file); 
    }
    file.close(true); 
  }
}; 
#endif 

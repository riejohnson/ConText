/* * * * *
 *  AzPmat.cpp
 *  Copyright (C) 2013-2015,2017 Rie Johnson
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

#include "AzPmat.hpp"
#include "AzPrint.hpp"
#include "AzP.h"

static bool do_sync=true; 

/*---  global  ---*/
AzPdevice dev; 
int max_threads = -1; 
int max_blocks = -1; 
bool __doDebug = false; 
bool __doTry = false; 
/*----------------*/

/*------------------------------------------------*/
void AzPmat::copy_col2col(const AzPmat *m, const int *cols, int cnum) 
{
  const char *eyec = "AzPmat::copy_col2col"; 
  if (cnum <= 0) return; 
  if (cols == NULL) return; 
  AzX::throw_if_null(m, eyec, "m");
  AzX::throw_if((m->rowNum() != row_num), eyec, "#row mismatch"); 
  for (int ix = 0; ix < cnum; ++ix) {
    AzX::throw_if((cols[ix] < 0 || cols[ix] >= m->colNum() || cols[ix] >= col_num), eyec, "invalid col#"); 
  }
  AzPintArr pia_cols; 
  pia_cols.reset(cols, cnum); 
  u._copy_cols2cols(_dptr_u(), m->_dptr(), row_num, pia_cols._dptr(), pia_cols.size()); 
}

/*------------------------------------------------*/
void AzPmat::copy_scol2dcol(const AzPmat *m, const int *src_cols, const int *dst_cols, int cnum, bool do_allow_negaindex) 
{
  const char *eyec = "AzPmat::copy_scol2dcol"; 
  if (cnum <= 0) return; 
  AzX::throw_if_null(m, eyec, "m");
  AzX::throw_if((m->rowNum() != row_num), eyec, "#row mismatch"); 
  
  AzPintArr pia_scols, pia_dcols; 
  const int *_scols = NULL, *_dcols =  NULL; 
  if (src_cols == NULL) AzX::throw_if((cnum > m->colNum()), eyec, "src_cols(null) and cnum don't match");  
  else {
    for (int ix=0;ix<cnum;++ix) AzX::throw_if(src_cols[ix]>=m->colNum(), eyec, "invalid src_col#"); 
    if (!do_allow_negaindex) for (int ix=0;ix<cnum;++ix) AzX::throw_if(src_cols[ix]<0, eyec, "invalid src_col#");     
    pia_scols.reset(src_cols, cnum); 
    _scols = pia_scols._dptr(); 
  }
  if (dst_cols == NULL) AzX::throw_if((cnum > col_num), eyec, "dst_cols(null) and cnum don't match");  
  else {
    for (int ix=0;ix<cnum;++ix) AzX::throw_if(dst_cols[ix]>=col_num, eyec, "invalid dst_col#"); 
    if (!do_allow_negaindex) for (int ix=0;ix<cnum;++ix) AzX::throw_if(dst_cols[ix]<0, eyec, "invalid dst_col#");     
    pia_dcols.reset(dst_cols, cnum); 
    _dcols = pia_dcols._dptr();
  }

  u._copy_scol2dcol(_dptr_u(), m->_dptr(), row_num, _scols, _dcols, cnum); 
}

/*------------------------------------------------*/
void AzPmat::copy_dcol(const AzPmat *m, const AzPintArr2v &pia2v_dcols, int idx, bool do_allow_negaindex) 
{
  const char *eyec = "AzPmat::copy_dcol(m,PintaArr2v,idx,flag)"; 
  int cnum = pia2v_dcols.length(idx); 
  if (cnum <= 0) return; 
  AzX::throw_if_null(m, eyec, "m");
  AzX::throw_if((m->rowNum() != row_num), eyec, "#row mismatch"); 
  AzX::throw_if((cnum > m->colNum()), eyec, "too many destination columns?!");  

  const int *dst_cols = pia2v_dcols._hptr(idx); 
  for (int ix=0;ix<cnum;++ix) AzX::throw_if(dst_cols[ix]>=col_num, eyec, "invalid destination column#"); 
  if (!do_allow_negaindex) for (int ix=0;ix<cnum;++ix) AzX::throw_if(dst_cols[ix]<0, eyec, "negative dest col#");     
  const int *_dcols = pia2v_dcols._dptr(idx); 

  u._copy_scol2dcol(_dptr_u(), m->_dptr(), row_num, NULL, _dcols, cnum); 
}

/*------------------------------------------------*/
void AzPmat::set(const AzPmat *m, const AzPintArr2v &pia2v_cols, int idx, AzFloat coeff) {
  const char *eyec = "AzPmat::set(m,pia2v,idx,coeff)"; 
  int cnum = pia2v_cols.length(idx); 
  AzX::throw_if_null(m, eyec, "m");
  AzX::throw_if((cnum <= 0), eyec, "#colums must be positive"); 
  reform_noinit(m->rowNum(), cnum); 
  const int *cols = pia2v_cols._hptr(idx); 
  for (int ix = 0; ix < cnum; ++ix) AzX::throw_if((cols[ix] >= m->colNum()), eyec, "invalid col#"); 
  bool do_zero_negaindex = true; 
  u._copy_cols(_dptr_u(), m->_dptr(), row_num, pia2v_cols._dptr(idx), cnum, do_zero_negaindex, coeff); 
}

/*------------------------------------------------*/
void AzPmat::set(const AzPmat *m, const int *cols, int cnum, AzFloat coeff) /* removed do_zero_negaindex 8/28/2015 */
{
  const char *eyec = "AzPmat::set(m,cols,cnum,coeff)"; 
  AzX::throw_if_null(m, eyec, "m");
  AzX::throw_if((cnum <= 0), eyec, "#colums must be positive"); 
  reform_noinit(m->rowNum(), cnum); 
  for (int ix = 0; ix < cnum; ++ix) {
    AzX::throw_if((cols[ix] >= m->colNum()), eyec, "invalid col#"); 
  }
  AzPintArr pia_cols; 
  pia_cols.reset(cols, cnum); 
  bool do_zero_negaindex = true; 
  u._copy_cols(_dptr_u(), m->_dptr(), row_num, pia_cols._dptr(), pia_cols.size(), do_zero_negaindex, coeff); 
}

/*------------------------------------------------*/
void AzPmat::copy(const AzPmat *m, const int *cols, int cnum, bool do_zero_negaindex, AzFloat coeff)
{
  const char *eyec = "AzPmat::copy(m,cols,cnum,sw,coeff)"; 
  AzX::throw_if_null(m, eyec, "m"); 
  AzX::throw_if((cnum <= 0), eyec, "#colums must be positive"); 
  shape_chk(m->rowNum(), cnum, eyec); 
  for (int ix = 0; ix < cnum; ++ix) {
    AzX::throw_if((cols[ix] >= m->colNum()), eyec, "invalid col#"); 
  }
  AzPintArr pia_cols; 
  pia_cols.reset(cols, cnum); 
  u._copy_cols(_dptr_u(), m->_dptr(), row_num, pia_cols._dptr(), pia_cols.size(), do_zero_negaindex, coeff); 
}

/*------------------------------------------------*/
/* source = cols[destination} */
void AzPmat::add_d2s(const AzPmat *m, const AzPintArr2v &pia2v_cols, int idx, double coeff) 
{
  const char *eyec = "AzPmat::add_d2s(m,pia2v_cols,idx,coeff)"; 
  int cnum = pia2v_cols.length(idx); 
  AzX::throw_if((cnum != col_num), eyec, "#colums mismatch"); 
  AzX::throw_if((row_num != m->rowNum()), eyec, "#rows mismatch"); 
  const int *cols = pia2v_cols._hptr(idx); 
  for (int ix = 0; ix < cnum; ++ix) AzX::throw_if((cols[ix] >= m->colNum()), eyec, "invalid col#"); 
  u._add_cols_d2s(_dptr_u(), m->_dptr(), row_num, pia2v_cols._dptr(idx), cnum, (AzFloat)coeff); 
}

/*------------------------------------------------*/
/* source = cols[destination} */
void AzPmat::add_d2s(const AzPmat *m, const int *cols, int cnum, double coeff) 
{
  const char *eyec = "AzPmat::add_d2s(m,cols,cnum,coeff)"; 
  AzX::throw_if((cnum != col_num), eyec, "#colums mismatch"); 
  AzX::throw_if((row_num != m->rowNum()), eyec, "#rows mismatch"); 
  for (int ix = 0; ix < cnum; ++ix) {
    AzX::throw_if((cols[ix] >= m->colNum()), eyec, "invalid col#"); 
  }
  AzPintArr pia_cols; 
  pia_cols.reset(cols, cnum); 
  u._add_cols_d2s(_dptr_u(), m->_dptr(), row_num, pia_cols._dptr(), pia_cols.size(), (AzFloat)coeff); 
}

/*------------------------------------------------*/
/* destination = cols[source] */
void AzPmat::add_s2d(const AzPmat *m, const AzPintArr2v &piav2_cols, int idx, double coeff, bool do_z) {
  const char *eyec = "AzPmat::add_s2d(m,piav2,idx,coeff)"; 
  int cnum = piav2_cols.length(idx); 
  AzX::throw_if((cnum != m->col_num), eyec, "#colums mismatch"); 
  AzX::throw_if((row_num != m->rowNum()), eyec, "#rows mismatch"); 
  const int *cols = piav2_cols._hptr(idx); 
  for (int ix = 0; ix < cnum; ++ix) AzX::throw_if((cols[ix] >= col_num), eyec, "invalid col#"); 
  u._add_cols_s2d(_dptr_u(), m->_dptr(), row_num, piav2_cols._dptr(idx), cnum, (AzFloat)coeff, do_z); 
}

/*------------------------------------------------*/
/* destination = cols[source] */
void AzPmat::add_s2d(const AzPmat *m, const int *cols, int cnum, double coeff, bool do_z) 
{
  const char *eyec = "AzPmat::add_s2d(m,cols,cnum,coeff)"; 
  AzX::throw_if((cnum != m->col_num), eyec, "#colums mismatch"); 
  AzX::throw_if((row_num != m->rowNum()), eyec, "#rows mismatch"); 
  for (int ix = 0; ix < cnum; ++ix) {
    AzX::throw_if((cols[ix] >= col_num), eyec, "invalid col#"); 
  }
  AzPintArr pia_cols; 
  pia_cols.reset(cols, cnum); 
  u._add_cols_s2d(_dptr_u(), m->_dptr(), row_num, pia_cols._dptr(), pia_cols.size(), (AzFloat)coeff, do_z); 
}

/*------------------------------------------------*/
void AzPmat::add_rows_s2d(const AzPmat *m, const int *rows, int rnum, double coeff) 
{
  const char *eyec = "AzPmat::add_rows_s2d"; 
  AzX::throw_if((rnum != m->row_num), eyec, "#rows mismatch"); 
  AzX::throw_if((col_num != m->colNum()), eyec, "#cols mismatch"); 
  for (int ix = 0; ix < rnum; ++ix) {
    AzX::throw_if((rows[ix] >= row_num), eyec, "invalid row#"); 
  }
  AzPintArr pia_rows; 
  pia_rows.reset(rows, rnum); 
  u._add_rows_s2d(_dptr_u(), row_num, m->_dptr(), rnum, col_num, pia_rows._dptr(), (AzFloat)coeff); 
}

/*------------------------------------------------*/
void AzPmat::add_rowwise(int row1, int r_num, const AzPmat *m2, int row2, double coeff) /* this[row1::r_num,] += m2[row2::r_num,]*coeff */
{
  const char *eyec = "AzPmat::add_rowwise";  
  AzX::throw_if_null(m2, eyec, "m2"); 
  AzX::throw_if((row2 < 0 || r_num < 0 || row2+r_num > m2->rowNum()), "AzPmat::add_rowwise", "invalid request"); 
  AzIntArr ia(m2->rowNum(), -1); 
  for (int ix = row2; ix < row2+r_num; ++ix) ia.update(ix, ix-row2+row1); 
  add_rows_s2d(m2, ia.point(), ia.size(), coeff); 
}

/*------------------------------------------------*/
void AzPmat::set(const AzPmat *m, double coeff)
{
  reform_noinit(m->row_num, m->col_num); 
  u._copy(_dptr_u(), m->_dptr(), row_num*col_num, (AzFloat)coeff); 
}

/*------------------------------------------------*/
void AzPmat::set(const AzDmat *m) {
  set(m, 0, m->colNum()); 
}

/*------------------------------------------------*/
void AzPmat::set(const AzDmatc *m, const int *cols, int cols_num)
{
  const char *eyec = "AzPmat::set(dmatc,cols,cols_num)"; 
  int rnum = m->rowNum(), cnum = cols_num; 
  reform_noinit(rnum, cnum); 
  
  AzFloat *h_data = NULL; 
  AzBaseArray<AzFloat> _a; 
  _a.alloc(&h_data, rnum*cnum, eyec, "h_data"); 
  int offs = 0; 
  for (int dst_col = 0; dst_col < cols_num; ++dst_col) {
    const AZ_MTX_FLOAT *inp = m->rawcol(cols[dst_col]); 
    for (int row = 0; row < rnum; ++row) h_data[offs++] = (AzFloat)inp[row]; 
  } 
  u._copy01(_dptr_u(), h_data, rnum*cnum);   
}

/*------------------------------------------------*/
void AzPmat::set(const AzDvect *v)
{
  reform_noinit(v->rowNum(), 1); 
  mycopy01(_dptr_u(), v->point(), v->rowNum()); 
}

/*------------------------------------------------*/
void AzPmat::set(int col, const AzDvect *v)
{
  const char *eyec = "AzPmat::set(col, Dvect)"; 
  AzX::throw_if((row_num != v->rowNum()), eyec, "#row mismatch"); 
  AzX::throw_if((col < 0 || col >= col_num), eyec, "Invalid col#"); 
  mycopy01(_dptr_u(col,"set"), v->point(), row_num); 
}

/*------------------------------------------------*/
void AzPmat::set(int col1, const AzPmat *m2, int col2, double coeff)
{
  const char *eyec = "AzPmat::set(col1,m2,col2,coeff)"; 
  AzX::throw_if((row_num != m2->row_num), eyec, "#row mismatch"); 
  AzX::throw_if((col1 < 0 || col1 >= col_num), eyec, "the 1st column parameter is out of range"); 
  AzX::throw_if((col2 < 0 || col2 >= m2->col_num), eyec, "the 2nd column parameter is out of range"); 
  u._copy(_dptr_u(col1,eyec), m2->_dptr(col2,eyec), row_num, (AzFloat)coeff); 
}

/*------------------------------------------------*/
void AzPmat::set(int col1, int c_num, const AzPmat *m2, int col2, double coeff)
{
  const char *eyec = "AzPmat::set(col1,c_num,m2,col2,coeff)";  
  AzX::throw_if((row_num != m2->row_num), eyec, "#row mismatch"); 
  AzX::throw_if((col1 < 0 || col1+c_num > col_num || col2 < 0 || col2+c_num > m2->col_num), eyec, "invalid request"); 
  u._copy(_dptr_u(col1,"set,dst"), m2->_dptr(col2,"set,src"), row_num*c_num, (AzFloat)coeff); 
}

/*------------------------------------------------*/
void AzPmat::set(const AzPmat *m2, int m2_col0, int m2_col1)
{
  const char *eyec = "AzPmat::set(m,col0,col1)"; 
  int c_num = m2_col1 - m2_col0; 
  AzX::throw_if((m2_col0 < 0 || c_num <= 0 || m2_col1 > m2->col_num), eyec, "invalid request"); 
  reform_noinit(m2->row_num, c_num); 
  u._copy(_dptr_u(), m2->_dptr(m2_col0,"set(m,c0,c1),src"), row_num*c_num); 
}

/*------------------------------------------------*/
void AzPmat::set(int col, int c_num, const AzFloat *p_host, int data_len) 
{
  const char *eyec = "AzPmat::set(col1,c_num,data,data_len)"; 
  AzX::throw_if((data_len != c_num*row_num), eyec, "data length mismatch"); 
  AzX::throw_if((col < 0 || col+c_num > col_num), eyec, "invalid column"); 
  u._copy01(_dptr_u(col,"set,dst"), p_host, data_len); 
}

/*------------------------------------------------*/
void AzPmat::cbind(const AzDataArr<AzPmat> *am) 
{
  if (am->size() <= 0) {
    reform(0,0); 
    return; 
  }
  int rnum = am->point(0)->rowNum(); 
  int cnum = 0; 
  for (int ix = 0; ix < am->size(); ++ix) cnum += am->point(ix)->colNum(); 
  reform(rnum, cnum); 
  int col = 0; 
  for (int ix = 0; ix < am->size(); ++ix) {
    const AzPmat *m = am->point(ix);   
    AzX::throw_if((m->rowNum() != rnum), "AzPmat::cbind(am)", "Invalid #row"); 
    set(col, m->colNum(), m);
    col += m->colNum(); 
  }  
}

/*------------------------------------------------*/
void AzPmat::rbind(const AzPmat *m) 
{
  const char *eyec = "AzPmat::rbind"; 
  if (size() <= 0) {
    set(m); 
    return; 
  }
  if (m->size() <= 0) {
    return; 
  }
  
  AzX::throw_if((!can_rbind(m)), eyec, "#col mismatch"); 
  int r_num0 = row_num; 
  AzPmat m0(this); 
  reform(r_num0 + m->rowNum(), m->colNum()); 
  set_rowwise(0, r_num0, &m0, 0); 
  set_rowwise(r_num0, m->rowNum(), m, 0); 
}

/*------------------------------------------------*/
void AzPmat::set_rowwise(int row1, int r_num, const AzPmat *m2, int row2)
{
  const char *eyec = "AzPmat::set(row1, row_num, m2, row2)"; 
  if (r_num <= 0) return; 
  AzX::throw_if((m2->col_num != col_num), eyec, "#col mismatch"); 
  AzX::throw_if((row1 < 0 || row1 + r_num > row_num || row2 < 0 || row2 + r_num > m2->row_num), 
                eyec, "out of range"); 
  u._copy_rowwise(_dptr_u(), row_num, col_num, row1, 
                  m2->_dptr(), m2->row_num, row2, 
                  r_num); 
}                          

/*------------------------------------------------*/
/* m1[row1::r_num,col1::c_num] <- m2[row2::r_num,col2::c_num] */
void AzPmat::set_submatrix(int row1, int r_num, int col1, int c_num, const AzPmat *m2, int row2, int col2)
{
  const char *eyec = "AzPmat::set_submatrix(row1, r_num, col1, c_num, m2, row2, col2)"; 
  if (r_num <= 0 || c_num <= 0) return;
  AzX::throw_if((row1 < 0 || row1 + r_num > row_num || row2 < 0 || row2 + r_num > m2->row_num || 
                col1 < 0 || col1 + c_num > col_num || col2 < 0 || col2 + c_num > m2->col_num), eyec, "out of range"); 
  u._copy_rowwise(_dptr_u(col1), row_num, c_num, row1, 
                  m2->_dptr(col2), m2->row_num, row2, 
                  r_num); 
} 

/* different #row and #col, but #row*#col is the same */
/*------------------------------------------------*/
void AzPmat::fill(const AzPmat *m) 
{
  const char *eyec = "AzPmat::fill"; 
  int sz = size(); 
  AzX::throw_if((m->size() != sz), eyec, "size mismatch"); 
  u._copy(_dptr_u(), m->_dptr(), sz); 
}

/*------------------------------------------------*/
void AzPmat::set(double val) {
  u._setval(_dptr_u(), (AzFloat)val, row_num*col_num); 
}

/*------------------------------------------------*/
void AzPmat::setRow(int row, double val) 
{ 
  AzX::throw_if((row < 0 || row >= row_num), "AzPmat::setRow", "row# is out of range"); 
  u._setRow(_dptr_u(), row_num, col_num, row, (AzFloat)val); 
}

/*------------------------------------------------*/
void AzPmat::set(int col, double val) {
  u._setval(_dptr_u(col,"set(col,val)"), (AzFloat)val, row_num); 
}

/*------------------------------------------------*/
void AzPmat::setIdentity(int dim)
{
  AzDmat m(dim, dim);
  for (int ix = 0; ix < dim; ++ix) m.set(ix,ix, 1); 
  set(&m); 
}

/*------------------------------------------------*/
void AzPmat::add_noblas(const AzPmat *m, double coeff) 
{
  AzX::throw_if((row_num != m->row_num || col_num != m->col_num), "AzPmat::add_noblas(m,coeff)", "shape mismatch"); 
  u._add(_dptr_u(), m->_dptr(), row_num*col_num, (AzFloat)coeff); 
}

/*------------------------------------------------*/
void AzPmat::add_cublas(const AzPmat *m, double coeff) 
{
  AzX::throw_if((row_num != m->row_num || col_num != m->col_num), "AzPmat::add_cublas(m,coeff)", "shape mismatch"); 
  u._add_axpy(_dptr_u(), m->_dptr(), row_num*col_num, (AzFloat)coeff);  
}

/*------------------------------------------------*/
void AzPmat::add(double coeff, const AzPmat *m0, double coeff0) 
{
  AzX::throw_if((row_num != m0->row_num || col_num != m0->col_num), "AzPmat::add(coeff,m0,coeff0)", "shape mismatch"); 
  if (coeff == 1) u._add(_dptr_u(), m0->_dptr(), row_num*col_num, (AzFloat)coeff0); 
  else            u._add1(_dptr_u(), (AzFloat)coeff, m0->_dptr(), (AzFloat)coeff0, row_num*col_num); 
}

/*------------------------------------------------*/
void AzPmat::add_square(double coeff, const AzPmat *m0, double coeff0) 
{
  AzX::throw_if((row_num != m0->row_num || col_num != m0->col_num), "AzPmat::add_square(coeff,m0,coeff0)", "shape mismatch"); 
  u._add_sq1(_dptr_u(), (AzFloat)coeff, m0->_dptr(), (AzFloat)coeff0, row_num*col_num); 
}

/*------------------------------------------------*/
void AzPmat::add(double coeff, const AzPmat *m0, double coeff0, const AzPmat *m1, double coeff1) 
{
  const char *eyec = "AzPmat:;add(c,m0,c0,m1,c1)"; 
  AzX::throw_if((row_num != m0->row_num || col_num != m0->col_num), eyec, "1st matrix: shape mismatch"); 
  AzX::throw_if((row_num != m1->row_num || col_num != m1->col_num), eyec, "2nd matrix: shape mismatch");   
  u._add2(_dptr_u(), (AzFloat)coeff, m0->_dptr(), (AzFloat)coeff0, m1->_dptr(), (AzFloat)coeff1, row_num*col_num); 
}

/*------------------------------------------------*/
void AzPmat::add(double val) {
  u._addval(_dptr_u(), (AzFloat)val, row_num*col_num); 
}

/*------------------------------------------------*/
void AzPmat::add(int col1, const AzPmat *m2, int col2, double coeff)
{
  const char *eyec = "AzPmat::add(col1,m2,col2,coeff)"; 
  AzX::throw_if((row_num != m2->row_num), eyec, "#row mismatch"); 
  u._add(_dptr_u(col1,eyec), m2->_dptr(col2,eyec), row_num, (AzFloat)coeff); 
}

/*------------------------------------------------*/
void AzPmat::add(int col1, int c_num, const AzPmat *m2, int col2, double coeff)
{
  const char *eyec = "AzPmat::add(col1,m2,col2,coeff"; 
  AzX::throw_if((row_num != m2->row_num), eyec, "#row mismatch"); 
  AzX::throw_if((col1+c_num > col_num || col2+c_num > m2->col_num), eyec, "invalid request"); 
  u._add(_dptr_u(col1,eyec), m2->_dptr(col2,eyec), row_num*c_num, (AzFloat)coeff);   
}

/*------------------------------------------------*/
void AzPmat::add(int col1, double val) {
  u._addval(_dptr_u(col1,"AzPmat::add(col1,val)"), (AzFloat)val, row_num); 
}

/*------------------------------------------------*/
void AzPmat::elm_multi(const AzPmat *m2, bool do_inv)
{
  const char *eyec = "AzPmat::elm_multi(m2)"; 
  AzX::throw_if((row_num != m2->row_num || col_num != m2->col_num), eyec, "shape mismatch"); 
  u._elm_multi(_dptr_u(), m2->_dptr(), row_num*col_num, do_inv); 
}

/*------------------------------------------------*/
void AzPmat::elm_multi(int col1, const AzPmat *m2, int col2, bool do_inv)
{
  const char *eyec = "AzPmat::elm_multi(col1,m2,col2)"; 
  AzX::throw_if((row_num != m2->row_num), eyec, "#row mismatch"); 
  u._elm_multi(_dptr_u(col1,eyec), m2->_dptr(col2,eyec), row_num, do_inv);  
}

/*------------------------------------------------*/
void AzPmat::elm_multi(int col1, int c_num, const AzPmat *m2, int col2, bool do_inv)
{
  const char *eyec = "AzPmat::elm_multi(col1,c_num,m2,col2)"; 
  AzX::throw_if((row_num != m2->row_num), eyec, "#row mismatch"); 
  AzX::throw_if((col1 + c_num > col_num || col2 + c_num > m2->col_num), eyec, "col1+#col or col2+#col is out of range"); 
  u._elm_multi(_dptr_u(col1,eyec), m2->_dptr(col2,eyec), row_num*c_num, do_inv);  
}

/*------------------------------------------------*/
void AzPmat::multiply_eachcol(const AzPmat *m2, bool do_inv, bool do_chk)
{
  const char *eyec = "AzPmat::multiply_eachcol(m2)"; 
  if (do_chk) AzX::throw_if(col_num != m2->col_num || m2->rowNum() != 1, eyec, "wrong shape."); 
  else        AzX::throw_if(col_num != m2->size(), eyec, "wrong size"); 
  u._multiply_eachcol(_dptr_u(), row_num, col_num, m2->_dptr(), do_inv); 
}

/*------------------------------------------------*/
void AzPmat::multiply_eachrow(const AzPmat *m2, bool do_inv, bool do_chk)
{
  const char *eyec = "AzPmat::multiply_eachrow(m2)"; 
  if (do_chk) AzX::throw_if(row_num != m2->row_num || m2->colNum() != 1, eyec, "wrong shape."); 
  else        AzX::throw_if(row_num != m2->size(), eyec, "worng size"); 
  u._multiply_eachrow(_dptr_u(), row_num, col_num, m2->_dptr(), do_inv); 
}

/*------------------------------------------------*/
void AzPmat::add_eachrow(const AzPmat *m2, double coeff) {
  const char *eyec = "AzPmat::add_eachrow"; 
  AzX::throw_if_null(m2, eyec, "m2"); 
  AzX::throw_if((row_num != m2->row_num || m2->colNum() != 1), eyec, "wrong shape."); 
  u._add_eachrow(_dptr_u(), row_num, col_num, m2->_dptr(), (AzFloat)coeff); 
}

/*------------------------------------------------*/
void AzPmat::scale_by_sqrt(const AzPmat *m_sq, double epsilon, bool do_inv) 
{
  const char *eyec = "AzPmat::scale_by_sqrt"; 
  shape_chk_tmpl(m_sq, eyec); 
  AzX::throw_if((do_inv && epsilon <= 0), eyec, "epsilon must be positive when do_inv is ON"); 
  u._scale_by_sqrt(_dptr_u(), size(), m_sq->_dptr(), (AzFloat)epsilon, do_inv); 
}
 
/*------------------------------------------------*/
void AzPmat::adam_delta(const AzPmat *m_g1, const AzPmat *m_g2, double b1t, double b2t, double eps) {
  const char *eyec = "AzPmat::adam_delta"; 
  AzX::throw_if_null(m_g1, eyec, "g1");
  AzX::throw_if_null(m_g2, eyec, "g2");
  m_g1->shape_chk_tmpl(m_g2, eyec);
  set(m_g1); 
  u._adam_delta(size(), _dptr_u(), m_g2->_dptr(), (AzFloat)b1t, (AzFloat)b2t, (AzFloat)eps); 
}

/*------------------------------------------------*/
void AzPmat::multiply_noblas(double val) { u._multiply     (_dptr_u(), (AzFloat)val, row_num*col_num); }
void AzPmat::multiply_cublas(double val) { u._multiply_scal(_dptr_u(), (AzFloat)val, row_num*col_num); }
void AzPmat::multiply_noblas(int col, double val) { u._multiply     (_dptr_u(col,"AzPmat::multiply_noblas"), (AzFloat)val, row_num); }
void AzPmat::multiply_cublas(int col, double val) { u._multiply_scal(_dptr_u(col,"AzPmat::multiply_cublas"), (AzFloat)val, row_num); }

/*------------------------------------------------*/
void AzPmat::multiply_noblas(int col, int cnum, double val) 
{
  const char *eyec = "AzPmat::multiply_noblas(col,cnum,val)"; 
  AzX::throw_if((col < 0 || cnum <= 0 || col+cnum > col_num), eyec, "invalid range"); 
  u._multiply(_dptr_u(col,eyec), (AzFloat)val, row_num*cnum); 
}
void AzPmat::multiply_cublas(int col, int cnum, double val) 
{
  const char *eyec = "AzPmat::multiply_cublas(col,cnum,val)"; 
  AzX::throw_if((col < 0 || cnum <= 0 || col+cnum > col_num), eyec, "invalid range"); 
  u._multiply_scal(_dptr_u(col,eyec), (AzFloat)val, row_num*cnum); 
}

/*------------------------------------------------*/
void AzPmat::truncate(double minval, double maxval) 
{
  if (minval > maxval) return; 
  u._trun(_dptr_u(), row_num*col_num, (AzFloat)minval, (AzFloat)maxval); 
}

/*------------------------------------------------*/
void AzPmat::truncate(int col, double minval, double maxval) 
{
  if (minval > maxval) return; 
  u._trun(_dptr_u(col,"truncate"), row_num, (AzFloat)minval, (AzFloat)maxval); 
}

/*------------------------------------------------*/
void AzPmat::repmat_from(const AzPmat *m_src, 
                         int num_r, int num_c)
{ 
  int r_num = m_src->rowNum(), c_num = m_src->colNum(); 
  reform(num_r*r_num, num_c*c_num); 
  u._add_repmat(m_src->_dptr(), r_num, c_num, _dptr_u(), num_r, num_c); 
}

/*------------------------------------------------*/
void AzPmat::add_repmat(const AzPmat *m_src, 
                        int num_r, int num_c)
{ 
  int r_num = m_src->rowNum(), c_num = m_src->colNum(); 
  shape_chk(num_r*r_num, num_c*c_num, "AzPmat::add_repmat"); 
  u._add_repmat(m_src->_dptr(), r_num, c_num, _dptr_u(), num_r, num_c); 
}

/*------------------------------------------------*/
void AzPmat::transpose_from_noblas(const AzPmat *m) 
{
  int r_num = m->row_num, c_num = m->col_num; 
  reform_noinit(c_num, r_num); 
  u._transpose_noblas(m->_dptr(), r_num, c_num, _dptr_u()); 
}

/*------------------------------------------------*/
void AzPmat::transpose_from_cublas(const AzPmat *m) 
{
  int r_num = m->row_num, c_num = m->col_num; 
  reform_noinit(c_num, r_num); 
  u._transpose_cublas(m->_dptr(), r_num, c_num, _dptr_u()); 
}

/*------------------------------------------------*/
void AzPmat::binarize()  { u._binarize( _dptr_u(), row_num*col_num); }
void AzPmat::binarize1() { u._binarize1(_dptr_u(), row_num*col_num); }
void AzPmat::mark_eq(AzFloat val) { u._mark_eq(_dptr_u(), row_num*col_num, val); }
void AzPmat::mark_positive()      { u._mark_gt(_dptr_u(), row_num*col_num, 0); }
void AzPmat::mark_negative()      { u._mark_lt(_dptr_u(), row_num*col_num, 0); }
void AzPmat::mark_gt(AzFloat val) { u._mark_gt(_dptr_u(), row_num*col_num, val); }
void AzPmat::mark_lt(AzFloat val) { u._mark_lt(_dptr_u(), row_num*col_num, val); }
void AzPmat::mark_le(AzFloat val) { u._mark_le(_dptr_u(), row_num*col_num, val); }
void AzPmat::mark_ge(AzFloat val) { u._mark_ge(_dptr_u(), row_num*col_num, val); }
void AzPmat::mark_in(AzFloat v0, AzFloat v1, bool is_exclusive) { /* (x in [v0,v1])?1:0 */
  AzX::throw_if(v0 >= v1, "AzPmat::mark_in", "1st argument must be smaller than the 2nd."); 
  AzPmat m(this); 
  if (is_exclusive) { m.mark_gt(v0); mark_lt(v1); elm_multi(&m); }
  else              { m.mark_ge(v0); mark_le(v1); elm_multi(&m); }
} 
void AzPmat::mark_le_rowth(const AzDvect *v_rowth, AzFloat coeff) { 
  AzX::throw_if((v_rowth->rowNum() != row_num), "AzPmat::mark_le_rowth", "dimension mismatch"); 
  AzPmat m(v_rowth); 
  u._mark_le_rowth(_dptr_u(), row_num, col_num, m._dptr(), coeff); 
}
void AzPmat::mark_gt_colth(const AzDvect *v_colth, AzFloat coeff) { 
  const char *eyec = "AzPmat::mark_gt_colth(dvect)"; 
  AzX::throw_if_null(v_colth, eyec, "v_colth"); AzX::throw_if((v_colth->rowNum()!=col_num), eyec, "dimension mismatch"); 
  AzPmat m(v_colth); 
  u._mark_gt_colth(_dptr_u(), row_num, col_num, m._dptr(), coeff); 
}
void AzPmat::mark_gt_colth(const AzPmat *m, AzFloat coeff) { 
  const char *eyec = "AzPmat::mark_gt_colth(pmat)"; 
  AzX::throw_if_null(m, eyec, "colth"); AzX::throw_if((m->rowNum()!=1||m->colNum()!=col_num), eyec, "dimension mismatch"); 
  u._mark_gt_colth(_dptr_u(), row_num, col_num, m->_dptr(), coeff); 
}
 
/*------------------------------------------------*/
void AzPmat::get_eachCol(const int *rows, int rnum, AzPmat *m) const {
  const char *eyec = "AzPmat::get_eachCol"; 
  AzX::throw_if_null(rows, eyec, "rows"); AzX::throw_if_null(m, eyec, "m"); 
  AzX::throw_if((rnum != col_num), eyec, "number of given row indexes must be the number of columns."); 
  for (int ix = 0; ix < rnum; ++ix) AzX::throw_if((rows[ix]<0 || rows[ix]>=row_num), eyec, "invalid row#"); 
  m->reform(1, col_num); 
  AzPintArr pia; pia.reset(rows, rnum); 
  u._get_eachCol(_dptr(), row_num, col_num, pia._dptr(), m->_dptr_u()); 
}

/*------------------------------------------------*/
void AzPmat::exp(AzPmat *m_mask) 
{
  if (m_mask != NULL) {
    m_mask->reform_tmpl(this); 
    u._exp(_dptr_u(), row_num*col_num, m_mask->_dptr_u()); 
  } else {
    u._exp(_dptr_u(), row_num*col_num, NULL); 
  }
}

/*------------------------------------------------*/
void AzPmat::log()    { u._log(   _dptr_u(), row_num*col_num); }
void AzPmat::squareroot()   { u._sqrt(  _dptr_u(), row_num*col_num); }
void AzPmat::square() { u._square(_dptr_u(), row_num*col_num); }
void AzPmat::pow(double val) { u._pow(_dptr_u(), row_num*col_num, (AzFloat)(val)); }
void AzPmat::inverse() { u._inverse(_dptr_u(), row_num*col_num); }

/*------------------------------------------------*/
double AzPmat::sum_cublas(int col) const {
  if (col < 0) return u._sum_cublas(_dptr(), row_num*col_num); 
  else         return u._sum_cublas(_dptr(col,"sum_cublas"), row_num); 
}

/*------------------------------------------------*/
double AzPmat::sum_noblas(int col) const {
  if (col < 0) return u._sum_noblas(_dptr(), row_num*col_num); 
  else         return u._sum_noblas(_dptr(col,"sum_noblas"), row_num); 
}

/*------------------------------------------------*/
double AzPmat::squareSum_noblas(int col) const {
  if (col < 0) return u._squareSum_noblas(_dptr(), row_num*col_num); 
  else         return u._squareSum_noblas(_dptr(col,"squareSum_noblas"), row_num);   
}

/*------------------------------------------------*/
double AzPmat::norm2_cublas(int col) const {
  if (col < 0) return u._norm2_cublas(_dptr(), row_num*col_num); 
  else         return u._norm2_cublas(_dptr(col,"squareSum_cublas"), row_num);   
}

/*------------------------------------------------*/
double AzPmat::absSum_noblas(int col) const {
  if (col < 0) return u._absSum_noblas(_dptr(), row_num*col_num); 
  else         return u._absSum_noblas(_dptr(col,"absSum_noblas"), row_num);   
}

/*------------------------------------------------*/
double AzPmat::absSum_cublas(int col) const {
  if (col < 0) return u._absSum_cublas(_dptr(), row_num*col_num); 
  else         return u._absSum_cublas(_dptr(col,"absSum_cublas"), row_num);   
}

/*------------------------------------------------*/
int AzPmat::nz(int col) const {
  if (col < 0) return u._nz(_dptr(), row_num*col_num); 
  else         return u._nz(_dptr(col,"nz"), row_num); 
}

/*------------------------------------------------*/
void AzPmat::colSquareSum_noblas(const AzPmat *m) {
  reform(1, m->col_num); 
  u._add_colSquareSum(m->_dptr(), m->row_num, m->col_num, _dptr_u());   
}

/*------------------------------------------------*/
/* slow */
void AzPmat::colSquareSum_cublas(const AzPmat *m) {
  reform_noinit(1, m->col_num); 
  AzDmat md(1, m->col_num); 
  for (int col = 0; col < m->colNum(); ++col) {
    AzFloat val = u._norm2_cublas(m->_dptr(col), m->row_num); 
    md.set(0, col, val*val); 
  }
  set(&md); 
}

/*------------------------------------------------*/
void AzPmat::colAbsSum(const AzPmat *m) {
  reform(1, m->col_num); 
  u._add_colAbsSum(m->_dptr(), m->row_num, m->col_num, _dptr_u());  
}

/*------------------------------------------------*/
void AzPmat::add_colSquareSum(const AzPmat *m) {
  shape_chk(1, m->col_num, "AzPmat::add_colSquareSum"); 
  u._add_colSquareSum(m->_dptr(), m->row_num, m->col_num, _dptr_u());   
}

/*------------------------------------------------*/
void AzPmat::add_colAbsSum(const AzPmat *m) {
  shape_chk(1, m->col_num, "AzPmat::add_colAbsSum"); 
  u._add_colAbsSum(m->_dptr(), m->row_num, m->col_num, _dptr_u());  
}

/*------------------------------------------------*/
double AzPmat::min(int col, int *out_row) const { return u._min(_dptr(col,"min"), row_num, out_row); }
double AzPmat::max(int col, int *out_row) const { return u._max(_dptr(col,"max"), row_num, out_row); }

/*------------------------------------------------*/
double AzPmat::min(int *out_row, int *out_col) const {
  int index; 
  double val = u._min(_dptr(), row_num*col_num, &index); 
  if (out_row != NULL) *out_row = index % row_num; 
  if (out_col != NULL) *out_col = index / row_num;   
  return val; 
}

/*------------------------------------------------*/
double AzPmat::max(int *out_row, int *out_col) const {
  int index; 
  double val = u._max(_dptr(), row_num*col_num, &index); 
  if (out_row != NULL) *out_row = index % row_num; 
  if (out_col != NULL) *out_col = index / row_num;   
  return val; 
}

/*------------------------------------------------*/
void AzPmat::max_eachCol(AzIntArr *ia_ind, AzPmat *m_max) const {
  AzFloat *val_ptr = NULL; 
  if (m_max != NULL) {
    m_max->reform_noinit(1, col_num); 
    val_ptr = m_max->_dptr_u(); 
  }

  int *ind_ptr = NULL; 
  AzPintArr pia; 
  if (ia_ind != NULL) {
    ia_ind->reset(col_num, -1); 
    pia.alloc(col_num);
    ind_ptr = pia._dptr_u();     
  }
  if (size() <= 0) return; 

  u._max_eachCol(_dptr(), row_num, col_num, ind_ptr, val_ptr);   
  
  if (ia_ind != NULL) {
    pia.get(ia_ind); 
  }
}

/*------------------------------------------------*/
void AzPmat::min_eachCol(AzIntArr *ia_ind, AzPmat *m_min) const {
  AzFloat *val_ptr = NULL; 
  if (m_min != NULL) {
    m_min->reform_noinit(1, col_num); 
    val_ptr = m_min->_dptr_u(); 
  }

  int *ind_ptr = NULL; 
  AzPintArr pia; 
  if (ia_ind != NULL) {
    ia_ind->reset(col_num, -1); 
    pia.alloc(col_num);
    ind_ptr = pia._dptr_u();     
  }
  if (size() <= 0) return; 

  u._min_eachCol(_dptr(), row_num, col_num, ind_ptr, val_ptr);   
  
  if (ia_ind != NULL) {
    pia.get(ia_ind); 
  }
}

/*------------------------------------------------*/
void AzPmat::get(AzDmat *m, bool do_chk) const
{
  const char *eyec = "AzPmat::get(Dmat,do_chk)"; 
  m->reform_chk(row_num, col_num, do_chk, eyec); 
  for (int col = 0; col < m->colNum(); ++col) {
    get(col, m->col_u(col), true); 
  }
}

/*------------------------------------------------*/
void AzPmat::get(int col, AzDvect *v, bool do_chk) const
{
  const char *eyec = "AzPmat::get(col,Dvect)"; 
  v->reform_chk(row_num, do_chk, eyec); 
  mycopy10(v->point_u(), _dptr(col,"get"), row_num); 
}

/*------------------------------------------------*/
void AzPmat::get(AzDvect *v, bool do_chk) const
{
  const char *eyec = "AzPmat::get(Dvect)"; 
  int sz = row_num*col_num; 
  v->reform_chk(sz, do_chk, eyec); 
  mycopy10(v->point_u(), _dptr(), sz); 
}

/*------------------------------------------------*/
void AzPmat::copy_to(AzDmatc *m, int dmatc_col) const 
{
  const char *eyec = "AzPmat::copy_to(AzDmatc,dmatc_col)"; 
  AzX::throw_if((m == NULL), eyec, "null input"); 
  AzX::throw_if((row_num != m->rowNum()), eyec, "#row mismatch"); 
  AzX::throw_if((dmatc_col + col_num > m->colNum()), eyec, "destination is too small"); 
  
  int sz = row_num*col_num; 
  AzFloat *elm = NULL; 
  AzBaseArray<AzFloat> _arr(sz, &elm); 
  mycopy10(elm, _dptr(), sz);  
  m->rawset(dmatc_col, elm, sz); 
}

/*------------------------------------------------*/
/* high overhead */
double AzPmat::get(int row, int col) const
{
  const char *eyec = "AzPmat::get(row, col)"; 
  AzX::throw_if((row < 0 || row >= row_num || col < 0 || col >= col_num), eyec, "out of range"); 
  double host_var; 
  mycopy10(&host_var, _column(col, _dptr(), row_num)+row, 1); 
  return host_var; 
}

/*------------------------------------------------*/
/* high overhead */
void AzPmat::set(int row, int col, double val)
{
  const char *eyec = "AzPmat::set(row, col, val)"; 
  AzX::throw_if((row < 0 || row >= row_num || col < 0 || col >= col_num), eyec, "out of range"); 
  mycopy01(_column(col, _dptr_u(), row_num)+row, &val, 1); 
}

/*------------------------------------------------*/
/* t(m1) * m2 */
void AzPmat::prod10(const AzPmat *m1, 
                    const AzPmat *m2, 
                    const AzPstreams *streams,                     
                    double alpha, double beta)
{
  const char *eyec = "AzPmat::prod10"; 
  int num = m1->row_num; 
  AzX::throw_if((m2->row_num != num), eyec, "shape mismatch"); 
  int r_num = m1->col_num; 
  int c_num = m2->col_num; 
  if (beta != 0) shape_chk(r_num, c_num, eyec); 
  else           reform(r_num, c_num); 
  u._prod10(_dptr_u(), r_num, c_num, m1->_dptr(), m1->row_num, m2->_dptr(), m2->row_num, num, streams, (AzFloat)alpha, (AzFloat)beta); 
}

/*------------------------------------------------*/
/* t(m1) * m2 */
void AzPmat::prod10_mask(const AzSmat *m_mask, /* 0: don't compute */
                    const AzPmat *m1, const AzPmat *m2,             
                    double alpha, double beta) {
  const char *eyec = "AzPmat::prod10_mask"; 
#ifdef __AZ_GPU__
  AzX::no_support(true, eyec, "prod10_mask on GPU"); 
#else  
  int num = m1->row_num, r_num = m1->col_num, c_num = m2->col_num;
  AzX::throw_if((m2->row_num != num), eyec, "shape mismatch"); 
  AzX::throw_if(m_mask->rowNum() != r_num || m_mask->colNum() != c_num, 
                eyec, "The mask shape is wrong."); 
  if (beta != 0) shape_chk(r_num, c_num, eyec); 
  else           reform(r_num, c_num); 
  u._prod10_mask(m_mask, _dptr_u(), r_num, c_num, m1->_dptr(), m1->row_num, 
                 m2->_dptr(), m2->row_num, num, (AzFloat)alpha, (AzFloat)beta); 
#endif                  
}

/*------------------------------------------------*/
/* m1 * t(m2) */
void AzPmat::prod01(const AzPmat *m1, 
                     const AzPmat *m2, 
                     const AzPstreams *streams,                      
                     double alpha, double beta)
{
  const char *eyec = "AzPmat::prod01"; 
  int num = m1->col_num; 
  AzX::throw_if((m2->col_num != num), eyec, "shape mismatch"); 
  int r_num = m1->row_num; 
  int c_num = m2->row_num; 
  if (beta != 0) shape_chk(r_num, c_num, eyec); 
  else           reform(r_num, c_num); 
  u._prod01(_dptr_u(), r_num, c_num, m1->_dptr(), m1->row_num, m2->_dptr(), m2->row_num, num, streams, (AzFloat)alpha, (AzFloat)beta); 
}

/*------------------------------------------------*/
/* m1 * m2 */
void AzPmat::prod00(const AzPmat *m1, 
                     const AzPmat *m2, 
                     const AzPstreams *streams, 
                     double alpha, double beta)
{
  const char *eyec = "AzPmat::prod00"; 
  int num = m1->col_num; 
  AzX::throw_if((m2->row_num != num), eyec, "shape mismatch"); 
  int r_num = m1->row_num; 
  int c_num = m2->col_num; 
  if (beta != 0) shape_chk(r_num, c_num, eyec); 
  else           reform(r_num, c_num); 
  u._prod00(_dptr_u(), r_num, c_num, m1->_dptr(), m1->row_num, m2->_dptr(), m2->row_num, num, streams, (AzFloat)alpha, (AzFloat)beta); 
}

/*------------------------------------------------*/
/*------------------------------------------------*/
void AzPmat::colSum(const AzPmat *m_inp, bool do_chk) /* this <- colSum(inp) */ {
  const char *eyec = "AzPmat::colSum"; 
  AzX::throw_if_null(m_inp, eyec, "input");   
  int r_num = 1; 
  int c_num = m_inp->col_num; 
  reform_chk(r_num, c_num, do_chk, eyec); 

  AzPmat m_one(1, m_inp->row_num); m_one.set(1); 
  prod(&m_one, m_inp, false, false); 
}

/*------------------------------------------------*/
void AzPmat::rowSum(const AzPmat *m_inp, bool do_chk) /* this <- rowSum(inp) */ {
  const char *eyec = "AzPmat::rowSum"; 
  AzX::throw_if_null(m_inp, eyec, "input"); 
  int r_num = m_inp->row_num; 
  int c_num = 1; 
  reform_chk(r_num, c_num, do_chk, eyec); 

  AzPmat m_one(m_inp->col_num, 1); m_one.set(1); 
  prod(m_inp, &m_one, false, false); 
}

/*------------------------------------------------*/
void AzPmat::add_colSum(const AzPmat *m_inp) /* this += colSum(inp) */ {
  const char *eyec = "AzPmat::add_colSum"; 
  AzX::throw_if_null(m_inp, eyec, "input");   
  int r_num = 1; 
  int c_num = m_inp->col_num; 
  shape_chk(r_num, c_num, eyec); 

  AzPmat m_one(1, m_inp->row_num); m_one.set(1); 
  add_prod(&m_one, m_inp, false, false); 
}

/*------------------------------------------------*/
void AzPmat::add_rowSum(const AzPmat *m_inp) /* this += rowSum(inp) */ {
  const char *eyec = "AzPmat::add_rowSum"; 
  AzX::throw_if_null(m_inp, eyec, "input"); 
  int r_num = m_inp->row_num; 
  int c_num = 1; 
  shape_chk(r_num, c_num, eyec); 

  AzPmat m_one(m_inp->col_num, 1); m_one.set(1); 
  add_prod(m_inp, &m_one, false, false); 
}

/*------------------------------------------------*/
void AzPmat::normalize1()
{
  AzPmat m_colSum; 
  m_colSum.colSum(this); 
  AzDvect v_sum; 
  m_colSum.get(&v_sum); 
  const double *sum = v_sum.point(); 
  for (int col = 0; col < v_sum.rowNum(); ++col) {
    if (sum[col] != 0) divide(col, sum[col]); 
  }
}

/*------------------------------------------------*/
void AzPmatVar::set(const AzPmatVar *mv0, 
                    int d0_begin, /* mv0's data index */
                    int d0_end)   /* mv0's data index */
{
  const char *eyec = "AzPmatVar::set(var,begin,end)"; 
  AzX::throw_if_null(mv0, eyec, "input");   
  if (d0_begin < 0) d0_begin = 0; 
  if (d0_end < 0) d0_end = mv0->data_num; 
  AzX::throw_if((d0_end-d0_begin <= 0 || d0_end > mv0->data_num), eyec, "requested data range is invalid"); 
  
  int dnum = d0_end - d0_begin; 
  data_num = dnum; 

  if (d0_begin == 0 && d0_end == mv0->data_num) {
    ia_dcolind.reset(&mv0->ia_dcolind); 
    pia_dcolind.reset(&mv0->pia_dcolind); 
    m.set(&mv0->m); 
  }
  else {
    int src_pos = d0_begin*2, len = 2*dnum; 
    pia_dcolind.reset(&mv0->pia_dcolind, src_pos, len); 
    pia_dcolind.get(&ia_dcolind);     
    int col0 = ia_dcolind.get(0); 
    int col1 = ia_dcolind.get(dnum*2-1); 
    m.set(&mv0->m, col0, col1);     
    pia_dcolind.add(-col0); 
    ia_dcolind.add(-col0); 
  } 
}

/*------------------------------------------------*/
void AzPmatVar::set(const AzPmatVar *mv0, 
                    const int *dxs0, /* mv0's data points */
                    int dxs0_num)   /* size of dxs0 */
{
  const char *eyec = "AzPmatVar::set(var,dxs,num)"; 
  AzX::throw_if_null(mv0, eyec, "input");   
  int dnum = dxs0_num; 
  data_num = dnum; 

  /*---  generate column index  ---*/
  ia_dcolind.reset(dnum*2, 0); 
  const AzIntArr *ia_inp_dcolind = mv0->h_index(); 
  
  int my_col = 0; 
  int max_cnum = 0;
  
  int *h_dcolind = ia_dcolind.point_u(); 
  for (int ix = 0; ix < dxs0_num; ++ix) {
    int dx0 = dxs0[ix]; 
    int inp_col0 = ia_inp_dcolind->get(dx0*2); 
    int inp_col1 = ia_inp_dcolind->get(dx0*2+1); 
    
    h_dcolind[ix*2] = my_col; 
    int my_cnum = inp_col1 - inp_col0; 
    max_cnum = MAX(max_cnum, my_cnum); 
    my_col += my_cnum; 
    h_dcolind[ix*2+1] = my_col; 
  }

  /*---  copy data  ---*/
  pia_dcolind.reset(&ia_dcolind); 
  m.reform(mv0->m.rowNum(), my_col); 
  AzPintArr pia_dxs0, pia_dcolind0; 
  pia_dxs0.reset(dxs0, dxs0_num); /* copy to device */
  u._copy_vardata(m.rowNum(), mv0->pia_dcolind._dptr(), pia_dxs0._dptr(), pia_dxs0.size(), max_cnum, mv0->m._dptr(), 
                  pia_dcolind._dptr(), m._dptr_u());                
}  

/*------------------------------------------------*/
void AzPmatVar::_set(int cc, const AzPmat *_m, const AzPintArr *_pia_dcolind, const AzIntArr *_ia_dcolind, 
                     bool do_noinit) /* used only when _m is null */ {
  const char *eyec = "AzPmatVar::_set(cc,m,pia,ia)"; 
  if (_pia_dcolind != NULL) {
    pia_dcolind.reset(_pia_dcolind); 
    pia_dcolind.get(&ia_dcolind); 
  }
  else if (_ia_dcolind != NULL) {
    pia_dcolind.reset(_ia_dcolind); 
    ia_dcolind.reset(_ia_dcolind); 
  }
  else {
    AzX::throw_if(true, eyec, "No data index"); 
  }
  AzX::throw_if((ia_dcolind.size()%2 != 0), eyec, "#dcolind should be even"); 
  data_num = ia_dcolind.size()/2; 
  const int *dcolind = ia_dcolind.point(); 
  if (_m == NULL) {
    AzX::throw_if((cc <= 0), eyec, "#channel must be positive"); 
    int cnum = (data_num <= 0) ? 0 : dcolind[data_num*2-1]; 
    if (do_noinit) m.reform_noinit(cc, cnum); 
    else           m.reform(cc, cnum);  
  }
  else {
    m.set(_m); 
  }

  int cnum = m.colNum(); 
  int rnum = m.rowNum();
  for (int dx = 0; dx < data_num; ++dx) {
    int col0 = dcolind[dx*2]; 
    int col1 = dcolind[dx*2+1]; 
    AzX::throw_if((col1-col0 < 0), eyec, "end cannot be smaller than begin"); 
    AzX::throw_if((col0 < 0 || col1 > cnum), eyec, "column index is pointing outside of matrix"); 
    AzX::throw_if((dx > 0 && col0 != dcolind[dx*2-1]), eyec, "data is out of order"); 
  }
  AzX::throw_if((data_num > 0 && dcolind[data_num*2-1] != cnum), eyec, "data size mismatch"); 
}

/*------------------------------------------------*/
/*------------------------------------------------*/
void AzPmat::set(const AzDmat *m, int col0, int col1) {
  const char *eyec = "AzPmat::set(dmat,col0,col1)"; 
  AzX::throw_if((col0 < 0 || col1-col0 <= 0 || col1 > m->colNum()), eyec, "invalid range");
  AzX::throw_if_null(m, eyec, "input"); 
  int rnum = m->rowNum(); 
  int cnum = col1-col0; 
  reform_noinit(rnum, cnum); 
  
  AzFloat *h_data = NULL; 
  AzBaseArray<AzFloat> _a; 
  _a.alloc(&h_data, rnum*cnum, eyec, "h_data"); 
  int offs = 0; 
  for (int col = col0; col < col1; ++col) {
    const double *inp = m->col(col)->point(); 
    int row; 
    for (row = 0; row < rnum; ++row) h_data[offs++] = (AzFloat)inp[row]; 
  } 
  u._copy01(_dptr_u(), h_data, rnum*cnum);   
}

/*------------------------------------------------*/
void AzPmatVar::set(const AzDmatcVar *mv, const int *dxs, int dxs_num) {
  AzX::throw_if_null(mv, "AzPmatVar::set(dmatcvar,dxs,num)", "input");   
  AzIntArr ia_cols, ia_dcolind; 
  for (int ix = 0; ix < dxs_num; ++ix) {
    int dx = dxs[ix]; 
    int col0 = mv->col_begin(dx), col1 = mv->col_end(dx); 
    ia_dcolind.put(ia_cols.size()); 
    for (int col = col0; col < col1; ++col) ia_cols.put(col); 
    ia_dcolind.put(ia_cols.size());     
  }
  AzPmat m; m.set(mv->data(), ia_cols.point(), ia_cols.size()); 
  set(&m, &ia_dcolind); 
}

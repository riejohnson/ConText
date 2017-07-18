/* * * * *
 *  AzSmat.cpp 
 *  Copyright (C) 2011-2017 Rie Johnson
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


#include "AzUtil.hpp"
#include "AzSmat.hpp"
#include "AzPrint.hpp"

#define AzVectSmall 32

/*-------------------------------------------------------------*/
void AzSmat::initialize(int inp_row_num, int inp_col_num, bool asDense) {
  const char *eyec = "AzSmat::initialize (asDense)"; 
  AzX::throw_if(inp_col_num < 0 || inp_row_num < 0, eyec, "#column and #row must be non-negative"); 
  AzX::throw_if(column != NULL || col_num > 0, eyec, "_release() must be called before this"); 
  col_num = inp_col_num; 
  row_num = inp_row_num; 
  a.alloc(&column, col_num, eyec, "column"); 
  if (asDense) {
    for (int cx = 0; cx < col_num; ++cx) column[cx] = AzSvect::new_svect(this->row_num, asDense); 
  }
  dummy_zero.reform(row_num); 
}

/*-------------------------------------------------------------*/
void AzSmat::initialize(const AzSmat *inp) {
  AzX::throw_if_null(inp, "AzSmat::initialize(AzSmat*)");
  bool asDense = false; 
  initialize(inp->row_num, inp->col_num, asDense); 
  if (inp->column != NULL) {
    for (int cx = 0; cx < this->col_num; ++cx) {
      if (inp->column[cx] != NULL) column[cx] = AzSvect::new_svect(inp->column[cx]); 
    }
  }
}

/*-------------------------------------------------------------*/
void AzSmat::reform(int row_num, int col_num, bool asDense) {
  _release(); 
  initialize(row_num, col_num, asDense); 
}

/*-------------------------------------------------------------*/
void AzSmat::resize(int new_col_num) {
  const char *eyec = "AzSmat::resize"; 
  if (new_col_num == col_num) return; 
  AzX::throw_if (new_col_num < 0, eyec, "new #columns must be positive"); 
  a.realloc(&column, new_col_num, eyec, "column"); 
  col_num = new_col_num; 
}

/*-------------------------------------------------------------*/
void AzSmat::resize(int new_row_num, int new_col_num) {
  resize(new_col_num); 

  if (column != NULL) {
    for (int col = 0; col < col_num; ++col) {
      if (column[col] != NULL) column[col]->resize(new_row_num); 
    }
  }
  dummy_zero.resize(new_row_num); 
  row_num = new_row_num; 
}

/*-------------------------------------------------------------*/
void AzSvect::resize(int new_row_num) {
  AzX::throw_if(new_row_num < row_num, "AzSvect::resize", "no support for shrinking"); 
  row_num = new_row_num; 
}

/*-------------------------------------------------------------*/
void AzSmat::set(const AzSmat *inp) {
  if (inp->row_num != this->row_num || 
      inp->col_num != this->col_num) {
    reform(inp->row_num, inp->col_num); 
  }
  for (int cx = 0; cx < this->col_num; ++cx) {
    if (inp->column[cx] == NULL) {
      delete column[cx]; 
      column[cx] = NULL; 
    }
    else {
      if (column[cx] == NULL) column[cx] = AzSvect::new_svect(row_num); 
      column[cx]->set(inp->column[cx]); 
    }
  }
}

/*-------------------------------------------------------------*/
void AzSmat::set(const AzSmat *inp, int col0, int col1) {
  AzX::throw_if(col0 < 0 || col1 < 0 || col1 > inp->col_num || col0 >= col1, 
                "AzSmat::set(inp,col0,col1)", "out of range"); 
  if (inp->row_num != this->row_num || 
      inp->col_num != col1-col0) {
    reform(inp->row_num, col1-col0); 
  }
  for (int cx = col0; cx < col1; ++cx) {
    int my_cx = cx - col0; 
    if (inp->column[cx] == NULL) {
      delete column[my_cx]; 
      column[my_cx] = NULL; 
    }
    else {
      if (column[my_cx] == NULL) column[my_cx] = AzSvect::new_svect(row_num); 
      column[my_cx]->set(inp->column[cx]); 
    }
  }
}

/*-------------------------------------------------------------*/
int AzSmat::set(const AzSmat *inp, const int *cols, int cnum,  /* new2old */
                bool do_zero_negaindex) {
  if (row_num != inp->row_num || col_num != cnum) {
    reform(inp->row_num, cnum); 
  }
  int negaindex = 0; 
  int my_col; 
  for (my_col = 0; my_col < cnum; ++my_col) {
    int col = cols[my_col]; 
    if (col < 0 && do_zero_negaindex) {
      delete column[my_col]; 
      column[my_col] = NULL;     
      ++negaindex; 
      continue; 
    }   
    AzX::throw_if(col < 0 || col >= inp->col_num, "AzSmat::set(inp,cols,cnum)", "invalid col#"); 
    if (inp->column[col] == NULL) {
      delete column[my_col]; 
      column[my_col] = NULL; 
    }
    else {
      if (column[my_col] == NULL) column[my_col] = AzSvect::new_svect(row_num); 
      column[my_col]->set(inp->column[col]); 
    }
  }
  return negaindex; 
}

/*-------------------------------------------------------------*/
void AzSmat::set(int col0, int col1, const AzSmat *inp, int i_col0) {
  const char *eyec = "AzSmat::set(col0,col1,inp)"; 
  AzX::throw_if(col0 < 0 || col1-col0 <= 0 || col1 > col_num, eyec, "requested columns are out of range"); 
  int i_col1 = i_col0 + (col1-col0); 
  AzX::throw_if(i_col0 < 0 || i_col1 > inp->col_num, eyec, "requested columns are out of range in the input matrix"); 
  AzX::throw_if(row_num != inp->row_num, eyec, "#rows mismatch"); 
  
  int i_col = i_col0; 
  for (int col = col0; col < col1; ++col, ++i_col) {
    if (inp->column[i_col] == NULL) {
      delete column[col]; column[col] = NULL; 
    }
    else {
      if (column[col] == NULL) column[col] = AzSvect::new_svect(row_num); 
      column[col]->set(inp->column[i_col]);       
    }
  }
}             
                 
/*-------------------------------------------------------------*/
void AzSmat::reduce(const int *cols, int cnum) { /* new2old; must be sorted */
  if (column == NULL) {
    reform(row_num, cnum); 
    return; 
  }

  int negaindex = 0; 
  int new_col; 
  for (new_col = 0; new_col < cnum; ++new_col) {
    int old_col = cols[new_col]; 
    AzX::throw_if(old_col < 0 || old_col >= col_num, "AzSmat::reduce(cols,cnum)", "invalid col#"); 
    AzX::throw_if(new_col > 0 && old_col <= cols[new_col-1], "AzSmat::reduce(cols,cnum)", "column# must be sorted"); 
    
    if (old_col == new_col) {}
    else if (column[old_col] == NULL) {
      delete column[new_col]; 
      column[new_col] = NULL; 
    }
    else {
      if (column[new_col] == NULL) column[new_col] = AzSvect::new_svect(row_num); 
      column[new_col]->set(column[old_col]); 
    }
  }
  resize(cnum); 
}

/*-------------------------------------------------------------*/
void AzSmat::transpose(AzSmat *m_out, 
                       int col_begin, int col_end) const {
  int col_b = col_begin, col_e = col_end; 
  if (col_b < 0) {
    col_b = 0; 
    col_e = col_num; 
  }
  else {
    AzX::throw_if (col_b >= col_num || col_e < 0 || col_e > col_num || col_e - col_b <= 0, 
                   "AzSmat::transpose", "column range error"); 
  }

  _transpose(m_out, col_b, col_e); 
}

/*-------------------------------------------------------------*/
void AzSmat::_transpose(AzSmat *m_out, 
                        int col_begin, 
                        int col_end) const {
  int row_num = rowNum(); 

  m_out->reform(col_end - col_begin, row_num); 

  AzIntArr ia_row_count; 
  ia_row_count.reset(row_num, 0); 
  int *row_count = ia_row_count.point_u(); 

  for (int cx = col_begin; cx < col_end; ++cx) {
    /* rewind(cx); */
    AzCursor cursor; 
    for ( ; ; ) {
      double val; 
      int rx = next(cursor, cx, val); 
      if (rx < 0) break;

      ++row_count[rx]; 
    }
  }
  for (int rx = 0; rx < row_num; ++rx) {
    if (row_count[rx] > 0) m_out->col_u(rx)->clear_prepare(row_count[rx]); 
  }

  for (int cx = col_begin; cx < col_end; ++cx) {
    /* rewind(cx); */
    AzCursor cursor; 
    for ( ; ; ) {
      double val; 
      int rx = next(cursor, cx, val); 
      if (rx < 0) break;

      m_out->col_u(rx)->set_inOrder(cx - col_begin, val); 
    }
  }
}

/*-------------------------------------------------------------*/
void AzSvect::clear_prepare(int num) {
  AzX::throw_if (num > row_num, "AzSvect::prepare", "input is too large"); 
  clear(); 
  elm_num = 0; 
  if (num > 0) {
    int elm_num_max = num; 
    a.alloc(&elm, elm_num_max, "AzSvect::prepare", "elm"); 
  }  
}

/*-------------------------------------------------------------*/
bool AzSmat::isZero() const {
  for (int cx = 0; cx < col_num; ++cx) {
    if (column[cx] != NULL && !column[cx]->isZero()) return false; 
  }
  return true; 
}

/*-------------------------------------------------------------*/
bool AzSmat::isZero(int col_no) const {
  if (col_no < 0 || col_no >= col_num || 
      column[col_no] == NULL) {
    return true; 
  }
  return column[col_no]->isZero();
}

/*-------------------------------------------------------------*/
double AzSmat::max(int *out_row, int *out_col, 
                   bool ignoreZero) const {
  int max_row = -1, max_col = -1; 
  double max_val = 0; 
  for (int cx = 0; cx < col_num; ++cx) {
    double local_max = 0; 
    int local_rx = -1; 
    if (column[cx] == NULL) {
      if (!ignoreZero) {
        local_max = 0; 
        local_rx = 0; 
      }
    }
    else {
      local_max = column[cx]->max(&local_rx, ignoreZero); 
    }
    if (local_rx >= 0) {
      if (max_col < 0 || local_max > max_val) {
        max_col = cx; 
        max_row = local_rx; 
        max_val = local_max; 
      }
    }
  }
  if (out_row != NULL) *out_row = max_row; 
  if (out_col != NULL) *out_col = max_col; 
  return max_val; 
}

/*-------------------------------------------------------------*/
double AzSmat::min(int *out_row, int *out_col, 
                   bool ignoreZero) const {
  int min_row = -1, min_col = -1; 
  double min_val = 0; 
  for (int cx = 0; cx < col_num; ++cx) {
    double local_min = 0; 
    int local_rx = -1; 
    if (column[cx] == NULL) {
      if (!ignoreZero) {
        local_min = 0; 
        local_rx = 0; 
      }
    }
    else {
      local_min = column[cx]->min(&local_rx, ignoreZero); 
    }
    if (local_rx >= 0) {
      if (min_col < 0 || local_min < min_val) {
        min_col = cx; 
        min_row = local_rx; 
        min_val = local_min; 
      }
    }
  }
  if (out_row != NULL) {
    *out_row = min_row; 
  }
  if (out_col != NULL) {
    *out_col = min_col; 
  }
  return min_val; 
}

/*-------------------------------------------------------------*/
void AzSmat::set(int row_no, int col_no, double val) {
  check_col(col_no, "AzSmat::set (row, col, val)"); 
  if (column[col_no] == NULL) column[col_no] = AzSvect::new_svect(row_num); 
  column[col_no]->set(row_no, val); 
}

/*-------------------------------------------------------------*/
void AzSmat::add(int row_no, int col_no, double val) {
  check_col(col_no, "AzSmat::add"); 
  if (val == 0) return; 
  if (column[col_no] == NULL) column[col_no] = AzSvect::new_svect(row_num); 
  column[col_no]->add(row_no, val); 
}

/*-------------------------------------------------------------*/
void AzSmat::add(const AzSmat *m1) {
  if (m1 == NULL) return; 
  AzX::throw_if(row_num!=m1->rowNum() || col_num!=m1->colNum(), "AzSmat::add", "shape mismatch"); 
  for (int cx = 0; cx < col_num; ++cx) {
    AzIFarr ifa1; m1->col(cx)->nonZero(&ifa1); 
    if (ifa1.size() <= 0) continue; /* nothing to add */
    AzIFarr ifa0; col(cx)->nonZero(&ifa0); 
    if (ifa0.size() <= 0) col_u(cx)->load(&ifa1); 
    else {
      ifa0.concat(&ifa1); 
      ifa0.squeeze_Sum(); 
      ifa0.sort_Int(); 
      col_u(cx)->load(&ifa0);
    }
  }
}

/*-------------------------------------------------------------*/
void AzSmat::multiply(int row_no, int col_no, double val) {
  check_col(col_no, "AzSmat::multiply (row, col, val)"); 
  if (column[col_no] == NULL) return; 
  column[col_no]->multiply(row_no, val); 
}

/*-------------------------------------------------------------*/
void AzSmat::multiply(double val) {
  for (int cx = 0; cx < col_num; ++cx) {
    if (column[cx] != NULL) column[cx]->multiply(val); 
  }
}

/*-------------------------------------------------------------*/
double AzSmat::get(int row_no, int col_no) const {
  check_col(col_no, "AzSmat::get"); 
  if (column[col_no] == NULL) return 0; 
  return column[col_no]->get(row_no); 
}

/*-------------------------------------------------------------*/
void AzSmat::dump(const AzOut &out, const char *header, 
                  const AzStrPool *sp_row, 
                  const AzStrPool *sp_col, 
                  int cut_num) const {
  if (out.isNull()) return; 

  AzPrint o(out); 

  const char *my_header = ""; 
  if (header != NULL) my_header = header; 
  o.printBegin(my_header, ",", "="); 
  /* (row,col)\n */
  o.pair_inBrackets(row_num, col_num, ","); 
  o.printEnd(); 
 
  for (int cx = 0; cx < col_num; ++cx) {
    if (column[cx] == NULL) continue; 

    /* column=cx (col_header) */
    o.printBegin("", " ", "="); 
    o.print("column", cx); 
    if (sp_col != NULL) o.inParen(sp_col->c_str(cx)); 
    o.printEnd(); 
    column[cx]->dump(out, "", sp_row, cut_num); 
  }
  o.flush(); 
}

/*-------------------------------------------------------------*/
void AzSmat::normalize() {
  for (int cx = 0; cx < col_num; ++cx) {
    if (column[cx] != NULL) column[cx]->normalize(); 
  }
}

/*-------------------------------------------------------------*/
void AzSmat::normalize1() {
  for (int cx = 0; cx < col_num; ++cx) {
    if (column[cx] != NULL) column[cx]->normalize1(); 
  }
}

/*-------------------------------------------------------------*/
void AzSmat::clear() {
  for (int cx = 0; cx < col_num; ++cx) {
    if (column[cx] != NULL) {
      delete column[cx]; 
      column[cx] = NULL; 
    }
  }
}

/*-------------------------------------------------------------*/
int AzSmat::next(AzCursor &cursor, int col, double &out_val) const {
  check_col(col, "AzSmat::next");
  if (column[col] == NULL) {
    out_val = 0; 
    return AzNone; 
  }
  return column[col]->next(cursor, out_val); 
}

/*-------------------------------------------------------------*/
/*-------------------------------------------------------------*/
/*-------------------------------------------------------------*/
void AzSvect::reform(int inp_row_num, bool asDense) {
  _release(); 
  initialize(inp_row_num, asDense); 
}

/*-------------------------------------------------------------*/
void AzSvect::initialize(int inp_row_num, bool asDense) {
  const char *eyec = "AzSvect::initialize"; 
  AzX::throw_if (inp_row_num < 0, eyec, "#row must be non-negative"); 
  AzX::throw_if(elm != NULL || elm_num > 0, eyec, "occupied"); 
  row_num = inp_row_num; 
  if (asDense) {
    a.alloc(&elm, row_num, eyec, "elm"); 
    elm_num = row_num; 
    for (int ex = 0; ex < elm_num; ++ex) {
      elm[ex].no = ex; 
      elm[ex].val = 0; 
    }
  }
}

/*-------------------------------------------------------------*/
void AzSvect::set(const AzReadOnlyVector *inp, double coefficient) {
  AzX::throw_if_null(inp, "AzSvect::set(readonly)");
  if (row_num != inp->rowNum()) reform(inp->rowNum()); 
  clear_prepare(inp->nonZeroRowNum()); 

  /* inp->rewind(); */
  AzCursor cursor; 
  for ( ; ; ) {
    double val; 
    int row = inp->next(cursor, val); 
    if (row < 0) break; 
    set_inOrder(row, val); 
  }
  if (coefficient != 1) multiply(coefficient); 
}

/* !!!!! Assuming there is no duplicated data index */
/*-------------------------------------------------------------*/
void AzSvect::nonZero(AzIFarr *ifa, 
                      const AzIntArr *ia_sorted) const { /* filter: must be sorted */
  int idx_num; 
  const int *idx = ia_sorted->point(&idx_num); 
  if (idx_num <= 0) return; 
  int xx = 0; 
  for (int ex = 0; ex < elm_num; ++ex) {
    if (elm[ex].val != 0) {
      for ( ; xx < idx_num; ++xx) {
        AzX::throw_if(xx > 0 && idx[xx] <= idx[xx-1], "AzSvect::nonZero", "filter must be sorted"); 
        if (idx[xx] == elm[ex].no) {
          ifa->put(elm[ex].no, elm[ex].val); 
          ++xx; 
          break; 
        }
        else if (idx[xx] > elm[ex].no) {
          break; 
        }
      }
      if (xx >= idx_num) break; 
    }
  }
}

/*-------------------------------------------------------------*/
void AzSvect::nonZero(AzIFarr *ifa) const {
  for (int ex = 0; ex < elm_num; ++ex) {
    if (elm[ex].val != 0) ifa->put(elm[ex].no, elm[ex].val); 
  }
}

/*-------------------------------------------------------------*/
void AzSvect::all(AzIFarr *ifa) const {
  ifa->prepare(row_num); 
  int last_no = -1; 
  for (int ex = 0; ex < elm_num; ++ex) {
    for (int rx = last_no+1; rx < elm[ex].no; ++rx) ifa->put(rx, 0); 
    ifa->put(elm[ex].no, elm[ex].val); 
    last_no = elm[ex].no; 
  }
  for (int rx = last_no+1; rx < row_num; ++rx) ifa->put(rx, 0);   
  AzX::throw_if (ifa->size() != row_num, "AzSvect::all", "something is wrong"); 
}

/*-------------------------------------------------------------*/
void AzSvect::zeroRowNo(AzIntArr *ia) const {
  ia->reset(); 
  int last_no = -1; 
  for (int ex = 0; ex < elm_num; ++ex) {
    for (int rx = last_no+1; rx < elm[ex].no; ++rx) ia->put(rx); 
    if (elm[ex].val == 0) ia->put(elm[ex].no); 
    last_no = elm[ex].no; 
  }
  for (int rx = last_no+1; rx < row_num; ++rx) ia->put(rx);   
}

/* returns the first one */
/*-------------------------------------------------------------*/
int AzSvect::nonZeroRowNo() const {
  for (int ex = 0; ex < elm_num; ++ex) {
    if (elm[ex].val != 0) return elm[ex].no; 
  }
  return -1; 
}

/*-------------------------------------------------------------*/
void AzSvect::nonZeroRowNo(AzIntArr *intq) const {
  for (int ex = 0; ex < elm_num; ++ex) {
    if (elm[ex].val != 0) intq->put(elm[ex].no); 
  }
}

/*-------------------------------------------------------------*/
int AzSmat::nonZeroRowNo(AzIntArr *ia_nzrows) const {
  ia_nzrows->reset(); 
  if (column == NULL) return 0; 
  for (int col = 0; col < col_num; ++col) {
    if (column[col] != NULL) {
      AzIntArr ia; column[col]->nonZeroRowNo(&ia); 
      ia_nzrows->concat(&ia); 
    }
  }
  int nz = ia_nzrows->size(); 
  ia_nzrows->unique(); 
  return nz; 
}

/*-------------------------------------------------------------*/
int AzSvect::nonZeroRowNum() const {
  int count = 0; 
  for (int ex = 0; ex < elm_num; ++ex) {
    if (elm[ex].val != 0) ++count;  
  }
  return count; 
}

/*-------------------------------------------------------------*/
bool AzSvect::isZero() const {
  for (int ex = 0; ex < elm_num; ++ex) {
    if (elm[ex].val != 0) return false; 
  }
  return true; 
}

/*-------------------------------------------------------------*/
void AzSmat::log_of_plusone() {
  if (column == NULL) return; 
  for (int col = 0; col < col_num; ++col) {
    if (column[col] != NULL) column[col]->log_of_plusone(); 
  }
}
/*-------------------------------------------------------------*/
void AzSvect::log_of_plusone() {
  for (int ex = 0; ex < elm_num; ++ex) {
    AzX::throw_if (elm[ex].val < 0, "AzSvect::log_of_plusone", "components must be non-negative"); 
    if (elm[ex].val > 0) elm[ex].val = log(elm[ex].val+1); 
  }
}

/*-------------------------------------------------------------*/
double AzSvect::max(int *out_row_no, 
                    bool ignoreZero) const {
  double max_val = -1; 
  int max_row = -1; 
  int ex; 
  for (ex = 0; ex < elm_num; ++ex) {
    if (ignoreZero && elm[ex].val == 0) continue; 
    if (max_row < 0 || elm[ex].val > max_val) {
      max_val = elm[ex].val; 
      max_row = elm[ex].no;  
    }
  }
  if (!ignoreZero && max_val < 0 && elm_num < row_num) {
    max_val = 0; 
    for (ex = 0; ex < elm_num; ++ex) if (elm[ex].no != ex) break;
    if (ex == 0) max_row = 0; 
    else         max_row = elm[ex - 1].no + 1; 
  }
  if (out_row_no != NULL) *out_row_no = max_row; 
  if (max_row < 0 && ignoreZero) max_val = 0; 

  return max_val; 
}

/*-------------------------------------------------------------*/
double AzSvect::maxAbs(int *out_row_no, 
                       double *out_real_val) const {
  double real_val = -1, max_val = -1; 
  int max_row = -1; 
  for (int ex = 0; ex < elm_num; ++ex) {
    double abs_val = fabs(elm[ex].val); 
    if (max_row < 0 || abs_val > max_val) {
      max_val = abs_val; 
      max_row = elm[ex].no;  
      real_val = elm[ex].val; 
    }
  }
  if (max_row < 0) {
    max_val = 0; 
    max_row = 0; 
    real_val = 0; 
  }
  if (out_row_no != NULL) *out_row_no = max_row; 
  if (out_real_val != NULL) *out_real_val = real_val; 

  return max_val; 
}

/*-------------------------------------------------------------*/
double AzSvect::min(int *out_row_no, 
                    bool ignoreZero) const {
  double min_val = 1.0; 
  int min_row = -1; 

  int ex; 
  for (ex = 0; ex < elm_num; ++ex) {
    if (ignoreZero && elm[ex].val == 0) continue; 
    if (min_row < 0 || elm[ex].val < min_val) {
      min_val = elm[ex].val; 
      min_row = elm[ex].no;  
    }
  }
  if (!ignoreZero && min_val > 0 && elm_num < row_num) {
    min_val = 0; 
    for (ex = 0; ex < elm_num; ++ex) if (elm[ex].no != ex) break;
    if (ex == 0) min_row = 0; 
    else         min_row = elm[ex - 1].no + 1; 
  }
  if (out_row_no != NULL) *out_row_no = min_row; 
  if (ignoreZero && min_row < 0) min_val = 0; 

  return min_val; 
}

/*-------------------------------------------------------------*/
double AzSvect::minPositive(int *out_row_no) const {
  double min_val = 0; 
  int min_row = -1; 
  for (int ex = 0; ex < elm_num; ++ex) {
    if (elm[ex].val > 0) {
      if (min_row < 0 || elm[ex].val < min_val) {
        min_val = elm[ex].val; 
        min_row = elm[ex].no;  
      }
    }
  }
  if (out_row_no != NULL) *out_row_no = min_row; 
  return min_val; 
}

/*-------------------------------------------------------------*/
void AzSvect::set(int row_no, double val) {
  check_row(row_no, "AzSvect::set"); 
  _checkVal(val); 

  int where = to_insert(row_no); 
  elm[where].val = (AZ_MTX_FLOAT)val; 
}

/*-------------------------------------------------------------*/
void AzSmat::set(double val) {
  if (val == 0) {
    zeroOut(); 
    return; 
  }
  for (int col = 0; col < col_num; ++col) {
    if (column[col] == NULL) column[col] = AzSvect::new_svect(row_num); 
    column[col]->set(val); 
  }
}

/*-------------------------------------------------------------*/
void AzSvect::set(double val) {
  _checkVal(val); 

  if (elm_num == row_num) {
    for (int rx = 0; rx < row_num; ++rx) {
      elm[rx].no = rx; 
      elm[rx].val = (AZ_MTX_FLOAT)val; 
    }
  }
  else {
    const char *eyec = "AzSvect::set (all)"; 
    a.free(&elm); 
    a.alloc(&elm, row_num, eyec, "elm"); 
    elm_num = row_num; 
    for (int ex = 0; ex < elm_num; ++ex) {
      elm[ex].no = ex; 
      elm[ex].val = (AZ_MTX_FLOAT)val; 
    }
  }
}

/*-------------------------------------------------------------*/
void AzSvect::change_rowno(int new_row_num, const AzIntArr *ia_old2new, 
                           bool do_zero_negaindex) {
  const char *eyec = "AzSvect::change_rowno"; 
  AzX::throw_if(ia_old2new->size() != row_num, eyec, "insufficient index"); 
  for (int ex = 0; ex < elm_num; ++ex) {
    int new_row = ia_old2new->get(elm[ex].no); 
    if (new_row != elm[ex].no) {
      if (new_row < 0) {
        if (do_zero_negaindex) elm[ex].val = 0; 
        else                   AzX::throw_if(true, eyec, "negative index"); 
      }
      else if (new_row >= new_row_num) {
        AzX::throw_if(true, eyec, "new row# is out of range"); 
      }
      elm[ex].no = new_row; 
    }
    AzX::throw_if(ex > 0 && elm[ex].no <= elm[ex-1].no, eyec, "rows are out of order.");  /* added feb02_2015 */
  }
  row_num = new_row_num; 
}

/*-------------------------------------------------------------*/
void AzSmat::change_rowno(int new_row_num, const AzIntArr *ia_old2new, 
                           bool do_zero_negaindex) {
  if (column == NULL) {
    row_num = new_row_num; 
    return; 
  }
  for (int col = 0; col < col_num; ++col) {
    if (column[col] != NULL) column[col]->change_rowno(new_row_num, ia_old2new, do_zero_negaindex);
  }
  row_num = new_row_num; 
}                    

/*-------------------------------------------------------------*/
void AzSvect::add(int row_no, double val) {
  check_row(row_no, "AzSvect::add"); 
  if (val == 0) return; 

  int where = to_insert(row_no); 
  double new_val = (double)elm[where].val + val; 
  _checkVal(new_val); 
  elm[where].val = (AZ_MTX_FLOAT)new_val; 
}

/*-------------------------------------------------------------*/
void AzSvect::plus_one_log() {
  for (int ex = 0; ex < elm_num; ++ex) {
    if (elm[ex].val == 0); 
    else if (elm[ex].val < 0) {
      AzX::throw_if(true, "AzSvect::plus_one_log", "Negative input"); 
    }
#if 0     
    else elm[ex].val = log(elm[ex].val+1); 
#else
    else elm[ex].val = (AZ_MTX_FLOAT)log((double)elm[ex].val+(double)1); /* 03/15/2015 */
#endif   
  }
}
/*-------------------------------------------------------------*/
void AzSmat::plus_one_log() {
  for (int col = 0; col < col_num; ++col) {
    if (column[col] != NULL) column[col]->plus_one_log(); 
  }
}

/*-------------------------------------------------------------*/
double AzSvect::sum() const {
  double sum = 0; 
  for (int ex = 0; ex < elm_num; ++ex) {
    sum += (double)elm[ex].val; 
  }
  return sum; 
}

/*-------------------------------------------------------------*/
double AzSvect::absSum() const {
  double sum = 0; 
  for (int ex = 0; ex < elm_num; ++ex) {
    sum += fabs((double)elm[ex].val); 
  }
  return sum; 
}

/*-------------------------------------------------------------*/
void AzSvect::multiply(int row_no, double val) {
  check_row(row_no, "AzSvect::multiply"); 
  if (val == 1) return; 
  
  int where = find(row_no); 
  if (where < 0) return; 

  for ( ; where < elm_num; ++where) {
    if (elm[where].no == row_no) {
      double new_val = (double)elm[where].val * val; 
      _checkVal(new_val); 
      elm[where].val = (AZ_MTX_FLOAT)new_val; 
    }
    else if (elm[where].no > row_no) {
      return; 
    }
  }
}

/*-------------------------------------------------------------*/
void AzSvect::multiply(double val) {
  if (val == 1) return; 
  for (int ex = 0; ex < elm_num; ++ex) {
    double new_val = (double)elm[ex].val * val; 
    _checkVal(new_val); 
    elm[ex].val = (AZ_MTX_FLOAT)((double)elm[ex].val * val); 
  }
}

/*-------------------------------------------------------------*/
void AzSvect::scale(const double *vect1) {
  for (int ex = 0; ex < elm_num; ++ex) {
    double new_val = (double)elm[ex].val * vect1[elm[ex].no]; 
    _checkVal(new_val); 
    elm[ex].val = (AZ_MTX_FLOAT)(new_val); 
  }
}

/*-------------------------------------------------------------*/
int AzSvect::to_insert(int row_no) {
  const char *eyec = "AzSvect::to_insert"; 
  int where = 0; 
  if (elm_num != 0) {
    where = find_forRoom(row_no); 
    for ( ; where < elm_num; ++where) {
      if      (elm[where].no == row_no) return where;  
      else if (elm[where].no > row_no)  break;
    }
  }

  int elm_num_max = a.size(); 
  if (elm_num >= elm_num_max) {
    elm_num_max += inc(); 
    elm_num_max = MIN(elm_num_max, row_num);
    a.realloc(&elm, elm_num_max, eyec, "elm"); 
  }

  for (int ex = elm_num - 1; ex >= where; --ex) elm[ex + 1] = elm[ex]; 
  ++elm_num; 

  /*-----  initialize the new element  -----*/
  elm[where].no = row_no; 
  elm[where].val = 0; 

  return where; 
}

/*-------------------------------------------------------------*/
double AzSvect::get(int row_no) const {
  check_row(row_no, "AzSvect::get"); 
  int where = find(row_no); 
  if (where < 0) return 0; 
  
  for ( ; where < elm_num; ++where) {
    if      (elm[where].no == row_no) return elm[where].val; 
    else if (elm[where].no > row_no)  return 0; 
  }
  return 0; 
}

/*-------------------------------------------------------------*/
int AzSvect::find(int row_no, int from_this) const {
  bool isFound; 
  int where = find_forRoom(row_no, from_this, &isFound); 
  if (!isFound) return -1; 
  return where; 
}

/*-------------------------------------------------------------*/
int AzSvect::find_forRoom(int row_no, 
                          int from_this, 
                          bool *out_isFound) const {
  if (out_isFound != NULL) *out_isFound = false; 
  if (elm_num == row_num) {  
    if (out_isFound != NULL) *out_isFound = true; 
    return row_no; 
  }
  if (elm_num == 0) return 0; /* failed */

  /*---  Decide where to search  ---*/
  int lx = 0; 
  if (from_this >= 0) lx = from_this; 
  int hx = elm_num - 1; 

  /*---  Just scan if not many  ---*/
  if (hx - lx <= AzVectSmall) {
    for ( ; lx <= hx; ++lx) {
      if (elm[lx].no == row_no) {
        if (out_isFound != NULL) *out_isFound = true; 
        return lx; 
      }
      else if (elm[lx].no > row_no) {
        return lx; /* failed */
      }
    }
  }

  /*---  Do binary search if many  ---*/
  for ( ; ; ) {
    if (lx > hx) return lx; 
    int mx = (lx + hx) / 2; 
    int cmp = row_no - elm[mx].no; 
    if (cmp < 0) hx = mx - 1; 
    else if (cmp > 0) lx = mx + 1; 
    else {
      if (out_isFound != NULL) *out_isFound = true; 
      return mx; 
    }
  }
}

/*-------------------------------------------------------------*/
double AzSvect::selfInnerProduct() const {
  double n2 = 0; 
  for (int ex = 0; ex < elm_num; ++ex) {
    if (elm[ex].val != 0) n2 += (double)elm[ex].val * (double)elm[ex].val; 
  }
  return n2; 
}

/*-------------------------------------------------------------*/
double AzSvect::normalize() {
  double norm2 = 0; 
  for (int ex = 0; ex < elm_num; ++ex) norm2 += (elm[ex].val * elm[ex].val); 

  if (norm2 != 0) {
    norm2 = sqrt(norm2); 
    for (int ex = 0; ex < elm_num; ++ex) {
      double new_val = (double)elm[ex].val / norm2; 
      _checkVal(new_val); 
      elm[ex].val = (AZ_MTX_FLOAT)new_val; 
    }
  }
  return norm2; 
}

/*-------------------------------------------------------------*/
double AzSvect::normalize1() {

  double sum = 0; 
  for (int ex = 0; ex < elm_num; ++ex) sum += elm[ex].val; 
  if (sum != 0) {
    for (int ex = 0; ex < elm_num; ++ex) {
      double new_val = (double)elm[ex].val / sum; 
      _checkVal(new_val); 
      elm[ex].val = (AZ_MTX_FLOAT)((double)elm[ex].val / sum); 
    }
  }
  return sum; 
}

/*-------------------------------------------------------------*/
void AzSvect::clear() {
  a.free(&elm); 
  elm_num = 0; 
}

/*-------------------------------------------------------------*/
void AzSvect::zeroOut() {
  elm_num = 0; 
}

/*-------------------------------------------------------------*/
int AzSvect::next(AzCursor &cursor, double &out_val) const {
  int nonzero_ex = MAX(cursor.get(), 0); 
  for ( ; nonzero_ex < elm_num; ++nonzero_ex) if (elm[nonzero_ex].val != 0) break;

  cursor.set(nonzero_ex + 1);  /* prepare for next "next" */

  if (nonzero_ex < elm_num) {
    out_val = elm[nonzero_ex].val; 
    return elm[nonzero_ex].no; 
  }

  /*---  end of the elements  ---*/
  out_val = 0; 
  return AzNone; 
} 

/*-------------------------------------------------------------*/
void AzSvect::dump(const AzOut &out, const char *header, 
                     const AzStrPool *sp_row, 
                     int cut_num) const {
  if (out.isNull()) return; 

  const char *my_header = ""; 
  if (header != NULL) my_header = header; 

  AzPrint o(out); 
  int indent = 3; 
  o.printBegin(my_header, ",", "=", indent); 
  o.print("elm_num", elm_num); 
  o.printEnd(); 

  if (cut_num > 0) {
    _dump(out, sp_row, cut_num); 
    return; 
  }

  for (int ex = 0; ex < elm_num; ++ex) {
    if (elm[ex].val == 0) continue;  

    o.printBegin("", " ", "=", indent); 
    o.inBrackets(elm[ex].no, 4); 
    o.print(elm[ex].val, 5, true); 
    if (sp_row != NULL) {
      const char *row_header = sp_row->c_str(elm[ex].no); 
      o.inParen(row_header); 
    }
    o.printEnd(); 
  }
  o.flush(); 
}

/*-------------------------------------------------------------*/
void AzSvect::_dump(const AzOut &out, 
                     const AzStrPool *sp_row, 
                     int cut_num) const {
  AzIFarr ifa_nz; 
  nonZero(&ifa_nz); 

  ifa_nz.sort_Float(false); 
  ifa_nz.cut(cut_num); 
  int num = ifa_nz.size(); 

  AzPrint o(out); 
  for (int ix = 0; ix < num; ++ix) {
    int row_no; 
    double val = ifa_nz.get(ix, &row_no); 

    int indent = 3; 
    o.printBegin("", " ", "=", indent); 
    o.inBrackets(row_no, 4); 
    o.print(val, 5, true); 
    if (sp_row != NULL) {
      const char *row_header = sp_row->c_str(row_no); 
      o.inParen(row_header); 
    }
    o.printEnd();
  }
  o.newLine(); 
  o.flush();  
}

/*-------------------------------------------------------------*/
void AzSvect::zerooutNegative() {
  for (int ex = 0; ex < elm_num; ++ex) {
    if (elm[ex].val < 0) elm[ex].val = 0; 
  }
}

/*-------------------------------------------------------------*/
void AzSvect::cut(double min_val) {
  for (int ex = 0; ex < elm_num; ++ex) {
    if (fabs(elm[ex].val) < min_val) elm[ex].val = 0; 
  }
}

/*-------------------------------------------------------------*/
void AzSmat::cut(double min_val) {
  if (column == NULL)  return; 
  for (int cx = 0; cx < col_num; ++cx) {
    if (column[cx] != NULL) column[cx]->cut(min_val); 
  }
}
 
/*-------------------------------------------------------------*/
void AzSvect::only_keep(int num) {
  AzX::throw_if(num <= 0, "AzSvect::only_keep", "Expected a positive number"); 
  int nz = nonZeroRowNum(); 
  if (nz <= num) return; 
 
  if (min() >= 0) { /* no negative values */
    AzIFarr ifa; ifa.prepare(nz); 
    nonZero(&ifa); 
    ifa.sort_Float(false); 
    ifa.cut(num); 
    ifa.sort_Int(true); 
    load(&ifa);       
  }
  else {   
    AzIFarr ifa; ifa.prepare(nz); 
    for (int ex = 0; ex < elm_num; ++ex) {
      if (elm[ex].val != 0) ifa.put(ex, fabs(elm[ex].val)); 
    }   
    ifa.sort_Float(false); 
    ifa.cut(num); 
    AzIFarr ifa_val; ifa_val.prepare(ifa.size()); 
    for (int ix = 0; ix < ifa.size(); ++ix) {
      int ex; 
      ifa.get(ix, &ex); 
      ifa_val.put(elm[ex].no, elm[ex].val); 
    }
    ifa_val.sort_Int(true); 
    load(&ifa_val);    
  }
}

/*-------------------------------------------------------------*/
void AzSmat::only_keep(int num) {
  if (column == NULL)  return; 
  for (int cx = 0; cx < col_num; ++cx) {
    if (column[cx] != NULL) column[cx]->only_keep(num); 
  }
}

/*-------------------------------------------------------------*/
void AzSmat::zerooutNegative() {
  if (column == NULL) return;
  for (int cx = 0; cx < col_num; ++cx) {
    if (column[cx] != NULL) column[cx]->zerooutNegative();
  }
}

/*-------------------------------------------------------------*/
void AzSmat::zeroOut() {
  if (column == NULL) return; 
  for (int cx = 0; cx < col_num; ++cx) {
    if (column[cx] != NULL) column[cx]->zeroOut(); 
  }
}

/*-------------------------------------------------------------*/
void AzSvect::load(const AzIntArr *ia_row, double val, 
                   bool do_ignore_negative) {
  int num; 
  const int *row = ia_row->point(&num); 
  AzIFarr ifa_row_val; ifa_row_val.prepare(num); 
  for (int ix = 0; ix < num; ++ix) {
    if (!do_ignore_negative || row[ix] >= 0) ifa_row_val.put(row[ix], val); 
  }
  load(&ifa_row_val); 
}

/* must be called in the order of the rows */
/*-------------------------------------------------------------*/
void AzSvect::set_inOrder(int row_no, double val) {
  const char *eyec = "AzSvect::set_inOrder"; 
  check_row(row_no, eyec);
  AzX::throw_if (elm_num > 0 && elm[elm_num-1].no >= row_no, eyec, "input is not in the order"); 

  int where = elm_num; 

  int elm_num_max = a.size(); 
  if (elm_num >= elm_num_max) {
    elm_num_max += inc(); 
    elm_num_max = MIN(elm_num_max, row_num);
    a.realloc(&elm, elm_num_max, eyec, "elm"); 
  }
  ++elm_num; 

  elm[where].no = row_no; 
  elm[where].val = (AZ_MTX_FLOAT)val; 
}

/*-------------------------------------------------------------*/
void AzSvect::load(const AzIFarr *ifa_row_val) {  
  const char *eyec = "AzSvect::load"; 
  int num = ifa_row_val->size(); 
  
  clear_prepare(num); 

  int prev_row = -1; 

  for (int ix = 0; ix < num; ++ix) {
    int row; 
    double val = ifa_row_val->get(ix, &row);     
    if (row < 0 || row >= row_num || /* out of range */
        row <= prev_row) { /* out of order */
      cout << "row=" << row << " row_num=" << row_num << " prev_row=" << prev_row << endl; 
      AzX::throw_if(true, "AzSvect::load", "Invalid input"); 
    }

    elm[elm_num].no = row; 
    _checkVal(val); 
    elm[elm_num].val = (AZ_MTX_FLOAT)val; 
    ++elm_num; 
    prev_row = row;  /* corrected on 2/18/2014 */    
  }
}

/*-------------------------------------------------------------*/
bool AzSmat::isSame(const AzSmat *inp) const {
  if (col_num != inp->col_num) return false; 
  for (int cx = 0; cx < col_num; ++cx) {
    bool isZero1 = isZero(cx); 
    bool isZero2 = inp->isZero(cx); 
    if (isZero1 != isZero2) return false; 
    if (isZero1) continue; 
    if (!column[cx]->isSame(inp->column[cx])) return false; 
  }
  return true; 
}

/*-------------------------------------------------------------*/
bool AzSvect::isSame(const AzSvect *inp) const {
  if (row_num != inp->row_num) return false; 

  int ex = 0, ex1 = 0; 
  for (ex = 0; ex < elm_num; ++ex) {
    if (elm[ex].val == 0) continue;  /* ignore zero */
    for ( ; ex1 < inp->elm_num; ++ex1) {
      if (inp->elm[ex1].val == 0) continue; /* ignore zero */
      if (inp->elm[ex1].no != elm[ex].no || 
          inp->elm[ex1].val != elm[ex].val) {
        return false; /* different! */
      }
      break; /* matched */
    }
    if (ex1 >= inp->elm_num) return false; 
    ++ex1; /* matched */
  }
  for ( ; ex1 < inp->elm_num; ++ex1) {
    if (inp->elm[ex1].val != 0) return false;  /* extra non-zero components */
  }
  return true; 
}

/*-------------------------------------------------------------*/
AZint8 AzSmat::nonZeroNum(double *ratio) const {
  if (ratio != NULL) *ratio = 0; 
  AZint8 out = 0; 
  if (column == NULL) return out; 
  for (int cx = 0; cx < col_num; ++cx) if (column[cx] != NULL) out += column[cx]->nonZeroRowNum(); 
  if (ratio != NULL && out != 0) *ratio = (double)out/(double)((double)row_num*(double)col_num); 
  return out; 
}

/*-------------------------------------------------------------*/
int AzSmat::nonZeroColNum() const {
  if (column == NULL) return 0; 
  int num = 0; 
  for (int col = 0; col < col_num; ++col) {
    if (column[col] != NULL && !column[col]->isZero()) ++num; 
  }
  return num; 
}

/*-------------------------------------------------------------*/
void AzSmat ::cap(double cap_val) {
  for (int col = 0; col < col_num; ++col) {
    if (column[col] != NULL) column[col]->cap(cap_val); 
  }
}

/*-------------------------------------------------------------*/
void AzSvect::cap(double cap_val) {
  AzX::throw_if(cap_val <= 0, "AzSvect::cap", "cap value must be non-negative"); 
  for (int ex = 0; ex < elm_num; ++ex) {
    if (elm[ex].val > cap_val) elm[ex].val = (AZ_MTX_FLOAT)cap_val;      
  }
}

/*-------------------------------------------------------------*/
void AzSmat::rbind(const AzSmat *m1) {
  const char *eyec = "AzSmat::rbind"; 
  if (rowNum() == 0 || colNum() == 0) {
    set(m1); 
    return; 
  }
  
  AzX::throw_if(m1->colNum() != colNum(), eyec, "#col mismatch"); 
  int offs = rowNum(); 
  resize(offs+m1->rowNum(), colNum()); 
  for (int cx = 0; cx < colNum(); ++cx) {
    const AzSvect *v0 = col(cx); 
    const AzSvect *v1 = m1->col(cx); 
    AzIFarr ifa; ifa.prepare(v0->nonZeroRowNum()+v1->nonZeroRowNum());
    v0->nonZero(&ifa);  
    AzCursor cur;     
    for ( ; ; ) {
      double val;
      int row = v1->next(cur, val); 
      if (row < 0) break; 
      ifa.put(row+offs, val); 
    }
    col_u(cx)->load(&ifa); 
  }
}

/*-------------------------------------------------------------*/
void AzSmat::cbind(const AzSmat *m1) {
  const char *eyec = "AzSmat::cbind"; 
  if (colNum() <= 0 || rowNum() <= 0) {
    set(m1); 
    return; 
  }
  AzX::throw_if (m1->rowNum() != rowNum(), eyec, "#row mismatch"); 
  int offs = colNum(); 
  resize(m1->rowNum(), offs+m1->colNum()); 
  for (int cx = 0; cx < m1->colNum(); ++cx) {
    col_u(offs+cx)->set(m1->col(cx)); 
  }
}

/*------------------------------------------------------------*/ 
void AzSvect::polarize() {
  if (min() >= 0) {
    resize(row_num*2); 
    return; 
  }

  int org_row_num = row_num; 
  int org_elm_num = elm_num; 
  resize(row_num*2); 
  for (int ex = 0; ex < org_elm_num; ++ex) { 
    if (elm[ex].val < 0) {
      set_inOrder(elm[ex].no+org_row_num, -elm[ex].val); 
      elm[ex].val = 0; 
    }
  }
}

/*-------------------------------------------------------------*/
void AzSmat::binarize() {
  if (column == NULL) return; 
  for (int col = 0; col < col_num; ++col) {
    if (column[col] != NULL) column[col]->binarize(); 
  }
}

/*-------------------------------------------------------------*/
void AzSmat::binarize1() {
  if (column == NULL) return; 
  for (int col = 0; col < col_num; ++col) {
    if (column[col] != NULL) column[col]->binarize1(); 
  }
}

/*-------------------------------------------------------------*/
void AzSvect::binarize() {
  for (int ex = 0; ex < elm_num; ++ex) {
    if (elm[ex].val > 0)      elm[ex].val = 1; 
    else if (elm[ex].val < 0) elm[ex].val = -1; 
  }
}

/*-------------------------------------------------------------*/
void AzSvect::binarize1() {
  for (int ex = 0; ex < elm_num; ++ex) {
    if (elm[ex].val != 0) elm[ex].val = 1; 
  }
}
/*-------------------------------------------------------------*/
bool AzSvect::isOneOrZero() const {
  for (int ex = 0; ex < elm_num; ++ex) {
    if (elm[ex].val != 0 && elm[ex].val != 1) return false; 
  }
  return true; 
}
/*-------------------------------------------------------------*/
bool AzSmat::isOneOrZero() const {
  if (column == NULL) return true; 
  for (int col = 0; col < col_num; ++col) {
    if (column[col] != NULL) {
      bool ret = column[col]->isOneOrZero();
      if (!ret) return false; 
    }
  }
  return true; 
}
/*-------------------------------------------------------------*/
bool AzSmat::isOneOrZero(int col) const {
  if (column == NULL) return true; 
  if (column[col] == NULL) return true;  
  return column[col]->isOneOrZero(); 
}

/*-------------------------------------------------------------*/
/*
 * To write AzSvect from AzSmat, omit #row to save space.   
 * For compatibility, write the size of float (4 | 8).  
*/
/*-------------------------------------------------------------*/
void AzSmat::_read(AzFile *file) {
  const char *eyec = "AzSmat::_read(file)"; 
  AzX::throw_if(col_num > 0 || column != NULL, eyec, "occupied"); 

  /*---  for compatibility  ---*/
  bool is_old = false; 
  int float_size = 0; 
  int marker = file->readInt(); 
  if (marker == -1) { /* new version after 12/14/2014 */
    float_size = file->readInt(); 
    col_num = file->readInt(); 
  }
  else if (marker < 0) {
    AzX::throw_if(true, AzInputError, eyec, "Unknown marker was detected.  The file is not AzSmat."); 
  }
  else { /* old version that did not write marker or float size */
    is_old = true; 
    col_num = marker; 
  }

  /*---  ---*/  
  row_num = file->readInt();  
  if (col_num > 0) {
    a.alloc(&column, col_num, eyec, "column"); 
    if (is_old) _read_cols_old(file); 
    else        _read_cols(file, float_size); 
  }
  dummy_zero.reform(row_num); 
}

/*-------------------------------------------------------------*/
void AzSmat::write(AzFile *file) const {
  file->writeInt(-1);                    /* added 12/15/2014 */
  file->writeInt(sizeof(AZ_MTX_FLOAT));  /* added 12/15 2014 */
  file->writeInt(col_num); 
  file->writeInt(row_num); 
  _write_cols(file); 
}

/*-------------------------------------------------------------*/
void AzSmat::_read_cols_old(AzFile *file) { /* for compatibility */
  for (int cx = 0; cx < col_num; ++cx) {
    column[cx] = AzObjIOTools::read<AzSvect>(file); 
  }
}

/*-------------------------------------------------------------*/
void AzSmat::_read_cols(AzFile *file, int float_size) {
  for (int cx = 0; cx < col_num; ++cx) {
    column[cx] = AzSvect::new_svect(file, float_size, row_num);    
    if (column[cx]->isZero()) {
      delete column[cx]; 
      column[cx] = NULL; 
    }
  }
}

/*-------------------------------------------------------------*/
void AzSmat::_write_cols(AzFile *file) const {
  AzSvect v_zero(row_num); 
  for (int cx = 0; cx < col_num; ++cx) {
    if (column[cx] == NULL) v_zero._write(file, row_num); 
    else                    column[cx]->_write(file, row_num); 
  }  
}
  
/*-------------------------------------------------------------*/
void AzSvect::_write(AzFile *file, int rnum) const {
  if (rnum >= 0) {
    /* save space by not writing #row; caller should be AzSmat */
    AzX::throw_if(row_num != rnum, "AzSvect::write", "#row conflict!"); 
  }
  else {
    file->writeInt(row_num); 
  }
  file->writeInt(elm_num); 
  file->writeBytes(elm, sizeof(elm[0])*elm_num); 
}

/*-------------------------------------------------------------*/
void AzSvect::check_rowno() const {
  for (int ex = 0; ex < elm_num; ++ex) {
    if (elm[ex].no < 0 || elm[ex].no >= row_num) {
      AzBytArr s; s << "#row=" << row_num << ", elm[ex].no=" << elm[ex].no << ", ex=" << ex; 
      AzX::throw_if(true, "AzSvect::check_rowno", "row# is out of range. ", s.c_str()); 
    }
  }
}

/*-------------------------------------------------------------*/
void AzSvect::_read_old(AzFile *file) { /* old version */
  int float_size = 8, rnum = -1; 
  _read(file, float_size, rnum); 
}

/*-------------------------------------------------------------*/
void AzSvect::_read(AzFile *file, int float_size, int rnum) {
  const char *eyec = "AzSvect::_read(file, float_size, #row)"; 
  AzX::throw_if(elm != NULL, eyec, "occupied"); 
  
  if (rnum >= 0) row_num = rnum; 
  else           row_num = file->readInt(); 
  elm_num = file->readInt(); 

  a.alloc(&elm, elm_num, eyec, "elm"); 

  if (elm_num <= 0) return; 

  if (float_size == -1 || float_size == sizeof(AZ_MTX_FLOAT)) {
    file->seekReadBytes(-1, sizeof(elm[0])*elm_num, elm); 
  }
  else if (float_size == 4) { /* file is single-precision */
    AZI_VECT_ELM4 *elm4 = NULL; 
    AzBaseArray<AZI_VECT_ELM4> _a(elm_num, &elm4); 
    file->seekReadBytes(-1, sizeof(elm4[0])*elm_num, elm4); 
    for (int ex = 0; ex < elm_num; ++ex) {
      elm[ex].no = elm4[ex].no; 
      elm[ex].val = elm4[ex].val; 
    }    
  }
  else if (float_size == 8) { /* file is double-precision */
    AZI_VECT_ELM8 *elm8 = NULL; 
    AzBaseArray<AZI_VECT_ELM8> _a(elm_num, &elm8); 
    file->seekReadBytes(-1, sizeof(elm8[0])*elm_num, elm8);
    for (int ex = 0; ex < elm_num; ++ex) {
      elm[ex].no = elm8[ex].no; 
      elm[ex].val = (AZ_MTX_FLOAT)elm8[ex].val; 
    }     
  }
  else {
    AzX::throw_if(true, eyec, "Unexpected floating-point variable size."); 
  }
  if (rnum >= 0) check_rowno(); 
}

/*-------------------------------------------------------------*/
/*-------------------------------------------------------------*/
/* to copy from AzSmatc */
void AzSvect::rawset(const AZI_VECT_ELM *ielm, int ielm_num) {
  const char *eyec = "AzSvect::set(ielm,ielm_num)"; 
  AzX::throw_if(ielm_num < 0, eyec, "ielm_num<0"); 
  a.free(&elm); 
  elm_num = ielm_num; 
  if (elm_num <= 0) return; 
  AzX::throw_if(ielm == NULL, eyec, "null input"); 
  for (int ex = 0; ex < ielm_num; ++ex) {
    AzX::throw_if(ielm[ex].no < 0 || ielm[ex].no >= row_num, eyec, "row# is out of range"); 
    AzX::throw_if(ex > 0 && ielm[ex].no <= ielm[ex-1].no, eyec, "elements are not sorted"); 
  }  
  a.alloc(&elm, ielm_num, eyec); 
  memcpy(elm, ielm, sizeof(elm[0])*ielm_num); 
}

/*-------------------------------------------------------------*/
/*-------------------------------------------------------------*/
void AzSmatc::set(const AzSmatc *m) {
  const char *eyec = "AzSmatc::set(Smatc)"; 
  AzX::throw_if_null(m, eyec);
  reform(m->rowNum(), m->colNum()); 
  int elm_num = (int)m->elmNum(); 
  arr.alloc(&elm, elm_num, eyec); 
  ia_be.reset(&m->ia_be); 
  memcpy(elm, m->elm, elm_num*sizeof(elm[0])); 
} 

/*-------------------------------------------------------------*/
void AzSmatc::set(const AzSmatc *m, const int *cxs, int cxs_num) {
  const char *eyec = "AzSmatc::set(m,cxs,cxs_num)"; 
  AzX::throw_if_null(m, eyec);
  reform(m->rowNum(), cxs_num); 
  int elm_num = 0; 
  for (int ix = 0; ix < cxs_num; ++ix) elm_num += m->col_size(cxs[ix]); 
  
  arr.alloc(&elm, elm_num, eyec); 
  int ex = 0; 
  int ocol; 
  for (ocol = 0; ocol < cxs_num; ++ocol) {
    int icol = cxs[ocol]; 
    const AZI_VECT_ELM *inp = m->rawcol_elm(icol); 
    int inp_num = m->col_size(icol); 
    AzX::throw_if(ex+inp_num > elm_num, eyec, "something is wrong 1"); 
    ia_be(ocol, ex); 
    memcpy(elm+ex, inp, inp_num*sizeof(elm[0])); 
    ex += inp_num; 
  }
  AzX::throw_if(ex != elm_num, eyec, "something is wrong 2"); 
  AzX::throw_if(ocol != col_num, eyec, "something is wrong 3"); 
  ia_be(ocol, ex);   
}  

/*-------------------------------------------------------------*/
/* not tested */
void AzSmatc::set(const AzSmat *m) {
  const char *eyec = "AzSmatc::set(Smat)"; 
  AzX::throw_if_null(m, eyec);
  reform(m->rowNum(), m->colNum()); 
  int elm_num = Az64::to_int((size_t)m->nonZeroNum()); 
  arr.alloc(&elm, elm_num, eyec); 
  int ex = 0; 
  for (int cx = 0; cx < m->colNum(); ++cx) {
    ia_be(cx, ex); 
    int num; const AZI_VECT_ELM *ielm = m->rawcol_elm(cx, &num); 
    for (int ix = 0; ix < num; ++ix) {
      if (ielm[ix].val != 0) elm[ex++] = ielm[ix];  
    }    
  }
  AzX::throw_if(ex != elm_num, eyec, "something is wrong 2"); 
  ia_be(col_num, ex); 
} 

/*-------------------------------------------------------------*/
void AzSmatc::write(AzFile *file) const {
  const char *eyec = "AzSmatc::write(file)";  
  int float_size = sizeof(AZ_MTX_FLOAT); 
  int marker = -1; 
  file->writeInt(marker); 
  file->writeInt(float_size); 
  file->writeInt(col_num); 
  file->writeInt(row_num); 

  for (int cx = 0; cx < col_num; ++cx) {
    int num = col_size(cx); 
    file->writeInt(num); 
    file->writeBytes(rawcol_elm(cx), num*sizeof(elm[0])); 
  }
}

/*-------------------------------------------------------------*/
void AzSmatc::read(AzFile *file) {
  const char *eyec = "AzSmatc::read(file)";  
  reset(); 
  AZint8 sz = file->size(); 
  
  int float_size = 0; 
  int marker = file->readInt();
  AzX::throw_if(marker != -1, AzInputError, eyec, 
                "Unknown marker was detected.  The file is not AzSmat or old AzSmat before 12/14/2014."); 
  float_size = file->readInt();
  col_num = file->readInt();
  row_num = file->readInt();

  int elm_sz = sizeof(AZI_VECT_ELM); 
  if      (float_size == 4) elm_sz = sizeof(AZI_VECT_ELM4); 
  else if (float_size == 8) elm_sz = sizeof(AZI_VECT_ELM8); 
 
  AZint8 offs = file->tell(); 
  AZint8 e_num = (sz - offs - col_num*sizeof(int)) / elm_sz; 
  int elm_num = Az64::to_int(e_num, eyec); 
  arr.alloc(&elm, elm_num, eyec); 

  ia_be.reset(col_num+1, -1);  
  int ex = 0; 
  for (int col = 0; col < col_num; ++col) {   
    ia_be(col, ex); 
    ex = read_svect(file, float_size, ex); 
  }
  ia_be(col_num, ex); 
  if (ex != elm_num) {
    arr.realloc(&elm, ex, "realloc at AzSmatc::read(file) to release extra memory"); 
  }
}

/*-------------------------------------------------------------*/
void AzSmatc::check_rowno(int where, int elm_num) const {
  const char *eyec = "AzSmatc::check_rowno"; 
  for (int ex = where; ex < where+elm_num; ++ex) {
    if (elm[ex].no < 0 || elm[ex].no >= row_num) {
      AzBytArr s; s << "#row=" << row_num << ", elm[ex].no=" << elm[ex].no << ", ex=" << ex; 
      AzX::throw_if(true, eyec, "row# is out of range. ", s.c_str()); 
    }
    if (ex > where && elm[ex].no <= elm[ex-1].no) {
      AzBytArr s; s << "ex=" << ex << " elm[ex-1].no=" << elm[ex-1].no << " elm[ex].no=" << elm[ex].no; 
      AzX::throw_if(true, eyec, "row# is out of order. ", s.c_str()); 
    }
  }
}

/*-------------------------------------------------------------*/
int AzSmatc::read_svect(AzFile *file, int float_size, int where) {
  const char *eyec = "AzSmatc::read_col(file, float_size, ex)"; 

  int elm_num = file->readInt(); 
  if (elm_num <= 0) return where; 

  AzX::throw_if(where+elm_num > arr.size(), eyec, "something is wrong ... ");  
  
  if (float_size == -1 || float_size == sizeof(AZ_MTX_FLOAT)) {
    file->seekReadBytes(-1, sizeof(elm[0])*elm_num, elm+where); 
  }
  else if (float_size == 4) { /* file is single-precision */
    AZI_VECT_ELM4 *elm4 = NULL; 
    AzBaseArray<AZI_VECT_ELM4> _a(elm_num, &elm4); 
    file->seekReadBytes(-1, sizeof(elm4[0])*elm_num, elm4); 
    for (int ex = 0; ex < elm_num; ++ex) {
      elm[where+ex].no = elm4[ex].no; 
      elm[where+ex].val = elm4[ex].val; 
    }    
  }
  else if (float_size == 8) { /* file is double-precision */
    AZI_VECT_ELM8 *elm8 = NULL; 
    AzBaseArray<AZI_VECT_ELM8> _a(elm_num, &elm8); 
    file->seekReadBytes(-1, sizeof(elm8[0])*elm_num, elm8);
    for (int ex = 0; ex < elm_num; ++ex) {
      elm[where+ex].no = elm8[ex].no; 
      elm[where+ex].val = (AZ_MTX_FLOAT)elm8[ex].val; 
    }     
  }
  else {
    AzX::throw_if(true, eyec, "Unexpected floating-point variable size."); 
  }
  check_rowno(where, elm_num); 
  return where+elm_num; 
}

/*-------------------------------------------------------------*/
void AzSmatc::copy_to_smat(AzSmat *ms, const int *cxs, int cxs_num) const {
  ms->reform(row_num, cxs_num); 
  for (int ix = 0; ix < cxs_num; ++ix) {
    int cx = cxs[ix]; 
    ms->col_u(ix)->rawset(rawcol_elm(cx), col_size(cx)); 
  }
}

/*-------------------------------------------------------------*/
/* static */
double AzSvect::first_positive(const AZI_VECT_ELM *elm, int elm_num, int *row) {
  if (row != NULL) *row = -1; 
  if (elm == NULL || elm_num <= 0) return -1; 

  for (int ix = 0; ix < elm_num; ++ix) {
    if (elm[ix].val > 0) {
      if (row != NULL) *row = elm[ix].no; 
      return elm[ix].val;       
    }
  }
  return -1; 
}

/*-------------------------------------------------------------*/
void AzSmatc::check_consistency() {
  const char *eyec = "AzSmatc::check_consistency"; 
  AzX::throw_if(ia_be.size() != col_num+1, eyec, "ia_be.size()!=#col+1");   
  int elm_num = arr.size(); 
  for (int col = 0; col < col_num; ++col) {
    int bx = ia_be[col], ex = ia_be[col+1]; 
    AzX::throw_if(bx < 0 || bx > elm_num, eyec, "begin is out of range"); 
    AzX::throw_if(ex < 0 || ex > elm_num, eyec, "end is out of range");     
    AzX::throw_if(bx > ex, eyec, "begin > end"); 
  }
  for (int ix = 0; ix < elm_num; ++ix) {
    int row = elm[ix].no; 
    AzX::throw_if(row < 0 || row >= row_num, eyec, "row# is out of range");     
  }
}

/*-------------------------------------------------------------*/
/*-------------------------------------------------------------*/
void AzSmatbc::write(AzFile *file, const AzIntArr &ia_no, int row_num, int col_num, const AzIntArr &ia_be) /* static */{
  check_consistency(ia_no, row_num, col_num,  ia_be);
  int version = 1; 
  file->writeInt(version); 
  file->writeInt(row_num); 
  file->writeInt(col_num); 
  ia_be.write(file); 
  ia_no.write(file);   
}

/*-------------------------------------------------------------*/
void AzSmatbc::read(AzFile *file) {
  int version = file->readInt(); 
  row_num = file->readInt(); 
  col_num = file->readInt(); 
  if (version == 0) {
    AzIntArr ia_begin, ia_end; 
    ia_begin.read(file); 
    ia_end.read(file);  
    ia_be.reset(&ia_begin); 
    ia_be.put(ia_end[ia_end.size()-1]); 
  }
  else {
    ia_be.read(file); 
  }
  ia_no.read(file);   
}

/*-------------------------------------------------------------*/
template <class M>  /* M: AzSmat | AzSmatc | AzSmatbc */
void AzSmatbc::set(const M *m, const int *cxs, int cxs_num) {
  const char *eyec = "AzSmatbc::set"; 
  reform(m->rowNum(), cxs_num); 
  AZint8 e_num = 0; 
  for (int ix = 0; ix < cxs_num; ++ix) e_num += m->col_size(cxs[ix]); 
  int elm_num = Az64::to_int(e_num, eyec); 
  ia_no.prepare(elm_num); 
  bool do_elm = !m->is_bc(); 
  for (int ocol = 0; ocol < cxs_num; ++ocol) {
    int col = cxs[ocol]; 
    ia_be(ocol, ia_no.size()); 
    int num = m->col_size(col); 
    if (do_elm) {
      const AZI_VECT_ELM *rawelm = m->rawcol_elm(col); 
      for (int ix = 0; ix < num; ++ix) {
        if (rawelm[ix].val == 0) continue; 
        AzX::throw_if(rawelm[ix].val != 1, AzInputError, eyec, "value <> 1"); 
        ia_no.put(rawelm[ix].no); 
      }
    }
    else {
      ia_no.concat(m->rawcol_int(col), num); 
    }
  }
  ia_be(cxs_num, ia_no.size()); 
}  
template void AzSmatbc::set<AzSmat>(const AzSmat *); 
template void AzSmatbc::set<AzSmatc>(const AzSmatc *);
template void AzSmatbc::set<AzSmatbc>(const AzSmatbc *);

/*-------------------------------------------------------------*/
const int *AzSmatbc::rawcol_int(int col, int *out_num) const {
  check_col(col, "AzSmatbc::rawcol_int"); 
  int pos = ia_be[col], sz = ia_be[col+1] - pos; 
  if (out_num != NULL) *out_num = sz; 
  return ia_no.point() + pos; 
}

/*-------------------------------------------------------------*/
void AzSmatbc::copy_to_smat(AzSmat *m, const int *cxs, int num) const {
  m->reform(row_num, num); 
  for (int ocol = 0; ocol < num; ++ocol) {
    int icol = cxs[ocol]; 
    check_col(icol, "AzSmatbc::copy_to_smat"); 
    AzIFarr ifa; ifa.prepare(ia_be[icol+1]-ia_be[icol]); 
    for (int ix = ia_be[icol]; ix < ia_be[icol+1]; ++ix) ifa.put(ia_no[ix], 1); 
    if (ifa.size() > 0) m->col_u(ocol)->load(&ifa); 
  }  
}  

/*-------------------------------------------------------------*/
double AzSmatbc::first_positive(int col, int *row) const {
  check_col(col, "AzSmatbc::first_positive"); 
  if (row != NULL) *row = -1; 
  for (int ix = ia_be[col]; ix < ia_be[col+1]; ++ix) {
    if (row != NULL) *row = ia_no[ix];  
    return 1; 
  }
  return -1; 
} 

/*-------------------------------------------------------------*/
void AzSmatbc::set(const AzIntArr &_ia_no, const AzIntArr &_ia_be) {
  ia_no.reset(&_ia_no); ia_be.reset(&_ia_be);
  check_consistency(); 
}

/*-------------------------------------------------------------*/
/* static */
void AzSmatbc::check_consistency(const AzIntArr &ia_row, int rnum, int cnum, 
                                 const AzIntArr &ia_be) {
  const char *eyec = "AzSmatbc::check_consistency"; 
  AzX::throw_if(ia_be.size() != cnum+1, eyec, "ia_be.size()!=#col+1");   
  for (int col = 0; col < cnum; ++col) {
    int bx = ia_be[col], ex = ia_be[col+1]; 
    AzX::throw_if(bx < 0 || bx > ia_row.size(), eyec, "begin is out of range"); 
    AzX::throw_if(ex < 0 || ex > ia_row.size(), eyec, "end is out of range");     
    AzX::throw_if(bx > ex, eyec, "begin > end"); 
  }
  for (int ix = 0; ix < ia_row.size(); ++ix) {
    int row = ia_row[ix]; 
    AzX::throw_if(row < 0 || row >= rnum, eyec, "row# is out of range");     
  }
}

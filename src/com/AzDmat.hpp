/* * * * *
 *  AzDmat.hpp 
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

#ifndef _AZ_DMAT_HPP_
#define _AZ_DMAT_HPP_

#include "AzUtil.hpp"
#include "AzStrPool.hpp"
#include "AzSmat.hpp"

//! dense vector 
class AzDvect : public virtual AzReadOnlyVector {
protected:
  int num; 
  double *elm; /* updated only through AzBaseArray functions */
               /* must not be NULL no matter what ... */
  AzBaseArray<double> a; 
  inline void _release() { a.free(&elm); num = 0; }
public:
  #define _AzDvect_init_ num(0), elm(NULL) 
  AzDvect() : _AzDvect_init_ {} 
  AzDvect(int inp_num) : _AzDvect_init_ { reform(inp_num); }
  AzDvect(const double inp[], int inp_num) : _AzDvect_init_ { set(inp, inp_num); }
  template <class T> AzDvect(const T inp[], int inp_num) : _AzDvect_init_ { set(inp, inp_num); }
  AzDvect(const AzDvect *inp) : _AzDvect_init_ { set(inp); }
  inline void set(const AzDvect *inp, double coeff=1) {
    _set(inp->elm, inp->num); 
    if (coeff != 1) multiply(coeff); 
  }
  inline void set_chk(const AzDvect *inp, double coeff=1) {
    AzX::throw_if (num != inp->num, "AzDvect::set_chk", "dimensionality mismatch"); 
    set(inp, coeff); 
  }

  AzDvect(const AzDvect &inp) : _AzDvect_init_ { set(inp.elm, inp.num); }
  AzDvect & operator =(const AzDvect &inp) {
    if (this == &inp) return *this; 
    set(&inp); 
    return *this; 

  }
  AzDvect(const AzSvect *inp) : _AzDvect_init_ { set(inp); }
  void set(const AzSvect *inp);  /* not tested */ 
  ~AzDvect() {}
  AzDvect(AzFile *file) : _AzDvect_init_ { _read(file); }

  void reset() { _release(); }
  inline void reform(int new_row_num) {
    _reform_noset(new_row_num); 
    zeroOut(); 
  }
  inline void reform_chk(int rnum, bool do_chk, const char *msg) {
    AzX::throw_if (do_chk && num != rnum, "AzDvect::reform_chk", msg); 
    reform(rnum); 
  }
  void resize(int new_row_num); 
  
  void write(AzFile *file) const; 
  void write(const char *fn) const { AzFile::write(fn, this); }
  void read(const char *fn) { _release(); AzFile::read(fn, this); }
  void read(AzFile *file)   { _release(); _read(file); }

  inline int size() const { return num; }
  inline int rowNum() const { return num; }  

  inline void destroy() { _release(); }
  bool isZero() const; 
  void load(const AzIFarr *ifa_row_val); 
  void cut(double min_val); 
  void binarize(); 
  void binarize1(); 
  void values(const int exs[], int ex_num, AzIFarr *ifa_ex_value); /* output */
  void nonZero(AzIFarr *ifq) const; 
  void nonZeroRowNo(AzIntArr *iq) const; 
  int nonZeroRowNum() const; 
  void all(AzIFarr *ifa) const {
    AzX::throw_if_null(ifa, "AzDvect::all", "ifa"); 
    ifa->prepare(num); 
    for (int row = 0; row < num; ++row) ifa->put(row, elm[row]); 
  }
  void zeroRowNo(AzIntArr *ia) const {
    AzX::throw_if_null(ia, "AzDvect::zeroRowNo", "ia");     
    ia->reset(); 
    for (int row = 0; row < num; ++row) if (elm[row] == 0) ia->put(row); 
  }
  
  inline void set(int row, double val) {
    check_row(row, "AzDvect::set"); 
    elm[row] = val; 
  }
  inline void set(const double *inp, int inp_num) {
    AzX::throw_if(inp == NULL || inp_num < 0, "AzDvect::set(array)", "Invalid input"); 
    _set(inp, inp_num); 
  }

  inline void set_chk(const double *inp, int inp_num) {
    AzX::throw_if (num != inp_num, "AzDvect::set_chk(array)", "shape mismatch"); 
    set(inp, inp_num); 
  }
  template <class T> 
  inline void set_chk(const T *inp, int inp_num) {
    AzX::throw_if (num != inp_num, "AzDvect::set_chk(array)", "shape mismatch"); 
    set(inp, inp_num); 
  }
  template <class T>
  void set(const T *inp, int inp_num) {
    AzX::throw_if(inp == NULL || inp_num < 0, "AzDvect::set(inp_tmpl, num)", "Invalid input"); 
    if (inp_num != num) _reform_noset(inp_num); 
    for (int ex = 0; ex < num; ++ex) elm[ex] = inp[ex];  
  }
   void set(double val); 

  inline double get(int row) const {
    check_row(row, "AzDvect::get"); 
    return elm[row]; 
  }
  inline double get_nochk(int row) const {
    return elm[row]; 
  }
  inline void add(int row, double val) {
    check_row(row, "AzDvect::add"); 
    if (val != 0) elm[row] += val; 
  }
  inline void add(double val, const AzIntArr *ia_rows) {
    if (ia_rows == NULL) return; 
    add(val, ia_rows->point(), ia_rows->size()); 
  }
  void add_nochk(double val, const AzIntArr *ia_rows) {
    if (ia_rows == NULL) return; 
    add_nochk(val, ia_rows->point(), ia_rows->size()); 
  }
  void add(double val, const int *rows, int rows_num); 
  void add_nochk(double val, const int *rows, int rows_num); 
  void add(double val); 
  void add(const double *inp, int inp_num, double coefficient=1); 
  void add(const AzSvect *vect1, double coefficient=1); 
  inline void add(const AzDvect *vect1, double coefficient=1) {
    AzX::throw_if_null(vect1, "AzDvect::add(vect)");
    add(vect1->elm, vect1->num, coefficient); 
  }

  inline void multiply(int row, double val) {
    check_row(row, "AzDvect::multiply"); 
    if (val != 1) elm[row] *= val; 
  }

  void abs() {
    for (int ex = 0; ex < num; ++ex) if (elm[ex] < 0) elm[ex] = -elm[ex]; 
  }
  double sum() const; 
  double absSum() const; 
  double sum(const int *row, int row_num) const; 
  double absSum(const int *row, int row_num) const; 
  inline double sum(const AzIntArr *ia_rows) const {
    if (ia_rows == NULL) return sum(); 
    return sum(ia_rows->point(), ia_rows->size()); 
  }
  inline double absSum(const AzIntArr *ia_rows) const {
    if (ia_rows == NULL) return absSum(); 
    return absSum(ia_rows->point(), ia_rows->size()); 
  }
  inline double average(const AzIntArr *ia_rows=NULL) const {
    if (num == 0) return 0; 
    if (ia_rows == NULL) return sum() / (double)num; 
    int pop = ia_rows->size(); 
    return sum(ia_rows->point(), pop) / (double)pop; 
  }

  void multiply(double val); 
  inline void divide(double val) {
    AzX::throw_if (val == 0, "AzDvect::divide", "division by zero"); 
    multiply((double)1/val); 
  }

  void scale(const AzDvect *dbles1, bool isInverse=false); 

  double innerProduct(const AzSvect *vect1) const; 
  double innerProduct(const AzDvect *dbles1) const; 
  double selfInnerProduct() const; 
  inline double squareSum() const { return selfInnerProduct(); }

  double normalize(); 
  double normalize1(); 

  inline void zeroOut() {
    for (int ex = 0; ex < num; ++ex) elm[ex] = 0; 
  }
  
  int next(AzCursor &cursor, double &out_val) const; 

  double max(int *out_row = NULL) const;  
  double maxAbs(int *out_row = NULL, double *out_real_val = NULL) const; 

  void max_abs(const AzDvect *v);  /* keep max */
  void add_abs(const AzDvect *v); 

  double min(int *out_row = NULL) const;  
  double max(const AzIntArr *ia_dx, int *out_row = NULL) const;  
  double min(const AzIntArr *ia_dx, int *out_row = NULL) const;  

  void dump(const AzOut &out, const char *header, 
            const AzStrPool *sp_row = NULL, 
            int cut_num = -1) const; 

  inline double *point_u() { return elm; }
  inline const double *point() const { return elm; }

  void square(); 
  static inline double sum(const double val[], int num) {
    if (val == NULL) return 0; 
    double sum = 0; 
    for (int ix = 0; ix < num; ++ix) sum += val[ix]; 
    return sum; 
  }
  static inline bool isNull(const AzDvect *v) {
    if (v == NULL) return true; 
    if (v->rowNum() == 0) return true; 
    return false; 
  }

  void mysqrt(double eps=0) {
    if (elm == NULL) return; 
    for (int ix = 0; ix < num; ++ix) {
      if (elm[ix]<0) {
        if (fabs(elm[ix]) <= eps) {
          elm[ix] = 0; 
          continue; 
        }
        AzX::throw_if(true, "AzDvect::mysqrt", "Negative component"); 
      }
      elm[ix]=sqrt(elm[ix]); 
    }
  }
  static void sdev(const AzDvect *v_avg, const AzDvect *v_avg2, AzDvect *v_sdev) {
    v_sdev->add(v_avg2); 
    AzDvect v(v_avg); v.scale(v_avg); 
    v_sdev->add(&v, -1); /* avg(x^2)-avg(x)^2 */
    double eps = 1e-10; 
    v_sdev->mysqrt(eps); 
  }
  void rbind(const AzDvect *v); 
  void polarize(); 
  bool isSame(const double *inp, int inp_num) const; 
  inline bool isSame(const AzDvect *v) const {
    AzX::throw_if_null(v, "AzDvect::isSame", "null input");
    return isSame(v->elm, v->num); 
  }
  void scale_smat(AzSmat *ms); 
  
protected:
  /*-------------------------------------------------------------*/
  inline void _set(const double *inp, int inp_num) {
    _reform_noset(inp_num); 
    memcpy(elm, inp, sizeof(inp[0]) * num); 
  }
  /*-------------------------------------------------------------*/
  void _reform_noset(int inp_num) { /* must be immediately followed by setting values */
    if (num == inp_num) return; 
    const char *eyec = "AzDvect::_reform_noset"; 
    a.free(&elm); num = 0; 
    AzX::throw_if (inp_num < 0, eyec, "dim must be non-negative"); 
    num = inp_num; 
    if (num > 0) a.alloc(&elm, num, eyec, "elm"); 
  }
  /*-------------------------------------------------------------*/

  void _read(AzFile *file); 
  void _dump(const AzOut &out, const AzStrPool *sp_row, 
             int cut_num = -1) const; 
  inline void check_row(int row, const char *eyec) const {
    AzX::throw_if(row < 0 || row >= num, eyec, "row# is out of range"); 
  }
}; 

//! dense matrix 
class AzDmat : public virtual AzReadOnlyMatrix {
protected:
  bool isLocked; 
  int col_num, row_num; 
  AzDvect **column;  /* updated only through AzPtrArray functions */
                     /* must not be NULL no matter what ... */
  AzObjPtrArray<AzDvect> a; 
  AzDvect dummy_zero; 

  void _release() {
    checkLock("_release"); 
    a.free(&column); col_num = 0; 
    row_num = 0; 
    dummy_zero.reform(0); 
  }
public: 
  #define _AzDmat_init_ row_num(0), col_num(0), column(NULL), isLocked(false)
  AzDmat() : _AzDmat_init_ {}
  AzDmat(int row_num, int col_num) : _AzDmat_init_ { reform(row_num, col_num); }
  AzDmat(const AzDmat *inp) : _AzDmat_init_ { set(inp); }

  /*----------------------------------------------------*/
  inline void set(const AzDmat *m_inp, double coeff=1) {
    AzX::throw_if_null(m_inp, "AzDmat::set(Dmat)");
    _set(m_inp); 
    if (coeff != 1) multiply(coeff); 
  }
  /*----------------------------------------------------*/
  inline void set_chk(const AzDmat *inp, double coeff=1) {
    AzX::throw_if (row_num != inp->row_num || col_num != inp->col_num, 
                   "AzDmat::set_chk", "dimensionality mismatch"); 
    set(inp, coeff); 
  }
  
  AzDmat(const AzDmat &inp) : _AzDmat_init_ { set(&inp); }
  AzDmat & operator =(const AzDmat &inp) {
    if (this == &inp) return *this; 
    set(&inp); 
    return *this; 
  }
  AzDmat(const AzSmat *inp) : _AzDmat_init_ { initialize(inp); }
  void set(const AzSmat *inp) { _release(); initialize(inp); }
  AzDmat(AzFile *file) : _AzDmat_init_ { _read(file); }
  ~AzDmat() {}
  void reset() { _release(); }

  inline void lock()   { isLocked = true; }
  inline void unlock() { isLocked = false; }

  void resize(int new_col_num); 
  void resize(int new_row_num, int new_col_num); 
  inline void reform(int rnum, int cnum) { _reform(rnum, cnum, true); }
  inline void reform_chk(int rnum, int cnum, const char *msg) { reform_chk(rnum, cnum, true, msg); }
  inline void reform_chk(int rnum, int cnum, bool do_chk, const char *msg) {
    AzX::throw_if (do_chk && (row_num != rnum || col_num != cnum), "AzDmat::reform_or_chk", msg); 
    reform(rnum, cnum); 
  }
  void destroy() { _release(); }

  /*--------------------------------------------------*/
  void write(AzFile *file) const; 
  void read(AzFile *file) { _release(); _read(file); }
  void write(const char *fn) const { AzFile::write(fn, this); }
  void read(const char *fn) { _release(); AzFile::read(fn, this); }  

  void convert(AzSmat *m_out) const; 
  inline void convert_to(AzSmat *m_out) const { convert(m_out); }

  void transpose(AzDmat *m_out, int col_begin = -1, int col_end = -1) const; 
  void transpose_to(AzDmat *m_out, int col_begin = -1, int col_end = -1) const { 
    transpose(m_out, col_begin, col_end); 
  }
  void transpose_from(const AzSmat *m_inp); 

  void cut(double min_val); 

  inline void set(double val) {
    for (int col = 0; col < colNum(); ++col) col_u(col)->set(val); 
  }
  int set(const AzDmat *inp, const int *cols, int cnum, bool do_zero_negaindex=false); 
  void set(int col0, int col1, const AzDmat *inp, int icol0=0); /* this[col0:col1-1] <- inp[icol0::(col1-col0)] */
  inline void reduce(const AzIntArr *ia) {
    AzX::throw_if_null(ia, "AzDmat::reduce", "ia"); 
    reduce(ia->point(), ia->size()); 
  }
  void reduce(const int *cols, int cnum); 
  void set(const AzDmat *inp, int col0, int col1) { /* this <- inp[col0:col1-1] */
    AzX::throw_if (col0 < 0 || col1 > inp->col_num || col1-col0 <= 0, "AzDmat::set(m,c0,c1)", "invalid request"); 
    AzIntArr ia; ia.range(col0, col1); 
    set(inp, ia.point(), ia.size()); 
  }
  void set(int row, int col, double val); 
  void add(int row, int col, double val); 
  void add(const AzDmat *inp, double coeff=1); 
  void add(const AzSmat *inp, double coeff=1); 
  void add(double val); 
  void add(const AzDvect *v, double coeff=1) { /* add the vector to each column */
    AzX::throw_if (v->rowNum() != row_num, "AzDmat::add(dvect)", "#row mismatch"); 
    for (int col = 0; col < col_num; ++col) col_u(col)->add(v, coeff); 
  }
    
  void multiply(int row, int col, double val); 
  void multiply(double val); 
  inline void divide(double val) {
    AzX::throw_if (val == 0, "AzDmat::divide", "division by zero"); 
    multiply((double)1/val); 
  }
  
  double get(int row, int col) const; 
  inline double get_nochk(int row, int col) const { /* for speed */
    return column[col]->get_nochk(row); 
  }
  
  /*---  this never returns NULL  ---*/
  inline AzDvect *col_u(int col) {
    check_col(col, "AzDmat::col_u"); 
    if (column[col] == NULL) column[col] = new AzDvect(row_num); 
    return column[col]; 
  }

  /*---  this never returns NULL ---*/
  inline const AzDvect *col(int col) const {
    check_col(col, "AzDmat::col"); 
    if (column[col] == NULL) {
      AzX::throw_if (dummy_zero.rowNum() != row_num, "AzDmat::col", "wrong dummy_zero"); 
      return &dummy_zero; 
    }
    return column[col]; 
  }

  inline int rowNum() const { return row_num; }
  inline int colNum() const { return col_num; }

  void normalize(); 
  void normalize1(); 

  void binarize(); 
  void binarize1(); 

  bool isZero() const; 
  bool isZero(int col) const; 

  void zeroOut(); 

  double max(int *out_row, int *out_col) const; 

  inline void dump(const AzOut &out, const char *header, 
            const AzStrPool *sp_row = NULL, const AzStrPool *sp_col = NULL, 
            int cut_num = -1) const {
    dump(out, header, -1, sp_row, sp_col, cut_num); 
  }
  void dump(const AzOut &out, const char *header, int max_col, 
            const AzStrPool *sp_row = NULL, const AzStrPool *sp_col = NULL, 
            int cut_num = -1) const; 

  void scale(const AzDvect *vect1, bool isInverse=false);
  void scale(const AzDmat *m, bool isInverse=false); 
  void abs() {
    for (int col = 0; col < col_num; ++col) if (column[col] != NULL) column[col]->abs(); 
  }
  
  void square(); 

  inline void load(int col, const AzIFarr *ifa_row_val) {
    col_u(col)->load(ifa_row_val); 
  }

  inline void average(AzDvect *v_avg) const { average_sdev(v_avg, NULL); }
  void average_sdev(AzDvect *v_avg, AzDvect *v_stdev) const; 

  void rbind(const AzDmat *m); 
  void undo_rbind(int added_len); 
  void cbind(const AzDmat *m) {
    const char *eyec = "AzDmat::cbind"; 
    AzX::throw_if_null(m, eyec, "m"); 
    if (colNum() <= 0 || rowNum() <= 0) {
      set(m); 
      return; 
    }
    AzX::throw_if (m->rowNum() != rowNum(), eyec, "shape mismatch"); 
    int org_num = colNum(); 
    int new_num = org_num + m->colNum(); 
    resize(new_num); 
    for (int cx = 0; cx < m->colNum(); ++cx) {
      col_u(org_num+cx)->set(m->col(cx)); 
    }
  }   

  template <class T>
  void set(const T *inp, int inp_num) {
    AzX::throw_if_null(inp, "AzDmat::set(inp_arr,inp_num)");
    AzX::throw_if (inp_num != row_num * col_num, "AzDmat::set(array)", "length conflict"); 
    int cur = 0; 
    for (int col = 0; col < col_num; ++col) {
      column[col]->set_chk(inp+cur, row_num); 
      cur += row_num; 
    }
  }
  double sum() const {
    double val = 0; 
    for (int cx = 0; cx < col_num; ++cx) val += column[cx]->sum(); 
    return val; 
  }
  void sum(AzDvect *v) const {
    AzX::throw_if_null(v, "AzDmat::sum(v)", "v");
    v->reform(row_num); 
    for (int cx = 0; cx < col_num; ++cx) v->add(col(cx)); 
  }
  
  double min() const {
    double mymin = 0; 
    for (int cx = 0; cx < colNum(); ++cx) {
      double colmin = col(cx)->min(); 
      if (cx == 0 || colmin < mymin) mymin = colmin; 
    }
    return mymin; 
  }
  double max() const {
    double mymax = 0; 
    for (int cx = 0; cx < colNum(); ++cx) {
      double colmax = col(cx)->max(); 
      if (cx == 0 || colmax > mymax) mymax = colmax; 
    }
    return mymax; 
  }
  double squareSum() const {
    double sum = 0; 
    for (int cx = 0; cx < col_num; ++cx) sum += col(cx)->selfInnerProduct(); 
    return sum; 
  }
  void prod(const AzDmat *m0, const AzDmat *m1, bool is_m0_tran, bool is_m1_tran); /* this <- op(m0)*op(m1) */
  
protected:
  void check_col(int col, const char *msg="") const {
    AzX::throw_if(col < 0 || col >= col_num, "AzDmat::check_col", msg); 
  } 
  inline void checkLock(const char *who) {
    AzX::throw_if(isLocked, "AzDmat::checkLock", 
            "Illegal attempt to change the pointers of a locked matrix by", who); 
  }
  void _reform(int rnum, int cnum, bool do_zeroOut); 
  void _set(const AzDmat *inp); 
    
  void _read(AzFile *file); 

  /*---  call _release() before calling initialize(...)  ---*/
  /*---  dimensionality may be changed  ---*/
  void initialize(const AzSmat *inp); 
  void _transpose(AzDmat *m_out, int col_begin, int col_end) const; 
}; 

/*-----------------------------------------------------------------------*/
class AzDSmat {
protected: 
  const AzDmat *md; 
  const AzSmat *ms; 
public: 
  AzDSmat() : md(NULL), ms(NULL) {}
  AzDSmat(const AzDmat *md_inp) { reset(md_inp); }
  AzDSmat(const AzSmat *ms_inp) { reset(ms_inp); }
  inline void reset(const AzDmat *md_inp) { md = md_inp; ms = NULL; }
  inline void reset(const AzSmat *ms_inp) { ms = ms_inp; md = NULL; }
  
  inline int rowNum() const { 
    if (md != NULL) return md->rowNum(); 
    if (ms != NULL) return ms->rowNum(); 
    return 0; 
  }
  inline int colNum() const { 
    if (md != NULL) return md->colNum(); 
    if (ms != NULL) return ms->colNum(); 
    return 0; 
  }
  inline void add_to(AzDvect *v_dst, int col, double coeff) const {
    if      (md != NULL) v_dst->add(md->col(col), coeff); /* dst += md[,col]*coeff */
    else if (ms != NULL) v_dst->add(ms->col(col), coeff); 
    else                 AzX::throw_if(true, "AzDSmat::add", "No data"); 
  }
  inline void add_to(AzDmat *m_dst, double coeff) const {
    if      (md != NULL) m_dst->add(md, coeff); /* dst += md*coeff */
    else if (ms != NULL) m_dst->add(ms, coeff); 
    else                 AzX::throw_if(true, "AzDSmat::add", "No data"); 
  }
  inline bool is_ready() const { return (md != NULL || ms != NULL); }
  inline bool is_dense() const { return (md != NULL); }
  inline bool is_sparse() const { return (ms != NULL); }
  inline const AzSmat *sparse() const { 
    AzX::throw_if_null(ms, "AzDSmat::sparse()", "No data"); 
    return ms; 
  }
  inline const AzDmat *dense() const {
    AzX::throw_if_null(md, "AzDSmat::dense()", "No data"); 
    return md;
  }
};

/*********************************************************************/
/*********************************************************************/
class AzDmatc /* c for compact */ {
protected: 
  int row_num, col_num; 
  AZ_MTX_FLOAT *elm; 
  AzBaseArray<AZ_MTX_FLOAT,AZint8> arr; 
public:
  AzDmatc() : row_num(0), col_num(0), elm(NULL) {}
  AzDmatc(int r_num, int c_num) : row_num(0), col_num(0), elm(NULL) { reform(r_num, c_num); }
  AzDmatc(const char *fn) : row_num(0), col_num(0), elm(NULL) { read(fn); }
  void reset() {
    arr.free(&elm, "AzDmatc::reset()"); 
    row_num = col_num = 0; 
  }  
  void destroy() { reset(); }
  void set(const AzSmat *m); 
  void set(const AzDmatc *m) {
    reset(); 
    row_num = m->row_num; col_num = m->col_num; 
    arr.alloc(&elm, size(), "AzDmatc::set"); 
    for (AZint8 ex = 0; ex < size(); ++ex) elm[ex] = m->elm[ex];    
  }
  void copy_to(AzDmat *m) {
    m->reform(row_num, col_num); 
    for (int col = 0; col < col_num; ++col) m->col_u(col)->set(rawcol(col), row_num); 
  }
  
  void write(const char *fn) const { AzFile::write(fn, this); }
  void write(AzFile *file) const {
    int float_sz = sizeof(elm[0]); 
    file->writeInt(float_sz); 
    file->writeInt(row_num); 
    file->writeInt(col_num); 
    file->writeBytes(elm, sizeof(elm[0]), size()); 
  }
  void read(const char *fn) { AzFile::read(fn, this); }   
  void read(AzFile *file) {
    const char *eyec = "AzDmatc::read(file)"; 
    reset(); 
    int float_sz = file->readInt(); 
    if (float_sz != sizeof(elm[0])) {
      AzBytArr s("Float size mismatch.  Expected "); s << (int)sizeof(elm[0]) << "; actual is " << float_sz; 
      throw new AzException(AzInputError, eyec, s.c_str()); 
    }
    row_num = file->readInt(); 
    col_num = file->readInt(); 
    if (row_num <= 0 && col_num <= 0) return; 
    arr.alloc(&elm, size(), eyec); 
    file->seekReadBytes(-1, sizeof(elm[0]), size(), elm); 
  }  
  
  int colNum() const { return col_num; }
  int rowNum() const { return row_num; }
  AZint8 size() const { return (AZint8)row_num*(AZint8)col_num; }
  void reform(int _row_num, int _col_num) {
    const char *eyec = "AzDmatc::reform"; 
    if (row_num < 0 || col_num < 0) throw new AzException(eyec, "#row and #column must be non-negative"); 
    arr.free(&elm); 
    row_num = _row_num; col_num = _col_num;     
    arr.alloc(&elm, size(), eyec); 
    for (int ex = 0; ex < size(); ++ex) elm[ex] = 0; 
  }  
  void copy_from(const AzDmat &m) {
    reform(m.rowNum(), m.colNum()); 
    for (int col = 0; col < col_num; ++col) rawset(col, m.col(col)->point(), row_num); 
  }
  template <class T>
  void rawset(int col, const T *val, AZint8 val_num) {
    const char *eyec = "AzDmatc::rawset"; 
    check_col(col, eyec); 
    if (val_num <= 0) return; 
    AZint8 offs = (AZint8)col*(AZint8)row_num; 
    if (offs + val_num > size()) {
      throw new AzException(eyec, "Too many values"); 
    }
    AZ_MTX_FLOAT *myelm = elm + offs; 
    for (AZint8 ex = 0; ex < val_num; ++ex) myelm[ex] = (AZ_MTX_FLOAT)val[ex];  
  } 
  const AZ_MTX_FLOAT *rawcol(int col, AZint8 *max_num=NULL) const {
    check_col(col, "AzDmatc::rawcol"); 
    if (max_num != NULL) *max_num = (AZint8)(col_num-col)*(AZint8)row_num; 
    return elm+(AZint8)row_num*(AZint8)col; 
  }    
  void writeText(const char *fn, int digits) const {
    AzFile file(fn); 
    file.open("wb"); 
    writeText(&file, digits); 
    file.close(true); 
  }
  void writeText(AzFile *file, int digits) const {
    for (int col = 0; col < col_num; ++col) {
      const AZ_MTX_FLOAT *elm = rawcol(col); 
      AzBytArr s; 
      for (int row = 0; row < row_num; ++row) {
        if (row > 0) s << " "; 
        s.cn(elm[row], digits);
      }        
      s.nl(); 
      s.writeText(file); 
    }
  }
  double first_positive(int col, int *row) const; 
  double min() const; 
  double max() const; 

  void truncate(double min_val, double max_val); 
  void calibrate0(double min_val, double max_val, double eps); 
  
  AzDmatc & operator =(const AzDmatc &inp) {
    if (this == &inp) return *this; 
    set(&inp); 
    return *this; 
  }  
  
protected:
  void check_col(int col, const char *msg="") const {
    AzX::throw_if(col < 0 || col >= col_num, "AzDmatc::check_col", msg); 
  }  
}; 

#include "AzMatVar.hpp"
typedef AzMatVar<AzDmatc> AzDmatcVar; 

#endif 

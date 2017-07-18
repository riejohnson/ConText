/* * * * *
 *  AzSmat.hpp 
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

#ifndef _AZ_SMAT_HPP_
#define _AZ_SMAT_HPP_

#include "AzUtil.hpp"
#include "AzStrPool.hpp"
#include "AzReadOnlyMatrix.hpp"

/*--- 
 * 12/15/2014:                                      
 *  - Changed AZ_MTX_FLOAT to be switchable btw double and float using a compiler option.          
 *  - Changed file format for AzSmat and AzSvect to save disk space.  
 *     NOTE: Old AzSmat file format is still supported.  
 *     NOTE: Old AzSvect file format is NOT supported, as there is no known release that writes AzSvect.
 *  - Added Replaced new AzSvect(..) with AzSvect::new_svect so that the success of new is ensured.    
 ---*/
#ifdef __AZ_SMAT_SINGLE__
typedef float AZ_MTX_FLOAT; 
#else
typedef double AZ_MTX_FLOAT; 
#endif 
#define _checkVal(x) 
/* static double _too_large_ = 16777216; */
/* static double _too_small_ = -16777216; */

typedef struct { int no; AZ_MTX_FLOAT val; } AZI_VECT_ELM; 
typedef struct { int no; float val;  } AZI_VECT_ELM4;
typedef struct { int no; double val; } AZI_VECT_ELM8;  

class AzSmat; 

//! sparse vector 
class AzSvect : /* implements */ public virtual AzReadOnlyVector {
protected:
  int row_num; 
  AZI_VECT_ELM *elm; 
  AzBaseArray<AZI_VECT_ELM> a; 
  int elm_num; 

  void _release() {
    a.free(&elm); elm_num = 0; 
  }
 
  void init_reset(int inp_row_num, bool asDense=false) { 
    initialize(inp_row_num, asDense); 
  }
  void init_reset(const AzSvect *inp) {
    if (inp == NULL) return; 
    row_num = inp->row_num; set(inp); 
  }
  void init_reset(const AzReadOnlyVector *inp) {
    if (inp == NULL) return;   
    row_num = inp->rowNum(); set(inp); 
  }
  void init_reset(AzFile *file, int float_size, int rnum) {
    _read(file, float_size, rnum); 
  }
  
  /*---  new_svect: used only by AzSmat  ---*/
  static AzSvect *new_svect() {
    AzSvect *v = NULL;
    try { v = new AzSvect(); }
    catch (std::bad_alloc &ba) { throw new AzException(AzAllocError, "AzSvect::new_svect()", ba.what()); }
    if (v == NULL) throw new AzException(AzAllocError, "AzSvect::new_svect()", "new");  
    return v; 
  }     
  static AzSvect *new_svect(int _row_num, bool asDense=false) {
    AzSvect *v = new_svect(); v->init_reset(_row_num, asDense); return v; 
  }   
  static AzSvect *new_svect(const AzSvect *inp) {
    AzSvect *v = new_svect(); v->init_reset(inp); return v; 
  } 
  static AzSvect *new_svect(const AzReadOnlyVector *inp) {
    AzSvect *v = new_svect(); v->init_reset(inp); return v; 
  }   
  static AzSvect *new_svect(AzFile *file, int float_no, int rnum) {
    AzSvect *v = new_svect(); v->init_reset(file, float_no, rnum); return v; 
  }

  AzSvect(AzFile *file) /* for compatibility; called only by AzSmat */
    : row_num(0), elm(NULL), elm_num(0) { _read_old(file); }
  
public:
  friend class AzDvect;    /* to speed up set, add, innerProduct */ 
  friend class AzPmatSpa;  /* to speed up set */ 
  friend class AzSmat;     /* to let AzSmat use new_svect and _write */
  friend class AzSmatc;    /* to speed up transfer_from */
  friend class AzObjIOTools; /* for file compatibility so that AzSvect(file) can be called. */
  #define _AzSvect_init_ row_num(0), elm(NULL), elm_num(0) 
  AzSvect() : _AzSvect_init_ {}
  AzSvect(int _row_num, bool asDense=false) : _AzSvect_init_ { init_reset(_row_num, asDense); }    
  AzSvect(const AzSvect *inp) : _AzSvect_init_ { init_reset(inp); }
  AzSvect(const AzReadOnlyVector *inp) : _AzSvect_init_ { init_reset(inp); }
  AzSvect(const AzSvect &inp) : _AzSvect_init_ { init_reset(&inp); }
  AzSvect & operator =(const AzSvect &inp) {
    if (this == &inp) return *this; 
    _release(); 
    row_num = inp.row_num; 
    set(&inp); 
    return *this; 
  }
  ~AzSvect() {}

  void read(AzFile *file) { _release(); _read(file, -1, -1); }
  void resize(int new_row_num); /* new #row must be greater than #row */
  void reform(int new_row_num, bool asDense=false); 
  void change_rowno(int new_row_num, const AzIntArr *ia_old2new, bool do_zero_negaindex=false);  
  void write(AzFile *file) const { _write(file, -1); }

  inline int rowNum() const { return row_num; }  
  void load(const AzIntArr *ia_row, double val, bool do_ignore_negative_rowno=false); 
  void load(const AzIFarr *ifa_row_val); 
  void load(AzIFarr *ifa_row_val) {
    ifa_row_val->sort_Int(true); 
    load((const AzIFarr *)ifa_row_val); 
  }   
  bool isZero() const;
  bool isOneOrZero() const; 

  void cut(double min_val); 
  void only_keep(int num); /* keep this number of components with largest fabs */
  void zerooutNegative(); /* not tested */
  void nonZero(AzIFarr *ifq, const AzIntArr *ia_sorted_filter) const; 

  int nonZeroRowNo() const; /* returns the first one */
  void nonZero(AzIFarr *ifa) const; 
  void all(AzIFarr *ifa) const;  /* not tested */
  void zeroRowNo(AzIntArr *ia) const;  /* not tested */
  void nonZeroRowNo(AzIntArr *intq) const; 
  int nonZeroRowNum() const; 

  void set_inOrder(int row_no, double val); 
  void set(int row_no, double val); 
  void set(double val); 
  void set(const AzReadOnlyVector *vect1, double coefficient=1);  
  
  void rawset(const AZI_VECT_ELM *_elm, int _elm_num);
  
  double get(int row_no) const; 

  double sum() const; 
  double absSum() const; 

  void add(int row_no, double val); 

  void multiply(int row_no, double val); 
  void multiply(double val); 
  inline void divide(double val) {
    AzX::throw_if(val == 0, "AzSvect::divide", "division by zero"); 
    multiply((double)1/val); 
  }
 
  void plus_one_log();  /* x <- log(x+1) */

  void scale(const double *vect1); 

  double selfInnerProduct() const; 
  inline double squareSum() const { return selfInnerProduct(); }
  
  void log_of_plusone(); /* log(x+1) */
  double normalize(); 
  double normalize1(); 
  void binarize();  /* posi -> 1, nega -> -1 */
  void binarize1(); /* nonzero -> 1 */
  
  void clear(); 
  void zeroOut(); 

  int next(AzCursor &cursor, double &out_val) const; 

  double minPositive(int *out_row_no = NULL) const; 
  double min(int *out_row_no = NULL, bool ignoreZero=false) const; 
  double max(int *out_row_no = NULL, bool ignoreZero=false) const; 
  double maxAbs(int *out_row_no = NULL, double *out_real_val = NULL) const; 

  void polarize(); 
  
  void dump(const AzOut &out, const char *header, 
            const AzStrPool *sp_row = NULL, 
            int cut_num = -1) const; 

  void clear_prepare(int num); 
  bool isSame(const AzSvect *inp) const; 
  void cap(double cap_val); 

  static double first_positive(const AZI_VECT_ELM *elm, int elm_num, int *row=NULL); 
  double first_positive(int *row=NULL) const { return first_positive(elm, elm_num, row); }
  
protected:
  void check_row(int row, const char *eyec) const {
    AzX::throw_if(row < 0 || row >= row_num, eyec, "Invalid row#");  
  }
  void _write(AzFile *file, int rnum) const; 
  void check_rowno() const; 
  void _read_old(AzFile *file); /* read old version */
  void _read(AzFile *file, int float_size, int rnum); 

  void initialize(int inp_row_num, bool asDense); 
  int to_insert(int row_no); 
  int find(int row_no, int from_this = -1) const; 
  int find_forRoom(int row_no, 
                   int from_this = -1, 
                    bool *out_isFound = NULL) const; 
  void _dump(const AzOut &out, const AzStrPool *sp_row, int cut_num = -1) const; 
  inline int inc() const {
    return MIN(4096, MAX(32, elm_num)); 
  }
}; 

//! sparse matrix 
class AzSmat : /* implements */ public virtual AzReadOnlyMatrix {
protected:
  int col_num, row_num; 
  AzSvect **column; /* NULL if and only if col_num=0 */
  AzObjPtrArray<AzSvect> a; 
  AzSvect dummy_zero; 
  void _release() {
    a.free(&column); col_num = 0; 
    row_num = 0; 
  }
public: 
  #define _AzSmat_init_ col_num(0), row_num(0), column(NULL)
  AzSmat() : _AzSmat_init_ {}
  AzSmat(int inp_row_num, int inp_col_num, bool asDense=false) : _AzSmat_init_ {
    initialize(inp_row_num, inp_col_num, asDense); 
  }
  AzSmat(const AzSmat *inp) : _AzSmat_init_ { initialize(inp); }
  AzSmat(const AzSmat &inp) : _AzSmat_init_ { initialize(&inp); }
  AzSmat & operator =(const AzSmat &inp) {
    if (this == &inp) return *this; 
    _release(); 
    initialize(&inp); 
    return *this; 
  }
  AzSmat(AzFile *file) : _AzSmat_init_ { _read(file); }
  AzSmat(const char *fn) : _AzSmat_init_ { read(fn); }
  ~AzSmat() {}
  void read(AzFile *file) { _release(); _read(file); }
  void read(const char *fn) { AzFile::read(fn, this); }
  inline void reset() { _release(); }
  void resize(int new_col_num); 
  void resize(int new_row_num, int new_col_num); /* new #row must be greater than #row */
  void reform(int row_num, int col_num, bool asDense=false); 

  void change_rowno(int new_row_num, const AzIntArr *ia_old2new, bool do_zero_negaindex=false); 
  int nonZeroRowNo(AzIntArr *ia_nzrows) const; 
  
  void write(AzFile *file) const; 
  void write (const char *fn) const { AzFile::write(fn, this); }

  bool isZero() const; 
  bool isZero(int col_no) const; 
  bool isOneOrZero() const; 
  bool isOneOrZero(int col_no) const; 
  
  int nonZeroColNum() const; 
  AZint8 nonZeroNum(double *ratio=NULL) const; 
  
  void transpose(AzSmat *m_out, int col_begin = -1, int col_end = -1) const; 
  inline void transpose_to(AzSmat *m_out, int col_begin = -1, int col_end = -1) const {
    transpose(m_out, col_begin, col_end); 
  }
  void cut(double min_val); 
  void only_keep(int num); /* keep this number of components per column with largest fabs */
  void zerooutNegative(); /* not tested */

  void set(const AzSmat *inp); 
  void set(const AzSmat *inp, int col0, int col1); /* this <- inp[,col0:col1-1] */
  int set(const AzSmat *inp, const int *cols, int cnum, bool do_zero_negaindex=false); /* return #negatives in cols */
  int set(const AzSmat *inp, const AzIntArr &ia_cols, bool do_zero_negaindex=false) {
    return set(inp, ia_cols.point(), ia_cols.size(), do_zero_negaindex);  
  }
  void set(int col0, int col1, const AzSmat *inp, int icol0=0); /* this[col0:col1-1] <- inp[icol0::(col1-col0)] */
  inline void reduce(const AzIntArr *ia_cols) { 
    AzX::throw_if_null(ia_cols, "AzSmat::reduce");
    reduce(ia_cols->point(), ia_cols->size());
  }
  void reduce(const int *cols, int cnum);  /* new2old; must be sorted; this <- selected columns of this */
  void set(const AzReadOnlyMatrix *inp);  
  void set(int row_no, int col_no, double val); 
  void set(double val); 

  void add(const AzSmat *m1); 
  void add(int row_no, int col_no, double val); 

  void multiply(int row_no, int col_no, double val); 
  void multiply(double val); 
  inline void divide(double val) {
    AzX::throw_if(val == 0, "AzSmat::divide", "division by zero"); 
    multiply((double)1/val); 
  }

  void plus_one_log();  /* x <- log(x+1) */  
  double get(int row_no, int col_no) const; 

  double sum() const {
    if (column == NULL) return 0; 
    double mysum = 0; 
    for (int col = 0; col < col_num; ++col) {
      if (column[col] != NULL) mysum += column[col]->sum(); 
    }
    return mysum; 
  }
  double sum(int col) const {
    check_col(col, "AzSmat::sum(col)"); 
    if (column == NULL) return 0; 
    if (column[col] == NULL) return 0; 
    return column[col]->sum(); 
  }
  
  /*---  this never returns NULL  ---*/
  inline const AzSvect *col(int col_no) const {
    check_col(col_no, "AzSmat::col"); 
    if (column[col_no] == NULL) {
      AzX::throw_if (dummy_zero.rowNum() != row_num, "AzSmat::col", "#col of dummy_zero is wrong"); 
      return &dummy_zero; 
    }
    return column[col_no]; 
  }

  /*---  this never returns NULL  ---*/
  inline AzSvect *col_u(int col_no) {
    check_col(col_no, "AzSmat::col_u"); 
    if (column[col_no] == NULL) column[col_no] = AzSvect::new_svect(row_num); 
    return column[col_no]; 
  }
  inline int rowNum() const { return row_num; }
  inline int colNum() const { return col_num; }
  inline int dataNum() const { return col_num; }
  inline int size() const { return row_num*col_num; }
  
  void log_of_plusone(); /* log(x+1) */
  void normalize(); 
  void normalize1(); 
  void binarize();  /* posi -> 1, nega -> -1 */
  void binarize1(); /* nonzero -> 1 */
  
  inline void destroy() {
    reform(0,0); 
  }
  inline void destroy(int col) {
    if (col >= 0 && col < col_num && column != NULL) {
      delete column[col]; 
      column[col] = NULL; 
    } 
  }

  inline void load(int col, AzIFarr *ifa_row_val) {
    col_u(col)->load(ifa_row_val); 
  }

  void clear(); 
  void zeroOut(); 

  int next(AzCursor &cursor, int col, double &out_val) const; 

  double max(int *out_row=NULL, int *out_col=NULL, bool ignoreZero=false) const;
  double min(int *out_row=NULL, int *out_col=NULL, bool ignoreZero=false) const; 

  void dump(const AzOut &out, const char *header, 
            const AzStrPool *sp_row = NULL, const AzStrPool *sp_col = NULL, 
            int cut_num = -1) const; 

  bool isSame(const AzSmat *inp) const; 

  inline static bool isNull(const AzSmat *inp) {
    if (inp == NULL) return true; 
    if (inp->col_num == 0) return true; 
    if (inp->row_num == 0) return true; 
    return false; 
  }
  void cap(double cap_val); 

  void rbind(const AzSmat *m1); 
  void cbind(const AzSmat *m1); 

  /*---  to match smatc  ---*/                     
  int col_size(int cx) const {
    check_col(cx, "AzSmat::col_size"); 
    if (column == NULL || column[cx] == NULL) return 0; 
    return column[cx]->elm_num; 
  }
  bool is_bc() const { return false; }
  const AZI_VECT_ELM *rawcol_elm(int cx, int *out_num=NULL) const {
    check_col(cx, "AzSmat::rawcol_elm"); 
    int num = col_size(cx); 
    if (out_num != NULL) *out_num = num; 
    if (num > 0) return column[cx]->elm; 
    return NULL;    
  }
  const int *rawcol_int(int cx, int *out_num=NULL) const { AzX::no_support(true, "AzSmat::rawcol_int", "rawcol_int"); return NULL; }
  AZint8 elmNum() const {
    if (column == NULL) return 0; 
    AZint8 e_num = 0; 
    for (int col = 0; col < col_num; ++col) e_num += col_size(col); 
    return e_num; 
  }
  void copy_to_smat(AzSmat *ms) const { 
    AzX::throw_if_null(ms, "AzSmat::copy_to_smat");   
    ms->set(this); 
  }
  void copy_to_smat(AzSmat *ms, const int *cxs, int cxs_num) const {
    AzX::throw_if_null(ms, "AzSmat::copy_to_smat(cols)"); 
    ms->set(this, cxs, cxs_num); 
  }
  double first_positive(int col, int *row=NULL) const { 
    return AzSvect::first_positive(rawcol_elm(col), col_size(col), row); 
  }
                     
protected:
  void check_col(int col, const char *eyec) const {
    AzX::throw_if(col < 0 || col >= col_num, eyec, "Invalid col#");  
  }
  void _read(AzFile *file); 
  void initialize(int row_num, int col_num, bool asDense); 
  void initialize(const AzSmat *inp); 
  void _transpose(AzSmat *m_out, int col_begin, int col_end) const; 

  void _read_cols_old(AzFile *file); /* for compatibility */
  void _read_cols(AzFile *file, int float_size); 
  void _write_cols(AzFile *file) const;   
}; 

#include "AzMatVar.hpp"
typedef AzMatVar<AzSmat> AzSmatVar; 

/*********************************************************************/
/*         to generate AzSmatbc or AzSmatc efficiently ...           */
/*********************************************************************/
template <class ValArr>  /* ValArr: AzIntArr | AzValArr<AZI_VECT_ELM> */
class Az_bc_c {
protected: 
  AzIntArr ia_be; 
  ValArr arr; 
  bool is_committed; 
  void check_col(int col, const char *eyec) const {
    AzX::throw_if(col<0 || col>=colNum(), eyec, "col# is out of range");  
  }
  void check_commit(const char *eyec) const {
    AzX::throw_if(!is_committed, eyec, "Not committed");  
  }
public:
  Az_bc_c() : is_committed(false) {}
  Az_bc_c(int ini_be, int ini_no) : is_committed(false) { reset(ini_be, ini_no); }
  void reset(int ini_be, int ini_no) {
    is_committed = false; 
    ia_be.reset(); ia_be.prepare(ini_be); 
    arr.reset();   arr.prepare(ini_no);     
  }
  void destroy() { ia_be.reset(); arr.reset(); is_committed = false; }
  int size(int col) { 
    check_col(col, "Az_bc_c::size(col)"); 
    return _end(col)-_beg(col); 
  }
  const AzIntArr &be() const { 
    check_commit("Az_bc_c::be"); 
    return ia_be; 
  }
  const ValArr &valarr() const { 
    check_commit("Az_bc_c::valarr"); 
    return arr; 
  }
  void remove_col(int col) {
    check_col(col, "remove_col"); 
    AzIntArr ia_rmv; ia_rmv.range(_beg(col), _end(col)); 
    arr.remove(ia_rmv.point(), ia_rmv.size());
    AzIntArr ia(ia_be.point()+col+1, ia_be.size()-col-1); 
    ia_be.cut(col); ia_be.concat(&ia, -ia_rmv.size()); 
  }
  void check_index_order() const {
    for (int col = 1; col < ia_be.size(); ++col) {
      AzX::throw_if(ia_be[col]<ia_be[col-1], "Az_bc_c::check_index_order", "not ascending"); 
    }
  }
  void put(const ValArr &_arr) {
    ia_be.put(arr.size()); 
    arr.concat(_arr);     
  }  
  void unique_put(ValArr &_arr) {
    _arr.unique(); 
    ia_be.put(arr.size()); 
    arr.concat(_arr);     
  }
  int colNum() const { return ((is_committed) ? ia_be.size()-1 : ia_be.size()); }
  int elmNum() const { return arr.size(); }
  void commit() {
    AzX::throw_if(is_committed, "Az_bc::commit", "Already committed"); 
    ia_be.put(arr.size()); 
    is_committed = true;       
  }
  bool isCommitted() const { return is_committed; }
  void prepmem(AZint8 now, AZint8 all) {
    prepmem(ia_be, now, all); 
    prepmem(arr, now, all);  
  }
  template <class Arr>
  static int prepmem(Arr &arr, AZint8 now, AZint8 all) {
    int est = (int)MIN((AZint8)arr.size() / now * all + 1024*1024, AzSigned32Max);     
    arr.prepare(est); 
    return est; 
  }  
  void check_overflow(const char *eyec, int num, int data_no=-1) {
    if ((AZint8)arr.size() + (AZint8)num > AzSigned32Max) { 
      AzBytArr s("Detected index overflow.  Going over the 2GB limit.  "); 
      s << "Divide the input data into batches."; 
      if (data_no >= 0) s << "  data#=" << data_no; 
      AzX::throw_if(true, AzInputError, eyec, s.c_str()); 
    }   
  }
protected: 
  int _beg(int col) { return ia_be[col]; }
  int _end(int col) { return (col+1 < ia_be.size()) ? ia_be[col+1] : arr.size(); }
}; 
typedef Az_bc_c<AzIntArr> Az_bc; 
typedef Az_bc_c< AzValArr<AZI_VECT_ELM> > Az_c; 

/*********************************************************************/
/* sparse matrix with compact format; no update is allowed           */
/* NOTE: The number of non-zero elements must not exceed 2GB,        */
class AzSmatc {
protected: 
  int row_num, col_num; 
/*  AzIntArr ia_begin, ia_end; */
  AzIntArr ia_be; /* 6/3/2017 */  
  AZI_VECT_ELM *elm; 
  AzBaseArray<AZI_VECT_ELM> arr;   

public:
  AzSmatc() : row_num(0), col_num(0), elm(NULL) {}
  AzSmatc(int _row_num, int _col_num) 
            : row_num(0), col_num(0), elm(NULL) {
    reform(_row_num, _col_num);               
  }
  AzSmatc(const char *fn) : row_num(0), col_num(0), elm(NULL) {
    read(fn);  
  }
  void reset() {
    ia_be.reset(); 
    arr.free(&elm, "AzSmatc::reset()"); 
  }  
  void destroy() { reset(); }
  
  void write(const char *fn) const { AzFile::write(fn, this); }
  void write(AzFile *file) const; /* read in the same format as AzSmat */
  void read(const char *fn) { AzFile::read(fn, this); }  
  void read(AzFile *file); /* write in the same format as AzSmat */

  int colNum() const { return col_num; }
  int rowNum() const { return row_num; }
  int dataNum() const { return col_num; }  
  AZint8 elmNum() const { return (AZint8)arr.size(); }
  int col_size(int cx) const {
    check_col(cx); 
    return ia_be[cx+1]-ia_be[cx]; 
  }
  bool is_bc() const { return false; }
  const AZI_VECT_ELM *rawcol_elm(int cx, int *num=NULL) const {
    check_col(cx); 
    if (num != NULL) *num = (ia_be[cx+1]-ia_be[cx]); 
    return elm+ia_be[cx];     
  }  
  const int *rawcol_int(int cx, int *out_num=NULL) const { AzX::no_support(true, "AzSmatc::rawcol_int", "rawcol_int"); return NULL; }
  void reform(int rnum, int cnum) {
    AzX::throw_if(rnum < 0 || cnum < 0, "AzSmatc::reform", "#row and #col must be non-negative"); 
    arr.free(&elm); 
    row_num = rnum; col_num = cnum; 
    ia_be.reset(col_num+1, 0); 
  }
  
  void check_consistency(); 
  void set(int rnum, const Az_c &inp_c) { set(rnum, inp_c.valarr(), inp_c.be()); }  
  void set(int rnum, const AzValArr<AZI_VECT_ELM> &_arr, const AzIntArr &_ia_be) {
    const char *eyec = "AzSmatc::set(int,ValArr<AZI_VECT_ELM>,IntArr &)"; 
    row_num = rnum; col_num = _ia_be.size()-1; 
    AzX::throw_if(row_num < 0 || col_num < 0, "AzSmatc::set", "Negative #row|#col?"); 
    ia_be.reset(&_ia_be); 
    int sz = _arr.size(); 
    arr.free(&elm, eyec); arr.alloc(&elm, sz, eyec); 
    memcpy(elm, _arr.point(), sizeof(AZI_VECT_ELM)*sz); 
    check_consistency(); 
  }
  static void write(AzFile *file, int rnum, const Az_bc &bc) { /* for seq2-bown */
    AzSmatc mc; mc.set(rnum, bc); mc.write(file); 
  }
  void set(int rnum, const Az_bc &inp_bc) { set(rnum, inp_bc.valarr(), inp_bc.be()); }
  void set(int rnum, const AzIntArr &_ia_no, const AzIntArr &_ia_be) { /* for seq2-bown */
    const char *eyec = "AzSmatc::set(int,IntArr &,IntArr &)"; 
    row_num = rnum; col_num = _ia_be.size()-1; 
    AzX::throw_if(row_num < 0 || col_num < 0, "AzSmatc::set", "Negative #row|#col?"); 
    ia_be.reset(&_ia_be); 
    int sz = _ia_no.size(); 
    arr.free(&elm, eyec); arr.alloc(&elm, sz, eyec); 
    for (int ex = 0; ex < sz; ++ex) { elm[ex].no = _ia_no[ex]; elm[ex].val = 1; }
    check_consistency(); 
  }     
  void set(const AzSmat *m);  /* not tested */
  void set(const AzSmatc *m);
  void set(const AzSmatc *m, const int *cxs, int cxs_num); 
  
  void copy_to_smat(AzSmat *ms, const int *cxs, int cxs_num) const;
  void copy_to_smat(AzSmat *ms) const {
    AzIntArr ia; ia.range(0, col_num); 
    copy_to_smat(ms, ia.point(), ia.size());     
  }
  double first_positive(int col, int *row=NULL) const { 
    return AzSvect::first_positive(rawcol_elm(col), col_size(col), row); 
  }
  double min() const {
    if (row_num <= 0 || col_num <= 0) return -1; 
    double val = ((AZint8)row_num*(AZint8)col_num > arr.size()) ? 0 : elm[0].val; 
    for (int ex = 0; ex < arr.size(); ++ex) val = MIN(val, elm[ex].val); 
    return val; 
  }
  double max() const {
    if (row_num <= 0 || col_num <= 0) return -1;     
    double val = ((AZint8)row_num*(AZint8)col_num > arr.size()) ? 0 : elm[0].val; 
    for (int ex = 0; ex < arr.size(); ++ex) val = MAX(val, elm[ex].val); 
    return val; 
  }   
  void multiply(double val) {
    for (int ix = 0; ix < elmNum(); ++ix) elm[ix].val *= (AZ_MTX_FLOAT)val;  
  }
  void binarize() {
    for (int ix = 0; ix < elmNum(); ++ix) {
      AZ_MTX_FLOAT val = elm[ix].val; 
      elm[ix].val = (AZ_MTX_FLOAT)((val>0) ? 1 : ((val<0) ? -1 : 0)); 
    }
  }
  AzSmatc & operator =(const AzSmatc &inp) {
    if (this == &inp) return *this; 
    set(&inp); 
    return *this; 
  }
  
protected:  
  void check_rowno(int where, int elm_num) const; 
  int read_svect(AzFile *file, int float_size, int where); 
  void check_col(int col) const {
    AzX::throw_if(col<0 || col>=col_num, "AzSmatc::check_col", "col# is out of range");  
  }
}; 

#include "AzMatVar.hpp"
typedef AzMatVar<AzSmatc> AzSmatcVar; 

/*-------------------------------------------------------------*/
class AzSmatbc {
protected: 
  int row_num, col_num; 
/*  AzIntArr ia_begin, ia_end; */
  AzIntArr ia_be; /* 6/2/2017 */
  AzIntArr ia_no; 
public:   
  AzSmatbc() : row_num(0), col_num(0) {}
  AzSmatbc(int rnum, int cnum) : row_num(0), col_num(0) { reform(rnum, cnum); }
  void reform(int rnum, int cnum) {
    AzX::throw_if(row_num < 0 || col_num < 0, "AzSmatbc::reform", "#row and #cold must be non-negative");     
    row_num = rnum; col_num = cnum; 
    ia_be.reset(col_num+1,0); ia_no.reset();
  }  
  void set(const AzIntArr &_ia_no, const AzIntArr &_ia_be); 
  void check_consistency() const { check_consistency(ia_no, row_num, col_num,  ia_be); }
  static void check_consistency(const AzIntArr &ia_no, int rnum, int cnum, const AzIntArr &ia_be); 
  void reset() { reform(0,0); }
  void destroy() { reform(0,0); }
  void write(AzFile *file) const { write(file, ia_no, row_num, col_num, ia_be); }
  static void write(AzFile *file, const AzIntArr &ia_no, int row_num, int col_num, AzIntArr const &ia_be); 
  static void write(AzFile *file, int row_num, const Az_bc &bc) {
    write(file, bc.valarr(), row_num, bc.colNum(), bc.be()); 
  }
  void read(AzFile *file); 
  void write(const char *fn) const { AzFile::write<AzSmatbc>(fn, this); }
  void read(const char *fn) { AzFile::read<AzSmatbc>(fn, this); }
  int rowNum() const { return row_num; }
  int colNum() const { return col_num; }
  int col_size(int col) const { check_col(col, "AzSmatbc::col_size"); return (ia_be[col+1]-ia_be[col]); }
  bool is_bc() const { return true; }
  const AZI_VECT_ELM *rawcol_elm(int col, int *out_num=NULL) const { AzX::no_support(true, "AzSmatbc::rawcol", "rawcol"); return NULL; }
  const int *rawcol_int(int col, int *out_num=NULL) const; 
  AZint8 elmNum() const { return (AZint8)ia_no.size(); }
  AZint8 nonZeroNum() const { return (AZint8)ia_no.size(); }  
  template <class M> void set(const M *m) { AzIntArr ia; ia.range(0, m->colNum()); set(m, ia.point(), ia.size()); }
  template <class M> void set(const M *m, const int *cxs, int cxs_num); 
  void copy_to_smat(AzSmat *m) const {
    AzIntArr ia; ia.range(0, col_num); 
    copy_to_smat(m, ia.point(), ia.size()); 
  }
  void copy_to_smat(AzSmat *m, const int *cxs, int num) const;   
  double min() const {
    if (elmNum() < (AZint8)row_num*(AZint8)col_num) return 0; 
    return 1; 
  }
  double max() const {
    if (elmNum() > 0) return 1; 
    return 0;     
  }
  double first_positive(int col, int *row=NULL) const; 
  
protected:
  void check_col(int col, const char *eyec) const {
    AzX::throw_if(col<0 || col>=col_num, eyec, "col# is out of range"); 
  }  
}; 
typedef AzMatVar<AzSmatbc> AzSmatbcVar; 
#endif 
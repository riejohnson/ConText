/* * * * *
 *  AzPmatSpa.hpp
 *  Copyright (C) 2015-2017 Rie Johnson
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

#ifndef _AZ_PMAT_SPA_HPP_
#define _AZ_PMAT_SPA_HPP_

#include "AzSmat.hpp"
#include "AzDmat.hpp"
#include "AzPmat.hpp"
#include "AzParam.hpp"
#include "AzPrint.hpp"

#ifdef __AZ_GPU__
#include "AzPmatSpa_gpu.cuh"
#else
#include "AzPmatSpa_cpu.hpp"  
#endif

class AzPmatSpa_flags {
public:
  bool do_cu_up, do_cu_dw, do_cu_x; 
  AzPmatSpa_flags(); 
  static void resetParam(AzParam &azp); 
  static void printParam(AzPrint &o); 
  static void printHelp(AzHelp &h); 
}; 

/***************************************************************/  
class AzPmatSpa {
protected:
  AzPmatSpa_flags f; 
  int row_num, col_num; 
  _AzParr<AzFloat> csc_vals; 
  _AzParr<int> csc_ptrs;  
  _AzParr<int> csc_rows; 
  _AzParr<int> csc_cols; /* in the order of csc_vals */
  
  _AzParr<AzFloat> csr_vals; 
  _AzParr<int> csr_ptrs;  
  _AzParr<int> csr_cols;
  
  _AzParr<int> nzptrs; /* pointing csr_vals */
  _AzParr<int> nzrows; /*                   */

  _AzPmatSpa u; 
  _AzPmat uu; 
  
public:
  AzPmatSpa() : row_num(0), col_num(0) {}
  AzPmatSpa(const AzPmatSpa_flags &_f) : row_num(0), col_num(0) { f = _f; }

  void add_to(AzPmat *m_dst, double coeff) const; 
  void sub_from(AzPmat *m_dst) const {
    add_to(m_dst, -1); 
  }

  void prod_dense1_sparse0(AzPmat *m_dst, const AzPmat *md, bool do_add) const {
    if (f.do_cu_up) {
      AzPmat m; 
      _prod_sparse1_dense0_cu(&m, md, do_add); 
      m_dst->transpose_from(&m);        
    }
    else _prod_dense1_sparse0_nocu(m_dst, md, do_add); 
  }

  void prod_sparse0_dense1(AzPmat *m_dst, const AzPmat *md, double alpha, bool do_add) const {
    if (f.do_cu_dw) {
      AzPmat md_tran; md_tran.transpose_from(md); 
      _prod_sparse0_dense0_cu_nz(m_dst, &md_tran, (AzFloat)alpha, do_add); 
    }
    else _prod_sparse0_dense1_nocu(m_dst, md, (AzFloat)alpha, do_add);  /* faster than cusparse */
  }
  
  /*---  cusparse  ---*/
  void _prod_sparse1_dense0_cu(AzPmat *m_dst, const AzPmat *md, bool do_add) const; 
  void _prod_sparse0_dense0_cu_all(AzPmat *m_dst, const AzPmat *md, AzFloat alpha, bool do_add) const; 
  void _prod_sparse0_dense0_cu_nz(AzPmat *m_dst, const AzPmat *md, AzFloat alpha, bool do_add) const; 
#ifdef __AZ_CSRMM2__
  void _prod_sparse_dense_cu(AzPmat *m_dst, const AzPmat *md, 
                            bool do_tran_s, bool do_tran_d, bool do_add) const; 
#endif   
  /*---  non-cusparse  ---*/  
  void _prod_dense1_sparse0_nocu(AzPmat *m_dst, const AzPmat *md, bool do_add) const; 
  void _prod_sparse0_dense1_nocu(AzPmat *m_dst, const AzPmat *md, AzFloat alpha, bool do_add) const; 
  
  /*---  ---*/
  inline bool is_row_indexed() const {
    return (csr_ptrs.size() == row_num + 1);  
  }
  void gen_row_index(bool do_gen_row_index, AzIIFarr *iifa=NULL); 
  
  inline int rowNum() const { return row_num; }
  inline int colNum() const { return col_num; }
  inline int size() const { return row_num*col_num; }
  void reform(int rnum, int cnum) {
    if (row_num == rnum && col_num == cnum) {
      _zeroOut(); 
    }
    else {
      row_num = rnum; 
      col_num = cnum;     
      _zeroOut(); 
    }
  }
  inline void destroy() { reform(0,0); }  
  inline void reset() { reform(0,0); } 

  template <class M> /* M: AzSmatc | AzSmat */
  void set(const M *ms, const int *cxs, int cxs_num, bool do_gen_row_index); 
  void set(const AzDataArr<AzIFarr> &aifa, int r_num, 
           const AzIntArr *ia_row_old2new, /* may be NULL */
           const int *cxs, int cxs_num, bool do_gen_row_index); 
  
  template <class M> /* M: AzSmatc | AzSmat */
  void set(const M *ms, bool do_gen_row_index) {
    AzIntArr ia; ia.range(0, ms->colNum()); 
    set(ms, ia.point(),  ia.size(), do_gen_row_index); 
  }
  template <class M> /* M: AzSmatc | AzSmat */
  void set(const M *ms, int col0, int col1, bool do_gen_row_index) {
    AzIntArr ia; ia.range(col0, col1); 
    set(ms, ia.point(),  ia.size(), do_gen_row_index); 
  }  
  void set(const AzDataArr<AzIFarr> &aifa, int r_num, 
           const AzIntArr *ia_row_old2new, /* may be NULL */
           bool do_gen_row_index) {
    AzIntArr ia; ia.range(0, aifa.size()); 
    set(aifa, r_num, ia_row_old2new, ia.point(), ia.size(), do_gen_row_index); 
  }             
  
  void check_consistency() const;  

  /*---  ---*/
  AzPmatSpa(const AzPmatSpa &inp) {
    AzX::throw_if(true, "AzPmatSpa(const&)", "= is prohibited");   
  }
  void set(const AzPmatSpa *inp) {
    row_num = inp->row_num; col_num = inp->col_num; 
    csc_vals.reset(&inp->csc_vals); 
    csc_ptrs.reset(&inp->csc_ptrs); 
    csc_rows.reset(&inp->csc_rows); 
    csc_cols.reset(&inp->csc_cols);     
    csr_vals.reset(&inp->csr_vals); 
    csr_ptrs.reset(&inp->csr_ptrs); 
    csr_cols.reset(&inp->csr_cols);
    nzptrs.reset(&inp->nzptrs); 
    nzrows.reset(&inp->nzrows);     
  }
  AzPmatSpa & operator =(const AzPmatSpa &inp) {
    AzX::throw_if(true, "AzPmatSpa operator =", "= is prohibited");     
  }  
  void dropout(double dout, bool is_test, AzPrng &rng, bool do_scale); 
  
protected:   
  void show(const char *header, 
            const _AzParr<AzFloat> &vals, const _AzParr<int> &ptrs, const _AzParr<int> &inds) const; 
  void _gen_row_index(AzIIFarr *iifa); 
  void _zeroOut() {
    csc_vals.free(); 
#if 0     
    AzIntArr ia; ia.reset(col_num+1, 0);        
    csc_ptrs.reset_from_host(ia.point(), ia.size(), "AzPmatSpa::_zeroOut", "csc_ptrs"); 
#else
    csc_ptrs.reset_with_zero(col_num+1, "AzPmatSpa::_zeroOut", "csc_ptrs (reset_with_zero)"); 
#endif  
    csc_rows.free(); 
    csc_cols.free(); 
    
    csr_vals.free(); csr_ptrs.free(); csr_cols.free(); 
    
    nzptrs.free(); nzrows.free(); 
  }
  void _set_csc_csr(const AzFloat *hvals, int nz_num, 
                    const AzIntArr &ia_ptrs, const AzIntArr &ia_rows, const AzIntArr &ia_cols, 
                    bool do_gen_row_index, AzIIFarr &iifa); 
}; 


/***************************************************************/  
class AzPmatSpaVar {
protected:
  int data_num; 
  AzPmatSpa m; /* row: #channel, col: item (like a patch of pixels) */
  AzIntArr ia_dcolind; /* column range in m: begin1, end1, begin2, end2, ... */
  AzPintArr pia_dcolind; 
public:
  friend class AzPmatApp; 

  AzPmatSpaVar() : data_num(0) {}
  AzPmatSpaVar(const AzPmatSpa_flags &_f) : data_num(0), m(_f) {}
  inline int dataNum() const { return data_num; }
  inline int rowNum() const { return m.rowNum(); }
  inline int colNum() const { return m.colNum(); }  
  void reset() {
    data_num = 0; 
    ia_dcolind.reset(); 
    pia_dcolind.reset(); 
    m.reset();     
  }
  inline void destroy() { reset(); }
  template <class M> /* M: AzSmatVar | AzSmatcVar | AzSmatbcVar */
  void set(const M *mv0, const int *dx0s, int dx0s_num, bool do_gen_rowindex); /* this[0::dxs_num] <- selected data of mv0 */   
  template <class M> void set(const M &mv0, bool do_gen_rowindex) {
    AzIntArr ia; ia.range(0, mv0.dataNum()); 
    set(&mv0, ia.point(), ia.size(), do_gen_rowindex); 
  }
  void set(const AzSmat &ms, const bool do_gen_rowindex) {
    AzSmatVar msv; msv.reset(&ms); 
    AzIntArr ia_dxs; ia_dxs.range(0, msv.dataNum()); 
    set(&msv, ia_dxs.point(), ia_dxs.size(), do_gen_rowindex); 
  }
  void set(const AzSmat &ms, const AzIntArr &ia_ind, const bool do_gen_rowindex) {
    AzSmatVar msv; msv.reset(&ms, &ia_ind); 
    AzIntArr ia_dxs; ia_dxs.range(0, msv.dataNum()); 
    set(&msv, ia_dxs.point(), ia_dxs.size(), do_gen_rowindex); 
  }  
  void set(const AzPmatSpa &_m) { /* regard one column as one data point */
    m.set(&_m);  
    data_num = m.colNum(); 
    ia_dcolind.reset(); 
    for (int col = 0; col < data_num; ++col) { ia_dcolind.put(col); ia_dcolind.put(col+1); }
    pia_dcolind.reset(&ia_dcolind); 
  } 
  const AzPmatSpa *data() const { return &m; }
  AzPmatSpa *data_u() { return &m; }  /* use this with caution. */
  void check_data_consistency(const char *eyec); 
  void set(const AzPmatSpaVar *inp) {
    data_num = inp->data_num; 
    ia_dcolind.reset(&inp->ia_dcolind); 
    pia_dcolind.reset(&inp->pia_dcolind); 
    m.set(&inp->m); 
  }  
/*  const AzIntArr *h_index() const { return &ia_dcolind; } */
  const AzPintArr *d_index() const { return &pia_dcolind; }
  int get_begin(int ix) const {
    check_datano(ix, "AzPmatSpaVar::get_begin"); 
    return ia_dcolind[ix*2];   
  }
  int get_end(int ix) const {
    check_datano(ix, "AzPmatSpaVar::get_end"); 
    return ia_dcolind[ix*2+1];   
  }
  void separate_columns(); 
  
protected: 
  void check_datano(int ix, const char *eyec) const {  
    AzX::throw_if((ix < 0 || ix >= data_num), eyec, "invalid index"); 
  }      
};

/*****  To absorb the interface differences between AzPmat and AzPmatSpa; for template use  *****/
class AzPs {
public: 
  static void add(AzPmat *md, const AzPmatSpa *ms, double coeff=1) { ms->add_to(md, coeff); }
  static void add(AzPmat *m0, const AzPmat *m1, double coeff=1) { m0->add(m1, coeff); }
  static void sub(AzPmat *m0, const AzPmat *m1)    { m0->sub(m1); }
  static void sub(AzPmat *md, const AzPmatSpa *ms) { ms->sub_from(md); }
  static void elm_multi(AzPmat *md, const AzPmatSpa *ms, bool do_inv=false) { 
    AzPmat m(ms->rowNum(), ms->colNum()); ms->add_to(&m, 1); 
    md->elm_multi(&m, do_inv); 
  }
  static void elm_multi(AzPmat *m0, const AzPmat *m1, bool do_inv=false)    { m0->elm_multi(m1, do_inv); }  
  template <class M> /* M: AzPmat | AzPmatSpa */
  static void set(AzPmat *md, const M *ms, double coeff=1) {
    md->reform(ms->rowNum(), ms->colNum()); 
    add(md, ms, coeff);  
  }
  #define AzPsAzPmat(x,y) AzPmat x; AzPs::set(&x, y); 
 
  static void set(AzPmatVar *mv, const AzPmatSpaVar *msv, double coeff=1) {
    mv->reform(msv->rowNum(), msv->d_index()); 
    set(mv->data_u(), msv->data(), coeff);  
  }      
  static void prod(AzPmat *m_dst, const AzPmat *md, const AzPmatSpa *ms, 
                   bool tr_d, bool tr_s) {
    AzX::no_support(!(tr_d && !tr_s), "AzPs::prod(dense,sparse)", "Anything other than tran(dense) & !tran(sparse)"); 
    bool do_add = false; 
    ms->prod_dense1_sparse0(m_dst, md, do_add);                        
  }
  static void add_prod(AzPmat *m_dst, const AzPmatSpa *ms, const AzPmat *md, 
                       bool tr_s, bool tr_d, double alpha=1, double beta=1) {
    AzX::no_support(!(!tr_s && tr_d), "AzPs::add_prod(sparse,dense)", "Anything other than !tran(sparse) & tran(dense)");
    m_dst->multiply(beta); 
    bool do_add = true; 
    ms->prod_sparse0_dense1(m_dst, md, alpha, do_add); 
  }
  static void prod(AzPmat *m_dst, const AzPmatSpa *ms, const AzPmat *md, 
                       bool tr_s, bool tr_d, double alpha=1) {
    AzX::no_support(!(!tr_s && tr_d), "AzPs::prod(sparse,dense)", "Anything other than !tran(sparse) & tran(dense)");
    bool do_add = false; 
    ms->prod_sparse0_dense1(m_dst, md, alpha, do_add); 
  }  
  static void add_prod(AzPmat *m_dst, const AzPmat *m0, const AzPmat *m1, 
                       bool tr0, bool tr1, double alpha=1, double beta=1) {
    m_dst->add_prod(m0, m1, tr0, tr1, alpha, beta); 
  }  
  static void prod(AzPmat *m_dst, const AzPmat *m0, const AzPmat *m1, 
                   bool tr0, bool tr1, double alpha=1) {                         
    m_dst->prod(m0, m1, tr0, tr1, alpha); 
  }  
}; 
#endif 
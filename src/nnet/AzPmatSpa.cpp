/* * * * *
 *  AzPmatSpa.cpp
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

#include "AzPmatSpa.hpp"

/*-----  global  -----*/
#ifdef __AZ_GPU__
/* x: row indexing, up: upward (dense1_spasre0), dw: downward (sparse0_dense1) */
bool __dflt_do_cu_x = true;  /* for small mini-batch, this must be false as cu is very slow. */
bool __dflt_do_cu_up = true; 
bool __dflt_do_cu_dw = false; 
#else
bool __dflt_do_cu_x = false;  
bool __dflt_do_cu_up = false; 
bool __dflt_do_cu_dw = false; 
#endif 
/*--------------------*/

extern bool __doDebug; 

/*------------------------------------------------*/ 
AzPmatSpa_flags::AzPmatSpa_flags() {
  do_cu_up = __dflt_do_cu_up; do_cu_dw = __dflt_do_cu_dw; do_cu_x = __dflt_do_cu_x;  
}
#define kw_dont_cu_x "NoCusparseIndex"
#define kw_dont_cu_up "NoCusparseFprop"  
#define kw_dont_cu_dw "NoCusparseBprop" 
#define kw_do_cu_x "CusparseIndex"
#define kw_do_cu_up "CusparseFprop"  
#define kw_do_cu_dw "CusparseBprop" 
void AzPmatSpa_flags::resetParam(AzParam &azp) {
  azp.swOff(&__dflt_do_cu_x, kw_dont_cu_x); 
  azp.swOff(&__dflt_do_cu_up, kw_dont_cu_up); 
  azp.swOff(&__dflt_do_cu_dw, kw_dont_cu_dw); 
  azp.swOn(&__dflt_do_cu_x, kw_do_cu_x); 
  azp.swOn(&__dflt_do_cu_up, kw_do_cu_up); 
  azp.swOn(&__dflt_do_cu_dw, kw_do_cu_dw);  
}  
void AzPmatSpa_flags::printParam(AzPrint &o) {
  o.printSw(kw_do_cu_x, __dflt_do_cu_x); 
  o.printSw(kw_do_cu_up, __dflt_do_cu_up); 
  o.printSw(kw_do_cu_dw, __dflt_do_cu_dw);    
}
void AzPmatSpa_flags::printHelp(AzHelp &h) {
#ifdef __AZ_GPU__  
  h.item(kw_dont_cu_x, "Do not use cusparse for row indexing."); 
//  h.item(kw_dont_cu_up, "Do not use cusparse for forward propagation", " use cusparse");   
//  h.item(kw_do_cu_dw, "Use cusparse for backward propagation", " no cusparse");     
#endif  
}

/*------------------------------------------------*/
/* not cusparse */
void AzPmatSpa::_prod_dense1_sparse0_nocu(AzPmat *m_dst, const AzPmat *md, bool do_add) const
{
  const char *eyec = "AzPmatSpa::_prod_dense1_sparse0"; 
  int dst_r_num = md->colNum(); 
  int dst_c_num = colNum(); 
  AzX::throw_if((rowNum() != md->rowNum()), eyec, "shape mismatch");  
  if (do_add) m_dst->shape_chk(dst_r_num, dst_c_num, "AzPmatSpa::_prod_dense1_sparse0,m_dst"); 
  else        m_dst->reform_noinit(dst_r_num, dst_c_num); 
  
  u._prod_dense1_sparse0(m_dst->_dptr_u(), dst_r_num, dst_c_num, 
                         md->_dptr(), md->rowNum(), md->colNum(), 
                         csc_vals._dptr(), csc_ptrs._dptr(), csc_rows._dptr(), do_add); 
}

/*------------------------------------------------*/
/* not cusparse */
void AzPmatSpa::_prod_sparse0_dense1_nocu(AzPmat *m_dst, const AzPmat *md, AzFloat alpha, bool do_add) const
{
  const char *eyec = "AzPmatSpa::_prod_sparse0_dense1_nocu"; 
  AzX::throw_if((!is_row_indexed()), eyec, "rows needs to be indexed"); 

  int dst_r_num = rowNum(); 
  int dst_c_num = md->rowNum(); 
  AzX::throw_if((colNum() != md->colNum()), eyec, "shape mismatch");  
  if (do_add) m_dst->shape_chk(dst_r_num, dst_c_num, "AzPmatSpa::_prod_sparse0_dense1,m_dst"); 
  else        m_dst->reform(dst_r_num, dst_c_num); /* this must NOT be reform_noinit */

  u._prod_sparse0_dense1_a(m_dst->_dptr_u(), dst_r_num, dst_c_num, 
        csr_vals._dptr(), nzptrs._dptr(), nzrows._dptr(), nzrows.size(), csr_cols._dptr(), 
        md->_dptr(), md->rowNum(), md->colNum(), alpha, do_add); 
} 

/*------------------------------------------------*/
void AzPmatSpa::add_to(AzPmat *m_dst, double coeff) const
{
  const char *eyec = "AzPmatSpa::add_sparse"; 
  m_dst->shape_chk(row_num, col_num, eyec); 
  u._add_sparse(m_dst->_dptr_u(), row_num, col_num, 
     csc_vals._dptr(), csc_rows._dptr(), csc_cols._dptr(), csc_vals.size(), (AzFloat)coeff); 
} 

/*------------------------------------------------*/  
/*------------------------------------------------*/
void AzPmatSpa::_prod_sparse1_dense0_cu(AzPmat *m_dst, const AzPmat *md, bool do_add) const
{
  const char *eyec = "AzPmatSpa::_prod_sparse1_dense0_cu"; 
  int dst_r_num = colNum(); 
  int dst_c_num = md->colNum(); 
  AzX::throw_if((rowNum() != md->rowNum()), eyec, "shape mismatch");  
  if (do_add) m_dst->shape_chk(dst_r_num, dst_c_num, "AzPmatSpa::_prod_sparse1_dense0_cu,m_dst"); 
  else        m_dst->reform(dst_r_num, dst_c_num); 
  
  AzFloat alpha = 1; 
  AzFloat beta = (do_add) ? (AzFloat)1 : (AzFloat)0; 
  u._prod_csr_dense(m_dst->_dptr_u(), dst_r_num, dst_c_num, 
                          csc_vals._dptr(), csc_ptrs._dptr(), csc_rows._dptr(), csc_vals.size(), 
                          md->_dptr(), md->rowNum(), md->colNum(), 
                          alpha, beta); 
}       

/*------------------------------------------------*/
void AzPmatSpa::_prod_sparse0_dense0_cu_nz(AzPmat *m_dst, const AzPmat *md, AzFloat alpha, bool do_add) const
{
  const char *eyec = "AzPmatSpa::_prod_sparse0_dense0_cu_nz"; 
  AzX::throw_if((!is_row_indexed()), eyec, "rows needs to be indexed"); 

  int dst_r_num = nzrows.size(); 
  int dst_c_num = md->colNum(); 
  AzX::throw_if((colNum() != md->rowNum()), eyec, "shape mismatch");  
  AzPmat m_new_dst(dst_r_num, dst_c_num); 

  AzFloat my_beta = 0; 
  u._prod_csr_dense(m_new_dst._dptr_u(), dst_r_num, dst_c_num, 
                    csr_vals._dptr(), nzptrs._dptr(), csr_cols._dptr(), csr_vals.size(), 
                    md->_dptr(), md->rowNum(), md->colNum(), 
                    alpha, my_beta);            
  dst_r_num = rowNum(); 
  if (do_add) m_dst->shape_chk(dst_r_num, dst_c_num, "AzPmatSpa::_prod_sparse0_dense0_cusparse_nz,m_dst"); 
  else        m_dst->reform(dst_r_num, dst_c_num); 

  AzIntArr ia_nzrows(nzrows.size(), -1); 
  nzrows.copy_to_host(ia_nzrows.point_u(), ia_nzrows.size()); 
  m_dst->add_rows_s2d(&m_new_dst, ia_nzrows.point(), ia_nzrows.size(), do_add); 
} 

/*------------------------------------------------*/
void AzPmatSpa::_prod_sparse0_dense0_cu_all(AzPmat *m_dst, const AzPmat *md, AzFloat alpha, bool do_add) const
{
  const char *eyec = "AzPmatSpa::_prod_sparse0_dense0_cu_all"; 
  AzX::throw_if((!is_row_indexed()), eyec, "rows needs to be indexed"); 

  int dst_r_num = rowNum(); 
  int dst_c_num = md->colNum(); 
  AzX::throw_if((colNum() != md->rowNum()), eyec, "shape mismatch");  
  if (do_add) m_dst->shape_chk(dst_r_num, dst_c_num, "AzPmatSpa::_prod_sparse0_dense0_cu_all,m_dst"); 
  else        m_dst->reform(dst_r_num, dst_c_num); 

  AzFloat beta = (do_add) ? (AzFloat)1 : (AzFloat)0;   
  u._prod_csr_dense(m_dst->_dptr_u(), dst_r_num, dst_c_num, 
                    csr_vals._dptr(), csr_ptrs._dptr(), csr_cols._dptr(), csr_vals.size(), 
                    md->_dptr(), md->rowNum(), md->colNum(), 
                    alpha, beta);                        
} 

/*------------------------------------------------*/
void AzPmatSpa::set(const AzDataArr<AzIFarr> &aifa, 
                     int r_num, 
                     const AzIntArr *ia_row_old2new, /* may be NULL */
                     const int *cxs, /* col#'s */ 
                     int cxs_num, 
                     bool do_gen_row_index)
{
  const char *eyec = "AzPmatSpa::set(aifa,#row,row_old2new,cxs,#cxs)"; 
  AzX::throw_if((cxs == NULL), eyec, "Null input");   

  reform(r_num, cxs_num); 
 
  int nz_num = 0; 
  for (int ix = 0; ix < cxs_num; ++ix) {
    nz_num += aifa[cxs[ix]]->size(); 
  }
  AzFloat *hvals = NULL; 
  AzBaseArray<AzFloat> _hvals(nz_num, &hvals);   
  AzIntArr ia_rows(nz_num, -1); 
  AzIntArr ia_ptrs; 
  AzIntArr ia_cols(nz_num, -1); /* for addition and multiplication */

  /*---  column-wise  ---*/  
  AzIIFarr iifa; 
  if (do_gen_row_index && !f.do_cu_x) iifa.prepare(nz_num); 
  int index = 0; 
  for (int col = 0; col < col_num; ++col) {
    ia_ptrs.put(index); /* beginning of this column */
    const AzIFarr *ifa = aifa[cxs[col]]; 
    int last_row = -1; 
    for (int ix = 0; ix < ifa->size(); ++ix) {
      int row = ifa->getInt(ix); 
      double val = ifa->get(ix);       
 
      if (ia_row_old2new != NULL) row = (*ia_row_old2new)[row]; 
      AzX::throw_if((row < 0 || row >= row_num), eyec, "invalid row#"); 
      AzX::throw_if((row <= last_row), eyec, "not sorted in the order of rows");  

      hvals[index] = (AzFloat)val; 
      ia_rows(index, row); 
      ia_cols(index, col);      
      if (do_gen_row_index && !f.do_cu_x) {
        iifa.put(row, col, val); 
      }
      ++index; 
      last_row = row;       
    }
  }
  ia_ptrs.put(index); 
  AzX::throw_if((index != nz_num), eyec, "number doesn't match");  
  _set_csc_csr(hvals, nz_num, ia_ptrs, ia_rows, ia_cols, do_gen_row_index, iifa); 
}

/*------------------------------------------------*/
template <class M> /* M: AzSmatc | AzSmat | AzSmatbc */
void AzPmatSpa::set(const M *ms, 
                    const int *cxs, /* col#'s */
                    int cxs_num, 
                    bool do_gen_row_index)
{
  const char *eyec = "AzPmatSpa::set(m,cxs,#cxs)"; 
  AzX::throw_if_null(ms, eyec, "matrix"); 
  AzX::throw_if_null(cxs, eyec, "data indexes"); 
  
  reform(ms->rowNum(), cxs_num); 
 
  int nz_num = 0; 
  for (int ix = 0; ix < cxs_num; ++ix) {
    nz_num += ms->col_size(cxs[ix]); 
  }
  AzFloat *hvals = NULL; 
  AzBaseArray<AzFloat> _hvals(nz_num, &hvals);   
  AzIntArr ia_rows(nz_num, -1); 
  AzIntArr ia_ptrs; 
  AzIntArr ia_cols(nz_num, -1); /* for addition and multiplication */

  /*---  column-wise  ---*/  
  AzIIFarr iifa; 
  if (do_gen_row_index && !f.do_cu_x) iifa.prepare(nz_num); 
  bool do_elm = !ms->is_bc(); 
  int index = 0; 
  for (int col = 0; col < col_num; ++col) {
    ia_ptrs.put(index); /* beginning of this column */
    int elm_num = ms->col_size(cxs[col]); 
    const AZI_VECT_ELM *myelm = NULL; const int *elmno = NULL; 
    if (do_elm) myelm = ms->rawcol_elm(cxs[col]); 
    else        elmno = ms->rawcol_int(cxs[col]); 
    for (int ix = 0; ix < elm_num; ++ix) {
      AzFloat val; int no; 
      if (do_elm) { val = myelm[ix].val; no = myelm[ix].no; }
      else        { val = 1;             no = elmno[ix]; }
      hvals[index] = val; 
      ia_rows(index, no); 
      ia_cols(index, col);      
      if (do_gen_row_index && !f.do_cu_x) iifa.put(no, col, val); 
      ++index;       
    }
  }
  ia_ptrs.put(index); 
  AzX::throw_if((index != nz_num), eyec, "number doesn't match");  

  _set_csc_csr(hvals, nz_num, ia_ptrs, ia_rows, ia_cols, do_gen_row_index, iifa); 
}
template void AzPmatSpa::set<AzSmatc>(const AzSmatc *, const int *, int, bool); 
template void AzPmatSpa::set<AzSmat>(const AzSmat *, const int *, int, bool); 
template void AzPmatSpa::set<AzSmatbc>(const AzSmatbc *, const int *, int, bool); 

/*------------------------------------------------*/
void AzPmatSpa::_set_csc_csr(const AzFloat *hvals, int nz_num, 
                              const AzIntArr &ia_ptrs, const AzIntArr &ia_rows, const AzIntArr &ia_cols, 
                              bool do_gen_row_index, 
                              AzIIFarr &iifa)
{
  const char *eyec = "AzPmatSpa::_set_csc_csr";   

  csc_vals.reset_from_host(hvals, nz_num, eyec, "csc_vals"); 
  csc_ptrs.reset_from_host(ia_ptrs.point(), ia_ptrs.size(), eyec, "csc_ptrs"); 
  csc_rows.reset_from_host(ia_rows.point(), ia_rows.size(), eyec, "csc_rows"); 
  csc_cols.reset_from_host(ia_cols.point(), ia_cols.size(), eyec, "csc_cols"); 

  if (do_gen_row_index && !f.do_cu_x) gen_row_index(do_gen_row_index, &iifa); 
  else                                gen_row_index(do_gen_row_index); 
}
  
/*------------------------------------------------*/
void AzPmatSpa::show(const char *header, 
                      const _AzParr<AzFloat> &vals, 
                      const _AzParr<int> &ptrs, 
                      const _AzParr<int> &inds) const
{                      
  cout << header << endl; 
  AzFloat *hval = NULL; 
  AzBaseArray<AzFloat> _a(vals.size(), &hval);      
  AzIntArr ia_ptrs(ptrs.size(), -1); 
  AzIntArr ia_inds(inds.size(), -1); 
  vals.copy_to_host(hval, vals.size()); 
  ptrs.copy_to_host(ia_ptrs.point_u(), ia_ptrs.size()); 
  inds.copy_to_host(ia_inds.point_u(), ia_inds.size()); 
  cout << "vals: "; 
  for (int ix = 0; ix < vals.size(); ++ix) {
    cout << ix << ":" << hval[ix] << " "; 
  }
  cout << endl; 
  ia_ptrs.print(log_out, "ia_ptrs"); 
  ia_inds.print(log_out, "ia_inds"); 
}
      
/*------------------------------------------------*/
void AzPmatSpa::gen_row_index(bool do_gen_row_index, AzIIFarr *iifa)
{
  const char *eyec = "AzPmatSpa::gen_row_index"; 

  if (!do_gen_row_index) {
    csr_vals.free(); 
    csr_ptrs.free(); 
    csr_cols.free();     
    nzptrs.free(); 
    nzrows.free(); 
    return; 
  }
  
  if (iifa != NULL && iifa->size() == csc_vals.size()) {
    _gen_row_index(iifa); 
  }
  else if (csc_vals.size() == 0) { /* to go around a bug in cusparse */
    AzIIFarr iifa_empty; 
    _gen_row_index(&iifa_empty);      
  }
  else {
    csr_vals.free_alloc(csc_vals.size(), "AzPmatSpa::gen_row_index,csr_vals"); 
    csr_ptrs.free_alloc(row_num + 1, "AzPmatSpa::gen_row_index,csr_ptrs"); 
    csr_cols.free_alloc(csc_vals.size(), "AzPmatSpa::gen_row_index,csr_cols");   
    u._csc2csr(row_num, col_num, 
               csc_vals._dptr(), csc_ptrs._dptr(), csc_rows._dptr(), csc_vals.size(), 
               csr_vals._dptr_u(), csr_ptrs._dptr_u(), csr_cols._dptr_u());   
  }
               
  AzIntArr ia_ptrs(csr_ptrs.size(), -1); 
  csr_ptrs.copy_to_host(ia_ptrs.point_u(), ia_ptrs.size()); 

  AzIntArr ia_nzptrs, ia_nzrows; 
  for (int row = 0; row < ia_ptrs.size()-1; ++row) {
    int bx = ia_ptrs[row], ex = ia_ptrs[row+1]; 
    if (bx < ex) {
      if (bx < 0 || bx >= csr_vals.size()) {
        if (f.do_cu_x) AzX::throw_if(true, eyec, "A bug in cusparseScsr2csc.  Specify NoCusparseIndex."); 
        else           AzX::throw_if(true, eyec, "Something is wrong with row indexing."); 
      }
      ia_nzptrs.put(bx); 
      ia_nzrows.put(row); 
    }
  }
  ia_nzptrs.put(csr_vals.size());  
  nzptrs.reset_from_host(ia_nzptrs.point(), ia_nzptrs.size(), eyec, "nzptrs");  
  nzrows.reset_from_host(ia_nzrows.point(), ia_nzrows.size(), eyec, "nzrows");      
}  

/*------------------------------------------------*/
void AzPmatSpa::_gen_row_index(AzIIFarr *iifa)
{
  const char *eyec = "AzPmatSpa::_gen_row_index"; 
  int nz_num = iifa->size(); 
  iifa->sort_IntInt(true); /* ascending order */

  AzFloat *h_csr_vals = NULL; 
  AzBaseArray<AzFloat> _a(nz_num, &h_csr_vals); 
  AzIntArr ia_ptrs(row_num+1, 0); 
  AzIntArr ia_cols(nz_num, -1); 
  
  int index = 0; 
  int last_row = -1; 
  for (int ix = 0; ix < nz_num; ++ix) {
    int row, col; 
    h_csr_vals[ix] = (AzFloat)iifa->get(ix, &row, &col); 
    ia_cols(ix, col); 
    if (ix == 0 || row != last_row) {
      for (int rx = last_row+1; rx <= row; ++rx) {
        ia_ptrs(rx, ix);  
      }
    }
    last_row = row; 
  }
  for (int rx = last_row+1; rx <= row_num; ++rx) {
    ia_ptrs(rx, nz_num); 
  }
  csr_vals.reset_from_host(h_csr_vals, nz_num, eyec, "csr_vals"); 
  csr_ptrs.reset_from_host(ia_ptrs.point(), ia_ptrs.size(), eyec, "csr_ptrs"); 
  csr_cols.reset_from_host(ia_cols.point(), ia_cols.size(), eyec, "csr_cols"); 
}  
  
/*------------------------------------------------*/
void AzPmatSpa::check_consistency() const
{
  const char *eyec = "AzPmatSpa::check_consistency"; 

  AzX::throw_if((csc_ptrs.size() != col_num+1), eyec, "colPtr length is wrong"); 
  
  /*---  coldata  ---*/
  AzIntArr ia_ptrs(csc_ptrs.size(), 0); csc_ptrs.copy_to_host(ia_ptrs.point_u(), ia_ptrs.size()); 
  AzIntArr ia_rows(csc_rows.size(), 0); csc_rows.copy_to_host(ia_rows.point_u(), ia_rows.size()); 
  AzIntArr ia_cols(csc_cols.size(), 0); csc_cols.copy_to_host(ia_cols.point_u(), ia_cols.size()); 
  
  for (int col = 0; col < col_num; ++col) {
    int bx = ia_ptrs[col]; 
    int ex = ia_ptrs[col+1]; 
    int last_row = -1; 
    for (int ix = bx; ix < ex; ++ix) {
      AzX::throw_if((ix < 0 || ix >= csc_vals.size()), eyec, "colPtr is wrong"); 
      int row = ia_rows[ix]; 
      AzX::throw_if((row < 0 || row >= row_num), eyec, "row# is out of range"); 
      AzX::throw_if((ix > bx && row <= last_row), eyec, "row# is out of order");  
      AzX::throw_if((ia_cols[ix] != col), eyec, "col# is out of order"); 
      last_row = row; 
    }
  }
}

/*------------------------------------------------*/
template <class M> /* M: AzSmatcVar | AzSmatVar | AzsmatbcVar */
void AzPmatSpaVar::set(const M *mv0, 
                       const int *dxs0, /* mv0's data points */
                       int dxs0_num,   /* size of dxs0 */
                       bool do_gen_rowindex)
{
  const char *eyec = "AzPmatSpaVar::set(mvc0,dxs,num)"; 
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

  m.set(mv0->data(), ia_cols.point(), ia_cols.size(), do_gen_rowindex); 
  pia_dcolind.reset(&ia_dcolind); 
  if (__doDebug) check_data_consistency(eyec); 
} 
template void AzPmatSpaVar::set<AzSmatcVar>(const AzSmatcVar *, const int *, int, bool); 
template void AzPmatSpaVar::set<AzSmatVar>(const AzSmatVar *, const int *, int, bool); 
template void AzPmatSpaVar::set<AzSmatbcVar>(const AzSmatbcVar *, const int *, int, bool); 

/*------------------------------------------------*/
void AzPmatSpaVar::check_data_consistency(const char *eyec)
{
  AzX::throw_if((data_num*2 != ia_dcolind.size()), eyec, "conflict in the length of dcolind"); 
  const int *dcolind = ia_dcolind.point(); 
  for (int dx = 0; dx < data_num; ++dx) {
    int col0 = dcolind[dx*2]; 
    int col1 = dcolind[dx*2+1]; 
    AzX::throw_if((col1-col0 < 0), eyec, "end cannot be smalle than begin"); 
    AzX::throw_if((col0 < 0 || col1 > m.colNum()), eyec, "column index is pointing outside of matrix"); 
    AzX::throw_if((dx > 0 && col0 != dcolind[dx*2-1]), eyec, "data is out of order"); 
  }
  AzX::throw_if((data_num > 0 && dcolind[data_num*2-1] != m.colNum()), eyec, "#column mismatch"); 
}

/*------------------------------------------------*/
void AzPmatSpaVar::separate_columns()
{
  data_num = m.colNum(); 
  ia_dcolind.reset(); 
  for (int col = 0; col < m.colNum(); ++col) {
    ia_dcolind.put(col); ia_dcolind.put(col+1); 
  }
  pia_dcolind.reset(&ia_dcolind);
  check_data_consistency("AzPmatSpaVar::separate_columns"); 
}


#ifdef __AZ_CSRMM2__
/*------------------------------------------------*/
void AzPmatSpa::_prod_sparse_dense_cu(AzPmat *m_dst, const AzPmat *md, 
                                      bool do_tran_s, bool do_tran_d, bool do_add) const
{
  const char *eyec = "AzPmatSpa::_prod_sparse_dense_cu"; 
  AzX::throw_if((!is_row_indexed()), eyec, "rows needs to be indexed"); 

  int dst_r_num = (do_tran_s) ? colNum() : rowNum(); 
  int dst_c_num = (do_tran_d) ? md->rowNum() : md->colNum(); 
  int num0 = (do_tran_s) ? rowNum() : colNum(); 
  int num1 = (do_tran_d) ? md->colNum() : md->rowNum(); 
  AzX::throw_if((num0 != num1), eyec, "shape mismatch");  
  if (do_add) m_dst->shape_chk(dst_r_num, dst_c_num, "AzPmatSpa::_prod_sparse_dense_cu,m_dst"); 
  else        m_dst->reform(dst_r_num, dst_c_num); 
  
  AzFloat alpha = 1; 
  AzFloat beta = (do_add) ? (AzFloat)1 : (AzFloat)0; 
  u._prod_csr_dense_mm2(do_tran_s, do_tran_d, m_dst->_dptr_u(), dst_r_num, dst_c_num, 
                          csr_vals._dptr(), csr_ptrs._dptr(), csr_cols._dptr(), csr_vals.size(), 
                          md->_dptr(), md->rowNum(), md->colNum(), 
                          alpha, beta); 
}       
#endif 

/*------------------------------------------------*/
void AzPmatSpa::dropout(double dout, bool is_test, AzPrng &rng, bool do_scale) {
  if (dout <= 0) return; 
  int sz = csc_vals.size(); 
  if (sz <= 0) return; 
  if (is_test) {
    if (!do_scale) return; 
    AzFloat val = (AzFloat)(1-dout); 
    uu._multiply(csc_vals._dptr_u(), val, sz);  
    if (csr_vals.size() > 0) uu._multiply(csr_vals._dptr_u(), val, csr_vals.size());
  }
  else {
    AzPmat m_mask(sz, 1); 
    rng.uniform_01(&m_mask);  /* [0,1] */
    m_mask.mark_gt((AzFloat)dout);  /* ([i,j] > dropout) ? 1 : 0 */
    bool do_inv = false; 
    uu._elm_multi(csc_vals._dptr_u(), m_mask._dptr(), sz, do_inv); 
    if (is_row_indexed()) gen_row_index(true);  /* reset row index */
  }
}    
 
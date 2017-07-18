/* * * * *
 *  AzpData_sparse.hpp
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

#ifndef _AZP_DATA_SPARSE_HPP_
#define _AZP_DATA_SPARSE_HPP_

#include "AzpData_.hpp"

/*---  sparse variable-sized/fixed-sized data  ---*/
/***
 *  AzSmatbc consumes smaller CPU memory than AzSmat, 
 *  but # of non-zero components must be no greater than 2GB.   
 ***/
template <class Mat, class MatY> /* Mat,MatY: AzSmatbc | AzSmatc */
class AzpData_sparse : public virtual /* implements */ AzpData_ {
protected: 
  int dsno; 
  bool do_allow_diffidx; 
  bool do_dense_y; /* for compatibility, but probably no use */

  AzDicc dic;   
  AzMatVar<Mat> msv_x; 
  Mat ms_x; 
  AzDmatc md_y; 
  MatY ms_y; 
  AzMatVar<MatY> msv_y; 
  int data_num, rnum, cnum; 
  int dummy_ydim; 

  int current_batch, released_batch; 
  int total_data_num; 
  bool is_spa_y, is_var_x, is_var_y; 
  
  AzStrPool sp_x_ext, sp_y_ext; 
  double min_tar, max_tar; 
 
  virtual AzpData_ *clone_nocopy() const { return new AzpData_sparse(); }   

  /*---  For on-the-fly region generation (experimental and slow)  ---*/
  int psz, pstep, padding; 
  AzIntArr _ia_nn, *ia_nn; 
  bool do_nobow; 
  bool do_gen_regions() const { return (psz > 1 || pstep > 1 || padding > 0); }
  void pfx_for_gen_regions(AzBytArr &s_pfx) const { s_pfx.reset("ds"); s_pfx << (MAX(0,dsno)) << "_"; }
  
public:
  AzpData_sparse() : current_batch(-1), data_num(0), rnum(0), cnum(0), total_data_num(0), is_spa_y(false), is_var_x(false), is_var_y(false), 
                     dummy_ydim(-1), min_tar(1e+10), max_tar(-1e+10), dsno(-1),
                     do_allow_diffidx(false), do_dense_y(false), released_batch(-1), 
                     ia_nn(NULL), psz(-1), pstep(1), padding(0), do_nobow(false) {                        
    sp_x_ext.put(AzpData_Ext_xsmatbc, AzpData_Ext_xsmatbcvar, AzpData_Ext_xsmatcvar, AzpData_Ext_x); /* cvar for seq2-bown */
    sp_y_ext.put(AzpData_Ext_ysmatbc, AzpData_Ext_ysmatbcvar, AzpData_Ext_ysmatcvar, AzpData_Ext_y, AzpData_Ext_ysmatc);     
  }
          
  virtual void reset_dsno(int _dsno=-1) { dsno = _dsno; }          
  virtual double min_target() const { return min_tar; }
  virtual double max_target() const { return max_tar; }
  virtual bool is_vg_x() const { return is_var_x; } /* each data point is a vector group */
  virtual bool is_vg_y() const { return is_var_y; }
  virtual bool is_sparse_x() const { return true; }  
  virtual bool is_sparse_y() const { return is_spa_y; }
  virtual int xdim(int data_no=0) const { 
    if (data_no == 0) return rnum; 
    return 0; 
  }
  virtual int dimensionality() const { return 1; }
  virtual int size(int index) const { 
    if (is_var_x) return -1; 
    if (index == 0) return 1; 
    return -1; 
  } 
  #define kw_do_allow_diffidx "AllowDifferentIndexes"
  #define kw_do_dense_y "DenseY"
  #define kw_psz "patch_size="
  #define kw_pstep "patch_stride="
  #define kw_padding "padding="
  #define kw_do_bow "Bow"
  #define kw_do_nobow "Seq"
  virtual void resetParam_data(AzParam &azp) {
    const char *eyec = "AzpData_sparse::resetParam_data"; 
    azp.swOn(&do_allow_diffidx, kw_do_allow_diffidx); 
    AzXi::check_input(s_x_ext, &sp_x_ext, eyec, kw_x_ext); 
    if (s_y_ext.length() > 0) AzXi::check_input(s_y_ext, &sp_y_ext, eyec, kw_y_ext);        
    is_var_x = is_var_ext(s_x_ext); 
    is_spa_y = is_spa_ext(s_y_ext); 
    is_var_y = is_var_ext(s_y_ext); 
    if (s_y_ext.equals(AzpData_Ext_y)) {
      azp.swOn(&do_dense_y, kw_do_dense_y); 
      if (!do_dense_y) is_spa_y = true; 
    }   
    AzX::throw_if(is_var_y && !is_var_x, AzInputError, eyec, "Y can be variabled-sized only when X is variable-sized."); 
    AzX::throw_if(is_bc_ext(s_x_ext) && !ms_x.is_bc(), eyec, "x_ext is binary compact (\"bc\"), but the matrix type is not."); 
    AzX::throw_if(!is_bc_ext(s_x_ext) && ms_x.is_bc(), eyec, "x_ext is not binary compact (\"bc\"), but the matrix type is."); 
    AzX::throw_if(is_bc_ext(s_y_ext) && !ms_y.is_bc(), eyec, "y_ext is binary compact (\"bc\"), but the matrix type is not."); 
    AzX::throw_if(!is_bc_ext(s_y_ext) && ms_y.is_bc(), eyec, "y_ext is not binary compact (\"bc\"), but the matrix type is.");      
    
    AzBytArr s_pfx; pfx_for_gen_regions(s_pfx); 
    azp.reset_prefix(s_pfx.c_str()); 
    azp.vInt(kw_psz, &psz); azp.vInt(kw_pstep, &pstep); azp.vInt(kw_padding, &padding); 
    azp.swOn(&do_nobow, kw_do_nobow); 
    azp.reset_prefix();
    if (do_gen_regions()) {
      AzX::throw_if(!msv_x.data()->is_bc(), eyec, 
        "To do region generation on the fly, the input matrix must be binary compact (*.xsmatbcvar).");       
      AzXi::throw_if_nonpositive(psz, eyec, kw_psz); 
      AzXi::throw_if_nonpositive(pstep, eyec, kw_pstep);       
      AzXi::throw_if_negative(padding, eyec, kw_padding);       
    }    
  }
  virtual void printParam_data(const AzOut &out, const char *pfx) const {
    AzPrint o(out, pfx); 
    if (dsno <= 0) { /* to avoid printing the same parameters repeatedly. */
      o.printSw(kw_do_allow_diffidx, do_allow_diffidx); 
      o.printSw(kw_do_dense_y, do_dense_y); 
    }
    if (do_gen_regions()) {
      AzBytArr s_pfx; pfx_for_gen_regions(s_pfx); 
      o.reset_prefix(s_pfx.c_str()); 
      o.printV(kw_psz, psz); o.printV(kw_pstep, pstep); o.printV(kw_padding, padding); 
      if (do_nobow) o.printSw(kw_do_nobow, do_nobow); 
      else          o.printSw(kw_do_bow, !do_nobow);       
      o.reset_prefix(); 
    }
    o.printEnd(); 
  }
  virtual void printHelp_data(AzHelp &h) const {}
  virtual void reset() { destroy(); }
  virtual int ydim() const { return (is_var_y) ? msv_y.rowNum() : ((is_spa_y) ? ms_y.rowNum() : md_y.rowNum()); }
  virtual int dataNum_total() const { return total_data_num; }
  virtual void destroy() {
    AzpData_::destroy(); 
    msv_x.destroy(); 
    ms_x.destroy(); 
    md_y.destroy(); 
    ms_y.destroy(); 
    msv_y.destroy(); 
    total_data_num = data_num = rnum = cnum = 0; 
    current_batch = -1; 
  }
  virtual int dataNum() const { return data_num; }
  virtual int colNum() const { return cnum; }
  virtual int colNum(int dx) const {
    AzX::throw_if(dx < 0 || dx >= dataNum(), "AzpData_sparse::colNum(dx)", "index is out of range"); 
    if (is_var_x) return msv_x.get_end(dx) - msv_x.get_begin(dx); 
    else          return 1;       
  }
  virtual const AzIntArr *dataIndex(int dsno=0) const { 
    const char *eyec = "AzpData_sparse::dataIndex"; 
    AzX::throw_if(dsno != 0, eyec, "Invalid dsno"); 
    AzX::throw_if(msv_x.dataNum() <= 0, eyec, "No data");  
    return msv_x.index(); 
  }
  virtual const AzDicc &get_dic(int dsno) const {
    AzX::throw_if((dsno != 0), "AzpData_sparse::get_dic", "dsno<>0?!"); 
    return dic; 
  }

  /*---  sparse variable-sized data  ---*/
  virtual void _gen_data(const int *dxs, int d_num, AzPmatSpaVar *m_data, int dsno) const { 
    const char *eyec = "AzpData_sparse::_gen_data(dxs,num,spavar)"; 
    AzX::throw_if_null(m_data, dxs, eyec); 
    AzX::throw_if((dsno != 0), eyec, "dsno<>0?!"); 
    AzX::throw_if(!is_vg_x(), eyec, "Not spavar");   
    bool do_gen_rowindex = true;     
    if (do_gen_regions()) gen_regions(dxs, d_num, *m_data, do_gen_rowindex); 
    else                  m_data->set(&msv_x, dxs, d_num, do_gen_rowindex);    
  }

  /*---  sparse features  ---*/
  virtual void _gen_data(const int *dxs, int d_num, AzPmatSpa *m_data, int dsno) const {
    const char *eyec = "AzpData_sparse::_gen_data(dxs,num,spa)"; 
    AzX::throw_if((dsno != 0), eyec, "dsno<>0?!"); 
    AzX::throw_if(is_vg_x(), "spavar");   
    bool do_gen_rowindex=true; 
    m_data->set(&ms_x, dxs, d_num, do_gen_rowindex); 
  }  
  virtual void gen_data(const int *dxs, int d_num, AzSmatVar &msv_data, int dsno=0) const {
    const char *eyec = "AzpData_sparse::gen_data(dxs,num,spavar)"; 
    AzX::throw_if((dsno != 0), eyec, "dsno<>0?!"); 
    AzX::throw_if(!is_vg_x(), eyec, "Not spavar");    
    AzMatVar<Mat> msv; msv.set(&msv_x, dxs, d_num); 
    AzSmat ms; msv.data()->copy_to_smat(&ms); 
    msv_data.set(&ms, msv.h_index());     
  } /* used by MLike */
  
  /*---  dense target  ---*/
  virtual void gen_targets(const int *dxs, int d_num, AzPmat *m_out_y) const {
    AzX::throw_if(is_spa_y, "AzpData_sparse::gen_targets(dense)(dxs,d_num..)", "target is sparse"); 
    m_out_y->set(&md_y, dxs, d_num); 
  }  
  virtual void gen_membership(AzDataArr<AzIntArr> &aia_cl_dxs) const {
    if      (md_y.colNum() > 0) _gen_membership(md_y, aia_cl_dxs); 
    else if (ms_y.colNum() > 0) _gen_membership(ms_y, aia_cl_dxs); 
    else {
      AzX::no_support(true, "AzpData_sparse::gen_membership", "Membership inquiry for variable-sized targets"); 
    }      
  }

  /*---  sparse target  ---*/ 
  virtual void gen_targets(const int *dxs, int d_num, AzSmat *m_out_y) const {
    AzX::throw_if(!is_spa_y, "AzpData_sparse::gen_targets(smat)(dxs,d_num..)", "target is dense");  
    if (is_var_y) {
      AzMatVar<MatY> msv; msv.set(&msv_y, dxs, d_num); 
      msv.data()->copy_to_smat(m_out_y); 
    }
    else {   
      ms_y.copy_to_smat(m_out_y, dxs, d_num); 
    }
  }  
  virtual void gen_targets(const int *dxs, int d_num, AzPmatSpa *m_out_y) const {
    AzX::throw_if(!is_spa_y, "AzpData_sparse::gen_targets(sparse)(dxs,d_num..)", "target is dense"); 
    bool gen_rowindex = false;     
    if (is_var_y) {
      AzMatVar<MatY> msv; msv.set(&msv_y, dxs, d_num); 
      m_out_y->set(msv.data(), gen_rowindex); 
    }
    else {
      m_out_y->set(&ms_y, dxs, d_num, gen_rowindex); 
    }
  }  

  /*------------------------------------------*/  
  virtual void reset_data(const AzOut &_out, const char *nm, int _dummy_ydim=-1, 
                          AzpData_binfo *bi=NULL) /* used by sparse_multi */ {
    const char *eyec = "AzpData_sparse::reset_data"; 
    out = _out; 
    s_nm.reset(nm); 
    dummy_ydim = _dummy_ydim; 
    total_data_num = 0; 
    if (bi != NULL) bi->reset(batch_num); 
    int x_row = -1, xs_row = -1, y_row = -1, ys_row = -1, ysv_row = -1, x2_row = -1; 
    int bx; 
    for (bx = batch_num-1; bx >= 0; --bx) {
      AzBytArr s_batchnm; 
      _reset_data(bx); 
      if (bx == batch_num-1) {
        x_row = msv_x.rowNum(); 
        xs_row = ms_x.rowNum(); 
        y_row = md_y.rowNum(); 
        ys_row = ms_y.rowNum(); 
        ysv_row = msv_y.rowNum(); 
      }
      else {
        AzX::throw_if((msv_x.rowNum() != x_row || ms_x.rowNum() != xs_row || 
            md_y.rowNum() != y_row || ms_y.rowNum() != ys_row || msv_y.rowNum() != ysv_row), 
            AzInputError, eyec, "Data dimensionality conflict between batches");    
      }
      if (is_var_y) {
        min_tar = MIN(min_tar, msv_y.data()->min()); 
        max_tar = MAX(max_tar, msv_y.data()->max()); 
      }
      else if (is_spa_y) {
        min_tar = MIN(min_tar, ms_y.min()); 
        max_tar = MAX(max_tar, ms_y.max()); 
      }
      else {
        min_tar = MIN(min_tar, md_y.min()); 
        max_tar = MAX(max_tar, md_y.max()); 
      }    
      if (bi != NULL) bi->update(bx, data_num); 
      total_data_num += data_num; 
      current_batch = bx; 
      AzTimeLog::print("#data = ", data_num, out); 
    }
    if (dummy_ydim <= 0) {
      AzBytArr s; s << "target-min,max=" << min_tar << "," << max_tar; 
      AzTimeLog::print(s.c_str(), out);
    }
  }

  /*------------------------------------------*/   
  virtual void first_batch() {
    released_batch = -1; 
    if (current_batch != 0) {
      current_batch = -1; 
      next_batch(); 
    }
  }
  
  /*------------------------------------------*/    
  virtual void next_batch() {  
    if (batch_num == 1 && current_batch == 0) return; 
    if (released_batch >= 0) current_batch = released_batch; /* resume the sequence */
    current_batch = (current_batch + 1) % batch_num;      
    bool do_print = true, do_print_stat = false; 
    _reset_data(current_batch, do_print, do_print_stat); 
    released_batch = -1; 
  }
  
  /*------------------------------------------*/   
  virtual void release_batch() {
    released_batch = current_batch; /* suspend the sequence */
    current_batch = -1; 
    msv_x.reset(); ms_x.reset(); md_y.reset(); ms_y.reset(); msv_y.reset(); 
    /* data_num = 0; */ 
  }

protected:
  bool is_var_ext(const AzBytArr &s_ext) const { return s_ext.endsWith("var"); }
  bool is_spa_ext(const AzBytArr &s_ext) const { return s_ext.contains("smat"); }
  bool is_bc_ext(const AzBytArr &s_ext) const { return s_ext.contains("bc"); }
  virtual void _reset_data(int batch_no, bool do_print=true, bool do_print_stat=true) {  
    const char *eyec = "AzpData_sparse::_reset_data";  
    if (do_print) AzTimeLog::print("... ", s_nm.c_str(), " batch#", batch_no+1, out); 
    AzBytArr s_x_fn, s_y_fn; 
    const char *x_fn = gen_batch_fn(batch_no, s_x_ext.c_str(), &s_x_fn);    

    msv_x.reset(); 
    ms_x.reset(); /* added on 04/07/2015 */
    md_y.reset(); /* replaced pmat on 08/10/2015 */
    ms_y.reset(); 
    msv_y.reset(); 
  
    if (is_var_ext(s_x_ext)) { 
      msv_x.read(x_fn); /* Mat is smatvar | smatcvar | smatbcvar */
      data_num = msv_x.dataNum(); cnum = msv_x.colNum(); rnum = msv_x.rowNum(); 
    }
    else if (is_spa_ext(s_x_ext)) {  
      ms_x.read(x_fn); 
      data_num = ms_x.colNum();  cnum = ms_x.colNum(); rnum = ms_x.rowNum(); 
    }
    else if (s_x_ext.equals(AzpData_Ext_x)) {
      AzSmat ms; 
      AzTextMat::readMatrix(x_fn, &ms); 
      ms_x.set(&ms); 
      data_num = ms_x.colNum(); cnum = ms_x.colNum(); rnum = ms_x.rowNum();  
    } 
    else {
      AzX::throw_if(true, eyec, "???", s_x_ext.c_str());  
    }
    
    AzX::throw_if((data_num == 0), AzInputError, eyec, "no data: ", x_fn);     
    if (dummy_ydim > 0) {
      if (!is_var_y) ms_y.reform(dummy_ydim, data_num);  /* 2/5/2015: so that U can be tested */
      else {
        MatY ms(dummy_ydim, msv_x.colNum()); 
        msv_y.reset(&ms, msv_x.index());         
      }
    }
    else {
      const char *y_fn = gen_batch_fn(batch_no, s_y_ext.c_str(), &s_y_fn); 
      if (is_var_ext(s_y_ext)) {
        msv_y.read(y_fn); 
        AzX::throw_if((msv_y.dataNum() != data_num), AzInputError, eyec, "#data mismatch btw features and targets"); 
        if (msv_x.index()->compare(msv_y.index()) != 0) {
          AzX::throw_if(!do_allow_diffidx, AzInputError, eyec, "data index mismatch btw features and targets");  
          AzPrint::writeln(out, "Y's data indexe differs from X's."); 
        }
      }    
      else if (s_y_ext.equals(AzpData_Ext_y)) {
        if (is_spa_y) readY_text(y_fn, dummy_ydim, data_num, &ms_y);         
        else          readY_text(y_fn, dummy_ydim, data_num, &md_y); 
      }
      else if (is_spa_ext(s_y_ext)) {
        readY_bin(y_fn, dummy_ydim, data_num, &ms_y); 
      }    
      else {
        AzX::throw_if(true, eyec, "???", s_y_ext.c_str());  
      }
    }

    /*---  read word-mapping:  added on 8/20/2015 and moved here on 9/14/2015  ---*/
    if (batch_no == 0) {
      AzBytArr s_xtext_fn; 
      const char *xtext_fn = gen_xtext_fn(&s_xtext_fn); 
      if (!AzFile::isExisting(xtext_fn)) {
        AzBytArr s("Unable to open the word-mapping file ("); s << xtext_fn << ").  "; 
        s << "To re-generate it, \"prepText gen_regions\" needs to be called again."; 
        AzX::throw_if(true, AzInputError, eyec, s.c_str()); 
      }
      AzDic mydic(xtext_fn); mydic.copy_words_only_to(dic);
      ia_nn = NULL; 
      if (mydic.get_max_n() > 1) { mydic.get_n(_ia_nn); ia_nn = &_ia_nn; }
    }
        
    if (do_print_stat) show_x_stat(); 
  }   
  void show_x_stat() const {
    const Mat *m = (is_vg_x()) ? msv_x.data() : &ms_x; 
    AzBytArr s("  #row="); s << m->rowNum() << " #col=" << m->colNum(); 
    s << " nz per col=" << (double)m->elmNum()/(double)m->colNum(); 
    AzTimeLog::print(s.c_str(), out);  
  }   
  /*-------------------------------------------------------------------------*/  
  virtual void _gen_data(const int *, int, AzPmat *, int) const { no_support("AzpData_sparse", "_gen_data(dense)"); }
  virtual void _gen_data(int, int, AzPmat *, int) const { no_support("AzpData_sparse", "_gen_data(dense) for test"); }
  virtual void _gen_data(const int *, int, AzPmatVar *, int) const { no_support("AzpData_sparse", "_gen_data(dense variable-sized)"); }
  virtual void _gen_data(int, int, AzPmatVar *, int) const { no_support("AzpData_sparse", "_gen_data(dense variable-sized) for test"); }     

  /*-------------------------------------------------------------------------*/    
  /*---       on-the-fly region generation (experimantal & slow)          ---*/  
  void gen_regions(const int *dxs, int dnum, AzPmatSpaVar &mv_out,
                   bool do_rowindex) const {       
    AzIntArr ia_dcolind; 
    int chk=dnum/10, ini = chk*500; 
    Az_bc bc(ini, ini*psz*3);
    for (int ix = 0; ix < dnum; ++ix) {
      ia_dcolind.put(bc.colNum());     
      int dx = dxs[ix]; 
      gen_bow_nobow(dx, bc); 
      ia_dcolind.put(bc.colNum()); 
      if (ix+1 == chk) bc.prepmem(ix+1, dnum); 
    }
    bc.commit(); 
    int row_num = (do_nobow) ? dic.size()*psz : dic.size(); 
    AzSmatbc m(row_num, bc.colNum()); 
    AzSmatbcVar mv; mv.reset(&m, &ia_dcolind); 
    mv.data_u()->set(bc.valarr(), bc.be()); 
    mv_out.set(mv, do_rowindex); 
  }
  /*-------------------------------------------------------------------------*/
  void gen_bow_nobow(int dx, Az_bc &bc) const {
    int col0 = msv_x.get_begin(dx), t_num = msv_x.get_end(dx)-col0; 
    int pnum = DIVUP(t_num+padding*2-psz, pstep) + 1; 
    int tx0 = -padding; 
    for (int pno = 0; pno < pnum; ++pno) {
      int tx1 = tx0 + psz; 
      AzIntArr ia; 
      for (int tx = MAX(0, tx0); tx < MIN(t_num, tx1); ++tx) {
        int num; const int *rows = msv_x.data()->rawcol_int(col0+tx, &num); 
        for (int ix = 0; ix < num; ++ix) {
          int row = rows[ix];  
          if (do_nobow) ia.put(row+(tx-tx0)*dic.size());            
          else {
            if (ia_nn != NULL && tx+(*ia_nn)[row] > tx1) continue; 
            ia.put(row);       
          }
        }
      }
      bc.unique_put(ia); 
      if (tx1 >= t_num+padding) break;
      tx0 += pstep;  
    }    
  }  
};   
#endif 

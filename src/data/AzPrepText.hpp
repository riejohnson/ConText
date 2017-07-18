/* * * * *
 *  AzPrepText.hpp 
 *  Copyright (C) 2014-2017 Rie Johnson
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

#ifndef _AZ_PREP_TEXT_HPP_
#define _AZ_PREP_TEXT_HPP_
 
#include "AzUtil.hpp"
#include "AzStrPool.hpp" 
#include "AzParam.hpp"
#include "AzPrint.hpp"
#include "AzSmat.hpp"
#include "AzDic.hpp"

class AzPrepText {
public:  

protected: 
  AzOut out; 
public:
  AzPrepText(const AzOut &_out) : out(_out) {}
  void gen_vocab(int argc, const char *argv[]) const; 
  void gen_regions(int argc, const char *argv[]) const; 
  void show_regions(int argc, const char *argv[]) const; 
  void gen_nbw(int argc, const char *argv[]) const; 
  void gen_nbwfeat(int argc, const char *argv[]) const; 
  void gen_b_feat(int argc, const char *argv[]) const; 
  void split_text(int argc, const char *argv[]) const;
  void merge_vocab(int argc, const char *argv[]) const;                    

  void adapt_word_vectors(int argc, const char *argv[]) const; 
  void write_wv_word_mapping(int argc, const char *argv[]) const; 
  
  /*-----*/                            
  void gen_regions_unsup(int argc, const char *argv[]) const; 
  void gen_regions_parsup(int argc, const char *argv[]) const;   
  void show_regions_XY(int argc, const char *argv[]) const;  
 
protected:                       
  /*---  for gen_regions  ---*/
  void gen_nobow_regions(int t_num, const AzDataArr<AzIntArr> &aia_nx_tok, 
                       int dic_sz, int pch_sz, int pch_step, int padding,  
                       bool do_allow_zero, int unkw, 
                       /*---  output  ---*/
                       Az_bc &bc, 
                       AzIntArr *ia_pos) /* patch position: may be NULL */ const; 
  void gen_nobow_regions_pos(int t_num, const AzDataArr<AzIntArr> &aia_tok, 
                       int dic_sz, int pch_sz, const AzIntArr &ia_tx0, int unkw, 
                       /*---  output  ---*/
                       Az_bc &bc, 
                       AzIntArr *ia_pos) const; 

  void gen_bow_regions(int t_num, const AzDataArr<AzIntArr> &aia_tokno, 
                       const AzIntArr &ia_nn, bool do_contain, 
                       int pch_sz, int pch_step, int padding,  
                       bool do_allow_zero, bool do_skip_stopunk, 
                       /*---  output  ---*/
                       Az_bc &bc, 
                       AzIntArr *ia_pos) /* patch position: may be NULL */ const; 
  void gen_bow_regions_pos(int t_num, const AzDataArr<AzIntArr> &aia_tokno, 
                       const AzIntArr &ia_nn, bool do_contain, 
                       int pch_sz, const AzIntArr &ia_tx0, 
                       /*---  output  ---*/
                       Az_bc &bc, 
                       AzIntArr *ia_pos) const;                   
  
  /*---  for show_regions  ---*/
  void _show_regions(const AzSmatVar *mv, const AzDic *dic, bool do_wordonly) const; 
  
  /*---  for nbw  ---*/
  static void scale(AzSmat *ms, const AzDvect *v);                          

  /*---  ---*/
  void write_Y(const AzSmat &m_y, const AzBytArr &s_y_fn, const AzBytArr *s_batch_id=NULL) const {
    write_Y(out, m_y, s_y_fn, s_batch_id); 
  }
  void write_X(const AzSmat &m_x, const AzBytArr &s_x_fn, const AzBytArr *s_batch_id=NULL) const {
    write_X(out, m_x, s_x_fn, s_batch_id); 
  }
  
  int add_unkw(AzDic &dic) const; 

  void union_vocab(const AzStrPool &sp_fns, AzStrPool &out_sp, bool do_1stfirst) const; 
  void join_vocab(const AzStrPool &sp_fns, AzStrPool &out_sp) const; 

  /*-----*/
  void write_regions(const Az_bc &bc, int row_num,
                           const AzIntArr &ia_dcolind, 
                           const AzBytArr &s_batch_id, 
                           const char *outnm, const char *ext) const {
    write_regions(out, bc, row_num, ia_dcolind, s_batch_id, outnm, ext); 
  }                             
             
  void gen_Y(const AzIntArr &ia_tokno, int dic_sz, const AzIntArr &ia_pos, 
             int xpch_sz, /* patch size used to generate X */
             int min_dist, int max_dist, int gap, bool do_nolr, 
             Az_bc &ybc) const; 
  void gen_Y_ngram_bow(const AzIntArr &ia_nn, 
                       const AzDataArr<AzIntArr> &aia_tokno, int dic_sz, 
                       const AzIntArr &ia_pos, 
                       int xpch_sz, /* patch size used to generate X */
                       int min_dist, int max_dist, int gap, bool do_nolr, 
                       Az_bc &ybc) const; 

  template <class Typ>
  void reduce_xy(int min_x, int min_y, int col_beg, 
                 Az_bc &xbc, Typ &ybc_c) const;                 

  /*---  for partially-supervised target  ---*/
  class feat_info { /* just to collect statistics ... */
  public:
    double min_val, max_val, sum, count; 
    feat_info() : min_val(99999999), max_val(-99999999), sum(0), count(0) {}
    void update(const AzIFarr &ifa) {
      if (ifa.size() <= 0) return; 
      min_val = MIN(min_val, ifa.findMin()); 
      max_val = MAX(max_val, ifa.findMax()); 
      count += ifa.size(); 
      sum += ifa.sum(); 
    }
    void show(AzBytArr &s) const {
      if (count <= 0) s << "No info"; 
      s << "min," << min_val << ",max," << max_val << ",avg," << sum/count << ",count," << count; 
    }    
  };  
  int gen_Y_ifeat(int top_num_each, int top_num_total, const AzSmat &m_feat, 
                   int tok_num, const AzIntArr &ia_pos, 
                   int xpch_sz, int min_dist, int max_dist, 
                   bool do_nolr, 
                   int f_pch_sz, int f_pch_step, int f_padding, 
                   Az_c &yc, 
                   feat_info fi[2]) const;   
  void set_ifeat(const AzSmat &m_feat, int top_num, 
                 int col, int offs, AzIFarr &ifa_ctx, feat_info fi[2]) const; 
  void write_Y_smatc(const AzSmatc &m_y, const AzIntArr &ia_dcolind, 
                                const AzBytArr &s_batch_id, const char *outnm, 
                                const char *y_ext) const; 
                 
  /*---  for show_regions_XY  ---*/                 
  void read_XY(const AzBytArr &s_rnm, const AzBytArr &s_ext, const AzBytArr &s_bid, AzSmatVar &mv) const; 
  void _show_regions_XY(const AzSmatVar &mv_x, const AzSmatVar &mv_y, 
                               const AzDic &xdic, const AzDic &y_dic, 
                               bool do_wordonly) const;                             
                               
public:
  /*---  for gen_vocab  ---*/
  static void put_in_voc(int nn, 
                  AzStrPool &sp_voc, 
                  const AzStrPool &sp_words, 
                  int wx, /* position in sp_words */
                  AZint8 count, 
                  bool do_stop_if_all, 
                  const AzStrPool *sp_stop, 
                  bool do_remove_number);   
  static AzByte gen_1byte_index(const AzStrPool *sp_words, int wx); 
  static int write_vocab(const char *fn, const AzStrPool *sp, 
                          int max_num, int min_count, bool do_write_count); 
  
  /*-----*/     
  static void gen_nobow_dic(const AzDic &inp_dic, int pch_sz, AzDic &out_dic);  
  static void check_batch_id(const AzBytArr &s_batch_id);  
  static void check_y_ext(const AzBytArr &s_y_ext, const char *eyec);                                  
  static void write_regions(const AzOut &out, const Az_bc &bc, int row_num,
                           const AzIntArr &ia_dcolind, const AzBytArr &s_batch_id, 
                           const char *outnm, const char *ext); 
  static void write_dic(const AzDic &dic, int row_num, const char *nm, const char *ext); 
  static void write_Y(const AzOut &out, const AzSmat &m_y, const AzBytArr &s_y_fn, const AzBytArr *s_batch_id=NULL); 
  static void write_X(const AzOut &out, const AzSmat &m_x, const AzBytArr &s_x_fn, const AzBytArr *s_batch_id=NULL); 

  static void check_size(const AzOut &out, const AzSmat &m);   
  
  static void zerovec_to_randvec(double wvec_rand_param, AzDmat &m_wvec); 
  static void read_word2vec(const char *fn, AzDic &dic, AzDmat &m_wvec, bool do_ignore_dup, bool is_text=false); 
}; 
#endif 
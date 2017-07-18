/* * * * *
 *  AzTools_text.hpp 
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

#ifndef _AZ_TOOLS_TEXT_HPP_
#define _AZ_TOOLS_TEXT_HPP_
 
#include "AzUtil.hpp"
#include "AzSmat.hpp"
#include "AzStrPool.hpp" 
#include "AzDic.hpp"

#define kw_do_allow_multi "MultiLabel"
#define kw_do_allow_nocat kw_do_allow_multi 
class AzTools_text {
public:                              
  static int scan_files_in_list(const char *inp_fn,  const char *ext,  
                                const AzOut &out, 
                                /*---  output  ---*/
                                AzStrPool *out_sp_list, /* may be NULL */
                                AzIntArr *ia_data_num); /* may be NULL; number of docs in each file */   
  static void tokenize(AzByte *data, int inp_len, /* used as work area */
                bool do_utf8dashes, bool do_lower, 
                AzStrPool &sp_tok, bool do_char=false, bool do_byte=false); 
  static void tokenize(AzByte *buff, int &len, 
                       const AzDic *dic, int nn, bool do_lower, bool do_utf8dashes,                     
                       AzIntArr *ia_tokno, bool do_char=false, bool do_byte=false); 
  static int tokenize(AzByte *buff, int &len, const AzDic *dic, AzIntArr &ia_nn, 
                       bool do_lower, bool do_utf8dashes,                   
                       AzDataArr<AzIntArr> &aia_tokno, bool do_char=false, bool do_byte=false); 
  static int replace_utf8dashes(AzByte *data, int len); 

  static void identify_tokens(const AzStrPool *sp_tok, int nn, const AzDic *dic, 
                              AzIntArr *ia_tokno) {
    if (nn == 1) identify_1gram(sp_tok, dic, ia_tokno); 
    else         identify_ngram(sp_tok, nn, dic, ia_tokno);     
  }                       
  static void identify_1gram(const AzStrPool *sp_tok, const AzDic *dic, 
                             AzIntArr *ia_tokno); 
  static void identify_ngram(const AzStrPool *sp_tok, int nn, const AzDic *dic_word, 
                             AzIntArr *ia_tokno);   
                             
  static void feat_to_text(const AzSvect *v_feat, 
                           const AzDic *dic_word, 
                           bool do_wordonly,                            
                           AzBytArr &s, /* output */                          
                           bool do_sort=false, 
                           int print_max=-1, 
                           char dlm='~'); /* for n-grams */  
  
  /* return the number of unknown  */
  static void count_words_get_cats(const AzOut &out, bool do_ignore_bad, 
                            bool do_count_unk, /* count unkown */
                            const char *fn, const char *txt_ext, const char *cat_ext, 
                            const AzDic &dic, const AzDic &dic_cat, 
                            int max_nn, 
                            bool do_allow_multi, bool do_allow_nocat, 
                            bool do_lower, bool do_utf8dashes, 
                            AzSmat *m_count, AzSmat *m_cat, 
                            bool do_no_cat=false); 
  static void parse_cats(const AzBytArr *s_cat, 
                         AzByte dlm,  /* e.g., | for, e.g., GSPO|M11|M12 */
                         bool do_allow_multicat, bool do_allow_nocat, 
                         const AzDic *dic_cat, AzIntArr *ia_cats, /*output */
                         int &multi_cat, int &no_cat, /* inout */                            
                         AzBytArr &s_err);  /* output */                 
                      
  static void get_bytes(const AzByte *data, int len, AzStrPool &sp_ch); 
  static void get_utf8chars(const AzByte *data, int len, AzStrPool &sp_ch, AzIntArr *ia_blk=NULL); 

  static void to_char_ngrams(AzByte *data, int inp_len, bool do_utf8dashes, bool do_lower, int nn, const AzDic &dic, 
                            AzIntArr &ia_chng, AzIntArr *out_ia_blk=NULL);
  static void to_char_ngrams(AzByte *data, int inp_len, bool do_utf8dashes, bool do_lower, int nn, 
                            AzStrPool &sp_chng, AzIntArr *out_ia_blk=NULL); 
  static void to_char_ngrams(int nn, const AzStrPool &sp_ch, const AzIntArr &ia_blk, 
                             AzStrPool &sp_chng, AzIntArr *out_ia_blk=NULL); 
       
protected:                          
  static void _parse_cats(const AzBytArr *s_cat, 
                          AzByte dlm,  /* e.g., | for, e.g., GSPO|M11|M12 */
                          bool do_allow_multicat, 
                          const AzDic *dic_cat, 
                          AzIntArr *ia_cats, /*output */
                          int &multi_cat, /* inout */
                          int &no_cat); /* inout */                            
}; 
#endif 

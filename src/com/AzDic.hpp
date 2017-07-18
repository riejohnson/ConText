/* * * * *
 *  AzDic.hpp 
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

#ifndef _AZ_DIC_HPP_
#define _AZ_DIC_HPP_

#include "AzStrPool.hpp"
#include "AzTools.hpp"

/**************************************************************/
class AzDic {
protected:
  AzStrPool sp_words; /* in the original order; not searchable */
  AzStrPool sp_sorted;  /* sorted and pointing to sp_words; searchable */
  
  static const int reserved_len=64; 
  static const int version=0;   
public:
  AzDic() {}
  AzDic(const char *fn, bool do_ignore_dup=false, bool do_allow_blank=false) {  
    reset(fn, do_ignore_dup, do_allow_blank); 
  }
  AzDic(const AzBytArr *s_fn, bool do_ignore_dup=false, bool do_allow_blank=false) {
    reset(s_fn->c_str(), do_ignore_dup, do_allow_blank); 
  }
  AzDic(const AzStrPool *sp, bool do_ignore_dup=false) {
    reset(sp, do_ignore_dup); 
  }
  int size() const { return sp_sorted.size(); }
  void reset() {
    sp_words.reset(); 
    sp_sorted.reset(); 
  }
  void reset(const AzDic *dic2) {
    if (dic2 == NULL) reset(); 
    else {
      sp_words.reset(&dic2->sp_words); 
      sp_sorted.reset(&dic2->sp_sorted); 
    }
  }
  void readText(const char *fn, bool do_ignore_dup=false, bool do_allow_blank=false) { reset(fn, do_ignore_dup, do_allow_blank); }
  void reset(const char *fn, bool do_ignore_dup=false, bool do_allow_blank=false) {  
    sp_words.reset(); sp_sorted.reset(); 
    sp_words.reset(100000, 10);   
    AzByte dlm = '\t'; 
    bool do_count = true; 
    AzTools::readList(fn, &sp_words, &dlm, do_count, do_allow_blank);
    _index(do_ignore_dup); 
  }
  void reset(const AzStrPool *sp, bool do_ignore_dup=false) {
    sp_words.reset(sp); 
    _index(do_ignore_dup); 
  }
  void cut(int num) {
    if (num < 0 || num >= sp_words.size()) return; 
    AzIntArr ia_keep; ia_keep.range(0, num); 
    sp_words.reduce(ia_keep.point(), ia_keep.size()); 
    _index(); 
  }
  inline void reduce(const AzIntArr *ia) { 
    if (ia == NULL) throw new AzException("AzDic::reduce", "null input"); 
    reduce(ia->point(), ia->size()); 
  }
  void reduce(const int *exs, int exs_num) { /* only keep these */
    sp_words.reduce(exs, exs_num); 
    _index(); 
  }
  int find(const char *word) const {
    if (sp_sorted.size() <= 0) return -1; 
    int sorted_id = sp_sorted.find(word); 
    if (sorted_id < 0) return sorted_id; 
    return sp_sorted.getValue(sorted_id); 
  }
  int find_ngram(const AzStrPool *sp_words, int wx, /* position in sp_words */
                 int nn) const {
    AzX::throw_if_null(sp_words, "AzDic::find_ngram", "sp_words"); 
    if (wx < 0 || wx+nn > sp_words->size() || size() <= 0) return -1; 
    AzBytArr s;   
    sp_words->compose_ngram(s, wx, nn); 
    return find(s.c_str()); 
  } 
  const char *get(int id) const {
    if (id < 0) return ""; 
    return sp_words.c_str(id); 
  }
  void get(int id, AzBytArr *s_word) const {
    s_word->reset(); 
    if (id < 0) return; 
    sp_words.get(id, s_word); 
  }
  
  template <class V>
  void vect_to_str(const V *v, AzBytArr *s, const char *dlm=" ", bool do_show_null=false) const {
    AzIntArr ia_row; 
    v->nonZeroRowNo(&ia_row); 
    get(ia_row, s, dlm, do_show_null); 
  }
  inline void get(const AzIntArr &ia_index, AzBytArr *s, const char *dlm=" ", bool do_show_null=false) const {
    get(ia_index.point(), ia_index.size(), s, dlm, do_show_null); 
  }
  inline void get(const int *index, int num, AzBytArr *s, const char *dlm=" ", bool do_show_null=false) const {
    int len = s->length(); 
    int ix; 
    for (ix = 0; ix < num; ++ix) {
      if (index[ix] >= 0) {
        if (ix > 0) s->c(dlm); 
        s->c(c_str(index[ix]));      
      }
      else if (do_show_null) {
        if (ix > 0) s->c(dlm); 
        s->c("*n*");       
      }
    }
  }  
  const char *c_str(int id) const {
    return sp_words.c_str(id); 
  }
  AZint8 count(int id) const {
    return sp_words.getCount(id); 
  }
  /*---  returns mapped  ---*/
  int map_to(const AzDic &dic2, AzIntArr &ia_1to2) const {
    int mapped = 0; 
    ia_1to2.reset(size(), -1); 
    int *_1to2 = ia_1to2.point_u(); 
    for (int ix = 0; ix < size(); ++ix) {
      _1to2[ix] = dic2.find(c_str(ix)); 
      if (_1to2[ix] >= 0) ++mapped; 
    }
    return mapped; 
  } 
  bool is_same(const AzDic *dic2) const {
    if (dic2 == NULL) {
      return (sp_words.size() == 0); 
    }
    return sp_words.has_same_strings_in_same_order(&dic2->sp_words); 
  }
  void writeText(AzFile *file, bool do_count) const {
    int ix; 
    for (ix = 0; ix < sp_words.size(); ++ix) {
      AzBytArr s; get(ix, &s); 
      if (do_count) s << "\t" << (double)sp_words.getCount(ix); 
      s.nl(); 
      s.writeText(file); 
    }
  }
  void writeText(const char *fn, bool do_count=false) const {
    AzFile file(fn); 
    file.open("wb"); 
    writeText(&file, do_count); 
    file.close(true); 
  }
  inline void writeText(const AzBytArr *s_fn, bool do_count=false) const {
    writeText(s_fn->c_str(), do_count); 
  }
  void write(AzFile *file) {
    AzTools::write_header(file, version, reserved_len);     
    sp_words.write(file); 
    sp_sorted.write(file); 
  }
  void read(AzFile *file) {
    int my_version = AzTools::read_header(file, reserved_len);     
    sp_words.read(file); 
    sp_sorted.read(file); 
  }
  void append(const AzDic *dic2) {
    sp_words.append(&dic2->sp_words); 
    _index();  
  }
  void append(const AzStrPool *sp2) {
    sp_words.append(sp2); 
    _index();   
  }  
  void add_prefix(const char *pref) {
    sp_words.add_prefix(pref); 
    sp_sorted.reset(&sp_words); 
    sp_sorted.commit(); 
  }
  void copy_to(AzStrPool *sp) const {
    sp->reset(&sp_words); 
  }
  const AzStrPool &ref() const { return sp_words; }
  int wordNum(int idx) const {
    const AzByte *ptr = sp_words.point(idx); 
    int len = sp_words.getLen(idx); 
    int count = 0; 
    for (int offs = 0; offs < len; ++offs) if (*(ptr+offs) == ' ') ++count; 
    return count+1; 
  }      
  
  /*-------------------------------------------------------------------------*/
  /* rearrange columns of m_xi (corresponding to dic_xi) in the order of dic_x */
  template <class M> /* AzSmat or AzDmat */
  static int rearrange_cols(const AzDic &dic_x, const AzDic &dic_xi, /* input */
                            M &m_xi, /* inout */
                            bool do_die_if_nothing_is_mapped=true) {
    const char *eyec = "AzDic::rearrange_cols"; 
    AzX::throw_if((dic_xi.size() != m_xi.colNum()), AzInputError, eyec, "#columns and size of vocabulary don't match"); 
    AzIntArr ia_x_to_xi; 
    int mapped = dic_x.map_to(dic_xi, ia_x_to_xi); 
    AzX::throw_if((do_die_if_nothing_is_mapped && mapped == 0), eyec, "Nothing was mapped?!"); 
    bool do_zero_negaindex = true;   
    M m_xi2; m_xi2.set(&m_xi, ia_x_to_xi.point(), ia_x_to_xi.size(), do_zero_negaindex); 
    m_xi.set(&m_xi2); 
    return mapped; 
  } 
  void get_n(AzIntArr &ia_nn) const {
    ia_nn.prepare(sp_words.size()); 
    for (int ex=0;ex<sp_words.size();++ex) ia_nn.put(sp_words.get_n(ex));     
  }
  int get_n(int ex) const { return sp_words.get_n(ex); }
  int get_max_n() const { return sp_words.get_max_n(); }
  int get_min_n() const { return sp_words.get_min_n(); }  
  void copy_words_only_to(AzStrPoolc &spc) const { spc.reset(sp_words, '\n'); }

  void check_zero_len(const char *msg="") const {
    const char *eyec = "AzDic::check_zero_len"; 
    for (int ix = 0; ix < size(); ++ix) {
      AzX::throw_if((strlen(c_str(ix)) == 0), AzInputError, eyec, msg, "zero-length entry"); 
    }  
  }
  
protected: 
  void _index(bool do_ignore_dup=false) {
    sp_sorted.reset(&sp_words);     
    int ix; 
    for (ix = 0; ix < sp_words.size(); ++ix) {
      sp_sorted.setValue(ix, ix); 
    }
    sp_sorted.commit(do_ignore_dup); 
  }   
};  

#define AzDicc AzStrPoolc
#endif 
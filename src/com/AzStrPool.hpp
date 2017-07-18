/* * * * *
 *  AzStrPool.hpp 
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

#ifndef _AZ_STR_POOL_HPP_
#define _AZ_STR_POOL_HPP_

#include "AzUtil.hpp"

typedef struct {
public:
  AZint8 offs; 
  int len; 
  AZint8 count;
  int value; 
  const AzByte *bytes; /* we need this for qsort */
} AzSpEnt; 

typedef struct {
  int begin; 
  int end; 
  int min_len; 
  int max_len; 
} AzSpIndex; 

//! Store byte arrays or strings.  Searchable after committed. 
//  max(#entry): 2^31-1
//  max(total length of data): 2^63-1 (since June 2014)
class AzStrPool {
protected:
  int cap; 
  AzSpEnt *ent; 
  AzBaseArray<AzSpEnt> a_ent; 
  int ent_num; 

  AzByte *data; 
  AzBaseArray<AzByte,AZint8> a_data; 
  AZint8 data_len; 

  bool isCommitted;  

  AzSpIndex *my_idx; 
  AzBaseArray<AzSpIndex> a_index; 

  int init_ent_num;
  AZint8 init_data_len; 

  #define _AzStrPool_init_ cap(1024*1024),ent_num(0),ent(NULL),data(NULL),\
                           data_len(0),isCommitted(true),my_idx(NULL),\
                           init_ent_num(65536),init_data_len(655360)
  void initialize() {
    cap = 1024*1024; ent_num = 0; ent = NULL; data = NULL; 
    data_len = 0; isCommitted = true; my_idx = NULL;   
    init_ent_num = 65536; init_data_len = 655360; 
  } 
public:
  AzStrPool() : _AzStrPool_init_ {}
  AzStrPool(int init_num, AZint8 avg_data_len) : _AzStrPool_init_ {
    init_ent_num = MAX(init_num, 64); 
    init_data_len = init_ent_num * MAX(1,avg_data_len);
  }
  AzStrPool(const AzStrPool *inp_sp) : _AzStrPool_init_ { _copy(inp_sp); }
  AzStrPool(const AzStrPool &inp_sp) : _AzStrPool_init_ { _copy(&inp_sp); }
  AzStrPool(AzFile *file) : _AzStrPool_init_ {_read(file); }
  ~AzStrPool() {}
  AzStrPool & operator =(const AzStrPool &inp) { /* prohibit assign operator */
    if (this == &inp) return *this; 
    throw new AzException("AzStrPool =", "no support"); 
  }
  AzStrPool(const char *str1, const char *str2=NULL, const char *str3=NULL, 
           const char *str4=NULL, const char *str5=NULL) : _AzStrPool_init_ {
    init_ent_num = 32; init_data_len = 256; 
    put(str1, str2, str3, str4, str5);     
  }
  void reset(); 
  void reset(const AzStrPool *inp) { reset(); _copy(inp); }
  
  inline void reset(int init_num, AZint8 avg_data_len) {
    reset(); 
    init_ent_num = MAX(init_num, 64); 
    init_data_len = init_ent_num * MAX(1,avg_data_len); 
  }
  inline void reset_nocap(int init_num, AZint8 avg_data_len) { reset(init_num, avg_data_len, -1); }
  inline void reset(int init_num, AZint8 avg_data_len, int _cap) {
    reset(init_num, avg_data_len); 
    cap = _cap; 
  }
  
  void read(AzFile *file) { reset(); _read(file); }
  void write(AzFile *file) const; 
  void write(const char *fn) const { AzFile::write(fn, this); }
  void read_compact(AzFile *file); 
  void write_compact(AzFile *file) const; 
  
  int put(const AzByte *bytes, int bytes_len, AZint8 count=1, int value=-1); 
  virtual int put(const char *str, AZint8 count=1) { return put((AzByte *)str, Az64::cstrlen(str), count); }
  int put(const AzBytArr *byteq, AZint8 count=1) { return put(byteq->point(), byteq->getLen(), count); }
  void put(const char *str1, const char *str2, const char *str3=NULL, 
           const char *str4=NULL, const char *str5=NULL) {
    if (str1 != NULL) put(str1);         
    if (str2 != NULL) put(str2); 
    if (str3 != NULL) put(str3); 
    if (str4 != NULL) put(str4); 
    if (str5 != NULL) put(str5);     
  }

  inline int putv(const AzBytArr *bq, int value) {
    AZint8 count = 1; 
    return put(bq->point(), bq->getLen(), count, value); 
  }        
  inline virtual int putv(const char *str, int value) {
    AZint8 count = 1; 
    return put((AzByte *)str, Az64::cstrlen(str), count, value); 
  }  
  inline int getValue(int ent_no) const {
    checkRange(ent_no, "AzStrPool::getValue"); 
    return ent[ent_no].value; 
  }
  inline void setValue(int ent_no, int value) {
    checkRange(ent_no, "AzStrPool::setValue"); 
    ent[ent_no].value = value; 
  }
  void add_to_value(int added) {
    for (int ex = 0; ex < ent_num; ++ex) ent[ex].value += added; 
  }
  void setCount(int ent_no, AZint8 count) {
    checkRange(ent_no, "AzStrPool::setCount"); 
    ent[ent_no].count = count; 
  }
  void put(AzDataArray<AzBytArr> *aStr) {
    AzX::throw_if_null(aStr, "AzStrPool::put(aStr)"); 
    int num = aStr->size(); 
    for (int ix = 0; ix < num; ++ix) put(aStr->point(ix)); 
  }
  void put(const AzStrPool *inp) {
    AzX::throw_if_null(inp, "AzStrPool::put(AzStrPool *)"); 
    int num = inp->size(); 
    for (int ix = 0; ix < num; ++ix) {
      int len; 
      const AzByte *str = inp->point(ix, &len); 
      AZint8 count = inp->getCount(ix); 
      int value = inp->getValue(ix); 
      put(str, len, count, value); 
    }
  }
  inline void append(const AzStrPool *inp) { put(inp); }
  virtual void add_prefix(const char *str) {
    AzBytArr s_str(str); 
    add_prefix(&s_str); 
  }
  virtual void add_prefix(const AzBytArr *s_str) {
    if (size() <= 0) return; 
    AzStrPool sp_tmp(this); 
    reset();
    reset(sp_tmp.size(), sp_tmp.data_len / sp_tmp.size() + s_str->length()); 
    for (int ix = 0; ix < sp_tmp.size(); ++ix) {
      AzBytArr s(s_str->c_str(), sp_tmp.c_str(ix)); 
      put(s.point(), s.length(), sp_tmp.getCount(ix), sp_tmp.getValue(ix));     
    }
  }

  void commit(bool do_ignore_value_conflict=false); 
  void build_index(); 
  int size() const { return ent_num; }

  const AzByte *point(int ent_no, int *out_len) const; 
  const AzByte *point(int ent_no) const; 
  virtual const char *c_str(int ent_no) const { return (char *)point(ent_no); }
  int getLen(int ent_no) const; 
  AZint8 getCount(int str_no) const; 

  void getAllCount(AzIFarr *ifa_count) const {
    AzX::throw_if_null(ifa_count, "AzStrPool::getAllCount"); 
    ifa_count->reset(); ifa_count->prepare(ent_num); 
    for (int ex = 0; ex < ent_num; ++ex) ifa_count->put(ex, (double)ent[ex].count); 
  }
  double getAllCount() const {
    double count = 0; 
    for (int ex = 0; ex < ent_num; ++ex) count += (double)ent[ex].count; 
    return count; 
  }
  
  void removeEntry(int ent_no); 
  void removeEntries(const AzIntArr *ia_rmv_ex) {
    if (ia_rmv_ex == NULL) return; 
    int num; 
    const int *rmv_ex = ia_rmv_ex->point(&num); 
    for (int ix = 0; ix < num; ++ix) removeEntry(rmv_ex[ix]); 
  }

  int find(const AzByte *bytes, int bytes_len) const;  
  int find(const AzBytArr *s) const { 
    AzX::throw_if_null(s, "AzStrPool::find(AzBytArr *)"); 
    return find(s->point(), s->getLen()); 
  }
  virtual int find(const char *str) const { return find((AzByte *)str, Az64::cstrlen(str)); }
  int find_anyway(const AzByte *bytes, int bytes_len) const; /* find even if it's slow ... */
  int find_anyway(const char *str) const { return find_anyway((AzByte *)str, Az64::cstrlen(str)); }
  
  virtual void dump(const AzOut &file, const char *header) const; 

  virtual void get(int ent_no, AzBytArr *s) const {
    AzX::throw_if_null(s, "AzStrPool::get"); 
    int len; const AzByte *bytes = point(ent_no, &len); 
    s->concat(bytes, len); 
  }
  virtual void get(const int *exs, int num, const char *dlm, AzBytArr *s) const {
    for (int ix = 0; ix < num; ++ix) {
      if (ix > 0) s->c(dlm); 
      if (exs[ix] < 0) s->c("*null*"); 
      else             s->c(c_str(exs[ix])); 
    }
  }

  bool isThisSearchReady() const { return isCommitted; }
  inline bool isThisCommitted() const { return isCommitted; }

  void reduce(int min_count); 

  virtual bool has_same_strings_in_same_order(const AzStrPool *sp) const {
    if (sp->size() != size()) return false; 
    for (int ix = 0; ix < sp->size(); ++ix) if (strcmp(sp->c_str(ix), c_str(ix)) != 0) return false; 
    return true; 
  }
 
  double keep_topfreq(int keep_num, bool do_keep_ties, bool do_release_mem=false); 
  void reduce(const AzIntArr *ia, bool do_release_mem=false) {
    AzX::throw_if_null(ia, "AzStrPool::reduce"); 
    reduce(ia->point(), ia->size(), do_release_mem); 
  }
  void reduce(const int *exs, int exs_num, bool do_release_mem=false) { /* must be sorted */
    const char *eyec = "AzStrPool::reduce(exs,exs_num)"; 
    AzX::throw_if(do_release_mem, eyec, "No support for do_release_mem yet"); 
    int new_ex = 0; 
    for (int ix = 0; ix < exs_num; ++ix) {
      int ex = exs[ix]; 
      AzX::throw_if(ex < 0 || ex >= ent_num, eyec, "index is out of range"); 
      AzX::throw_if(ix > 0 && ex <= exs[ix-1], eyec, "index array must be sorted and have no duplication"); 
      if (ex != new_ex) ent[new_ex] = ent[ex];       
      ++new_ex;
    }
    ent_num = new_ex; 
    if (isCommitted) build_index(); 
  }

  void compose_ngram(AzBytArr &s_ngram, /* output: n-gram starting from [wx] */
                     int wx, int nn, 
                     bool do_stop_if_all=false, 
                     const AzStrPool *sp_stop=NULL, 
                     bool do_remove_number=false) const {
    if (do_stop_if_all) _compose_ngram_stop_if_all(s_ngram, wx, nn, sp_stop, do_remove_number); 
    else                _compose_ngram_stop_if_one(s_ngram, wx, nn, sp_stop, do_remove_number); 
  }                       
  int get_n(int ex) const; 
  int get_max_n() const;                      
  int get_min_n() const; 
  
protected:
  void _read(AzFile *file); 
  void _copy(const AzStrPool *sp2); 
  int genIndexKey(const AzByte *bytes, int bytes_len) const; 

  int inc_ent(); 
  AZint8 inc_data(AZint8 min_inc); 

  inline void checkRange(int ent_no, const char *eyec) const {
    AzX::throw_if (ent_no < 0 || ent_no >= ent_num, eyec, "out of range"); 
  }
  
  void _compose_ngram_stop_if_one(AzBytArr &s_ngram, int wx, int nn, const AzStrPool *sp_stop, 
                             bool do_remove_number) const;  
  void _compose_ngram_stop_if_all(AzBytArr &s_ngram, int wx, int nn, const AzStrPool *sp_stop, 
                             bool do_remove_number) const;                             
}; 

/**************************************************************/
/* c for compact */
class AzStrPoolc {
protected:
  int num; 
  AzBaseArr<AzByte,AZint8> data; 
public: 
  AzStrPoolc() : num(0) {}
  AzStrPoolc(const AzStrPoolc &sp_c) : num(0) { reset(sp_c); }
  AzStrPoolc & operator =(const AzStrPoolc &sp_c) { if (this != &sp_c) reset(sp_c); return *this; }
  AzStrPoolc(const AzStrPool &sp, AzByte dlm) : num(0) { reset(sp, dlm); }
  int size() const { return num; }
  void reset() { num = 0; data.free(); }
  void reset(const AzStrPoolc &inp) {
    num = inp.num; 
    data.free_alloc(inp.data.size(), "AzStrPoolc::reset(copy)", "data"); 
    memcpy(data.point_u(), inp.data.point(), inp.data.size()); 
  }
  void reset(const AzStrPool &sp, AzByte dlm); 
  bool is_same(const AzStrPoolc &inp) const {
    if (data.size() != inp.data.size()) return false; 
    return (memcmp(data.point(), inp.data.point(), data.size()) == 0);     
  }
  void write(AzFile *file) const { 
    file->writeInt(num); 
    file->writeInt8(data.size()); 
    file->writeBytes(data.point(), data.size()); 
  }
  void read(AzFile *file) {
    num = file->readInt(); 
    AZint8 len = file->readInt8(); 
    data.free_alloc(len, "AzStrPoolc::read", "data"); 
    file->readBytes(data.point_u(), data.size()); 
  }
  void writeText(const char *fn) const {
    AzFile file(fn); file.open("wb"); 
    file.writeBytes(data.point(), data.size()); 
    file.close(true);     
  } 
}; 
#endif 

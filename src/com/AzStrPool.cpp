/* * * * *
 *  AzStrPool.cpp 
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
#include "AzStrPool.hpp"
#include "AzPrint.hpp"

int azsp_compare_Ent(const void *v1, const void *v2); 

static int index_size = 65536; 
static int version = 0, reserved_len = 32; /*6/20/2017 */
/*------------------------------------------------------------------*/
/* For making files comparable by diff by avoiding fillers in entries. */
void AzStrPool::write_compact(AzFile *file) const {
  file->writeInt(version);AzBytArr s;s.reset(reserved_len,0);file->writeBytes(s.point(),s.length());  
  file->writeInt(ent_num); 
  file->writeInt8(data_len); 
  file->writeBool(isCommitted); 
  if (data_len > 0) file->writeBytes(data, data_len); 
  for (int ex = 0; ex < ent_num; ++ex) {
    AzSpEnt *ep = &ent[ex];      
    file->writeInt8(ep->offs); 
    file->writeInt(ep->len); 
    file->writeInt8(ep->count); 
    file->writeInt(ep->value); 
  }
}

/*------------------------------------------------------------------*/
void AzStrPool::read_compact(AzFile *file) {
  const char *eyec = "AzStrPool::read_compact(file)"; 
  reset(); 
  int my_version=file->readInt();AzBytArr s;s.reset(reserved_len,0);file->readBytes(s.point_u(),s.length());
  ent_num = file->readInt(); 
  data_len = file->readInt8(); 
  isCommitted = file->readBool(); 
  if (data_len > 0) {
    a_data.alloc(&data, data_len, eyec, "data"); 
    file->seekReadBytes(-1, data_len, data); 
  }
  a_ent.alloc(&ent, ent_num, eyec, "ent"); 
  for (int ex = 0; ex < ent_num; ++ex) {
    AzSpEnt *ep = &ent[ex];     
    ep->offs = file->readInt8(); 
    ep->len = file->readInt(); 
    ep->count = file->readInt8(); 
    ep->value = file->readInt(); 
    ep->bytes = data + ep->offs; 
  }
  if (isCommitted) build_index(); 
}

/*------------------------------------------------------------------*/
void AzStrPool::_read(AzFile *file) {
  const char *eyec = "AzStrPool::read(file)"; 
  ent_num = file->readInt(); 
  data_len = file->readInt8(); 
  isCommitted = file->readBool(); 
  if (data_len > 0) {
    a_data.alloc(&data, data_len, eyec, "data"); 
    file->seekReadBytes(-1, data_len, data); 
  }
  a_ent.alloc(&ent, ent_num, eyec, "ent"); 
  file->seekReadBytes(-1, sizeof(ent[0])*ent_num, ent); 
  for (int ex = 0; ex < ent_num; ++ex) {
    AzSpEnt *ep = &ent[ex]; 
    ep->bytes = data + ep->offs; 
  }
  if (isCommitted) build_index(); 
}

/*------------------------------------------------------------------*/
void AzStrPool::_copy(const AzStrPool *sp2) {
  const char *eyec = "AzStrPool::_copy"; 
  AzX::throw_if_null(sp2, eyec); 
  ent_num = sp2->ent_num; 
  a_ent.alloc(&ent, ent_num, eyec, "ent"); 
  if (ent_num > 0) memcpy(ent, sp2->ent, sizeof(ent[0]) * ent_num); 

  data_len = sp2->data_len; 
  a_data.alloc(&data, data_len, eyec, "data"); 
  memcpy(data, sp2->data, data_len); 

  isCommitted = sp2->isCommitted; 
  if (isCommitted) {
    for (int ex = 0; ex < ent_num; ++ex) ent[ex].bytes = data + ent[ex].offs; 
    build_index(); 
  }
}

/*------------------------------------------------------------------*/
void AzStrPool::write(AzFile *file) const {
  file->writeInt(ent_num); 
  file->writeInt8(data_len); 
  file->writeBool(isCommitted); 
  if (data_len > 0) file->writeBytes(data, data_len); 
  file->writeBytes(ent, (AZint8)sizeof(ent[0])*ent_num); 
}

/*------------------------------------------------------------------*/
int AzStrPool::put(const AzByte *bytes, int bytes_len, 
                   AZint8 count, int value) {
  const char *eyec = "AzStrPool::put"; 
  AzX::throw_if(bytes_len < 0, eyec, "negative length"); 
  if (isCommitted) {
    int ex = find(bytes, bytes_len); 
    if (ex != AzNone) { 
      AzX::throw_if(ent[ex].value != value, eyec, "value conflict"); 
      ent[ex].count += count; 
      return ex; 
    }
  }

  isCommitted = false; 

  int ent_num_max = a_ent.size(); 
  if (ent_num >= ent_num_max) {
    ent_num_max += inc_ent();     
    a_ent.realloc(&ent, ent_num_max, eyec, "ent"); 
  }
  AZint8 data_len_max = a_data.size(); 
  if (data_len + bytes_len + 1 > data_len_max) {
    data_len_max += inc_data(data_len+bytes_len+1-data_len_max); 
    a_data.realloc(&data, data_len_max, eyec, "data"); 
  }
  AzSpEnt *ep = &ent[ent_num]; 
  ep->offs = data_len; 
  ep->len = bytes_len; 
  ep->count = count; 
  ep->value = value; 
  ep->bytes = NULL; /* set only after sorted */
  memcpy(data + ep->offs, bytes, bytes_len); 
  data_len += bytes_len; 
  *(data + data_len) = '\0'; 
  ++data_len; 
  ++ent_num; 
  return (ent_num - 1); 
}

/*------------------------------------------------------------------*/
const AzByte *AzStrPool::point(int ent_no, int *out_bytes_len) const {
  *out_bytes_len = 0; 
  checkRange(ent_no, "AzStrPool::point"); 

  *out_bytes_len = ent[ent_no].len; 
  return data + ent[ent_no].offs; 
}

/*------------------------------------------------------------------*/
const AzByte *AzStrPool::point(int ent_no) const {
  checkRange(ent_no, "AzStrPool::point"); 
  return data + ent[ent_no].offs; 
}

/*------------------------------------------------------------------*/
int AzStrPool::getLen(int ent_no) const {
  checkRange(ent_no, "AzStrPool::getLen"); 
  return ent[ent_no].len; 
}

/*------------------------------------------------------------------*/
AZint8 AzStrPool::getCount(int ent_no) const {
  checkRange(ent_no, "AzStrPool::getCount"); 
  return ent[ent_no].count; 
}

/*------------------------------------------------------------------*/
void AzStrPool::removeEntry(int ent_no) {
  checkRange(ent_no, "AzStrPool::removeEntry"); 
  for (int ex = ent_no + 1; ex < ent_num; ++ex) ent[ex-1] = ent[ex]; 
  --ent_num; 
  a_index.free(&my_idx); 
}

/*------------------------------------------------------------------*/
void AzStrPool::commit(bool do_ignore_value_conflict) {
  const char *eyec = "AzStrPool::commit"; 
  isCommitted = true; 
  if (ent_num == 0) return; 
  for (int ex = 0; ex < ent_num; ++ex) ent[ex].bytes = data + ent[ex].offs; 

  /*-----  sort entries by strings  -----*/
  qsort(ent, ent_num, sizeof(ent[0]), azsp_compare_Ent); 

  int out_len = 0; 
  int new_id = 0; 
  for (int ex = 0; ex < ent_num; ) {
    out_len += (ent[ex].len + 1);
    if (new_id != ex) ent[new_id] = ent[ex]; 
    AZint8 count = 0; 
    for ( ; ex < ent_num; ++ex) {
      if (azsp_compare_Ent(&ent[ex], &ent[new_id]) != 0) break;
      if (ent[new_id].value != ent[ex].value) {
        AzX::throw_if(!do_ignore_value_conflict, eyec, "value conflict", c_str(ex)); 
      }
      count += ent[ex].count; 
    }
    ent[new_id].count = count; 
    ++new_id; 
  }
  ent_num = new_id; 

  /*-----  -----*/
  if (out_len > 0) {
    AzByte *new_data = NULL; 
    AzBaseArray<AzByte,AZint8> a_new_data; 
    a_new_data.alloc(&new_data, out_len, eyec, "new_data"); 

    AZint8 new_data_len = 0; 
    for (int ex = 0; ex < ent_num; ++ex) {
      AzSpEnt *ep = &ent[ex]; 
      memcpy(new_data + new_data_len, data + ep->offs, ep->len); 
      ep->offs = new_data_len;  
      ep->bytes = new_data + ep->offs; 

      AzByte *last_byte_ptr = new_data + ep->offs + ep->len; 
      *last_byte_ptr = 0; 

      new_data_len += (ep->len + 1); 
    }

    a_data.transfer_from(&a_new_data, &data, &new_data, eyec); 
    data_len = new_data_len; 
  }
  else {
    a_data.free(&data); 
    data_len = 0; 
  }

  /*-----  index it  -----*/
  build_index(); 
}

/*------------------------------------------------------------------*/
int AzStrPool::genIndexKey(const AzByte *bytes, int bytes_len) const {
  if (index_size == 256) return (int)bytes[0]; 
  int key = 0; 
  if (bytes_len >= 1) key = 256 * bytes[0]; 
  if (bytes_len >= 2) key += bytes[1]; 
  return key; 
}

/*------------------------------------------------------------------*/
void AzStrPool::build_index() {
  const char *eyec = "AzStrPool::build_index"; 
  AzX::throw_if(!isCommitted, eyec, "not sorted yet"); 
  a_index.free(&my_idx); 
  if (ent_num < index_size) return; 

  a_index.alloc(&my_idx, index_size, eyec, "my_idx"); 
  for (int ix = 0; ix < index_size; ++ix) {
    AzSpIndex *xp = &my_idx[ix]; 
    xp->begin = xp->end = AzNone; 
    xp->min_len = xp->max_len = 0; 
  }

  for (int ex = 0; ex < ent_num; ++ex) {
    int key = genIndexKey(ent[ex].bytes, ent[ex].len); 
    AzSpIndex *xp = &my_idx[key]; 
    if (xp->begin == AzNone) {
      xp->begin = ex; 
      xp->min_len = ent[ex].len; 
      xp->max_len = ent[ex].len; 
    }
    xp->end = ex + 1; 

    xp->min_len = MIN(xp->min_len, ent[ex].len); 
    xp->max_len = MAX(xp->max_len, ent[ex].len); 
  }
}

/*------------------------------------------------------------------*/
int AzStrPool::find(const AzByte *bytes, int bytes_len) const {
  const char *eyec = "AzStrPool::find"; 
  AzX::throw_if(!isCommitted, eyec, "sort first"); 

  int lx = 0, hx = ent_num - 1; 
  if (my_idx != NULL) {
    int key = genIndexKey(bytes, bytes_len); 
    AzSpIndex *xp = &my_idx[key]; 
    if (xp->begin >= xp->end || 
        bytes_len < xp->min_len || bytes_len > xp->max_len) {
      /* ++fail; */
      return AzNone; 
    }
    lx = xp->begin; 
    hx = xp->end - 1; 
  }

  AzSpEnt inp; 
  inp.bytes = bytes; 
  inp.len = bytes_len; 
 
  for ( ; ; ) {
    if (lx > hx) break;
    int mx = (lx + hx) / 2; 
    int cmp = azsp_compare_Ent(&inp, &ent[mx]); 
    if (cmp < 0) hx = mx - 1; 
    else if (cmp > 0) lx = mx + 1; 
    else {
      /* ++hit; */
      return mx; 
    }
  }

  /*++fail; */
  return AzNone; 
}

int AzStrPool::find_anyway(const AzByte *bytes, int len) const { /* find even if it's slow ... */
  if (isCommitted) return find(bytes, len); 
  for (int ix = 0; ix < size(); ++ix) {
    if (ent[ix].len == len && memcmp(data+ent[ix].offs, bytes, len) == 0) return ix; 
  }
  return AzNone; 
}

/*------------------------------------------------------------------*/
void AzStrPool::reset() {
  a_ent.free(&ent); ent_num = 0;  
  a_data.free(&data); data_len = 0; 
  a_index.free(&my_idx); 
  initialize(); 
}

/*------------------------------------------------------------------*/
int azsp_compare_Ent(const void *v1, const void *v2) {
  AzSpEnt *p1 = (AzSpEnt *)v1; 
  AzSpEnt *p2 = (AzSpEnt *)v2; 
  int cmp_len = MIN(p1->len, p2->len); 
  int cmp = memcmp(p1->bytes, p2->bytes, cmp_len); 
  if (cmp != 0) return cmp; 
  if (p1->len < p2->len) return -1; 
  if (p1->len > p2->len) return 1; 
  return 0; 
}

/*------------------------------------------------------------------*/
void AzStrPool::dump(const AzOut &out, const char *header) const {
  if (out.isNull()) return; 

  AzPrint o(out); 
  if (header == NULL) o.writeln("***  AzStrPool::dump  ***"); 
  else                o.writeln(header); 
  int num = size(); 
  for (int ix = 0; ix < num; ++ix) {
    o.printBegin("", ", "); 
    o.inBrackets(ix,5); 
    o.print(getCount(ix),4); 
    o.print(c_str(ix)); 
    o.printEnd(); 
  }
}

/*------------------------------------------------------------------*/
int AzStrPool::inc_ent() {
  if (cap > 0) return MIN(cap, MAX(init_ent_num, ent_num)); 
  else         return MAX(init_ent_num, ent_num); 
}

/*------------------------------------------------------------------*/
AZint8 AzStrPool::inc_data(AZint8 min_inc) {
  AZint8 inc; 
  if (cap > 0) inc = MIN(cap*10, MAX(init_data_len, data_len)); 
  else         inc = MAX(init_data_len, data_len); 
  inc = MAX(inc, min_inc); 
  return inc; 
}

/*------------------------------------------------------------------*/
void AzStrPool::reduce(int min_count) {
  const char *eyec = "AzStrPool::reduce"; 
  AzX::throw_if(!isCommitted, eyec, "sort first"); 

  AzByte *temp_data = NULL; 
  AzBaseArray<AzByte,AZint8> _a(data_len, &temp_data); 
  memcpy(temp_data, data, data_len); 

  AZint8 new_offs = 0; 
  int new_ex = 0; 
  for (int ex = 0; ex < ent_num; ++ex) {
    if (ent[ex].count >= min_count) {
      if (new_ex != ex) ent[new_ex] = ent[ex]; 
      ent[new_ex].offs = new_offs; 
      memcpy(data + new_offs, temp_data + ent[ex].offs, ent[ex].len);  
      new_offs += ent[ex].len; 
      *(data + new_offs) = '\0'; 
      ++new_offs; 

      ++new_ex; 
    }
  }

  a_ent.realloc(&ent, new_ex, eyec, "ent"); 
  ent_num = new_ex; 

  a_data.realloc(&data, new_offs, eyec, "data"); 
  data_len = new_offs; 

  for (int ex = 0; ex < ent_num; ++ex) ent[ex].bytes = data + ent[ex].offs; 

  /*-----  index it  -----*/
  build_index(); 
}

/*------------------------------------------------------------------*/
double AzStrPool::keep_topfreq(int keep_num, bool do_keep_ties, bool do_release_mem) {
  if (ent_num <= keep_num) return -1; 
  AzIFarr ifa_count; 
  for (int ex = 0; ex < ent_num; ++ex) {
    ifa_count.put(ex, (double)ent[ex].count); 
  }
  ifa_count.sort_Float(false); /* descending order */
  double th = ifa_count.get(keep_num-1); 
  if (do_keep_ties) {
    int ix; 
    for (ix = keep_num; ix < ifa_count.size(); ++ix) {
      double count = ifa_count.get(ix); 
      if (count < th) break; 
    }
    ifa_count.cut(ix); 
  }
  else {
    ifa_count.cut(keep_num); 
  }
  AzIntArr ia_ent; 
  ifa_count.int1(&ia_ent); 
  ia_ent.sort(true); 
  reduce(ia_ent.point(), ia_ent.size(), do_release_mem); 
  return th; 
}

/*------------------------------------------------------------------*/
/* reject if at least one of the words is stop-words */
void AzStrPool::_compose_ngram_stop_if_one(AzBytArr &s_ngram, /* output */
                            int wx, int nn, 
                            const AzStrPool *sp_stop, 
                            bool do_remove_number) const {
  s_ngram.reset(); 
  if (wx < 0 || wx+nn > size()) return; 
  for (int ix = wx; ix < wx+nn; ++ix) {
    if (ix > wx) s_ngram.c(' '); 
    if (sp_stop != NULL && sp_stop->find(c_str(ix)) >= 0 || 
        do_remove_number && strpbrk(c_str(ix), "0123456789") != NULL) {
      s_ngram.reset(); 
      return; 
    }
    s_ngram << c_str(ix); 
  }
}

/*------------------------------------------------------------------*/
/* reject if all of the words are stop-words */
void AzStrPool::_compose_ngram_stop_if_all(AzBytArr &s_ngram, /* output */
                            int wx, int nn, 
                            const AzStrPool *sp_stop, 
                            bool do_remove_number) const {
  s_ngram.reset(); 
  if (wx < 0 || wx+nn > size()) return; 
  int ix; 
  for (ix = wx; ix < wx+nn; ++ix) {
    AzBytArr s_word; get(ix, &s_word);     
    if (sp_stop != NULL && sp_stop->find(c_str(ix)) >= 0 || 
        do_remove_number && strpbrk(c_str(ix), "0123456789") != NULL) continue; 
    break;  /* not a stop-word */
  }
  if (ix >= wx+nn) return; 
  
  for (ix = wx; ix < wx+nn; ++ix) {
    if (ix > wx) s_ngram.c(' '); 
    s_ngram << c_str(ix); 
  }  
}

/*------------------------------------------------------------------*/
int AzStrPool::get_n(int ex) const {
  checkRange(ex, "AzStrPool::get_n");   
  int len; const AzByte *byt = point(ex, &len); 
  if (len <= 0) return 0; 
  int nn = 1; 
  for (int ix = 0; ix < len; ++ix) if (byt[ix] <= ' ') ++nn; 
  return nn; 
}
    
/*------------------------------------------------------------------*/
int AzStrPool::get_max_n() const {
  int max_n = 0; 
  for (int ex = 0; ex < ent_num; ++ex) {
    int len; 
    const AzByte *byt = point(ex, &len); 
    if (len <= 0) continue; 
    int nn = 1; 
    for (int ix = 0; ix < len; ++ix) if (byt[ix] <= ' ') ++nn; 
    max_n = MAX(max_n, nn); 
  }  
  return max_n; 
}

/*------------------------------------------------------------------*/
int AzStrPool::get_min_n() const {
  int min_n = -1;
  for (int ex = 0; ex < ent_num; ++ex) {
    int len; 
    const AzByte *byt = point(ex, &len); 
    if (len <= 0) continue; 
    int nn = 1; 
    for (int ix = 0; ix < len; ++ix) if (byt[ix] <= ' ') ++nn; 
    if (min_n < 0 || nn < min_n) min_n = nn; 
  }  
  return min_n; 
}

/*------------------------------------------------------------------*/
void AzStrPoolc::reset(const AzStrPool &sp, AzByte dlm) {
  const char *eyec = "AzStrPoolc::reset"; 
  num = sp.size(); 
  AZint8 len = 0; 
  for (int ix = 0; ix < sp.size(); ++ix) len += (sp.getLen(ix) + 1); 
  data.free_alloc(len, eyec, "data"); 
  AZint8 offs = 0; 
  AzByte *ptr = data.point_u(); 
  for (int ix = 0; ix < sp.size(); ++ix) {
    int wlen = sp.getLen(ix); 
    memcpy(ptr+offs, sp.point(ix), wlen); 
    offs += wlen; 
    *(ptr+offs) = dlm;  
    ++offs; 
  }
  AzX::throw_if((offs != len), eyec, "length mismatch"); 
}  

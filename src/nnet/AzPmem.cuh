/* * * * *
 *  AzPmem.cuh
 *  Copyright (C) 2015-2016 Rie Johnson
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

#ifndef _AZ_PMEM_CUH_
#define _AZ_PMEM_CUH_

#include "AzUtil.hpp"
#include "AzPrint.hpp"

extern bool __doDebug; 

#ifdef __AZ_GPU__ 
#include "AzCuda.cuh"
class _AzPmem {
public: 
  static void *_alloc(size_t sz, const char *str1="", const char *str2="", const char *str3="") {
    void *ptr = NULL; 
    cudaError_t ret = cudaMalloc(&ptr, sz); 
    AzCuda::throwIfError(ret, str1, str2, str3, sz);  
    return ptr; 
  }
  static void *_alloc_nothrow(size_t sz, AzBytArr &s) {
    void *ptr = NULL; 
    cudaError_t ret = cudaMalloc(&ptr, sz); 
    if (ret != cudaSuccess) {
      s << cudaGetErrorString(ret); 
      cudaGetLastError(); /* reset error */      
      ptr = NULL; 
    }
    return ptr; 
  }
  static void _free(void *ptr, const char *str1="", const char *str2="", const char *str3="") {
    if (ptr == NULL) return;   
    cudaError_t ret = cudaFree(ptr); 
    AzCuda::throwIfError(ret, str1, str2, str3);  
  }
};   
#else
class _AzPmem {
public:
  static void *_alloc(size_t sz, const char *str1="", const char *str2="", const char *str3="") {
    void *ptr = NULL; 
    char *myptr = NULL; 
    AzMemTools<size_t>::alloc(&myptr, sz, str1, str2); 
    ptr = (void *)myptr;  
    return ptr; 
  }
  static void *_alloc_nothrow(size_t sz) { _alloc(sz, "nothrow was requested but throwing ..."); }
  static void _free(void *ptr, const char *str1="", const char *str2="", const char *str3="") {
    if (ptr == NULL) return; 
    char *myptr = (char *)ptr; 
    AzMemTools<size_t>::free(&myptr);  
  }  
};   
#endif

class AzPmemEnt {
public:
  size_t sz; 
  char *ptr; 
  int next, prev; 
  bool is_used; 
  AzPmemEnt() : sz(0), ptr(NULL), next(-1), prev(-1), is_used(false) {}
  void set(void *_ptr, size_t _sz, bool _is_used) {
    ptr = (char *)_ptr; sz = _sz; is_used = _is_used; 
  }
  void init() {
    ptr = NULL; sz = 0; is_used = false; next = -1; prev = -1; 
  }
  void print(int no, AzBytArr &s) const {
    s << "[" << no << "] ptr="; 
    if (ptr == NULL) s << "NULL "; 
    else             s << "* ";
    s << "sz=" << (double)sz << " is_used=" << is_used << " next=" << next << " prev=" << prev; 
  } 
}; 

class AzPmem {
protected: 
  size_t memsz; 
  void *mem; 
/*  static const int azcpmem_max_ent = 1000; */
  static const int azcpmem_max_ent = 100000; 
  static const int azcpmem_bound = 8;
  AzDataArr<AzPmemEnt> ent; 

  double curr_extra, extra_count, curr_mem, max_mem; 
  
  double ent_overflow; 
  double entnum, entnum_max; 
  int max_no; 
  
public:   
  AzPmem() : mem(NULL), memsz(0), curr_extra(0), extra_count(0), curr_mem(0), max_mem(0), ent_overflow(0),
             entnum(0), entnum_max(0), max_no(-1) {}
  ~AzPmem() { term(); }

  void init(double gb) { /* giga byte */
    size_t _memsz = (size_t)(gb*(double)1024*(double)1024*(double)1024); 
    init(_memsz); 
  }
  void init(size_t _memsz) { 
    const char *eyec = "AzPmem::init"; 
    AzX::pthrow_if((mem != NULL), eyec, "mem must be NULL at this point"); 
    ent.reset(); 
    memsz = MAX(_memsz, 0); 
    if (memsz <= 0) return; 

    AzBytArr s("Allocating device memory: "); s << (double)memsz; 
    AzPrint::writeln(log_out, s.c_str()); 
#if 0     
    mem = _AzPmem::_alloc(memsz, "AzPmem::init", "mem"); 
#else
    AzBytArr s_err; 
    mem = _AzPmem::_alloc_nothrow(memsz, s_err); 
    if (mem == NULL) { /* if fails, disable the device memory handler and keep going */
      AzPrint::writeln(log_out, "Pre-allocation of device memory failed with \"", s_err.c_str(), 
                       "\".  Disabling device memory handler ... "); 
      memsz = 0; 
      return; 
    }
#endif 
    
    ent.reset(azcpmem_max_ent); 
    ent(0)->set(mem, memsz, false);  /* initially, we only have one big area */
    curr_extra = extra_count = 0; max_no = 0; 
    entnum = entnum_max = 1; 
    ent_overflow = 0; 
  }
  void term() {  
    _AzPmem::_free(mem, "AzPmem::term", "mem"); 
    mem = NULL; 
    memsz = 0; 
    ent.reset(); 
    if (entnum_max > 0) {
      AzBytArr s("AzPmem stat: #concurrent="); s << entnum_max << " max-entry#=" << max_no; 
      if (ent_overflow > 0) s << " table-expansion=" << ent_overflow;
      if (extra_count > 0)  s << " for-memory=" << extra_count-ent_overflow; 
      AzPrint::writeln(log_out, s); 
    }
    entnum = entnum_max = ent_overflow = extra_count = 0; 
  }

  /*--------------------------------------------------------------------------------*/
  void *alloc(int &no, size_t _sz, const char *str1="", const char *str2="") {
    const char *eyec = "AzPmem::alloc"; 
    size_t sz = align(_sz);  
    curr_mem += sz; max_mem = MAX(max_mem, curr_mem); 
    ++entnum; entnum_max = MAX(entnum, entnum_max); 
    
    no = first(); 
    for ( ; no >= 0; ) {
      const AzPmemEnt *ep = ent[no];     
      AzX::pthrow_if((ep->ptr == NULL), eyec, "null ptr?!");  
      if (!ep->is_used && ep->sz == sz) {
        ent(no)->is_used = true; 
        return ep->ptr; 
      }
      if (!ep->is_used && ep->sz > sz) {
        int new_no = insert_after(no); 
        if (new_no < 0) break; /* failed */
        ent(new_no)->set(ep->ptr+sz, ep->sz-sz, false); 
        ent(no)->set(ep->ptr, sz, true);     
        return ep->ptr; 
      }
      no = ep->next; 
    }
  
    /*---  if the pre-allocated memory cannot meet the requirement, allocate memory --- */
    no = -1; 
    void *ptr = _AzPmem::_alloc(_sz, "AzPmem::alloc extra", str1, str2);   
    ++curr_extra; ++extra_count; 
    return ptr; 
  }

  /*--------------------------------------------------------------------------------*/
  void free(int &no, void *ptr, size_t _sz, const char *str1="") {
    const char *eyec = "AzPmem::free"; 
    size_t sz = align(_sz); 
    curr_mem -= sz; 
    --entnum; 
    
    if (no >= 0) {
      const AzPmemEnt *ep = ent[no]; 
      AzX::pthrow_if((ep->ptr != ptr), eyec, "ptr mismatch.  something is wrong ... "); 
      AzX::pthrow_if((sz != ep->sz), eyec, "size mismatch.  something is wrong ... "); 
      ent(no)->is_used = false; 
      merge_next_if_possible(no); 
      merge_next_if_possible(ep->prev);      
      no = -1; 
    }
    else {    
      /*---  this ptr isn't pointing to the pre-allocated memory ... ---*/
      AzX::pthrow_if((curr_extra <= 0), eyec, "extra: # of free requests is greater than # of alloc requests ... "); 
      --curr_extra; 
      _AzPmem::_free(ptr, eyec, str1); 
    }      
    if (__doDebug) check_consistency(); 
  }
  
  void print(AzBytArr &s) const {
    s<<"curr_mem="<<curr_mem<<" max_mem="<< max_mem<<" curr_extra="<<curr_extra<<" extra_count="<<extra_count;s.nl(); 
    int no = first(); 
    for ( ; no >= 0; ) {
      ent[no]->print(no, s); 
      no = ent[no]->next; 
    }
  }

protected:   
  int first() const { return (ent.size() > 0) ? 0 : -1; }
  size_t align(size_t _sz) { int b = azcpmem_bound; return (_sz+b-1)/b*b; }
  
  /*--------------------------------------------------------------------------------*/
  int insert_after(int no) {
    int ex; 
    for (ex = 0; ex < ent.size(); ++ex) if (ent[ex]->ptr == NULL) break; 
    if (ex >= ent.size()) return -1; 

    int new_no = ex; 
    ent(new_no)->init(); 

    ent(new_no)->prev = no;   
    ent(new_no)->next = ent[no]->next; 
    ent(no)->next = new_no; 
    if (ent[new_no]->next >= 0) {
      ent(ent[new_no]->next)->prev = new_no;    
    }
    max_no = MAX(max_no, new_no); 
    return new_no; 
  }

  /*--------------------------------------------------------------------------------*/
  void merge_next_if_possible(int no) {
    const char *eyec = "AzPmem::merge_next_if_possible"; 
    if (no < 0 || ent[no]->is_used) return; 
    int next = ent[no]->next; 
    if (next < 0 || ent[next]->is_used) return; 
  
    AzX::pthrow_if((ent[no]->ptr + ent[no]->sz != ent[next]->ptr), eyec, "The two areas should be adjacent ... ?!"); 
  
    ent(no)->set(ent[no]->ptr, ent[no]->sz + ent[next]->sz, false); 
    ent(no)->next = ent[next]->next; 
    if (ent[no]->next >= 0) ent(ent[no]->next)->prev = no; 
    ent(next)->init(); 
  }
 
  /*--------------------------------------------------------------------------------*/  
  void check_consistency() const {
    const char *eyec = "AzPmem::check_consistency"; 
    int no = first(); 
    int prev = -1; 
    size_t sz = 0; 
    for ( ; no >= 0; ) {
      AzX::pthrow_if((ent[no]->prev != prev), eyec, "wrong prev"); 
      AzX::pthrow_if((ent[no]->ptr == NULL), eyec, "null ptr"); 
      AzX::pthrow_if((prev >= 0 && ent[prev]->ptr+ent[prev]->sz != ent[no]->ptr), eyec, "areas are not adjacnet"); 
      sz += ent[no]->sz; 
      prev = no; no = ent[no]->next; 
    }
    AzX::pthrow_if((sz != memsz), eyec, "total memory size doesn't match"); 
  }
}; 
#endif 
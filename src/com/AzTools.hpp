/* * * * *
 *  AzTools.hpp 
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

#ifndef _AZ_TOOLS_HPP_
#define _AZ_TOOLS_HPP_

#include "AzUtil.hpp"
#include "AzSmat.hpp"
#include "AzDmat.hpp"
#include "AzStrPool.hpp"
#include "AzPrint.hpp"

class AzTools {
public: 
  static int read_header(AzFile *file, int reserved_len) {
    int version = file->readInt(); 
    AzBytArr s; s.reset(reserved_len, 0); 
    file->readBytes(s.point_u(), s.length());   
    return version; 
  }
  static void write_header(AzFile *file, int version, int reserved_len) {
    file->writeInt(version); 
    AzBytArr s; s.reset(reserved_len, 0);     
    file->writeBytes(s.point(), s.length());  
  }
  
  static void writeMatrix(const AzDmat &m, const char *fn, const AzOut &out, int digits=7); 
  static int writeList(const char *fn, const AzStrPool *sp_list); 
  static void readList(const char *fn, AzStrPool *sp_list, 
                       const AzByte *dlm=NULL,  /* if specified, read the first token (delimited by dlm) only */
                       bool do_count=false, /* assume the count follows the delimiter */
                       bool do_allow_blank=false); 

  /*--- (NOTE) This skips preceding white characters.  ---*/
  static const AzByte *getString(const AzByte **wpp, const AzByte *data_end, int *byte_len); 
  static void getString(const AzByte **wpp, const AzByte *data_end, AzBytArr &string) {
    int str_len; const AzByte *str = getString(wpp, data_end, &str_len); 
    string.concat(str, str_len); 
  }
  
  /*-----  -----*/ 
  static int chomp(const AzByte *inp, int inp_len); 
  
  /*---  (NOTE) This doesn't skip preceding white characters.  ---*/
protected:   
  static const AzByte *_getString(const AzByte **wpp, const AzByte *data_end, 
                                  AzByte wdlm, int *byte_len);                                   
public:                                   
  static void getString(const AzByte **wpp, const AzByte *data_end, 
                        AzByte wdlm, AzBytArr &string) {
    int str_len; const AzByte *str = _getString(wpp, data_end, wdlm, &str_len); 
    string.concat(str, str_len); 
  }

  static void getStrings(const AzByte *data, int data_len, AzStrPool *sp_tokens); 
  static void getStrings(const AzByte *data, int len, AzByte dlm, AzStrPool *sp_out);
  inline static void getStrings(const char *str, AzByte dlm, AzStrPool *sp_out) {
    if (str == NULL) return; 
    getStrings((AzByte *)str, Az64::cstrlen(str), dlm, sp_out); 
  }

  static void getInts(const char *str, AzByte dlm, AzIntArr *ia) {
    AzX::throw_if_null(str, "AzTools::getInts"); 
    AzX::throw_if_null(ia, "AzTools::getInts", "ia"); 
    ia->reset(); 
    AzStrPool sp; 
    getStrings(str, dlm, &sp); 
    for (int ix = 0; ix < sp.size(); ++ix) ia->put(atol(sp.c_str(ix))); 
  }
  static void getFloats(const char *str, AzByte dlm, AzDvect &v) {
    AzX::throw_if_null(str, "AzTools::getFloats"); 
    AzIFarr ifa; 
    AzStrPool sp; 
    getStrings(str, dlm, &sp); 
    for (int ix = 0; ix < sp.size(); ++ix) ifa.put(-1, atof(sp.c_str(ix))); 
    v.reform(ifa.size()); 
    for (int ix = 0; ix < ifa.size(); ++ix) v.set(ix, ifa.get(ix)); 
  }

  static void shuffle(int rand_seed, AzIntArr *iq, bool withReplacement = false); 
  static void shuffle2(AzIntArr &ia); 
  static inline int big_rand() {
    return (rand() % 32768) * 32768 + (rand() % 32768); 
  }
  inline static int rand_large() {
    int rand_max = RAND_MAX; 
    if (rand_max <= 0x7fff) return rand() * (rand_max + 1) + rand(); 
    return rand(); 
  }
  static void sample(int nn, int kk, AzIntArr *ia); /* without replacement */  
  static inline double rand01() {
    int denomi = 32760; 
    return ( (rand() % denomi)/(double)denomi ); 
  }
  static bool isSpecified(const AzBytArr *s) {
    if (s == NULL || s->length() <= 0) return false; 
    return isSpecified(s->c_str()); 
  }
  static bool isSpecified(const char *input) {
    if (input == NULL || strlen(input) == 0) return false; 
    if (*input == '_' && strlen(input) == 1) return false; 
    return true;   
  }

  static void unstrip(AzBytArr &s, AzByte ch=' ') {
    AzBytArr s_tmp(&s); s.reset(); 
    s << ch << s_tmp << ch; 
  }
  static void strip(AzBytArr &s) {
    AzBytArr s_tmp(&s); s.reset(); 
    strip(s_tmp.point(), s_tmp.length(), s); 
  }
  static void strip(const AzByte *inp_str, int inp_len, 
                    AzBytArr &str_out) { /* output */
    int out_len; 
    const AzByte *out_str = strip(inp_str, inp_str + inp_len, &out_len);
    str_out.concat(out_str, out_len); 
  }
  static const AzByte *strip(const AzByte *data, const AzByte *data_end, int *byte_len); 

  /*---  milestone ... ---*/
  inline static void check_milestone(int &milestone, int index, int inc) {
    if (milestone > 0 && index >= milestone) {
      fprintf(stdout, "."); fflush(stdout); 
      milestone += inc; 
    }
  }
  inline static void finish_milestone(int &milestone) {
    if (milestone > 0) fprintf(stdout, "\n"); 
    milestone = -1; 
  }   
  template <class M> static void show_smat_stat(const M &m, AzBytArr &s) {
    s << m.rowNum() << " x " << m.colNum() << " (" << (double)m.elmNum()/(double)m.colNum() << ") "; 
  }
  template <class M> static void show_dmat_stat(const M &m, AzBytArr &s) {
    s << m.rowNum() << " x " << m.colNum(); 
  }
  
  /*------------------------------------------------*/
  static AzOut *reset_out(const char *fn, AzOfs &ofs, AzOut &out) {
    AzOut *out_ptr = NULL;  
    if (isSpecified(fn)) {
      ofs.open(fn, ios_base::out); 
      ofs.set_to(out); 
      out_ptr = &out; 
    }
    return out_ptr; 
  }    
}; 

/*--------------------------------------------------------------------*/
template <class M> 
class AzMats_file {
protected: 
  int num; 
  int cur; 
  char mode; 
  AzFile file; 
public: 
  #define _AzMats_file_init_ num(0), cur(0), mode(0)
  AzMats_file() : _AzMats_file_init_ {}
  AzMats_file(const char *fn) : _AzMats_file_init_ { reset_for_read(fn); }
  AzMats_file(const char *fn, int num) : _AzMats_file_init_ { reset_for_write(fn, num); }
  int matNum() const { return num; }
  int reset_for_read(const char *fn) {
    cur = 0; 
    file.reset(fn); 
    file.open("rb"); 
    mode = 'r'; 
    num = file.readInt(); 
    return num; 
  }
  void reset_for_write(const char *fn, int _num) {
    cur = 0; 
    file.reset(fn); 
    file.open("wb"); 
    mode = 'w'; 
    num = _num; 
    file.writeInt(num); 
  }  
  void read(M *mat) {
    const char *eyec = "AzMat_file::read"; 
    if (mode != 'r') throw new AzException(eyec, "call reset_for_read first"); 
    if (mat == NULL) throw new AzException(eyec, "null input"); 
    if (cur >= num) throw new AzException(eyec, "index is out of range"); 
    mat->read(&file); 
    ++cur; 
  }
  void write(const M *mat) {
    const char *eyec = "AzMat_file::write"; 
    AzX::throw_if(mode != 'w', eyec, "call reset_for_write first"); 
    AzX::throw_if_null(mat, eyec); 
    AzX::throw_if(cur >= num, eyec, "index is out of range"); 
    mat->write(&file); 
    ++cur; 
  }
  void done() {
    if      (mode == 'r') file.close(); 
    else if (mode == 'w') file.close(true); 
    mode = 0; 
  }
  static void merge(const AzOut &out, const char *out_fn, int fn_num, const char *fn[]) {
    int num = 0; 
    for (int fx = 0; fx < fn_num; ++fx) {
      AzMats_file mfile; 
      num += mfile.reset_for_read(fn[fx]); 
      mfile.done(); 
    }
    AzMats_file ofile(out_fn, num); 
    for (int fx = 0; fx < fn_num; ++fx) {
      AzTimeLog::print(fn[fx], " ... ", out); 
      AzMats_file ifile; 
      int mat_num = ifile.reset_for_read(fn[fx]); 
      for (int mx = 0; mx < mat_num; ++mx) {
        M mat; 
        ifile.read(&mat); 
        ofile.write(&mat); 
      }
      ifile.done(); 
    }
    ofile.done();     
    AzTimeLog::print("Done ... ", out); 
  }
}; 
#endif 

/* * * * *
 *  AzPrint.hpp 
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

#ifndef _AZ_PRINT_HPP
#define _AZ_PRINT_HPP

#include "AzUtil.hpp"

//! Printing tools
class AzPrint {
protected:
  ostream *o; /* may be NULL */
  const char *dlm, *name_dlm; 
  int count; 
  bool useDlm, did_write; 
  int level; 
  
  const char *pfx; 
  AzBytArr s; /* work */
public:
  AzPrint(const AzOut &out, const char *_pfx="") : o(NULL), dlm(NULL), name_dlm(NULL), did_write(false), 
                                                       count(0), useDlm(true), level(0), pfx("") {
    if (!out.isNull()) o = out.o; 
    level = out.getLevel(); 
    pfx = _pfx; 
  }
  ~AzPrint() { /* 6/5/2017 */
    printEnd(); 
    flush(); 
  }
  void reset(const AzOut &out, const char *_pfx="") {
    o = NULL; 
    if (!out.isNull()) o = out.o; 
    level = out.getLevel(); 
    dlm = NULL; 
    count = 0; 
    did_write = false; 
    pfx = _pfx; 
  }
  void reset_prefix(const char *_pfx="") { pfx = _pfx; }

  void resetDlm(const char *inp_dlm) {
    dlm = inp_dlm; 
  }

  /*--------*/
  void ppBegin(const char *caller, 
                      const char *desc=NULL, 
                      const char *inp_dlm=NULL) {
    AzBytArr s; 
    if (level > 0) s.concat(caller); 
    else           s.concat(desc); 
    printBegin(s.c_str(), inp_dlm); 
  }
  void ppEnd() {
    printEnd(); 
  }
  /*--------*/

  void printBegin(const char *kw, 
                         const char *inp_dlm=NULL, 
                         const char *inp_name_dlm=NULL, 
                         int indent=0) {
    if (o == NULL) return; 
    did_write = false; 
    dlm = inp_dlm; 
    name_dlm = inp_name_dlm; 
    if (name_dlm == NULL) name_dlm = dlm; 
    if (indent > 0) { /* indentation */
      AzBytArr s; s.fill(' ', indent);  
      *o<<s.c_str(); 
      did_write = true; 
    }
    if (kw != NULL && strlen(kw) > 0) {   
      *o<<kw<<": "; 
      did_write = true; 
    }
    count = 0; 
  }
  void printSw(const char *_kw, bool val) {
    if (o == NULL) return; 
    const char *kw = cpfx(_kw); 
    if (val) {
      /*---  print only when it's on  ---*/
      itemBegin(); 
      *o<<kw<<":"; 
      if (val) *o<<"ON"; 
      else     *o<<"OFF"; 
    }
  }
  template<class T>
  void printV(const char *_kw, T val, bool do_force=false) {
    if (o == NULL) return; 
    const char *kw = cpfx(_kw);     
    if (val != -1 || do_force) {
      itemBegin();  
      *o<<kw<<val; 
    }
  }  
  template<class T>
  void printV_posiOnly(const char *_kw, T val) {
    if (o == NULL) return; 
    const char *kw = cpfx(_kw);     
    if (val > 0) {
      itemBegin();  
      *o<<kw<<val; 
    }
  }
  void printV(const char *_kw, const char *val) {
    if (o == NULL) return; 
    const char *kw = cpfx(_kw);     
    itemBegin(); 
    *o<<kw<<val; 
  }
  void printV(const char *_kw, const AzBytArr &s) {
    if (o == NULL) return; 
    const char *kw = cpfx(_kw);     
    itemBegin(); 
    *o<<kw<<s.c_str(); 
  }
  void printV_if_not_empty(const char *_kw, const AzBytArr &s) {
    if (o == NULL) return; 
    if (s.length() <= 0) return; 
    const char *kw = cpfx(_kw); 
    itemBegin(); 
    *o<<kw<<s.c_str(); 
  }  
  void printEnd(const char *str=NULL) {
    if (o == NULL) return; 
    if (str != NULL) *o<<str; 
    if (str != NULL || count > 0 || did_write) *o<<endl; /* if condition added on 1/1/2015 */
/*    if (dlm == NULL) o->flush();  6/5/2017 */
    count = 0; did_write = false; 
  }
  void flush() {
    if (o == NULL) return; 
    o->flush(); 
  }

  /*-------------------*/
  void reset_options() {
    if (o == NULL) return; 
    o->unsetf(ios_base::scientific); 
    o->unsetf(ios_base::fixed); 
    o->precision(3); 
  }
  void set_precision(int p) {
    if (o == NULL) return; 
    o->precision(p); 
  }
  void set_scientific() {
    if (o == NULL) return; 
    *o<<scientific; 
  }

  /*-------------------*/
  void print(const AzBytArr &s) {
    if (o == NULL) return; 
    print(s.c_str()); 
  }
  void print(const char *str) {
    if (o == NULL) return; 
    itemBegin(); 
    *o<<str; 
  }

  /*---  print continuously without delimiter  ---*/
  void print_cont(const char *s1, const char *s2, const char *s3=NULL) {
    if (o == NULL) return; 
    itemBegin(); 
    if (s1 != NULL) *o<<s1; 
    if (s2 != NULL) *o<<s2; 
    if (s3 != NULL) *o<<s3; 
  }

  /*-----*/
  void print(int val, int width=-1, bool doFillZero=false) {
    print(NULL, val, width, doFillZero); 
  }
  void print(AZint8 val, int width=-1, bool doFillZero=false) {
    print(NULL, val, width, doFillZero); 
  }
  void print(double val, int prec=-1, bool doSci=false) {
    print(NULL, val, prec, doSci); 
  }
  template <class T>
  void print(const char *name, T val, int width_prec=-1, 
                    bool doZero_doSci=false) {
    if (o == NULL) return; 
    itemBegin(); 
    if (name != NULL) {
      *o<<name; 
      if (name_dlm != NULL) *o<<name_dlm; 
    }
    AzBytArr s; s.cn(val, width_prec, doZero_doSci);
    *o<<s.c_str(); 
  }

  /*-----*/
  void inParen(int val, int width=-1, bool doFillZero=false) {
    if (o == NULL) return; 
    itemBegin(); 
    AzBytArr s; inParen(s, val, width, doFillZero); 
    *o<<s.c_str(); 
  }
  void inBrackets(int val, int width=-1, bool doFillZero=false) {
    if (o == NULL) return; 
    itemBegin(); 
    AzBytArr s; inBrackets(s, val, width, doFillZero); 
    *o<<s.c_str(); 
  }
  void inParen(double val, int prec=-1, bool doSci=false) {
    if (o == NULL) return; 
    itemBegin(); 
    AzBytArr s; inParen(s, val, prec, doSci); 
    *o<<s.c_str(); 
  }
  void inBrackets(double val, int prec=-1, bool doSci=false) {
    if (o == NULL) return; 
    itemBegin(); 
    AzBytArr s; inBrackets(s, val, prec, doSci); 
    *o<<s.c_str(); 
  }
  void inDoubleQuotes(const char *str) {
    if (o == NULL) return; 
    itemBegin(); 
    *o<<"\""<<str<<"\""; 
  }
  void inParen(const char *str) {
    if (o == NULL) return; 
    itemBegin(); 
    *o<<"("<<str<<")"; 
  }
  void inBrackets(const char *str) {
    if (o == NULL) return; 
    itemBegin(); 
    *o<<"["<<str<<"]"; 
  }
  void inParen(const AzBytArr &s) {
    if (o == NULL) return; 
    inParen(s.c_str()); 
  }
  void inBrackets(const AzBytArr &s) {
    if (o == NULL) return; 
    inBrackets(s.c_str()); 
  }

  /*---  ---*/
  template <class T, class U>
  void pair_inParen(T val1, U val2, const char *pair_dlm=NULL) { 
    if (o == NULL) return; 
    itemBegin(); 
    _print_pair("(", ")", val1, val2, pair_dlm); 
  }

  template <class T, class U>
  void pair_inBrackets(T val1, U val2, const char *pair_dlm=NULL) {
    if (o == NULL) return; 
    itemBegin(); 
    _print_pair("[", "]", val1, val2, pair_dlm);  
  }

  /*-------------------*/
  void newLine() {
    if (o == NULL) return; 
    *o<<endl; 
  }
  template <class T>
  void writeln(const T inp) {
    if (o == NULL) return; 
    *o<<inp<<endl; 
  }
  template <class T>
  void write(const T inp) {
    if (o == NULL) return; 
    *o<<inp; 
  }
  void writeln(const AzBytArr &s) {
    if (o == NULL) return; 
    *o<<s.c_str()<<endl; 
  }
  void write(const AzBytArr &s) {
    if (o == NULL) return; 
    *o<<s.c_str(); 
  }

  void disableDlm() {
    useDlm = false; 
  }
  void enableDlm() {
    useDlm = true; 
  }

  template <class T>
  static void write(const AzOut &out, const T inp) {
    if (out.isNull()) return; 
    *out.o<<inp; 
  }
  template <class T, class U>
  static void write(const AzOut &out, const T inp0, const U inp1) {
    write(out, inp0); write(out, inp1); 
  }
  template <class T, class U, class V>
  static void write(const AzOut &out, const T inp0, const U inp1, const V inp2) {
    write(out, inp0, inp1); write(out, inp2); 
  }
  template <class T, class U, class V, class W>
  static void write(const AzOut &out, const T inp0, const U inp1, const V inp2, const W inp3) {
    write(out, inp0, inp1, inp2); write(out, inp3); 
  }  
  template <class T>
  static void writeln(const AzOut &out, const T inp) {
    if (out.isNull()) return; 
    *out.o<<inp<<endl; 
  }
  template <class T, class U>
  static void writeln(const AzOut &out, const T inp0, const U inp1) {
    write(out, inp0); 
    writeln(out, inp1); 
  }
  template <class T, class U, class V>
  static void writeln(const AzOut &out, const T inp0, const U inp1, const V inp2) {
    write(out, inp0, inp1); writeln(out, inp2); 
  }
  template <class T, class U, class V, class W>
  static void writeln(const AzOut &out, const T inp0, const U inp1, const V inp2, const W inp3) {
    write(out, inp0, inp1, inp2); writeln(out, inp3); 
  }
  
  static void write(const AzOut &out, const AzBytArr &s) {
    write(out, s.c_str()); 
  }
  static void writeln(const AzOut &out, const AzBytArr &s) {
    writeln(out, s.c_str()); 
  }
  template <class T>
  static void force_writeln(const T inp) {
    cout<<inp<<endl; 
  }
  static void force_writeln(const AzBytArr &s) {
    force_writeln(s.c_str()); 
  }

  void print_space(int len) {
    AzBytArr s; s.fill(' ', len); 
    print(s); 
  }

protected:
  void itemBegin() {
    if (o == NULL) return; 
    if (useDlm) {
      if (dlm == NULL)    *o<<endl<<"   ";  
      else if (count > 0) *o<<dlm; 
    }
    ++count; 
  }

  /*-------------------*/
  template <class T, class U>
  void _print_pair(const char *left, const char *right, 
                          T val1, U val2, const char *pair_dlm) {
    if (o == NULL) return; 
    *o<<left<<val1; 
    if (pair_dlm != NULL) *o<<pair_dlm; 
    else if (dlm != NULL) *o<<dlm; 
    else                  *o<<" "; 
    *o<<val2<<right; 
  }

  /*-------------------*/
  template <class T>
  static void inParen(AzBytArr &s, T val, int width_prec=-1, 
                             bool doZero_doSci=false) {
    s << "("; s.cn(val, width_prec, doZero_doSci); s << ")"; 
  }
  template <class T>
  static void inBrackets(AzBytArr &s, T val, int width_prec=-1, 
                                bool doZero_doSci=false) {
    s << "["; s.cn(val, width_prec, doZero_doSci); s << "]"; 
  }
  
  const char *cpfx(const char *kw) {
    s.reset(pfx); 
    s.concat(kw); 
    return s.c_str(); 
  }   
}; 
#endif 

/* * * * *
 *  AzTools.cpp 
 *  Copyright (C) 2011-2015,2017 Rie Johnson
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

#include "AzTools.hpp"
#include "AzPrint.hpp"
/*--------------------------------------------------------*/
int AzTools::writeList(const char *fn, const AzStrPool *sp_list) {
  int max_len = 0; 
  AzFile file(fn); 
  file.open("wb"); 
  int num = sp_list->size(); 
  for (int ix = 0; ix < num; ++ix) {
    AzBytArr s; 
    sp_list->get(ix, &s); 
    s.concat("\n");
    max_len = MAX(max_len, s.getLen());  
    s.writeText(&file);  
  }
  file.close(true); 
  return max_len; 
}

/*--------------------------------------------------------*/
void AzTools::readList(const char *fn, 
                       AzStrPool *sp_list, /* output */
                       const AzByte *dlm, 
                       bool do_count, 
                       bool do_allow_blank) {
  AzX::throw_if_null(sp_list, "AzTools::readList", "sp_list");                          
  AzFile file(fn); 
  file.open("rb"); 
  int sz = file.size_under2G("AzTools::readList, list file"); 

  AzByte *buff = NULL; 
  int buff_len = sz+1; 
  AzBaseArray<AzByte> _a(buff_len, &buff);

  for ( ; ; ) {
    int len = file.gets(buff, buff_len); 
    if (len <= 0) break; 
    AZint8 count = 1; 
    int str_len; 
    const AzByte *str = buff; 
    if (do_allow_blank) str_len = chomp(buff, len);  
    else                str = strip(buff, buff+len, &str_len); 
    if (dlm != NULL) {
      int ix; 
      for (ix = 0; ix < str_len; ++ix) if (*(str+ix) == *dlm) break; 
      if (do_count && ix < str_len) {
        count = (AZint8)(atof((char *)(str+ix+1))); /* +1 for dlm */        
      }        
      str_len = ix;   
    } 
    sp_list->put(str, str_len, count); 
  }
}

/*--------------------------------------------------------*/
int AzTools::chomp(const AzByte *inp, int inp_len) {
  if (inp_len <= 0) return 0; 
  AzX::throw_if_null(inp, "AzTools::chomp"); 
  const AzByte *wp2 = inp+inp_len-1; 
  for ( ; wp2>=inp; --wp2) if (*wp2 != 0x0a && *wp2 != 0x0d) break; 
  return Az64::ptr_diff(wp2+1-inp); 
}

/*--------------------------------------------------------*/
/* (NOTE) This doesn't skip preceding white characters.   */
/*--------------------------------------------------------*/
const AzByte *AzTools::_getString(const AzByte **wpp, const AzByte *data_end, 
                                 AzByte dlm, int *token_len) { 
  const AzByte *token = *wpp; 
  const AzByte *wp = *wpp; 
  for ( ; wp < data_end; ++wp) if (*wp == dlm) break;
  *token_len = Az64::ptr_diff(wp-token, "AzTools::getString"); 

  /*-----  point the next to the delimiter  -----*/
  if (wp < data_end) ++wp;  
  *wpp = wp; 
  return token; 
}

/*--------------------------------------------------------*/
void AzTools::getStrings(const AzByte *data, int data_len, 
                         AzByte dlm, 
                         AzStrPool *sp_out) { /* output */
  const char *eyec = "AzTools::getStrings";                          
  AzX::throw_if_null(data, eyec); AzX::throw_if_null(sp_out, eyec); 
  const AzByte *data_end = data + data_len; 
  const AzByte *wp = data; 
  for ( ; wp < data_end; ) {
    int len; 
    const AzByte *token = _getString(&wp, data_end, dlm, &len); 
    sp_out->put(token, len); 
  }
}

/*--------------------------------------------------------*/
/* (NOTE) This skips preceding white characters.          */
/*--------------------------------------------------------*/
const AzByte *AzTools::getString(const AzByte **wpp, const AzByte *data_end, int *byte_len) {
  const char *eyec = "AzTools::getString"; 
  AzX::throw_if_null(wpp, eyec); AzX::throw_if_null(*wpp, eyec); 
  const AzByte *wp = *wpp; 
  for ( ; wp < data_end; ++wp) if (*wp > ' ') break;
  const AzByte *token = wp; 

  for ( ; wp < data_end; ++wp) if (*wp <= ' ') break;
  const AzByte *token_end = wp; 

  *byte_len = Az64::ptr_diff(token_end-token, "AzTools::getString2"); 

  *wpp = token_end; 
  return token; 
}

/*--------------------------------------------------------*/
void AzTools::getStrings(const AzByte *data, int data_len, 
                         AzStrPool *sp_tokens) {                           
  if (data_len <= 0) return; 
  const char *eyec = "AzTools::getStrings"; 
  AzX::throw_if_null(data, eyec); AzX::throw_if_null(sp_tokens, eyec); 
  const AzByte *wp = data, *data_end = data + data_len; 
  for ( ; ; ) {
    int len; 
    const AzByte *str = getString(&wp, data_end, &len); 
    if (len <= 0) break; 
    sp_tokens->put(str, len); 
  }
}

/*--------------------------------------------------------*/
const AzByte *AzTools::strip(const AzByte *data, const AzByte *data_end,  int *byte_len) {
  const char *eyec = "AzTools::strip"; 
  AzX::throw_if_null(data, eyec); AzX::throw_if_null(byte_len, eyec, "byte_len"); 
  const AzByte *bp; 
  for (bp = data; bp < data_end; ++bp) if (*bp > ' ') break; 
  const AzByte *ep; 
  for (ep = data_end - 1; ep >= data; --ep) if (*ep > ' ') break; 
  ++ep; 

  *byte_len = Az64::ptr_diff(ep-bp, "AzTools::strip"); 

  if (*byte_len < 0) { /* blank line */
    bp = data; 
    *byte_len = 0; 
  }
  return bp; 
}

/*--------------------------------------------------------*/
void AzTools::shuffle(int rand_seed, AzIntArr *iq, bool withReplacement) {
  if (rand_seed > 0) srand(rand_seed); 

  int num = iq->size(); 
  AzIntArr ia_temp; 
  int unit = 0; 
  ia_temp.reset(num, AzNone); 
  for (int ix = 0; ix < num; ++ix) {
    for ( ; ; ) {
      int rand_no = rand_large(); 
      rand_no = rand_no % num; 
      if (withReplacement) {
        ia_temp.update(ix, iq->get(rand_no)); 
        break; 
      }
      if (ia_temp.get(rand_no) == AzNone) {
        ia_temp.update(rand_no, iq->get(ix)); 
        break; 
      }
    }
    if (unit > 0 && !log_out.isNull()) {
      if ((ix+1) % unit == 0) {
        AzPrint::write(log_out, "."); 
        log_out.flush(); 
      }
    }
  }
  if (unit > 0 && !log_out.isNull()) {
    AzPrint::writeln(log_out, ""); 
    log_out.flush(); 
  }
  iq->reset(); 
  iq->concat(&ia_temp); 
}

/*--------------------------------------------------------*/
/* without replacement */
void AzTools::shuffle2(AzIntArr &ia) {
  int num = ia.size(); 
  AzIntArr ia_idx; ia_idx.range(0, num); 
  for (int taken = 0; taken < num; ++taken) {
    int rand_no = rand_large() % (num-taken) + taken;    
    int temp = ia_idx[rand_no]; 
    ia_idx(rand_no, ia_idx[taken]); 
    ia_idx(taken, temp); 
  } 
  AzIntArr ia_temp(&ia); 
  for (int ix = 0; ix < num; ++ix) ia(ix, ia_temp[ia_idx[ix]]); 
}

/*------------------------------------------------*/
void AzTools::sample(int nn, int kk, AzIntArr *ia) { 
  AzX::throw_if_null(ia, "AzTools::sample"); 
  if (kk >= nn) {
    ia->range(0, nn); 
    return; 
  }
  ia->reset(); ia->prepare(kk); 
  AzIntArr ia_is_used; ia_is_used.reset(nn, false); 
  int *is_used = ia_is_used.point_u(); 
  for ( ; ; ) {
    int val = rand_large() % nn; 
    if (!is_used[val]) {
      ia->put(val); 
      is_used[val] = true;  
      if (ia->size() >= kk) break; 
    }
  }
}

/*------------------------------------------------*/
void AzTools::writeMatrix(const AzDmat &m, const char *fn, const AzOut &out, int digits) {
  AzBytArr s("Writing ", fn, ": "); 
  if (AzBytArr::endsWith(fn, "dmat")) {  
    show_dmat_stat(m, s); s << " (dmat) "; 
    AzTimeLog::print(s, out); 
    m.write(fn); 
  }
  else if (AzBytArr::endsWith(fn, "smat")) {
    AzSmat ms; m.convert(&ms); 
    show_smat_stat(ms, s); s << " (smat) "; 
    AzTimeLog::print(s, out); 
    ms.write(fn);
  }
  else {
    show_dmat_stat(m, s); s << " (text) "; 
    AzTimeLog::print(s, out); 
    m.writeText(fn, digits); 
  }
  AzTimeLog::print("Done ... ", out); 
}

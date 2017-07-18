/* * * * *
 *  AzTextMat.cpp 
 *  Copyright (C) 2015-2017 Rie Johnson
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

#include "AzTextMat.hpp"
#include "AzTools.hpp"

/*------------------------------------------------------------------*/
void AzTextMat::readVector(const char *y_fn, AzDvect *v_y, int max_data_num) {
  AzSmat m_y; 
  readData_Large(y_fn, 1, &m_y, max_data_num); 
  int y_data_num = m_y.colNum(); 
  v_y->reform(y_data_num); 
  for (int dx = 0; dx < y_data_num; ++dx) {
    v_y->set(dx, m_y.get(0, dx)); 
  }
}                       

/*------------------------------------------------------------------*/
void AzTextMat::_parseDataLine(const AzByte *inp, 
                              int inp_len, 
                              int f_num, 
                              const char *data_fn, 
                              int line_no, 
                              /*---  output  ---*/
                              AzIFarr &ifa_ex_val) {
  const char *eyec = "AzTextMat::_parseDataLine"; 
  ifa_ex_val.prepare(f_num); 
  const AzByte *wp = inp, *line_end = inp + inp_len; 
  int ex = 0; 
  for ( ; wp < line_end; ) {
    int str_len; 
    const AzByte *str = AzTools::getString(&wp, line_end, &str_len); 
    if (str_len > 0) {
      if (ex >= f_num) {
        AzBytArr s("Error in "); s << data_fn << ": Line#=" << line_no; 
        AzPrint::writeln(log_out, s); 
        s.reset("Too many values per line: expected "); s << f_num << " values.";  
        AzX::throw_if(true, AzInputNotValid, eyec, s.c_str()); 
      }
      
      double val = my_atof((char *)str, eyec, line_no);    
      if (val != 0) ifa_ex_val.put(ex, val); 
      ++ex;
    }    
  }
  if (ex < f_num) {
    AzTimeLog::print("Error in Line#=", line_no, log_out); 
    AzX::throw_if(true, AzInputNotValid, eyec, "Too few values"); 
  }
}

/*------------------------------------------------------------------*/
int AzTextMat::countFeatures(const AzByte *line, const AzByte *line_end) {
  const AzByte *wp = line; 
  int count = 0; 
  for ( ; wp < line_end; ) {
    AzBytArr s; 
    AzTools::getString(&wp, line_end, s); 
    if (s.length() > 0) ++count; 
  }
  return count;
}

/*------------------------------------------------------------------*/
template <class Mat> /* Mat: AzSmat | AzDmat */
void AzTextMat::readData_Large(const char *data_fn, 
                         int expected_f_num, 
                         /*---  output  ---*/
                         Mat *m_feat,
                         int max_data_num) {
  const char *eyec = "AzTextMat::readData_Large"; 

  /*---  find the number of lines and the maximum line length  ---*/
  AzIntArr ia_line_len; 
  bool do_throw_if_null = true; 
  AzFile::scan(data_fn, 1024*1024, &ia_line_len, do_throw_if_null, max_data_num+1);  /* +1 for sparse indicator */
  
  int data_num = ia_line_len.size(); 
  int max_line_len = ia_line_len.max(); 
  AzX::throw_if((data_num <= 0), AzInputNotValid, eyec, "Empty data"); 

  AzBytArr ba_buff; 
  AzByte *buff = ba_buff.reset(max_line_len+256, 0); /* +256 just in case */
  
  /*---  1st line indicates sparse/dense  ---*/
  AzFile file(data_fn); 
  
  file.open("rb"); 
  int line0_len = file.gets(buff, max_line_len); 
  AzBytArr s_line0(buff, line0_len); 
  
  int line_no = 0; 
  bool isSparse = false;
  int f_num = if_sparse(s_line0, expected_f_num); 
  if (f_num > 0) {
    isSparse = true; 
    --data_num; /* b/c 1st line is information */
    AzX::throw_if((data_num <= 0), AzInputNotValid, eyec, "Empty sparse data file"); 
    line_no = 1; /* first data line */
  }
  else {
    f_num = expected_f_num; 
    if (f_num <= 0) {
      const AzByte *line0 = s_line0.point(); 
      f_num = countFeatures(line0, line0+line0_len); 
    }
    AzX::throw_if((f_num <= 0), AzInputNotValid, eyec, "No feature in the first line"); 
    file.seek(0);  /* rewind to the beginning */
  }

  /*---  read features  ---*/
  if (max_data_num > 0) {
    data_num = MIN(data_num, max_data_num); 
  }
  m_feat->reform(f_num, data_num); 

  for (int dx = 0; dx < data_num; ++dx, ++line_no) {
    int len = ia_line_len.get(line_no); 
    file.readBytes(buff, len); 
    buff[len] = '\0';  /* to make it a C string */
    if (isSparse) parseDataLine_Sparse(buff, len, f_num, data_fn, line_no+1, m_feat, dx); 
    else          parseDataLine(buff, len, f_num, data_fn, line_no+1, m_feat, dx); 
  }
  file.close(); 
}
template void AzTextMat::readData_Large<AzSmat>(const char *, int, AzSmat *, int); 
template void AzTextMat::readData_Large<AzDmat>(const char *, int, AzDmat *, int); 

/*------------------------------------------------------------------*/
void AzTextMat::_parseDataLine_Sparse(const AzByte *inp, 
                              int inp_len, 
                              int f_num, 
                              const char *data_fn, 
                              int line_no, 
                              /*---  output  ---*/
                              AzIFarr &ifa_ex_val) {
  const char *eyec = "AzTextMat::_parseDataLine_Sparse"; 

  AzIntArr ia_isUsed; 
  ia_isUsed.reset(f_num, 0); 
  int *isUsed = ia_isUsed.point_u(); 

  const AzByte *wp = inp, *line_end = inp + inp_len; 
  for ( ; wp < line_end; ) {
    AzBytArr str_token; 
    AzTools::getString(&wp, line_end, str_token); 
    if (str_token.getLen() > 0) {
      int ex; 
      double val; 
      decomposeFeat(str_token.c_str(), line_no, &ex, &val); 
      if (ex < 0 || ex >= f_num) {
        AzBytArr s("Error in line# "); s << line_no << ": invalid feature# " << ex; 
        AzX::throw_if(true, AzInputError, eyec, s.c_str());     
      }
      if (isUsed[ex]) {
        AzBytArr s("Error in line# "); s << line_no << ": feature# " << ex << " appears more than once."; 
        AzX::throw_if(true, AzInputError, eyec, s.c_str()); 
      }
      if (val != 0) {
        ifa_ex_val.put(ex, val); 
      }
      isUsed[ex] = 1; 
    }
  }
}

/*-------------------------------------------------------------*/
/* ex:val */
void AzTextMat::decomposeFeat(const char *token,
                              int line_no,  
                              /*---  output  ---*/
                              int *ex, 
                              double *val) {
  *ex = my_fno(token, "AzTextMat::decomposeFeat", line_no); 
  *val = 1; 
  const char *ptr = strchr(token, ':');   
  if (ptr != NULL) {
    *val = my_atof(ptr+1, "AzTextMat::decomposeFeat", line_no); 
  }
}

/*------------------------------------------------------------------*/
int AzTextMat::if_sparse(AzBytArr &s_line, 
                         int expected_f_num, 
                         const char *str) {
  const char *eyec = "AzTextMat::if_sparse"; 

  int sparse_f_num = -1; 
  AzBytArr s_sparse(str);  
  AzStrPool sp_tok; 
  AzTools::getStrings(s_line.point(), s_line.length(), &sp_tok); 
  if (sp_tok.size() > 0 && 
      s_sparse.compare(sp_tok.c_str(0)) == 0) {
    if (sp_tok.size() >= 2) {
      sparse_f_num = atol(sp_tok.c_str(1)); 
    }
    AzX::throw_if((sparse_f_num <= 0), AzInputError, eyec, 
            "1st line of sparse data file must be \"sparse dd\" where dd is the feature dimensionality."); 
    AzX::throw_if((expected_f_num > 0 && sparse_f_num != expected_f_num), AzInputError, eyec, 
            "Conflict in feature dim: feature definition file vs. data file.");
  }
  return sparse_f_num; 
}
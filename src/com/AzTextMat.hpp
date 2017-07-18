/* * * * *
 *  AzTextMat.hpp 
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

#ifndef _AZ_TEXT_MAT_HPP_
#define _AZ_TEXT_MAT_HPP_

#include "AzUtil.hpp"
#include "AzSmat.hpp"
#include "AzDmat.hpp"
#include "AzStrPool.hpp"

class AzTextMat {
public: 
  template <class M> /* M: AzDmatc | AzSmatc */
  static void readMatrix(const char *fn, M *m_data, int max_data_num=-1) {
    AzSmat m; 
    readMatrix(fn, &m, max_data_num); 
    m_data->set(&m); 
  }
  static void readMatrix(const char *fn, AzDmat *m_data, int max_data_num=-1) {
    readData_Large(fn, -1, m_data, max_data_num); 
  }  
  static void readMatrix(const char *fn, AzSmat *m_data, int max_data_num=-1) {
    readData_Large(fn, -1, m_data, max_data_num); 
  }
  static void readVector(const char *fn, AzDvect *v_data, int max_data_num=-1); 

protected:
  template <class Mat> /* Mat: AzDmat | AzSmat */
  static void readData_Large(const char *data_fn, 
                         int expected_f_num, 
                         /*---  output  ---*/
                         Mat *m_feat, 
                         /*---  ---*/
                         int max_data_num=-1); 
  template <class Mat> /* Mat: AzSmat | AzDmat */
  static void parseDataLine(const AzByte *inp, 
                              int inp_len, 
                              int f_num, 
                              const char *data_fn, 
                              int line_no, 
                              /*---  output  ---*/
                              Mat *m_feat, 
                              int col) {
    AzIFarr ifa_ex_val; 
    _parseDataLine(inp, inp_len, f_num, data_fn, line_no, ifa_ex_val);   
    m_feat->load(col, &ifa_ex_val);  
  }                           
  static void _parseDataLine(const AzByte *inp, 
                            int inp_len, 
                            int f_num, 
                            const char *data_fn, /* for printing error */
                            int line_no, 
                            /*---  output  ---*/
                            AzIFarr &ifa_ex_val); 

  static int countFeatures(const AzByte *line, 
                           const AzByte *line_end);

  static void scanData(const char *data_fn, 
                         /*---  output  ---*/
                         int &out_data_num, 
                         int &out_max_len); 

  /*---  For the sparse data format  ---*/
  template <class Mat> /* Mat: AzSmat | AzDmat */
  static void parseDataLine_Sparse(const AzByte *inp, 
                              int inp_len, 
                              int f_num, 
                              const char *data_fn, 
                              int line_no, 
                              /*---  output  ---*/
                              Mat *m_feat, 
                              int col) {
    AzIFarr ifa_ex_val; 
    _parseDataLine_Sparse(inp, inp_len, f_num, data_fn, line_no, ifa_ex_val); 
    m_feat->load(col, &ifa_ex_val);
  }   
  static void _parseDataLine_Sparse(const AzByte *inp, 
                              int inp_len, 
                              int f_num, 
                              const char *data_fn, 
                              int line_no, 
                              /*---  output  ---*/
                              AzIFarr &ifa_ex_val); 

  static void decomposeFeat(const char *token, 
                            int line_no, 
                            /*---  output  ---*/
                            int *ex, 
                            double *val); 
  static int if_sparse(AzBytArr &s_line, int expected_f_num, const char *str="sparse"); 
  inline static double my_atof(const char *str, 
                           const char *eyec, 
                           int line_no) {
    if (*str == '\0' || *str >= '0' && *str <= '9' || 
        *str == '+' || *str == '-') {
      return atof(str); 
    }
    AzBytArr s("Invalid number expression in line# ");
    s << line_no << " of the input data file: " << str; 
    AzX::throw_if(true, AzInputError, eyec, s.c_str()); return 0; 
  }

  inline static int my_fno(const char *str, 
                           const char *eyec, 
                           int line_no) {
    if (*str >= '0' && *str <= '9' || 
        *str == '+') {
      return atol(str); 
    }
    AzBytArr s("Invalid field# expression in line# ");
    s << line_no << " of the input data file: " << str; 
    AzX::throw_if(true, AzInputError, eyec, s.c_str()); return -1; 
  }
}; 

#endif 

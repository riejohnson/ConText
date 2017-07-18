/* * * * *
 *  AzCuda_PmatSpa.cu
 *  Copyright (C) 2015 Rie Johnson
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

#include "AzCuda_PmatSpa.cuh" 

#ifdef __AZ_GPU__
  #include "AzCuda.cuh"
  #include "AzCuda_Pmat.cuh"  /* azc_config */
  static void chk_err(const char *eyec, int bb, int tt) {
    AzCuda::check_error(eyec, bb, tt); 
  }   
#else
  extern bool __doDebug; 
  #include "AzPrint.hpp"
  static bool azc_config(int num, int &bb, int &tt, const char *msg="") { return true; }
  static void chk_err(const char *eyec, int bb, int tt) {
    if (__doDebug) AzPrint::writeln(log_out, eyec); 
  }  
#endif 

/*------------------------------------------------*/  
__global__ void azc2_prod_dense1_sparse0(
                  AzFloat *dst, int r_num, int c_num, 
                  const AzFloat *src1, int r_num1, int c_num1, /* dense */
                  const AzFloat *csc_vals, const int *csc_ptrs, const int *csc_rows, 
                  bool do_add)
{
  int num = r_num*c_num; 
  int ex; 
  for (ex = azc_thno; ex < num; ex += azc_thnum) {
#if 1  
    int row = ex % r_num; 
    int col = ex / r_num;       
#else 
    int col = ex % c_num; 
    int row = ex / c_num; 
#endif    
      
    const AzFloat *v1 = _column(row, src1, r_num1); 
      
    double val = 0; 
    int bx = csc_ptrs[col]; 
    int ex = csc_ptrs[col+1]; 
    for (int ix = bx; ix < ex; ++ix) {
      val += csc_vals[ix] * v1[csc_rows[ix]]; 
    }
    if (do_add) _entry(row, col, dst, r_num) += val; 
    else        _entry(row, col, dst, r_num) = val; 
  }
}
void azc2call_prod_dense1_sparse0(
                  AzFloat *dst, int r_num, int c_num, 
                  const AzFloat *src1, int r_num1, int c_num1, /* dense */
                  const AzFloat *csc_vals, const int *csc_ptrs, const int *csc_rows, 
                  bool do_add)
{                           
  int bb, tt; 
  azc_config(r_num*c_num, bb, tt, "azc2call_add_prod_dense1_sparse0"); 
  azc_kernel(azc2_prod_dense1_sparse0,bb,tt)(dst, r_num, c_num, src1, r_num1, c_num1, csc_vals, csc_ptrs, csc_rows, do_add); 
  chk_err("azc2call_prod_dense1_sparse0",bb,tt); 
}
 
/*------------------------------------------------*/
/* dst += sparse * tran(dense) */
__global__ void azc2_prod_sparse0_dense1(
                  AzFloat *dst, int r_num, int c_num, 
                  const AzFloat *csr_vals, 
                  const int *nzrow_ptrs, const int *nzrow_rows, int nzrow_num, /* only for nonzero rows */
                  const int *csr_cols, 
                  const AzFloat *src2, int r_num2, int c_num2, /* dense */  
                  bool do_add)
{
  int num = nzrow_num*c_num; 
  for (int ex = azc_thno; ex < num; ex += azc_thnum) { /* going through dst (output) entries */    
    int col = ex % c_num, rx = ex / c_num;     

    double val = 0; 
    int row = nzrow_rows[rx]; 
    int b_ix = nzrow_ptrs[rx]; 
    int e_ix = nzrow_ptrs[rx+1]; 
    for (int ix = b_ix; ix < e_ix; ++ix) {
      AzFloat val2 = _entry(col, csr_cols[ix], src2, r_num2); 
      val += csr_vals[ix] * val2; 
    }
    if (do_add) _entry(row, col, dst, r_num) += val; 
    else        _entry(row, col, dst, r_num) = val; 
  }
}
void azc2call_prod_sparse0_dense1(
                  AzFloat *dst, int r_num, int c_num, 
                  const AzFloat *csr_vals, 
                  const int *nzrow_ptrs, const int *nzrow_rows, int nzrow_num, /* only for nonzero rows */
                  const int *csr_cols,                    
                  const AzFloat *src2, int r_num2, int c_num2, /* dense */                  
                  bool do_add)
{                           
  int bb, tt;   
  int num = nzrow_num*c_num; 
  if (num <= 0) return; 
  azc_config(num, bb, tt, "azc2call_add_prod_sparse0_dense1");  
  azc_kernel(azc2_prod_sparse0_dense1,bb,tt)(dst, r_num, c_num, csr_vals, 
         nzrow_ptrs, nzrow_rows, nzrow_num, csr_cols, 
         src2, r_num2, c_num2, do_add); 
  chk_err("azc2call_prod_sparse0_dense1",bb,tt); 
}

/*------------------------------------------------*/
/* alpha*sparse * tran(dense) */
__global__ void azc2_prod_sparse0_dense1_a(
                  AzFloat *dst, int r_num, int c_num, 
                  const AzFloat *csr_vals, 
                  const int *nzrow_ptrs, const int *nzrow_rows, int nzrow_num, /* only for nonzero rows */
                  const int *csr_cols, 
                  const AzFloat *src2, int r_num2, int c_num2, /* dense */  
                  AzFloat alpha, bool do_add)
{
  int num = nzrow_num*c_num; 
  for (int ex = azc_thno; ex < num; ex += azc_thnum) { /* going through dst (output) entries */    
    int col = ex % c_num, rx = ex / c_num;     

    double val = 0; 
    int row = nzrow_rows[rx]; 
    int b_ix = nzrow_ptrs[rx]; 
    int e_ix = nzrow_ptrs[rx+1]; 
    for (int ix = b_ix; ix < e_ix; ++ix) {
      AzFloat val2 = _entry(col, csr_cols[ix], src2, r_num2); 
      val += csr_vals[ix] * val2; 
    }
    
    if (do_add) _entry(row, col, dst, r_num) += alpha*val; 
    else        _entry(row, col, dst, r_num) = alpha*val; 
  }
}
void azc2call_prod_sparse0_dense1_a(
                  AzFloat *dst, int r_num, int c_num, 
                  const AzFloat *csr_vals, 
                  const int *nzrow_ptrs, const int *nzrow_rows, int nzrow_num, /* only for nonzero rows */
                  const int *csr_cols,                    
                  const AzFloat *src2, int r_num2, int c_num2, /* dense */                  
                  AzFloat alpha, bool do_add)
{                           
  int bb, tt;   
  int num = nzrow_num*c_num; 
  if (num <= 0) return; 
  azc_config(num, bb, tt, "azc2call_add_prod_sparse0_dense1_ab");  
  if (alpha == 1) /* faster */
    azc_kernel(azc2_prod_sparse0_dense1,bb,tt)(dst, r_num, c_num, csr_vals, 
               nzrow_ptrs, nzrow_rows, nzrow_num, csr_cols, src2, r_num2, c_num2, do_add); 
  else 
    azc_kernel(azc2_prod_sparse0_dense1_a,bb,tt)(dst, r_num, c_num, csr_vals, 
         nzrow_ptrs, nzrow_rows, nzrow_num, csr_cols, src2, r_num2, c_num2, alpha, do_add); 
  chk_err("azc2call_prod_sparse0_dense1_ab",bb,tt); 
}

/*------------------------------------------------*/
__global__ void azc2_add_sparse(
                  AzFloat *dst, int r_num, int c_num, 
                  const AzFloat *csc_vals, const int *csc_rows, const int *csc_cols, int vals_num, 
                  AzFloat coeff)
{
  for (int ex = azc_thno; ex < vals_num; ex += azc_thnum) {
    _entry(csc_rows[ex], csc_cols[ex], dst, r_num) += coeff*csc_vals[ex];     
  }
}
void azc2call_add_sparse(
                  AzFloat *dst, int r_num, int c_num, 
                  const AzFloat *csc_vals, const int *csc_rows, const int *csc_cols, int vals_num, 
                  AzFloat coeff)
{
  if (vals_num <= 0) return; 
  int bb, tt; 
  azc_config(vals_num, bb, tt, "azccall_add_sparse"); 
  azc_kernel(azc2_add_sparse,bb,tt)(dst, r_num, c_num, csc_vals, csc_rows, csc_cols, vals_num, coeff); 
  chk_err("azc2call_add_sparse",bb,tt); 
}

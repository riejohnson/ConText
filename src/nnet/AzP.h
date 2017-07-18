/* * * * *
 *  AzP.h
 *  Copyright (C) 2014-2016 Rie Johnson
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

#ifndef _AZ_P_H_
#define _AZ_P_H_
#ifdef __AZ_GPU__
  #include <cuda_runtime.h>
  #include <cublas_v2.h> 
  #include <cusparse_v2.h>   
  #include <curand.h>
  #define azc_kernel(f,a,b)   f<<<a,b>>>
  #define azc_thno (blockDim.x * blockIdx.x + threadIdx.x) 
  #define azc_thnum (blockDim.x * gridDim.x) 
#else
  #include "AzP_cpu.h"
#endif 

#ifdef __AZ_DOUBLE__
  typedef double AzFloat;
  #define azc_exp_arg_min (-700)
  #define azc_exp_arg_max (700)
  #define azc_epsilon 1e-307
#else
  typedef float AzFloat;
  #define azc_exp_arg_min (-80)
  #define azc_exp_arg_max (80)
  #define azc_epsilon 1e-37
#endif 

/*---  used only for GPU though  ---*/
#define azc_numShared 4096
  
/* c: column, e: ptr to elements, r: the number of rows */
#define _column(c,e,r) ((e)+(c)*(r))
/* r: row, c: column, p: ptr to elements, rn: the number of rows */
#define _entry(r,c,p,rn) (*((p)+(c)*(rn)+(r))) 
#define myexp(x) exp(MAX(azc_exp_arg_min,MIN(azc_exp_arg_max,(x))))
#define azc_exp_mask(x) ( ((x) < azc_exp_arg_min || (x) > azc_exp_arg_max) ? 0 : 1 )
#endif 
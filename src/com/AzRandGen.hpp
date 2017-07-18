/* * * * *
 *  AzpRandGen.hpp
 *  Copyright (C) 2014-2015,2017 Rie Johnson
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

#ifndef _AZ_RAND_GEN_HPP_
#define _AZ_RAND_GEN_HPP_

#include "AzDmat.hpp"

/* random number generator */
class AzRandGen {
protected:   
  bool do_myown;  /* since 4/27/2017 */
  int myown_seed; 
  unsigned long long myown_val; 
public:
  AzRandGen() : do_myown(false), myown_seed(1), myown_val((unsigned long long)myown_seed) {}
  void _srand_(int seed) {
    do_myown = true; 
    myown_seed = seed; 
    myown_val = (unsigned long long)seed; 
  }
  inline unsigned long long _rand_ULL() {
    if (!do_myown) return rand(); 
    else {
      myown_val = myown_val * (unsigned long long)25214903917 + 11; 
      return myown_val; 
    }   
  }
  int _rand_() {
    if (do_myown) return _rand_ULL() % (unsigned long long)2147483648; 
    else          return rand(); 
  }
  
  /*---  Gaussian  ---*/
  void gaussian(double param, AzDmat *m) { /* must be formatted by caller */
    AzX::throw_if((m == NULL), "AzRandGen::gaussian", "null pointer"); 
    for (int col = 0; col < m->colNum(); ++col) gaussian(param, m->col_u(col)); 
  }
  void gaussian(double param, AzDvect *v) {
    AzX::throw_if((v == NULL), "AzRandGen::gaussian", "null pointer");   
    gaussian(v->point_u(), v->rowNum()); 
    v->multiply(param); 
  }

  /*---  uniform  ---*/
  void uniform(double param, AzDmat *m) { /* must be formatted by caller */
    for (int col = 0; col < m->colNum(); ++col) uniform(param, m->col_u(col)); 
  }
  void uniform(double param, AzDvect *v) {
    AzX::throw_if((v == NULL), "AzRandGen::uniform", "null pointer");   
    double *val = v->point_u();     
    uniform_01(val, v->rowNum());  /* [0,1] */
    for (int ex = 0; ex < v->rowNum(); ++ex) {
      val[ex] = (val[ex]*2-1)*param;  /* -> [-param, param] */
    }
  }
  void uniform_01(AzDvect &v) { uniform_01(v.point_u(), v.rowNum()); }
  void uniform_01(AzDmat &m) { for (int col = 0; col < m.colNum(); ++col) uniform_01(*m.col_u(col)); }
  template <class T> /* T: float | double */
  void uniform_01(T *val, size_t sz) { /* [0,1] */
    int denomi = 30000; 
    for (size_t ix = 0; ix < sz; ++ix) {
      val[ix] = (T)((double)(_rand_() % denomi) / (double)denomi); 
    }
  }

  /*---  polar method  ---*/
  template <class T> /* T: float | double */
  void gaussian(T *val, size_t sz) {
    for (size_t ix = 0; ix < sz; ) {
  	  double u1 = 0, u2 = 0, s = 0; 
  	  for ( ; ; ) {
        u1 = uniform1(); u2 = uniform1(); 
        s = u1*u1 + u2*u2; 
    		if (s > 0 && s < 1) break;
      } 
  	  double x1 = u1 * sqrt(-2*log(s)/s); 
      val[ix++] = x1; 
      if (ix >= sz) break; 
	  
	    double x2 = u2 * sqrt(-2*log(s)/s); 
      val[ix++] = x2; 
    }
  }  
  
protected:   
  virtual double uniform1() {
    int mymax = 32000; 
  	double mymaxmax = mymax * (mymax-1) + mymax-1; 
  	int rand0 = _rand_() % mymax;
    int rand1 = _rand_() % mymax; 
  	double output = (rand0 * mymax + rand1) / mymaxmax; /* (0,1) */
  	output = 2 * output - 1;  /* (-1,1) */
  	return output; 
  }  
}; 

#endif
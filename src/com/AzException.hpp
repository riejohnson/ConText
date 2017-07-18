/* * * * *
 *  AzException.hpp 
 *  Copyright (C) 2011,2012,2017 Rie Johnson
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

#ifndef _AZ_EXCEPTION_HPP_
#define _AZ_EXCEPTION_HPP_

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <iostream>
#include <sstream>
using namespace std; 

enum AzRetCode {
  AzNormal=0, 
  AzAllocError=10,
  AzFileIOError=20,
  AzInputError=30, 
  AzInputMissing=31, 
  AzInputNotValid=32, 
  AzCudaErr=33, 
  AzConflict=100,  /* all others */
}; 

/*-----------------------------------------------------*/
class AzException {
public:
  AzException(const char *string1, 
              const char *string2, 
              const char *string3=NULL) 
  {
    reset(AzConflict, string1, string2, string3); 
  }

  AzException(AzRetCode retcode, 
              const char *string1, 
              const char *string2, 
              const char *string3=NULL) 
  {
    reset(retcode, string1, string2, string3); 
  }

  template <class T>
  AzException(AzRetCode retcode, 
              const char *string1, 
              const char *string2, 
              const char *string3, 
              T anything)
  {
    reset(retcode, string1, string2, string3); 
    s3 << "; " << anything; 
  }

  void reset(AzRetCode retcode, 
             const char *str1, 
             const char *str2, 
             const char *str3)
  {
    this->retcode = retcode; 
    if (str1 != NULL) s1 << str1; 
    if (str2 != NULL) s2 << str2; 
    if (str3 != NULL) s3 << str3; 
  }

  AzRetCode getReturnCode() {
    return retcode;   
  }

  string getMessage() {
    if (retcode == AzNormal) {}
    else if (retcode == AzAllocError)    message << "!Memory alloc error!"; 
    else if (retcode == AzFileIOError)   message << "!File I/O error!"; 
    else if (retcode == AzInputError)    message << "!Input error!"; 
    else if (retcode == AzInputMissing)  message << "!Missing input!"; 
    else if (retcode == AzInputNotValid) message << "!Input value is not valid!"; 
    else if (retcode == AzCudaErr)       message << "!cuda error!";  
    else if (retcode == AzConflict)      message << "Conflict"; 
    else                                 message << "Unknown error"; 

    if (retcode != AzNormal) message << ": "; 
    if (s1.str().find("Az") == 0) message << "(Detected in " << s1.str() << ") " << endl; 
    else                          message << s1.str() << " "; 
    message << s2.str(); 
    if (s3.str().length() > 0) message << " " << s3.str(); 
    message << endl; 
    return message.str(); 
  }

protected:
  AzRetCode retcode; 

  stringstream s1, s2, s3; 
  stringstream message; 
};

/*---  ---*/
class AzX {
public:
  inline static void throw_if(bool cond, const char *eyec, const char *str1="", const char *str2="") {
    if (!cond) return; 
    throw new AzException(eyec, str1, str2);
  } 
  inline static void throw_if(bool cond, AzRetCode ret, const char *eyec, const char *str1="", const char *str2="") {
    if (!cond) return; 
    throw new AzException(ret, eyec, str1, str2);
  }   
  inline static void no_support(bool cond, const char *eyec, const char *item) {
    if (!cond) return; 
  throw new AzException(AzInputError, eyec, "NO SUPPORT for ", item); 
  }
  inline static void throw_if_null(const void *ptr, const char *eyec, const char *str1="null input", const char *str2="") {
    if (ptr != NULL) return; 
    throw new AzException(eyec, str1, str2); 
  }
  inline static void throw_if_null(const void *ptr1, const void *ptr2, const char *eyec, 
                                   const char *str1="null input", const char *str2="") {
    if (ptr1 != NULL || ptr2 != NULL) return; 
    throw new AzException(eyec, str1, str2); 
  }
  inline static void throw_if_null(const void *ptr1, const void *ptr2, const void *ptr3, const char *eyec, 
                                   const char *str1="null input", const char *str2="") {
    if (ptr1 != NULL || ptr2 != NULL || ptr3 != NULL) return; 
    throw new AzException(eyec, str1, str2); 
  } 
  inline static void throw_if_null(const void *ptr, AzRetCode ret, const char *eyec, const char *str1="null input", const char *str2="") {
    if (ptr != NULL) return; 
    throw new AzException(ret, eyec, str1, str2); 
  }
  
  /* Print before throwing.  Use these to handle cuda errors.  */
  inline static void pthrow_if(bool cond, const char *eyec, const char *str1="", const char *str2="") {
    if (!cond) return; 
    AzException *ep = new AzException(eyec, str1, str2); 
    cout << ep->getMessage() << endl; 
    throw ep; 
  }  
  inline static void pthrow(AzRetCode ret, const char *eyec, const char *str1="", const char *str2="") {
    pthrow_if(true, ret, eyec, str1, str2); 
  }
  inline static void pthrow_if(bool cond, AzRetCode ret, const char *eyec, const char *str1="", const char *str2="") {
    if (!cond) return; 
    AzException *ep = new AzException(ret, eyec, str1, str2); 
    cout << ep->getMessage() << endl; 
    throw ep; 
  }   
}; 
#endif 

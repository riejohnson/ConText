/* * * * *
 *  AzpCompoSet_.hpp
 *  Copyright (C) 2014-2015 Rie Johnson
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

#ifndef _AZP_COMPO_SET__HPP_
#define _AZP_COMPO_SET__HPP_

#include "AzpLoss_.hpp"
#include "AzpWeight_.hpp"
#include "AzpActiv_.hpp"
#include "AzpDropout_.hpp"
#include "AzpPooling_.hpp"
#include "AzpPooling_var_.hpp"
#include "AzpResNorm_.hpp"
#include "AzpPatch_.hpp"
#include "AzpPatch_var_.hpp"

/* component set used by each layer */
class AzpCompoSet_ {
public:   
  virtual ~AzpCompoSet_() {}
  virtual const AzpLoss_ *loss() const = 0; 
  virtual const AzpPatch_ *patch() const = 0; 
  virtual const AzpPatch_var_ *patch_var() const = 0; 
  virtual const AzpWeight_ *weight() const = 0; 
  virtual const AzpActiv_ *activ() const = 0; 
  virtual const AzpDropout_ *dropout() const = 0; 
  virtual const AzpPooling_ *pool() const = 0; 
  virtual const AzpPooling_var_ *pool_var() const = 0; 
  virtual const AzpResNorm_ *resnorm() const = 0; 
  void check_if_ready(const char *msg) const {
    AzX::throw_if((loss() == NULL || patch() == NULL || patch_var() == NULL || 
        weight() == NULL || activ() == NULL || dropout() == NULL || 
        pool() == NULL || pool_var() == NULL || resnorm() == NULL), 
        "AzpCompoSet_::check_if_ready", msg, "Some component is missing."); 
  }
}; 

/************************************************/
class AzpNetCompoPtrs {
public: 
  AzpLoss_ *loss; 
  bool check_if_ready(const char *msg, bool dont_throw=false) const {
    const char *eyec = "AzpNetCompoSet::check_if_ready"; 
    if (loss == NULL) {
      if (dont_throw) return false; 
      AzX::throw_if(true, eyec, msg, "Some component is not ready"); 
    }
    return true; 
  }
  AzpNetCompoPtrs() : loss(NULL) {}
  ~AzpNetCompoPtrs() {
    reset(); 
  }
  void reset() {
    delete loss; loss = NULL; 
  }
  void reset(const AzpCompoSet_ *cset) {
    AzX::throw_if((cset == NULL), "AzpNetCompoSet::reset(compo_set)", "No component set"); 
    reset(); 
    loss = cset->loss()->clone();
  }
  void reset(const AzpNetCompoPtrs *inp) {
    reset(); 
    if (inp == NULL || !inp->check_if_ready("", true)) return; 
    loss = inp->loss->clone();
  }
  void write(AzFile *file) const {
    loss->write(file); 
  }    
  void read(AzFile *file) {
    loss->read(file); 
  }
};

/************************************************/
class AzpLayerCompoPtrs {
public: 
  AzpPatch_ *patch; 
  AzpPatch_var_ *patch_var; 
  AzpWeight_ *weight, *weight2; 
  AzpActiv_ *activ; 
  AzpDropout_ *dropout; 
  AzpPooling_ *pool; 
  AzpPooling_var_ *pool_var;   
  AzpResNorm_ *resnorm; 
  
  #define AzpLayerCompo_Num 9
  AzpCompo_ *compo[AzpLayerCompo_Num]; 
  int num; 
  
  bool check_if_ready(const char *msg, bool dont_throw=false) const {
    const char *eyec = "AzpLayerCompoSet::check_if_ready"; 
    if (patch == NULL || patch_var == NULL || 
        weight == NULL || weight2 == NULL || 
        activ == NULL || dropout == NULL || 
        pool == NULL || pool_var == NULL || resnorm == NULL) {
      if (dont_throw) return false; 
      AzX::throw_if(true, eyec, msg, "Some component is not ready"); 
    }
    return true; 
  }
  AzpLayerCompoPtrs() : num(AzpLayerCompo_Num), 
                        patch(NULL), patch_var(NULL), weight(NULL), weight2(NULL), 
                        activ(NULL), dropout(NULL), 
                        pool(NULL), pool_var(NULL), resnorm(NULL) {
    for (int ix = 0; ix < num; ++ix) compo[ix] = NULL; 
  }
  ~AzpLayerCompoPtrs() {
    reset(); 
  }
  void reset() {
    for (int ix = 0; ix < num; ++ix) {
      delete compo[ix]; compo[ix] = NULL; 
    }
    patch=NULL; patch_var=NULL; weight=NULL; weight2=NULL; 
    activ=NULL; dropout=NULL; pool=NULL; pool_var=NULL; resnorm=NULL;   
  }
  void reset(const AzpCompoSet_ *cset) {
    AzX::throw_if((cset == NULL), "AzpLayerCompoSet::reset(compo_set)", "No component set"); 
    reset(); 
    int ix = 0; 
    patch = cset->patch()->clone();         compo[ix++] = patch; 
    patch_var = cset->patch_var()->clone(); compo[ix++] = patch_var; 
    weight = cset->weight()->clone();       compo[ix++] = weight; 
    weight2 = cset->weight()->clone();       compo[ix++] = weight2;     
    activ = cset->activ()->clone();         compo[ix++] = activ; 
    dropout = cset->dropout()->clone();     compo[ix++] = dropout; 
    pool = cset->pool()->clone();           compo[ix++] = pool; 
    pool_var = cset->pool_var()->clone();   compo[ix++] = pool_var; 
    resnorm = cset->resnorm()->clone();     compo[ix++] = resnorm; 
  }
  void reset(const AzpLayerCompoPtrs *inp) {
    reset(); 
    if (inp == NULL || !inp->check_if_ready("", true)) return; 
    int ix = 0; 
    patch = inp->patch->clone();         compo[ix++] = patch; 
    patch_var = inp->patch_var->clone(); compo[ix++] = patch_var; 
    weight = inp->weight->clone();       compo[ix++] = weight;  
    weight2 = inp->weight2->clone();     compo[ix++] = weight2;      
    activ = inp->activ->clone();         compo[ix++] = activ; 
    dropout = inp->dropout->clone();     compo[ix++] = dropout; 
    pool = inp->pool->clone();           compo[ix++] = pool; 
    pool_var = inp->pool_var->clone();   compo[ix++] = pool_var; 
    resnorm = inp->resnorm->clone();     compo[ix++] = resnorm; 
  }
  void write(AzFile *file) const {
    for (int ix = 0; ix < num; ++ix) compo[ix]->write(file); 
  }    
  void read(AzFile *file) {
    for (int ix = 0; ix < num; ++ix) {
//      cout << "AzpCompoSet_::read, ix=" << ix << endl; 
      compo[ix]->read(file); 
    }
  }
};
#endif 
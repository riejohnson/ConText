/* * * * *
 *  AzMultiConn.hpp
 *  Copyright (C) 2017 Rie Johnson
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

#ifndef _AZ_MULTI_CONN_HPP_
#define _AZ_MULTI_CONN_HPP_

#include "AzUtil.hpp"
#include "AzStrPool.hpp"
#include "AzParam.hpp"
#include "AzPrint.hpp"

/**
  * - Layer#0 is always the first layer. 
  * - The last layer is always the top layer. 
  **/
/*------------------------------------------------------------*/ 
class AzMultiConn {
public:  
  int lsz; 
  AzIIarr iia_conn; 
  AzStrPool sp_conn; 
  AzIntArr ia_order; 
  AzDataArr<AzIntArr> aia_below, aia_above; 
  bool do_add, do_show_nicknm; 
  bool is_multi; /* set by resetParam.  false if there is no conn parameter(s). */
  AzDataArr<AzIntArr> aia_orelease; /* to save memory: 11/26/2016 */
/*  static const int version = 0; */
  static const int version = 1; /* for read|write_compact of sp_conn: 6/20/2017 */
  static const int reserved_len = 64; 
  
public: 
  AzMultiConn() : do_add(true), is_multi(false), do_show_nicknm(true) {}
  void reset() {
    iia_conn.reset(); 
    sp_conn.reset(); 
    ia_order.reset(); 
    aia_below.reset(); 
    aia_above.reset(); 
    aia_orelease.reset(); 
  }    
  void write(AzFile *file) const {
    AzTools::write_header(file, version, reserved_len); 
    file->writeBool(is_multi);    
    file->writeBool(do_add); 
    file->writeInt(lsz); 
    iia_conn.write(file); 
    sp_conn.write_compact(file); 
    ia_order.write(file); 
    aia_below.write(file); 
    aia_above.write(file); 
  }
  void read(AzFile *file) {
    int my_version = AzTools::read_header(file, reserved_len);  
    is_multi = file->readBool(); 
    do_add = file->readBool(); 
    lsz = file->readInt(); 
    iia_conn.read(file); 
    if (my_version == 0) sp_conn.read(file); 
    else                 sp_conn.read_compact(file); 
    ia_order.read(file); 
    aia_below.read(file); 
    aia_above.read(file); 
  }
  
  virtual bool is_multi_conn() const { return is_multi; }
  virtual void setup(bool is_warmstart) {
    order_layers(lsz, iia_conn, ia_order, aia_below, aia_above);  
    insert_connectors(ia_order, aia_below, aia_above); 
    reset_for_output_release(); 
  }
  virtual bool is_additive() const { return do_add; }
    
  virtual void resetParam(AzParam &azp, int hid_num, bool is_warmstart=false); 
  virtual void printParam(const AzOut &out) const; 

  virtual const AzIntArr &order() const { return ia_order; }
  virtual int below(int lx) {
    check_lay_ind(lx, "AzMultiConn::below"); 
    if (aia_below[lx]->size() <= 0) return -1; 
    return (*aia_below[lx])[0]; 
  }
  virtual int above(int lx) {
    check_lay_ind(lx, "AzMultiConn::above"); 
    if (aia_above[lx]->size() <= 0) return -1; 
    return (*aia_above[lx])[0]; 
  }  
  virtual const AzIntArr &all_below(int lx) {
    check_ind(lx, "AzMultiConn::all_below"); 
    return (*aia_below[lx]); 
  }
  virtual const AzIntArr &all_above(int lx) {
    check_ind(lx, "AzMultiConn::all_above"); 
    return (*aia_above[lx]); 
  }

  static void show_below_above(const AzIntArr &ia_below, const AzIntArr &ia_above, AzBytArr &s); 
 
  template <class M> void release_output(int curr_lx, AzDataArr<M> &amat) const {
    if (aia_orelease.size() <= 0) return; 
    const AzIntArr *ia = aia_orelease[curr_lx]; 
    for (int ix = 0; ix < ia->size(); ++ix) amat((*ia)[ix])->reset(); 
  }
 
protected: 
  virtual void check_lay_ind(int lx, const char *eyec) const {
    AzX::throw_if(lx < 0 || lx >= lsz, eyec, "out of range (layers)"); 
  }
  virtual void check_ind(int lx, const char *eyec) const {
    AzX::throw_if(lx < 0 || lx >= ia_order.size(), eyec, "out of range (layers and connectors)");  
  }

  virtual void _parse_binary_conns(int hid_num, AzParam &azp, AzIIarr &iia_conn, AzStrPool &sp_conn) const; 
  virtual void _parse_conns(int hid_num, AzParam &azp, AzIIarr &iia_conn, AzStrPool &sp_conn) const; 
  virtual int parse_layerno(const char *str, int hid_num) const; 
  virtual void order_layers(int layer_num, const AzIIarr &iia_conn, 
                          /*---  output  ---*/
                          AzIntArr &ia_order, AzDataArr<AzIntArr> &aia_below, AzDataArr<AzIntArr> &aia_above) const; 
  virtual void insert_connectors(AzIntArr &ia_order, AzDataArr<AzIntArr> &aia_below, AzDataArr<AzIntArr> &aia_above) const; /* inout */
  void reset_for_output_release(); 
}; 
#endif 

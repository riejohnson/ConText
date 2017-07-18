/* * * * *
 *  AzMultiConn.cpp
 *  Copyright (C) 2016-2017 Rie Johnson
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
#include "AzMultiConn.hpp"
/*------------------------------------------------------------*/ 
#define kw_conn "conn"
#define kw_newconn "conn="
#define conn_dlm '-'
#define kw_top "top"
#define kw_do_add "AdditiveConn"
#define kw_do_concat "ConcatConn"
/* output: iia_conn, sp_conn */
void AzMultiConn::resetParam(AzParam &azp, int hid_num, bool is_warmstart) {
  const char *eyec = "AzMultiConn::resetParam"; 
  if (is_warmstart) return; 
  lsz = hid_num+1; 
  iia_conn.reset(); 
  sp_conn.reset(); 

  _parse_conns(hid_num, azp, iia_conn, sp_conn);  /* new interface */
  if (iia_conn.size() <= 0) _parse_binary_conns(hid_num, azp, iia_conn, sp_conn);  /* old interface:for compatibility */
  iia_conn.unique(); /* remove duplicated pairs */
  
  /*---  default  ---*/
  if (iia_conn.size() == 0) {
    is_multi = false; 
    AzBytArr s_conn; 
    for (int lx = 0; lx < hid_num; ++lx) {
      int below = lx, above = lx + 1; 
      iia_conn.put(below, above); 
      s_conn << lx << conn_dlm; 
    }
    s_conn << "top"; 
    sp_conn.put(&s_conn); 
  }
  else {
    is_multi = true; 
    bool do_concat = false; 
    azp.swOn(&do_concat, kw_do_concat); 
    azp.swOn(&do_add, kw_do_add);
    if (do_concat) do_add = false;  
  }
}

/*------------------------------------------------------------*/ 
/* old interface: conn0=0-1 conn1=1-top conn2=0-top ... */
/* can be static */
void AzMultiConn::_parse_binary_conns(int hid_num, AzParam &azp, AzIIarr &iia_conn, AzStrPool &sp_conn) const {
  const char *eyec = "AzMultiConn::_parse_binary_conns"; 
  int top = hid_num; 
  for (int ix = 0; ; ++ix) {
    AzBytArr s_conn; 
    AzBytArr s_kw(kw_conn); s_kw << ix << "="; 
    azp.vStr(s_kw.c_str(), &s_conn); 
    if (s_conn.length() <= 0) break; 
    
    AzStrPool sp(32,32); 
    AzTools::getStrings(s_conn.point(), s_conn.length(), conn_dlm, &sp); 

    AzX::throw_if((sp.size() != 2), AzInputError, "Expected the format like n-m for", kw_conn); 
    int below = parse_layerno(sp.c_str(0), hid_num); 
    int above = parse_layerno(sp.c_str(1), hid_num); 
    AzX::throw_if((below == top), AzInputError, eyec, "No edge is allowed to go out of the top layer"); 
    iia_conn.put(below, above); 
    sp_conn.put(&s_conn); 
  }
}
  
/*------------------------------------------------------------*/ 
/* new interface: conn=0-1-top,0-top */
/* can be static */
void AzMultiConn::_parse_conns(int hid_num, AzParam &azp, AzIIarr &iia_conn, AzStrPool &sp_conn) const {
  const char *eyec = "AzpCNet3_multi::_parse_conns"; 
  int top = hid_num; 
  AzBytArr s_conn; 
  azp.vStr(kw_newconn, &s_conn); 
  if (s_conn.length() <= 0) return; 
  
  sp_conn.put(&s_conn); /* for compatibility */
    
  AzStrPool sp0(32,32); 
  AzTools::getStrings(s_conn.point(), s_conn.length(), ',', &sp0); 
  AzBytArr s; 
  bool doing_nicknm = false; 
  for (int ix = 0; ix < sp0.size(); ++ix) {
    if (do_show_nicknm && ix > 0) s << ','; 
    
    AzStrPool sp(32,32); 
    AzTools::getStrings(sp0.point(ix), sp0.getLen(ix), '-', &sp); 
    AzX::throw_if(sp.size() < 2, AzInputError, "Specify e.g., 0-1-2-top,0-top following ", kw_conn); 
    AzIntArr ia_layno; 
    int below = -1; 
    for (int jx = 0; jx < sp.size(); ++jx) {
      int above = parse_layerno(sp.c_str(jx), hid_num); 
      if (jx > 0) {
        AzX::throw_if(below == top, AzInputError, eyec, "No edge is allowed to go out of the top layer"); 
        iia_conn.put(below, above); 
      }
      below = above; 
      if (do_show_nicknm) {
        if (jx > 0) s << '-'; 
        AzBytArr s_nicknm, s_pfx; s_pfx << below; azp.reset_prefix(s_pfx.c_str()); 
        #define kw_nicknm "name=" /* must be same as defined in AzpReLayer.hpp" */
        azp.vStr(kw_nicknm, &s_nicknm); azp.reset_prefix();
        if (s_nicknm.length() > 0) { s << s_nicknm.c_str(); doing_nicknm = true; }
        else if (below == top)     s << kw_top; 
        else                       s << below;         
      }
    }
  }
  if (doing_nicknm) {
    AzBytArr s0(sp_conn.c_str(0)); 
    s0 << "   ( " << s << " )"; 
    sp_conn.reset(); 
    sp_conn.put(&s0);     
  }
}  

/*------------------------------------------------------------*/ 
/* input: sp_conn */
void AzMultiConn::printParam(const AzOut &out) const {
  AzPrint o(out); 
  if (sp_conn.size() == 1) {
    o.printV(kw_newconn, sp_conn.c_str(0)); 
  }
  else { /* old interface: conn0=0-1 conn1=1-2 ... */
    for (int ix = 0; ix < sp_conn.size(); ++ix) {
      AzBytArr s_kw(kw_conn); s_kw << ix << "="; 
      o.printV(s_kw.c_str(), sp_conn.c_str(ix)); 
    }
  }
  if (is_multi) {
    if (do_add) o.printSw(kw_do_add, do_add); 
    else        o.printSw(kw_do_concat, true); 
  }
  o.printEnd(); 
}

/*------------------------------------------------------------*/ 
/* can be static */
int AzMultiConn::parse_layerno(const char *str, int hid_num) const {
  const char *eyec = "AzMultiConn::parse_layerno"; 
  int layer_no = -1; 
  if (strcmp(str, kw_top) == 0) layer_no = hid_num; 
  else {
    AzX::throw_if((*str < '0' || *str > '9'), AzInputError, eyec, "Invalid layer#", str); 
    layer_no = atol(str); 
    AzX::throw_if((layer_no < 0 || layer_no > hid_num), AzInputError, eyec, "layer# is out of range", str); 
  }
  return layer_no; 
}  

/*------------------------------------------------------------*/ 
/* can be static */
void AzMultiConn::order_layers(int layer_num, 
                          const AzIIarr &iia_conn, 
                          /*---  output  ---*/
                          AzIntArr &ia_order, 
                          AzDataArr<AzIntArr> &aia_below, 
                          AzDataArr<AzIntArr> &aia_above) const {
  const char *eyec = "AzMultiConn::order_layers"; 
  AzIntArr ia_done(layer_num, 0); 
  int *done = ia_done.point_u(); 
  aia_below.reset(layer_num); 
  aia_above.reset(layer_num); 
  for (int ix = 0; ix < iia_conn.size(); ++ix) {
    int below, above; 
    iia_conn.get(ix, &below, &above); 
    aia_below(above)->put(below); 
    aia_above(below)->put(above); 
  }

  ia_order.reset(); 
  int num = 0; 
  for ( ; num < layer_num; ) {
    int org_num = num; 
    for (int lx = 0; lx < layer_num; ++lx) {
      if (done[lx] == 1) continue; 
      bool is_ready = true; 
      for (int bx = 0; bx < aia_below[lx]->size(); ++bx) {
        int below = (*aia_below[lx])[bx]; 
        if (done[below] != 1) {
          is_ready = false; 
          break; 
        }
      }
      if (is_ready) {      
        ia_order.put(lx); 
        done[lx] = 1; 
        ++num; 
        break; 
      }
    }
    AzX::throw_if(num==org_num, AzInputError, eyec, "deadlock"); /* if not making a progress, there may be a loop */
  }  
  
  int top_num = 0, bottom_num = 0; 
  int top_lx = -1; 
  for (int lx = 0; lx < ia_order.size(); ++lx) {
    if (aia_above[lx]->size() <= 0) {
      top_lx = lx; 
      ++top_num; 
    }
  }
  AzX::throw_if(ia_order.size() <= 0, eyec, "No layers are specified."); 
  AzX::throw_if(ia_order[0] != 0, eyec, "Layer#0 must be the first layer."); /* ReNet assumes this */
  AzX::throw_if(top_num != 1, AzInputError, eyec, "There is more than one top layer or no top layer or an unconnected layer."); 
  AzX::throw_if(top_lx != ia_order[ia_order.size()-1], eyec, "The last layer must be the top layer."); 
}  

/*------------------------------------------------------------*/ 
/* can be static */
void AzMultiConn::insert_connectors(AzIntArr &ia_order,  /* inout */
                                AzDataArr<AzIntArr> &aia_below, /* inout */
                                AzDataArr<AzIntArr> &aia_above) const { /* inout */
  const char *eyec = "AzMultiConn::insert_connectors"; 

  int layer_num = ia_order.size();
  /*---  count connectors to be inserted  ---*/
  int conn_num = 0; 
  for (int lx = 0; lx < layer_num; ++lx) {
    if (aia_below[lx]->size() > 1) ++conn_num; 
    if (aia_above[lx]->size() > 1) ++conn_num; 
  }

  /*---  copy the current edges  ---*/
  AzDataArr<AzIntArr> aia_b(layer_num + conn_num); 
  AzDataArr<AzIntArr> aia_a(layer_num + conn_num); 
  for (int lx = 0; lx < layer_num; ++lx) {
    aia_b(lx)->reset(aia_below[lx]); 
    aia_a(lx)->reset(aia_above[lx]); 
  }

  /*---  insert connection where multiple input/output  ---*/
  AzIntArr ia_o; 
  int cx = layer_num; 
  for (int ix = 0; ix < ia_order.size(); ++ix) {
    int lx = ia_order.get(ix); 
    if (aia_b[lx]->size() > 1) { /* multiple inputs */
      aia_b(cx)->reset(aia_b[lx]); 
      aia_a(cx)->put(lx); 
      for (int ix = 0; ix < aia_b[cx]->size(); ++ix) {
        int below = (*aia_b[cx])[ix]; 
        int count = aia_a(below)->replace(lx, cx); 
        AzX::throw_if(count != 1, eyec, "something is wrong"); 
      }
      aia_b(lx)->reset(); 
      aia_b(lx)->put(cx); 
      ia_o.put(cx);       
      ++cx; 
    }

    ia_o.put(lx); 
      
    if (aia_above[lx]->size() > 1) { /* multiple outputs */
      aia_b(cx)->put(lx); 
      aia_a(cx)->reset(aia_above[lx]); 
      for (int ix = 0; ix < aia_a[cx]->size(); ++ix) {
        int above = (*aia_a[cx])[ix]; 
        int count = aia_b(above)->replace(lx, cx); 
        AzX::throw_if(count != 1, eyec, "something is wrong-2"); 
      }
      aia_a(lx)->reset(); 
      aia_a(lx)->put(cx); 
      ia_o.put(cx); 
      ++cx; 
    }
  }
  
  /*---  output  ---*/
  aia_below.reset(&aia_b); 
  aia_above.reset(&aia_a); 
  ia_order.reset(&ia_o); 
}

/*------------------------------------------------------------*/ 
/* static */
void AzMultiConn::show_below_above(const AzIntArr &ia_below, const AzIntArr &ia_above, AzBytArr &s) {
  s << "("; 
  for (int ix = 0; ix < ia_below.size(); ++ix) {
    if (ix > 0) s << ","; 
    s << ia_below[ix]; 
  }
  s << ") -> ("; 
  for (int ix = 0; ix < ia_above.size(); ++ix) {
    if (ix > 0) s << ","; 
    s << ia_above[ix]; 
  }    
  s << ")"; 
}

/*------------------------------------------------------------*/ 
void AzMultiConn::reset_for_output_release() {
  AzIntArr ia_last_user(ia_order.max()+1, -1); 
  aia_orelease.reset(ia_order.max()+1); 
  for (int ox = 0; ox < ia_order.size(); ++ox) {
    int lx = ia_order[ox]; 
    const AzIntArr *ia_below = aia_below[lx]; 
    for (int ix = 0; ix < ia_below->size(); ++ix) ia_last_user((*ia_below)[ix], lx); 
  }  
  for (int lx = 0; lx < ia_last_user.size(); ++lx) {
    int last_user = ia_last_user[lx]; 
    if (last_user >= 0) aia_orelease(last_user)->put(lx);     
  }
}

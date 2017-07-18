/* * * * *
 *  AzPrepText.cpp 
 *  Copyright (C) 2014-2017 Rie Johnson
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

#include "AzParam.hpp"
#include "AzTools.hpp"
#include "AzTools_text.hpp"
#include "AzPrepText.hpp"
#include "AzHelp.hpp"
#include "AzTextMat.hpp"
#include "AzRandGen.hpp"

/*-------------------------------------------------------------------------*/
class AzPrepText_Param_ {
public: 
  AzPrepText_Param_() {}
  virtual void reset(int argc, const char *argv[], const AzOut &out) {
    if (argc < 2) {
      if (argc >= 1) {
        AzPrint::writeln(out, ""); 
        AzPrint::writeln(out, argv[0]); /* action */
      }    
      printHelp(out); 
      throw new AzException(AzNormal, "", ""); 
    }
    const char *action = argv[0]; 
    char param_dlm = ' '; 
    AzParam azp(param_dlm, argc-1, argv+1); 
    AzPrint::writeln(out, ""); 
    AzPrint::writeln(out, "--------------------------------------------------");     
    AzPrint::writeln(out, action, " ", azp.c_str()); 
    AzPrint::writeln(out, "--------------------------------------------------"); 
    resetParam(out, azp); 
    azp.check(out); 
  }
  virtual void resetParam(const AzOut &out, AzParam &azp) = 0; 
  virtual void printHelp(const AzOut &out) const {  /* derived classes should override this */
    AzPrint::writeln(out, "Help is not available."); 
  }  
}; 

/*-------------------------------------------------------------------------*/ 
class AzPrepText_gen_vocab_Param : public virtual AzPrepText_Param_ {
public:
  AzBytArr s_inp_fn, s_voc_fn, s_stop_fn, s_txt_ext; 
  bool do_lower, do_remove_number, do_utf8dashes, do_write_count, do_stop_if_all; 
  int min_count, nn, max_num; 
  bool do_char, do_byte; 
  
  AzPrepText_gen_vocab_Param(int argc, const char *argv[], const AzOut &out)
     : do_lower(false), do_remove_number(false), do_utf8dashes(false), min_count(-1), nn(1), 
       max_num(-1), do_write_count(false), do_char(false), do_byte(false), do_stop_if_all(false) {
    reset(argc, argv, out); 
  }
  
  /*-----------------------------------------------*/
  #define kw_inp_fn "input_fn="
  #define kw_voc_fn "vocab_fn="
  #define kw_do_lower "LowerCase"
  #define kw_min_count "min_word_count="
  #define kw_do_remove_number "RemoveNumbers"
  #define kw_txt_ext "text_fn_ext="
  #define kw_nn "n="
  #define kw_stop_fn "stopword_fn="
  #define kw_do_utf8dashes "UTF8"
  #define kw_max_num "max_vocab_size="
  #define kw_do_write_count "WriteCount"
  #define kw_do_char "Char"
  #define kw_do_byte "Byte"  
  #define kw_do_stop_if_all "StopIfAll"

  #define help_do_lower "Convert upper-case to lower-case characters."
  #define help_do_utf8dashes "Convert UTF8 en dash, em dash, single/double quotes to ascii characters."
  /*-------------------------------------------------------------------------*/
  virtual void resetParam(const AzOut &out, AzParam &azp) {
    const char *eyec = "AzPrepText_gen_vocab_Param::resetParam"; 
    AzPrint o(out); 
    azp.vStr(o, kw_inp_fn, s_inp_fn); 
    azp.vStr_prt_if_not_empty(o, kw_txt_ext, s_txt_ext);   
    azp.vStr(o, kw_voc_fn, s_voc_fn); 
    azp.swOn(o, do_lower, kw_do_lower); 
    azp.vStr_prt_if_not_empty(o, kw_stop_fn, s_stop_fn);     
    azp.vInt(o, kw_min_count, min_count); 
    azp.vInt(o, kw_max_num, max_num); 
    azp.vInt(o, kw_nn, nn); 
    azp.swOn(o, do_remove_number, kw_do_remove_number); 
    azp.swOn(o, do_utf8dashes, kw_do_utf8dashes); 
    azp.swOn(o, do_write_count, kw_do_write_count); 
    azp.swOn(o, do_char, kw_do_char); 
    if (!do_char) azp.swOn(o, do_byte, kw_do_byte); 
    azp.swOn(o, do_stop_if_all, kw_do_stop_if_all); 
  
    AzXi::throw_if_empty(s_inp_fn, eyec, kw_inp_fn); 
    AzXi::throw_if_empty(s_voc_fn, eyec, kw_voc_fn); 
    o.printEnd(); 
  }
      
  void printHelp(const AzOut &out) const {
    AzHelp h(out); 
    h.begin("", "", ""); 
    h.item_required(kw_inp_fn, "Path to the input token file or the list of token files.  If it ends with \".lst\", the file should contain the list of token filenames.  A token file should contain one document per line, and each document should be tokens delimited by space."); 
//    h.item(kw_txt_ext, "Filename extension of input token files."); 
    h.item_required(kw_voc_fn, "Path to the file that the vocabulary is written to."); 
    h.item(kw_stop_fn, "Path to a stopword file.  The words in this file will be excluded."); 
    h.item(kw_min_count, "Minimum word counts to be included in the vocabulary file.", "No limit"); 
    h.item(kw_max_num, "Maximum number of words to be included in the vocabulary file.  The most frequent ones will be included.", "No limit"); 
    h.item(kw_do_lower, help_do_lower);     
    h.item(kw_do_utf8dashes, help_do_utf8dashes); 
    h.item(kw_do_remove_number, "Exclude words that contain numbers.");     
    h.item(kw_do_write_count, "Write word counts as well as the words to the vocabulary file."); 

    h.item(kw_nn, "n for n-grams.  E.g., if n=3, only tri-grams are included.");   
    h.end(); 
  }    
};                      
                  
/*-------------------------------------------------------------------------*/
void AzPrepText::gen_vocab(int argc, const char *argv[]) const {
  AzPrepText_gen_vocab_Param p(argc, argv, out);  

  AzStrPool sp_voc[256]; 
  for (int vx = 0; vx < 256; ++vx) {
    sp_voc[vx].reset(100000,20); 
  }
  
  AzStrPool *sp_stop = NULL, _sp_stop; 
  if (p.s_stop_fn.length() > 0) {
    AzTools::readList(p.s_stop_fn.c_str(), &_sp_stop); 
    _sp_stop.commit(); 
    if (_sp_stop.size() > 0) sp_stop = &_sp_stop; 
  }  

  AzIntArr ia_data_num; 
  AzStrPool sp_list; 
  int buff_size = AzTools_text::scan_files_in_list(p.s_inp_fn.c_str(), p.s_txt_ext.c_str(), 
                                                   out, &sp_list, &ia_data_num); 
  buff_size += 256;  /* just in case */
  AzBytArr s_buff; 
  AzByte *buff = s_buff.reset(buff_size, 0);                                                    

  for (int fx = 0; fx < sp_list.size(); ++fx) {
    AzBytArr s_fn(sp_list.c_str(fx), p.s_txt_ext.c_str()); 
    const char *fn = s_fn.c_str();   
    AzTimeLog::print(fn, out); 
    AzFile file(fn); file.open("rb"); 
    for ( ; ; ) {
      int len = file.gets(buff, buff_size); 
      if (len <= 0) break;
    
      AzStrPool sp;     
      AzTools_text::tokenize(buff, len, p.do_utf8dashes, p.do_lower, sp, p.do_char, p.do_byte); 
      for (int wx = 0; wx < sp.size(); ++wx) {
        int index = gen_1byte_index(&sp, wx); 
        put_in_voc(p.nn, sp_voc[index], sp, wx, 1, p.do_stop_if_all, sp_stop, p.do_remove_number); 
      }
    }
    int num = 0; 
    for (int vx = 0; vx < 256; ++vx) {
      sp_voc[vx].commit(); 
      num += sp_voc[vx].size(); 
    }    
    AzTimeLog::print(" ... size: ", num, out); 
  }
  if (p.min_count > 1) {
    int num = 0; 
    for (int vx = 0; vx < 256; ++vx) {
      sp_voc[vx].reduce(p.min_count); 
      num += sp_voc[vx].size(); 
    }
    AzTimeLog::print("Removed <min_count -> ", num, out);     
  }
  AzTimeLog::print("Merging ... ", out); 
  for (int vx = 1; vx < 256; ++vx) {
    sp_voc[0].put(&sp_voc[vx]); 
    sp_voc[vx].reset(); 
  }
  sp_voc[0].commit(); 
  AzTimeLog::print("Writing to ", p.s_voc_fn.c_str(), out); 
  int sz = write_vocab(p.s_voc_fn.c_str(), &sp_voc[0], p.max_num, p.min_count, p.do_write_count); 
  AzTimeLog::print("Done: size=", sz, out); 
}

/*-------------------------------------------------------------------------*/
void AzPrepText::put_in_voc(int nn, 
                  AzStrPool &sp_voc, 
                  const AzStrPool &sp_words, 
                  int wx, /* position in sp_words */
                  AZint8 count, 
                  bool do_stop_if_all, 
                  const AzStrPool *sp_stop, 
                  bool do_remove_number) /* static */ {
  if (wx < 0 || wx+nn > sp_words.size()) return; 
  if (nn == 1) {
    if (sp_stop != NULL && sp_stop->find(sp_words.c_str(wx)) >= 0 ||       
        do_remove_number && strpbrk(sp_words.c_str(wx), "0123456789") != NULL) {}
    else sp_voc.put(sp_words.c_str(wx), count); 
  }
  else {
    AzBytArr s_ngram;   
    sp_words.compose_ngram(s_ngram, wx, nn, do_stop_if_all, sp_stop, do_remove_number); 
    if (s_ngram.length() > 0) {
      sp_voc.put(s_ngram.c_str(), count); 
    }
  }
} 

/*-------------------------------------------------------------------------*/
/* static */
AzByte AzPrepText::gen_1byte_index(const AzStrPool *sp_words, int wx)
{
  if (wx < 0 || wx >= sp_words->size()) return 0; 
  int len; 
  const AzByte *ptr = sp_words->point(wx, &len); 
  if (len == 0) return 0; 
  return (AzByte)((int)(*ptr) + (int)(*(ptr+len/2))); 
} 

/*-------------------------------------------------------------*/
int AzPrepText::write_vocab(const char *fn, const AzStrPool *sp, 
                            int max_num, 
                            int min_count, 
                            bool do_write_count)
{
  AzFile file(fn); file.open("wb"); 
  AzIFarr ifa_ix_count; ifa_ix_count.prepare(sp->size()); 
  for (int ix = 0; ix < sp->size(); ++ix) {
    double count = (double)sp->getCount(ix); 
    if (count >= min_count) ifa_ix_count.put(ix, count); 
  } 
  ifa_ix_count.sort_FloatInt(false, true); /* float: descending, int: ascending */
  ifa_ix_count.cut(max_num); 
  for (int ix = 0; ix < ifa_ix_count.size(); ++ix) {
    int idx; 
    ifa_ix_count.get(ix, &idx); 
    AzBytArr s(sp->c_str(idx)); 
    if (do_write_count) s << '\t' << sp->getCount(idx); 
    s.nl(); s.writeText(&file);  
  }
  file.close(true); 
  return ifa_ix_count.size(); 
}
 
/*-------------------------------------------------------------------------*/
#define xtext_ext ".xtext"
/*-------------------------------------------------------------------------*/
class AzPrepText_gen_regions_Param : public virtual AzPrepText_Param_ {
public:
  bool do_contain; 
  bool do_char, do_byte; 
  bool do_bow, do_skip_stopunk; 
  AzBytArr s_inp_fn, s_txt_ext, s_cat_ext, s_voc_fn, s_cat_dic_fn, s_rnm, s_x_ext, s_y_ext; 
  bool do_lower, do_utf8dashes; 
  int pch_sz, pch_step, padding; 
  bool do_allow_zero, do_allow_multi, do_allow_nocat;  
  bool do_ignore_bad; 
  bool do_region_only, do_write_pos; 
  AzBytArr s_batch_id; 
  AzBytArr s_inppos_fn; 
  bool do_unkw; 
  int shift_right, shift_left; /* used only with inppos_fn */
  
  AzPrepText_gen_regions_Param(int argc, const char *argv[], const AzOut &out) 
    : do_bow(false), do_skip_stopunk(false), do_lower(false), pch_sz(-1), pch_step(1), padding(0), 
      do_allow_zero(false), do_allow_multi(false), do_allow_nocat(false), do_utf8dashes(false), 
      do_region_only(false), s_x_ext(".xsmatbcvar"), s_y_ext(".y"), do_write_pos(false), do_ignore_bad(false), do_unkw(false), 
      shift_right(-1), shift_left(-1), do_char(false), do_byte(false), do_contain(false), 
      s_txt_ext(".txt.tok"), s_cat_ext(".cat") /* 07/09/2017 */ {
    reset(argc, argv, out); 
  }      
  /*-------------------------------------------------------------------------*/
  #define kw_do_contain "Contain"
  #define kw_do_region_only "RegionOnly"
  #define kw_cat_ext "label_fn_ext="
  #define kw_cat_dic_fn "label_dic_fn="
  #define kw_rnm "region_fn_stem="
  #define kw_pch_sz "patch_size="
  #define kw_pch_step "patch_stride="
  #define kw_padding "padding="
  #define kw_do_allow_zero_old "AllowZeroRegion"
  #define kw_do_allow_zero "NoSkip"
/*  #define kw_do_allow_multi "AllowMulti"  defined in AzTools_text */
  #define kw_do_bow_old "Bow-convolution"
  #define kw_do_bow "Bow"  
  #define kw_do_skip_stopunk "VariableStride"
  #define kw_y_ext "y_ext="   
  #define kw_x_ext "x_ext="
  #define kw_batch_id "batch_id="
  #define kw_inppos_fn "input_pos_fn="
  #define kw_do_write_pos "WritePositions"
  #define kw_do_ignore_bad "ExcludeMultiNo"
  #define kw_do_unkw "Unkw"
  #define kw_shift_left "shift_left="
  #define kw_shift_right "shift_right="
  /*-------------------------------------------------------------------------*/  
  virtual void resetParam(const AzOut &out, AzParam &azp) {
    const char *eyec = "AzPrepText_gen_regions_Param::resetParam";   
    AzPrint o(out); 
    azp.swOn(o, do_contain, kw_do_contain); 
    azp.swOn(o, do_char, kw_do_char); 
    if (!do_char) azp.swOn(o, do_byte, kw_do_byte); 
    azp.swOn(o, do_region_only, kw_do_region_only); 
    if (!do_region_only) {
      azp.vStr(o, kw_cat_ext, s_cat_ext);     
      azp.vStr(o, kw_cat_dic_fn, s_cat_dic_fn);     
    }
    azp.swOn(o, do_bow, kw_do_bow, kw_do_bow_old);
    azp.vStr(o, kw_inp_fn, s_inp_fn); 
    azp.vStr(o, kw_txt_ext, s_txt_ext); 
    azp.vStr(o, kw_voc_fn, s_voc_fn); 
    azp.vStr(o, kw_rnm, s_rnm); 
    azp.swOn(o, do_lower, kw_do_lower); 
    azp.swOn(o, do_utf8dashes, kw_do_utf8dashes); 
    azp.vInt(o, kw_pch_sz, pch_sz); 
    azp.swOn(o, do_allow_multi, kw_do_allow_multi); 
    do_allow_nocat = do_allow_multi; 
    azp.swOn(o, do_ignore_bad, kw_do_ignore_bad); 
    azp.swOn(o, do_write_pos, kw_do_write_pos); 
    azp.vStr(o, kw_y_ext, s_y_ext); 
    azp.vStr(o, kw_x_ext, s_x_ext); 
    azp.vStr_prt_if_not_empty(o, kw_batch_id, s_batch_id); 
    
    azp.vStr_prt_if_not_empty(o, kw_inppos_fn, s_inppos_fn); /* positions of regions for which vectors are generated */
    if (s_inppos_fn.length() <= 0) {
      azp.vInt(o, kw_pch_step, pch_step); 
      azp.vInt(o, kw_padding, padding); 
      azp.swOn(o, do_allow_zero, kw_do_allow_zero, kw_do_allow_zero_old); 
      azp.swOn(o, do_skip_stopunk, kw_do_skip_stopunk); 
    } 
    else {
      pch_step = padding = -1;  /* these won't be used.  Setting -1 to avoid displaying */ 
      azp.vInt(o, kw_shift_left, shift_left); 
      if (shift_left <= 0) azp.vInt(o, kw_shift_right, shift_right); 
    }
    azp.swOn(o, do_unkw, kw_do_unkw); 
    
    AzXi::throw_if_both(do_unkw && do_bow, eyec, kw_do_unkw, kw_do_bow); 
    
    AzXi::throw_if_empty(s_inp_fn, eyec, kw_inp_fn); 
    if (!do_region_only) {   
      AzXi::throw_if_empty(s_cat_ext, eyec, kw_cat_ext);     
      AzXi::throw_if_empty(s_cat_dic_fn, eyec, kw_cat_dic_fn);  
    }
    AzXi::throw_if_empty(s_voc_fn, eyec, kw_voc_fn);   
    AzXi::throw_if_empty(s_rnm, eyec, kw_rnm);     
    AzX::throw_if(s_rnm.contains('+'), AzInputError, eyec, kw_rnm, " must not conatin \'+\'."); 
    AzXi::throw_if_empty(s_y_ext, eyec, kw_y_ext);         
    
    AzStrPool sp_x_ext(10,10); sp_x_ext.put(".xsmatbcvar", ".xsmatcvar"); /* We need .xsmatcvar for seq2-bown */
    AzXi::check_input(s_x_ext, &sp_x_ext, eyec, kw_x_ext);       
    
    AzXi::throw_if_nonpositive(pch_sz, eyec, kw_pch_sz); 
    if (s_inppos_fn.length() <= 0) {
      AzXi::throw_if_nonpositive(pch_step, eyec, kw_pch_step); 
      AzXi::throw_if_negative(padding, eyec, kw_padding);        
    }
    o.printEnd(); 
  }

  #define help_inp_fn "Input filename without extension."
  #define help_txt_ext "Filename extension of the tokenized text file(s)."
  #define help_cat_ext "Filename extension of the label file(s).  Required for target file generation."
  #define help_cat_dic_fn "Path to the label dictionary file (input).  The file should list the labels used in the label files, one per each line.  Required for target file generation."
  #define help_voc_fn "Path to the vocabulary file generated by \"gen_vocab\" (input)." 
  #define help_y_ext "Filename extension of the target file (output).  \".y\" | \".ysmat\".  Use \".ysmat\" when the number of classes is large."
  #define help_batch_id "Batch ID, e.g., \"1of5\" (the first batch out of 5), \"2of5\" (the second batch out of 5).  Specify this when making multiple files for one dataset."
  void printHelp(const AzOut &out) const {
    AzHelp h(out); 
    h.item_required(kw_inp_fn, help_inp_fn); 
    h.item_required(kw_cat_dic_fn, help_cat_dic_fn); 
    h.item_required(kw_voc_fn, help_voc_fn); 
    h.item_required(kw_rnm, "Pathname stem of the region file, target file, and word-mapping file (output).  To make the pathnames, the respective extensions will be attached."); 
    h.item(kw_x_ext, "Filename extension of the region file (output).  \".xsmatcvar\" | \".xsmatbcvar\".", ".xsmatbcvar");     
/*    h.item(kw_y_ext, help_y_ext, ".y"); */
    h.item_required(kw_pch_sz, "Region size."); 
    h.item(kw_pch_step, "Region stride.", "1"); 
    h.item(kw_padding, "Padding size.", "0"); 
    h.item(kw_txt_ext, help_txt_ext, ".txt.tok"); 
    h.item(kw_cat_ext, help_cat_ext, ".cat");     
    h.item(kw_do_bow, "Use bag-of-word representation for sparse region vectors."); 
    h.item(kw_do_skip_stopunk, "Take variable strides.");      
    h.item(kw_do_lower, help_do_lower); 
    h.item(kw_do_utf8dashes, help_do_utf8dashes); 
    h.item(kw_do_allow_multi, "Allow multiple labels per data point for multi-label classification.  Data points without any label are also allowed.");  
    h.item(kw_do_allow_zero, "Allow zero vector to be a sparse region vector."); 
    h.item(kw_do_region_only, "Generate a region file only.  Do not generate a target file.");
      
    h.item(kw_batch_id, help_batch_id); 
    h.end(); 
  }   
}; 

/*-------------------------------------------------------------------------*/
int AzPrepText::add_unkw(AzDic &dic) const {
  AzStrPool sp_words(&dic.ref()); 
  AzBytArr s("__UNK__"); 
  for ( ; ; ) {
    if (dic.find(s.c_str()) < 0) break; 
    s << "_"; 
  }
  AzTimeLog::print("Adding ", s.c_str(), out); 
  int id = sp_words.put(&s); 
  dic.reset(&sp_words); 
  return id; 
}

/*-------------------------------------------------------------------------*/
void AzPrepText::gen_regions(int argc, const char *argv[]) const {
  const char *eyec = "AzPrepText::gen_regions"; 
 
  AzPrepText_gen_regions_Param p(argc, argv, out);   
  check_y_ext(p.s_y_ext, eyec); 
  check_batch_id(p.s_batch_id); 
  AzDic dic_word(p.s_voc_fn.c_str()); /* read the vocabulary set */
  AzX::throw_if(dic_word.size() <= 0, AzInputError, eyec, "empty dic: ", p.s_voc_fn.c_str()); 
  
  int max_nn = dic_word.get_max_n(); 
  AzIntArr ia_nn; 
  if (max_nn == 1) ia_nn.put(1); 
  else {
    AzX::no_support((max_nn == 0), eyec, "Empty vocabulary"); 
    int min_nn = dic_word.get_min_n(); 
    AzBytArr s("Vocabulary with "); 
    if (min_nn != max_nn) s << min_nn << "-"; 
    s << max_nn << " grams"; 
    AzPrint::writeln(out, s.c_str());     
    for (int ix = min_nn; ix <= max_nn; ++ix) ia_nn.put(ix); 
  }
  int unkw_id = -1; 
  AzXi::throw_if_both(p.do_unkw && max_nn != 1, eyec, kw_do_unkw, "n-grams with n>1"); 
  if (p.do_unkw) unkw_id = add_unkw(dic_word); 
  AzXi::throw_if_both(p.do_unkw && p.do_bow, eyec, kw_do_unkw, kw_do_bow); 
  AzX::no_support(max_nn>1 && !p.do_bow, eyec, "n-gram sequential"); 
  AzX::no_support(max_nn>1 && p.do_skip_stopunk, eyec, "n-gram VariableStride"); 
  
  AzDataArr<AzIntArr> aia_inppos;   
  bool do_pos = false; 
  if (p.s_inppos_fn.length() > 0) {
    AzTimeLog::print("Reading ", p.s_inppos_fn.c_str(), out); 
    AzFile::read(p.s_inppos_fn.c_str(), &aia_inppos);
    do_pos = true; 
    if (p.shift_left > 0) {
      AzPrint::writeln(out, "Shifting positions to left: ", p.shift_left); 
      for (int ix=0; ix<aia_inppos.size(); ++ix) aia_inppos(ix)->add(-p.shift_left); 
    }
    else if (p.shift_right > 0) {
      AzPrint::writeln(out, "Shifting positions to right: ", p.shift_right); 
      for (int ix=0; ix<aia_inppos.size(); ++ix) aia_inppos(ix)->add(p.shift_right); 
    }
  }

  AzDic dic_cat; 
  if (!p.do_region_only) dic_cat.reset(p.s_cat_dic_fn.c_str());  /* read categories */

  /*---  scan files to determine buffer size and #data  ---*/
  AzStrPool sp_list; 
  AzIntArr ia_data_num; 
  int buff_size = AzTools_text::scan_files_in_list(p.s_inp_fn.c_str(), p.s_txt_ext.c_str(), 
                                                   out, &sp_list, &ia_data_num); 
  int data_num = ia_data_num.sum(); 
  AzDataArr<AzIntArr> aia_outpos(data_num); 
  
  /*---  read data and generate features  ---*/
  AzSmat m_cat; 
  if (!p.do_region_only) m_cat.reform(dic_cat.size(), data_num); 

  int chkmem=100, wlen=5; /* estimamte required memory size after 100 documents; assume a word is 5 char long. */
  int ini = buff_size*chkmem/wlen*2;
  Az_bc bc(ini, ini*p.pch_sz*ia_nn.size()); 
  AzIntArr ia_dcolind(data_num*2); 
  
  AzX::throw_if((do_pos && aia_inppos.size() != data_num), AzInputError, eyec, kw_inppos_fn, "#data mismatch");  
  
  buff_size += 256; 
  AzBytArr s_buff; 
  AzByte *buff = s_buff.reset(buff_size, 0); 
  int no_cat = 0, multi_cat = 0; 
  int data_no = 0; 
  for (int fx = 0; fx < sp_list.size(); ++fx) { /* for each file */
    AzBytArr s_txt_fn(sp_list.c_str(fx), p.s_txt_ext.c_str()); 
    const char *fn = s_txt_fn.c_str(); 
    AzStrPool sp_cat;     
    if (!p.do_region_only) {
      AzBytArr s_cat_fn(sp_list.c_str(fx), p.s_cat_ext.c_str()); 
      AzTools::readList(s_cat_fn.c_str(), &sp_cat); 
    }
    AzTimeLog::print(fn, out);   
    AzFile file(fn); 
    file.open("rb"); 
    int num_in_file = 0; 
    for ( ; ; ++num_in_file) {  /* for each document */
      int len = file.gets(buff, buff_size); 
      if (len <= 0) break; 

      /*---  categories  ---*/      
      if (!p.do_region_only) {
        AzX::throw_if(num_in_file >= sp_cat.size(), AzInputError, eyec, "#data mismatch: btw text file and cat file"); 
        AzBytArr s_cat; sp_cat.get(num_in_file, &s_cat);
        AzIntArr ia_cats; AzBytArr s_err; 
        AzTools_text::parse_cats(&s_cat, '|', p.do_allow_multi, p.do_allow_nocat, 
                                 &dic_cat, &ia_cats, multi_cat, no_cat, s_err); 
        if (s_err.length() > 0) {
          if (p.do_ignore_bad) continue; 
          AzX::throw_if(true, AzInputError, eyec, s_err.c_str()); 
        }
        m_cat.col_u(data_no)->load(&ia_cats, 1);                               
      }
           
      /*---  text  ---*/
      AzIntArr *ia_opos = aia_outpos(data_no); 
      AzDataArr<AzIntArr> aia_xtokno; 
      int t_num = AzTools_text::tokenize(buff, len, &dic_word, ia_nn, p.do_lower, p.do_utf8dashes, 
                                         aia_xtokno, p.do_char, p.do_byte);  
      bc.check_overflow(eyec, t_num*p.pch_sz*ia_nn.size(), data_no);   
      ia_dcolind.put(bc.colNum()); 
      if (p.do_bow) {
        if (do_pos) gen_bow_regions_pos(t_num, aia_xtokno, ia_nn, p.do_contain, p.pch_sz, aia_inppos[data_no], 
                                        bc, ia_opos);     
        else        gen_bow_regions(t_num, aia_xtokno, ia_nn, p.do_contain, p.pch_sz, p.pch_step, p.padding, 
                                    p.do_allow_zero, p.do_skip_stopunk, bc, ia_opos); 
      }
      else {
        if (do_pos) gen_nobow_regions_pos(t_num, aia_xtokno, dic_word.size(), p.pch_sz, aia_inppos[data_no], unkw_id, 
                                          bc, ia_opos);     
        else        gen_nobow_regions(t_num, aia_xtokno, dic_word.size(), p.pch_sz, p.pch_step, p.padding, 
                                      p.do_allow_zero, unkw_id, bc, ia_opos); 
      }        
      ia_dcolind.put(bc.colNum()); 
            
      ++data_no;
      if (data_no == chkmem) {
        bc.prepmem(data_no, data_num);  
      }
    } /* for each doc */
    AzX::throw_if(!p.do_region_only && num_in_file != sp_cat.size(), AzInputError, eyec, "#data mismatch2: btw text file and cat file");  
  } /* for each file */
  if (!p.do_region_only) m_cat.resize(data_no); 
  cout << "#data=" << data_no << " no-cat=" << no_cat << " multi-cat=" << multi_cat << endl; 
  bc.commit(); 
  
  cout << bc.elmNum() << " " << bc.colNum() << endl;   
  
  /*---  write files  ---*/
  const char *outnm = p.s_rnm.c_str(); 
  int row_num = (p.do_bow) ? dic_word.size() : dic_word.size()*p.pch_sz;
  write_regions(bc, row_num, ia_dcolind, p.s_batch_id, outnm, p.s_x_ext.c_str());  
  write_dic(dic_word, row_num, outnm, xtext_ext); 

  if (!p.do_region_only) { /* labeled data */
    AzBytArr s_y_fn(outnm, p.s_y_ext.c_str()); 
    write_Y(m_cat, s_y_fn, &p.s_batch_id); 
  }
  
  if (p.do_write_pos) {
    AzBytArr s_pos_fn(outnm, ".pos"); 
    if (p.s_batch_id.length() > 0) s_pos_fn << "." << p.s_batch_id.c_str();
    AzFile::write(s_pos_fn.c_str(), &aia_outpos);   
  }
  AzTimeLog::print("Done ... ", out); 
}

/*-------------------------------------------------------------------------*/
void AzPrepText::check_size(const AzOut &out, const AzSmat &m) /* static */ {
  if (Az64::can_be_int(m.elmNum())) return; 

  AzBytArr s("# of non-zero components exceeded 2GB ("); s << (double)m.elmNum()/(double)AzSigned32Max; 
  s << " times larger).  Divide into batches."; 
  AzX::throw_if(true, AzInputError, "AzPrepText::check_size", s.c_str()); 
}

/*-------------------------------------------------------------------------*/
void AzPrepText::check_batch_id(const AzBytArr &s_batch_id) {
  const char *eyec = "AzPrepText::check_batch_id"; 
  if (s_batch_id.length() <= 0) return; 
  AzBytArr s(kw_batch_id); s << " should look like \"1of5\""; 
  const char *batch_id = s_batch_id.c_str(); 
  const char *of_str = strstr(batch_id, "of"); 
  AzX::throw_if (of_str == NULL, AzInputError, eyec, s.c_str()); 
  const char *wp; 
  for (wp = batch_id; wp < batch_id+s_batch_id.length(); ++wp) {
    if (wp >= of_str && wp < of_str+2) continue; 
    AzX::throw_if(*wp < '0' || *wp > '9', AzInputError, eyec, s.c_str());  
  }
  int batch_no = atol(batch_id); 
  int batch_num = atol(of_str+2); 
  AzX::throw_if(batch_no < 1 || batch_no > batch_num, AzInputError, eyec, s.c_str(), " batch# must start with 1 and must not exceed the number of batches. "); 
}  

/*-------------------------------------------------------------------------*/
void AzPrepText::gen_nobow_regions(int t_num, 
                       const AzDataArr<AzIntArr> &aia_nx_tok,
                       int dic_sz, int pch_sz, int pch_step, int padding,  
                       bool do_allow_zero, 
                       int unkw,  /* for unigram only */
                       /*---  output  ---*/
                       Az_bc &bc, 
                       AzIntArr *ia_pos) /* patch position: may be NULL */ const {
  int pch_num = DIVUP(t_num+padding*2-pch_sz, pch_step) + 1; 
  if (pch_num <= 0) return; 
  if (ia_pos != NULL) ia_pos->reset(); 

  int tx0 = -padding; 
  for (int pch_no = 0; pch_no < pch_num; ++pch_no) {
    int tx1 = tx0 + pch_sz; 
    AzIntArr ia; 
    for (int tx = MAX(0, tx0); tx < MIN(t_num, tx1); ++tx) {
      for (int nx = 0; nx < aia_nx_tok.size(); ++nx) {
        int tok = (*aia_nx_tok[nx])[tx]; 
        if      (tok >= 0 ) ia.put(tok +(tx-tx0)*dic_sz); 
        else if (unkw >= 0) ia.put(unkw+(tx-tx0)*dic_sz); 
      }
    }
    if (do_allow_zero || ia.size()>0) {
      bc.unique_put(ia); 
      if (ia_pos != NULL) ia_pos->put(tx0); 
    }
   
    if (tx1 >= t_num+padding) break;
    tx0 += pch_step;     
  }  
}

/*-------------------------------------------------------------------------*/
void AzPrepText::gen_nobow_regions_pos(int t_num, const AzDataArr<AzIntArr> &aia_tok, 
                       int dic_sz, int pch_sz, const AzIntArr &ia_tx0, int unkw, 
                       /*---  output  ---*/
                       Az_bc &bc, 
                       AzIntArr *ia_pos) const {
  if (ia_pos != NULL) ia_pos->reset(); 
 
  for (int col = 0; col < ia_tx0.size(); ++col) {
    int tx0 = ia_tx0[col], tx1 = tx0 + pch_sz;  
    AzIntArr ia; 
    for (int tx = MAX(0,tx0); tx < MIN(t_num,tx1); ++tx) {
      if (tx >= 0 && tx < t_num) {
        for (int nx = 0; nx < aia_tok.size(); ++nx) {
          int tok = (*aia_tok[nx])[tx]; 
          if      (tok >= 0 ) ia.put(tok +(tx-tx0)*dic_sz); 
          else if (unkw >= 0) ia.put(unkw+(tx-tx0)*dic_sz); 
        } 
      }
    }
    bc.unique_put(ia); 
    if (ia_pos != NULL) ia_pos->put(tx0);     
  }
}

/*-------------------------------------------------------------------------*/
void AzPrepText::gen_bow_regions(int t_num, const AzDataArr<AzIntArr> &aia_tokno, 
                       const AzIntArr &ia_nn, bool do_contain, 
                       int pch_sz, int pch_step, int padding,  
                       bool do_allow_zero, bool do_skip_stopunk,
                       /*---  output  ---*/
                       Az_bc &bc, 
                       AzIntArr *ia_pos) /* patch position: may be NULL */ const {
  int pch_num = DIVUP(t_num+padding*2-pch_sz, pch_step) + 1; 
  if (ia_pos != NULL) ia_pos->reset(); 

  int tx0 = -padding; 
  for (int pch_no = 0; pch_no < pch_num; ++pch_no) {
    int tx1 = tx0 + pch_sz; 
    
    AzIntArr ia; 
    for (int tx = MAX(0, tx0); tx < MIN(t_num, tx1); ++tx) {
      for (int nx = 0; nx < aia_tokno.size(); ++nx) {
        if (do_contain && tx+ia_nn[nx] > tx1) continue;   
        int tok = (*aia_tokno[nx])[tx]; 
        if (tok >= 0) ia.put(tok); 
      }
    }
    if (do_allow_zero || !do_skip_stopunk || ia.size() > 0) {
      bc.unique_put(ia); 
      if (ia_pos != NULL) ia_pos->put(tx0); 
    }
   
    if (tx1 >= t_num+padding) break;
    
    int dist = 1; 
    if (do_skip_stopunk && !do_contain) {
      /*---  to avoid repeating the same bow  ---*/
      int tx; 
      for (tx = MAX(0,tx0); tx < t_num; ++tx) {
        int nx; for (nx = 0; nx < aia_tokno.size(); ++nx) if ((*aia_tokno[nx])[tx] >= 0) break; 
        if (nx < aia_tokno.size()) break; 
      }
      int dist0 = tx-tx0+1; /* to lose a word, we have to slide a window this much */

      for (tx = MAX(0,tx1); tx < t_num; ++tx) {
        int nx; for (nx = 0; nx < aia_tokno.size(); ++nx) if ((*aia_tokno[nx])[tx] >= 0) break; 
        if (nx < aia_tokno.size()) break; 
      }
      int dist1 = tx-tx1+1; /* to get a new word, we have to slide a window this much */
      dist = MIN(dist0, dist1); 
    }
    tx0 += MAX(dist, pch_step); 
  }    
} 

/*-------------------------------------------------------------------------*/
void AzPrepText::gen_bow_regions_pos(int t_num, 
                       const AzDataArr<AzIntArr> &aia_tokno, 
                       const AzIntArr &ia_nn, bool do_contain, 
                       int pch_sz, const AzIntArr &ia_tx0,
                       /*---  output  ---*/
                       Az_bc &bc, 
                       AzIntArr *ia_pos) const {
  if (ia_pos != NULL) ia_pos->reset(); 
  for (int col = 0; col < ia_tx0.size(); ++col) {
    int tx0 = ia_tx0[col]; 
    int tx1 = tx0 + pch_sz; 
    
    AzIntArr ia_rows; 
    for (int tx = MAX(0, tx0); tx < MIN(t_num, tx1); ++tx) {
      for (int nx = 0; nx < aia_tokno.size(); ++nx) {
        if (do_contain && tx+ia_nn[nx] > tx1) continue;        
        int tok = (*aia_tokno[nx])[tx]; 
        if (tok >= 0 ) ia_rows.put(tok); 
      }
    }
    bc.unique_put(ia_rows); 
    if (ia_pos != NULL) ia_pos->put(tx0); 
  }  
} 

/*-------------------------------------------------------------------------*/
/* static */
void AzPrepText::gen_nobow_dic(const AzDic &inp_dic, int pch_sz, AzDic &out_dic) {
  for (int px = 0; px < pch_sz; ++px) {
    AzBytArr s_pref; s_pref << px << ":"; 
    AzDic tmp_dic; tmp_dic.reset(&inp_dic); 
    tmp_dic.add_prefix(s_pref.c_str()); 
    if (px == 0) out_dic.reset(&tmp_dic); 
    else         out_dic.append(&tmp_dic); 
  }
}  
    
/*-------------------------------------------------------------------------*/
/*-------------------------------------------------------------------------*/
class AzPrepText_show_regions_Param : public virtual AzPrepText_Param_ {
public:
  AzBytArr s_rnm, s_x_ext; 
  bool do_wordonly; 
  AzPrepText_show_regions_Param(int argc, const char *argv[], const AzOut &out) 
    : do_wordonly(false), s_x_ext(".xsmatbcvar") {
    reset(argc, argv, out); 
  } 
  #define kw_do_wordonly "ShowWordOnly"
  void resetParam(const AzOut &out, AzParam &azp) {
    const char *eyec = "AzPrepText_show_regions_Param::resetParam"; 
    AzPrint o(out); 
    azp.vStr(o, kw_rnm, s_rnm); 
    azp.vStr(o, kw_x_ext, s_x_ext); 
    azp.swOn(o, do_wordonly, kw_do_wordonly); 
    AzXi::throw_if_empty(s_rnm, eyec, kw_rnm); 
    o.printEnd(); 
  }
  void printHelp(const AzOut &out) const {
    AzHelp h(out); 
    h.item_required(kw_rnm, "Filename stem of the region file generated by \"gen_regions\"."); 
    h.item(kw_x_ext, "File extension of region file.", ".xsmatbcvar"); 
    h.item(kw_do_wordonly, "Show words only without values."); 
    h.end(); 
  }
}; 

/*-------------------------------------------------------------------------*/
void AzPrepText::show_regions(int argc, const char *argv[]) const {
  AzPrepText_show_regions_Param p(argc, argv, out);   
  
  AzBytArr s_x_fn(p.s_rnm.c_str(), p.s_x_ext.c_str()), s_xtext_fn(p.s_rnm.c_str(), xtext_ext); 
  AzSmatVar mv; 
  if (s_x_fn.endsWith("smatbcvar")) {
    AzSmatbcVar mvbc(s_x_fn.c_str()); 
    AzSmat m; mvbc.data()->copy_to_smat(&m); 
    mv.reset(&m, mvbc.index()); 
  }
  else mv.read(s_x_fn.c_str()); 
  AzDic dic(s_xtext_fn.c_str()); 
  _show_regions(&mv, &dic, p.do_wordonly); 
}

/*-------------------------------------------------------------------------*/
void AzPrepText::_show_regions(const AzSmatVar *mv, 
                               const AzDic *dic,
                               bool do_wordonly) const {
  const AzSmat *ms = mv->data(); 
  AzBytArr s; s << "#data=" << mv->dataNum() << " #col=" << ms->colNum(); 
  AzPrint::writeln(out, s); 
  for (int dx = 0; dx < mv->dataNum(); ++dx) {
    int col0 = mv->col_begin(dx), col1 = mv->col_end(dx); 
    s.reset("data#"); s << dx; s.nl(); 
    AzPrint::writeln(out, s); 
    int col; 
    for (col = col0; col < col1; ++col) {
      s.reset("  "); 
      AzTools_text::feat_to_text(ms->col(col), dic, do_wordonly, s); 
      AzPrint::writeln(out, s); 
    }
  }
} 
 
/*-------------------------------------------------------------------------*/
/*-------------------------------------------------------------------------*/
class AzPrepText_gen_nbw_Param : public virtual AzPrepText_Param_ {
public:
  AzBytArr s_trn_fn, s_txt_ext, s_cat_ext, s_cat_dic_fn, s_voc_fn; 
  AzBytArr s_nbw_fn; 
  double alpha; 
  bool do_lower, do_utf8dashes, do_ignore_bad; 
  
  AzPrepText_gen_nbw_Param(int argc, const char *argv[], const AzOut &out) 
    : alpha(1), do_lower(false), do_utf8dashes(false), do_ignore_bad(false), 
      s_txt_ext(".txt.tok"), s_cat_ext(".cat") /* 07/08/2016 */ {
    reset(argc, argv, out); 
  }      
  /*-------------------------------------------------------------------------*/
  #define kw_trn_fn "train_fn="
  #define kw_nbw_fn "nbw_fn="
  #define kw_alpha "alpha="
  #define kw_cat_ext "label_fn_ext="
  #define kw_cat_dic_fn "label_dic_fn="
  /*-------------------------------------------------------------------------*/  
  virtual void resetParam(const AzOut &out, AzParam &azp) {
    const char *eyec = "AzPrepText_gen_nbw_Param::resetParam"; 
    AzPrint o(out); 
    azp.vStr(o, kw_voc_fn, s_voc_fn); 
    azp.vStr(o, kw_trn_fn, s_trn_fn); 
    azp.vStr(o, kw_nbw_fn, s_nbw_fn);     
    azp.vStr(o, kw_txt_ext, s_txt_ext); 
    azp.vStr(o, kw_cat_ext, s_cat_ext);     
    azp.vStr(o, kw_cat_dic_fn, s_cat_dic_fn);     
    azp.vFloat(o, kw_alpha, alpha); 
    azp.swOn(o, do_lower, kw_do_lower); 
    azp.swOn(o, do_utf8dashes, kw_do_utf8dashes); 
    azp.swOn(o, do_ignore_bad, kw_do_ignore_bad); 
    
    AzXi::throw_if_empty(s_voc_fn, eyec, kw_voc_fn); 
    AzXi::throw_if_empty(s_trn_fn, eyec, kw_trn_fn);    
    AzXi::throw_if_empty(s_nbw_fn, eyec, kw_nbw_fn);   
    AzXi::throw_if_empty(s_txt_ext, eyec, kw_txt_ext);  
    AzXi::throw_if_empty(s_cat_ext, eyec, kw_cat_ext);  
    AzXi::throw_if_empty(s_cat_dic_fn, eyec, kw_cat_dic_fn);    
    o.printEnd(); 
  }  
  void printHelp(const AzOut &out) const {  
    AzHelp h(out); 
    h.begin("", "", ""); 
    h.item_required(kw_voc_fn, help_voc_fn); 
    h.item_required(kw_trn_fn, help_inp_fn); 
    h.item_required(kw_nbw_fn, "Path to the file which the NB weights will be written to (output).  If it ends with \"dmat\", it is written in the dense matrix formatn and can be used in \"gen_nbw\".  If not, a text file is generated for browsing the NB-weights but not for use in \"gen_nbw\"."); 
    h.item_required(kw_cat_dic_fn, help_cat_dic_fn);     
    h.item(kw_txt_ext, help_txt_ext, ".txt.tok"); 
    h.item(kw_cat_ext, help_cat_ext, ".cat"); 
    h.item(kw_alpha, "The value that should be added for smoothing", "1"); 
    h.item(kw_do_lower, help_do_lower);     
    h.item(kw_do_utf8dashes, help_do_utf8dashes);    
    h.end(); 
  }  
}; 
/*-------------------------------------------------------------------------*/
void AzPrepText::gen_nbw(int argc, const char *argv[]) const {
  const char *eyec = "AzPrepText::gen_nbw"; 

  AzPrepText_gen_nbw_Param p(argc, argv, out);   
  AzDic dic(p.s_voc_fn.c_str()); /* vocabulary */
  AzDic dic_cat(p.s_cat_dic_fn.c_str());  /* read categories */
  AzX::throw_if(dic_cat.size() < 2, AzInputError, eyec, "#class must be no smaller than 2"); 
  
  bool do_allow_multi = false, do_allow_nocat = false; 
  
  int nn = dic.get_max_n(); 
  AzPrint::writeln(out, "max n for n-gram in the vocabulary file: ", nn); 

  /*---  read training data  ---*/
  bool do_count_unk = false; 
  AzSmat m_trn_count, m_trn_cat; 
  AzTools_text::count_words_get_cats(out, p.do_ignore_bad, do_count_unk, 
        p.s_trn_fn.c_str(), p.s_txt_ext.c_str(), p.s_cat_ext.c_str(), 
        dic, dic_cat, nn, 
        do_allow_multi, do_allow_nocat, p.do_lower, p.do_utf8dashes, 
        &m_trn_count, &m_trn_cat); 

  AzDmat m_trn_sum(m_trn_count.rowNum(), m_trn_cat.rowNum()); 
  for (int col = 0; col < m_trn_count.colNum(); ++col) {
    int cat; 
    m_trn_cat.col(col)->max(&cat); 
    m_trn_sum.col_u(cat)->add(m_trn_count.col(col)); 
  }
  AzDmat m_val(dic.size(), dic_cat.size()); 
  for (int cat = 0; cat < dic_cat.size(); ++cat) {
    AzDvect v_posi(m_trn_sum.col(cat)); 
    AzDvect v_nega(m_trn_sum.rowNum()); 
    for (int nega = 0; nega < dic_cat.size(); ++nega) { /* one vs. others */
      if (nega == cat) continue; 
      v_nega.add(m_trn_sum.col(nega)); 
    }
    v_posi.add(p.alpha); v_posi.normalize1(); /* add smoothing factor and divide by sum */
    v_nega.add(p.alpha); v_nega.normalize1(); 
    double *val = m_val.col_u(cat)->point_u();  
    for (int row = 0; row < m_val.rowNum(); ++row) {
      val[row] = log(v_posi.get(row)/v_nega.get(row)); 
    }
  }  
  /*---  write nb-weights  ---*/
  if (p.s_nbw_fn.endsWith("dmat")) {
    AzTimeLog::print("Writing nbw (dmat): ", p.s_nbw_fn.c_str(), out); 
    m_val.write(p.s_nbw_fn.c_str()); 
  }
  else {
    AzTimeLog::print("Writing nbw (text): ", p.s_nbw_fn.c_str(), out); 
    int digits = 5; 
    m_val.writeText(p.s_nbw_fn.c_str(), digits); 
  }
  AzTimeLog::print("Done ... ", out); 
}

/*-------------------------------------------------------------------------*/
/*-------------------------------------------------------------------------*/
class AzPrepText_gen_nbwfeat_Param : public virtual AzPrepText_Param_ {
public:
  AzBytArr s_inp_fn, s_txt_ext, s_cat_ext, s_cat_dic_fn, s_voc_fn; 
  AzBytArr s_outnm, s_x_ext, s_y_ext, s_nbw_fn, s_batch_id; 
  bool do_lower, do_utf8dashes, do_no_cat, do_ignore_bad; 
  
  AzPrepText_gen_nbwfeat_Param(int argc, const char *argv[], const AzOut &out) 
    : do_lower(false), do_utf8dashes(false), do_no_cat(false), s_x_ext(".xsmatcvar"), s_y_ext(".y"), do_ignore_bad(false), 
      s_txt_ext(".txt.tok"), s_cat_ext(".cat") /* 07/09/2017 */ {
    reset(argc, argv, out); 
  }      
  /*-------------------------------------------------------------------------*/
  #define kw_inp_fn "input_fn="
  #define kw_outnm "output_fn_stem="
  #define kw_x_ext "x_ext="
  #define kw_do_no_cat "NoCat"
  #define help_x_ext "Extension of the feature file.  \".xsmatcvar\" (binary; for CNN) | \".x\" (text format; for NN)."
  #define help_do_no_cat "Do not make a target file."
  #define help_outnm "Pathname stem of the feature file and target file (output).  To make the pathnames of the feature file and target file, the respective extensions will be attached."
  /*-------------------------------------------------------------------------*/  
  virtual void resetParam(const AzOut &out, AzParam &azp) {
    const char *eyec = "AzPrepText_gen_nbwfeat_Param::resetParam"; 
    AzPrint o(out); 
    azp.vStr(o, kw_voc_fn, s_voc_fn); 
    azp.vStr(o, kw_inp_fn, s_inp_fn); 
    azp.vStr(o, kw_nbw_fn, s_nbw_fn);     
    azp.vStr(o, kw_txt_ext, s_txt_ext); 
    azp.vStr(o, kw_cat_ext, s_cat_ext);     
    azp.vStr(o, kw_cat_dic_fn, s_cat_dic_fn);     
    azp.vStr(o, kw_outnm, s_outnm); 
    azp.vStr(o, kw_x_ext, s_x_ext); 
    azp.vStr(o, kw_y_ext, s_y_ext);     
    azp.swOn(o, do_lower, kw_do_lower); 
    azp.swOn(o, do_utf8dashes, kw_do_utf8dashes); 
    azp.swOn(o, do_no_cat, kw_do_no_cat, false); 
    
    azp.swOn(o, do_ignore_bad, kw_do_ignore_bad); 
    azp.vStr_prt_if_not_empty(o, kw_batch_id, s_batch_id); 
    AzXi::throw_if_empty(s_voc_fn, eyec, kw_voc_fn); 
    AzXi::throw_if_empty(s_inp_fn, eyec, kw_inp_fn); 
    AzXi::throw_if_empty(s_outnm, eyec, kw_outnm); 
    AzXi::throw_if_empty(s_x_ext, eyec, kw_x_ext); 
    AzXi::throw_if_empty(s_y_ext, eyec, kw_y_ext);      
    AzXi::throw_if_empty(s_nbw_fn, eyec, kw_nbw_fn);       
    if (!do_no_cat) AzXi::throw_if_empty(s_cat_ext, eyec, kw_cat_ext);  
    AzXi::throw_if_empty(s_cat_dic_fn, eyec, kw_cat_dic_fn);    
    o.printEnd(); 
  }
  void printHelp(const AzOut &out) const {
    AzHelp h(out); 
    h.begin("", "", ""); 
    h.item_required(kw_voc_fn, help_voc_fn); 
    h.item_required(kw_inp_fn, help_inp_fn); 
    h.item_required(kw_nbw_fn, "Path to the file of the NB weights (input).  It should end with \"dmat\""); 
    h.item_required(kw_cat_dic_fn, help_cat_dic_fn); 
    h.item_required(kw_outnm, help_outnm); 
    h.item(kw_txt_ext, help_txt_ext, ".txt.tok"); 
    h.item(kw_cat_ext, help_cat_ext, ".cat");     
    h.item(kw_x_ext, help_x_ext, ".xsmatcvar"); 
    h.item(kw_y_ext, help_y_ext, ".y"); 
    h.item(kw_do_lower, help_do_lower);     
    h.item(kw_do_utf8dashes, help_do_utf8dashes);    
    h.item(kw_do_no_cat, help_do_no_cat); 
    h.item(kw_batch_id, help_batch_id); 
    h.end(); 
  }   
}; 
/*-------------------------------------------------------------------------*/
void AzPrepText::gen_nbwfeat(int argc, const char *argv[]) const {
  const char *eyec = "AzPrepText::gen_nbwfeat"; 

  AzPrepText_gen_nbwfeat_Param p(argc, argv, out);  
  check_y_ext(p.s_y_ext, eyec);  
  check_batch_id(p.s_batch_id);   
  AzDic dic(p.s_voc_fn.c_str()); /* vocabulary */
  AzDic dic_cat(p.s_cat_dic_fn.c_str());  /* read categories */
  AzX::throw_if(dic_cat.size() < 2, AzInputError, eyec, "#class must be no smaller than 2"); 
  AzX::throw_if(!p.s_nbw_fn.endsWith("dmat"), AzInputError, eyec, kw_nbw_fn, " should end with \"dmat\"");     
  
  AzTimeLog::print("Reading ", p.s_nbw_fn.c_str(), out); 
  AzDmat m_val;   
  m_val.read(p.s_nbw_fn.c_str()); 

  if (m_val.rowNum() != dic.size() || m_val.colNum() != dic_cat.size()) {
    AzBytArr s("Conflict in NB-weight dimensions.  Expected: "); 
    s << dic.size() << " x " << dic_cat.size() << "; actual: " << m_val.rowNum() << " x " << m_val.colNum(); 
    AzX::throw_if(true, AzInputError, eyec, s.c_str()); 
  }
    
  bool do_allow_multi = false, do_allow_nocat = false; 
  int nn = dic.get_max_n(); 
  AzPrint::writeln(out, "max n for n-gram in the vocabulary file: ", nn); 
  
  /*---  read data  ---*/
  bool do_count_unk = false; 
  AzSmat m_count, m_cat; 
  AzTools_text::count_words_get_cats(out, p.do_ignore_bad, do_count_unk, 
        p.s_inp_fn.c_str(), p.s_txt_ext.c_str(), p.s_cat_ext.c_str(), 
        dic, dic_cat, nn, 
        do_allow_multi, do_allow_nocat, p.do_lower, p.do_utf8dashes, 
        &m_count, &m_cat, p.do_no_cat); 

  AzTimeLog::print("Binarizing ... ", out); 
  m_count.binarize(); /* binary features */

  /*---  multiply nb-weights and write files  ---*/
  for (int cat = 0; cat < dic_cat.size(); ++cat) {
    AzTimeLog::print("Cat", cat, out); 
    AzSmat m_feat(&m_count); 
    scale(&m_feat, m_val.col(cat));  /* multiply NB-weights */
 
    /*---  generate binary labels for one vs. others training  ---*/
    AzSmat m_bcat(2, m_cat.colNum()); 
    for (int col = 0; col < m_cat.colNum(); ++col) {
      if (m_cat.get(cat, col) == 1) m_bcat.set((int)0, (int)col, (double)1); /* positive */
      else                          m_bcat.set((int)1, (int)col, (double)1); /* negative */
    }
    
    /*---  write files  ---*/
    AzBytArr s_x_fn(p.s_outnm.c_str()), s_y_fn(p.s_outnm.c_str()); 
    if (dic_cat.size() != 2) {
      s_x_fn << ".cat" << cat; s_y_fn << ".cat" << cat; 
    }
    s_x_fn << p.s_x_ext.c_str(); 
    write_X(m_feat, s_x_fn, &p.s_batch_id); 

    s_y_fn << p.s_y_ext.c_str();
    write_Y(m_bcat, s_y_fn, &p.s_batch_id); 

    AzBytArr s_xtext_fn(p.s_outnm.c_str(), xtext_ext);     
    dic.writeText(s_xtext_fn.c_str(), false);     
    if (dic_cat.size() == 2) break; 
  }
  if (dic_cat.size() > 2) {
    AzBytArr s_y_fn(p.s_outnm.c_str(), p.s_y_ext.c_str()); 
    write_Y(m_cat, s_y_fn);       
  }
}

/*-------------------------------------------------------------------------*/ 
void AzPrepText::scale(AzSmat *ms, const AzDvect *v) 
{
  for (int col = 0; col < ms->colNum(); ++col) {
    AzIFarr ifa; 
    ms->col(col)->nonZero(&ifa); 
    AzIFarr ifa_new; ifa_new.prepare(ifa.size()); 
    for (int ix = 0; ix < ifa.size(); ++ix) {
      double val = ifa.get(ix); 
      int row = ifa.getInt(ix); 
      ifa_new.put(row, val*v->get(row)); 
    }
    ms->col_u(col)->load(&ifa_new); 
  }
}

/*-------------------------------------------------------------------------*/
/*-------------------------------------------------------------------------*/
class AzPrepText_gen_b_feat_Param : public virtual AzPrepText_Param_ {
public:
  AzBytArr s_inp_fn, s_txt_ext, s_cat_ext, s_cat_dic_fn, s_voc_fn; 
  AzBytArr s_outnm, s_x_ext, s_y_ext, s_batch_id; 
  bool do_log, do_bin, do_unit; 
  bool do_lower, do_utf8dashes, do_no_cat, do_ignore_bad; 
  
  AzPrepText_gen_b_feat_Param(int argc, const char *argv[], const AzOut &out) 
    : do_lower(false), do_utf8dashes(false), s_x_ext(".x"), s_y_ext(".y"), 
      do_no_cat(false), do_log(false), do_bin(false), do_unit(false), do_ignore_bad(false) {
    reset(argc, argv, out); 
  }      
  /*-------------------------------------------------------------------------*/   
  #define kw_outnm "output_fn_stem="
  #define kw_x_ext "x_ext="
  #define kw_do_no_cat "NoCat"  
  #define kw_do_log "LogCount"
  #define kw_do_bin "Binary"
  #define kw_do_unit "Unit"
  /*-------------------------------------------------------------------------*/  
  virtual void resetParam(const AzOut &out, AzParam &azp) {
    const char *eyec = "AzPrepText_b_gen_b_feat_Param::resetParam"; 
    AzPrint o(out); 
    azp.vStr(o, kw_voc_fn, s_voc_fn); 
    azp.vStr(o, kw_inp_fn, s_inp_fn); 
    azp.vStr(o, kw_txt_ext, s_txt_ext); 
    azp.vStr(o, kw_cat_ext, s_cat_ext);     
    azp.vStr(o, kw_cat_dic_fn, s_cat_dic_fn);     
    azp.vStr(o, kw_outnm, s_outnm); 
    azp.vStr(o, kw_x_ext, s_x_ext); 
    azp.vStr(o, kw_y_ext, s_y_ext);     
    azp.swOn(o, do_lower, kw_do_lower); 
    azp.swOn(o, do_utf8dashes, kw_do_utf8dashes); 
    azp.swOn(o, do_no_cat, kw_do_no_cat, false); 
    azp.swOn(o, do_log, kw_do_log); 
    azp.swOn(o, do_bin, kw_do_bin); 
    azp.swOn(o, do_unit, kw_do_unit); 
    azp.swOn(o, do_ignore_bad, kw_do_ignore_bad); 
    azp.vStr_prt_if_not_empty(o, kw_batch_id, s_batch_id); 

    AzXi::throw_if_empty(s_voc_fn, eyec, kw_voc_fn); 
    AzXi::throw_if_empty(s_inp_fn, eyec, kw_inp_fn); 
    AzXi::throw_if_empty(s_outnm, eyec, kw_outnm);  
    AzXi::throw_if_empty(s_x_ext, eyec, kw_x_ext);  
    AzXi::throw_if_empty(s_y_ext, eyec, kw_y_ext);         
    if (!do_no_cat) {
      AzXi::throw_if_empty(s_cat_ext, eyec, kw_cat_ext);  
      AzXi::throw_if_empty(s_cat_dic_fn, eyec, kw_cat_dic_fn);    
    }
    AzXi::throw_if_both(do_log && do_bin, eyec, kw_do_log, kw_do_bin);  
    o.printEnd(); 
  }
  
  void printHelp(const AzOut &out) const {
    AzHelp h(out); 
    h.begin("", "", ""); 
    h.item_required(kw_voc_fn, help_voc_fn); 
    h.item_required(kw_inp_fn, help_inp_fn); 
    h.item(kw_txt_ext, help_txt_ext); 
    h.item_required(kw_cat_ext, help_cat_ext); 
    h.item_required(kw_cat_dic_fn, help_cat_dic_fn); 
    h.item_required(kw_outnm, help_outnm); 
    h.item(kw_x_ext, help_x_ext, ".x"); 
    h.item(kw_y_ext, help_y_ext, ".y"); 
    h.item(kw_do_lower, help_do_lower);     
    h.item(kw_do_utf8dashes, help_do_utf8dashes);    
    h.item(kw_do_no_cat, help_do_no_cat); 
    h.item(kw_do_log, "Set components to log(x+1) where x is the frequency of the word in each document."); 
    h.item(kw_do_bin, "Set components to 1 if the word appears in the document; 0 otherwise."); 
    h.item(kw_do_unit, "Scale the feature vectors to unit vectors."); 
    h.item(kw_batch_id, help_batch_id); 
    h.end(); 
  }   
}; 
/*-------------------------------------------------------------------------*/
void AzPrepText::gen_b_feat(int argc, const char *argv[]) const {
  const char *eyec = "AzPrepText::gen_b_feat"; 

  AzPrepText_gen_b_feat_Param p(argc, argv, out);   
  check_y_ext(p.s_y_ext, eyec);  
  check_batch_id(p.s_batch_id); 
  AzDic dic(p.s_voc_fn.c_str()); /* vocabulary */
  AzDic dic_cat; 
  if (!p.do_no_cat) dic_cat.reset(p.s_cat_dic_fn.c_str()); /* read categories */
  
  bool do_allow_multi = false, do_allow_nocat = false; 
  int nn=dic.get_max_n(); 
  AzPrint::writeln(out, "max n for n-gram in the vocabulary file: ", nn); 
  
  /*---  read data  ---*/
  bool do_unk = false; 
  AzSmat m_feat, m_cat; 
  AzTools_text::count_words_get_cats(out, p.do_ignore_bad, do_unk, 
        p.s_inp_fn.c_str(), p.s_txt_ext.c_str(), p.s_cat_ext.c_str(), 
        dic, dic_cat, nn, 
        do_allow_multi, do_allow_nocat, p.do_lower, p.do_utf8dashes, 
        &m_feat, &m_cat, p.do_no_cat); 

  /*---  convert features if requested  ---*/
  if (p.do_bin) {
    AzTimeLog::print("Binarizing ... ", out); 
    m_feat.binarize(); /* binary features */
  }
  else if (p.do_log) {
    AzTimeLog::print("log(x+1) ... ", out); 
    m_feat.plus_one_log(); 
  }
  if (p.do_unit) {
    AzTimeLog::print("Converting to unit vectors ... ", out); 
    m_feat.normalize();  
  }

  /*---  write files  ---*/
  AzBytArr s_x_fn(p.s_outnm.c_str(), p.s_x_ext.c_str()); 
  write_X(m_feat, s_x_fn, &p.s_batch_id); 

  AzBytArr s_xtext_fn(p.s_outnm.c_str(), xtext_ext);   
  dic.writeText(s_xtext_fn.c_str(), false);     

  if (!p.do_no_cat) {
    AzBytArr s_y_fn(p.s_outnm.c_str(), p.s_y_ext.c_str());    
    write_Y(m_cat, s_y_fn, &p.s_batch_id); 
  }
}
 
/*-------------------------------------------------------------------------*/
void AzPrepText::write_X(const AzOut &out, const AzSmat &m_x, const AzBytArr &s_x_fn, 
                         const AzBytArr *s_batch_id) /* static */ /* may be NULL */ {
  check_size(out, m_x);                            
  AzBytArr s_fn(&s_x_fn); 
  if (s_batch_id != NULL && s_batch_id->length() > 0) s_fn << "." << s_batch_id->c_str(); 
  const char *fn = s_fn.c_str(); 
  AzBytArr s(": "); AzTools::show_smat_stat(m_x, s);  
  if (s_x_fn.endsWith("smatcvar")) {
    AzSmatVar msv; msv.reset(&m_x); /* NOTE: AzSmatVar and AzSmatcVar share the file format */
    AzTimeLog::print(fn, s.c_str(), " (smatcvar)", out);  
    msv.write(fn);    
  }
  else if (s_x_fn.endsWith("smatc")) {
    AzTimeLog::print(fn, s.c_str(), " (smatc)", out);  
    m_x.write(fn); /* NOTE: AzSmat and AzSmatc share the file format */
  }
  else if (s_x_fn.endsWith("smatbc")) {
    AzTimeLog::print(fn, s.c_str(), " (smatbc)", out);      
    AzSmatbc mbc; mbc.set(&m_x); 
    mbc.write(fn); 
  }
  else if (s_x_fn.endsWith(".x")) {
    int digits = 5; 
    bool do_sparse = true; 
    AzTimeLog::print(fn, s.c_str(), " (text)", out);  
    m_x.writeText(fn, digits, do_sparse); 
  }
  else {
    AzX::throw_if(true, AzInputError, "AzPrepText::write_X", "Unknown file type: ", s_x_fn.c_str());  
  }
}  

/*-------------------------------------------------------------------------*/
void AzPrepText::write_Y(const AzOut &out, const AzSmat &m_y, 
                         const AzBytArr &s_y_fn, 
                         const AzBytArr *s_batch_id) /* may be NULL */ /* static */ {
  check_size(out, m_y); 
  AzBytArr s(": "); AzTools::show_smat_stat(m_y, s);  
  AzBytArr s_fn(&s_y_fn); 
  if (s_batch_id != NULL && s_batch_id->length() > 0) s_fn << "." << s_batch_id->c_str(); 
  const char *fn = s_fn.c_str();     
  if (s_y_fn.endsWith("smatc")) {
    AzTimeLog::print(fn, s.c_str(), " (smatc)", out);
    m_y.write(fn); 
  }
  else if (s_y_fn.endsWith("smatbc")) {
    AzTimeLog::print(fn, s.c_str(), " (smatbc)", out);
    AzSmatbc mbc; mbc.set(&m_y); 
    mbc.write(fn); 
  }
  else if (s_y_fn.endsWith(".y")) {
    int digits=5; 
    bool do_sparse = true;   
    AzTimeLog::print(fn, s.c_str(), " (text)", out);
    m_y.writeText(fn, digits, do_sparse); 
  }
  else {
    AzX::throw_if(true, AzInputError, "AzPrepText::write_Y", "Unknown file type: ", s_y_fn.c_str()); 
  }
}   

/*-----------------------------------------------------------------*/
/*-----------------------------------------------------------------*/
#define kw_ext "ext="
#define kw_num "num_batches="
#define kw_split "split="
#define kw_seed "random_seed="
#define kw_id_fn "id_fn_stem="
/*-----------------------------------------------------------------*/
class AzPrepText_split_text_Param : public virtual AzPrepText_Param_ {
public: 
  AzBytArr s_inp_fn, s_ext, s_outnm, s_split, s_id_fn; 
  AzIntArr ia_ratio; 
  int num, seed; 
  AzPrepText_split_text_Param(int argc, const char *argv[], const AzOut &out) : num(0), seed(1) {
    reset(argc, argv, out); 
  }
  void resetParam(const AzOut &out, AzParam &azp) {
    const char *eyec = "AzPrepText_split_Param::resetParam"; 
    AzPrint o(out); 
    azp.vStr(o, kw_inp_fn, s_inp_fn);
    azp.vStr_prt_if_not_empty(o, kw_ext, s_ext);  
    azp.vStr(o, kw_outnm, s_outnm);     
    azp.vStr_prt_if_not_empty(o, kw_split, s_split); 
    if (s_split.length() <= 0) azp.vInt(o, kw_num, num); 
    else {
      AzTools::getInts(s_split.c_str(), ':', &ia_ratio); 
      AzX::throw_if((ia_ratio.min() <= 0), eyec, kw_split, "must not include a non-positive value."); 
      num = ia_ratio.size(); 
    }
    azp.vInt(o, kw_seed, seed); 
    azp.vStr_prt_if_not_empty(o, kw_id_fn, s_id_fn); 
    AzXi::throw_if_nonpositive(num, eyec, kw_seed); 
    AzXi::throw_if_empty(s_inp_fn, eyec, kw_inp_fn);     
    AzXi::throw_if_empty(s_outnm, eyec, kw_outnm);      
    o.printEnd(); 
  }
  void printHelp(const AzOut &out) const {
    AzHelp h(out); 
    h.begin("", "", ""); 
    h.item_required(kw_inp_fn, "Input filename or the list of input filenames  If it ends with \".lst\" it should contain the list of input filenames.");  
    h.item(kw_ext, "Filename extension.  Optional."); 
    h.item_required(kw_outnm, help_outnm); 
    h.item_required(kw_num, "Number of partitions.");     
    h.item(kw_split, "How to split the data.  Use \":\" as a delimiter.  For example, \"4:1\" will split the data into four fifth and one fifth.  \"1:1:1\" will split the data into three partitions of the same size.    Required if \"num_batches=\" is not specified.");    
    h.item(kw_seed, "Seed for random number generation.", "1"); 
    h.item(kw_id_fn, "Used to make pathnames by attaching, e.g., \".1of5\" to write the data indexes of the resulting text files.  Optional."); 
    h.end(); 
  }     
}; 

/*-----------------------------------------------------------------*/
void AzPrepText::split_text(int argc, const char *argv[]) const {
  const char *eyec = "AzPrepText::split_text"; 
  AzPrepText_split_text_Param p(argc, argv, out);   

  srand(p.seed); 
  
  AzIntArr ia_data_num; 
  AzStrPool sp_list; 
  int buff_size = AzTools_text::scan_files_in_list(p.s_inp_fn.c_str(), p.s_ext.c_str(), 
                                                   out, &sp_list, &ia_data_num); 
  buff_size += 256;  /* just in case */
  AzBytArr s_buff; 
  AzByte *buff = s_buff.reset(buff_size, 0);                                                    

  int data_num = ia_data_num.sum(); 

  AzTimeLog::print("Assigning batch id: ", data_num, out); 
  AzIntArr ia_dx2group;  
  if (p.ia_ratio.size() <= 0) {
    /*---  divide into partitions of the same size approximately  ---*/
    ia_dx2group.reset(data_num, -1); 
    for (int dx = 0; dx < data_num; ++dx) {
      int gx = AzTools::rand_large() % p.num; 
      ia_dx2group(dx, gx);  
    }
  }
  else {
    /*---  use the specified ratio  ---*/
    double sum = p.ia_ratio.sum(); 
    AzIntArr ia_num; 
    for (int ix = 0; ix < p.ia_ratio.size(); ++ix) ia_num.put((int)floor((double)data_num * (double)p.ia_ratio[ix]/sum)); 
    int extra = data_num - ia_num.sum(); 
    for (int ix = 0; extra > 0; ix = (ix+1)%ia_num.size()) {
      ia_num.increment(ix); --extra; 
    }
    AzX::throw_if((ia_num.sum() != data_num), eyec, "Partition sizes do not add up ..."); 
    ia_num.print(log_out, "Sizes: "); 
    
    ia_dx2group.reset(); 
    for (int gx = 0; gx < ia_num.size(); ++gx) {
      for (int ix = 0; ix < ia_num[gx]; ++ix) ia_dx2group.put(gx); 
    }
    AzTools::shuffle2(ia_dx2group); 
  }

  AzTimeLog::print("Reading and writing ... ", out);   
  AzDataArr<AzFile> afile(p.num); 
  for (int gx = 0; gx < p.num; ++gx) {
    AzBytArr s_fn(&p.s_outnm); 
    s_fn << "." << gx+1 << "of" << p.num;     
    if (p.s_ext.length() > 0) s_fn << p.s_ext.c_str();     
    afile(gx)->reset(s_fn.c_str()); 
    afile(gx)->open("wb"); 
  }
  
  int dx = 0; 
  for (int fx = 0; fx < sp_list.size(); ++fx) {
    AzBytArr s_fn(sp_list.c_str(fx), p.s_ext.c_str()); 
    const char *fn = s_fn.c_str();   
    AzTimeLog::print(fn, out); 
    AzFile file(fn); file.open("rb"); 
    for ( ; ; ) {
      int len = file.gets(buff, buff_size); 
      if (len <= 0) break;  
  
      int gx = ia_dx2group[dx];     
      afile(gx)->writeBytes(buff, len); 
      ++dx; 
    }
  }
  for (int gx = 0; gx < p.num; ++gx) {
    afile(gx)->close(true);  
  }
  
  /*---  write data indexes to the id files  ---*/
  if (p.s_id_fn.length() > 0) {
    for (int gx = 0; gx < p.num; ++gx) {
      AzBytArr s_fn(p.s_id_fn.c_str()); s_fn << "." << gx+1 << "of" << p.num;  
      AzTimeLog::print("Writing data indexes to: ", s_fn.c_str(), out); 
      AzFile file(s_fn.c_str()); file.open("wb"); 
      for (int dx = 0; dx < data_num; ++dx) {
        if (ia_dx2group[dx] == gx) {
          AzBytArr s; s << dx; s.nl(); 
          s.writeText(&file); 
        }        
      }
      file.close(true); 
    }    
  }

  AzTimeLog::print("Done ... ", log_out); 
} 

/*-----------------------------------------------------------------*/
/* static */
void AzPrepText::check_y_ext(const AzBytArr &s_y_ext, const char *eyec) {
  AzStrPool sp_y_ext(10,10); sp_y_ext.put(".y", ".ysmat", ".ysmatbc"); 
  AzXi::check_input(s_y_ext, &sp_y_ext, eyec, kw_y_ext);   
}    

/*-------------------------------------------------------------------------*/ 
/*-------------------------------------------------------------------------*/ 
class AzPrepText_merge_vocab_Param : public virtual AzPrepText_Param_ {
public:
  AzBytArr s_inp_fns, s_voc_fn; 
  int min_count, max_num; 
  bool do_write_count; 
  bool do_join; 
  bool do_1stfirst; /* make the same output as "merge_sort_dic" */
                    /* so that the order of the input files matter */
  AzPrepText_merge_vocab_Param(int argc, const char *argv[], const AzOut &out)
       : min_count(-1), max_num(-1), do_write_count(false), do_join(false), do_1stfirst(false) {
      reset(argc, argv, out); 
  }

  /*-----------------------------------------------*/
  #define kw_inp_fns "input_fns="
  #define kw_do_join "Join"
  #define kw_do_1stfirst "UseInputFileOrder"
  
  /*-------------------------------------------------------------------------*/
  virtual void resetParam(const AzOut &out, AzParam &azp) {
    const char *eyec = "AzPrepText_merge_vocab_Param::resetParam"; 
    AzPrint o(out); 
    azp.vStr(o, kw_inp_fns, s_inp_fns);  
    azp.vStr(o, kw_voc_fn, s_voc_fn);   
    azp.vInt(o, kw_min_count, min_count); 
    azp.vInt(o, kw_max_num, max_num); 
    azp.swOn(o, do_write_count, kw_do_write_count); 
    azp.swOn(o, do_join, kw_do_join); 
    AzXi::throw_if_empty(s_inp_fns, eyec, kw_inp_fns); 
    AzXi::throw_if_empty(s_voc_fn, eyec, kw_voc_fn); 
    if (!do_join) azp.swOn(&do_1stfirst, kw_do_1stfirst); 
    o.printEnd(); 
  }            
  void printHelp(const AzOut &out) const {
    AzHelp h(out); 
    h.begin("", "", ""); 
    h.item_required(kw_inp_fns, "Input vocabulary filenames to be merged delimited by \"+\".");  
    h.item_required(kw_voc_fn, "Pathname of the output vocabulary file."); 
    h.item(kw_min_count, "Minimum word counts to be included in the vocabulary file.", "No limit"); 
    h.item(kw_max_num, "Maximum number of words to be included in the vocabulary file.  The most frequent ones will be included.", "No limit"); 
    h.item(kw_do_write_count, "Write word counts as well as the words to the vocabulary file."); 
    h.item(kw_do_join, "Take the intersection of the input vocabulary files.  Default: Union");     
    h.end(); 
  }   
};                      
                  
/*-------------------------------------------------------------------------*/
void AzPrepText::merge_vocab(int argc, const char *argv[]) const {
  AzPrepText_merge_vocab_Param p(argc, argv, out);  

  AzStrPool sp_fns(100,100); 
  AzTools::getStrings(p.s_inp_fns.c_str(), '+', &sp_fns); 
  AzStrPool sp; 
  if (p.do_join) join_vocab(sp_fns, sp); 
  else           union_vocab(sp_fns, sp, p.do_1stfirst); 
  AzTimeLog::print("Merged ... ", sp.size(), out); 
  write_vocab(p.s_voc_fn.c_str(), &sp, p.max_num, p.min_count, p.do_write_count); 
}

/*-------------------------------------------------------------------------*/
void AzPrepText::union_vocab(const AzStrPool &sp_fns, AzStrPool &out_sp, bool do_1stfirst) const {
  AzTimeLog::print("Union ... ", out); 
  out_sp.reset(); 
  for (int ix = 0; ix < sp_fns.size(); ++ix) {
    AzDic dic(sp_fns.c_str(ix)); 
    AzBytArr s(sp_fns.c_str(ix)); s << ": size=" << dic.size(); 
    AzTimeLog::print(s.c_str(), out);     
    out_sp.put(&dic.ref()); 
  }
  if (!do_1stfirst) out_sp.commit();  
  else {
    /*---  don't commit, and just make sure there is no duplicate  ---*/
    AzStrPool sp(&out_sp); 
    sp.commit(); 
    if (sp.size() != out_sp.size()) {
      for (int ix = 0; ix < sp.size(); ++ix) if (sp.getCount(ix) > 1) AzPrint::writeln(out, "Duplicated entry: ", sp.c_str(ix)); 
      AzX::throw_if(true, AzInputError, "AzPrepText::union_vocab", 
                    "Duplicated entries are not allowed."); 
    }
  }
}  

/*-------------------------------------------------------------------------*/
void AzPrepText::join_vocab(const AzStrPool &sp_fns, AzStrPool &out_sp) const {
  AzTimeLog::print("Join ... ", out); 
  AzDataArr<AzDic> adic(sp_fns.size()); 
  int min_pop = -1, min_ix = -1; 
  for (int ix = 0; ix < sp_fns.size(); ++ix) {
    adic(ix)->reset(sp_fns.c_str(ix)); 
    AzBytArr s(sp_fns.c_str(ix)); s << ": size=" << adic[ix]->size();     
    AzTimeLog::print(s.c_str(), out);     
    if (ix == 0 || adic[ix]->size() < min_pop) {
      min_pop = adic[ix]->size(); 
      min_ix = ix;       
    }
  } 
  const AzDic *min_dic = adic[min_ix]; 
  AzIntArr ia_ok; ia_ok.reset(min_dic->size(), 1); 
  AzDvect v_count(min_dic->size());   
  for (int ix = 0; ix < sp_fns.size(); ++ix) {
    const AzDic *dic = adic[ix]; 
    for (int jx = 0; jx < min_dic->size(); ++jx) {
      if (ia_ok[jx] != 1) continue; 
      int idx = (ix == min_ix) ? jx : dic->find(min_dic->c_str(jx)); 
      if (idx < 0) ia_ok(jx, 0); 
      else         v_count.add(jx, (double)dic->count(idx)); 
    }
  }  
  out_sp.reset(); 
  for (int ix = 0; ix < ia_ok.size(); ++ix) {
    if (ia_ok[ix] != 1) continue; 
    out_sp.put(min_dic->c_str(ix), (AZint8)v_count.get(ix)); 
  }  
}

/*-------------------------------------------------------------------------*/
/*-------------------------------------------------------------------------*/
class AzPrepText_gen_regions_unsup_Param : public virtual AzPrepText_Param_ {
public: 
  AzBytArr s_xtyp, s_xdic_fn, s_ydic_fn, s_inp_fn, s_txt_ext, s_rnm; 
  AzBytArr s_batch_id, s_x_ext, s_y_ext; 
  int dist, min_x, min_y; 
  int gap; 
  int pch_sz, pch_step, padding; 
  bool do_lower, do_utf8dashes; 
  bool do_nolr; /* only when bow */
  bool do_no_skip; 
  bool do_rightonly, do_leftonly; 
  
  #define kw_bow "Bow"
  #define kw_seq "Seq"   
  AzPrepText_gen_regions_unsup_Param(int argc, const char *argv[], const AzOut &out) 
    : dist(-1), min_x(1), min_y(1), pch_sz(-1), pch_step(1), padding(-1), gap(0), 
      s_x_ext(".xsmatbc"), s_y_ext(".ysmatbc"), s_xtyp(kw_bow), do_rightonly(false), do_leftonly(false), 
      do_lower(false), do_utf8dashes(false), do_nolr(false), do_no_skip(false) {
    reset(argc, argv, out); 
  }

  /*-------------------------------------------------------------------------*/  
  #define kw_xtyp "x_type="
  #define kw_xdic_fn "x_vocab_fn="    
  #define kw_ydic_fn "y_vocab_fn="
  #define kw_txt_ext "text_fn_ext="  
  #define kw_min_x "min_x="
  #define kw_min_y "min_y="
  #define kw_dist "dist="
  #define kw_gap "gap="
  #define kw_do_nolr "MergeLeftRight"
  #define kw_x_ext_old "x_fn_ext="    
  #define kw_y_ext_old "y_fn_ext=" 
  #define kw_do_no_skip "NoSkip"
  #define kw_do_leftonly "LeftOnly"
  #define kw_do_rightonly "RightOnly"

  #define ytext_ext ".ytext"
 
  #define help_rnm "Pathname stem of the region file, target file, and word-mapping file (output).  To make the pathnames, the respective extensions will be attached."
  #define help_do_nolr "Do not distinguish the target regions on the left and right."
  #define help_xtyp "Vector representation for X (sparse region vectors).  Bow | Seq"
  /*-------------------------------------------------------------------------*/
  void resetParam(const AzOut &out, AzParam &azp) {
    const char *eyec = "AzPrepText_gen_regions_unsup_Param::resetParam"; 
    AzPrint o(out); 
    azp.vStr(o, kw_xtyp, s_xtyp); 
    azp.vStr(o, kw_xdic_fn, s_xdic_fn);    
    azp.vStr(o, kw_ydic_fn, s_ydic_fn);       
    azp.vStr(o, kw_inp_fn, s_inp_fn);      
    azp.vStr(o, kw_rnm, s_rnm);  
    azp.vStr_prt_if_not_empty(o, kw_txt_ext, s_txt_ext);  
    azp.vInt(o, kw_pch_sz, pch_sz);      
    azp.vInt(o, kw_pch_step, pch_step);   
    azp.vInt(o, kw_padding, padding);   
    azp.vInt(o, kw_dist, dist);   
    azp.vInt(o, kw_gap, gap); 
    azp.swOn(o, do_lower, kw_do_lower); 
    azp.swOn(o, do_utf8dashes, kw_do_utf8dashes);     
    azp.swOn(o, do_nolr, kw_do_nolr); 
    azp.vStr_prt_if_not_empty(o, kw_batch_id, s_batch_id); 
    azp.vStr(o, kw_x_ext, s_x_ext, kw_x_ext_old); 
    azp.vStr(o, kw_y_ext, s_y_ext, kw_y_ext_old); 
    azp.swOn(o, do_no_skip, kw_do_no_skip, false); 
    if (do_no_skip) {
      min_x = min_y = 0;       
    }    
    azp.swOn(o, do_leftonly, kw_do_leftonly); 
    if (!do_leftonly) azp.swOn(o, do_rightonly, kw_do_rightonly); 
    if (do_leftonly || do_rightonly) do_nolr = true;

    AzX::throw_if(min_x>1 || min_y>1, AzInputError, eyec, "min_x and min_y must be no greater than 1.");     
    AzStrPool sp_typ(10,10); sp_typ.put(kw_bow, kw_seq); 
    AzXi::check_input(s_xtyp, &sp_typ, eyec, kw_xtyp);     
    AzXi::throw_if_empty(s_xdic_fn, eyec, kw_xdic_fn);  
    AzXi::throw_if_empty(s_ydic_fn, eyec, kw_ydic_fn);     
    AzXi::throw_if_empty(s_inp_fn, eyec, kw_inp_fn);      
    AzXi::throw_if_empty(s_rnm, eyec, kw_rnm);      
    AzXi::throw_if_nonpositive(pch_sz, eyec, kw_pch_sz); 
    AzXi::throw_if_nonpositive(pch_step, eyec, kw_pch_step); 
    AzXi::throw_if_negative(padding, eyec, kw_padding); 
    AzXi::throw_if_nonpositive(dist, eyec, kw_dist); 
    AzXi::throw_if_negative(gap, eyec, kw_gap); 
    AzStrPool sp_x_ext(10,10); sp_x_ext.put(".xsmatbc", ".xsmatbcvar"); /* var for LSTM */
    AzXi::check_input(s_x_ext, &sp_x_ext, eyec, kw_x_ext);     
    AzStrPool sp_y_ext(10,10); sp_y_ext.put(".ysmatbc", ".ysmatbcvar"); /* var for LSTM */
    AzXi::check_input(s_y_ext, &sp_y_ext, eyec, kw_y_ext);           
    o.printEnd(); 
  }
  void printHelp(const AzOut &out) const {
    AzHelp h(out); h.begin("", "", "");  h.nl(); 
    h.writeln("To generate from unlabeled data a region file and target file for tv-embedding learning with unsupervised target.  The task is to predict adjacent regions based on the current region.\n", 3);
    h.item_required(kw_xtyp, help_xtyp, kw_bow); 
    h.item_required(kw_inp_fn, "Path to the input token file or the list of token files.  If the filename ends with \".lst\", the file should be the list of token filenames.  The input token file(s) should contain one document per line, and each document should be tokens delimited by space."); 
    h.item_required(kw_rnm, help_rnm); 
    h.item_required(kw_xdic_fn, "Path to the vocabulary file generated by \"gen_vocab\", used for X (features)."); 
    h.item_required(kw_ydic_fn, "Path to the vocabulary file generated by \"gen_vocab\", used for Y (target).");    
    h.item_required(kw_pch_sz, "Region size."); 
    h.item_required(kw_dist, "Size of adjacent regions used to produce Y (target).");        
    h.item(kw_pch_step, "Region stride.", "1");     
    h.item(kw_padding, "Padding size.", "0");      
    h.item(kw_gap, "Gap between the X region (features) and Y region (target).", "0"); 
    h.item(kw_do_lower, help_do_lower);     
    h.item(kw_do_utf8dashes, help_do_utf8dashes); 
    h.item(kw_do_nolr, help_do_nolr); 
    h.item(kw_do_rightonly, "Use this for training a forward LSTM (left to right) so that the region to the right (future) of the current time step is regarded as a target region."); 
    h.item(kw_do_leftonly, "Use this for training a backward LSTM (right to left) so that the region to the left (past) of the current time step is regarded as a target region."); 
    /* txt_ext, x_ext, y_ext, do_no_skip, min_x, min_y */
    h.end(); 
  }   
}; 

/*-------------------------------------------------------------------------*/
/* Note: X and Y use different dictionaries */
void AzPrepText::gen_regions_unsup(int argc, const char *argv[]) const {
  const char *eyec = "AzPrepText::gen_regions_unsup"; 
  AzPrepText_gen_regions_unsup_Param p(argc, argv, out);   
  check_batch_id(p.s_batch_id);     
 
  bool do_xseq = p.s_xtyp.equals(kw_seq); 

  AzDic ydic(p.s_ydic_fn.c_str()); 
  int ydic_nn = ydic.get_max_n(); 
  AzPrint::writeln(out, "y dic n=", ydic_nn); 
  AzX::throw_if((ydic.size() <= 0), AzInputError, eyec, "No Y (target) vocabulary."); 
  
  AzDic xdic(p.s_xdic_fn.c_str()); 
  int xdic_nn = xdic.get_max_n(); 
  AzPrint::writeln(out, "x dic n=", xdic_nn);   
  AzX::throw_if((xdic.size() <= 0), AzInputError, eyec, "No vocabulary.");   
  AzX::no_support((xdic_nn > 1 && do_xseq), eyec, "X with multi-word vocabulary and Seq option");    

  /*---  scan files to determine buffer size and #data  ---*/
  AzOut noout; 
  AzStrPool sp_list; 
  AzIntArr ia_data_num; 
  int buff_size = AzTools_text::scan_files_in_list(p.s_inp_fn.c_str(), p.s_txt_ext.c_str(), 
                                                   noout, &sp_list, &ia_data_num);   
  int data_num = ia_data_num.sum(); 
  
  /*---  read data and generate features  ---*/
  int chkmem=100, wlen=5; /* estimate required memory size after 100 documents; assume a word is 5 char long. */  
  int ini = buff_size*chkmem/wlen*2; 
  Az_bc xbc(ini, ini*p.pch_sz*xdic_nn), ybc(ini, ini*p.dist*2*ydic_nn); 
  AzIntArr ia_dcolind(data_num*2); 
  
  buff_size += 256; 
  AzBytArr s_buff; 
  AzByte *buff = s_buff.reset(buff_size, 0); 
  int no_data = 0, data_no = 0, cnum = 0, cnum_before_reduce = 0; 
  int l_dist = -p.dist, r_dist = p.dist; 
  if (p.do_leftonly) r_dist = 0; 
  if (p.do_rightonly) l_dist = 0; 

  AzIntArr ia_xnn; for (int ix = 1; ix <= xdic_nn; ++ix) ia_xnn.put(ix); 
  AzIntArr ia_ynn; for (int ix = 1; ix <= ydic_nn; ++ix) ia_ynn.put(ix); 
  
  for (int fx = 0; fx < sp_list.size(); ++fx) { /* for each file */
    AzBytArr s_fn(sp_list.c_str(fx), p.s_txt_ext.c_str()); 
    const char *fn = s_fn.c_str(); 
    AzTimeLog::print(fn, out);   
    AzFile file(fn); 
    file.open("rb"); 
    int num_in_file = ia_data_num.get(fx); 
    int inc = num_in_file / 50, milestone = inc; 
    int dx = 0; 
    for ( ; ; ++dx) {  /* for each doc */
      AzTools::check_milestone(milestone, dx, inc); 
      int len = file.gets(buff, buff_size); 
      if (len <= 0) break; 
      
      int col_beg = xbc.colNum(); 
      
      /*---  X  ---*/
      AzIntArr ia_x_pos;    
      AzBytArr s_data(buff, len); 
      int my_len = s_data.length();

      bool do_skip_stopunk = (do_xseq)?false:true, do_allow_zero = false; 
      if (p.do_no_skip) {
        do_allow_zero = true; 
        do_skip_stopunk = false; 
      }
      if (xdic_nn > 1) do_skip_stopunk = false; /* 6/4/2017: prohibit VariableStride with n-grams */
      if (do_xseq) do_skip_stopunk = false;   /* 6/4/2017: prohibit VariableStride if sequential */
      bool do_contain = (xdic_nn > 1); 
      int unkw = -1;       

      AzDataArr<AzIntArr> aia_xtokno; 
      int xtok_num = AzTools_text::tokenize(s_data.point_u(), my_len, &xdic, ia_xnn, p.do_lower, p.do_utf8dashes, aia_xtokno);        
      xbc.check_overflow(eyec, xtok_num*p.pch_sz*xdic_nn, data_no); 
      if (do_xseq) gen_nobow_regions(xtok_num, aia_xtokno, xdic.size(), 
                                     p.pch_sz, p.pch_step, p.padding, do_allow_zero, unkw, 
                                     xbc, &ia_x_pos); 
      else         gen_bow_regions(xtok_num, aia_xtokno, ia_xnn, do_contain,
                                   p.pch_sz, p.pch_step, p.padding, do_allow_zero, do_skip_stopunk, 
                                   xbc, &ia_x_pos);  

      if (ia_x_pos.size() <= 0) {
        ++no_data; 
        continue; 
      }
      
      /*---  Y  ---*/
      s_data.reset(buff, len); 
      my_len = s_data.length();        
      if (ydic_nn > 1) { /* n-grams */
        AzDataArr<AzIntArr> aia_ytokno; 
        int ytok_num = AzTools_text::tokenize(s_data.point_u(), my_len, &ydic, ia_ynn, p.do_lower, p.do_utf8dashes, aia_ytokno);  
        AzX::throw_if((xtok_num != ytok_num), eyec, "conflict in the numbers of X tokens and Y tokens"); 
        ybc.check_overflow(eyec, ytok_num*ydic_nn*p.dist*2, data_no);         
        gen_Y_ngram_bow(ia_ynn, aia_ytokno, ydic.size(), ia_x_pos, 
                        p.pch_sz, l_dist, r_dist, p.gap, p.do_nolr, ybc); 
      }
      else { /* words */
        int nn = 1; 
        AzIntArr ia_ytokno; 
        AzTools_text::tokenize(s_data.point_u(), my_len, &ydic, nn, p.do_lower, p.do_utf8dashes, &ia_ytokno);  
        int ytok_num = ia_ytokno.size(); 
        AzX::throw_if((xtok_num != ytok_num), eyec, "conflict in the numbers of X tokens and Y tokens"); 
        ybc.check_overflow(eyec, ytok_num*ydic_nn*p.dist*2, data_no);         
        gen_Y(ia_ytokno, ydic.size(), ia_x_pos, 
              p.pch_sz, l_dist, r_dist, p.gap, p.do_nolr, ybc);      
      }
      
      cnum_before_reduce += xbc.colNum()-col_beg; 
      reduce_xy(p.min_x, p.min_y, col_beg, xbc, ybc); 
      if (xbc.colNum() <= col_beg) {
        ++no_data; 
        continue;         
      }
      cnum += xbc.colNum()-col_beg; 
      
      ++data_no;         
      ia_dcolind.put(col_beg); ia_dcolind.put(xbc.colNum()); 
      AzX::throw_if(xbc.colNum() != ybc.colNum(), eyec, "Conflict btw X index size and Y index size");         
      if (data_no == chkmem) {
        xbc.prepmem(data_no, data_num);
        ybc.prepmem(data_no, data_num);
      }           
    } /* for each doc */
    AzTools::finish_milestone(milestone); 
    AzBytArr s("   #data="); s<<data_no<<" no_data="<<no_data<<" #col="<<cnum; AzPrint::writeln(out, s); 
  } /* for each file */
  AzBytArr s("#data="); s<<data_no<<" no_data="<<no_data<<" #col="<<cnum<<" #col_all="<<cnum_before_reduce;
  s<<" #x="<<xbc.elmNum()<<" #y="<<ybc.elmNum(); 
  AzPrint::writeln(out, s);  
  xbc.commit(); ybc.commit(); 

  const char *outnm = p.s_rnm.c_str(); 
  AzTimeLog::print("Generating X ... ", out);
  int x_row_num = (xdic_nn <= 1 && do_xseq) ? xdic.size()*p.pch_sz : xdic.size();   
  write_regions(xbc, x_row_num, ia_dcolind, p.s_batch_id, outnm, p.s_x_ext.c_str()); 
  write_dic(xdic, x_row_num, outnm, xtext_ext);  
  AzTimeLog::print("Generating Y ... ", out);  
  int y_row_num = (p.do_nolr) ? ydic.size() : ydic.size()*2; 
  write_regions(ybc, y_row_num, ia_dcolind, p.s_batch_id, outnm, p.s_y_ext.c_str()); 
  write_dic(ydic, y_row_num, outnm, ytext_ext);  
  AzTimeLog::print("Done ... ", out); 
}

/*-------------------------------------------------------------------------*/
void AzPrepText::write_Y_smatc(const AzSmatc &m_y, const AzIntArr &ia_dcolind, 
                                const AzBytArr &s_batch_id, const char *outnm, 
                                const char *y_ext) const {
  int data_num = ia_dcolind.size()/2; 
  AzBytArr s(": ("); s<<m_y.rowNum()<<" x "<<m_y.colNum()<<", #data="<<data_num<<" )";                                   
  
  AzBytArr s_fn(outnm, y_ext); 
  if (s_batch_id.length() > 0) s_fn << "." << s_batch_id.c_str(); 
  const char *fn = s_fn.c_str(); 
  AzTimeLog::print(fn, s.c_str(), out); 
  AzFile file(fn); file.open("wb"); 
  if (AzBytArr::endsWith(y_ext, "var")) {
    /* write as AzSmatcVar with consistency check */      
    AzSmatcVar::write_hdr(&file, data_num, ia_dcolind, true, m_y.colNum()); 
  }
  m_y.write(&file); 
  file.close(true); 
}                                  

/*-------------------------------------------------------------------------*/
/* xy_ext must end with smatcvar or smatbcvar */
void AzPrepText::write_regions(const AzOut &out, const Az_bc &bc, 
                           int row_num, 
                           const AzIntArr &ia_dcolind, 
                           const AzBytArr &s_batch_id, 
                           const char *outnm, const char *xy_ext) /* static */ {
  AzX::throw_if(!bc.isCommitted(), "AzPrepText::write_regions", "bc is not commited."); 
  int data_num = ia_dcolind.size()/2; 
  AzBytArr s_xy(": "); s_xy << row_num << " x " << bc.colNum() << " (" << (double)bc.elmNum()/(double)bc.colNum(); 
  s_xy << "), #data=" << data_num; 

  AzBytArr s_xy_fn(outnm, xy_ext); 
  if (s_batch_id.length() > 0) s_xy_fn << "." << s_batch_id.c_str(); 
  const char *xy_fn = s_xy_fn.c_str(); 
  AzTimeLog::print(xy_fn, s_xy.c_str(), out); 
  AzFile file(xy_fn); file.open("wb"); 
  if (AzBytArr::endsWith(xy_ext, "var")) {
  /* write as AzSmat[b]cVar with consistency check */    
    AzSmatbcVar::write_hdr(&file, data_num, ia_dcolind, true, bc.colNum()); 
  }
  if (AzBytArr::contains(xy_ext, "bc")) AzSmatbc::write(&file, row_num, bc); 
  else                                  AzSmatc::write(&file, row_num, bc); /* we need this for seq2-bown */
  file.close(true); 
}  

/*-------------------------------------------------------------------------*/
void AzPrepText::write_dic(const AzDic &dic, int row_num, const char *nm, const char *ext) /* static */ {
  const char *eyec = "AzPrepText::write_dic"; 
  int pch_sz = row_num / dic.size(); 
  AzX::throw_if(row_num % dic.size() != 0, eyec, "Conflict in #row and size of vocabulary"); 
  AzBytArr s_fn(nm, ext); 
  if (pch_sz > 1) {
    AzDic seq_dic; 
    gen_nobow_dic(dic, pch_sz, seq_dic); 
    AzX::throw_if(seq_dic.size() != row_num, eyec, "Something is wrong ... "); 
    seq_dic.writeText(s_fn.c_str()); 
  }
  else {
    dic.writeText(s_fn.c_str()); 
  }
} 

/*-------------------------------------------------------------------------*/
/* not tested */
void AzPrepText::gen_Y_ngram_bow(const AzIntArr &ia_nn, 
                             const AzDataArr<AzIntArr> &aia_tokno, int dic_sz, 
                             const AzIntArr &ia_pos, 
                             int xpch_sz, /* patch size used to generate X */
                             int min_dist, int max_dist, int gap, 
                             bool do_nolr, 
                             Az_bc &bc) const {
  const char *eyec = "AzPrepText::gen_Y_ngram_bow"; 
  int t_num = aia_tokno[0]->size(); 
  for (int ix = 0; ix < ia_pos.size(); ++ix) {
    int xtx0 = ia_pos[ix]; 
    int xtx1 = xtx0 + xpch_sz; 
    if (gap > 0) { xtx0 -= gap; xtx1 += gap; }
    
    AzIntArr ia_ctx;   
  
    int base = xtx0+min_dist;  /* left */
    for (int nx = 0; nx < aia_tokno.size(); ++nx) {   
      int nn = ia_nn[nx]; 
      for (int tx = MAX(0,base); tx <= MIN(t_num,xtx0)-nn; ++tx) {         
        int tokno = (*aia_tokno[nx])[tx]; 
        if (tokno >= 0) ia_ctx.put(tokno); 
      }
    }
   
    base = xtx1; /* right */
    for (int nx = 0; nx < aia_tokno.size(); ++nx) {  
      int nn = ia_nn[nx];   
      for (int tx = MAX(0,base); tx <= MIN(t_num,xtx1+max_dist)-nn; ++tx) {           
        int tokno = (*aia_tokno[nx])[tx]; 
        if (tokno >= 0) {
          if (!do_nolr) tokno += dic_sz; 
          ia_ctx.put(tokno); 
        }
      }
    }
    bc.unique_put(ia_ctx);  
  }
}

/*-------------------------------------------------------------------------*/
template <class Typ>
void AzPrepText::reduce_xy(int min_x, int min_y, int col_beg, 
                           Az_bc &xbc, Typ &ybc_c) const {
  const char *eyec = "AzPrepText::reduce_xy";                           
  AzX::throw_if(ybc_c.colNum() != xbc.colNum(), eyec, "X and Y have different sizes");                            
  if (min_x <= 0 && min_y <= 0) return; 
  AzX::throw_if(min_x > 1 || min_y > 1, AzInputError, eyec, "min_x and min_y must be 1 or 0."); 
  for (int col = col_beg; col < xbc.colNum(); ++col) { 
    if (xbc.size(col) < min_x || ybc_c.size(col) < min_y) {
      xbc.remove_col(col); 
      ybc_c.remove_col(col);     
      --col; 
    }
  }  
}                         
template void AzPrepText::reduce_xy< Az_bc >(int, int, int, Az_bc &, Az_bc &) const; 
template void AzPrepText::reduce_xy< Az_c >(int, int, int, Az_bc &, Az_c &) const; 

/*-------------------------------------------------------------------------*/
void AzPrepText::gen_Y(const AzIntArr &ia_tokno, int dic_sz, 
                        const AzIntArr &ia_pos, 
                        int xpch_sz, /* patch size used to generate X */
                        int min_dist, int max_dist, int gap, 
                        bool do_nolr, 
                        Az_bc &bc) const {
  int t_num; const int *tokno = ia_tokno.point(&t_num); 
  for (int ix = 0; ix < ia_pos.size(); ++ix) {
    int xtx0 = ia_pos[ix]; 
    int xtx1 = xtx0 + xpch_sz;  
    if (gap > 0) { xtx0 -= gap; xtx1 += gap; }    
        
    AzIntArr ia;      
    for (int tx = MAX(0,xtx0+min_dist); tx < MIN(t_num,xtx0); ++tx) if (tokno[tx] >= 0) ia.put(tokno[tx]); 
    int offs = (do_nolr) ? 0 : dic_sz; 
    for (int tx = MAX(0,xtx1); tx < MIN(t_num,xtx1+max_dist); ++tx) if (tokno[tx] >= 0) ia.put(tokno[tx]+offs); 
    bc.unique_put(ia); 
  }
}
 
/*-------------------------------------------------------------------------*/
/*-------------------------------------------------------------------------*/
class AzPrepText_gen_regions_parsup_Param : public virtual AzPrepText_Param_ {
public: 
  AzBytArr s_xtyp, s_xdic_fn, s_inp_fn, s_txt_ext, s_rnm; 
  AzBytArr s_batch_id, s_x_ext, s_y_ext;
  AzBytArr s_feat_fn; 
  int dist, min_x, min_y; 
  int pch_sz, pch_step, padding; 
  int f_pch_sz, f_pch_step, f_padding; 
  bool do_lower, do_utf8dashes; 
  bool do_nolr, do_binarize;
  bool do_leftonly, do_rightonly, do_no_skip; 
  int top_num_each, top_num_total; 
  double scale_y, min_yval; 
 
  AzPrepText_gen_regions_parsup_Param(int argc, const char *argv[], const AzOut &out) 
    : dist(0), min_x(1), min_y(1), pch_sz(-1), pch_step(1), padding(0),
      s_x_ext(".xsmatbc"), s_y_ext(".ysmatc"),     
      top_num_each(-1), top_num_total(-1), 
      f_pch_sz(-1), f_pch_step(-1), f_padding(-1), 
      do_lower(false), do_utf8dashes(false), do_nolr(false), do_binarize(false), 
      do_leftonly(false), do_rightonly(false), do_no_skip(false),  
      scale_y(-1), min_yval(-1) {
    reset(argc, argv, out); 
  }

  /*-------------------------------------------------------------------------*/  
  #define kw_feat_fn "embed_fn="
  #define kw_top_num_each "num_top_each="
  #define kw_top_num_total "num_top="  
  #define kw_do_binarize "BinarizeY"
  #define kw_scale_y "scale_y="
  #define kw_min_yval "min_yval="
  #define kw_f_pch_sz "f_patch_size="
  #define kw_f_pch_step "f_patch_stride="
  #define kw_f_padding "f_padding="  
  /*-------------------------------------------------------------------------*/
  void resetParam(const AzOut &out, AzParam &azp) {
    const char *eyec = "AzPrepText_gen_regions_parsup_Param::resetParam"; 
    AzPrint o(out); 
    azp.vInt(o, kw_top_num_each, top_num_each); 
    azp.vInt(o, kw_top_num_total, top_num_total);     
    azp.vStr(o, kw_feat_fn, s_feat_fn); 
    azp.vStr(o, kw_xtyp, s_xtyp); 
    azp.vStr(o, kw_xdic_fn, s_xdic_fn);         
    azp.vStr(o, kw_inp_fn, s_inp_fn);      
    azp.vStr(o, kw_rnm, s_rnm);  
    azp.vStr_prt_if_not_empty(o, kw_txt_ext, s_txt_ext);  
    azp.vInt(o, kw_pch_sz, pch_sz);      
    azp.vInt(o, kw_pch_step, pch_step);   
    azp.vInt(o, kw_padding, padding);
    
    f_pch_sz = pch_sz; 
    f_pch_step = 1; 
    f_padding = f_pch_sz - 1; 
    azp.vInt(o, kw_f_pch_sz, f_pch_sz);      
    azp.vInt(o, kw_f_pch_step, f_pch_step);   
    azp.vInt(o, kw_f_padding, f_padding);
   
    azp.vInt(o, kw_dist, dist);   
    azp.swOn(o, do_lower, kw_do_lower); 
    azp.swOn(o, do_utf8dashes, kw_do_utf8dashes);     
    azp.swOn(o, do_nolr, kw_do_nolr); 
    azp.vStr_prt_if_not_empty(o, kw_batch_id, s_batch_id); 
    azp.vStr(o, kw_x_ext, s_x_ext, kw_x_ext_old); 
    azp.vStr(o, kw_y_ext, s_y_ext, kw_y_ext_old);     
    azp.swOn(o, do_binarize, kw_do_binarize); 
    if (!do_binarize) {
      azp.vFloat(o, kw_scale_y, scale_y); 
    }
    azp.vFloat(o, kw_min_yval, min_yval); 
    AzX::no_support(min_yval > 0, eyec, kw_min_yval); 
    azp.swOn(o, do_no_skip, kw_do_no_skip, false); 
    if (do_no_skip) {
      min_x = min_y = 0;       
    }   
    azp.swOn(o, do_leftonly, kw_do_leftonly); 
    if (!do_leftonly) azp.swOn(o, do_rightonly, kw_do_rightonly); 
    if (do_leftonly || do_rightonly) do_nolr = true;
    
    AzX::throw_if(min_x>1 || min_y>1, AzInputError, eyec, "min_x and min_y must be no greater than 1."); 
    AzXi::throw_if_empty(s_feat_fn, eyec, kw_feat_fn);      
    AzXi::throw_if_empty(s_xtyp, eyec, kw_xtyp); 
    AzXi::throw_if_empty(s_xdic_fn, eyec, kw_xdic_fn);     
    AzXi::throw_if_empty(s_inp_fn, eyec, kw_inp_fn);      
    AzXi::throw_if_empty(s_rnm, eyec, kw_rnm);      
    AzXi::throw_if_nonpositive(pch_sz, eyec, kw_pch_sz); 
    AzXi::throw_if_nonpositive(pch_step, eyec, kw_pch_step); 
    AzXi::throw_if_negative(padding, eyec, kw_padding); 
    AzXi::throw_if_nonpositive(f_pch_sz, eyec, kw_f_pch_sz); 
    AzXi::throw_if_nonpositive(f_pch_step, eyec, kw_f_pch_step); 
    AzXi::throw_if_negative(f_padding, eyec, kw_f_padding);     
    AzXi::throw_if_nonpositive(dist, eyec, kw_dist); 
    if (f_pch_sz > dist) {
      AzBytArr s(kw_f_pch_sz); s << " must be no greater than " << kw_dist << "."; 
      AzX::throw_if(true, AzInputError, eyec, s.c_str()); 
    }
    
    AzStrPool sp_x_ext(10,10); sp_x_ext.put(".xsmatbc", ".xsmatbcvar"); 
    AzXi::check_input(s_x_ext, &sp_x_ext, eyec, kw_x_ext);     
    AzStrPool sp_y_ext(10,10); sp_y_ext.put(".ysmatc", ".ysmatcvar");     
    AzXi::check_input(s_y_ext, &sp_y_ext, eyec, kw_y_ext);            
    o.printEnd(); 
  }
  void printHelp(const AzOut &out) const {    
    AzHelp h(out); h.begin("", "", "");  h.nl(); 
    h.writeln("To generate from unlabeled data a region file and target file for tv-embedding learning with partially-supervised target.  The task is to predict the internal features obtained by applying a supervised model to adjacent regions, based on the current region.\n", 3); 
    h.item_required(kw_xtyp, help_xtyp, kw_bow); 
    h.item_required(kw_inp_fn, "Path to the input token file or the list of token files.  If the filename ends with \".lst\", the file should be the list of token filenames.  The input token file(s) should contain one document per line, and each document should be tokens delimited by space."); 
    h.item_required(kw_rnm, help_rnm); 
    h.item_required(kw_xdic_fn, "Path to the vocabulary file generated by \"gen_vocab\", used for X (features).");   
    h.item_required(kw_pch_sz, "Region size."); 
    h.item(kw_pch_step, "Region stride.", "1"); 
    h.item(kw_padding, "Padding size.", "0");   
    h.item_required(kw_feat_fn, "Pathname to the internal-feature file.");     
    h.item_required(kw_f_pch_sz, "Region size that was used to generate the internal-feature file."); 
    h.item_required(kw_f_pch_step, "Region stride that was used to generate the internal-feature file."); 
    h.item_required(kw_f_padding, "Padding size that was used to generate the internal-feature file.");      
    h.item_required(kw_dist, "Size of adjacent regions used to produce Y (target).");         
    h.item(kw_top_num_total, "Number of the larget internal feature components to retain.  The rest will be set to zero."); 
    h.item(kw_do_lower, help_do_lower);     
    h.item(kw_do_utf8dashes, help_do_utf8dashes); 
    h.item(kw_do_nolr, help_do_nolr); 
    /* txt_ext, x_ext, y_ext, do_no_skip, min_x, min_y, top_num_each */
    h.end(); 
  }   
}; 

/*-------------------------------------------------------------------------*/
void AzPrepText::gen_regions_parsup(int argc, const char *argv[]) const {
  const char *eyec = "AzPrepText::gen_regions_parsup"; 
  AzPrepText_gen_regions_parsup_Param p(argc, argv, out);   
  check_batch_id(p.s_batch_id); 
  
  AzMats_file<AzSmat> mfile; 
  int feat_data_num = mfile.reset_for_read(p.s_feat_fn.c_str()); 

  AzStrPool sp_typ(10,10); sp_typ.put(kw_bow, kw_seq); 
  AzXi::check_input(p.s_xtyp.c_str(), &sp_typ, eyec, kw_xtyp);  
  bool do_xseq = p.s_xtyp.equals(kw_seq); 
  int l_dist = -p.dist, r_dist = p.dist; 
  if (p.do_leftonly) r_dist = 0; 
  if (p.do_rightonly) l_dist = 0; 
   
  AzDic dic(p.s_xdic_fn.c_str()); 
  int xdic_nn = dic.get_max_n(); 
  AzPrint::writeln(out, "x dic n=", xdic_nn);
  AzX::throw_if((dic.size() <= 0), AzInputError, eyec, "No vocabulary"); 
  AzX::no_support((xdic_nn > 1 && do_xseq), eyec, "X with multi-word vocabulary and Seq option"); 
  AzIntArr ia_xnn; for (int ix = 1; ix <= xdic_nn; ++ix) ia_xnn.put(ix); 
  
  /*---  scan files to determine buffer size and #data  ---*/
  AzOut noout; 
  AzStrPool sp_list; 
  AzIntArr ia_data_num; 
  int buff_size = AzTools_text::scan_files_in_list(p.s_inp_fn.c_str(), p.s_txt_ext.c_str(), 
                                                   noout, &sp_list, &ia_data_num);   
  int data_num = ia_data_num.sum(); 
  AzX::throw_if ((data_num != feat_data_num), eyec, "#data mismatch"); 
  
  /*---  read data and generate features  ---*/
  Az_bc xbc; 
  Az_c yc; 
  AzIntArr ia_dcolind; 

  buff_size += 256; 
  AzBytArr s_buff; 
  AzByte *buff = s_buff.reset(buff_size, 0); 
  int no_data = 0, data_no = 0, cnum = 0, cnum_before_reduce = 0; 
  feat_info fi[2];
  int y_row_num = 0;   
  for (int fx = 0; fx < sp_list.size(); ++fx) { /* for each file */
    AzBytArr s_fn(sp_list.c_str(fx), p.s_txt_ext.c_str()); 
    const char *fn = s_fn.c_str(); 
    AzTimeLog::print(fn, log_out);   
    AzFile file(fn); 
    file.open("rb"); 
    int num_in_file = ia_data_num.get(fx); 
    int inc = num_in_file / 50, milestone = inc; 
    int dx = 0; 
    for ( ; ; ++dx) {  /* for each doc */
      AzTools::check_milestone(milestone, dx, inc); 
      int len = file.gets(buff, buff_size); 
      if (len <= 0) break; 

      bool do_skip_stopunk = (do_xseq) ? false : true;   
      bool do_allow_zero = false;  
      if (p.do_no_skip) {
        do_skip_stopunk = false; 
        do_allow_zero = true;     
      }  
      if (xdic_nn > 1) do_skip_stopunk = false; 
      bool do_contain = (xdic_nn > 1); 
      int unkw = -1; 
      
      int col_beg = xbc.colNum();      
      /*---  X  ---*/
      AzBytArr s_data(buff, len); 
      int my_len = s_data.length();
      int tok_num = 0; 
      AzIntArr ia_pos; 

      AzDataArr<AzIntArr> aia_xtokno; 
      tok_num = AzTools_text::tokenize(s_data.point_u(), my_len, &dic, ia_xnn, p.do_lower, p.do_utf8dashes, aia_xtokno);        
      xbc.check_overflow(eyec, tok_num*p.pch_sz, data_no); 
      if (do_xseq) gen_nobow_regions(tok_num, aia_xtokno, dic.size(), 
                                     p.pch_sz, p.pch_step, p.padding, do_allow_zero, unkw, 
                                     xbc, &ia_pos); 
      else         gen_bow_regions(tok_num, aia_xtokno, ia_xnn, do_contain, 
                                   p.pch_sz, p.pch_step, p.padding, do_allow_zero, do_skip_stopunk, 
                                   xbc, &ia_pos);  

      AzSmat m_feat; 
      mfile.read(&m_feat);      
      if (ia_pos.size() <= 0) {
        ++no_data; 
        continue; 
      }
      if (p.top_num_each > 0 || p.top_num_total > 0 || p.scale_y > 0) {
        double min_ifeat = m_feat.min(); 
        AzX::no_support((min_ifeat < 0), eyec, "Negative values for internal-feature components."); 
      }
 
      /*---  Y (ifeat: internal features generated by a supervised model) ---*/ 
      y_row_num = gen_Y_ifeat(p.top_num_each, p.top_num_total, m_feat, tok_num, ia_pos, 
                  p.pch_sz, l_dist, r_dist, p.do_nolr, 
                  p.f_pch_sz, p.f_pch_step, p.f_padding, 
                  yc, fi); 
                                        
      cnum_before_reduce += (xbc.colNum() - col_beg); 
      reduce_xy(p.min_x, p.min_y, col_beg, xbc, yc);   
      if (xbc.colNum() <= col_beg) {
        ++no_data; 
        continue; 
      }
      ia_dcolind.put(col_beg); ia_dcolind.put(xbc.colNum()); 
      cnum += (xbc.colNum() - col_beg); 
      ++data_no;         
      AzX::throw_if(xbc.colNum() != yc.colNum(), eyec, "Conflict btw X #col and Y #col");         
    } /* for each doc */
    AzTools::finish_milestone(milestone); 
    AzBytArr s("   #data="); s << data_no << " no_data=" << no_data << " #col=" << cnum; 
    AzPrint::writeln(out, s); 
  } /* for each file */
  mfile.done();   

  xbc.commit(); 
  yc.commit(); 
  
  AzBytArr s("#data="); s<<data_no<<" no_data="<<no_data<<" #col="<<cnum<<" #col_all="<<cnum_before_reduce;        
  AzPrint::writeln(out, s); 
  s.reset("all:"); fi[0].show(s); AzPrint::writeln(out, s); 
  s.reset("top:"); fi[1].show(s); AzPrint::writeln(out, s); 

  AzSmatc m_y; m_y.set(y_row_num, yc); 
  yc.destroy(); 

  if (p.do_binarize) {
    AzTimeLog::print("Binarizing Y ... ", log_out); 
    m_y.binarize(); 
  }
  else if (p.scale_y > 0) {
    double max_top = fi[1].max_val; 
    double scale = 1; 
    if (max_top < p.scale_y) for ( ; ; scale *= 2) if (max_top*scale >= p.scale_y) break; 
    if (max_top > p.scale_y*2) for ( ; ; scale /= 2) if (max_top*scale <= p.scale_y*2) break; 
    s.reset("Multiplying Y with "); s << scale; AzPrint::writeln(out, s); 
    m_y.multiply(scale); 
  }  
  
  const char *outnm = p.s_rnm.c_str(); 
  AzTimeLog::print("Generating X ... ", out);  
  int x_row_num = (xdic_nn <= 1 && do_xseq) ? dic.size()*p.pch_sz : dic.size();     
  write_regions(xbc, x_row_num, ia_dcolind, p.s_batch_id, outnm, p.s_x_ext.c_str());
  write_dic(dic, x_row_num, outnm, xtext_ext); 

  AzTimeLog::print("Generating Y ... ", out);  
  write_Y_smatc(m_y, ia_dcolind, p.s_batch_id, outnm, p.s_y_ext.c_str());   
}

/*-------------------------------------------------------------------------*/
int AzPrepText::gen_Y_ifeat(int top_num_each, int top_num_total, const AzSmat &m_feat, 
                              int t_num, const AzIntArr &ia_pos, 
                              int xpch_sz, int min_dist, int max_dist, 
                              bool do_nolr, 
                              int f_pch_sz, int f_pch_step, int f_padding, 
                              Az_c &yc, 
                              feat_info fi[2]) const {
  const char *eyec = "AzPrepText::gen_Y_ifeat"; 
  int feat_sz = m_feat.rowNum(); 
  int f_pch_num = DIVUP(t_num+f_padding*2-f_pch_sz, f_pch_step) + 1; 
  if (m_feat.colNum() != f_pch_num) {
    AzBytArr s("#patch mismatch: Expcected: "); s << f_pch_num << " Actual: " << m_feat.colNum(); 
    AzX::throw_if(true, AzInputError, eyec, s.c_str()); 
  }

  for (int ix = 0; ix < ia_pos.size(); ++ix) {
    int xtx0 = ia_pos[ix]; 
    int xtx1 = xtx0 + xpch_sz; 
     
    AzIFarr ifa_ctx; 
    int offs = 0;     
    for (int tx = xtx0+min_dist; tx < xtx0; ++tx) {
      if (tx + f_pch_sz > xtx0) break; 
      set_ifeat(m_feat, top_num_each, (tx+f_padding)/f_pch_step, offs, ifa_ctx, fi); 
    }
    
    if (!do_nolr) offs = feat_sz; 
    for (int tx = xtx1; tx < xtx1+max_dist; ++tx) {
      if (tx + f_pch_sz > xtx1+max_dist) break; 
      set_ifeat(m_feat, top_num_each, (tx+f_padding)/f_pch_step, offs, ifa_ctx, fi); 
    }
    ifa_ctx.squeeze_Max(); 
    if (top_num_total > 0 && ifa_ctx.size() > top_num_total) {
      ifa_ctx.sort_Float(false);  /* WARNING: sort results depend on how the compiler breaks ties */
      ifa_ctx.cut(top_num_total);       
    }
    ifa_ctx.sort_Int(true); 
    AzValArr<AZI_VECT_ELM> _arr(ifa_ctx.size()); 
    for (int ix = 0; ix < ifa_ctx.size(); ++ix) {
      AZI_VECT_ELM elm; 
      elm.val = (AZ_MTX_FLOAT)ifa_ctx.get(ix, &elm.no); 
      _arr.put(elm);  
    }
    yc.put(_arr); 
  }
  return ((do_nolr) ? feat_sz : feat_sz*2); 
}                                   

/*-------------------------------------------------------------------------*/
void AzPrepText::set_ifeat(const AzSmat &m_feat, int top_num, 
                 int col, int offs, AzIFarr &ifa_ctx, feat_info fi[2]) const {                 
  if (col < 0 || col >= m_feat.colNum()) return; 
  AzIFarr ifa; m_feat.col(col)->nonZero(&ifa); 
  fi[0].update(ifa); 
  if (top_num > 0 && ifa.size() > top_num) {
    ifa.sort_FloatInt(false); /* descending order */
    ifa.cut(top_num); 
  }
  fi[1].update(ifa); 
  if (offs == 0) ifa_ctx.concat(&ifa); 
  else {
    for (int ix = 0; ix < ifa.size(); ++ix) {
      int row = ifa.getInt(ix); 
      double val = ifa.get(ix); 
      ifa_ctx.put(row+offs, val); 
    }    
  }
}   

/*-------------------------------------------------------------------------*/
/*-------------------------------------------------------------------------*/
class AzPrepText_show_regions_XY_Param : public virtual AzPrepText_Param_ {
public:
  AzBytArr s_rnm, s_x_ext, s_y_ext, s_batch_id; 
  bool do_wordonly; 
  AzPrepText_show_regions_XY_Param(int argc, const char *argv[], const AzOut &out) 
    : do_wordonly(false), s_x_ext(".xsmatbc"), s_y_ext(".ysmatbc") {
    reset(argc, argv, out); 
  } 
  #define kw_do_wordonly "ShowWordOnly"
  void resetParam(const AzOut &out, AzParam &azp) {
    const char *eyec = "AzPrepText_show_regions_Param::resetParam"; 
    AzPrint o(out); 
    azp.vStr(o, kw_rnm, s_rnm); 
    azp.vStr(o, kw_x_ext, s_x_ext); 
    azp.vStr(o, kw_y_ext, s_y_ext);     
    azp.vStr_prt_if_not_empty(o, kw_batch_id, s_batch_id); 
    azp.swOn(o, do_wordonly, kw_do_wordonly); 
    AzXi::throw_if_empty(s_rnm, eyec, kw_rnm); 
    o.printEnd(); 
  }
  void printHelp(const AzOut &out) const {
    AzHelp h(out); 
    h.item_required(kw_rnm, "Filename stem of the region file generated by \"gen_regions\"."); 
    h.item(kw_x_ext, "File extension of region file.", ".xsmatbc"); 
    h.item(kw_y_ext, "File extension of region file.", ".ysmatbc");  
    h.item(kw_do_wordonly, "Show words only without values."); 
    h.end(); 
  }
}; 

/*-------------------------------------------------------------------------*/
void AzPrepText::show_regions_XY(int argc, const char *argv[]) const {
  AzPrepText_show_regions_XY_Param p(argc, argv, out);   
  AzSmatVar mv_x, mv_y; 
  read_XY(p.s_rnm, p.s_x_ext, p.s_batch_id, mv_x); 
  read_XY(p.s_rnm, p.s_y_ext, p.s_batch_id, mv_y); 
  AzBytArr s_xtext_fn(p.s_rnm.c_str(), xtext_ext), s_ytext_fn(p.s_rnm.c_str(), ytext_ext);  
  AzDic xdic(s_xtext_fn.c_str()), ydic(s_ytext_fn.c_str()); 
  _show_regions_XY(mv_x, mv_y, xdic, ydic, p.do_wordonly); 
}

/*-------------------------------------------------------------------------*/
void AzPrepText::read_XY(const AzBytArr &s_rnm, const AzBytArr &s_ext, const AzBytArr &s_bid, AzSmatVar &mv) const {
  const char *eyec = "AzPrepText::read_XY"; 
  AzBytArr s_fn(&s_rnm); s_fn << s_ext; 
  if (s_bid.length() > 0) s_fn << "." << s_bid; 
  if (s_ext.contains("var")) {
    if (s_ext.endsWith("smatvar") || s_ext.endsWith("smatcvar")) mv.read(s_fn.c_str()); 
    else if (s_ext.endsWith("smatbcvar")) {
      AzSmatbcVar mvbc(s_fn.c_str()); 
      AzSmat m; mvbc.data()->copy_to_smat(&m); 
      mv.reset(&m, mvbc.index()); 
    }
    else AzX::throw_if(true, AzInputError, eyec, "Unsupported file extension"); 
  }  
  else {  
    AzSmat m; 
    if (s_ext.endsWith("smat")) m.read(s_fn.c_str()); 
    else if (s_ext.endsWith("smatbc")) {
      AzSmatbc mbc; mbc.read(s_fn.c_str()); 
      mbc.copy_to_smat(&m); 
    }
    else AzX::throw_if(true, AzInputError, eyec, "Unsupported file extension");     
    AzIntArr ia; ia.put(0); ia.put(m.colNum()); 
    mv.reset(&m, &ia); 
  }
}

/*-------------------------------------------------------------------------*/
void AzPrepText::_show_regions_XY(const AzSmatVar &mv_x, const AzSmatVar &mv_y, 
                               const AzDic &xdic, const AzDic &ydic, 
                               bool do_wordonly) const {
  const char *eyec = "AzPrepText::_show_regions_XY";                                  
  AzX::throw_if(mv_x.dataNum() != mv_y.dataNum(), AzInputError, eyec, "#data mismatch btw X and Y");                                  
  AzX::throw_if(mv_x.colNum() != mv_y.colNum(), AzInputError, eyec, "#col mismatch btw X and Y");   
  const AzSmat *mx = mv_x.data(), *my = mv_y.data(); 
  AzBytArr s; s << "#data=" << mv_x.dataNum() << " #col=" << mx->colNum(); 
  AzPrint::writeln(out, s); 
  for (int dx = 0; dx < mv_x.dataNum(); ++dx) {
    int col0 = mv_x.col_begin(dx), col1 = mv_x.col_end(dx); 
    s.reset("data#"); s << dx; s.nl(); 
    AzPrint::writeln(out, s); 
    int col; 
    for (col = col0; col < col1; ++col) {
      s.reset("  X: ");      
      AzTools_text::feat_to_text(mx->col(col), &xdic, do_wordonly, s); 
      s.nl(); 
      s << ("  Y: "); 
      AzTools_text::feat_to_text(my->col(col), &ydic, do_wordonly, s);
      AzPrint::writeln(out, s); 
    }
  }
} 

/*------------------------------------------------------------*/ 
/*------------------------------------------------------------*/ 
#define kw_w2vbin_fn "wordvec_bin_fn="
#define kw_w2vtxt_fn "wordvec_txt_fn="
#define kw_wvectxt_fn "wordvec_txt_vec_fn="
#define kw_wvecdic_fn "wordvec_txt_words_fn="
#define kw_do_ignore_dup "IgnoreDupWords"
#define kw_wvec_rand_param "rand_param="
#define kw_w_fn "weight_fn="
#define kw_do_verbose "Verbose"
#define kw_rand_seed "random_seed="
#define kw_wmap_fn "word_map_fn="    
/*------------------------------------------------------------*/ 
class AzPrepText_adapt_word_vectors_Param : public virtual AzPrepText_Param_ {
public: 
  AzBytArr s_w_fn, s_w2vbin_fn, s_w2vtxt_fn, s_wmap_fn; 
  AzBytArr s_wvectxt_fn, s_wvecdic_fn; 
  int rand_seed; 
  double wvec_rand_param; 
  bool do_verbose, do_ignore_dup; 
  AzPrepText_adapt_word_vectors_Param(int argc, const char *argv[], const AzOut &out) : 
    wvec_rand_param(-1), do_verbose(false), rand_seed(-1), do_ignore_dup(false) {
    reset(argc, argv, out); 
  }
  void resetParam(const AzOut &out, AzParam &p) {
    const char *eyec = "AzPrepText_adapt_word_vectors_Param::resetParam";  
    AzPrint o(out);     
    p.vStr(o, kw_w_fn, s_w_fn);   
    p.vStr_prt_if_not_empty(o, kw_w2vbin_fn, s_w2vbin_fn); 
    p.vStr_prt_if_not_empty(o, kw_w2vtxt_fn, s_w2vtxt_fn);     
    if (s_w2vbin_fn.length() <= 0 && s_w2vtxt_fn.length() <= 0) {
      /*---  old interface but keep it for compatibility  ---*/
      p.vStr_prt_if_not_empty(o, kw_wvectxt_fn, s_wvectxt_fn); 
      p.vStr_prt_if_not_empty(o, kw_wvecdic_fn, s_wvecdic_fn);     
    }
    p.vStr(o, kw_wmap_fn, s_wmap_fn); 
    p.vInt(o, kw_rand_seed, rand_seed); 
    p.vFloat(o, kw_wvec_rand_param, wvec_rand_param); 
    p.swOn(o, do_ignore_dup, kw_do_ignore_dup); 
    p.swOn(o, do_verbose, kw_do_verbose); 
    if      (s_wvectxt_fn.length() > 0) AzXi::throw_if_empty(s_wvecdic_fn, eyec, kw_wvecdic_fn); /* old interface */
    else if (s_wvecdic_fn.length() > 0) AzXi::throw_if_empty(s_wvectxt_fn, eyec, kw_wvectxt_fn); /* old interface */ 
    else                                AzXi::throw_if_both_or_neither(s_w2vbin_fn, s_w2vtxt_fn, eyec, kw_w2vbin_fn, kw_w2vtxt_fn); 
    AzXi::throw_if_empty(s_wmap_fn, eyec, kw_wmap_fn); 
    AzXi::throw_if_empty(s_w_fn, eyec, kw_w_fn);   
    if (rand_seed != -1) AzXi::throw_if_nonpositive(rand_seed, eyec, kw_rand_seed); 
    
    AzX::throw_if(!s_w_fn.endsWith("dmatc"), AzInputError, eyec, "The weight filename (\"weight_fn\") must end with \"dmatc\"."); 
  }
  void printHelp(const AzOut &out) const {   
    AzHelp h(out); h.begin("", "", "");  h.nl(); 
    h.writeln("adapt_word_vectors: convert a word vector file to a \"weight file\" so that the word vectors can be used to produce input to training.\n", 3); 
    h.item(kw_w2vbin_fn, "Path to a binary file containing word vectors and words in the word2vec binary format (input).  Either this or wordvec_txt_fn is requried.");
    h.item(kw_w2vtxt_fn, "Path to a text file containing word vectors and words in the word2vec text format (input).  Either this or wordvec_bin_fn is requried.");
#if 0     
    h.item(kw_wvectxt_fn, "Path to the word vector text file (input).  One line per vector.  Vector components should be delimited by space."); 
    h.item(kw_wvecdic_fn, "Path to the word vector word file (input).  One line per word."); 
#endif     
    h.item_required(kw_w_fn, "Path to the weight file to be generated (output).  It must end with \"dmatc\".");     
    h.item_required(kw_wmap_fn, "Path to the vocabulary file generated by \"prepText gen_vocab\".  Word vectors will be sorted in the order of this file.  Share this file with \"prepText gen_regions\" so that the dimensions of the resulting weights will correctly correspond to the dimensions of the sparse region vectors.");
    h.item(kw_rand_seed, "Random number generator seed."); 
    h.item(kw_wvec_rand_param, "x: scale of initialization.  If the word-mapping file (word_map_fn) contains words for which word vectors are not given, the word vectors for these unknown words will be randomly set by Gaussian distribution with zero mean with standard deviation x.", "0");     
    /* do_verbose */
    h.item(kw_do_ignore_dup, "Ignore it if there are duplicated words associated with word vectors.  If this is not turned on, the process will be terminated on the detection of duplicated words."); 
    h.end(); 
  }   
}; 

/*------------------------------------------------*/
void AzPrepText::adapt_word_vectors(int argc, const char *argv[]) const {
  const char *eyec = "AzPrepText::adapt_word_vectors"; 
  AzPrepText_adapt_word_vectors_Param p(argc, argv, out); 

  if (p.rand_seed > 0) srand(p.rand_seed);  
  
  AzTimeLog::print("Reading vectors ... ", log_out); 
  AzDic xdic(p.s_wmap_fn.c_str()), wvecdic; 
  AzDmat m_wvec; 
  if      (p.s_w2vbin_fn.length() > 0) read_word2vec(p.s_w2vbin_fn.c_str(), wvecdic, m_wvec, p.do_ignore_dup); 
  else if (p.s_w2vtxt_fn.length() > 0) read_word2vec(p.s_w2vtxt_fn.c_str(), wvecdic, m_wvec, p.do_ignore_dup, true); 
  else {
    wvecdic.reset(p.s_wvecdic_fn.c_str(), p.do_ignore_dup); 
    AzTextMat::readMatrix(p.s_wvectxt_fn.c_str(), &m_wvec); 
  }
  AzTimeLog::print("Processing ... ", log_out); 
  bool do_die_if = true; 
  int mapped = AzDic::rearrange_cols(xdic, wvecdic, m_wvec, do_die_if); 
  if (p.do_verbose) {
    AzBytArr s; s << wvecdic.size() << " -> " << xdic.size() << ": mapped=" << mapped; 
    AzTimeLog::print(s.c_str(), log_out); 
  }  
  if (p.wvec_rand_param > 0) {
    AzTimeLog::print("Random initialization of unknown words ... ", log_out); 
    zerovec_to_randvec(p.wvec_rand_param, m_wvec); 
  }
  
  AzDmat md_w; m_wvec.transpose_to(&md_w); m_wvec.destroy(); 
  AzDmatc m_w; m_w.copy_from(md_w); md_w.destroy();   
  AzBytArr s("Writing "); s << p.s_w_fn.c_str() << ": " << m_w.rowNum() << " x " << m_w.colNum();
  AzTimeLog::print(s.c_str(), " (binary: dmatc)", log_out); 
  m_w.write(p.s_w_fn.c_str());  

  AzTimeLog::print("Done ... ", log_out); 
}

/*------------------------------------------------*/ 
void AzPrepText::zerovec_to_randvec(double wvec_rand_param, AzDmat &m_wvec) {
  AzRandGen rg; 
  for (int col = 0; col < m_wvec.colNum(); ++col) {
    if (!m_wvec.isZero(col)) continue;    
    rg.gaussian(wvec_rand_param, m_wvec.col_u(col)); 
  }
}

/*--------------------------------------------------------------*/ 
/*  read word2vec word vectors as word2vec word-analogy.c does  */
/*--------------------------------------------------------------*/ 
void AzPrepText::read_word2vec(const char *fn, AzDic &dic, AzDmat &m_wvec, bool do_ignore_dup, bool is_text) {
  const char *eyec = "AzPrepText::read_word2vec_bin"; 
  AzBytArr s_err("An error was encountered while reading "); 
  s_err << fn << ".  The file is either corrupted or not in the word2vec "; 
  if (is_text) s_err << "text format."; 
  else         s_err << "binary format."; 

  AzFile file(fn); file.open("rb"); 
  FILE *f = file.ptr(); 
  long long words, size; 
  if (fscanf(f, "%lld", &words) != 1) throw new AzException(AzInputError, eyec, s_err.c_str()); 
  if (fscanf(f, "%lld", &size) != 1) throw new AzException(AzInputError, eyec, s_err.c_str()); 
  int row_num = Az64::to_int(size, "dimensionality of word2vec vectors"); 
  int col_num = Az64::to_int(words, "number of word2vec vectors"); 
  m_wvec.reform(row_num, col_num); 
  char word[4096];  /* assume no words/phrases are longer than this */
  AzStrPool sp; 
  for (int b = 0; b < words; b++) {
    char ch; 
    if (fscanf(f, "%s%c", word, &ch) != 2) throw new AzException(AzInputError, eyec, s_err.c_str()); 
    sp.put(word); 
    for (int a = 0; a < size; a++) {
      float val; 
      if (is_text) {
        char valtxt[4096];  /* assume no values are longer than this */
        if (fscanf(f, "%s%c", valtxt, &ch) != 2) throw new AzException(AzInputError, eyec, s_err.c_str());         
        val = (float)atof(valtxt); 
      }
      else {
        if (fread(&val, sizeof(float), 1, f) != 1) throw new AzException(AzInputError, eyec, s_err.c_str()); 
      }
      m_wvec.set(a, b, val);  
    }
  }
  dic.reset(&sp, do_ignore_dup);   
  file.close(); 
}

/*------------------------------------------------------------*/ 
/*------------------------------------------------------------*/ 
class AzPrepText_write_wv_word_mapping_Param : public virtual AzPrepText_Param_ {
public: 
  AzBytArr s_w2vbin_fn, s_w2vtxt_fn, s_wmap_fn; 
  bool do_ignore_dup; 
  AzPrepText_write_wv_word_mapping_Param(int argc, const char *argv[], const AzOut &out) : do_ignore_dup(false) {
    reset(argc, argv, out); 
  }
  void resetParam(const AzOut &out, AzParam &azp) {
    const char *eyec = "AzPrepText_write_wv_word_mapping_Param::resetParam";      
    AzPrint o(out); 
    azp.vStr_prt_if_not_empty(o, kw_w2vbin_fn, s_w2vbin_fn); 
    azp.vStr_prt_if_not_empty(o, kw_w2vtxt_fn, s_w2vtxt_fn);     
    azp.vStr(o, kw_wmap_fn, s_wmap_fn);     
    azp.swOn(o, do_ignore_dup, kw_do_ignore_dup); 
    AzXi::throw_if_both_or_neither(s_w2vbin_fn, s_w2vtxt_fn, eyec, kw_w2vbin_fn, kw_w2vtxt_fn);    
    AzXi::throw_if_empty(s_wmap_fn, eyec, kw_wmap_fn);       
  }
  void printHelp(const AzOut &out) {
    AzHelp h(out); h.begin("", "", "");  h.nl(); 
    h.writeln("write_wv_word_mapping: write word mapping of word vectors to a file\n", 3); 
    h.item(kw_w2vbin_fn, "Path to a binary file containing word vectors and words in the word2vec binary format (input).  Either this or wordvec_txt_fn is requried.");
    h.item(kw_w2vtxt_fn, "Path to a text file containing word vectors and words in the word2vec text format (input).  Either this or wordvec_bin_fn is requried.");
    h.item_required(kw_wmap_fn, "Path to the word mapping file to be written (output)."); 
    h.item(kw_do_ignore_dup, "Ignore it if there are duplicated words associated with word vectors.  If this is not turned on, the process will be terminated on the detection of duplicated words."); 
    h.end();
  }   
}; 

/*------------------------------------------------*/
void AzPrepText::write_wv_word_mapping(int argc, const char *argv[]) const {
  const char *eyec = "AzPrepText::write_wv_word_mapping"; 
  AzPrepText_write_wv_word_mapping_Param p(argc, argv, out); 
  AzTimeLog::print("Reading vectors ... ", log_out); 
  AzDic wvecdic; AzDmat m_wvec; 
  if (p.s_w2vbin_fn.length() > 0) read_word2vec(p.s_w2vbin_fn.c_str(), wvecdic, m_wvec, p.do_ignore_dup); 
  else                            read_word2vec(p.s_w2vtxt_fn.c_str(), wvecdic, m_wvec, p.do_ignore_dup, true);  
  AzBytArr s(p.s_wmap_fn.c_str()); s << ": size=" << wvecdic.size(); 
  AzTimeLog::print(s.c_str(), log_out); 
  wvecdic.writeText(p.s_wmap_fn.c_str());  
  AzTimeLog::print("Done ... ", log_out); 
} 

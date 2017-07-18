#***************************************************************************
# Functions below were used to extract and tokenize text used in [JZ17]. 
# They are not meant for general purposes.  
# 
# NOTE: "source sh-common.sh" is needed before "source dpcnn-functions.sh". 
# [JZ17] Deep pyramid convolutional neural networks for text categorization
#***************************************************************************

#---  Extract text and labels.  
#---  Input: ${orgdir}/train.csv, ${orgdir}/test.csv # Yelp.{f|p},Yahoo,Ama.{f|p},AG,Sogou,Dbpedia
#---                                                 #   donwloaded 
#---     
#---  Output:${odir}/${nm}-{train|test}.{txt|cat} # training/test text/label files
#---         ${odir}/${nm}.catdic                 # label dictionary 
ld_extract () {  
  local fnm=ld_extract

  if [ "$orgdir" = "" ]; then echo $fnm: orgdir is missing.; return 1; fi
  if [ "$odir"   = "" ]; then echo $fnm: odir is missing.; return 1; fi
  if [ "$nm"     = "" ]; then echo $fnm: nm is missing.; return 1; fi

  echo $fnm: orgdir=$orgdir odir=$odir nm=$nm  

  #---  Extract text and labels
  echo; echo Extracting text and labels ... 
  local typ; for typ in train test; do
    perl ld-extract.pl ${orgdir}/${typ}.csv ${odir}/${nm}-${typ} $do_dbpedia
    if [ $? != 0 ]; then echo $fnm: text extraction failed.; return 1; fi
  done

  #---  Prepare a label dictionary
  rm -f ${odir}/${nm}-train.catdic
  sort < ${odir}/${nm}-test.catdic > ${odir}/${nm}.catdic
  if [ $? != 0 ]; then echo $fnm: sort failed.; return 1; fi
  rm -f ${odir}/${nm}-test.catdic

  return 0
}

#---  Split training data into the training portion and the validation portion. 
#---  Split the training portion into data batches. 
#---  Tokenize training, validation, and test data. 
#---  Optionally, extract text and labels from a csv file in the charcnn format. 
#---
#---  Input: ${orgdir}/train.csv, ${orgdir}/test.csv
#---  Output: ${odir}/${nm}-td.${batch_id}.{txt.tok|cat} where batch_id is 1of${bb}, 2of${bb}, ...
#---          ${odir}/${nm}-dv.{txt.tok|cat}
#---          ${odir}/${nm}-test.${txt.tok|cat} 
ld_split_tokenize () {  
  local fnm=ld_split_tokenize
  if [ "$orgdir" = "" ]; then echo $fnm: orgdir is missing.; return 1; fi
  if [ "$odir" = "" ]; then echo $fnm: odir is missing.; return 1; fi
  if [ "$bb"   = "" ]; then echo $fnm: bb is missing.; return 1; fi
  if [ "$nm"  = "" ]; then echo $fnm: nm is missing.; return 1; fi
  if [ "$num"  = "" ]; then echo $fnm: num is missing.; return 1; fi
  if [ "$dev"  = "" ]; then dev=10; fi

# optional: max_jobs, do_dbpedia, reg
  if [ "$reg" = "" ]; then reg=${odir}/empty; touch $reg; fi
  if [ "$max_jobs" = "" ]; then max_jobs=1; fi
  echo $fnm: orgdir=$orgdir odir=$odir bb=$bb nm=$nm num=$num dev=$dev reg=$reg   
  echo $fnm: max_jobs=$max_jobs do_dbpedia=$do_dbpedia

  #---  Extract text and labels
  ld_extract
  if [ $? != 0 ]; then echo $fnm: ld_extract failed.; return 1; fi
  _onm=${odir}/${nm}

  #---  Split the training set ("train") into a developemnt set ("dv") and the rest ("td") 
  echo Splitting ... 
  local ext; for ext in txt cat; do
    local _tmp=${_onm}.tmp
    $prep_exe split_text input_fn=${_onm}-train.${ext} random_seed=1 output_fn_stem=${_tmp}.${ext} split=$((num-dev)):${dev}
    if [ $? != 0 ]; then echo $fnm: split_text failed.; return 1; fi
    mv ${_tmp}.${ext}.1of2 ${_onm}-td.${ext}
    mv ${_tmp}.${ext}.2of2 ${_onm}-dv.${ext} 
  done

  #---  Split the td set (used as training data) into $bb batches 
  local _split=1; local no; for (( no=1; no<=bb-1; no++ )) do _split=${_split}:1; done  # e.g., 1:1:1:1:1 for bb=5
  echo $fnm: _split=$_split
  local ext; for ext in txt cat; do
    if [ "$bb" = 1 ]; then
      mv ${_onm}-td.${ext} ${_onm}-td.${ext}.1of1
    else
      $prep_exe split_text input_fn=${_onm}-td.${ext} random_seed=1 output_fn_stem=${_onm}-td.${ext} split=$_split
      if [ $? != 0 ]; then echo $fnm: split_text failed.; return 1; fi
    fi
  done


  echo Tokenizing ... 
  _cmd_=(); local num=0
  #---  Tokenize dv and test.
  local set; for set in dv test; do
    _cmd_[$num]="perl to_tokens.pl ${_onm}-${set}.txt $reg .tok 1"; num=$((num+1))
  done 
  #---  Tokenize td. 
  local no; for (( no=1; no<=bb; no++ )); do 
    _cmd_[$num]="perl to_tokens.pl ${_onm}-td.txt.${no}of${bb} $reg .tok 1"; num=$((num+1))
  done
  do_jobs $max_jobs $num tok-${nm}
  if [ $? != 0 ]; then echo $nm: do_jobs failed.; return 1; fi

  echo Done ...

  #---  Rename td tokenized files
  local no; for (( no=1; no<=bb; no++ )) do 
    mv ${_onm}-td.txt.${no}of${bb}.tok ${_onm}-td.${no}of${bb}.txt.tok
    mv ${_onm}-td.cat.${no}of${bb}     ${_onm}-td.${no}of${bb}.cat
  done

  return 0
}

#***************************************************************************
# Functions below are for reproducing the experimental results in [JZ17]. 
# NOTE: "source sh-common.sh" is needed before "source dpcnn-functions.sh".
# 
# [JZ17] Deep pyramid convolutional neural networks for text categorization
#***************************************************************************

#---  Global variables 
  dvkw=dv
  tstkw=test
  trnkw=td
  options="LowerCase UTF8"
  showopt=
  _extarr_=  # Used to pass an array to _return0_if_all_exist()

#-------------------
#---  Supervised training
#-------------------
dpcnn_training () {
  local fnm=dpcnn_training
  #---  Required parameters given via global variables. 
  if [ "$nm"   = "" ]; then echo $fnm: nm is missing.;   return 1; fi
  if [ "$bb"   = "" ]; then echo $fnm: bb is missing.;   return 1; fi
  if [ "$p "   = "" ]; then echo $fnm: p is missing.;    return 1; fi
  if [ "$pnm"  = "" ]; then echo $fnm: pnm is missing.;  return 1; fi
  if [ "$lays" = "" ]; then echo $fnm: lays is missing.; return 1; fi
  if [ "$ss"   = "" ]; then echo $fnm: ss is missing.;   return 1; fi
  if [ "$mynm" = "" ]; then echo $fnm: mynm is missing.; return 1; fi
  echo $fnm: nm=$nm bb=$bb p=$p pnm=$pnm lays=$lays ss=$ss topdo=$topdo mynm=$mynm

  #---  Optional parameters.  Copy to local variables and set default values if missing. 
  # do_save
  local _tmp=$tmp;     if [ "$_tmb"   = "" ]; then _tmb=500;   fi
  local _epo=$epochs;  if [ "$_epo"   = "" ]; then _epo=150;   fi
  local _lam=$lam;     if [ "$_lam"   = "" ]; then _lam=1e-4;  fi
  local _tlam=$toplam; if [ "$_tlam"  = "" ]; then _tlam=1e-4; fi
  local _nodes=$nodes; if [ "$_nodes" = "" ]; then _nodes=250; fi
  local _psz=$psz;     if [ "$_psz"   = "" ]; then _psz=3;     fi
  local _gpu=$gpu;     if [ "$_gpu"   = "" ]; then _gpu=-1;    fi
  local _mem=$mem;     if [ "$_mem"   = "" ]; then _mem=4;     fi
  echo $fnm: tmb=$_tmb epochs=$_epo lam=$_lam toplam=$_tlam nodes=$_nodes psz=$_psz gpu=$_gpu mem=$_mem

  #---  network definition (except for layer#0) 
  local lp0=2 # inner loop count
  local lp1=$(( (lays-1)/lp0 ))

  local _conn="conn="  # connection parameter 
  local l=0
  local act=None
  local _pm=
  _pm="${_pm} nodes=$_nodes patch_size=$_psz padding=$(((_psz-1)/2))" # default parameters for all layers (used only by applicable layers)
                                                          
  local i1; for (( i1=0; i1<lp1; i1++ )); do
    local u=3 # 3 for Act, Patch, and Weight 
    _conn="${_conn}${l}-$((l+u*lp0+1)),"
    local i0; for (( i0=0; i0<lp0; i0++ )); do 
      local no=$(( lp0*i1+i0+1 )) # +1 for layer#0

      l=$((l+1)); _pm="${_pm} ${l}name=Act${no}    ${l}layer_type=Act   ${l}activ_type=$act"; act=Rect
      l=$((l+1)); _pm="${_pm} ${l}name=Patch${no}  ${l}layer_type=Patch "
      l=$((l+1)); _pm="${_pm} ${l}name=Weight${no} ${l}layer_type=DenWei  "
    done

    l=$((l+1))
    _pm="${_pm} ${l}layer_type=Pooling ${l}pooling_type=Max"
    if [ "$i1" = $((lp1-1)) ]; then # last block 
      _pm="${_pm} ${l}name=LastPool  ${l}num_pooling=1"
    else 
      _pm="${_pm} ${l}name=Pool${no} ${l}pooling_size=3 ${l}pooling_stride=2 ${l}pooling_padding=$(((3-1)/2))"
    fi
  done
  _pm="${_pm} layers=$((l+1))"
  if [ "$topdo" != "-1" ]; then _pm="${_pm} top_dropout=$topdo"; fi

  _conn="${_conn}0"
  local i; for (( i=1; i<=l; i++ )); do _conn="${_conn}-${i}"; done
  _conn="${_conn}-top"

  #---  unsupervised embeddings ("side" layer(s) attached to layer#0)
  local _skw=
  local _spm=
  local _sn=0
  if [ "$spnms" != "" ]; then
    local i; for (( i=0; ; i++ )); do
      local spnm=${spnms[i]}
      if [ "$spnm" = "" ] || [ "$spnm" = "_" ]; then break; fi

      local sfn=${outdir}/${nm}-td-uns-${spnm}-dim${dim}.epo10.ReLayer0 
      local dsno=$((i+1)); pfx=0side${i}_
      _spm="${_spm} ${pfx}layer_type=Weight+ ${pfx}layer_fn=${sfn} ${pfx}dsno=${dsno} dsno${dsno}=${spnm}"
      _skw=${_skw}-${spnm}
      _sn=$((_sn+1))
    done
    if [ "$_sn" != 0 ]; then _spm="num_sides=${_sn} ${_spm}"; fi
  fi

  #---  layer#0 type 
  local _pm0="0name=Embedding"
  if [ "$_sn" != 0 ]; then _pm0="${_pm0} 0layer_type=WeightS+"
  else                     _pm0="${_pm0} 0layer_type=Weight+"; fi

  #---  training parameters 
  local _tpm=
  _tpm="${_tpm} top_reg_L2=$_tlam reg_L2=$_lam step_size=$ss \
        loss=Log mini_batch_size=100 momentum=0.9 random_seed=1 \
        num_epochs=$_epo ss_scheduler=Few ss_decay=0.1 ss_decay_at=$((_epo*4/5)) \
        trnname=${nm}-${trnkw}- tstname=${nm}-${dvkw}-+${nm}-${tstkw}- data_dir=$tmpdir \
        datatype=sparse dsno0=$pnm \
        SaveDataMem num_batches=$bb test_mini_batch_size=$_tmb test_interval=5 \
        max_loss=5 inc=10000 $showopt"
  if [ "$do_save" = 1 ]; then 
    _tpm="${_tpm} save_fn=${outdir}/${mynm} save_interval=5 save_after=$(( _epo*4/5 ))"
  fi

  #---  training 
  local _logfn=${logdir}/${mynm}${_skw}.log; local _csvfn=${csvdir}/${mynm}${_skw}.csv
  echo; echo $fnm: Training ... see $_logfn and $_csvfn for progress.
  $exe $_gpu:$_mem train $_conn $_pm0 $_pm $_spm $_tpm evaluation_fn=$_csvfn > $_logfn
  if [ $? != 0 ]; then echo $fnm: training failed. See $_logfn; return 1; fi

  echo $fnm: success
  return 0
}

#-------------------
#---  Write prediction values to a file
#-------------------
dpcnn_predict () {
  local fnm=dpcnn_predict
  #---  Required parameters given via global variables. 
  if [ "$nm"   = "" ]; then echo $fnm: nm is missing.; return 1; fi
  if [ "$pnm"  = "" ]; then echo $fnm: pnm is missing.; return 1; fi
  if [ "$mynm" = "" ]; then echo $fnm: mynm is missing.; return 1; fi
  if [ "$epochs" = "" ]; then echo $fnm: epochs is missing.; return 1; fi
  echo $fnm: nm=$nm mynm=$mynm pnm=$pnm epochs=$epochs do_write_text=$do_write_text

  #---  Optional parameters.  Copy to local variables and set default valuse if missing. 
  local _gpu=$gpu; if [ "$_gpu" = "" ]; then _gpu=-1; fi
  local _tmb=$tmb; if [ "$_tmb" = "" ]; then _tmb=500; fi

  echo $fnm: gpu=$_gpu mem=$mem  

  local _mod_fn=${outdir}/${mynm}.epo${epochs}.ReNet
  local _pred_fn=${_mod_fn}.pred
  local _opt=
  if [ "$do_write_text" = 1 ]; then _pred_fn=${_pred_fn}.txt; _opt=WriteText
  else                              _pred_fn=${_pred_fn}.bin; 
  fi  

  local _pm="dsno0=$pnm"
  if [ "$spnms" != "" ]; then
    local i; for (( i=0; ; i++ )); do
      local spnm=${spnms[i]}
      if [ "$spnm" = "" ] || [ "$spnm" = "_" ]; then break; fi
      local dsno=$((i+1)); _pm="${_pm} dsno${dsno}=${spnm}" # "dsno" needs to be set exactly as was set for training.  
    done
  fi
  $exe $_gpu:$mem predict $_opt model_fn=$_mod_fn prediction_fn=$_pred_fn \
       datatype=sparse tstname=${nm}-${tstkw}- data_dir=$tmpdir \
       test_mini_batch_size=$_tmb $_pm
  if [ $? != 0 ]; then echo $fnm: predict failed.; return 1; fi

  echo $fnm: success
  return 0
}

#-----------------
_return0_if_all_exist__uns () {
  local _rnm=$1; local _bid=$2
  _extarr_=( xsmatbc ysmatbc xtext )
  _return0_if_all_exist $_rnm $_bid
  return $?
}

#-----------------
_return0_if_all_exist__sup_Ronly () {
  local _rnm=$1; local _bid=$2
  _extarr_=( xsmatbcvar xtext )
  _return0_if_all_exist $_rnm $_bid
  return $?
}

#-----------------
_return0_if_all_exist__sup () {
  local _rnm=$1; local _bid=$2
  _extarr_=( xsmatbcvar y xtext )
  _return0_if_all_exist $_rnm $_bid
  return $?
}

#-----------------
_return0_if_all_exist () {
  local _rnm=$1; local _bid=$2
  if [ "$dont_reuse" = 1 ]; then return 1; fi  # Don't reuse old files.  Create new ones.

  local i; for (( i=0; ; i++ )); do
    local _ext=${_extarr_[i]}
    if [ "$_ext" = "" ]; then break; fi
    local _fn=${_rnm}.${_ext}
    if [ "$_bid" != "" ] && [ "$_ext" != "xtext" ]; then _fn=${_fn}.${_bid}; fi
    if [ ! -e "$_fn" ]; then return 1; fi
    echo ... $_fn exists. 
  done
  if [ "_$id" = "" ]; then echo $_rnm: all files exist
  else                     echo $_rnm $_bid: all files exist
  fi
  return 0
}

#-----------------
#---  Unsupervised embedding training. 
#----------------
dpcnn_train_unsemb () {
  local fnm=dpcnn_train_unsemb
  #---  Required parameters given via global variables. 
  if [ "$nm"   = "" ]; then echo $fnm: nm is missing.;  return 1; fi
  if [ "$bb"   = "" ]; then echo $fnm: bb is missing.;  return 1; fi
  if [ "$pnm"  = "" ]; then echo $fnm: pnm is missing.; return 1; fi
  if [ "$dim"  = "" ]; then echo $fnm: dim is missing.; return 1; fi
  echo $fnm: nm=$nm bb=$bb pnm=$pnm dim=$dim

  #---  Optional parameters.  Copy to local variables and set default values if missing. 
  local _gpu=$gpu;      if [ "$_gpu"  = "" ]; then _gpu=-1;    fi
  local _uss=$uss;      if [ "$_uss"  = "" ]; then _uss=5.0;   fi  
  local _umom=$umom;    if [ "$_umom" = "" ]; then _umom=-1;   fi
  local _uact=$uact;    if [ "$_uact" = "" ]; then _uact=Rect; fi
  local _uepo=$uepochs; if [ "$_uepo" = "" ]; then _uepo=10;   fi
  local _umem=$umem;    if [ "$_umem" = "" ]; then _umem=1;    fi
  echo $fnm: uss=$_uss umom=$_umom uact=$_uact uepochs=$_uepo umem=$umem
  echo $fnm: gpu=$_gpu mem=$mem

  local _unsnm=${nm}-${trnkw}-uns-${pnm}

  local _mynm=${_unsnm}-dim${dim}
  local _lay0_fn=${outdir}/${_mynm}
  local _logfn=${logdir}/${_mynm}.log
  echo; echo $fnm: Training unsupervised embedding ... see $_logfn for progress. 
  $exe $_gpu:$_umem train num_batches=$bb trnname=$_unsnm data_dir=$tmpdir \
        datatype=sparse x_ext=.xsmatbc y_ext=.ysmatbc \
        step_size_decay_at=$(( _uepo*4/5 )) num_epochs=$_uepo \
        NoCusparseIndex zero_Y_weight=0.2 zero_Y_ratio=5 \
        layers=1 0layer_type=Weight+ 0nodes=$dim 0save_layer_fn=$_lay0_fn \
        0resnorm_type=Text  NoTest Regression \
        loss=Square ss_scheduler=Few ss_decay=0.1 inc=500000 \
        mini_batch_size=100 step_size=$_uss momentum=$_umom \
        reg_L2=0 activ_type=$_uact random_seed=1 $showopt > $_logfn
  if [ $? != 0 ]; then echo $fnm: unsupervised embedding training failed. See $_logfn; return 1; fi

  echo $fnm: success
  return 0
}

#---  Generate list training token files. 
#----------------
_dpcnn_gen_td_lst () {
  local _fnm=_dpcnn_gen_td_lst
  local _plst=$1
  if [ "$_plst" = "" ]; then echo $fnm: _plst is missing.; return 1; fi
  if [ "$nm" = ""    ]; then echo $fnm: nm is missing.;    return 1; fi
  if [ "$bb" = ""    ]; then echo $fnm: bb is missing.;    return 1; fi
  if [ "$idir" = ""  ]; then echo $fnm: idir is missing;   return 1; fi
#  if [ "$dont_reuse" != 1 ] && [ -e $_plst ]; then
#    echo $_fnm: Using existing $_plst ... 
#  else 
    rm -f $_plst
    local no; for (( no=1; no<=bb; no++ )); do
      echo ${idir}/${nm}-${trnkw}.${no}of${bb}.txt.tok >> $_plst
    done
#  fi
}

#---  Generate a vocabulary file of word uni-grams
#----------------
_dpcnn_gen_vocab () {
  local _fnm=_dpcnn_gen_vocab
  local _voc=$1
  local _max=$2    # size of vocabulary 
  local _stopfn=$3
  if [ "$_voc" = "" ]; then echo $_fnm: _voc is missing; return 1; fi
  if [ "$_max" = "" ]; then _max=30; fi

  if [ "$dont_reuse" != 1 ] && [ -e $_voc ]; then
    echo $_fnm: Using existing $_voc ... 
  else
    local _lst=${tmpdir}/${nm}.lst
    _dpcnn_gen_td_lst $_lst
    $prep_exe gen_vocab input_fn=$_lst vocab_fn=$_voc $options WriteCount stopword_fn=$_stopfn max_vocab_size=${_max}000
    if [ $? != 0 ]; then rm -f $_voc; echo $_fnm: gen_vocab failed.; return 1; fi
  fi
  return 0
}

#---  Generate a vocabulary file of word uni-, bi-, and tri-grams
#----------------
_dpcnn_gen_vocab3 () {
  local fnm=_dpcnn_gen_vocab3
  local _max_jobs=$1 
  local _voc=$2
  local _max=$3    # size of vocabulary 
  local _z=$4      
  if [ "$_max_jobs = "" ]; then echo $fnm: _max_jobs is missing; return 1; fi
  if [ "$_voc"     = "" ]; then echo $fnm: _voc is missing; return 1; fi
  if [ "$_max      = "" ]; then _max=200; fi

  if [ "$dont_reuse" != 1 ] && [ -e $_voc ]; then
    echo $fnm: Using exisintg $_voc ... 
    return 0; 
  fi
  local _lst=${tmpdir}/${nm}.lst
  _dpcnn_gen_td_lst $_lst
  local _fns=
  _cmd_=(); local num=0
  local _n; for _n in 1 2 3; do
    local _mincount=5
    local _voc0=${tmpdir}/${nm}${_z}-${_n}grams-tmp.vocab
    _cmd_[$num]="$prep_exe gen_vocab input_fn=$_lst vocab_fn=$_voc0 $options WriteCount n=$_n min_word_count=$_mincount"
    num=$((num+1))
    if [ "$_fns" != "" ]; then _fns=${_fns}+; fi
    _fns=$_fns$_voc0
  done
  do_jobs $max_jobs $num genV3${_z}-${nm}
  if [ $? != 0 ]; then rm -f ${tmpdir}/${nm}${_z}*grams-tmp.vocab; echo $fnm: gen_vocab failed.; return 1; fi

  $prep_exe merge_vocab input_fns=$_fns vocab_fn=$_voc WriteCount max_vocab_size=${_max}000 UseInputFileOrder
  if [ $? != 0 ]; then rm -f $_voc; echo $fnm: merge_vocab failed.; return 1; fi

  return 0
}

#----------------
#---  Generate regions files for supervised training. 
#---- # of concurrent processes: bb+2
#----------------
dpcnn_gen_regions () {
  local fnm=dpcnn_gen_regions
  #---  Required parameters given via global variables. 
  if [ "$nm" = ""   ]; then echo $fnm: nm is missing.;   return 1; fi
  if [ "$bb" = ""   ]; then echo $fnm: bb is missing.;   return 1; fi
  if [ "$p"  = ""   ]; then echo $fnm: p is missing.;    return 1; fi
  if [ "$pnm" = ""  ]; then echo $fnm: pnm is missing.;  return 1; fi
  if [ "$idir" = "" ]; then echo $fnm: idir is missing.; return 1; fi
  if [ "$xmax" = "" ]; then echo $fnm: xmax is missing.; return 1; fi
  if [ "$max_jobs" = "" ]; then echo $fnm: max_jobs is missing.; return 1; fi
  echo $fnm: nm=$nm bb=$bb p=$p pnm=$pnm idir=$idir xmax=$xmax max_jobs=$max_jobs

  #---  Optional parameters.
  local _catdic=$catdic; if [ "$_catdic" = "" ]; then _catdic=${idir}/${nm}.catdic; fi
  echo $fnm: catdic=$_catdic

  local _xvoc=${tmpdir}/${nm}-${trnkw}-${xmax}k.vocab
  _dpcnn_gen_vocab $_xvoc $xmax
  if [ $? != 0 ]; then echo $fnm: _dpcnn_gen_vocab failed.; return 1; fi

  #---  Generate region files for supervised training 
  local _pm="$options Bow NoSkip vocab_fn=$_xvoc patch_size=$p padding=$(((p-1)/2)) label_dic_fn=$_catdic"

  _cmd_=(); local num=0
  local _set; for _set in $dvkw $tstkw; do  # dv and tset
    local _rnm=${tmpdir}/${nm}-${_set}-${pnm}
    _return0_if_all_exist__sup $_rnm
    if [ $? != 0 ]; then 
      local _inp_fn=${idir}/${nm}-${_set}
      _cmd_[$num]="$prep_exe gen_regions $_pm region_fn_stem=$_rnm input_fn=$_inp_fn"
      num=$((num+1))
    fi
  done
  local _no; for (( _no=1; _no<=bb; _no++ )); do  # td
    local _bid=${_no}of${bb}
    local _rnm=${tmpdir}/${nm}-${trnkw}-${pnm}
    _return0_if_all_exist__sup $_rnm $_bid
    if [ $? != 0 ]; then
      local _inp_fn=${idir}/${nm}-${trnkw}.${_bid}
      _cmd_[$num]="$prep_exe gen_regions $_pm region_fn_stem=$_rnm input_fn=$_inp_fn batch_id=$_bid "
      num=$((num+1))
    fi
  done

  do_jobs $max_jobs $num  genR-${pnm}-${nm} # do_jobs (sh-common.sh) executes commands in _cmd_. 
  if [ $? != 0 ]; then echo $fnm: do_jobs failed.; return 1; fi

  echo $fnm: success
  return 0
}

#-------------------
#---  Generate region files for side layers (unsupervised embeddings). 
#-------------------
dpcnn_gen_regions_side () {
  local fnm=dpcnn_gen_regions_side
  #---  Required parameters. 
  if [ "$nm"  = "" ]; then echo nm is missing.; return 1; fi   # data name
  if [ "$bb"  = "" ]; then echo bb is missing.; return 1; fi   # number of training data batches
  if [ "$p"   = "" ]; then echo p is missing.; return 1; fi    # region size
  if [ "$pnm" = "" ]; then echo pnm is missing.; return 1; fi  # used to generate filenames
  if [ "idir" = "" ]; then echo idir is missing.; return 1; fi # input data directory
  if [ "$max_jobs" = "" ]; then echo $fnm: max_jobs is missing.; return 1; fi # max # of background processes
  echo $fnm: nm=$nm bb=$bb p=$p pnm=$pnm idir=$idir max_jobs=$max_jobs

  local _xvoc=${tmpdir}/${nm}-${pnm}.wmap  # generated by dpcnn_gen_regions_unsemb 
  if [ ! -e $_xvoc ]; then echo Failed to find $_xvoc, which was supposed to be made by dpcnn_gen_regions_unsemb with pnm=$pnm.; return 1; fi

  local _pm="$options Bow NoSkip vocab_fn=$_xvoc patch_size=$p padding=$(((p-1)/2)) Contain RegionOnly"
  _cmd_=(); local num=0
  local _no; for (( _no=1; _no<=bb; _no++ )); do  # td
    local _rnm=${tmpdir}/${nm}-${trnkw}-${pnm}
    local _bid=${_no}of${bb}
    _return0_if_all_exist__sup_Ronly $_rnm $_bid
    if [ $? != 0 ]; then
      local _inp_fn=${idir}/${nm}-${trnkw}.${_bid}
      _cmd_[$num]="$prep_exe gen_regions input_fn=$_inp_fn region_fn_stem=$_rnm $_pm batch_id=$_bid"
      num=$((num+1))
    fi
  done
  local _set; for _set in $dvkw $tstkw; do  # dv and test
    local _rnm=${tmpdir}/${nm}-${_set}-${pnm}
    _return0_if_all_exist__sup_Ronly $_rnm
    if [ $? != 0 ]; then
      local _inp_fn=${idir}/${nm}-${_set}
      _cmd_[$num]="$prep_exe gen_regions input_fn=$_inp_fn region_fn_stem=$_rnm $_pm"
      num=$((num+1))
    fi
  done

  do_jobs $max_jobs $num  genR4s-${pnm}-${nm} # execute command in _cmd_
  if [ $? != 0 ]; then echo $fnm: gen_regions failed.; return 1; fi

  echo $fnm: success
  return 0; 
}

#-------------------
#---  Generate region files for unsupervised embedding training. 
#-------------------
dpcnn_gen_regions_unsemb () {
  local fnm=dpcnn_gen_regions_unsemb
  if [ "$n" = ""    ]; then echo $fnm: n is missing.;   return 1; fi # max n for n-grams
  if [ "$nm" = ""   ]; then echo $fnm: nm is missing.;  return 1; fi # data name
  if [ "$bb" = ""   ]; then echo $fnm: bb is missing.;  return 1; fi # # of training data batches
  if [ "$p"  = ""   ]; then echo $fnm: p is missing.;   return 1; fi # region size
  if [ "$pnm" = ""  ]; then echo $fnm: pnm is missing.; return 1; fi # used to generate filenames
  if [ "$idir" = "" ]; then echo $fnm: idir is missing; return 1; fi # input data directory
  if [ "$max_jobs" = "" ]; then echo $fnm: max_jobs is missing.; return 1; fi # max # of background processes
  echo $fnm: n=$n nm=$nm bb=$bb p=$p pnm=$pnm idir=$idir max_jobs=$max_jobs

  local _z=u${pnm} # to avoid naming conflict with other processes. 

  #---  vocabulary for X (features)
  local _xvoc=
  if [ "$n" = 1 ]; then  # words
    if [ "$xmax" = "" ]; then xmax=30; fi  # vocabulary size: default 30K
    echo $fnm: xmax=$xmax
    _xvoc=${tmpdir}/${nm}${_z}-${trnkw}-${pnm}-${xmax}k.vocab
    _dpcnn_gen_vocab $_xvoc $xmax
    if [ $? != 0 ]; then echo $fnm: _dpcnn_gen_vocab failed.; return 1; fi
  elif [ "$n" = 3 ]; then # word {1,2,3}-grams
    if [ "$x3max" = "" ]; then x3max=200; fi # vocabulary size: default 200K
    echo $fnm: x3max=$x3max
    _xvoc=${tmpdir}/${nm}${_z}-${trnkw}-${pnm}-${x3max}k.vocab
    _dpcnn_gen_vocab3 $max_jobs $_xvoc $x3max $_z
    if [ $? != 0 ]; then echo $fnm: _dpcnn_gen_vocab3 failed.; return 1; fi
  else 
    echo $fnm: n must be either 1 or 3.; return 1
  fi

  #---  save X voc for later use. 
  cp -p $_xvoc ${tmpdir}/${nm}-${pnm}.wmap  # Used by dpcnn_regions_side later. 

  #---  vocabulary for Y (target)
  local _yvoc=
  if [ "$ymax" = "" ]; then ymax=30; fi # vocabulary size: default 30k
  echo $fnm: ymax=$ymax
  _yvoc=${tmpdir}/${nm}${_z}-${trnkw}-uns-minstop-${ymax}k.vocab
  _dpcnn_gen_vocab $_yvoc $ymax data/minstop.txt
  if [ $? != 0 ]; then echo $fnm: _dpcnn_gen_vocab failed.; return 1; fi

  #---  Generate region files
  local _opt2=; if [ "$p" = 9 ]; then _opt2=MergeLeftRight; fi 
  local _u_rnm=${tmpdir}/${nm}-${trnkw}-uns-${pnm} # training data filename for unsupervised training
  local _u_pm="$options x_type=Bow x_vocab_fn=$_xvoc y_vocab_fn=$_yvoc patch_size=$p padding=$(((p-1)/2)) \
               x_ext=.xsmatbc y_ext=.ysmatbc dist=$p $_opt2"

  _cmd_=(); local num=0
  local _no; for (( _no=1; _no<=bb; _no++ )); do
    local _bid=${_no}of${bb}
    local _inp_fn=${idir}/${nm}-${trnkw}.${_bid}.txt.tok

    #---  training data for unsupervised training 
    _return0_if_all_exist__uns $_u_rnm $_bid
    if [ $? != 0 ]; then
      _cmd_[$num]="$prep_exe gen_regions_unsup input_fn=$_inp_fn region_fn_stem=$_u_rnm $_u_pm batch_id=$_bid"
      num=$((num+1))
    fi
  done
  do_jobs $max_jobs $num genR4u-${pnm}-${nm} # execute commands in _cmd_
  if [ $? != 0 ]; then echo $fnm: gen_regions for $dvkw and $tstkw failed.; return 1; fi

  echo $fnm: success
  return 0
}

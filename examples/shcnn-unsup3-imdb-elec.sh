#!/bin/bash
  #####
  #####  IMDB: unsup3-tv.  Embedding learning with unsupervised target and sparse region vectors of bag-of-1-3grams. 
  #####
  #####  Step 1. Learning embedding from unlabeled data using unsupervised target and sparse region vectors of bag-of-1-3grams. 
  #####    Step 1.1. Generate input files.  Target: neighboring regions. 
  #####    Step 1.2. training. 
  #####  Step 2. Supervised training using the trained embedding to produce additional input. 
  #####    Step 2.1. Generate input files. 
  #####    Step 2.2. Training. 
  #####
  #####  NOTE: Unlabeled data is downloaded by find_file in sh-common.sh. 
  #####

  #-----------------#
  gpu=-1  # <= change this to, e.g., "gpu=0" to use a specific GPU. 
  source sh-common.sh
  #-----------------#

  nm=imdb
# nm=elec  # <= Uncomment this to train/tset on Elec

  dim=100 # <= Dimensionality of unsupervised embedding. 
  options="LowerCase UTF8"
  z=2 # to avoid name conflict with other scripts

  #***  Step 0. Download unlabeled data token files.  
  uns_lst=${tmpdir}/${nm}${z}-trnuns.lst # unlabeled data
  udir=for-semi # <= Where unlabeled data is.  Downloaded if udir=for-semi. 
  if [ "$nm" = "imdb" ]; then
    echo data/imdb-train.txt.tok    >  $uns_lst
    echo ${udir}/imdb-unlab.txt.tok >> $uns_lst
    find_file $udir imdb-unlab.txt.tok; if [ $? != 0 ]; then echo $shnm: find_file failed.; exit 1; fi   
  elif [ "$nm" = "elec" ]; then
    echo ${udir}/elec-25k-unlab00.txt.tok  > $uns_lst
    echo ${udir}/elec-25k-unlab01.txt.tok >> $uns_lst
    find_file $udir elec-25k-unlab00.txt.tok; if [ $? != 0 ]; then echo $shnm: find_file failed.; exit 1; fi 
    find_file $udir elec-25k-unlab01.txt.tok; if [ $? != 0 ]; then echo $shnm: find_file failed.; exit 1; fi     
  else 
    echo Unexpected dataset name: $nm; exit 1
  fi
  
  #***  Step 1. Learning embedding from unlabeled data using unsupervised target and region vectors of bag-of-1-3grams. 
  #---  Step 1.1. Generate input files.

  #---  vocabulary for Y (target)
  yszk=30
  yvoc=${tmpdir}/${nm}${z}-minstop-uns.vocab
  stop_fn=data/minstop.txt # function words
  $prep_exe gen_vocab input_fn=$uns_lst vocab_fn=$yvoc $options WriteCount stopword_fn=$stop_fn \
                  max_vocab_size=${yszk}000
  if [ $? != 0 ]; then echo $shnm: gen_vocab failed.; exit 1; fi

  #---  vocabulary for X (regions)
  fns=
  for nn in 1 2 3; do
    xvoc=${tmpdir}/${nm}${z}-trn-${nn}gram.vocab
    $prep_exe gen_vocab input_fn=data/${nm}-train.txt.tok vocab_fn=$xvoc $options WriteCount \
                  n=$nn
    if [ $? != 0 ]; then echo $shnm: gen_vocab failed.; exit 1; fi
    #---  NOTE: With larger unlabeled data, try input_fn=$uns_lst. 
    fns="$fns $xvoc"
  done
  xszk=30 # NOTE: With larger unlabeled data, try xszk=200. 
  xvoc=${tmpdir}/${nm}${z}-trn-123gram-${xszk}k.vocab
  echo Merging and sorting 1-3grams and retaining the most frequent ${xszk}000 ... 
  perl merge_dic.pl ${xszk}000 $fns > $xvoc
  if [ $? != 0 ]; then echo $shnm: merge_dic.pl failed.; exit 1; fi

  #---  Generate a region file and a target file. 
  pch_sz=5; pch_step=1; padding=$((pch_sz-1))
  dist=$pch_sz

  unsnm=${nm}-unsx3-p${pch_sz}
  $prep_exe gen_regions_unsup x_ext=.xsmatbc y_ext=.ysmatbc \
                   x_type=Bow input_fn=$uns_lst \
                   $opt2 x_vocab_fn=$xvoc y_vocab_fn=$yvoc region_fn_stem=${tmpdir}/${unsnm} $options \
                   patch_size=$pch_sz patch_stride=$pch_step padding=$padding dist=$dist 
  if [ $? != 0 ]; then echo $shnm: gen_regions failed.; exit 1; fi

  #---  Step 1.2. Embedding training. 
  epo1=8; epo=10
  lay0_fn=${outdir}/${unsnm}.dim${dim}
  logfn=${logdir}/${unsnm}-dim${dim}.log
  echo 
  echo Training embedding with unsupervised target.  
  echo This takes a while.  See $logfn for progress. 
  #
  # NOTE: To make it faster, try "momentum=0 step_size=5". 
  #
  $exe $gpu:1 train trnname=$unsnm data_dir=$tmpdir 0save_layer_fn=$lay0_fn \
        datatype=sparse x_ext=.xsmatbc y_ext=.ysmatbc NoCusparseIndex \
        step_size=0.5 ss_scheduler=Few ss_decay=0.1 ss_decay_at=$epo1 num_epochs=$epo \
        NoTest Regression loss=Square mini_batch_size=100 momentum=0.9 reg_L2=0 random_seed=1 \
        zero_Y_weight=0.2 zero_Y_ratio=5 \
        layers=1 0layer_type=Weight+ 0nodes=$dim 0activ_type=Rect 0resnorm_type=Text  \
         inc=500000 \
        > $logfn
  if [ $? != 0 ]; then echo $shnm: Embedding training failed.; exit 1; fi

  rm -f ${tmpdir}/${unsnm}*

  #***  Step 2. Supervised training using the trained embedding to produce additional input. 
  #---  Step 2.1. Generate input files.
  xvoc1=${tmpdir}/${nm}${z}-trn-${xszk}k.vocab
  $prep_exe gen_vocab input_fn=data/${nm}-train.txt.tok vocab_fn=$xvoc1 $options WriteCount max_vocab_size=${xszk}000
  for set in train test; do 
    opt=NoSkip
    #---  dataset#0: for the main layer (seq)
    rnm=${tmpdir}/${nm}${z}-${set}-p${pch_sz}seq
    $prep_exe gen_regions $opt $options patch_size=$pch_sz padding=$((pch_sz-1)) \
      region_fn_stem=$rnm input_fn=data/${nm}-${set} vocab_fn=$xvoc1 label_dic_fn=data/${nm}_cat.dic
    if [ $? != 0 ]; then echo $shnm: gen_regions failed.; exit 1; fi

    #---  dataset#1: for the side layer (bag-of-1-3grams)
    rnm=${tmpdir}/${nm}${z}-${set}-p${pch_sz}x3bow
    $prep_exe gen_regions $opt Bow $options RegionOnly patch_size=$pch_sz padding=$((pch_sz-1)) \
      region_fn_stem=$rnm input_fn=data/${nm}-${set} vocab_fn=$xvoc
      
    if [ $? != 0 ]; then echo $shnm: gen_regions failed.; exit 1; fi
  done

  #---  Step 2.2. Training 
  mynm=shcnn-unsup3-${nm}-dim${dim}
  logfn=${logdir}/${mynm}.log
  csvfn=${csvdir}/${mynm}.csv
  s_fn=${lay0_fn}.epo${epo}.ReLayer0
  nodes=1000
  echo 
  echo Supervised training using unsupervised embedding to produce additional input.   
  echo This takes a while.  See $logfn and $csvfn for progress.
  $exe $gpu:4 train V2 datatype=sparse \
        trnname=${nm}${z}-train-p${pch_sz} tstname=${nm}${z}-test-p${pch_sz} \
        dsno0=seq dsno1=x3bow data_dir=$tmpdir \
        loss=Square reg_L2=1e-4 step_size=0.1 top_dropout=0.5 \
        mini_batch_size=100 momentum=0.9 random_seed=1 \
        num_epochs=100 ss_scheduler=Few ss_decay=0.1 ss_decay_at=80 \
        layers=2 0layer_type=WeightS+ 0nodes=$nodes 0activ_type=Rect 0dsno=0 \
        num_sides=1 0side0_layer_type=Weight+ 0side0_layer_fn=$s_fn 0side0_dsno=1 \
        1layer_type=Pooling 1pooling_type=Max 1num_pooling=1 1resnorm_type=Text  \
        evaluation_fn=$csvfn test_interval=25  \
        > $logfn
  if [ $? != 0 ]; then echo $shnm: training failed.; exit 1; fi

  rm -f ${tmpdir}/${nm}${z}*
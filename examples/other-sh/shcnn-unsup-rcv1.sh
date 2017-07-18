#!/bin/bash
  #####  
  #####  RCV1: unsup-tv.  Embedding learning with unsupervised target. 
  #####
  #####  Step 1. Learning embedding from unlabeled data using unsupervised target. 
  #####    Step 1.1. Generate input files.  Target: neighboring regions. . 
  #####    Step 1.2. training. 
  #####  Step 2. Supervised training using the trained embedding to produce additional input. 
  #####    Step 2.1. Generate input files. 
  #####    Step 2.2. Training. 

  #-----------------#
  gpu=-1  # <= change this to, e.g., "gpu=0" to use a specific GPU. 
  source sh-common.sh
  #-----------------#
  rcv1dir=rcv1_data  # <= change this to the directory where your RCV1 data is. 
  rcv1udir=rcv1_unlab_data  # <= change this to the directory where RCV1 unlabeled data is. 
           #
           #    NOTE: We cannot provide RCV1 unlabeled data due to the copyright issue.  
           #
  dim=100 # <= change this to change dimensionality of unsupervised embedding. 
  options="LowerCase UTF8"
  z=a # to avoid name conflict with other scripts

  #***  Step 1. Learning embedding from unlabeled data using unsupervised target. 
  #---  Step 1.1. Generate input files. 
  uns_lst=${tmpdir}/rcv1${z}-uns.lst
  rm -f $uns_lst
  for ym in 609 610 611 612 701 702 703 704 705 706; do
    echo ${rcv1udir}/rcv1-199${ym}.txt.tok >> $uns_lst
  done

  #---  vocabulary for Y (target)
  yszk=30
  yvoc=${tmpdir}/rcv1${z}-stop-uns.vocab
  stop_fn=data/rcv1_stopword.txt
  $prep_exe gen_vocab input_fn=$uns_lst vocab_fn=$yvoc $options WriteCount stopword_fn=$stop_fn \
                  max_vocab_size=${yszk}000 RemoveNumbers
  if [ $? != 0 ]; then echo $shnm: gen_vocab failed.; exit 1; fi

  #---  vocabulary for X (regions)
  xszk=30
  xvoc=${tmpdir}/rcv1${z}-trn.vocab
  $prep_exe gen_vocab input_fn=${rcv1dir}/rcv1-1m-train.txt.tok vocab_fn=$xvoc $options WriteCount \
                  max_vocab_size=${xszk}000
  if [ $? != 0 ]; then echo $shnm: gen_vocab failed.; exit 1; fi

  #---  split text into 5 batches 
  batches=5
  $prep_exe split_text input_fn=$uns_lst num_batches=$batches random_seed=1 output_fn_stem=${tmpdir}/rcv1${z}-uns
  if [ $? != 0 ]; then echo $shnm: split_text failed.; exit 1; fi

  #---  Generate region files and target files of 5 batches. 
  pch_sz=20; pch_step=2; padding=$((pch_sz-1))
  dist=$pch_sz
  nm=rcv1${z}-uns-p${pch_sz}
  for no in 1 2 3 4 5; do 
    batch_id=${no}of${batches}
    inp_fn=${tmpdir}/rcv1${z}-uns.${batch_id}
    $prep_exe gen_regions_unsup x_ext=.xsmatbc y_ext=.ysmatbc x_type=Bow input_fn=$inp_fn \
                   $opt2 x_vocab_fn=$xvoc y_vocab_fn=$yvoc region_fn_stem=${tmpdir}/${nm} $options \
                   patch_size=$pch_sz patch_stride=$pch_step padding=$padding dist=$dist \
                   MergeLeftRight batch_id=$batch_id
    if [ $? != 0 ]; then echo $shnm: gen_regions failed.; exit 1; fi
  done

  #---  Step 1.2. Unsupervised embedding training. 
  epo1=8; epo=10  # "epo1=4; epo=5" is good enough, though. 
  lay0_fn=${outdir}/${nm}.dim${dim}
  logfn=${logdir}/rcv1-uns-dim${dim}.log
  echo 
  echo Training embedding with unsupervised target.  
  echo This takes a while.  See $logfn for progress. 
  # NOTE: To make it faster, try "momentum=0 step_size=5".
  $exe $gpu:1 train num_batches=$batches trnname=$nm data_dir=$tmpdir 0save_layer_fn=$lay0_fn \
        datatype=sparse x_ext=.xsmatbc y_ext=.ysmatbc NoCusparseIndex \
        step_size=0.5 ss_scheduler=Few ss_decay=0.1 ss_decay_at=$epo1 num_epochs=$epo \
        NoTest Regression loss=Square mini_batch_size=100 momentum=0.9 reg_L2=0 random_seed=1 \
        zero_Y_weight=0.2 zero_Y_ratio=5 \
        layers=1 0layer_type=Weight+ 0nodes=$dim 0activ_type=Rect 0resnorm_type=Text  \
         inc=500000 \
        > $logfn
  if [ $? != 0 ]; then echo $shnm: Embedding training failed.; exit 1; fi
  rm -f ${tmpdir}/${nm}.*
  rm -f ${tmpdir}/rcv1${z}-uns*

  #***  Step 2. Supervised training using the trained embedding to produce additional input. 
  #---  Step 2.1. Generate input files.
  for set in train test; do 
    rnm=${tmpdir}/rcv1${z}-1m-${set}-p${pch_sz}
    $prep_exe gen_regions Bow VariableStride \
      region_fn_stem=$rnm input_fn=${rcv1dir}/rcv1-1m-${set} vocab_fn=$xvoc \
      $options label_fn_ext=.lvl2 \
      label_dic_fn=data/rcv1-lvl2.catdic \
      patch_size=$pch_sz patch_stride=2 padding=$((pch_sz-1))
    if [ $? != 0 ]; then echo $shnm: gen_regions failed.; exit 1; fi
  done

  #---  Step 2.2. Training 
  gpumem=${gpu}:4  # pre-allocate 4GB GPU memory. 
  mynm=shcnn-unsup-rcv1-dim${dim}
  logfn=${logdir}/${mynm}.log
  csvfn=${csvdir}/${mynm}.csv
  s_fn=${lay0_fn}.epo${epo}.ReLayer0
  stepsize=0.5; nodes=1000
  echo 
  echo Supervised training using unsupervised embedding to produce additional input.   
  echo This takes a while.  See $logfn and $csvfn for progress. 
  $exe $gpumem train V2 \
        trnname=rcv1${z}-1m-train-p${pch_sz} tstname=rcv1${z}-1m-test-p${pch_sz} data_dir=$tmpdir \
        datatype=sparse \
        reg_L2=1e-4 step_size=$stepsize \
        mini_batch_size=100 momentum=0.9 random_seed=1 \
        num_epochs=100 ss_scheduler=Few ss_decay=0.1 ss_decay_at=80 \
        loss=Square \
        layers=2 0layer_type=WeightS+ 0nodes=$nodes 0activ_type=Rect \
        num_sides=1 0side0_layer_type=Weight+ 0side0_layer_fn=$s_fn \
        0pooling_type=Avg 0num_pooling=10 0resnorm_type=Text  \
        1layer_type=Patch 1patch_size=10 \
        evaluation_fn=$csvfn test_interval=25  \
        > $logfn
  if [ $? != 0 ]; then echo $shnm: training failed.; exit 1; fi
  rm -f ${tmpdir}/rcv1${z}*

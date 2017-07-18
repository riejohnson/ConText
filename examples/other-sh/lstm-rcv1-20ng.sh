#!/bin/bash
  #---  RCV1: Supervised LSTM 
  #-----------------#
  gpu=-1  # <= change this to, e.g., "gpu=0" to use a specific GPU. 
  mem=4   # pre-allocate 2GB device memory 
  source sh-common.sh
  #-----------------#  
  nm=rcv1-1m; ldir=rcv1_data; epo=25; catdic=data/rcv1-lvl2.catdic; lab_ext=.lvl2; pl=Avg
# nm=20ng;    ldir=20ng_data; epo=50; catdic=${ldir}/20ng.catdic;   lab_ext=.cat;  pl=Max; prepopt=NoSkip

  options="LowerCase UTF8"
  txt_ext=.txt.tok

  z=l1 # to avoid name conflict with other scripts

  #---  vocabulary
  xvoc=${tmpdir}/${nm}${z}-trn.vocab
  $prep_exe gen_vocab input_fn=${ldir}/${nm}-train vocab_fn=$xvoc $options WriteCount text_fn_ext=$txt_ext max_vocab_size=30000
  if [ $? != 0 ]; then echo $shnm: gen_vocab failed.; exit 1; fi

  #---  Generate input data for the final supervised training
  for set in train test; do 
    rnm=${tmpdir}/${nm}${z}-${set}-p1
    $prep_exe gen_regions region_fn_stem=$rnm input_fn=${ldir}/${nm}-${set} vocab_fn=$xvoc \
        $options text_fn_ext=$txt_ext label_fn_ext=$lab_ext \
        label_dic_fn=$catdic $prepopt \
        patch_size=1 patch_stride=1 padding=0
    if [ $? != 0 ]; then echo $shnm: gen_regions failed.; exit 1; fi
  done

  #---  Training
  mynm=lstm-${nm}
  logfn=${logdir}/${mynm}.log
  csvfn=${csvdir}/${mynm}.csv
  echo Training ... see $logfn and $csvfn 
  num_pool=10
  $exe $gpu:$mem train  optim=Rmsp NoGate_i NoGate_o top_dropout=0.5 \
              max_loss=5 inc=5000 trnname=${nm}${z}-train-p1 tstname=${nm}${z}-test-p1 data_dir=$tmpdir \
              test_interval=25 reg_L2=0 step_size=0.25e-3 evaluation_fn=$csvfn \
              layers=3 loss=Square mini_batch_size=50 test_mini_batch_size=500 random_seed=1 \
              datatype=sparse \
              num_epochs=$epo ss_scheduler=Few ss_decay=0.1 ss_decay_at=$((epo*4/5)) \
              0layer_type=Lstm2 0nodes=500 0chop_size=50 \
              1layer_type=Pooling 1num_pooling=$num_pool 1pooling_type=$pl \
              2layer_type=Patch 2patch_size=$num_pool > $logfn
  if [ $? != 0 ]; then echo $shnm: training failed.; exit 1; fi

  rm -f ${tmpdir}/${nm}${z}*
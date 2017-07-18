#!/bin/bash
  #---  LSTM 
  #-----------------#
  gpu=-1  # <= change this to, e.g., "gpu=0" to use a specific GPU. 
  mem=4   # pre-allocate 2GB device memory 
  source sh-common.sh
  #-----------------#  
  nm=imdb; ite=100; ite1=80; mb=50
# nm=elec; ite=50; ite1=40; mb=100  # <= Uncomment this to train/test on Elec.

  ddir=data 
  options="LowerCase UTF8"
  catdic=${ddir}/${nm}_cat.dic
  z=l1 # to avoid name conflict with other scripts

  #---  vocabulary
  xvoc=${tmpdir}/${nm}${z}-trn.vocab
  $prep_exe gen_vocab $options input_fn=${ddir}/${nm}-train.txt.tok vocab_fn=$xvoc WriteCount max_vocab_size=30000
  if [ $? != 0 ]; then echo $shnm: gen_vocab failed.; exit 1; fi

  #---  Generate input data for the final supervised training
  for set in train test; do 
    rnm=${tmpdir}/${nm}${z}-${set}-p1
    $prep_exe gen_regions $options NoSkip region_fn_stem=$rnm \
        input_fn=${ddir}/${nm}-${set} vocab_fn=$xvoc label_dic_fn=$catdic \
        patch_size=1
    if [ $? != 0 ]; then echo $shnm: gen_regions failed.; exit 1; fi
  done

  #---  Training
  mynm=lstm-${nm}
  logfn=${logdir}/${mynm}.log; csvfn=${csvdir}/${mynm}.csv
  echo; echo Training ... see $logfn and $csvfn 
  $exe $gpu:$mem train  NoGate_i NoGate_o top_dropout=0.5 test_mini_batch_size=500 \
              max_loss=5 inc=5000 trnname=${nm}${z}-train-p1 tstname=${nm}${z}-test-p1 data_dir=$tmpdir \
              test_interval=25 reg_L2=0 step_size=1 evaluation_fn=$csvfn \
              layers=2 loss=Square mini_batch_size=$mb momentum=0.9 random_seed=1 \
              datatype=sparse \
              num_epochs=$ite ss_scheduler=Few ss_decay=0.1 ss_decay_at=$ite1 \
              0layer_type=Lstm2 0nodes=500 0chop_size=50 \
              1layer_type=Pooling 1num_pooling=1 1pooling_type=Max > $logfn
  if [ $? != 0 ]; then echo $shnm: training failed.; exit 1; fi

  rm -f ${tmpdir}/${nm}${z}*
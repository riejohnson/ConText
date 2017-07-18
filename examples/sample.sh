#!/bin/bash

  gpu=-1  # <= change this to, e.g., "gpu=2" to use a specific GPU. 
  source sh-common.sh
  
  options=LowerCase  # This must be shared by gen_vocab and gen_regions
  
  #---  Step 1. Generate vocabulary
  echo Generaing a vocabulary file from training data ... 
  vocab_fn=${tmpdir}/s.voc
  $prep_exe gen_vocab $options input_fn=data/s-train.txt.tok vocab_fn=$vocab_fn max_vocab_size=30000
  if [ $? != 0 ]; then echo $shnm: gen_vocab failed.; exit 1; fi

  #---  Step 2. Generate input files for training and testing.  
  echo; echo Generating input files for training/testing ... 
  p=3 # region size
  for set in train test; do 
    rnm=${tmpdir}/s-${set}-p${p}
    $prep_exe gen_regions $options region_fn_stem=$rnm \
        input_fn=data/s-${set} vocab_fn=$vocab_fn label_dic_fn=data/s-cat.dic \
        patch_size=$p padding=$(((p-1)/2))
    if [ $? != 0 ]; then echo $shnm: gen_regions failed.; exit 1; fi
  done

  #---  Step 3. Training and testing using GPU
  echo; echo Training and testing ... See ${csvdir}/sample.csv 
  epo=20  # number of epochs
  ss=0.25 # step-size (learning rate) 
  $exe $gpu train layers=1 0layer_type=Weight+ 0nodes=500 0activ_type=Rect \
     0pooling_type=Max 0num_pooling=1 top_dropout=0.5 \
     loss=Square mini_batch_size=100 momentum=0.9 reg_L2=1e-4 step_size=$ss \
     datatype=sparse trnname=s-train-p${p} tstname=s-test-p${p} data_dir=$tmpdir \
     num_epochs=$epo ss_scheduler=Few ss_decay=0.1 ss_decay_at=$((epo*4/5)) \
     test_interval=5 random_seed=1 evaluation_fn=${csvdir}/sample.csv
  if [ $? != 0 ]; then echo $shnm: train failed.; exit 1; fi

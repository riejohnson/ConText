#!/bin/bash
  #----  Example of multi-label categorization. 
  #----  Input: token files (data/s-multilab-{train|test}.txt.tok) 
  #----         label files (data/s-multilab-{train|test}.cat)
  #----         label dictionary (data/s-multilab.catdic)

  gpu=-1  # <= change this to, e.g., "gpu=0" to use a specific GPU. 
  source sh-common.sh

  nm=s-multilab

  #---  Step 1. Generate vocabulary
  echo Generaing vocabulary from training data ... 
  vocab_fn=${tmpdir}/s-multilab.voc
  options=LowerCase
  
  $prep_exe gen_vocab input_fn=data/${nm}-train.txt.tok vocab_fn=$vocab_fn $options max_vocab_size=30000
  if [ $? != 0 ]; then echo $shnm: gen_vocab failed.; exit 1; fi

  #---  Step 2. Generate input files for training and testing.  
  echo; echo Generating region files ... 
  p=3  # region size
  pnm=p${p}
  for set in train test; do 
    rnm=${tmpdir}/${nm}-${set}-${pnm}
    $prep_exe gen_regions MultiLabel region_fn_stem=$rnm \
        input_fn=data/${nm}-${set} vocab_fn=$vocab_fn $options label_dic_fn=data/${nm}.catdic \
        patch_size=$p padding=$(((p-1)/2))
    if [ $? != 0 ]; then echo $shnm: gen_regions failed.; exit 1; fi
  done

  #---  Step 3. Training and test using GPU
  echo; echo Training with multi-label data ... 
  $exe $gpu train MultiLabel test_interval=5 random_seed=1 \
    datatype=sparse trnname=${nm}-train-${pnm} tstname=${nm}-test-${pnm} data_dir=$tmpdir \
    layers=1 0layer_type=Weight+ 0nodes=500 0activ_type=Rect 0pooling_type=Max 0num_pooling=1 \
    loss=BinLogi2 mini_batch_size=100 momentum=0.9 reg_L2=1e-4 step_size=0.25 top_dropout=0.5 \
    num_epochs=20 ss_scheduler=Few ss_decay=0.1 ss_decay_at=15
  if [ $? != 0 ]; then echo $shnm: training failed.; exit 1; fi


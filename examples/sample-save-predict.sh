#!/bin/bash
  #--- Example of saving a model and using 'predict'

  gpu=-1  # <= change this to, e.g., "gpu=0" to use a specific GPU. 
  source sh-common.sh 
  options=LowerCase  # This must be shared by gen_vocab and gen_regions. 
  
  #---  Step 1. Generate vocabulary
  echo Generaing a vocabulary file from training data ... 
  vocab_fn=${tmpdir}/s.voc
  $prep_exe gen_vocab $options input_fn=data/s-train.txt.tok vocab_fn=$vocab_fn max_vocab_size=30000
  if [ $? != 0 ]; then echo $shnm: gen_vocab failed.; exit 1; fi

  #---  Step 2. Generate input files for training and testing.  
  echo; echo Generating region files ... 
  p=3 # region size
  pnm=p${p}  # used in filenames
  for set in train test; do 
    rnm=${tmpdir}/s-${set}-${pnm}
    $prep_exe gen_regions $options region_fn_stem=$rnm \
       input_fn=data/s-${set} vocab_fn=$vocab_fn label_dic_fn=data/s-cat.dic \
       patch_size=$p padding=$(((p-1)/2))
    if [ $? != 0 ]; then echo $shnm: gen_regions failed.; exit 1; fi
  done

  #---  Step 3. Training and testing using GPU
  logfile=${logdir}/sample-save.log
  echo; echo Training and saving the models
  epo=20  # number of epochs 
  ss=0.25 # step-size (learning rate)
  save_fn=${outdir}/s-train-${pnm}.mod  # model filenames are generated from this. 
  $exe $gpu train save_interval=10 save_fn=$save_fn \
      test_interval=5 random_seed=1 \
      datatype=sparse trnname=s-train-${pnm} tstname=s-test-${pnm} data_dir=$tmpdir \
      layers=1 0layer_type=Weight+ 0nodes=500 0activ_type=Rect 0pooling_type=Max 0num_pooling=1 \
      loss=Square mini_batch_size=100 momentum=0.9 reg_L2=1e-4 step_size=$ss top_dropout=0.5 \
      num_epochs=$epo ss_scheduler=Few ss_decay=0.1 ss_decay_at=$((epo*4/5))
  if [ $? != 0 ]; then echo $shnm: train failed.; exit 1; fi

  #---  Step 4. Apply the saved model to test data
  mod_fn=${save_fn}.epo${epo}.ReNet  # path to the model file saved after $epo epochs.   
  #---  binary output (faster)
  $exe $gpu predict model_fn=$mod_fn \
      prediction_fn=${outdir}/s-test-${pnm}.pred.bin \
      datatype=sparse tstname=s-test-${pnm} data_dir=$tmpdir
  if [ $? != 0 ]; then echo $shnm: predict failed.; exit 1; fi

  #---  text output (slower)
  $exe $gpu predict model_fn=$mod_fn \
    prediction_fn=${outdir}/s-test-${pnm}.pred.txt WriteText \
    datatype=sparse tstname=s-test-${pnm} data_dir=$tmpdir
  if [ $? != 0 ]; then echo $shnm: predict failed.; exit 1; fi

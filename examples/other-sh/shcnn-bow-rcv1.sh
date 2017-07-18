#!/bin/bash
  ####  Input: token file (one article per line; tokens are delimited by white space): *.txt.tok
  ####         label file (one label per line): *.cat
  ####  The input files are not included in the package due to copyright.  

  #-----------------#
  gpu=-1  # <= change this to, e.g., "gpu=0" to use a specific GPU. 
  mem=2   # pre-allocate 2GB device memory 
  source sh-common.sh
  #-----------------#
  inpdir=rcv1_data  # <= change this to where rcv1 data is. 
  z=bo # to avoid name conflict with other scripts 

  #---  Step 1. Generate vocabulary
  echo Generaing vocabulary from training data ... 
  max_num=30000
  vocab_fn=${tmpdir}/rcv1${z}_trn-${max_num}.vocab
  options="LowerCase UTF8"
  
  $prep_exe gen_vocab input_fn=${inpdir}/rcv1-1m-train.txt.tok vocab_fn=$vocab_fn max_vocab_size=$max_num \
                            $options WriteCount RemoveNumbers stopword_fn=data/rcv1_stopword.txt
  if [ $? != 0 ]; then echo $shnm: gen_vocab failed.; exit 1; fi

  #---  Step 2. Generate region files (${tmpdir}/*.xsmatbcvar), target files (${tmpdir}/*.y) 
  #---          and word-mapping files (${tmpdir}/*.xtext).    
  echo; echo Generating region files ... 
  p=20
  for set in train test; do 
    rnm=${tmpdir}/rcv1${z}_${set}-p${p}
    $prep_exe gen_regions Bow VariableStride region_fn_stem=$rnm \
        input_fn=${inpdir}/rcv1-1m-${set} vocab_fn=$vocab_fn \
        $options label_fn_ext=.lvl2 label_dic_fn=data/rcv1-lvl2.catdic \
        patch_size=$p patch_stride=2 padding=$((p-1))
    if [ $? != 0 ]; then echo $shnm: gen_regions failed.; exit 1; fi
  done


  #---  Step 3. Training and test using GPU
  mynm=shcnn-bow-rcv1
  log_fn=${logdir}/${mynm}.log; csv_fn=${csvdir}/${mynm}.csv
  echo; echo Training and testing ... ; echo This takes a while.  See $log_fn and $csv_fn for progress. 
  plnum=10 # number of pooling units.
  $exe $gpu:$mem train datatype=sparse \
         data_dir=$tmpdir trnname=rcv1${z}_train-p${p} tstname=rcv1${z}_test-p${p} \
         save_fn=${outdir}/${mynm}-mod save_interval=100 \
         loss=Square num_epochs=100 \
         reg_L2=1e-4 momentum=0.9 mini_batch_size=100 random_seed=1 \
         step_size=0.5 ss_scheduler=Few ss_decay=0.1 ss_decay_at=80 \
         layers=2 0layer_type=Weight+ 0nodes=1000 0activ_type=Rect \
         0pooling_type=Avg 0num_pooling=$plnum 0resnorm_type=Text  \
         1layer_type=Patch 1patch_size=$plnum \
         test_interval=25 evaluation_fn=$csv_fn > ${log_fn}
  if [ $? != 0 ]; then echo $shnm: training failed.; exit 1; fi

  rm -f ${tmpdir}/rcv1${z}*

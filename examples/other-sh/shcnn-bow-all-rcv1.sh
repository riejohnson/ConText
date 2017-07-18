#!/bin/bash
  ####  Input: token file (one article per line; tokens are delimited by white space) 
  ####         label file (one label per line)
  ####  The input files are not included in the package due to copyright.  

  #-----------------#
  gpu=-1  # <= change this to, e.g., "gpu=0" to use a specific GPU. 
  mem=4   # pre-allocate 4GB device memory 
  source sh-common.sh
  #-----------------#
  inpdir=rcv1_data  # <= change this to where rcv1 data is. 

  z=alsz

for sz in r02k r03k r04k r05k r10k 1m 2m 3m; do
  #---  Step 1. Generate vocabulary
  echo Generaing vocabulary from training data ... 

  max_num=30000
  vocab_fn=${tmpdir}/rcv1${z}-${sz}_trn-${max_num}.vocab
  options="LowerCase UTF8"
  
  $prep_exe gen_vocab input_fn=${inpdir}/rcv1-${sz}-train.txt.tok vocab_fn=$vocab_fn max_vocab_size=$max_num \
                            $options WriteCount RemoveNumbers stopword_fn=data/rcv1_stopword.txt
  if [ $? != 0 ]; then echo $shnm: gen_vocab failed.; exit 1; fi

  #---  Step 2. Generate region files (${tmpdir}/*.xsmatbcvar) and target files (${tmpdir}/*.y) for training and testing CNN.  
  #     We generate region vectors of the convolution layer and write them to a file, instead of making them 
  #     on the fly during training/testing.   
  echo 
  echo Generating region files ... 

  pch_sz=20

  for set in train test; do 
    rnm=${tmpdir}/rcv1${z}-${sz}-${set}-p${pch_sz}
 
    $prep_exe gen_regions \
      Bow VariableStride \
      region_fn_stem=$rnm input_fn=${inpdir}/rcv1-${sz}-${set} vocab_fn=$vocab_fn \
      $options text_fn_ext=.txt.tok label_fn_ext=.lvl2 \
      label_dic_fn=data/rcv1-lvl2.catdic \
      patch_size=$pch_sz patch_stride=2 padding=$((pch_sz-1))
    if [ $? != 0 ]; then echo $shnm: gen_regions failed.; exit 1; fi
  done


  #---  Step 3. Training and test using GPU
  mynm=shcnn-bow-${sz}-rcv1
  log_fn=${logdir}/${mynm}.log
  csv_fn=${csvdir}/${mynm}.csv

  ss=0.5
  lam=1e-4
  if [ "$sz" = "r02k" ] || [ "$sz" = "r03k" ]; then
    lam=1e-3
  fi
  if [ "$sz" = "3m" ]; then
    ss=0.25
  fi

  echo 
  echo Training CNN and testing ... 
  echo This takes a while.  See $log_fn and $perf_fn for progress. 
  nodes=500 # number of feature maps (weight vectors).
  plnum=10
  $exe $gpu:$mem train \
         data_dir=$tmpdir trnname=rcv1${z}-${sz}-train-p${pch_sz} tstname=rcv1${z}-${sz}-test-p${pch_sz} \
         datatype=sparse \
         loss=Square num_epochs=100 \
         reg_L2=$lam momentum=0.9 mini_batch_size=100 random_seed=1 \
         step_size=$ss ss_scheduler=Few ss_decay=0.1 ss_decay_at=80 \
         layers=2 0layer_type=Weight+ 0nodes=$nodes 0activ_type=Rect \
         0pooling_type=Avg 0num_pooling=$plnum 0resnorm_type=Text  \
         1layer_type=Patch 1patch_size=$plnum \
          test_interval=25 evaluation_fn=$csv_fn \
         > ${log_fn}
  if [ $? != 0 ]; then echo $shnm: training failed.; exit 1; fi

  rm -f ${tmpdir}/rcv1${z}-${sz}*
done

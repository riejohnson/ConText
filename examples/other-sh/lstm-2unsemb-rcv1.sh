#!/bin/bash
  #---  RCV1: semi-supervised LSTM
  #---  NOTE: 5GB or more GPU device memory is required. 

  #-----------------#
  gpu=-1  # <= change this to, e.g., "gpu=0" to use a specific GPU. 
  mem=4   # pre-allocate 4GB device memory 
  source sh-common.sh
  #-----------------#
  ddir=data 
  ldir=rcv1_data       # <= change this to where rcv1 labeled data is. 
  udir=rcv1_unlab_data # <= change this to point to unlab_data.tar.gz 

  options="LowerCase UTF8"
  txt_ext=.txt.tok

  catdic=${ddir}/rcv1-lvl2.catdic
  z=l2 # to avoid name conflict with other scripts

#  dim=100; uss=0.25; ss=2.5
  dim=300; uss=0.1; ss=1  # a larger step-size exploded 
  unsepo=50; unsepo1=40; supepo=100; supepo1=80

  suplam=0; suplamtop=0

  #***  Prepare data for training unsupervised embeddings on unlabeled data 
  uns_lst=${tmpdir}/rcv1${z}-uns.lst # unlabeled data
  rm -f $uns_lst
  for ym in 609 610 611 612 701 702 703 704 705 706; do
    echo ${udir}/rcv1-199${ym} >> $uns_lst
  done

  #---  vocabulary for X (features) 
  xvoc=${tmpdir}/rcv1${z}-trn.vocab
  $prep_exe gen_vocab input_fn=${ldir}/rcv1-1m-train vocab_fn=$xvoc $options WriteCount text_fn_ext=$txt_ext max_vocab_size=30000
  if [ $? != 0 ]; then echo $shnm: gen_vocab failed.; exit 1; fi

  #---  vocabulary for Y (target)
  yvoc=${tmpdir}/rcv1${z}-stop-uns.vocab
  stop_fn=${ddir}/rcv1_stopword.txt
  $prep_exe gen_vocab input_fn=$uns_lst vocab_fn=$yvoc $options WriteCount text_fn_ext=$txt_ext max_vocab_size=30000 stopword_fn=$stop_fn RemoveNumbers
  if [ $? != 0 ]; then echo $shnm: gen_vocab failed.; exit 1; fi

  for unslay in LstmF LstmB; do
    #---  Generate input data for unsupervised training
    if [ "$unslay" = "LstmF" ]; then # forward (left to right)
      lr=RightOnly                    # predict the words on the right
      lrkw=F
    fi
    if [ "$unslay" = "LstmB" ]; then # backward (right to left) 
      lr=LeftOnly                     # predict words on the left
      lrkw=B
    fi

    unsnm=rcv1${z}-uns-${lrkw}

    #---  split text into 5 batches  
    batches=5
    $prep_exe split_text input_fn=$uns_lst ext=$txt_ext num_batches=$batches random_seed=1 output_fn_stem=${tmpdir}/rcv1${z}-uns
    if [ $? != 0 ]; then echo $shnm: split_text failed.; exit 1; fi

    #---  Generate input for unsupervised embedding training 
    for no in 1 2 3 4 5; do 
      batch_id=${no}of${batches}
      inp_fn=${tmpdir}/rcv1${z}-uns.${batch_id}
      $prep_exe gen_regions_unsup x_type=Bow input_fn=$inp_fn text_fn_ext=$txt_ext $lr \
                   x_vocab_fn=$xvoc y_vocab_fn=$yvoc region_fn_stem=${tmpdir}/${unsnm} $options \
                   patch_size=1 patch_stride=1 padding=0 dist=20 \
                   x_ext=.xsmatbcvar y_ext=.ysmatbcvar \
                   MergeLeftRight batch_id=$batch_id
      if [ $? != 0 ]; then echo $shnm: gen_regions failed.; exit 1; fi
    done
    echo waiting ... 
    wait
    echo done ...

    #---  Training unsupervised embeddings on unlabeled data 
    mynm=rcv1-Lstm${lrkw}-dim${dim}
    lay_fn=${outdir}/${mynm}.lay
    logfn=${logdir}/${mynm}.log
    echo Training unsupervised embedding on unlabeled data ... see $logfn
    $exe $gpu:$mem train  top_CountRegions num_batches=5 NoTest Regression NoCusparseIndex 0save_layer_fn=$lay_fn \
              inc=5000 trnname=$unsnm data_dir=$tmpdir \
              reg_L2=0 step_size=$uss \
              layers=1 loss=Square mini_batch_size=10 momentum=0.9 random_seed=1 \
              datatype=sparse x_ext=.xsmatbcvar y_ext=.ysmatbcvar \
              num_epochs=$unsepo step_size_scheduler=Few step_size_decay=0.1 step_size_decay_at=$unsepo1 \
              nodes=$dim 0layer_type=$unslay zero_Y_ratio=10 zero_Y_weight=1 chop_size=50 > $logfn
    if [ $? != 0 ]; then echo $shnm: training failed.; exit 1; fi
  done  

  #---  Generate region files for the final supervised training
  for set in train test; do 
    rnm=${tmpdir}/rcv1${z}-${set}-p1
    $prep_exe gen_regions Bow VariableStride \
        region_fn_stem=$rnm input_fn=${ldir}/rcv1-1m-${set} vocab_fn=$xvoc \
        $options text_fn_ext=$txt_ext label_fn_ext=.lvl2 \
        label_dic_fn=$catdic \
        patch_size=1 patch_stride=1 padding=0
    if [ $? != 0 ]; then echo $shnm: gen_regions failed.; exit 1; fi
  done

  #---  Training with labeled data 
  fnF=${outdir}/rcv1-LstmF-dim${dim}.lay.epo${unsepo}.ReLayer0
  fnB=${outdir}/rcv1-LstmB-dim${dim}.lay.epo${unsepo}.ReLayer0

  mynm=lstm-2unsemb-rcv1-dim${dim}
  logfn=${logdir}/${mynm}.log
  csvfn=${csvdir}/${mynm}.csv
  echo Training with labeled data ... see $logfn and $csvfn 
  $exe $gpu:$mem train  num_sides=2 0side0_layer_type=LstmF 0side0_layer_fn=$fnF 0side1_layer_type=LstmB 0side1_layer_fn=$fnB \
              NoGate_i NoGate_o top_dropout=0.5 top_reg_L2=$suplamtop test_mini_batch_size=500 \
              max_loss=5 inc=5000 trnname=rcv1${z}-train-p1 tstname=rcv1${z}-test-p1 data_dir=$tmpdir \
              test_interval=25 reg_L2=$suplam step_size=$ss evaluation_fn=$csvfn \
              layers=2 loss=Square mini_batch_size=50 momentum=0.9 random_seed=1 \
              datatype=sparse \
              num_epochs=$supepo ss_scheduler=Few ss_decay=0.1 ss_decay_at=$supepo1 \
              0layer_type=Lstm2 0nodes=500 0chop_size=50 \
              1layer_type=Pooling 1num_pooling=1 1pooling_type=Max > $logfn
  if [ $? != 0 ]; then echo $shnm: training failed.; exit 1; fi

  rm -f ${tmpdir}/rcv1${z}*

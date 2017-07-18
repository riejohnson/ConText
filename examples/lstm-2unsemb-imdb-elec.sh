#!/bin/bash
  #---  LSTM with two unsupervised embeddings (LstmF and LstmB) 

  #-----------------#
  gpu=-1  # <= change this to, e.g., "gpu=0" to use a specific GPU. 
  mem=4   # pre-allocate 4GB device memory 
  source sh-common.sh
  #-----------------#
  nm=imdb; supepo=100; u_stepsize=0.5; mb=50
# nm=elec; supepo=50; u_stepsize=0.25; mb=100  # Uncomment this to train/test on Elec.

  ddir=data 
  udir=for-semi # <= Where unlabeled data is.  Downloaded if udir=for-semi. 

  options="LowerCase UTF8"
  catdic=${ddir}/${nm}_cat.dic
  z=l2 # to avoid name conflict with other scripts
  dim=100    # dimensionality of unsupervised embeddings.
  unsepo=30  # number of epochs for unsupervised embedding training.

  #---  Download unlabeled data. 
  uns_lst=${tmpdir}/${nm}${z}-trnuns.lst # unlabeled data
  if [ "$nm" = "imdb" ]; then
    echo data/imdb-train.txt.tok    >  $uns_lst
    echo ${udir}/imdb-unlab.txt.tok >> $uns_lst 
    find_file $udir imdb-unlab.txt.tok; if [ $? != 0 ]; then echo $shnm: find_file failed.; exit 1; fi   
  elif [ "$nm" = "elec" ]; then
    echo ${udir}/elec-25k-unlab00.txt.tok  >  $uns_lst
    echo ${udir}/elec-25k-unlab01.txt.tok >> $uns_lst
    find_file $udir elec-25k-unlab00.txt.tok; if [ $? != 0 ]; then echo $shnm: find_file failed.; exit 1; fi 
    find_file $udir elec-25k-unlab01.txt.tok; if [ $? != 0 ]; then echo $shnm: find_file failed.; exit 1; fi     
  else 
    echo Unexpected dataset name: $nm; exit 1
  fi

  #***  Embedding learning on unlabeled data 
  #---  vocabulary for X (features) 
  xvoc=${tmpdir}/${nm}${z}-trn.vocab
  $prep_exe gen_vocab input_fn=${ddir}/${nm}-train.txt.tok vocab_fn=$xvoc $options WriteCount max_vocab_size=30000
  if [ $? != 0 ]; then echo $shnm: gen_vocab failed.; exit 1; fi
  #---  NOTE: With larger unlabeled data, try input_fn=$uns_lst and max_vocab_size=100000.  

  #---  vocabulary for Y (target)
  yvoc=${tmpdir}/${nm}${z}-minstop-uns.vocab
  stop_fn=data/minstop.txt   # function words
  $prep_exe gen_vocab $options input_fn=$uns_lst vocab_fn=$yvoc WriteCount max_vocab_size=30000 stopword_fn=$stop_fn
  if [ $? != 0 ]; then echo $shnm: gen_vocab failed.; exit 1; fi

  for unslay in LstmF LstmB; do  # Forward and backward 
    #---  Generate input data for unsupervised embedding learning
    if [ "$unslay" = "LstmF" ]; then # forward (left to right)
      lr=RightOnly                   # predict the words on the right
      lrkw=F
    fi
    if [ "$unslay" = "LstmB" ]; then # backward (right to left) 
      lr=LeftOnly                    # predict words on the left
      lrkw=B
    fi

    unsnm=${nm}${z}-uns-${lrkw}
    $prep_exe gen_regions_unsup $options x_type=Bow input_fn=$uns_lst $lr \
                   x_vocab_fn=$xvoc y_vocab_fn=$yvoc region_fn_stem=${tmpdir}/${unsnm} \
                   patch_size=1 patch_stride=1 padding=0 dist=5 \
                   x_ext=.xsmatbcvar y_ext=.ysmatbcvar NoSkip
    if [ $? != 0 ]; then echo $shnm: gen_regions_unsup failed.; exit 1; fi

    #---  Embedding training on unlabeled data 
    mynm=${nm}-Lstm${lrkw}-dim${dim}
    lay_fn=${outdir}/${mynm}
    logfn=${logdir}/${mynm}.log
    echo Training unsupervised embedding ... see $logfn
    $exe $gpu:$mem train  top_CountRegions NoTest Regression NoCusparseIndex 0save_layer_fn=$lay_fn \
              inc=5000 trnname=$unsnm data_dir=$tmpdir \
              reg_L2=0 step_size=$u_stepsize num_epochs=$unsepo \
              layers=1 loss=Square mini_batch_size=10 momentum=0.9 random_seed=1 \
              datatype=sparse x_ext=.xsmatbcvar y_ext=.ysmatbcvar \
              0layer_type=$unslay 0nodes=$dim 0chop_size=50 \
              zero_Y_ratio=10 zero_Y_weight=1 > $logfn
    if [ $? != 0 ]; then echo $shnm: Embedding training failed.; exit 1; fi
  done  

  #***  Generate input data for the final supervised training
  for set in train test; do 
    rnm=${tmpdir}/${nm}${z}-${set}-p1
    $prep_exe gen_regions $options NoSkip region_fn_stem=$rnm \
        input_fn=${ddir}/${nm}-${set} vocab_fn=$xvoc label_dic_fn=$catdic \
        patch_size=1
    if [ $? != 0 ]; then echo $shnm: gen_regions failed.; exit 1; fi
  done

  #***  Training with labeled data 
  fnF=${outdir}/${nm}-LstmF-dim${dim}.epo${unsepo}.ReLayer0
  fnB=${outdir}/${nm}-LstmB-dim${dim}.epo${unsepo}.ReLayer0

  mynm=lstm-2unsemb-${nm}; logfn=${logdir}/${mynm}.log; csvfn=${csvdir}/${mynm}.csv
  echo; echo Training with labeled data and unsupervised embeddings ... see $logfn and $csvfn
  suplam=0; suplamtop=1e-4  # regularization parameters.
  $exe $gpu:$mem train  num_sides=2 0side0_layer_type=LstmF 0side0_layer_fn=$fnF 0side1_layer_type=LstmB 0side1_layer_fn=$fnB \
              NoGate_i NoGate_o top_dropout=0.5 top_reg_L2=$suplamtop test_mini_batch_size=500 \
              max_loss=5 inc=5000 trnname=${nm}${z}-train-p1 tstname=${nm}${z}-test-p1 data_dir=$tmpdir \
              test_interval=25 reg_L2=$suplam step_size=1 evaluation_fn=$csvfn \
              layers=2 loss=Square mini_batch_size=$mb momentum=0.9 random_seed=1 \
              datatype=sparse \
              num_epochs=$supepo ss_scheduler=Few ss_decay=0.1 ss_decay_at=$((supepo*4/5)) \
              0layer_type=Lstm2 0nodes=500 0chop_size=50 \
              1layer_type=Pooling 1num_pooling=1 1pooling_type=Max > $logfn
  if [ $? != 0 ]; then echo $shnm: training failed.; exit 1; fi

  rm -f ${tmpdir}/${nm}${z}*
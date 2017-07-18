#!/bin/bash
  #---  RCV1: semi-supervised shallow CNN with 2 unsupervised LSTM embeddings 
  #---                                     and 3 unsupervised CNN embeddings.
  #---  NOTE: 5GB or more GPU device memory is required. 

  #-----------------#
  gpu=-1  # <= change this to, e.g., "gpu=0" to use a specific GPU. 
  mem=5   # pre-allocate 4GB device memory 
  source sh-common.sh
  #-----------------#
  ddir=data 
  ldir=rcv1_data   # <= Change this to where RCV1 labeled data is. 
  lstmdir=for-semi # <= Change this to where LSTM embeddings are.  Downloaded if lstmdir=for-semi. 
  cnndir=for-semi  # <= Change this to where CNN embeddings are.   Downloaded if cnndir=for-semi. 
  #####
  ##### WARNING: If your system uses Big Endian (Motorola convention), you cannot use the 
  #####    downloaded files!  They are in the little-endian format (Intel convention)!
  #####

  options="LowerCase UTF8"
  txt_ext=.txt.tok

  catdic=${ddir}/rcv1-lvl2.catdic
  z=3 # to avoid name conflict with other scripts

  dim=300
  unsite=50; supite=100; supite1=80

  #---  Prepare unsupervised embedding files. 
  lay_fn0=rcv1-LstmF-dim${dim}.lay.epo${unsite}.ReLayer0
  lay_fn1=rcv1-LstmB-dim${dim}.lay.epo${unsite}.ReLayer0
  for fn in $lay_fn0 $lay_fn1; do find_file $lstmdir $fn; if [ $? != 0 ]; then echo $shnm: find_file failed.; exit 1; fi; done 
  lay_fn0=${lstmdir}/${lay_fn0}; lay_fn1=${lstmdir}/${lay_fn1}
    
  lay_fn2=rcv1-uns-p20.dim100.epo10.ReLayer0
  lay_fn3=rcv1-unsx3-p20.dim100.epo10.ReLayer0
  lay_fn4=rcv1-parsup-p20p20.dim100.epo10.ReLayer0
  for fn in $lay_fn2 $lay_fn3 $lay_fn4; do find_file $cnndir $fn; if [ $? != 0 ]; then echo $shnm: find_file failed.; exit 1; fi; done
  lay_fn2=${cnndir}/${lay_fn2}; lay_fn3=${cnndir}/${lay_fn3}; lay_fn4=${cnndir}/${lay_fn4}   

  #---  
  voc01=${tmpdir}/rcv1${z}-01.wmap
  $exe $gpu write_word_mapping layer_type=LstmF layer0_fn=$lay_fn0 word_map_fn=$voc01
  if [ $? != 0 ]; then echo $shnm: write_word_mapping failed.; exit 1; fi

  for set in train test; do 
    #---  Generate region files for the LSTM embeddings
    rnm=${tmpdir}/rcv1${z}-${set}-p1
    $prep_exe gen_regions NoSkip \
        region_fn_stem=$rnm input_fn=${ldir}/rcv1-1m-${set} vocab_fn=$voc01 \
        $options text_fn_ext=$txt_ext label_fn_ext=.lvl2 label_dic_fn=$catdic \
        patch_size=1 patch_stride=1 padding=0
    if [ $? != 0 ]; then echo $shnm: gen_regions failed.; exit 1; fi

    #---  Generate region files for layer-0 
    p=21 # b/c we want an odd number here ... 
    rnm=${tmpdir}/rcv1${z}-${set}-p${p}
    $prep_exe gen_regions NoSkip Bow \
        region_fn_stem=$rnm input_fn=${ldir}/rcv1-1m-${set} vocab_fn=$voc01 \
        $options text_fn_ext=$txt_ext RegionOnly \
        patch_size=$p patch_stride=1 padding=$(((p-1)/2))
    if [ $? != 0 ]; then echo $shnm: gen_regions failed.; exit 1; fi
  done

  #---  Generate region files for CNN embeddings 
  wm2=${tmpdir}/rcv1${z}-2.wmap
  $exe $gpu write_word_mapping layer0_fn=$lay_fn2 layer_type=Weight+ word_map_fn=$wm2
  if [ $? != 0 ]; then echo $shnm: write_word_mapping failed.; exit 1; fi

  wm3=${tmpdir}/rcv1${z}-3.wmap
  $exe $gpu write_word_mapping layer0_fn=$lay_fn3 layer_type=Weight+ word_map_fn=$wm3
  if [ $? != 0 ]; then echo $shnm: write_word_mapping failed.; exit 1; fi

  wm4=${tmpdir}/rcv1${z}-4.wmap
  $exe $gpu write_word_mapping layer0_fn=$lay_fn4 layer_type=Weight+ word_map_fn=$wm4
  if [ $? != 0 ]; then echo $shnm: write_word_mapping failed.; exit 1; fi

  for set in train test; do 
    for no in 2 3 4; do
      p=21 # b/c we want an odd number here ... 
      rnm=${tmpdir}/rcv1${z}-${set}-${no}-p${p}bow
      $prep_exe gen_regions NoSkip Bow \
          region_fn_stem=$rnm input_fn=${ldir}/rcv1-1m-${set} vocab_fn=${tmpdir}/rcv1${z}-${no}.wmap \
          $options text_fn_ext=$txt_ext RegionOnly \
          patch_size=$p patch_stride=1 padding=$(((p-1)/2))
      if [ $? != 0 ]; then echo $shnm: gen_regions failed.; exit 1; fi
    done
  done

  #---  Training with labeled data and five unsupervised embeddings
  mynm=shcnn-5unsemb-rcv1
  logfn=${logdir}/${mynm}.log
  csvfn=${csvdir}/${mynm}.csv
  echo Training with labeled data and five unsupervised embeddings ... see $logfn and $csvfn
  $exe $gpu:$mem train  \
              num_sides=5 0side0_layer_type=LstmF 0side0_layer_fn=$lay_fn0 0side1_layer_type=LstmB 0side1_layer_fn=$lay_fn1 \
              0side2_layer_type=Weight+ 0side2_layer_fn=$lay_fn2 0side2_dsno=1 \
              0side3_layer_type=Weight+ 0side3_layer_fn=$lay_fn3 0side3_dsno=2 \
              0side4_layer_type=Weight+ 0side4_layer_fn=$lay_fn4 0side4_dsno=3 \
              0dsno=4 \
              top_dropout=0.5 test_mini_batch_size=500 \
              max_loss=5 inc=5000 trnname=rcv1${z}-train- tstname=rcv1${z}-test- data_dir=$tmpdir \
              test_interval=25 reg_L2=1e-4 step_size=0.5 evaluation_fn=$csvfn \
              layers=2 loss=Square mini_batch_size=50 momentum=0.9 random_seed=1 \
              datatype=sparse dsno0=p1 dsno1=2-p21bow dsno2=3-p21bow dsno3=4-p21bow dsno4=p21 \
              num_epochs=$supite ss_scheduler=Few ss_decay=0.1 ss_decay_at=$supite1 \
              0nodes=1000 0layer_type=WeightS+ 0activ_type=Rect \
              0num_pooling=10 0pooling_type=Avg 0resnorm_type=Text  \
              1layer_type=Patch 1patch_size=10 > $logfn
  if [ $? != 0 ]; then echo $shnm: training failed.; exit 1; fi

  rm -f ${tmpdir}/rcv1${z}*

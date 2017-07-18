#!/bin/bash
  #---  Semi-supervised LSTM with 2 unsupervised LSTM embeddings 
  #---                        and 3 unsupervised CNN embeddings. 

  #-----------------#
  gpu=-1  # <= change this to, e.g., "gpu=0" to use a specific GPU. 
  mem=4   # pre-allocate 4GB device memory 
  source sh-common.sh
  #-----------------#
  nm=imdb; mb=50; supite=100
# nm=elec; mb=100; supite=50  # Uncomment this to train/test on Elec.

  ddir=data 
  lstmdir=for-semi # <= Change this to where LSTM embeddings are.  Downloaded if lstmdir=for-semi. 
  cnndir=for-semi  # <= Change this to where CNN embeddings are.   Downloaded if cnndir=for-semi.  
  #####
  ##### WARNING: If your system uses Big Endian (Motorola convention), you cannot use the 
  #####    downloaded files!  They are in the little-endian format (Intel convention)!
  #####

  options="LowerCase UTF8"
  catdic=${ddir}/${nm}_cat.dic
  z=l5 # to avoid name conflict with other scripts
  dim=100  # dimensionality of unsupervised embeddings. 

  #---  Download unsupervised embedding files if required. 
  lay_fn0=${nm}-LstmF-dim${dim}.lay.epo30.ReLayer0
  lay_fn1=${nm}-LstmB-dim${dim}.lay.epo30.ReLayer0
  for fn in $lay_fn0 $lay_fn1; do find_file $lstmdir $fn; if [ $? != 0 ]; then echo $shnm: find_file failed.; exit 1; fi; done 
  lay_fn0=${lstmdir}/${lay_fn0}; lay_fn1=${lstmdir}/${lay_fn1}

  lay_fn2=${nm}-uns-p5.dim${dim}.epo10.ReLayer0
  lay_fn3=${nm}-unsx3-p5.dim${dim}.epo10.ReLayer0
  lay_fn4=${nm}-parsup-p3p5.dim${dim}.epo10.ReLayer0
  for fn in $lay_fn2 $lay_fn3 $lay_fn4; do find_file $cnndir $fn; if [ $? != 0 ]; then echo $shnm: find_file failed.; exit 1; fi; done
  lay_fn2=${cnndir}/${lay_fn2}; lay_fn3=${cnndir}/${lay_fn3}; lay_fn4=${cnndir}/${lay_fn4}
  
  #---  Generate region files for unsupervised LSTM embeddings
  voc01=${tmpdir}/${nm}${z}-01.wmap
  $exe $gpu write_word_mapping layer_type=LstmF layer0_fn=$lay_fn0 word_map_fn=$voc01
  if [ $? != 0 ]; then echo $shnm: write_word_mapping failed.; exit 1; fi

  for set in train test; do 
    rnm=${tmpdir}/${nm}${z}-${set}-p1
    $prep_exe gen_regions $options NoSkip region_fn_stem=$rnm \
        input_fn=${ddir}/${nm}-${set} vocab_fn=$voc01 label_dic_fn=$catdic \
        patch_size=1
    if [ $? != 0 ]; then echo $shnm: gen_regions failed.; exit 1; fi
  done

  #---  Generate region files for unsupervised CNN embeddings
  wm2=${tmpdir}/${nm}${z}-2.wmap
  $exe $gpu write_word_mapping layer0_fn=$lay_fn2 layer_type=Weight+ word_map_fn=$wm2
  if [ $? != 0 ]; then echo $shnm: write_word_mapping failed.; exit 1; fi

  wm3=${tmpdir}/${nm}${z}-3.wmap
  $exe $gpu write_word_mapping layer0_fn=$lay_fn3 layer_type=Weight+ word_map_fn=$wm3
  if [ $? != 0 ]; then echo $shnm: write_word_mapping failed.; exit 1; fi
  
  wm4=${tmpdir}/${nm}${z}-4.wmap
  $exe $gpu write_word_mapping layer0_fn=$lay_fn4 layer_type=Weight+ word_map_fn=$wm4
  if [ $? != 0 ]; then echo $shnm: write_word_mapping failed.; exit 1; fi

  for set in train test; do 
    for no in 2 3 4; do
      p=5
      rnm=${tmpdir}/${nm}${z}-${set}-${no}-p${p}bow
      $prep_exe gen_regions $options NoSkip Bow RegionOnly region_fn_stem=$rnm \
          input_fn=${ddir}/${nm}-${set} vocab_fn=${tmpdir}/${nm}${z}-${no}.wmap \
          patch_size=$p padding=$(((p-1)/2))
      if [ $? != 0 ]; then echo $shnm: gen_regions failed.; exit 1; fi
    done
  done

  #---  Training with labeled data and five unsupervised embeddings 
  mynm=lstm-5unsemb-${nm}-dim${dim}
  logfn=${logdir}/${mynm}.log; csvfn=${csvdir}/${mynm}.csv
  echo; echo Training with labeled data and five tv-embeddings ... see $logfn and $csvfn
  suplam=0; suplamtop=1e-4  # regularization parameters.
  $exe $gpu:$mem train  reg_L2=$suplam top_reg_L2=$suplamtop top_dropout=0.5 \
              num_sides=5 0side0_layer_type=LstmF 0side0_layer_fn=$lay_fn0 0side1_layer_type=LstmB 0side1_layer_fn=$lay_fn1 \
              0side2_layer_type=Weight+ 0side2_layer_fn=$lay_fn2 0side2_dsno=1 \
              0side3_layer_type=Weight+ 0side3_layer_fn=$lay_fn3 0side3_dsno=2 \
              0side4_layer_type=Weight+ 0side4_layer_fn=$lay_fn4 0side4_dsno=3 \
              NoGate_i NoGate_o test_mini_batch_size=500 \
              max_loss=5 inc=5000 trnname=${nm}${z}-train- tstname=${nm}${z}-test- data_dir=${tmpdir} \
              test_interval=25 step_size=1 evaluation_fn=$csvfn \
              layers=2 loss=Square mini_batch_size=$mb momentum=0.9 random_seed=1 \
              datatype=sparse dsno0=p1 dsno1=2-p5bow dsno2=3-p5bow dsno3=4-p5bow \
              num_epochs=$supite ss_scheduler=Few ss_decay=0.1 ss_decay_at=$((supite*4/5)) \
              0layer_type=Lstm2 0nodes=500 0chop_size=50 \
              1layer_type=Pooling 1num_pooling=1 1pooling_type=Max > $logfn
  if [ $? != 0 ]; then echo $shnm: training failed.; exit 1; fi

  rm -f ${tmpdir}/${nm}${z}*
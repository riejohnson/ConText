#!/bin/bash
  #####
  #####  parsup-tv.  Embedding learning with partially-supervised target. 
  #####
  #####  Step 1. Learning embedding from unlabeled data using partially-supervised target. 
  #####    Step 1.1. Generate input files for applying a model trained with labeled data to unlabeled data. 
  #####    Step 1.2. Apply a supervised model to unlabeled data to obtain embedding results (i.e., output of region embedding). 
  #####    Step 1.3. Generate input files for embedding training with partially-supervised target. 
  #####              Target: the embedding results (obtained in Step 1.2) derived from the neighboring regions.  
  #####    Step 1.4. Training. 
  #####  Step 2. Supervised training using the trained embedding to produce additional input. 
  #####    Step 2.1. Generate input files. 
  #####    Step 2.2. Training. 
  #####
  #####  NOTE: Unlabeled data is downloaded by find_file in sh-common.sh. 
  #####        A model file is also downloaded if moddir=for-semi.
  #####

  #-----------------#
  gpu=-1  # <= change this to, e.g., "gpu=0" to use a specific GPU. 
  source sh-common.sh
  #-----------------#
  dim=100       # <= Dimensionality of embedding. 
  udir=for-semi # <= Where unlabeled data is.  Downloaded if udir=for-semi.
  nm=imdb 
# nm=elec  # <= Uncomment this to train/test on Elec.

  options="LowerCase UTF8"
  txt_ext=.txt.tok

  #***  Step 0. Prepare a supervised model and download unlabeled data. 
  f_pch_sz=3; f_pch_step=1; f_padding=$((f_pch_sz-1))
  modkw=dwnld; moddir=for-semi; mod_fn=for-parsup-${nm}-p${f_pch_sz}.supmod.ReNet # Download the model file. 
# modkw=gend; moddir=$outdir;  mod_fn=shcnn-seq-${nm}-mod.epo100.ReNet # Use the output of shcnn-seq-imdb-elec.sh
  find_file $moddir $mod_fn
  if [ $? != 0 ]; then
    if [ "$moddir" != "for-semi" ]; then echo $shnm: This script uses the output of shcnn-seq-imdb-elec.sh. \
                                              Run it first to generate $mod_fn.; exit 1; fi
    echo $shnm: find_file failed.; exit 1
  fi
  #####
  ##### WARNING: If your system uses Big Endian (Motorola convention), you cannot use the 
  #####    downloaded model file!  It is in the little-endian format (Intel convention)!
  #####
  
  z=-${modkw}; mod_fn=${moddir}/${mod_fn} # to avoid filename conflict with other scripts

  uns_lst=${tmpdir}/${nm}${z}-trnuns.lst # unlabeled data
  if [ "$nm" = "imdb" ]; then
    echo data/imdb-train.txt.tok    >  $uns_lst
    echo ${udir}/imdb-unlab.txt.tok >> $uns_lst
    find_file $udir imdb-unlab.txt.tok; if [ $? != 0 ]; then echo $shnm: find_file failed.; exit 1; fi   
  elif [ "$nm" = "elec" ]; then
    echo ${udir}/elec-25k-unlab00.txt.tok  > $uns_lst
    echo ${udir}/elec-25k-unlab01.txt.tok >> $uns_lst 
    find_file $udir elec-25k-unlab00.txt.tok; if [ $? != 0 ]; then echo $shnm: find_file failed.; exit 1; fi 
    find_file $udir elec-25k-unlab01.txt.tok; if [ $? != 0 ]; then echo $shnm: find_file failed.; exit 1; fi     
  else 
    echo Unexpected dataset name: $nm; exit 1
  fi

  #***  Step 1. Learning embedding from unlabeled data using partially-supervised target. 
  #---  Step 1.1. Generate region files for applying a supervised model to unlabeled data. 

  #---  vocabulary for X (regions)
  xszk=30
  xvoc=${tmpdir}/${nm}${z}-for-parsup.vocab
  $prep_exe gen_vocab input_fn=data/${nm}-train.txt.tok vocab_fn=$xvoc $options WriteCount \
                  max_vocab_size=${xszk}000
  if [ $? != 0 ]; then echo $shnm: gen_vocab failed.; exit 1; fi

  #---  NOTE: With larger unlabeled data, try input_fn=$uns_lst and max_vocab_size=100000.  

  opt2="NoSkip RegionOnly"
  rnm=${tmpdir}/${nm}${z}-uns-p${f_pch_sz}
  #---  generate a region file of unlabeled data. 
  $prep_exe gen_regions region_fn_stem=$rnm input_fn=$uns_lst vocab_fn=$xvoc $options $opt2 \
         patch_size=$f_pch_sz patch_stride=$f_pch_step padding=$f_padding \
         text_fn_ext=  # to avoid attaching ".txt.tok" to the input filename ... 
  if [ $? != 0 ]; then echo $shnm: gen_regions failed.; exit 1; fi

  #---  Step 1.2. Apply a supervised model to unlabeled data to obtain embedded regions. 
  top_num=10 # retain the 10 largest values only.  
  emb_fn=${rnm}.emb.smats
  $exe $gpu write_embedded 0DisablePooling test_mini_batch_size=100 \
       datatype=sparse tstname=$rnm  \
       model_fn=$mod_fn num_top=$top_num embed_fn=$emb_fn
  if [ $? != 0 ]; then echo $shnm: write_embedded failed.; exit 1; fi

  #---  Step 1.3. Generate input files for embedding training with partially-supervised target.  
  p=5
  dist=$p
  unsnm=${nm}-parsup-p${f_pch_sz}p${p}
  $prep_exe gen_regions_parsup x_ext=.xsmatbc y_ext=.ysmatc x_type=Bow input_fn=$uns_lst \
                   scale_y=1 \
                   x_vocab_fn=$xvoc region_fn_stem=${tmpdir}/$unsnm $options \
                   patch_size=$p patch_stride=1 padding=$((p-1)) \
                   f_patch_size=$f_pch_sz f_patch_stride=$f_pch_step f_padding=$f_padding \
                   dist=$dist num_top=$top_num embed_fn=$emb_fn
  if [ $? != 0 ]; then echo $shnm: gen_regions_parsup failed.; exit 1; fi

  #---  Step 1.4. Embedding training. 
  gpumem=${gpu}:1 # pre-allocate 1GB GPU memory. 
  epo1=8; epo=10
  lay0_fn=${outdir}/${unsnm}-${modkw}-dim${dim}
  logfn=${logdir}/${unsnm}-${modkw}-dim${dim}.log
  echo 
  echo Training embedding with partially-supervised target.  
  echo This takes a while.  See $logfn for progress. 
  $exe $gpumem train trnname=$unsnm data_dir=$tmpdir 0save_layer_fn=$lay0_fn \
        NoCusparseIndex zero_Y_weight=0.2 zero_Y_ratio=5 \
        NoTest Regression loss=Square random_seed=1 \
        ss_scheduler=Few ss_decay=0.1 step_size=0.5 ss_decay_at=$epo1 num_epochs=$epo \
        datatype=sparse x_ext=.xsmatbc y_ext=.ysmatc \
        mini_batch_size=100 momentum=0.9 reg_L2=0 \
        layers=1 0nodes=$dim 0activ_type=Rect 0resnorm_type=Text  \
        inc=500000  \
        > $logfn
  if [ $? != 0 ]; then echo $shnm: Embedding training failed.; exit 1; fi

  rm -f ${tmpdir}/${unsnm}*

  #***  Step 2. Supervised training using the trained embedding to produce additional input. 
  #---  Step 2.1. Generate input files.
  for set in train test; do 
    opt=NoSkip
    #---  dataset#0: for the main layer (seq)
    $prep_exe gen_regions $opt $options region_fn_stem=${tmpdir}/${nm}${z}-${set}-p${p}seq \
      input_fn=data/${nm}-${set} vocab_fn=$xvoc label_dic_fn=data/${nm}_cat.dic \
      patch_size=$p padding=$((p-1))
    if [ $? != 0 ]; then echo $shnm: gen_regions failed.; exit 1; fi

    #---  dataset#1: for the side layer (bow)
    $prep_exe gen_regions $opt $options Bow RegionOnly \
        region_fn_stem=${tmpdir}/${nm}${z}-${set}-p${p}bow \
        input_fn=data/${nm}-${set} vocab_fn=$xvoc \
        patch_size=$p padding=$((p-1))
    if [ $? != 0 ]; then echo $shnm: gen_regions failed.; exit 1; fi
  done

  #---  Step 2.2. Training 
  mynm=shcnn-parsup-${nm}-dim${dim}-${nm}-${modkw}
  logfn=${logdir}/${mynm}.log; csvfn=${csvdir}/${mynm}.csv
  s_fn=${lay0_fn}.epo${epo}.ReLayer0
  nodes=1000 # number of feature maps 
  echo; echo Training ... This takes a while.  See $logfn and $csvfn for progress. 
  $exe $gpu:4 train V2 \
        trnname=${nm}${z}-train-p${p} tstname=${nm}${z}-test-p${p} dsno0=seq dsno1=bow data_dir=$tmpdir \
        datatype=sparse \
        reg_L2=1e-4 step_size=0.1 top_dropout=0.5 \
        mini_batch_size=100 momentum=0.9 random_seed=1 \
        num_epochs=100 ss_scheduler=Few ss_decay=0.1 ss_decay_at=80 \
        loss=Square \
        layers=1 0layer_type=WeightS+ \
        0activ_type=Rect 0nodes=$nodes 0resnorm_type=Text  \
        0pooling_type=Max 0num_pooling=1 \
        num_sides=1 0side0_layer_type=Weight+ 0side0_layer_fn=$s_fn 0side0_dsno=1 \
        evaluation_fn=$csvfn test_interval=25 > $logfn
  if [ $? != 0 ]; then echo $shnm: training failed.; exit 1; fi

  rm -f ${tmpdir}/${nm}${z}*
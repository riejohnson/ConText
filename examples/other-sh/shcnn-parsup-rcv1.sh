#!/bin/bash
  #####
  #####  RCV1: parsup-tv.  Embedding learning with partially-supervised target. 
  #####
  #####  Step 1. Learning embedding from unlabeled data using partially-supervised target. 
  #####    Step 1.1. Generate input files for applying a model trained with labeled data to unlabeled data. 
  #####    Step 1.2. Apply a supervised model to unlabeled data to obtain embedding results (i.e., output of region embedding). 
  #####    Step 1.3. Generate input files for training embedding with partially-supervised target. 
  #####              Target: embedded regions (obtained in Step 1.2) derived from the neighboring regions.  
  #####    Step 1.4. Training. 
  #####  Step 2. Supervised training using the trained embedding function to produce additional input. 
  #####    Step 2.1. Generate input files. 
  #####    Step 2.2. Training. 
  #####
  #####  NOTE: A model file is downloaded if moddir=for-semi.

  #-----------------#
  gpu=-1  # <= change this to, e.g., "gpu=0" to use a specific GPU. 
  source sh-common.sh
  #-----------------#
  dim=100 # <= change this to change dimensionality of embedding. 
  rcv1dir=rcv1_data # <= change this to the directory where RCV1 labeled data is. 
  rcv1udir=rcv1_unlab_data  # <= change this to the directory where RCV1 unlabeled data is. 
           #
           #    NOTE: We cannot provide RCV1 unlabeled data due to the copyright issue.  
           #

  options="LowerCase UTF8"
  txt_ext=.txt.tok

  #***  Step 0. Prepare a supervised model.  Donwloaded if moddir=for-semi.
  f_pch_sz=20; f_pch_step=2; f_padding=$((f_pch_sz-1))
  #------------------------------------
  modkw=dwnld; moddir=for-semi; mod_fn=for-parsup-rcv1-p${f_pch_sz}.supmod.ReNet # Download the file. 
# modkw=gend; moddir=$outdir; mod_fn=shcnn-bow-rcv1-mod.epo100.ReNet # Use the output of shcnn-bow-rcv1.sh
  #####
  ##### WARNING: If your system uses Big Endian (Motorola convention), you cannot use the 
  #####    downloaded model file!  It is in the little-endian format (Intel convention)!
  #####

  find_file $moddir $mod_fn
  if [ $? != 0 ]; then 
    if [ "$moddir" != "for-semi" ]; then echo $shnm: This script uses the output of shcnn-bow-rcv1.sh. \
                                              Run it first to generate $mod_fn.; exit 1; fi
    echo $shnm: find_file failed.; exit 1
  fi
  mod_fn=${moddir}/${mod_fn}
  #------------------------------------
  z=-${modkw} # to avoid name conflict with other scripts

  #***  Step 1. Embedding learning from unlabeled data using partially-supervised target. 
  #---  Step 1.1. Generate input files for applying a supervised model to unlabeled data.

  uns_lst=${tmpdir}/rcv1${z}-uns.lst
  rm $uns_lst
  for ym in 609 610 611 612 701 702 703 704 705 706; do
    echo ${rcv1udir}/rcv1-199${ym} >> $uns_lst
  done

  #---  vocabulary for X (regions)
  xvoc=${tmpdir}/rcv1${z}-trn.vocab
#  $prep_exe gen_vocab input_fn=${rcv1dir}/rcv1-1m-train.txt.tok vocab_fn=$xvoc $options WriteCount \
#                  max_vocab_size=30000
  $exe $gpu write_word_mapping model_fn=$mod_fn word_map_fn=$xvoc
  if [ $? != 0 ]; then echo $shnm: write_word_mapping failed.; exit 1; fi

  #---  NOTE: Also try input_fn=$uns_lst and max_vocab_size=100000. 

  #---  split text into 5 batches. 
  batches=5
  $prep_exe split_text input_fn=$uns_lst ext=$txt_ext num_batches=$batches random_seed=1 output_fn_stem=${tmpdir}/rcv1${z}-uns
  if [ $? != 0 ]; then echo $shnm: split_text failed.; exit 1; fi

  #---  Prepare data for training with unlabeled data
  opt2="Bow NoSkip RegionOnly"
  pch_sz=20; pch_step=1; padding=$((pch_sz-1))
  top_num=10
  dist=$pch_sz

  nm=rcv1${z}-parsup-p${f_pch_sz}p${pch_sz}

  for no in 1 2 3 4 5; do  # for each batch
    batch_id=${no}of${batches}
    rnm=${tmpdir}/rcv1${z}-uns${no}-p${f_pch_sz}
    #---  generate a region file for unlabeled data. 
    $prep_exe gen_regions \
         region_fn_stem=$rnm input_fn=${tmpdir}/rcv1${z}-uns.${batch_id} vocab_fn=$xvoc \
         $options $opt2 text_fn_ext=$txt_ext \
         patch_size=$f_pch_sz patch_stride=$f_pch_step padding=$f_padding
    if [ $? != 0 ]; then echo $shnm: gen_regions failed.; exit 1; fi  

    #---  Step 1.2. Apply a supervised model to unlabeled data to obtain embedding results. 
    emb_fn=${rnm}.emb.smats
    $exe $gpu write_embedded test_mini_batch_size=100 0DisablePooling \
       datatype=sparse tstname=$rnm  \
       model_fn=$mod_fn num_top=$top_num embed_fn=$emb_fn
    if [ $? != 0 ]; then echo $shnm: write_embedded failed.; exit 1; fi

    #---  Step 1.3. Generate input files for embedding training with partially-supervised target.  
    $prep_exe gen_regions_parsup x_ext=.xsmatbc y_ext=.ysmatc \
                   x_type=Bow input_fn=${tmpdir}/rcv1${z}-uns.${batch_id} text_fn_ext=$txt_ext \
                   scale_y=1 MergeLeftRight \
                   x_vocab_fn=$xvoc region_fn_stem=${tmpdir}/$nm $options \
                   patch_size=$pch_sz patch_stride=$pch_step padding=$padding \
                   f_patch_size=$f_pch_sz f_patch_stride=$f_pch_step f_padding=$f_padding \
                   dist=$dist num_top=$top_num embed_fn=$emb_fn batch_id=$batch_id
    if [ $? != 0 ]; then echo $shnm: gen_regions_parsup failed.; exit 1; fi
  done

  #---  Step 1.4. Embedding training  
  gpumem=${gpu}:1 # pre-allocate 1GB GPU memory. 
  epo1=8; epo=10  # "epo1=4; epo=5" is good enough, though. 
  lay0_fn=${outdir}/${nm}-dim${dim}
  logfn=${logdir}/${nm}-dim${dim}.log
  echo 
  echo Embedding learning with partially-supervised target.  
  echo This takes a while.  See $logfn for progress. 
  $exe $gpu:1 train num_batches=$batches trnname=$nm data_dir=$tmpdir 0save_layer_fn=$lay0_fn \
        datatype=sparse x_ext=.xsmatbc y_ext=.ysmatc NoCusparseIndex \
        step_size=0.5 ss_scheduler=Few ss_decay=0.1 ss_decay_at=$epo1 num_epochs=$epo \
        NoTest Regression loss=Square mini_batch_size=100 momentum=0.9 reg_L2=0 random_seed=1 \
        zero_Y_weight=0.2 zero_Y_ratio=5 \
        layers=1 0layer_type=Weight+ 0nodes=$dim 0activ_type=Rect 0resnorm_type=Text  \
         inc=500000 \
        > $logfn
  if [ $? != 0 ]; then echo $shnm: Embedding training failed.; exit 1; fi

  #***  Step 2. Supervised training using the trained embedding to produce additional input. 
  #---  Step 2.1. Generate input files.
  for set in train test; do 
    rnm=${tmpdir}/rcv1${z}-1m-${set}-p${pch_sz}
    $prep_exe gen_regions Bow VariableStride \
      region_fn_stem=$rnm input_fn=${rcv1dir}/rcv1-1m-${set} vocab_fn=$xvoc \
      $options text_fn_ext=$txt_ext label_fn_ext=.lvl2 \
      label_dic_fn=data/rcv1-lvl2.catdic \
      patch_size=$pch_sz patch_stride=2 padding=$((pch_sz-1))
    if [ $? != 0 ]; then echo $shnm: gen_regions failed.; exit 1; fi
  done

  #---  Step 2.2. Training. 
  gpumem=${gpu}:4 # pre-allocate 4GB GPU memory
  mynm=shcnn-parsup-rcv1-dim${dim}-${modkw}
  logfn=${logdir}/${mynm}.log
  csvfn=${csvdir}/${mynm}.csv
  s_fn=${lay0_fn}.epo${epo}.ReLayer0
  stepsize=0.5; nodes=1000
  echo 
  echo Supervised training using the partially-supervised embedding to produce additional input.   
  echo This takes a while.  See $logfn and $csvfn for progress. 
  $exe $gpumem train V2 \
        trnname=rcv1${z}-1m-train-p${pch_sz} tstname=rcv1${z}-1m-test-p${pch_sz} data_dir=$tmpdir \
        datatype=sparse \
        reg_L2=1e-4 step_size=$stepsize \
        mini_batch_size=100 momentum=0.9 random_seed=1 \
        num_epochs=100 ss_scheduler=Few ss_decay=0.1 ss_decay_at=80 \
        loss=Square \
        layers=2 0layer_type=WeightS+ 0nodes=$nodes 0activ_type=Rect \
        num_sides=1 0side0_layer_type=Weight+ 0side0_layer_fn=$s_fn \
        0pooling_type=Avg 0num_pooling=10 0resnorm_type=Text  \
        1layer_type=Patch 1patch_size=10 \
        evaluation_fn=$csvfn test_interval=25  \
        > $logfn
  if [ $? != 0 ]; then echo $shnm: training failed.; exit 1; fi

  rm -f ${tmpdir}/rcv1${z}*
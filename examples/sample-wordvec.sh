#!/bin/bash
  #####
  #####  Example: Using given word vectors as input to CNN, as is typically done. 
  #####           Word vectors are in one word2vec-format binary file. 
  #####
  #####  NOTE: If word vectors require too much GPU memory (causing "out of memory"), 
  #####        reduce the vocabulary ($xvoc below). 

  gpu=-1  # <= change this to, e.g., "gpu=0" to use a specific GPU. 
  source sh-common.sh

  #-----  Pre-trained word vectors  -----
  options=LowerCase
  dim=10                        # dimensionality of word vectors. 
  wvparam="wordvec_bin_fn=data/s-dim10.wvbin" # word vector file in word2vec binary format (Little Endian)
# wvparam="wordvec_txt_fn=data/s-dim10.wvtxt" # word vector file in word2vec text format
  #--------------------------------------

  #---  write vocabulary (word mapping) of word vectors to a file. 
  wm_fn=${tmpdir}/s-wordvec.wmap.txt
  $prep_exe write_wv_word_mapping $wvparam word_map_fn=$wm_fn # IgnoreDupWords
  if [ $? != 0 ]; then echo $shnm: write_wv_word_mapping failed.; exit 1; fi

  #---  Extract training data vocabulary.  
  trn_voc=${tmpdir}/s-trn.vocab
  $prep_exe gen_vocab input_fn=data/s-train.txt.tok vocab_fn=$trn_voc $options WriteCount max_vocab_size=30000
  if [ $? != 0 ]; then echo $shnm: gen_vocab failed.; exit 1; fi

  #---  Merge the word vector vocabulary and training data vocabulary.  
  xvoc=${tmpdir}/s-wv-trn.vocab
  $prep_exe merge_vocab input_fns=${wm_fn}+${trn_voc} vocab_fn=$xvoc
  if [ $? != 0 ]; then echo $shnm: merge_vocab failed.; exit 1; fi

  #---  NOTE: If word vectors require too much GPU memory (indicated by "out of memory"), 
  #---        remove some words from $xvoc (vocabulary).  

  #---  Convert text to one-hot vectors and generate target files. 
  for set in train test; do 
    #---  sparse region vectors of size 1 = one-hot vectors 
    rnm=${tmpdir}/s-${set}-p1
    $prep_exe gen_regions $options NoSkip region_fn_stem=$rnm \
        input_fn=data/s-${set} vocab_fn=$xvoc label_dic_fn=data/s-cat.dic \
        patch_size=1
    if [ $? != 0 ]; then echo $shnm: gen_regions failed.; exit 1; fi
  done

  #---  Convert word2vec word vector files to a weight file. 
  w_fn=${tmpdir}/s-wv-trn.dmatc # weight file
  $prep_exe adapt_word_vectors $wvparam word_map_fn=$xvoc weight_fn=$w_fn rand_param=0.01 random_seed=1
  if [ $? != 0 ]; then echo $shnm: adapt_word_vectors failed.; exit 1; fi

  #---  Training. 
  #---    Layer-0: word embedding layer
  #---    Layer-1&2: convolution layer of region size 3, stride 1, padding 2. 
  #---                     followed by max-pooling with one pooling region. 

  opt=0Fixed # Don't fine-tune word vectors. Comment out this line to fine-tune word vectors. 
  $exe $gpu train layers=3 $opt \
        0layer_type=Weight+ 0activ_type=None 0nodes=$dim 0weight_fn=$w_fn 0NoIntercept \
        1layer_type=Patch 1patch_size=3 1patch_stride=1 1padding=2 \
        2layer_type=Weight+ 2activ_type=Rect 2nodes=500 2pooling_type=Max 2num_pooling=1 \
        datatype=sparse data_dir=$tmpdir trnname=s-train-p1 tstname=s-test-p1 \
        reg_L2=1e-4 step_size=0.1 loss=Square mini_batch_size=100 momentum=0.9 random_seed=1 \
        num_epochs=20 ss_scheduler=Few ss_decay=0.1 ss_decay_at=15 test_interval=5
 if [ $? != 0 ]; then echo $shnm: training failed.; exit 1; fi

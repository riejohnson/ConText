#!/bin/bash
#****************************************************************************
# Training/testing DPCNN without unsupervised embeddings. 
#
# Step 0. Prepare input files.
#   Prepare token files and label files of training, validation, and test sets. 
#   and a label dictionary file according to naming conventions as follows: 
#   * Naming conventions for input files ( see data/s-dp.* for example) 
#     - training data:   ${nm}-td.${no}of${bb}.{txt.tok|cat} where bb is the number of batches, no=1,2,...,bb
#     - validation data: ${nm}-dv.{txt.tok|cat}
#     - test data:       ${nm}-test.{txt.tok|cat}
#     - label dictionary:${nm}.catdic
# 
# Step 1. Train DPCNN. 
# 
# Step 2. Apply a model to test data and write prediction values to a file. 
#****************************************************************************
  source sh-common.sh        # Constants and functions
  source dpcnn-functions.sh  # Constants and functions for DPCNN

  max_jobs=5    # <= Data preparation (dpcnn_gen_regions*) creates up to this number of background processes.
  dont_reuse=1  # <= 1: Don't reuse old files and create new files. 
                # <= 0: Reuse existing files (e.g., vocaburary, regions files). 
                # <= Set this to 1 if existing files are obsolete or broken and causing errors. 
                # <= Set this to 0 to avoid re-generating existing files. 

  #-------------------
  #---  Step 0. Prepare input files (token files, label files, and label dictionary). 
  #-------------------
  # - dpcnn-tokenize.sh can be used to produce token files and label files to prepare input files 
  #   from csv files of Yelp.p etc.  
  # - Please follow the naming conventions above if you do it by yourself. 
  # - Small toy data "s-dp" is provided at data/. 

  #-------------------
  #---  Step 1. Train DPCNN 
  #-------------------
  gpu=-1         # <= Change this to, e.g., "gpu=0" to use a specific GPU. 
  idir=gendata   # <= Change this to where the dataset files (tokenized text files and label files etc.) are.
  # <= Uncomment one of the following 6 lines. 
  #---
  nm=s-dp;     bb=2;  p=3; ss=0.25;  topdo=0.5; tepo=50; idir=data  # small sample data
# nm=yelppol;  bb=5;  p=3; ss=0.05;  topdo=0.5; tepo=135
# nm=yelpfull; bb=5;  p=3; ss=0.05;  topdo=0.5; tepo=125
# nm=yah10;    bb=10; p=5; ss=0.05;  topdo=0;   tepo=135
# nm=amafull;  bb=10; p=3; ss=0.025; topdo=0;   tepo=150
# nm=amapol;   bb=10; p=3; ss=0.025; topdo=0;   tepo=145
  #---
                 # nm: Data name.
                 # bb: Number of training data batches. 
                 # p:  Window size of the text region embedding layer (1st layer). 
                 # ss: Step size (learning rate)
                 # topdo: Dropout parameter applied to the input to the top (output) layer.
                 # tepo (test epochs) was chosen based on the validation performance. 
                 # 
  lays=15        # Number of weight layers. 
  nodes=250      # Number of feature maps.
  epochs=150     # Train for 150/bb epochs (bb is the number of train data batches).
                 # NOTE: Perform early stopping based on the validation performance 
                 #       after the learning rate is reduced at epochs*4/5.
  xmax=30        # Vocabulary size / 1000 (e.g., 30 means 30K) 
  pnm=p${p}      # Used to generate input filenames.

  if [ "$nm" = "s-dp" ]; then epochs=50; lays=5; fi # Changing parameters for toy data. 

  mynm=dpcnn-${nm}-${lays} # used to generate output filenames

  #---  Generate input files (regions files etc. )
  dpcnn_gen_regions 
  if [ $? != 0 ]; then echo dpcnn_gen_regions failed; exit 1; fi

  #---  Training without unsupervised embeddings. 
  do_save=1      # Save models in files. 
# do_save=0      # Don't save models.  
  dpcnn_training
  if [ $? != 0 ]; then echo dpcnn_training failed; exit 1; fi

  #---  Apply the model saved after $tepo epochs to test data and write prediction values to a file
# do_write_text=0 # Write in the binary format, faster 
  do_write_text=1 # Write in the text format, slower
  epochs=$tepo # Use the models saved after $tepo ecpochs. 
  dpcnn_predict 
  if [ $? != 0 ]; then echo dpcnn_predict failed; exit 1; fi

#!/bin/bash
#---------------------
# DPCNN with unsupervised embedding. 
# This script uses miniscule toy data and train DPCNN with one unsupervised embedding. 
#
# NOTE1: For light-weight demonstration, toy data is used here, and note that 
#        DPCNN is NOT suitable for miniscule data like this one. 
# NOTE2: Parameters used in [JZ17] on real-world data such as Yelp.p are 
#        in dpcnn-with-unsemb.sh
# 
# Step 1. Train an unsupervised embedding function. 
# Step 2. Train DPCNN using the unsupervised embedding trained above to produce additional input. 
# Step 3. Apply a model to the test data and write prediction values to a file. 
#---------------------
  source sh-common.sh
  source dpcnn-functions.sh  # Functions for DPCNN

  gpu=-1        # <= Change this to, e.g., "gpu=0" to use a specific GPU. 
  max_jobs=5    # <= Data preparation (dpcnn_gen_regions*) creates up to this number of background processes.
  dont_reuse=1  # <= 1: Don't reuse old files and create new files. 
                # <= 0: Reuse existing files (e.g., vocaburary, regions files). 
                # <= Set this to 1 if existing files are obsolete or broken and causing errors. 
                # <= Set this to 0 to avoid re-generating existing files. 

  nm=s-dp       # Data name
  idir=data     # Where the input files (token/label files and label dictionary) are. 
  bb=2          # Number of training data batches. 
  p0=3          # Window size of text region embedding (1st layer), 
  ss=0.25       # Step-size (learning rate) for the final supervised training. 
  topdo=0.5     # Dropout rate applied to the input to the top layer.  Default: No dropout. 
                # 
  nodes=250     # Number of feature maps.
  epochs=50     # 50/bb epochs
  dim=100       # Dimensionality of unsupervised embeddings. 
  xmax=30       # Size of uni-gram vocabulary 
  x3max=200     # Size of 1-3gram vocabulary for unsupervised embeddings
  trikw=tri${x3max}k # Used in the filenames for region files of 1-3grams. 
  lays=5        # Number of hidden weight layers


  #-------------------------------------
  #---  Step 1. Train an unsupervised embedding function.
  #-------------------------------------
  p=5        # Region size 5
  n=1        # Use bag of word uni-grams to represent regions.
  pnm=p${p}  # Used for generating filenames.

  #---  Unsupervised embedding training.
  dpcnn_gen_regions_unsemb;   if [ $? != 0 ]; then echo $shnm: dpcnn_gen_regions_unsup failed.; exit 1; fi
  dpcnn_train_unsemb;         if [ $? != 0 ]; then echo $shnm: dpcnn_train_unsemb failed.; exit 1; fi
  rm -f ${tmpdir}/${nm}-td-uns*${pnm}  # clean up

  #---  Generate region file for a side layer for the final training.
  #---- NOTE: dpcnn_gen_regions_side must be called AFTER dpcnn_gen_regions_unsemb with the same parameters.
  dpcnn_gen_regions_side; if [ $? != 0 ]; then echo $shnm: dpcnn_gen_regions_side failed.; exit 1; fi


  #-------------------------------------
  #---  Step 2. Final training using unsupervised embeddings. 
  #-------------------------------------
  #---  Prepare input files. 
  p=$p0; pnm=p${p}
  dpcnn_gen_regions # prepare regions file for layer-0, target file, etc.
  if [ $? != 0 ]; then echo $shnm: dpcnn_gen_regions failed.; exit 1; fi

  #---  Training
  spnms=( p5 ) # One unsupervised embedding 
  mynm=dpcnn-1uemb-${nm}-${lays}-p${p}-ss${ss}-${topdo}
  do_save=1  # Save models in files. 
# do_save=0  # Don't save models in files. 
  dpcnn_training   
  if [ $? != 0 ]; then echo dpcnn_training failed.; exit 1; fi


  #-------------------------------------
  #---  Step 3. Apply the model saved after $epochs epochs to test data.
  #-------------------------------------
  #---  Prediction values are written to: 
  #---     ${outdir}/${mynm}.epo${tepo}.ReNet.pred.{txt|bin}
  epochs=45
  do_write_text=1
  dpcnn_predict 
  if [ $? != 0 ]; then echo dpcnn_predict failed.; exit 1; fi

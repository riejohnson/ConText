#!/bin/bash
#****************************************************************************
# This script shows parameter settings of DPCNN with unsupervised embeddings 
# used in [JZ17].  
# 
# Step 0. Prepare input files (token files, label files, and label dictionary)
#         See Step 0 of dpcnn-without-unsemb.sh. 
# Step 1. Train 4 unsupervised embeddings. 
# Step 2. Train DPCNN using the 4 unsupervised embeddings to produce additional input. 
# Step 3. Apply a model to test data and write prediction values to a file. 
#
# NOTE1: To run this script, one of the 8 lines beginning with "nm=" needs to 
#        be uncommented.
#
# NOTE2: Large training data requires large disk space, e.g., Yelp.p: 16GB, 
#        Ama.p (largest): 63GB, due to the design choice to speed up training
#        at the expense of disk space. 
#
# NOTE3: See the lines below "max_jobs" for CPU memory amounts.  
#****************************************************************************
  source sh-common.sh        # Constants
  source dpcnn-functions.sh  # Functions for DPCNN
  #-----------------#

  gpu=-1        # <= change this to, e.g., "gpu=0" to use a specific GPU. 
  idir=gendata  # <= Change this to where the input files (token/label files and label dictionary) are. 
  max_jobs=5    # <= Data preparation (dpcnn_gen_regions*) creates up to this number of background processes.
     # A larger max_jobs (up to bb+2) makes dpcnn_gen_regions* faster but consumes more CPU memory.  
     # CPU memory amounts required by dpcnn_gen_regions_unsup (the most memory consuming) with 
     # max_jobs=5 are, e.g., Yelp.p:         3GB(p=5,n=1), 5GB(p=5,n=3), 5GB(p=9,n=1), 8GB (p=9,n=3)
     #                       Ama.p(largest): 7GB(p=5,n=1),11GB(p=5,n=3),12GB(p=9,n=1),17GB (p=9,n=3)
  dont_reuse=1  # <= 1: Don't reuse old files and create new files. 
                # <= 0: Reuse existing files (e.g., vocaburary, regions files). 
                # <= Set this to 1 if existing files are obsolete or broken and causing errors. 
                # <= Set this to 0 to avoid re-generating existing files. 

  #-------------------------------------
  #---  Parameters used in [JZ17] Table 2&3 row 1. 
  nodes=250  # Number of feature maps.
  epochs=150 # 150 epochs (=30 epochs if the number of batches (bb below) is 5). 
  tmb=500    # mini-batch size for testing. 
  dim=300    # dimensionality of unsupervised embeddings. 
  xmax=30    # size of uni-gram vocabulary 
  x3max=200  # size of 1-3gram vocabulary for unsupervised embeddings
  trikw=tri${x3max}k # used in the filenames for region files of 1-3grams. 
  lays=15    # number of hidden weight layers

  # <=  Uncomment one of the following 8 lines. 
# nm=yelppol;  bb=5;  p0=3; ss=0.05;  tepo=130; topdo=0.5 
# nm=yelpfull; bb=5;  p0=1; ss=0.05;  tepo=125; topdo=0.5
# nm=yah10;    bb=10; p0=5; ss=0.025; tepo=125 
# nm=amafull;  bb=10; p0=3; ss=0.025; tepo=125
# nm=amapol;   bb=10; p0=3; ss=0.025; tepo=130
# nm=ag;       bb=1;  p0=5; ss=0.05;  tepo=45;  topdo=0.5; dim=100; epochs=50
# nm=sog10;    bb=10; p0=3; ss=0.05;  tepo=135; topdo=0.5; dim=100;              tmb=100; uss=0.5; umom=0.9
# nm=dbpedia;  bb=5;  p0=3; ss=0.1;   tepo=140; topdo=0.5; dim=100

             # bb: number of batches. 
             # p0: window size of text region embedding (1st layer), 
             # ss: step-size (learning rate) for the final supervised training. 
             # topdo: dropout rate applied to the input to the top layer.  Default: No dropout. 
             # tmb: mini-batch size for testing.  Default: 500. 
             # uss: step-size for unsupervised embedding training. Default: 5
             # umom: momentum for unsupervised embedding training. Default: No momentum
             #
             # tepo (test epochs) was chosen based on the validation performance. 

  #-------------------------------------
  #---  Step 1. Train four unsupervised embeddings. 
  #---  NOTE: This takes long.  
  #---        To speed up, work on 4 embeddings in parallel if 4 GPUs are available. 
  #---        $gpu should be set appropriately to use a specific GPU. 
  #-------------------------------------
  for p in 5 9; do    # region size 5, 9
    for n in 1 3; do  # 1: uni-grams, 3: {1,2,3}-grams
      #---  Unsupervised embedding training.
      if [ "$n" = 1 ]; then pnm=p${p}; else pnm=p${p}${trikw}; fi  # pnm is used to generate filenames. 
      dpcnn_gen_regions_unsemb;   if [ $? != 0 ]; then echo $shnm: dpcnn_gen_regions_unsup failed.; exit 1; fi
      dpcnn_train_unsemb;         if [ $? != 0 ]; then echo $shnm: dpcnn_train_unsemb failed.; exit 1; fi
      rm -f ${tmpdir}/${nm}-td-uns*${pnm}.*  # clean up

      #---  Generate region file for a side layer for the final training.
      #---- NOTE: dpcnn_gen_regions_side must be called AFTER dpcnn_gen_regions_unsemb with the same parameters.
      dpcnn_gen_regions_side; if [ $? != 0 ]; then echo $shnm: dpcnn_gen_regions_side failed.; exit 1; fi
    done
  done


  #-------------------------------------
  #---  Step 2. Final training using unsupervised embeddings. 
  #-------------------------------------
  #---  Prepare input files. 
  p=$p0; pnm=p${p}
  dpcnn_gen_regions   # prepare regions file for layer-0, target file, etc.
  if [ $? != 0 ]; then echo $shnm: dpcnn_gen_regions failed.; exit 1; fi

  #---  Training
  spnms=( p5 p5${trikw} p9 p9${trikw} ) # 4 unsupervised embeddings
  mynm=dpcnn-uemb-${nm}-${lays}-p${p}-ss${ss}-${topdo}
# do_save=1  # Save models in files. 
  do_save=0  # Don't save models in files. 
  dpcnn_training   
  if [ $? != 0 ]; then echo dpcnn_training failed.; exit 1; fi


  #-------------------------------------
  #---  Step 3. Apply the models saved after $tepo epochs to test data. 
  #---  Prediction values are written to: 
  #---     ${outdir}/${mynm}.epo${tepo}.ReNet.pred.{txt|bin}
  #---  NOTE: To do this, change "do_save=0" above to "do_save=1" 
  #---        when training and uncomment the following lines. 
#  epochs=$tepo
#  do_write_text=1  # *.txt 
## do_write_text=0  # *.bin
#  dpcnn_predict 
#  if [ $? != 0 ]; then echo dpcnn_predict failed.; exit 1; fi

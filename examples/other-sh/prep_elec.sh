#!/bin/bash
  ####  This script does data preprocessing for the Elec experiments with 25K training data points. 
  ####  You don't have to run this script since the preprocessed files are included in the package.  

  #---  Step 1. Download data 
  txtdir=../elec

  #---  Step 3. Generate token files (data/*.tok; one review per line and tokens are separated by white space)
  echo Generating token files ... 

  dir=data
  cp -p ${txtdir}/elec-25k-train.txt ${dir}/
  cp -p ${txtdir}/elec-test.txt ${dir}/

  perl to_tokens.pl ${dir}/elec-25k-train.txt data/imdb_registered2.txt .tok & 
  perl to_tokens.pl ${dir}/elec-test.txt      data/imdb_registered2.txt .tok & 
  wait
  echo Done ... 
 

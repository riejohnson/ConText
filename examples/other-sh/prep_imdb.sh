#!/bin/bash
  ####  This script does data preprocessing for the IMDB experiments. 
  ####  You don't have to run this script since the preprocessed files are included in the package.  

  #---  Step 1. Donwload aclImdb_v1.tar.gz from http://ai.stanford.edu/~amaas/data/sentiment/ and unzip it.  

  #---  Step 2. Generate text files (data/*.txt; one review per line) and label files (data/*.cat; one review per line).  
  orgdir=aclImdb  # from aclImdb_v1.tar.gz 
  for t in train test; do
    for pn in pos neg; do
      cat=$pn
      nm=imdb-${t}-${pn}
      where=${orgdir}/${t}/${pn}
      perl extract_text-imdb.pl lst/${nm}.lst ${where}/ $cat data/${nm}.txt data/${nm}.cat
    done
  done

  cat_dic=data/imdb_cat.dic  # list of labels 
  echo neg >  $cat_dic 
  echo pos >> $cat_dic

  #---  Step 3. Generate token files (data/*.tok; one review per line and tokens are separated by white space)
  echo Generating token files ... 
  perl to_tokens.pl data/imdb-train-pos.txt data/imdb_registered2.txt .tok &
  perl to_tokens.pl data/imdb-train-neg.txt data/imdb_registered2.txt .tok &
  perl to_tokens.pl data/imdb-test-pos.txt  data/imdb_registered2.txt .tok &
  perl to_tokens.pl data/imdb-test-neg.txt  data/imdb_registered2.txt .tok &
  wait
  echo Done ... 

  for set in train test; do
    for ext in .txt.tok  .cat; do
      #---  merge the files for positive and negative. 
      mv data/imdb-${set}-pos${ext} data/imdb-${set}${ext}
      cat data/imdb-${set}-neg${ext} >> data/imdb-${set}${ext}
      rm data/imdb-${set}-neg${ext}
    done
  done

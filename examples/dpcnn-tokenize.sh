#!/bin/bash
#****************************************
# ld_split_tokenize (a functino in ld-functions.sh) does the following: 
# - Extract text and labels from csv files of Yelp.p etc. 
# - Split training data into a training portion and a validation portion. 
# - Tokenize text. 
# - Generate a label dictionary. 
#
#   * Naming conventions for output files ( see data/s-dp.* for example) 
#     - training data:   ${nm}-td.${no}of${bb}.{txt.tok|cat} where bb is the number of batches, no=1,2,...,bb
#     - validation data: ${nm}-dv.{txt.tok|cat}
#     - test data:       ${nm}-test.{txt.tok|cat}
#     - label dictionary:${nm}.catdic
#****************************************

  #-----------------#
  source sh-common.sh    # Constants
  source ld-functions.sh # Functions for large data.
  #-----------------#
  ddir=csv-data # <= Change this to where the csv data directories are
  max_jobs=5    # <= Change this to the desired number of background processes invoked at once. 

  #---  parameters for ld_split_tokenize 
  dev=10  # Generate a development set of 10K documents from the training set. 
  reg=data/imdb_registered2.txt
  odir=gendata; if [ ! -e $odir ]; then mkdir $odir; fi # output directory. 

  # nm: output pathnames will be ${tmpdir}/${onm}.* 
  # num: number of training documents divided by 1000
  # bb: number of batches to be generated.  
  # do_extract: extract text and labels from csv files. 

#---  uncomment one of the following 8 lines
# orgdir=${ddir}/yelp_review_polarity_csv; nm=yelppol;  num=560;  bb=5
# orgdir=${ddir}/yelp_review_full_csv;     nm=yelpfull; num=650;  bb=5
# orgdir=${ddir}/dbpedia_csv;              nm=dbpedia;  num=560;  bb=5; do_dbpedia=1
# orgdir=${ddir}/amazon_review_polarity_csv; nm=amapol; num=3600; bb=10
# orgdir=${ddir}/amazon_review_full_csv;   nm=amafull;  num=3000; bb=10
# orgdir=${ddir}/sogou_news_csv;           nm=sog10;    num=450;  bb=10
# orgdir=${ddir}/yahoo_answers_csv;        nm=yah10;    num=1400; bb=10
# orgdir=${ddir}/ag_news_csv;              nm=ag;       num=120;  bb=1

  #---  - Extract text and labels from csv files and generate a label dictionary.
  #---  - Split training data and the training portion and validation (development) portion. 
  #---  - Tokenize text.
  ld_split_tokenize  
  if [ $? != 0 ]; then echo $shnm: ld_split_tokenize failed.; exit 1; fi


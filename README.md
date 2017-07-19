## ConText v4.00: C++ program for neural networks for text categorization
ConText v4.00 provides a C++ implementation of neural networks for text categorization described in:    
* [JZ15a] [Effective use of word order for text categorization with convolutional neural networks](https://aclweb.org/anthology/N/N15/N15-1011.pdf).   Rie Johnson and Tong Zhang.  NAACL HLT 2015.    
* [JZ15b] [Semi-supervised convolutional neural networks for text categorization via region embedding](https://papers.nips.cc/paper/5849-semi-supervised-convolutional-neural-networks-for-text-categorization-via-region-embedding).  Rie Johnson and Tong Zhang.  NIPS 2015.  
* [JZ16] [Supervised and semi-supervised text categorization using LSTM for region embeddings](http://proceedings.mlr.press/v48/johnson16.pdf).  Rie Johnson and Tong Zhang.  ICML 2016.   
* [JZ17] [Deep pyramid convolutional neural networks for text categorization](http://riejohnson.com/paper/dpcnn-acl17.pdf).  Rie Johnson and Tong Zhang.  ACL 2017.   

ConText v4.00 is available at http://riejohnson.com/cnn_download.html. 

**_System Requirements_**: This software runs only on a CUDA-capable GPU such as Tesla K20.  That is, your system **must have a GPU** and an appropriate version of CUDA installed.  The provided `makefile` and example shell scripts are for Unix-like systems.  Testing was done on Linux.  In principle, the C++ code should compile and run also in other systems (e.g., Windows), but no guarantee.  See [`README`](README) for more details.   

**_Download & Documentation_**: See http://riejohnson.com/cnn_download.html#download.  

**_Getting Started_**
1. Download the code and extract the files, and read [`README`](README) (not `README.md`).  
2. Go to the top directory and build executables by entering `make`, after customizing `makefile` as needed.  
  (If you downloaded from GitHub, `make` also decompresses sample text files that exceed GitHub file size limit 
   and does `chmod +x` on shell scripts.) 
3. To confirm installation, go to `examples/` and enter `./sample.sh`.  
  (See [`README`](README) for installation trouble shooting.) 
4. Read Section 1 (Overview) of [User Guide](http://riejohnson.com/software/conText-v4-ug.pdf) to get an idea. 
5. Try some shell scripts at `examples/`.  There is a table of the scripts in Section 1.5 of 
[User Guide](http://riejohnson.com/software/conText-v4-ug.pdf). 

**_Data Source_**: The data files were derived from [Large Move Review Dataset (IMDB)](http://ai.stanford.edu/~amaas/data/sentiment/) 
[MDPHN11] and [Amazon reviews](http://snap.stanford.edu/data/web-Amazon.html) [ML13]. 

**_Licence_**: This program is free software issued under the [GNU General Public License V3](http://www.gnu.org/copyleft/gpl.html). 

**_References_**   
[MDPHN11] Andrew L. Maas, Raymond E. Daly, Peter T. Pham, Dan Huang, Andrew Y. Ng, and Christopher Potts.  Learning word vectors for sentiment analysis.  ACL 2011.   
[ML13] Julian McAuley and Jure Leskovec.  Hidden factors and hidden topics: understanding rating dimensions with review text.  RecSys 2013.   

**_Pull requests_**: This GitHub repository provides a snapshot of research code, which is constantly changing elsewhere for research purposes.  For this reason, it is very likely that pull requests will be declined. 

# COEN 240 Machine Learning - Final Project
This repo is the final project for COEN 240 - Machine Learning at Santa Clara University, Fall 2020

## Prerequisite / Installation:
- Python 3.6 or install Anaconda
- numpy
- gensim
- sklearn
- ntlk 
    nltk.download('stopwords')
    nltk.download('punkt')

## Input:
- Please download the data at http://qwone.com/~jason/20Newsgroups/20news-18828.tar.gz and extract the 20news-18828 folder to the project folder


## Usage:
- The project to be ran in the following order:
    Part 1 & 2  -`importData.py` to generate vocabulary set, vocabulary dictionary and statistics set in numpy pickle
                -(optional) `tfidfMatrixGenerator.py` to generate the csv files for the tf-idf matrix. please follow closely instructions to run this files
    
    Part 3      -`lda.py` to perform lda on the dataset.
    
    Part 4      -`doc2vec.py` to perform doc2vec on the dataset.
    
    Part 5      -`kmeans-bow.py` to perform kmeans from bags of words representation
                -`kmeans-tfidf.py` to perform kmeans from tf-idf representation 
                -`kmeans-doc2vec.py` to perform kmeans from doc2vec representation
                -`kmeans-topic.py` to perform kmeans from topic distribution generated from lda 
    
    Part 7      -`SVMs.py` to perform support vector machine classification for the dataset from tf-idf representation 
    
    Part 8      -`ExtraCredit.py` to perform multinominal naieve bayesian classification for the dataset from tf-idf represenation 
## Outputs:
    - details about outputs are in each file's descriptiion 

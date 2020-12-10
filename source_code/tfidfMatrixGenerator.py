"""
Santa Clara University

COEN 240 - Machine Learning

Final Project

Quan Bach
Anh Truong

By running this file, the weight matrix will be contructed based on TF-IDF algorithm.
NOTICE: due to the hardware constraints the documents are divided into batch of 5000 for processing.

*** ATTENTION *** this program is not optimized. Therefore, running this script might take more than 50 mininutes - depends on the hardware. Please refer to the TF-IDF folder for the generated matrices. If folder does not exist, this script must be ran.

Author hardware configuration:
    OS:         macOS Catalina
    Processor:  2.5Ghz Quad-Core Intel Core i7
    Memory:     16 GB 1600 MHz DDR3

!!! before running this file: 
    - run importData.py and uncomment line 142 to generate content_list
    - if encounter error due to exceeding memory, please add the provided ./utils/content_list.npy to folder npy 

*** ONLY RUN THIS FILE IF YOU WANT TO SEE THE TF-IDF TABLES IN CSV FORMAT ***
"""

import math
import os 
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np


# function to get unique values 
def unique(list1): 
    x = np.array(list1) 
    print(np.unique(x)) 

if __name__ == "__main__":
    
    if not os.path.exists('./tf-idf/'):
            os.makedirs('./tf-idf/')
    
    content_list = np.load('./npy/content_list.npy', allow_pickle=True)


    partition = math.ceil(len(content_list)/5000)
    vectorizer = TfidfVectorizer() # initialize vectorizer for the algorithm
    
    vocab_tfidf = []
    # partiioning the corprus into batch of max size 5000 each
    for idx in range(0,partition):
        if idx == partition - 1:
            mylist = content_list[idx*5000:]
        else:
            mylist = content_list[idx*5000 : (idx + 1)*5000]
            
        vectors = vectorizer.fit_transform(mylist)
        feature_names = vectorizer.get_feature_names()
        vocab_tfidf += feature_names
        dense = vectors.todense()
        denselist = dense.tolist()
        df = pd.DataFrame(denselist, columns=feature_names)
        compression_opts = dict(method='zip',archive_name= ('tf-idf' + str(idx+1) + '.csv'))
        df.to_csv(('./tf-idf/tf-idf' + str(idx+1) + '.zip'),index = False, compression=compression_opts)
        print('Saved ' + './tf-idf/tf-idf' + str(idx+1) + '.zip to tf-idf folder')
        
    
    vocab_tfidf = unique(vocab_tfidf) 
    np.save('./npy/vocab_tfidf.npy',vocab_tfidf)
    print('Saved vocab_tfidf.npy to npy folder')
    
  

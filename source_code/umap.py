# -*- coding: utf-8 -*-
"""
Created on Thu Dec 10 13:46:12 2020

@author: QB
"""
import pandas as pd
import numpy as np

import umap
import umap.plot

# Used to get the data
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

# Some plotting libraries
import matplotlib.pyplot as plt

#m matplotlib notebook
from bokeh.plotting import show, save, output_notebook, output_file
from bokeh.resources import INLINE
output_notebook(resources=INLINE)


if __name__ == '__main__':
    
    
    newsgroup_train = fetch_20newsgroups(subset='train', shuffle=True,remove=('headers','footers','quotes'), random_state=42)

    tfidf_vectorizer = TfidfVectorizer(min_df=5, stop_words='english')

    tfidf_word_doc_matrix = tfidf_vectorizer.fit_transform(newsgroup_train.data)
    
    tfidf_embedding = umap.UMAP(metric='hellinger',low_memory=True).fit_transform(tfidf_word_doc_matrix)
    
    category_labels = [newsgroup_train.target_names[x] for x in newsgroup_train.target]
    hover_df = pd.DataFrame(category_labels, columns=['category'])
    
    # show(f)
    fig = umap.plot.points(tfidf_embedding, labels=hover_df['category'])

"""
Santa Clara University

COEN 240 - Machine Learning

Final Project

Quan Bach
Anh Truong



By running thi file: 
    - Kmeans cluster with topic distibution will be performed and shown in output
    - true_k value can be modify for experimenting 
    - a plot will be saved to KMeans folder 
    
"""

import os
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

if __name__ == '__main__':
    
    # create the directory/folder if not yet exist
    if not os.path.exists('./KMeans/'):
        os.makedirs('./KMeans/')
           
    # import topic distibution by lda-tfidf 
    topic_dist = np.load('./npy/lda_tfidf_topic_distribution_20k.npy')
    print('...')
    print('Loaded topic distribution matrix')
    print('...')
    
    print ('Kmeans training with topic distribution in progress...')
    print('...')


    # numbers of clusters 
    true_k = 20
    model = KMeans(n_clusters=true_k, init='k-means++', max_iter=100, n_init=1)
    
    new_array = topic_dist[:,:,1]
    
    M = model.fit(new_array)
    
 
    #PCA
    pca = PCA(n_components=2).fit(new_array)
    datapoint = pca.transform(new_array)
    plt.figure
    plt.scatter(datapoint[:, 0], datapoint[:, 1], c='DarkBlue')

    # plot the centroids
    plt.scatter(
        model.cluster_centers_[:, 0], model.cluster_centers_[:, 1],
        s=250, marker='*',
        c='red', edgecolor='black',
        label='centroids'
        )
    plt.legend(scatterpoints=1)
    plt.grid()
    plt.savefig('./KMeans/kmeans_topic_dist.png')
    print('Saved plot to KMeans...')
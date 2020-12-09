"""
Santa Clara University

COEN 240 - Machine Learning

Final Project

Quan Bach
Anh Truong



By running thi file: 
    - Kmeans cluster with tf-idf will be performed and shown in output
    - true_k value can be modify for experimenting 
    - a txt file will be saved with the kmeans clustering info 
    - a plot will be saved to KMeans folder 
    - a grouping of clusters will be shown in output
    
*** NOTICE ***: this script will dowload the 20 news group dataset from sklearn and perform processing directly here before the lda models. 
Therefore, it does not used the vocabularies generated by other files previously. However, the content should be the same. 


"""

from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import os
import matplotlib.pyplot as plt



if __name__ == '__main__':
    if not os.path.exists('./KMeans/'):
           os.makedirs('./KMeans/')
    
    
    # load data 
    newsgroups_train = fetch_20newsgroups(subset='train', remove=('headers', 'footers', 'quotes'), shuffle = True)    
    print('Dataset loaded...')
    print('...')


    print('KMean with TF-IDF in progress')
    print('...')
    vectorizer = TfidfVectorizer(max_df=0.5,min_df=0.05,stop_words='english')

    X = vectorizer.fit_transform(newsgroups_train.data)
    true_k = 12

    model = KMeans(n_clusters=true_k, init='k-means++', max_iter=100, n_init=1)
    model.fit(X)
    
    print("Top terms per cluster:")

    order_centroids = model.cluster_centers_.argsort()[:, ::-1]

    terms = vectorizer.get_feature_names()
    
    out_file = open('./KMeans/kmean-tfidf.txt','w')
    for i in range(true_k):
        print ("Cluster %d:" % i)
        out_file.write("Cluster %d:" % i)
        for ind in order_centroids[i, :10]:
            print (' %s' % terms[ind])
            out_file.write(' %s' % terms[ind])
        print ()
        out_file.write('\n')
    
    out_file.close()
    print('Saved kmean-tfidf to KMean folder...')
    
    # plot
    X = X.toarray()
    y_km = model.fit_predict(X)
    
    # plot the 12 clusters
    for i in range (0,true_k):
        plt.scatter(
           X[y_km == i, 0], X[y_km == i, 1],
           s=50, c='lightgreen',
           marker='s', edgecolor='black',
           label='cluster ' + str(i)
        )

    # plot the centroids
    plt.scatter(
        model.cluster_centers_[:, 0], model.cluster_centers_[:, 1],
        s=250, marker='*',
        c='red', edgecolor='black',
        label='centroids'
        )
    plt.legend(scatterpoints=1)
    plt.grid()
    plt.savefig('./KMeans/kmeans_tfidf.png')
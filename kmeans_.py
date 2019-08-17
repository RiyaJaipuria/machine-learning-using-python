# kmeans

#import the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#import the dataset with pandas
dataset = pd.read_csv('Mall_Customers.csv')
X = dataset.iloc[:, [3,4]].values

#using elbow method to find optimal number of clusters
from sklearn.cluster import KMeans
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters = i,init = 'k-means++', max_iter = 300, n_init = 10, random_state = 0)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)
plt.plot(range(1, 11), wcss)
plt.title('the elbow method')
plt.xlabel('number of clusters')
plt.ylabel('WCSS')
plt.show()

#applying k-means on the dataset
kmeans = KMeans(n_clusters = 5, init = 'k-means++',  max_iter = 300, n_init = 10, random_state = 10)
y_kmeans = kmeans.fit_predict(X)

#visualizing the clusters
plt.scatter(X[y_kmeans == 0, 0], X[y_kmeans ==0,1], s = 100,c = 'red', label = 'careless')
plt.scatter(X[y_kmeans == 1, 0], X[y_kmeans ==1,1], s = 100,c = 'blue', label = 'standard')
plt.scatter(X[y_kmeans == 2, 0], X[y_kmeans ==2,1], s = 100,c = 'green', label = 'target')
plt.scatter(X[y_kmeans == 3, 0], X[y_kmeans ==3,1], s = 100,c = 'cyan', label = 'sensible')
plt.scatter(X[y_kmeans == 4, 0], X[y_kmeans ==4,1], s = 100,c = 'magenta', label = 'careful')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s =300, c = 'yellow', label = 'centroids')
plt.title('cluster of clients')
plt.xlabel('annual income (k$)')
plt.ylabel('spending score (1-100)')
plt.legend()
plt.show()

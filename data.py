import pandas as pd
import plotly.express as px
from sklearn.cluster import KMeans as km
import matplotlib.pyplot as plt
import seaborn as sb
import csv

data = pd.read_csv("data.csv")
scatter = px.scatter(data, x="petal_size", y="sepal_size")
#scatter.show()
print(data.head())

#to choose the number of clusters using WCSS
X = data.iloc[:,[0, 1]]
wcss = []
#for i in range (1, 11):
    #kmeans = km(n_clusters = i, init = "k-means++", random_state = 27)

    #to fit the data into the model
    #kmeans.fit(X)

    #to append the inertia of kmeans to return the wcss of that model
    #wcss.append(kmeans.inertia_)

#to plot a line chart to show an elbow-like structure on the graph
"""plt.figure(figsize=(10, 5))
sb.lineplot(range(1, 11), wcss, marker='o', color = 'blue')
plt.title('Elbow Method')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS')"""
#plt.show()

#to find kmeans with 3 clusters
kmeans = km(n_clusters = 3, init = "k-means++", random_state = 27)
Y = kmeans.fit_predict(X)

#to create a scatterplot for the clusters with different colors
plt.figure(figsize=(15, 7))
sb.scatterplot(X[Y==0,0], X[Y==0,1], color = "red", label = "Cluster 1")
sb.scatterplot(X[Y==1,0], X[Y==1,1], color = "blue", label = "Cluster 2")
sb.scatterplot(X[Y==2,0], X[Y==2,1], color = "yellow", label = "Cluster 3")
sb.scatterplot(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1], color = 'red', label = "Centroids", s = 100, marker = ',')
plt.title('Cluster of Flowers')
plt.xlabel('Petal Size')
plt.ylabel('Sepal Size')
plt.legend()
plt.show()
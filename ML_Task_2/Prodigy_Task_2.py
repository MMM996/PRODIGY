# This is the code for Task 2 Prodigy Techinfo internship
# Task 2: Customer group clustering 
# The Code is written by Muhammad Mudassir Majeed
# The date is Jan-24.
# Dataset: https://www.kaggle.com/datasets/vjchoudhary7/customer-segmentation-tutorial-in-python

#-----------------------------------------------------------------------------#

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Part 1: Load dataset
data = pd.read_csv('Task 2\\Mall_Customers.csv')

# Part 2: Premilinary EDA
pd.set_option('display.max_columns',None)
data.head()
data.info()
data.describe()

# Check for Null Values
data.isnull().sum()     # No Null Values

# Check for Duplicate Values
data.duplicated().sum()     # No Duplicates

# Part 3: Pre-processing
# Drop Unnecessary Columns
data = data.drop(['CustomerID'], axis = 1)
data.info()

# Encoding of non-Numeric values
from sklearn.preprocessing import LabelEncoder
label = LabelEncoder()
data['Gender'] = label.fit_transform(data['Gender'])
data.info()
data.head()

# Define Train Data
X = data 

# Standardize
from sklearn.preprocessing import StandardScaler
scalar = StandardScaler()
X = scalar.fit_transform(X)

# Part 4: Model Training
# K-Means
from sklearn.cluster import KMeans
Kmeans = KMeans(n_clusters=4, random_state = 42)
Kmeans.fit(X)
Kmeans_labels = Kmeans.predict(X)

# K-Means ++
Kmeans_pp = KMeans(n_clusters=4, init='k-means++', random_state=42)
Kmeans_pp.fit(X)
Kmeans_pp_labels = Kmeans_pp.predict(X)


# Part 5: Model Evaluation

# Inertia
inertia_kmeans = Kmeans.inertia_
inertia_Kmeans_pp = Kmeans_pp.inertia_

# Silhouette Score
from sklearn.metrics import silhouette_score
silhouette_Kmeans = silhouette_score(X, Kmeans.labels_)
silhouette_Kmeans_pp = silhouette_score(X, Kmeans_pp.labels_)

# Calinski-Harabasz Index
from sklearn.metrics import calinski_harabasz_score
Calinski_Kmeans = calinski_harabasz_score(X, Kmeans.labels_)
Calinski_Kmeans_pp = calinski_harabasz_score(X, Kmeans_pp.labels_)

# Davies-Bouldin Index
from sklearn.metrics import davies_bouldin_score
davies_Kmeans = davies_bouldin_score(X, Kmeans.labels_)
davies_Kmeans_pp = davies_bouldin_score(X, Kmeans_pp.labels_)

# Data in Table
from tabulate import tabulate
data_table = [
            ['Kmeans', inertia_kmeans, silhouette_Kmeans, Calinski_Kmeans, davies_Kmeans],
            ['Kmeans++', inertia_Kmeans_pp, silhouette_Kmeans_pp, Calinski_Kmeans_pp, davies_Kmeans_pp]
            ]

headers = ['Model Name', 'Inertia','Silhouette Score', 'Calinski Index', 'Davies Index']

table = tabulate(data_table, headers= headers, tablefmt ='fancy_grid', floatfmt={'.2f','.2f','.2f',
                                                                                '.2f','.2f'})
print(table)

# Visualize Clusters
plt.figure(figsize=(12, 5))

# Plot K-means clusters
plt.subplot(1, 2, 1)
plt.scatter(X[:, 0], X[:, 1], c=Kmeans_labels, cmap='viridis', edgecolors='k', s=50)
plt.scatter(Kmeans.cluster_centers_[:, 0], Kmeans.cluster_centers_[:, 1], c='red', marker='X', s=200, label='Centroids')
plt.title('K-means Clustering')
plt.legend()

# Plot K-means++ clusters
plt.subplot(1, 2, 2)
plt.scatter(X[:, 0], X[:, 1], c=Kmeans_pp_labels, cmap='viridis', edgecolors='k', s=50)
plt.scatter(Kmeans_pp.cluster_centers_[:, 0], Kmeans_pp.cluster_centers_[:, 1], c='red', marker='X', s=200, label='Centroids')
plt.title('K-means++ Clustering')
plt.legend()

plt.show()



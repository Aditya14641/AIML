
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from sklearn.cluster import AgglomerativeClustering
from mpl_toolkits.mplot3d import Axes3D

data = pd.read_csv('Mall_Customers.csv')

print(data.head())


X = data[['Annual Income (k$)', 'Spending Score (1-100)']]


wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, random_state=42, n_init=10)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)


plt.figure(figsize=(8,5))
plt.plot(range(1,11), wcss, marker='o')
plt.title('Elbow Method for Optimal k')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('WCSS')
plt.grid()
plt.show()

# From elbow plot, choose k=5
k = 5

\
kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
data['KMeans_Cluster'] = kmeans.fit_predict(X)


plt.figure(figsize=(8,6))
sns.scatterplot(x='Annual Income (k$)', y='Spending Score (1-100)',
                hue='KMeans_Cluster', data=data, palette='Set1')
plt.title('Customer Segments using KMeans')
plt.legend()
plt.grid()
plt.show()


linked = linkage(X, method='ward')

plt.figure(figsize=(10,7))
dendrogram(linked)
plt.title('Dendrogram')
plt.xlabel('Customers')
plt.ylabel('Euclidean distances')
plt.show()

# From dendrogram, choose 5 clusters

hierarchical = AgglomerativeClustering(n_clusters=5, linkage='ward')
data['Hierarchical_Cluster'] = hierarchical.fit_predict(X)

# Visualize Hierarchical Clusters
plt.figure(figsize=(8,6))
sns.scatterplot(x='Annual Income (k$)', y='Spending Score (1-100)',
                hue='Hierarchical_Cluster', data=data, palette='Set2')
plt.title('Customer Segments using Hierarchical Clustering')
plt.legend()
plt.grid()
plt.show()


le = LabelEncoder()
data['Gender_Numeric'] = le.fit_transform(data['Genre'])


profile = data.groupby('KMeans_Cluster').agg({
    'Age': ['mean', 'min', 'max'],
    'Gender_Numeric': 'mean',
    'Annual Income (k$)': ['mean'],
    'Spending Score (1-100)': ['mean']
}).round(1)

print("\nCluster Profiling (based on KMeans Clusters):")
print(profile)


fig = plt.figure(figsize=(10,8))
ax = fig.add_subplot(111, projection='3d')

sc = ax.scatter(data['Age'], data['Annual Income (k$)'], data['Spending Score (1-100)'],
                c=data['KMeans_Cluster'], cmap='Set1', s=50)

ax.set_xlabel('Age')
ax.set_ylabel('Annual Income (k$)')
ax.set_zlabel('Spending Score (1-100)')
ax.set_title('3D View of Customer Segments')
plt.colorbar(sc)
plt.show()

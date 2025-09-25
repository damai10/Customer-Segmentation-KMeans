#!/usr/bin/env python
# coding: utf-8

# Import the Dependencies

# In[17]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings("ignore", category=UserWarning)


# Data collection and Analysis

# In[18]:


#loading the data from csv file to a Pandas Dataframe
customer_data=pd.read_csv('Mall_Customers.csv')


# In[19]:


#first five rows in the dataframe
customer_data.head()


# In[20]:


#finding the number of rows and columns
customer_data.shape


# In[21]:


#getting some information about the dataset
customer_data.info()


# In[22]:


#checking for the missing value
customer_data.isnull().sum()


# In[23]:


# Age distribution
plt.figure(figsize=(6,4))
sns.histplot(customer_data['Age'], bins=20, kde=True, color='blue')
plt.title("Age Distribution")
plt.show()


# In[24]:


#Annual Income Distribution
plt.figure(figsize=(6,4))
sns.histplot(customer_data['Annual Income (k$)'], bins=20, kde=True, color='green')
plt.title("Annual Income Distribution")
plt.show()


# In[25]:


# Spending Score distribution
plt.figure(figsize=(6,4))
sns.histplot(customer_data['Spending Score (1-100)'], bins=20, kde=True, color='orange')
plt.title("Spending Score Distribution")
plt.show()


# In[26]:


plt.figure(figsize=(5,5))
sns.countplot(x='Gender', data=customer_data, palette='Set2')
plt.title("Gender Distribution")
plt.show()


# In[27]:


plt.figure(figsize=(6,4))
sns.boxplot(x='Gender', y='Annual Income (k$)', data=customer_data, palette='Set3')
plt.title("Income by Gender")
plt.show()

plt.figure(figsize=(6,4))
sns.boxplot(x='Gender', y='Spending Score (1-100)', data=customer_data, palette='Set3')
plt.title("Spending Score by Gender")
plt.show()


# Choosing the Annual Income Column & Spending Score column

# In[28]:


X = customer_data[["Age", "Annual Income (k$)", "Spending Score (1-100)"]].values


# In[29]:


print(X)


# Choosing the number of clusters

# WCSS --> Within Clusters Sum of Squares

# In[30]:


#finding wcss value for different number of clusters
WCSS = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
    kmeans.fit(X)
    WCSS.append(kmeans.inertia_)


# In[31]:


# Plot an elbow graph
plt.figure(figsize=(6, 4))
plt.plot(range(1, 11), WCSS, marker='o')
plt.title("The Elbow Method")
plt.xlabel("Number of clusters")
plt.ylabel("WCSS")
plt.show()


# Optimum Number of Clusters = 5

# Training the k-Means Clustering Model

# In[32]:


kmeans = KMeans(n_clusters=5, init='k-means++',random_state=0)

#return a label for each data point based on their structure
Y = kmeans.fit_predict(X)
print(Y)


# Clusters-0,1,2,3,4

# In[33]:


# Save clustered data 
customer_data.to_csv("Customer_Segments.csv", index=False)


# Visualizing the Clusters

# In[38]:


#plotting all the clusters and their Centroids

plt.figure(figsize=(8,8))
plt.scatter(X[Y==0,0],X[Y==0,1],s=50,c='green',label='Cluster 1')
plt.scatter(X[Y==1,0],X[Y==1,1],s=50,c='red',label='Cluster 2')
plt.scatter(X[Y==2,0],X[Y==2,1],s=50,c='yellow',label='Cluster 3')
plt.scatter(X[Y==3,0],X[Y==3,1],s=50,c='violet',label='Cluster 4')
plt.scatter(X[Y==4,0],X[Y==4,1],s=50,c='blue',label='Cluster 5')

#plot the centeroids
plt.scatter(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1],s=100,c='black',label='Centroids')

plt.title('Scatter plot of Customer Segmentation (Income vs Spending Score)')
plt.xlabel('Annual Income')
plt.ylabel('Spending Score')
plt.legend()
plt.show()


# In[39]:


#plotting all the clusters and their Centroids

plt.figure(figsize=(8,8))
plt.scatter(X[Y==0,0],X[Y==0,1],s=50,c='green',label='Cluster 1')
plt.scatter(X[Y==1,0],X[Y==1,1],s=50,c='red',label='Cluster 2')
plt.scatter(X[Y==2,0],X[Y==2,1],s=50,c='yellow',label='Cluster 3')
plt.scatter(X[Y==3,0],X[Y==3,1],s=50,c='violet',label='Cluster 4')
plt.scatter(X[Y==4,0],X[Y==4,1],s=50,c='blue',label='Cluster 5')

#plot the centeroids
plt.scatter(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,2],s=100,c='black',label='Centroids')

plt.title('Scatter plot of Customer Segmentation (Income vs Spending Score)')
plt.xlabel('Annual Income')
plt.ylabel('Spending Score')
plt.legend()
plt.show()


# In[40]:


#plotting all the clusters and their Centroids

plt.figure(figsize=(8,8))
plt.scatter(X[Y==0,0],X[Y==0,1],s=50,c='green',label='Cluster 1')
plt.scatter(X[Y==1,0],X[Y==1,1],s=50,c='red',label='Cluster 2')
plt.scatter(X[Y==2,0],X[Y==2,1],s=50,c='yellow',label='Cluster 3')
plt.scatter(X[Y==3,0],X[Y==3,1],s=50,c='violet',label='Cluster 4')
plt.scatter(X[Y==4,0],X[Y==4,1],s=50,c='blue',label='Cluster 5')

#plot the centeroids
plt.scatter(kmeans.cluster_centers_[:,1],kmeans.cluster_centers_[:,2],s=100,c='black',label='Centroids')

plt.title('Scatter plot of Customer Segmentation (Income vs Spending Score)')
plt.xlabel('Annual Income')
plt.ylabel('Spending Score')
plt.legend()
plt.show()


# In[ ]:





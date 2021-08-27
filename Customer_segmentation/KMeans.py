#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, normalize ,MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA


# In[2]:


creditcard_df= pd.read_csv('/Users/sana/Desktop/Academics/Projects/Customer_Segmentation/Credit_card_data.csv')


# In[3]:


creditcard_df


# In[4]:


type(creditcard_df)
creditcard_df.info()


# In[5]:


creditcard_df.describe()


# In[6]:


print(creditcard_df.isnull().sum())


# In[7]:


creditcard_df.loc[(creditcard_df['MINIMUM_PAYMENTS'].isnull() == True), 'MINIMUM_PAYMENTS'] = creditcard_df['MINIMUM_PAYMENTS'].mean()          
creditcard_df.isnull().sum()


# In[8]:


creditcard_df.loc[(creditcard_df['CREDIT_LIMIT'].isnull() == True),'CREDIT_LIMIT']=creditcard_df['CREDIT_LIMIT'].mean() 


# In[9]:


creditcard_df.isnull().sum()


# In[10]:


creditcard_df.isnull().sum().sum()


# In[11]:


creditcard_df.drop('CUST_ID',axis=1,inplace=True)


# In[12]:


creditcard_df


# In[13]:


n = len(creditcard_df.columns)
n


# In[14]:


creditcard_df.columns


# In[15]:


correlations=creditcard_df.corr()
sns.heatmap(correlations,annot=True)


# In[16]:


scaler=StandardScaler()
creditcard_df_scaled=scaler.fit_transform(creditcard_df)


# In[17]:


type(creditcard_df_scaled)


# In[18]:


creditcard_df_scaled


# In[19]:


cost=[]
range_values=range(1,20)
for i in range_values:
    kmeans=KMeans(i)
    kmeans.fit(creditcard_df_scaled)
    cost.append(kmeans.inertia_)
plt.plot(cost)


# In[20]:


kmeans = KMeans(7)
kmeans.fit(creditcard_df_scaled)#find the nearest clusters for given data
labels = kmeans.labels_
labels


# In[21]:


kmeans.cluster_centers_.shape


# In[22]:


cluster_centers=pd.DataFrame(data=kmeans.cluster_centers_,columns=[creditcard_df.columns])
cluster_centers


# In[23]:


# scaler=StandardScaler()
# cluster_centers=scaler.inverse_transform(cluster_centers)
# cluster_centers=pd.DataFrame(data=cluster_centers,columns=[creditcard_df_scaled.columns])
# cluster_centers


# In[24]:


labels.shape


# In[25]:


labels.max()


# In[26]:


labels.min()


# In[27]:


credit_df_cluster=pd.concat([creditcard_df,pd.DataFrame(({'cluster':labels}))],axis=1)
credit_df_cluster


# In[28]:


pca=PCA(n_components=2)
principal_comp=pca.fit_transform(creditcard_df_scaled)


# In[29]:


pca_df=pd.DataFrame(data=principal_comp,columns=['pca1','pca2'])
pca_df


# In[30]:


pca_df=pd.concat([pca_df,pd.DataFrame({'Cluster':labels})],axis=1)
pca_df


# In[31]:


plt.figure(figsize=(10,10))
ax=sns.scatterplot(x='pca1',y='pca2',hue='Cluster',data=pca_df,palette=['yellow','red','blue','pink','orange','black','purple'])
plt.show()


# In[ ]:





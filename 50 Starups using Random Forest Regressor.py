#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[2]:


dataset = pd.read_csv('50_Startups.csv')


# In[3]:


dataset


# In[4]:


dataset.corr()


# In[5]:


dataset.corr()


# In[6]:


import seaborn as sns


# In[7]:


sns.heatmap(dataset.corr(),annot=True)


# In[8]:


dataset.isnull().any()


# In[9]:


x = dataset.iloc[:,0:4].values


# In[10]:


x


# In[11]:


y = dataset.iloc[:,4].values


# In[12]:


y


# In[13]:


from sklearn.compose import ColumnTransformer


# In[14]:


from sklearn.preprocessing import OneHotEncoder


# In[15]:


ct = ColumnTransformer([('oh',OneHotEncoder(),[3])], remainder='passthrough')


# In[16]:


x = ct.fit_transform(x)


# In[17]:


x


# In[18]:


#removing dummy variables

x = x[:,1:]


# In[19]:


x


# In[20]:


x.shape


# In[21]:


from sklearn.model_selection import train_test_split


# In[22]:


x_train , x_test , y_train , y_test = train_test_split(x,y,test_size=0.2,random_state=0)


# In[23]:


x_train


# In[24]:


x_test


# In[25]:


y_train


# In[26]:


y_test


# In[27]:


from sklearn.ensemble import RandomForestRegressor


# In[28]:


rf = RandomForestRegressor(n_estimators=10,random_state=0,n_jobs=-1)


# In[29]:


rf.fit(x_train,y_train)


# In[30]:


y_pred = rf.predict(x_test)


# In[31]:


y_pred


# In[32]:


y_test


# In[33]:


from sklearn.metrics import r2_score


# In[34]:


r2_score(y_test,y_pred)*100


# In[ ]:





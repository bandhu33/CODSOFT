#!/usr/bin/env python
# coding: utf-8

# In[5]:


import pandas as pd 
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns


# In[6]:


iris=pd.read_csv(r"C:\Users\91769\Downloads\archive (3)\IRIS.csv")


# In[7]:


iris


# In[8]:


iris.head()


# In[10]:


iris['species'].value_counts()

#it will classify and make similar groups 


# In[11]:


iris.isnull().any()


# In[72]:


iris['species'].value_counts()

#it will classify and make similar groups 


# In[73]:


iris


# In[13]:


X = iris.drop(['Id', 'Species'], axis=1)
y = iris['Species']

print(X.shape)

print(y.shape)


# In[75]:


iris


# In[81]:


iris.species


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





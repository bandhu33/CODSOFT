#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd 


# In[4]:


titanic = pd.read_csv(r"C:\Users\91769\Downloads\archive (1)\tested.csv")


# In[5]:


titanic


# In[5]:


titanic.shape


# In[6]:


titanic.info()


# In[9]:


ports= pd.get_dummies(titanic.Embarked,prefix='Embarked')
ports.head()


# In[10]:


titanic = titanic.join(ports)


# In[12]:


titanic.drop(['Embarked'], axis=1, inplace=True)


# In[14]:


titanic.head()


# In[16]:


titanic.info()


# In[17]:


titanic.Sex= titanic.Sex.map({'male':0,'female':1})


# In[18]:


titanic


# In[19]:


titanic.columns


# #we will separate our data here because we need two kinds of data one is label and another is features label which our algorithms will predict 

# In[20]:


y = titanic.Survived.copy()
x = titanic.drop(['Survived'],axis = 1)


# In[22]:


x.columns


# In[23]:


x.drop(['Cabin','Ticket','Name','PassengerId'],axis= 1,inplace=True)


# In[24]:


x.info()


# In[26]:


x.isnull().sum()


# In[27]:


x.isnull().values.any()


# In[28]:


x[pd.isnull(x).any(axis=1)]


# In[29]:


x.Age.fillna(x.Age.mean(),inplace=True)


# In[30]:


x.isnull().values.any()


# In[31]:


x[pd.isnull(x).any(axis=1)]


# In[32]:


x.Fare.fillna(x.Fare.mean(),inplace=True)


# In[34]:


x[pd.isnull(x).any(axis=1)]


# In[71]:


from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


# In[63]:


x_train,X_test,y_train,y_test= train_test_split(x,y, test_size=0.2,random_state=7)


# In[39]:


from sklearn.linear_model import LogisticRegression
model = LogisticRegression()


# In[40]:


model


# In[64]:


model.fit(x_train,y_train)


# In[67]:


model.score(x_train,y_train)


# In[68]:


model.score(x_valid,y_valid)


# In[69]:


x_train_prediction = model.predict(x_train)


# In[72]:


training_data_accuracy = accuracy_score(y_train, x_train_prediction)
print('Accuracy score of training data : ', training_data_accuracy)


# In[ ]:


#now we will check for random person 


# In[86]:


other_data = (1,0,24.00,0,1,150,0,1,0)


# In[88]:


import numpy as np


# In[89]:


other_data_as_numpy_array=np.asarray(other_data)


# In[90]:


other_data_reshaped = other_data_as_numpy_array.reshape(1,-1)


# In[91]:


prediction = model.predict(other_data_reshaped)

if prediction[0]==0:
    print("Dead")
if prediction[0]==1:
    print("Alive")


# In[ ]:





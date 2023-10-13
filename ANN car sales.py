#!/usr/bin/env python
# coding: utf-8

# In[7]:


import pandas as pd


# In[8]:


file_path = r"C:\Users\91769\Downloads\archive (4)\car_purchasing.csv"
df = pd.read_csv(file_path, encoding='iso-8859-1')


# In[9]:


df


# In[10]:


df.isnull().any()


# In[11]:


df.isnull().sum()


# In[12]:


df.info()


# In[13]:


df


# In[14]:


#drop unwanted 
df.drop(columns=['customer name', 'customer e-mail', 'country', 'gender'], inplace=True)


# In[15]:


df


# In[16]:


X = df.drop('car purchase amount', axis=1)
y = df['car purchase amount']

#data splitting


# In[17]:


#data scalling 


# In[18]:


from sklearn.preprocessing import MinMaxScaler


# In[19]:


scaler = MinMaxScaler()
X = scaler.fit_transform(X)

y = scaler.fit_transform(y.values.reshape(-1, 1))


# In[20]:


from sklearn.model_selection import train_test_split


# In[21]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)


# In[22]:


print(X_train.shape)
print(y_train.shape)


# In[ ]:





# In[23]:


#ANN
from tensorflow.keras.models import Sequential


# In[24]:


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense


# In[25]:


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model = Sequential()
model.add(Dense(10, activation='relu', input_dim=4))
model.add(Dense(10, activation='relu'))
model.add(Dense(1, activation='linear'))

model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mean_absolute_error'])
model.summary()


# In[26]:


history = model.fit(X_train, y_train, epochs=50, validation_split=0.2)


# In[27]:


y_pred = model.predict(X_test)


# In[28]:


from sklearn.metrics import r2_score

R2 = r2_score(y_test, y_pred)
print("R2 Score=",R2 )


# In[ ]:





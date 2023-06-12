#!/usr/bin/env python
# coding: utf-8

# In[58]:


import numpy as np
import pandas as pd
import scipy 
from scipy import sparse
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt


# In[59]:


pip install mglearn


# In[60]:


from sklearn.datasets import load_iris
iris_dataset = load_iris()


# In[61]:


print ("keys of iris_dataset:\n",iris_dataset.keys()) #here it means the attributes or the column names of the dataset


# In[62]:


print(iris_dataset['DESCR'][:193] + "\n...")


# In[63]:


print(" Target_Names: ", iris_dataset['target_names'])


# In[64]:


print("feature_names : ", iris_dataset['feature_names'])


# In[65]:


print("type of data:" , type(iris_dataset['data']))


# In[66]:


print("shape of data:" ,iris_dataset['data'].shape)


# In[67]:


print("all data\n" ,iris_dataset['data'])


# In[68]:


print("first five\n" ,iris_dataset['data'][:5])


# In[69]:


print("type of target:" ,iris_dataset['target'].shape)


# In[70]:


print("target:\n" ,iris_dataset['target'])


# In[71]:


from sklearn.model_selection import train_test_split


# In[72]:


X_train , X_test, y_train,y_test = train_test_split(iris_dataset['data'],iris_dataset['target'] ,random_state=0)


# In[73]:


print("X_train Shape : " ,X_train.shape)
print("y_train Shape : " ,y_train.shape)


# In[74]:


print("X_test Shape : " ,X_test.shape)
print("y_test Shape : " ,y_test.shape)


# In[75]:


iris_dataframe = pd.DataFrame(X_train , columns=iris_dataset.feature_names)


# In[77]:


import mglearn


# In[80]:


pd.plotting.scatter_matrix(iris_dataframe, c=y_train, figsize=(15, 15), marker='o', 
                           hist_kwds={'bins':20}, s=60, alpha=.8, cmap=mglearn.cm3)


# In[81]:


from sklearn.neighbors import KNeighborsClassifier 


# In[83]:


knn=KNeighborsClassifier(n_neighbors=1)


# In[84]:


knn.fit(X_train ,y_train)


# In[85]:


X_new =np.array([[5,2.9,1,0.2]])
print("X_new.shape:" , X_new.shape)


# In[86]:


prediction = knn.predict(X_new)
print("Prediction:" , prediction)
print("Predicted target name :" , iris_dataset['target_names'][prediction])


# In[89]:


y_pred = knn.predict(X_test)
print("test set Prediction:\n" , y_pred)


# In[90]:


print("test set score:{:.2f}".format(np.mean(y_pred==y_test)))


# In[92]:


print("test set score:{:.2f}".format(knn.score(X_test,y_test)))


# In[ ]:





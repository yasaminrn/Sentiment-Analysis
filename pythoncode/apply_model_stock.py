#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#This notebook is used to apply the trained mode, defined in stock_sentiment_model notebook
#The trained logistic regression model and trained word tokenizer model are saved and loaded using pickle


# In[ ]:


#importing utilities


# In[ ]:


import re
import numpy as np
import pandas as pd


# In[ ]:


#plotting libraries, added for potential testing purpose, not used for prediction
import seaborn as sns
import matplotlib.pyplot as plt


# In[ ]:


#Transforms text to feature vectors that can be used as input to estimator


# In[ ]:


from sklearn.feature_extraction.text import TfidfVectorizer


# In[ ]:


#importing performance matrix for potential testing purpose, not used for prediction
from sklearn.metrics import confusion_matrix, classification_report


# In[ ]:


#importing pickle to load saved models (logistic regression and vectorizer)
import pickle


# In[ ]:


#FILE_NAMES are automatically taken from interface notebook.
#The command below for file name is included for potential single use of this notebook
#FILE_NAME = 'META.csv'


# In[ ]:


get_ipython().run_line_magic('run', 'applying_textprocess.ipynb')


# In[ ]:


X=data['text']


# In[ ]:


#Joining data as the vectorizer works on string


# In[ ]:


X=X.apply(lambda x: ' '.join(x))


# In[ ]:


#applying vectorizer and logistic regression models trained and saved in stock_sentimet_model notebook


# In[ ]:


vectorizer = pickle.load(open('stock_vectorizer.pk','rb'))


# In[ ]:


loaded_model = pickle.load(open('stock_LRmodel.sav', 'rb'))


# In[ ]:


x=data['text'].apply(lambda y: str(np.array(y)))
x


# In[ ]:


prediction = loaded_model.predict(vectorizer.transform(X))


# In[ ]:


#showing sample of prediction for first 100 tweets


# In[ ]:


prediction[:100]


# In[ ]:


np.unique(prediction)


# In[ ]:


#calculating the ratio of positive outputs to total number of tweets andlyzed
#This is used as means of gauging the current sentiment around the companies of interest


# In[ ]:


current_sentiment = sum(prediction)/len(prediction)


#!/usr/bin/env python
# coding: utf-8

# In[1]:


#This notebook is used for reading data and applying the pre-process on tweet data
#The pre-process is the first step to format the data for proper use of Natural language toolkit


# In[2]:


#importing utility libraries


# In[3]:


import re
import numpy as np
import pandas as pd


# In[4]:


#importing dataset


# In[5]:


DATASET_ENCODING = "ISO-8859-1"


# In[6]:


#File location and address


# In[7]:


#File names are taken automatically from interface company_list
#For single use of applying_textprocess, can use the FILE_NAME as shown in exapmle below.
#FILE_NAME = 'META.csv'


# In[8]:


#Defining column titles
#For training/testing data, two columns of text and target are used
#For new data, the scraping data has column labels. For current project we are only interested in 'text' column of new data.


# In[9]:


if re.search('train', FILE_NAME) or re.search('test', FILE_NAME):
    DATASET_COLUMNS=['text','target']


# In[10]:


if re.search('train', FILE_NAME) or re.search('test', FILE_NAME):
    df = pd.read_csv(FILE_NAME, encoding=DATASET_ENCODING, names=DATASET_COLUMNS)
else:
    df = pd.read_csv(FILE_NAME, encoding=DATASET_ENCODING)


# In[11]:


#checking for Null values, in whole table and in text column.


# In[12]:


np.sum(df.isnull().any(axis=1))


# In[13]:


np.sum(df['text'].isnull())


# In[14]:


if re.search('train', FILE_NAME):
    data=df[['text','target']]
elif re.search('test', FILE_NAME):
    data=df[['text','target']]
else: 
    data=df[['text']]


# In[15]:


get_ipython().run_line_magic('run', 'text_process.ipynb')


# In[16]:


#changing all tweets to lower case


# In[17]:


data['text']


# In[37]:


#As we want to overwrite data frame disbaling the default warning for overwrite


# In[18]:


pd.options.mode.chained_assignment = None  # default='warn'


# In[19]:


data['text']=data['text'].str.lower()


# In[20]:


#applying the cleaning_stopwords function defined in text_process notebook


# In[21]:


data['text']=data['text'].apply(lambda text: cleaning_stopwords(text))


# In[22]:


#applying the cleaning_punctuations function defined in text_process notebook


# In[23]:


data['text']=data['text'].apply(lambda x:cleaning_punctuations(x))


# In[24]:


#applying the cleaning_repeating_char function defined in text_process notebook


# In[25]:


data['text']=data['text'].apply(lambda x: cleaning_repeating_char(x))


# In[26]:


#Applying cleaning_URLs function defined in text_process notebook


# In[27]:


data['text']=data['text'].apply(lambda x: cleaning_URLs(x))


# In[28]:


#applying the cleaning_numbers function defined in text_process notebook


# In[29]:


data['text']=data['text'].apply(lambda x: cleaning_numbers(x))


# In[30]:


#applying the word_tokenize function


# In[31]:


data['text']=data['text'].apply(word_tokenize)


# In[32]:


#applying stemming_on_text function defined in text_process notebook


# In[33]:


data['text']=data['text'].apply(lambda x: stemming_on_text(x))


# In[34]:


#applying lemmatizer_on_text function defined in text_process notebook


# In[35]:


data['text']=data['text'].apply(lambda x: lemmatizer_on_text(x))


# In[36]:


data['text']


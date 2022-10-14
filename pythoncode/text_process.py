#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#This notebook contains functions for preprocessing the tweeets for sentiment analysis
#The functions are applied in applying_text_process notebook


# In[ ]:


#importing utility libraries


# In[ ]:


import re
import numpy as np
import pandas as pd


# In[ ]:


#natural language processing toolkit


# In[ ]:


#importing word_tokenize


# In[ ]:


from nltk.tokenize import word_tokenize
import nltk
nltk.download('punkt')


# In[ ]:


#importing WordNetLemmatizer


# In[ ]:


from nltk.stem import WordNetLemmatizer
import nltk
nltk.download('wordnet')


# In[ ]:


#defining list of stop words in English


# In[ ]:


stopwordlist = ['a', 'about', 'above', 'after', 'again', 'ain', 'all', 'am', 'an',
             'and','any','are', 'as', 'at', 'be', 'because', 'been', 'before',
             'being', 'below', 'between','both', 'by', 'can', 'd', 'did', 'do',
             'does', 'doing', 'down', 'during', 'each','few', 'for', 'from',
             'further', 'had', 'has', 'have', 'having', 'he', 'her', 'here',
             'hers', 'herself', 'him', 'himself', 'his', 'how', 'i', 'if', 'in',
             'into','is', 'it', 'its', 'itself', 'just', 'll', 'm', 'ma',
             'me', 'more', 'most','my', 'myself', 'now', 'o', 'of', 'on', 'once',
             'only', 'or', 'other', 'our', 'ours','ourselves', 'out', 'own', 're','s', 'same', 'she', "shes", 'should', "shouldve",'so', 'some', 'such',
             't', 'than', 'that', "thatll", 'the', 'their', 'theirs', 'them',
             'themselves', 'then', 'there', 'these', 'they', 'this', 'those',
             'through', 'to', 'too','under', 'until', 'up', 've', 'very', 'was',
             'we', 'were', 'what', 'when', 'where','which','while', 'who', 'whom',
             'why', 'will', 'with', 'won', 'y', 'you', "youd","youll", "youre",
             "youve", 'your', 'yours', 'yourself', 'yourselves']


# In[ ]:


#changing stopwordlist from list to set


# In[ ]:


STOPWORDS=set(stopwordlist)


# In[ ]:


#cleaning stop words from the list function


# In[ ]:


def cleaning_stopwords(text):
    return" ".join([word for word in str(text).split() if word not in STOPWORDS])


# In[ ]:


#cleaning and removing punctuation, using string library


# In[ ]:


import string
english_punctuations=string.punctuation


# In[ ]:


def cleaning_punctuations(text):
    translator=str.maketrans('','',english_punctuations)
    return text.translate(translator)


# In[ ]:


#cleaning and removing repeating characters(more than 2)


# In[ ]:


def cleaning_repeating_char(text):
    return re.sub(r'(.)\1+', r'\1\1', text)


# In[ ]:


#cleaning URLs


# In[ ]:


def cleaning_URLs(data):
    return re.sub('((WWW.[^s]+)|(https?://[^s]+))',' ', data)


# In[ ]:


#cleaning and removing numbers


# In[ ]:


def cleaning_numbers(data):
    return re.sub('[0-9]+', '', data)


# In[ ]:


#defining stemming function


# In[ ]:


st=nltk.PorterStemmer()
def stemming_on_text(data):
    text=[st.stem(word) for word in data]
    return data


# In[ ]:


#defining lemmatizer function


# In[ ]:


lm=WordNetLemmatizer()
def lemmatizer_on_text(data):
    text=[lm.lemmatize(word) for word in data]
    return data


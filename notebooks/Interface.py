#!/usr/bin/env python
# coding: utf-8

# In[1]:


#This notebook is the interface for the sentiment around a list of companies based on tweets
#The sentiment prection could be used for different purposes, including short term investment in stocks.


# In[2]:


#List of companies of interest for sentiment prediction


# In[3]:


company_list = ['TSLA', 'AAPL', 'META']


# In[4]:


#Tweets are taken using twitter API with unpaid developer account
#Find the scraping code, in scraing.ipynb notebook


# In[5]:


#Twitter API does not allow getting multiple queries within a short period time for non-paid developer accounts
#Therefore, the scraping part is taken out from automation process.
#Automated scraping can be performed with comment below
#%run scraping.ipynb


# In[6]:


#Applying sentiment analysis model saved in 'apply_model_stock.ipynb'trained on previously labled tweets


# In[7]:


sentiment = []
for i in range(len(company_list)):
    FILE_NAME = company_list[i]+'.csv'
    get_ipython().run_line_magic('run', 'apply_model_stock.ipynb')
    sentiment.append("{:.2f}".format(current_sentiment))


# In[8]:


#showing the ratio of positive tweets to the total number of tweets for each company


# In[9]:


print(sentiment)


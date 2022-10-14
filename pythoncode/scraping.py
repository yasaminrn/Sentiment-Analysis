#!/usr/bin/env python
# coding: utf-8

# In[1]:


#This notebook is used for scraping tweet for companies of interest using twitter API


# In[2]:


#importing libraries


# In[3]:


import pandas as pd


# In[4]:


import tweepy
from tweepy import OAuthHandler


# In[5]:


import time


# In[6]:


get_ipython().run_line_magic('run', 'key.ipynb')


# In[7]:


def auth():
    try:
        auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
        auth.set_access_token(access_token, access_token_secret)
        api = tweepy.API(auth)
    except:
        print("An error occurred during the authentication")
    return api


# In[8]:


api = auth()


# In[9]:


print(api)


# In[10]:


#text_query = stock_list
#this command is provided by a list for general use 
#as the tweeter API does not allow multiple queries in a short time period, they are run one by one
text_query = ['TSLA', 'AAPL', 'META']


# In[11]:


len(text_query)


# In[12]:


for j in range(0,len(text_query)):

    df_query_based_tweets = pd.DataFrame()

    try:
        # Creation of query method using appropriate parameters
        count = 1000
        tweets = tweepy.Cursor(api.search_tweets,q=text_query[j], lang="en").items(count)

        # Pulling information from tweets iterable object and adding relevant tweet information in our data frame
        for tweet in tweets:
            df_query_based_tweets = df_query_based_tweets.append(
                          {'Created at' : tweet._json['created_at'],
                                       'User ID': tweet._json['id'],
                              'User Name': tweet.user._json['name'],
                                        'text': tweet._json['text'],
                     'Description': tweet.user._json['description'],
                           'Location': tweet.user._json['location'],
             'Followers Count': tweet.user._json['followers_count'],
                 'Friends Count': tweet.user._json['friends_count'],
               'Statuses Count': tweet.user._json['statuses_count'],
         'Profile Image Url': tweet.user._json['profile_image_url'],
                         }, ignore_index=True)
    except BaseException as e:
        print('failed on_status,',str(e))
        time.sleep(3)
    
    df_query_based_tweets.to_csv(text_query[j]+'.csv', index=False)


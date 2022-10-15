
#This notebook is used for reading data and applying the pre-process on tweet data
#The pre-process is the first step to format the data for proper use of Natural language toolkit

#importing utility libraries

import re
import numpy as np
import pandas as pd

#importing dataset

DATASET_ENCODING = "ISO-8859-1"

#File location and address

#File names are taken automatically from interface company_list
#For single use of applying_textprocess, can use the FILE_NAME as shown in exapmle below.
#FILE_NAME = 'META.csv'


# In[8]:


#Defining column titles
#For training/testing data, two columns of text and target are used
#For new data, the scraping data has column labels. For current project we are only interested in 'text' column of new data.

if re.search('train', FILE_NAME) or re.search('test', FILE_NAME):
    DATASET_COLUMNS=['text','target']

if re.search('train', FILE_NAME) or re.search('test', FILE_NAME):
    df = pd.read_csv(FILE_NAME, encoding=DATASET_ENCODING, names=DATASET_COLUMNS)
else:
    df = pd.read_csv(FILE_NAME, encoding=DATASET_ENCODING)

#checking for Null values, in whole table and in text column.

np.sum(df.isnull().any(axis=1))

np.sum(df['text'].isnull())

if re.search('train', FILE_NAME):
    data=df[['text','target']]
elif re.search('test', FILE_NAME):
    data=df[['text','target']]
else: 
    data=df[['text']]

get_ipython().run_line_magic('run', 'text_process.ipynb')

#changing all tweets to lower case

data['text']

#As we want to overwrite data frame disbaling the default warning for overwrite

pd.options.mode.chained_assignment = None  # default='warn'

data['text']=data['text'].str.lower()

#applying the cleaning_stopwords function defined in text_process notebook

data['text']=data['text'].apply(lambda text: cleaning_stopwords(text))

#applying the cleaning_punctuations function defined in text_process notebook

data['text']=data['text'].apply(lambda x:cleaning_punctuations(x))

#applying the cleaning_repeating_char function defined in text_process notebook

data['text']=data['text'].apply(lambda x: cleaning_repeating_char(x))

#Applying cleaning_URLs function defined in text_process notebook

data['text']=data['text'].apply(lambda x: cleaning_URLs(x))

#applying the cleaning_numbers function defined in text_process notebook

data['text']=data['text'].apply(lambda x: cleaning_numbers(x))

#applying the word_tokenize function

data['text']=data['text'].apply(word_tokenize)

#applying stemming_on_text function defined in text_process notebook

data['text']=data['text'].apply(lambda x: stemming_on_text(x))

#applying lemmatizer_on_text function defined in text_process notebook

data['text']=data['text'].apply(lambda x: lemmatizer_on_text(x))

data['text']


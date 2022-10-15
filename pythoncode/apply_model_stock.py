
#This notebook is used to apply the trained mode, defined in stock_sentiment_model notebook
#The trained logistic regression model and trained word tokenizer model are saved and loaded using pickle

#importing utilities

import re
import numpy as np
import pandas as pd

#plotting libraries, added for potential testing purpose, not used for prediction
import seaborn as sns
import matplotlib.pyplot as plt

#Transforms text to feature vectors that can be used as input to estimator

from sklearn.feature_extraction.text import TfidfVectorizer

#importing performance matrix for potential testing purpose, not used for prediction
from sklearn.metrics import confusion_matrix, classification_report


#importing pickle to load saved models (logistic regression and vectorizer)
import pickle


#FILE_NAMES are automatically taken from interface notebook.
#The command below for file name is included for potential single use of this notebook
#FILE_NAME = 'META.csv'

get_ipython().run_line_magic('run', 'applying_textprocess.ipynb')

X=data['text']

#Joining data as the vectorizer works on string

X=X.apply(lambda x: ' '.join(x))

#applying vectorizer and logistic regression models trained and saved in stock_sentimet_model notebook

vectorizer = pickle.load(open('stock_vectorizer.pk','rb'))

loaded_model = pickle.load(open('stock_LRmodel.sav', 'rb'))

x=data['text'].apply(lambda y: str(np.array(y)))

prediction = loaded_model.predict(vectorizer.transform(X))

#showing sample of prediction for first 100 tweets

prediction[:100]

np.unique(prediction)

#calculating the ratio of positive outputs to total number of tweets andlyzed
#This is used as means of gauging the current sentiment around the companies of interest

current_sentiment = sum(prediction)/len(prediction)


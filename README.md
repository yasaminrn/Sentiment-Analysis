# Sentiment-Analysis
This project is intended to perform sentiment analysis of tweets.
The results will give an understanding of current perspective around companies and is considered to be means for short-term investment decision making.

# Training data
The training data is composed of processed data from two datasets of labeled tweets.
The first dataset is tweets related to stocks, from Kaggle dataset below:
https://doi.org/10.34740/KAGGLE/DSV/1217821
Total count: 5,791
Negative count: 2,106
Positive count: 3,685
The dataset has labels of 1 for positive and -1 for negative, which is transformed to 1 for positive and 0 for negative in preprocessing.
The second dataset has tweets around apple stock (NASDAQ: AAPL), from dataword:
https://data.world/crowdflower/apple-twitter-sentiment
Total count: 3,886
Negative count: 1,219
Positive count: 423
Other count: 2,667
The dataset has labels of 1 for negative, 5 for positive, and 3 for neutral. 
Only the positive and negative tweets are used as training data. The labels are transformed to 1 for positive and 0 for negative.
The dataset has multiple columns, only the text and labels (called target) are used for training purpose.

# Pre-processing of text data
In order to use the data for natural language processing toolkit (NLTK), the necessity text processing is performed in two notebooks:
text_process which includes the definition of functions needed for text processing
applying_textprocess which includes application of the text_process commands. 
Thes text process commands are applied to both training data and the new data for sentiment analysis.

# Sentiment analysis model
Sentiment analysis model is performed in notebook stock_sentiment_model.
Different classification models are trained and tested. Among the models the logistic regression with 77% accuracy is saved for use on new data for sentiment analysis.
The word vectorizer is also saved to be implemented on new as the step needed prior to implementation of the model.

![alt text](https://github.com/yasaminrn/Sentiment-Analysis/upload/main/data)/LR_CF.jpg)

# Scraping tweets
Tweets around different companies are scraped using tweepy library and tweeter API for developers.
The saved tweet data have different columns extracted for potential future use, but only the text column is used in current sentiment analysis.

# Applying the model to new tweet texts
The model is applied to new data in notebook apply_model_stock and interface notebooks.

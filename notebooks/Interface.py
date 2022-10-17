#This notebook is the interface for the sentiment around a list of companies based on tweets
#The sentiment prection could be used for different purposes, including short term investment in stocks.

#List of companies of interest for sentiment prediction
company_list = ['TSLA', 'AAPL', 'META']

#Tweets are taken using twitter API with unpaid developer account
#Find the scraping code, in scraing.ipynb notebook

#Twitter API does not allow getting multiple queries within a short period time for non-paid developer accounts
#Therefore, the scraping part is taken out from automation process.
#Automated scraping can be performed with comment below
#%run scraping.ipynb

#Applying sentiment analysis model saved in 'apply_model_stock.ipynb'trained on previously labled tweets
sentiment = []
for i in range(len(company_list)):
    FILE_NAME = company_list[i]+'.csv'
    get_ipython().run_line_magic('run', 'apply_model_stock.ipynb')
    sentiment.append("{:.2f}".format(current_sentiment))

#showing the ratio of positive tweets to the total number of tweets for each company
print(sentiment)


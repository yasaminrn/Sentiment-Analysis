# This notebook is used for developing and training model for sentiment analysis
# The classification model is performed using bernoli model, SVC model and logistic regression model
# Logistic regression model and word vectorizer are saved using pickle to be loaded for prediction around new data

#importing utilities
import re
import numpy as np
import pandas as pd

#importing plotting libraries
import seaborn as sns
import matplotlib.pyplot as plt

#sklearn: importing classifiers
from sklearn.svm import LinearSVC 
from sklearn.naive_bayes import BernoulliNB
from sklearn.linear_model import LogisticRegression

#sklearn: modelling 
from sklearn.model_selection import train_test_split

#Transforms text to feature vectors that can be used as input to estimator
from sklearn.feature_extraction.text import TfidfVectorizer

#Importing confusion matrics to show the graohical representation of model performances
from sklearn import metrics
from sklearn.metrics import confusion_matrix, classification_report

#importing roc and auc tools
from sklearn.metrics import roc_curve, auc

#saving and reloading models
import pickle


FILE_NAME = 'stockdata_training.csv'

#changing data type to lists
#text, target = list(data['text']), list(data['target'])
get_ipython().run_line_magic('run', 'applying_textprocess.ipynb')
#selecting text and target column for analysis
dataset=data[['text','target']]

#separating input feature and label
#X=dataset.text
#y=dataset.target
X=data['text']
y=data['target']

X=X.apply(lambda x: ' '.join(x))

#splitting to train and test subsets (90% vs 10%)
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.05, random_state =1000)

#fit TF-IDF Vectorizer
vectorizer = TfidfVectorizer(ngram_range=(1,2), max_features=15000)
vectorizer.fit(X_train)

print('No. of feature_words: ', len(vectorizer.get_feature_names()))

#Saving the trained vectorizer to be used in apply_model_stock notebook
with open('stock_vectorizer.pk', 'wb') as fin:
    pickle.dump(vectorizer, fin)

X_train = vectorizer.transform(X_train)
X_test = vectorizer.transform(X_test)

#function for model evaluation
#accuracy score, confusion matrix with plot. ROC_AUC curve

def model_Evaluate(model):
    # Predict values for Test dataset
    y_pred = model.predict(X_test)
    # Print the evaluation metrics for the dataset.
    print(classification_report(y_test, y_pred))
    # Compute and plot the Confusion matrix
    cf_matrix = confusion_matrix(y_test, y_pred)
    categories = ['Negative','Positive']
    group_names = ['True Neg','False Pos', 'False Neg','True Pos']
    group_percentages = ['{0:.2%}'.format(value) for value in cf_matrix.flatten() / np.sum(cf_matrix)]
    labels = [f'{v1}n{v2}' for v1, v2 in zip(group_names,group_percentages)]
    labels = np.asarray(labels).reshape(2,2)
    sns.heatmap(cf_matrix, annot = labels, cmap = 'Blues',fmt = '',
    xticklabels = categories, yticklabels = categories)
    plt.xlabel("Predicted values", fontdict = {'size':14}, labelpad = 10)
    plt.ylabel("Actual values" , fontdict = {'size':14}, labelpad = 10)
    plt.title ("Confusion Matrix", fontdict = {'size':18}, pad = 20)

# model building

#bernoli model
#instantiate the model
BNBmodel = BernoulliNB()
BNBmodel.fit(X_train, y_train)
model_Evaluate(BNBmodel)
y_pred1 = BNBmodel.predict_proba(X_test)[::,1]

#roc curve for bernoli model
fpr, tpr, _ = metrics.roc_curve(y_test, y_pred1)
roc_auc = auc(fpr, tpr)
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=1, label='ROC curve (area = %0.2f)' % roc_auc)
#create ROC curve
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC CURVE')
plt.legend(loc="lower right")
plt.show()

#SVC model
#instantiate the model
SVCmodel = LinearSVC()
SVCmodel.fit(X_train, y_train)
model_Evaluate(SVCmodel)
y_pred2 = SVCmodel.predict(X_test)

#roc curve of SVC model
fpr, tpr, _ = metrics.roc_curve(y_test, y_pred2)
roc_auc = auc(fpr, tpr)
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=1, label='ROC curve (area = %0.2f)' % roc_auc)
#create ROC curve
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC CURVE')
plt.legend(loc="lower right")
plt.show()

#logistic regression
#instantiate the model
LRmodel = LogisticRegression(C = 2, max_iter = 1000, n_jobs=-1)
#fit the model using the training data
LRmodel.fit(X_train,y_train)
model_Evaluate(LRmodel)
y_pred3 = LRmodel.predict_proba(X_test)[::,1]

#roc curve for Logistic regression model
fpr, tpr, _ = metrics.roc_curve(y_test, y_pred3)
roc_auc = auc(fpr, tpr)
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=1, label='ROC curve (area = %0.2f)' % roc_auc)
#create ROC curve
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC CURVE')
plt.legend(loc="lower right")
plt.show()


# As the logistic regression model is performing better and it is well suited for classification problem, we save this to use on Amazon tweets

pickle.dump(LRmodel, open('stock_LRmodel.sav', 'wb'))


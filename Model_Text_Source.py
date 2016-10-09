# -*- coding: utf-8 -*-
"""
Created on Wed May 04 19:22:48 2016

@author: mohit
"""

# Importing libraries

import nltk
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline

# Read the dataframe
df = pd.read_csv('C:\Users\mohit\Desktop\Spring BIA\Web Analytics\Politifact\politifact.csv')

# Change coloumn names
df.head()
del df['Unnamed: 0']
df.columns = ['source','text','label']

# Remove No Flip, Full Flop, Half Flip
df = df[df['label']!= 'No Flip']
df = df[df['label']!= 'Full Flop'] 
df = df[df['label']!= 'Half Flip']

# Discretize the label into binary outputs - True and False
df.loc[df['label'] == 'Half-True', 'label'] = 'True'
df.loc[df['label'] == 'Mostly True', 'label'] = 'True'
df.loc[df['label'] == 'Mostly False', 'label'] = 'False'
df.loc[df['label'] == 'Pants on Fire!', 'label'] = 'False'

################################# Adding The Source as the feature ##############################

l = []
for x in df['source']:
    y = x.strip().replace(' ','')
    l.append(y)
    
l2 = []
for a in l:
    a = 'u' + a + ' '
    l2.append(a)

df['source_feature'] = l2
del df['source']
df['text'] = df['source_feature'] + df['text'] 
df.head()
################################################ Data Preprartion #######################################
# Build classification model without including the source as the feature
# Clean the data

import string
from nltk.corpus import stopwords

def text_process(mess):
    """
    Takes in a string of text, then performs the following:
    1. Remove all punctuation
    2. Remove all stopwords
    3. Returns a list of the cleaned text
    """
    # Check characters to see if they are in punctuation
    nopunc = [char for char in mess if char not in string.punctuation]

    # Join the characters again to form the string.
    nopunc = ''.join(nopunc)
    
    # Now just remove any stopwords
    return [word for word in nopunc.split() if word.lower() not in stopwords.words('english')]

# Apply it to all the text
df['text'].apply(text_process)

# Convert text to vectors
# Bag of words approach
from sklearn.feature_extraction.text import CountVectorizer
bow_transformer = CountVectorizer(analyzer=text_process).fit(df['text'])

text_bow = bow_transformer.transform(df['text'])
print 'Shape of Sparse Matrix: ', text_bow.shape
print 'Amount of Non-Zero occurences: ', text_bow.nnz
print 'sparsity: %.2f%%' % (100.0 * text_bow.nnz / (text_bow.shape[0] * text_bow.shape[1]))

# Perfrom tf-idf normalization
from sklearn.feature_extraction.text import TfidfTransformer
tfidf_transformer = TfidfTransformer().fit(text_bow)

messages_tfidf = tfidf_transformer.transform(text_bow)
print messages_tfidf.shape

# Split into training and test
from sklearn.cross_validation import train_test_split

text_train, text_test, label_train, label_test = \
train_test_split(df['text'], df['label'], test_size=0.3)

print len(text_train), len(text_test), len(text_train) + len(text_test)

############################################# Model Building ##########################################
# Create a data pipeline
from sklearn.pipeline import Pipeline
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import roc_curve, auc
from sklearn import metrics
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn import tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier 
from sklearn.ensemble import GradientBoostingClassifier

pipeline1 = Pipeline([
    ('bow', CountVectorizer(analyzer=text_process)),  # strings to token integer counts
    ('tfidf', TfidfTransformer()),  # integer counts to weighted TF-IDF scores
    ('NVB', MultinomialNB()) # train on TF-IDF vectors w/ Naive Bayes classifier
])

pipeline2 = Pipeline([
    ('bow', CountVectorizer(analyzer=text_process)),  # strings to token integer counts
    ('tfidf', TfidfTransformer()),  # integer counts to weighted TF-IDF scores
    ('logistic', LogisticRegression()) # train on TF-IDF vectors w/ Logistic Regression
])

pipeline3 = Pipeline([
    ('bow', CountVectorizer(analyzer=text_process)),  # strings to token integer counts
    ('tfidf', TfidfTransformer()),  # integer counts to weighted TF-IDF scores
    ('svm', svm.SVC(probability=True)) # train on TF-IDF vectors w/ SVM
])

pipeline4 = Pipeline([
    ('bow', CountVectorizer(analyzer=text_process)),  # strings to token integer counts
    ('tfidf', TfidfTransformer()),  # integer counts to weighted TF-IDF scores
    ('CART', tree.DecisionTreeClassifier()) # train on TF-IDF vectors w/ CART
])

pipeline5 = Pipeline([
    ('bow', CountVectorizer(analyzer=text_process)),  # strings to token integer counts
    ('tfidf', TfidfTransformer()),  # integer counts to weighted TF-IDF scores
    ('KNN', KNeighborsClassifier()) # train on TF-IDF vectors w/ KNN
])

pipeline6 = Pipeline([
    ('bow', CountVectorizer(analyzer=text_process)),  # strings to token integer counts
    ('tfidf', TfidfTransformer()),  # integer counts to weighted TF-IDF scores
    ('RF',RandomForestClassifier()) # train on TF-IDF vectors w/ KNN
])

#Naive bayes
pipeline1.fit(text_train,label_train)
predictions1 = pipeline1.predict(text_test)

# Confusion Matrix
pd.crosstab(label_test,predictions1, rownames=['True'], colnames=['Predicted'], margins=True)
print metrics.accuracy_score(label_test, predictions1)
probs_NB = pipeline1.predict_proba(text_test)
print probs_NB

# Convert true false to 1,0 for ROC curve
label_test_roc = label_test.copy()
label_test_roc

label_test_roc[label_test_roc=='True'] = 1
label_test_roc[label_test_roc=='False'] = 0

fpr_NB, tpr_NB, _ = roc_curve(label_test_roc, probs_NB[:,1])
roc_auc_NB = auc(fpr_NB, tpr_NB)
print 'ROC AUC of NB:' ,roc_auc_NB

# Logistic Regression
pipeline2.fit(text_train,label_train)
predictions2 = pipeline2.predict(text_test)

# Confusion Matrix
pd.crosstab(label_test,predictions2, rownames=['True'], colnames=['Predicted'], margins=True)
print metrics.accuracy_score(label_test, predictions2)

# generate class probabilities
probs_LG = pipeline2.predict_proba(text_test)
print probs_LG

fpr_LG, tpr_LG, _ = roc_curve(label_test_roc, probs_LG[:,1])
roc_auc_LG = auc(fpr_LG, tpr_LG)
print 'ROC AUC of LG:' ,roc_auc_LG

# Support Vector Machines
pipeline3.fit(text_train,label_train)
predictions3 = pipeline3.predict(text_test)

# Confusion Matrix
pd.crosstab(label_test,predictions3, rownames=['True'], colnames=['Predicted'], margins=True)
print metrics.accuracy_score(label_test, predictions3)

# generate class probabilities
probs_SVM = pipeline3.predict_proba(text_test)
print probs_SVM

fpr_SVM, tpr_SVM, _ = roc_curve(label_test_roc, probs_SVM[:,1])
roc_auc_SVM = auc(fpr_SVM, tpr_SVM)
print 'ROC AUC of SVM:' ,roc_auc_SVM

# CART
pipeline4.fit(text_train,label_train)
predictions4 = pipeline4.predict(text_test)

# Confusion Matrix
pd.crosstab(label_test,predictions4, rownames=['True'], colnames=['Predicted'], margins=True)

# generate class probabilities
probs_DT = pipeline4.predict_proba(text_test)
print probs_DT

print metrics.accuracy_score(label_test, predictions4)

fpr_DT, tpr_DT, _ = roc_curve(label_test_roc, probs_DT[:,1])
roc_auc_DT = auc(fpr_DT, tpr_DT)
print 'ROC AUC of DT:' ,roc_auc_DT

# K - nearest neighbours
n_neighbors=3
pipeline5.fit(text_train,label_train)
predictions5 = pipeline5.predict(text_test)

# Confusion Matrix
pd.crosstab(label_test,predictions5, rownames=['True'], colnames=['Predicted'], margins=True)

print metrics.accuracy_score(label_test, predictions5)

# generate class probabilities
probs_KNN = pipeline5.predict_proba(text_test)
print probs_KNN

fpr_KNN, tpr_KNN, _ = roc_curve(label_test_roc, probs_KNN[:,1])
roc_auc_KNN = auc(fpr_KNN, tpr_KNN)
print 'ROC AUC of KNN:' ,roc_auc_KNN

# Random Forest
n_estimators = 200
pipeline6.fit(text_train,label_train)
predictions6 = pipeline6.predict(text_test)

# Confusion Matrix
pd.crosstab(label_test,predictions6, rownames=['True'], colnames=['Predicted'], margins=True)
print metrics.accuracy_score(label_test, predictions6)

# generate class probabilities
probs_RF = pipeline6.predict_proba(text_test)
print probs_RF

fpr_RF, tpr_RF, _ = roc_curve(label_test_roc, probs_RF[:,1])
roc_auc_RF = auc(fpr_RF, tpr_RF)
print 'ROC AUC of RF:' ,roc_auc_RF

################################################## ROC Curve ########################################

plt.figure()
plt.plot(fpr_LG, tpr_LG, label='LG(area = %f)' % roc_auc_LG)
plt.plot(fpr_SVM, tpr_SVM, label='SVM(area = %f)' % roc_auc_SVM)
plt.plot(fpr_NB, tpr_NB, label='NB(area = %f)' % roc_auc_NB)
plt.plot(fpr_DT, tpr_DT, label='DT(area = %f)' % roc_auc_DT)
plt.plot(fpr_KNN, tpr_KNN, label='KNN(area = %f)' % roc_auc_KNN)
plt.plot(fpr_RF, tpr_RF, label='RF(area = %f)' % roc_auc_RF)
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc="lower right")
plt.show()













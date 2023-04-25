#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import warnings
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import f1_score, roc_auc_score, roc_curve
from pylab import rcParams
from sklearn.svm import SVC
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.preprocessing import LabelEncoder
from collections import defaultdict
from nltk.corpus import wordnet as wn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import model_selection, naive_bayes, svm
from sklearn.metrics import accuracy_score
import nltk


# In[2]:


# Read CSV file into input_df
input_df = pd.read_csv('smaller_staff_cleaned_dataset.csv', sep=',')

print(input_df)


# In[3]:


input_df.dtypes


# In[4]:


# Break input_df into features and label
features = input_df.drop(['Review_Type','isPositive'], axis=1)
label = input_df['isPositive']
features


# In[5]:


label


# In[6]:


# Split input_df into test and train dataset
Staff_train, Staff_test, staff_train_label, staff_test_label = train_test_split(features, label, test_size=0.3, random_state=42)


# In[7]:


Staff_train


# In[8]:


Staff_test


# In[9]:


staff_train_label


# In[10]:


class_polarity=pd.crosstab(index=input_df["Review_Type"], columns="count")
class_polarity


# In[11]:


proportion_polarity=input_df["Review_Type"].value_counts(normalize=True)
proportion_polarity


# In[12]:


# setting up parameter of the NB algorithm
Naive = naive_bayes.MultinomialNB()


# In[13]:


import time
start = time.time()


# In[14]:


# fit the training dataset on the NB classifier
Naive.fit(Staff_train, staff_train_label)


# In[15]:


end = time.time()
print('Time taken to run NB : ', end - start, "seconds")


# In[16]:


# Predict using the model generated
prediction_NB = pd.DataFrame()
prediction_NB['Actual']     = staff_test_label
prediction_NB['Predicted']  = Naive.predict(Staff_test)
display (prediction_NB)


# In[17]:


# predict the labels on validation dataset
predictions_NB = Naive.predict(Staff_test)


# In[18]:


# Use accuracy_score function to get the accuracy
print("Naive Bayes Accuracy Score -> ",accuracy_score(predictions_NB, staff_test_label)*100)


# In[19]:


#Naive Bayes confusion matrix
confusion_matrix_NB = pd.crosstab(prediction_NB['Actual'], prediction_NB['Predicted'], rownames=['Actual'], colnames=['Predicted'])
print (confusion_matrix_NB)


# In[20]:


import time
start = time.time()


# In[21]:


# Intial and fit LogisticRegression model
lr = LogisticRegression(max_iter=1000)
lr.fit(Staff_train, staff_train_label) # Training step


# In[22]:


end = time.time()
print('Time taken to run lr : ', end - start, "seconds")


# In[23]:


# Predict using the model generated
prediction_lr = pd.DataFrame()
prediction_lr['Actual3']     = staff_test_label
prediction_lr['Predicted3']  = lr.predict(Staff_test)
display(prediction_lr)
#display (prediction_LR)
print(prediction_lr.dtypes)


# In[24]:


# Display confusion matrix
confusion_matrix_LR = pd.crosstab(prediction_lr['Actual3'], prediction_lr['Predicted3'])
print (confusion_matrix_LR)


# In[25]:


# predict the labels on validation dataset
prediction_lr = lr.predict(Staff_test)


# In[26]:


# Use accuracy_score function to get the accuracy
print("Logistic regression Accuracy Score -> ",accuracy_score(prediction_lr, staff_test_label)*100)


# In[27]:


# Classifier - Algorithm - SVM -setting up algorithm
SVM = svm.SVC(C=1.0, kernel='linear', degree=3, gamma='auto')


# In[28]:


import time
start = time.time()


# In[29]:


# fit the training dataset on the SVM classifier
SVM.fit(Staff_train,staff_train_label)


# In[30]:


end = time.time()
print('Time taken to run SVM : ', end - start, "seconds")


# In[31]:


# Predict using the model generated
prediction_SVM = pd.DataFrame()
prediction_SVM['Actual2']     = staff_test_label
prediction_SVM['Predicted2']  = SVM.predict(Staff_test)
display (prediction_SVM)


# In[32]:


# predict the labels on validation dataset
predictions_SVM = SVM.predict(Staff_test)


# In[33]:


# Use accuracy_score function to get the accuracy
print("SVM Accuracy Score -> ",accuracy_score(predictions_SVM, staff_test_label)*100)


# In[34]:


# Display confusion matrix
confusion_matrix_SVM = pd.crosstab(prediction_SVM['Actual2'], prediction_SVM['Predicted2'], rownames=['Actual2'], colnames=['Predicted2'])
print (confusion_matrix_SVM)


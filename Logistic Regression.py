# -*- coding: utf-8 -*-
"""
Created on Tue Feb 21 16:59:23 2023

@author: lenovo
"""

BUSINESS OBJECTIVE:-(y)Whether the client has subscribed a term deposit or not 

0=NOT SUBSRIBED
1=SUBSRIBED



#Importing the Necessary Liabrary
import pandas as pd
import numpy as np 
import seaborn as sns
import matplotlib.pylab as plt
import scipy
from scipy import stats
import pylab


#Loading The Dataset
df=pd.read_csv('C:/Users/lenovo/OneDrive/Documents/EXCLER ASSIGNMENTS/Logistic Regression/bank.csv')
#EDA
df.info()
df.isna().sum()
df.describe()
#First we need to convert non-numeric data to numeric data,using Label Encoding
from sklearn.preprocessing import LabelEncoder
l=LabelEncoder()
df['job']=l.fit_transform(df['job'])
df['education']=l.fit_transform(df['education'])
df['default']=l.fit_transform(df['default'])
df['housing']=l.fit_transform(df['housing'])
df['contact']=l.fit_transform(df['contact'])
df['month']=l.fit_transform(df['month'])
df['poutcome']=l.fit_transform(df['poutcome'])
df['y']=l.fit_transform(df['y'])
df['marital']=l.fit_transform(df['marital'])
df['loan']=l.fit_transform(df['loan'])

df.info()
df.shape
df.describe()

#Model Building
import statsmodels.formula.api as sm
model=sm.logit('y ~ age+job+marital+education+default+balance+housing+loan+contact+day+month+duration+campaign+pdays+previous+poutcome',data=df).fit()
model.summary()
model.summary2()#for AIC

pred=model.predict(df)

from sklearn import metrics
from sklearn.metrics import roc_curve,auc,classification_report,confusion_matrix

#Finding the Cut-off Value
fpr,tpr,thresholds=roc_curve(df.y,pred)
index=np.argmax(tpr - fpr)
cutoffvalue=thresholds[index]
cutoffvalue

#Area Under the Curve
auc=auc(fpr,tpr)
auc
print("Area under the ROC curve : %f" % auc)

#Visual Representation(ROC Curve)

plt.plot(fpr,tpr)
plt.xlabel('false positive rate')
plt.ylabel('true positive rate')
plt.title('Receiver Operating Characterstics')
plt.show()

# filling all the cells with zeroes
df['pred']=np.zeros(4521)

#taking threshold value and above the prob value will be treated as correct value 
df.loc[pred > cutoffvalue,'pred']=1

#Classification Report
classification_report=classification_report(df.y,df.pred)
classification_report

#Splitting the data into train and test data 
from sklearn.model_selection import train_test_split
train,test=train_test_split(df,test_size=0.3)

#Model Building

finalmodel=sm.logit('y ~ age+job+marital+education+default+balance+housing+loan+contact+day+month+duration+campaign+pdays+previous+poutcome',data=train).fit()
finalmodel.summary()
finalmodel.summary2()

#test data
test_predict=finalmodel.predict(test)

#Filling all the cells with Zeros
test['test_predict']=np.zeros(1357)


#ROC CURVE,Cutoff value
fpr, tpr, thresholds = metrics.roc_curve(test["y"], test_predict)
index=np.argmax(tpr - fpr)
cutoffvalue=thresholds[index]
cutoffvalue

# taking threshold value as 'optimal_threshold' and above the thresold prob value will be treated as 1 

test.loc[test_predict > cutoffvalue,'test_predict']=1

#Classification_report

classification_test = classification_report(test["test_predict"], test["y"])
classification_test
#Confusion Matrix
confusion_matrix=confusion_matrix(test.y,test['test_predict'])

#
#PLOT OF ROC
plt.plot(fpr, tpr);plt.xlabel("False positive rate");plt.ylabel("True positive rate")
#Area under the curve
roc_auc_test = metrics.auc(fpr, tpr)
roc_auc_test


#Train Data

train_predict = finalmodel.predict(train)
                                
# Creating new column 
# filling all the cells with zeroes
train["train_predict"] = np.zeros(938)


#ROC CURVE,Cutoff value
fpr, tpr, thresholds = metrics.roc_curve(train["y"], train_predict)
index=np.argmax(tpr - fpr)
cutoffvalue=thresholds[index]
cutoffvalue



# taking threshold value and above the prob value will be treated as correct value 
train.loc[train_predict > optimal_threshold,"train_predict"] = 1

# confusion matrix
confusion_matrx = pd.crosstab(train['train_predict'],train['y'])
confusion_matrx

#Classification_report

classification_test = classification_report(train["train_predict"],test["y"])
classification_test

#ROC CURVE AND AUC
fpr, tpr, threshold = metrics.roc_curve(train["y"], train_predict)

#PLOT OF ROC
plt.plot(fpr, tpr);plt.xlabel("False positive rate");plt.ylabel("True positive rate")
#Area under  the curve
roc_auc_train = metrics.auc(fpr, tpr)
roc_auc_train

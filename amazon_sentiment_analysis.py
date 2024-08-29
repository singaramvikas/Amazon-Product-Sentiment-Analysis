# -*- coding: utf-8 -*-
"""Amazon Sentiment Analysis.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1przi0n0XQzKfq5SwI0myM6TwYJyPotOg
"""

import pandas as pd
import numpy as np

df = pd.read_csv(r'/content/drive/MyDrive/Amazon Use Case/Reviews.csv')
df.head()

df.columns

df['Helpful%']=np.where(df['HelpfulnessDenominator']>0, df['HelpfulnessNumerator']/df['HelpfulnessDenominator'],-1)

df['Helpful%'].unique()

df.info()

df['%upvote']=pd.cut(df['Helpful%'],bins=[-1,0,0.2,0.4,0.6,0.8,1],labels=['Empty', '0-20','20-40','40-60','60-80','80-100'])

df.head()

df.groupby(['Score','%upvote']).agg('count')

df_s=df.groupby(['Score','%upvote']).agg({'Id':'count'}).reset_index()

#Creating a Pivot Table
pivot=df_s.pivot(index='%upvote',columns='Score')

#creating a heat map for the pivot table

import seaborn as sns

sns.heatmap(pivot,annot=True,cmap='YlGnBu')

df['Score'].unique()

df2=df[df['Score']!=3]

X=df2['Text']

df2['Score'].unique()

y_dict ={1:0,2:0,4:1,5:1}
y=df2['Score'].map(y_dict)

from sklearn.feature_extraction.text import CountVectorizer

#stop_words='english': This parameter specifies that common English words (like 'the', 'is', 'and', etc.) should be ignored during the tokenization process.
c=CountVectorizer(stop_words='english')

X_c=c.fit_transform(X)

X_c.shape

#building model to check for accuracy
from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test = train_test_split(X_c,y)

X_train.shape

from sklearn.linear_model import LogisticRegression

log = LogisticRegression()

ml=log.fit(X_train,y_train)

ml.score(X_test,y_test)

# Extracting top 20 positive words and top 20 negative words
w=c.get_feature_names_out()

coef=ml.coef_.tolist()[0]

coef_df=pd.DataFrame({'Word':w,'Coefficient':coef})
coef_df

coef_df=coef_df.sort_values(['Coefficient','Word'],ascending=False)

coef_df.head(20)

coef_df.tail(20)

"""### Automate Bag of word on the data, check model accuracy, fetch top 20 positive and negative words

"""

def text_fit(X,y,nlp_model,ml_model,coeff=1):
  X_c=nlp_model.fit_transform(X)
  print('features:{}'.format(X_c.shape[1]))
  X_train,X_test,y_train,y_test = train_test_split(X_c,y)
  ml=ml_model.fit(X_train,y_train)
  acc=ml.score(X_test,y_test)
  print(acc)
  print('\n')
  if coeff==1:

    w=c.get_feature_names_out()
    coef=ml.coef_.tolist()[0]
    coef_df=pd.DataFrame({'Word':w,'Coefficient':coef})
    coef_df=coef_df.sort_values(['Coefficient','Word'],ascending=False)
    print('\n')
    print('top 20 positive words')
    print(coef_df.head(20))
    print('\n')
    print('top 20 negative words')
    print(coef_df.tail(20))

from sklearn.feature_extraction.text import CountVectorizer

c=CountVectorizer(stop_words='english')

from sklearn.linear_model import LogisticRegression

text_fit(X,y,c,LogisticRegression())

# Automating Prediction using a function
from sklearn.metrics import confusion_matrix,accuracy_score
def predict(X,y,nlp_model,ml_model):
  X_c=nlp_model.fit_transform(X)
  X_train,X_test,y_train,y_test = train_test_split(X_c,y)
  ml=ml_model.fit(X_train,y_train)
  prediction = ml.predict(X_test)
  cm=confusion_matrix(prediction,y_test)
  print(cm)
  ac=accuracy_score(prediction,y_test)
  print(ac)

c= CountVectorizer()
lr = LogisticRegression()

predict(X,y,c,lr)

from sklearn.dummy import DummyClassifier

text_fit(X,y,c,DummyClassifier(),0)

from sklearn.feature_extraction.text import TfidfVectorizer

tfidf = TfidfVectorizer(stop_words='english')

text_fit(X,y,tfidf,lr,0)

predict(X,y,c,lr)

data=df[df['Score']==5]
data.head()

data['%upvote'].unique()

data2=data[data['%upvote'].isin(['80-100','60-80','20-40','0-20'])]

data2.head()

X=data2['Text']

data2['%upvote'].unique()

y_dict = {'80-100':1, '60-80':1, '20-40':0, '0-20':0}
y = data2['%upvote'].map(y_dict)

y.value_counts()

from sklearn.feature_extraction.text import TfidfVectorizer

tf = TfidfVectorizer()

X_c = tf.fit_transform(X)

y.value_counts()

# Apply Techniques for handling imbalanced data

from imblearn.over_sampling import RandomOverSampler

os = RandomOverSampler()

X_train_res,y_train_res=os.fit_resample(X_c,y)

X_train_res.shape

y_train_res.shape

from collections import Counter

print('Original data shape {}'.format(Counter(y)))
print('Resample data shape {}'.format(Counter(y_train_res)))

#Doing Cross Validation using grid search CV

from sklearn.linear_model import LogisticRegression

log_class = LogisticRegression()

from sklearn.model_selection import GridSearchCV

np.arange(-2,3)

grid={'C':10.0**np.arange(-2,3),'penalty':['l1','l2']}

clf=GridSearchCV(estimator=log_class,param_grid=grid,cv=5,n_jobs=-1,scoring='f1_macro')

clf.fit(X_train_res,y_train_res)

from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test= train_test_split(X_c,y)

pred = clf.predict(X_test)

from sklearn.metrics import confusion_matrix

confusion_matrix(y_test,pred)

accuracy_score(y_test,pred)


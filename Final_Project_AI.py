# -*- coding: utf-8 -*-
"""
Created on Mon Oct 26 09:29:19 2020

@author: Jagesh
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords 
from nltk.tokenize import WordPunctTokenizer

import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv('mauritius_restaurant.csv')

df = data[['review_id', 'user_id', 'business_id', 'text', 'stars', 'date']]

df_business = pd.read_csv('mauritius_business.csv')

#Check Null values in Dataframe
df.isnull().sum()

df.head()


df.shape

#Select only stars and text
column_data = df[['business_id', 'user_id', 'stars', 'text']]


import string
from nltk.corpus import stopwords
stop = []
for word in stopwords.words('english'):
    s = [char for char in word if char not in string.punctuation]
    stop.append(''.join(s))
    
    def text_process(mess):
        """Takes in a string of text, then performs the following:
        1. Remove all punctuation
        2. Remove all stopwords
        3. Returns a list of the cleaned text"""
        # Check characters to see if they are in punctuation
        nopunc = [char for char in mess if char not in string.punctuation]
    
        # Join the characters again to form the string.
        nopunc = ''.join(nopunc)
        
        # Now just remove any stopwords
        return " ".join([word for word in nopunc.split() if word.lower() not in stop])

column_data['text'] = column_data['text'].apply(text_process)

#Split train test for testing the model later
vld_size=0.15
X_train, X_valid, y_train, y_valid = train_test_split(column_data['text'], df['business_id'], test_size = vld_size)


userid_df = column_data[['user_id','text']]
business_df = column_data[['business_id', 'text']]

userid_df.head()

userid_df[userid_df['user_id']=='ZwVz20be-hOZnyAbevyMyQ']['text']

business_df.head()


userid_df = userid_df.groupby('user_id').agg({'text': ' '.join})
business_df = business_df.groupby('business_id').agg({'text': ' '.join})

userid_df.head()

userid_df.loc['ZwVz20be-hOZnyAbevyMyQ']['text']

from sklearn.feature_extraction.text import TfidfVectorizer

#userid vectorizer
userid_vectorizer = TfidfVectorizer(tokenizer = WordPunctTokenizer().tokenize, max_features=5000)
userid_vectors = userid_vectorizer.fit_transform(userid_df['text'])
userid_vectors.shape

userid_vectors


#Business id vectorizer
businessid_vectorizer = TfidfVectorizer(tokenizer = WordPunctTokenizer().tokenize, max_features=5000)
businessid_vectors = businessid_vectorizer.fit_transform(business_df['text'])
businessid_vectors.shape

#Matrix factorization
userid_rating_matrix = pd.pivot_table(column_data, values='stars', index=['user_id'], columns=['business_id'])
userid_rating_matrix.shape

userid_rating_matrix.head()

P = pd.DataFrame(userid_vectors.toarray(), index=userid_df.index, columns=userid_vectorizer.get_feature_names())
Q = pd.DataFrame(businessid_vectors.toarray(), index=business_df.index, columns=businessid_vectorizer.get_feature_names())

Q.head()

def matrix_factorization(R, P, Q, steps=25, gamma=0.001,lamda=0.02):
    for step in range(steps):
        for i in R.index:
            for j in R.columns:
                if R.loc[i,j]>0:
                    eij=R.loc[i,j]-np.dot(P.loc[i],Q.loc[j])
                    P.loc[i]=P.loc[i]+gamma*(eij*Q.loc[j]-lamda*P.loc[i])
                    Q.loc[j]=Q.loc[j]+gamma*(eij*P.loc[i]-lamda*Q.loc[j])
        e=0
        for i in R.index:
            for j in R.columns:
                if R.loc[i,j]>0:
                    e= e + pow(R.loc[i,j]-np.dot(P.loc[i],Q.loc[j]),2)+lamda*(pow(np.linalg.norm(P.loc[i]),2)+pow(np.linalg.norm(Q.loc[j]),2))
        if e<0.001:
            break
        
    return P,Q

%%time
P, Q = matrix_factorization(userid_rating_matrix, P, Q, steps=25, gamma=0.001,lamda=0.02)

Q.head()

Q.iloc[0].sort_values(ascending=False).head(10)

import pickle
output = open('mauritius_recommendation_model.pkl', 'wb')
pickle.dump(P,output)
pickle.dump(Q,output)
pickle.dump(userid_vectorizer,output)
output.close()

words = "i want to have dinner with beautiful views"
test_df= pd.DataFrame([words], columns=['text'])
test_df['text'] = test_df['text'].apply(text_process)
test_vectors = userid_vectorizer.transform(test_df['text'])
test_v_df = pd.DataFrame(test_vectors.toarray(), index=test_df.index, columns=userid_vectorizer.get_feature_names())

predictItemRating=pd.DataFrame(np.dot(test_v_df.loc[0],Q.T),index=Q.index,columns=['Rating'])
topRecommendations=pd.DataFrame.sort_values(predictItemRating,['Rating'],ascending=[0])[:7]

for i in topRecommendations.index:
    print(df_business[df_business['business_id']==i]['name'].iloc[0])
    print(df_business[df_business['business_id']==i]['categories'].iloc[0])
    print(str(df_business[df_business['business_id']==i]['stars'].iloc[0])+ ' '+str(df_business[df_business['business_id']==i]['review_count'].iloc[0]))
    print('')
    
    
"""
Created on Fri Sep 24 07:59:06 2021

@author: koreynishimoto

Some words

"""

import pandas as pd

import numpy as np
import scipy 
import joblib
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from sklearn import datasets
from sklearn import metrics
from sklearn.linear_model import LogisticRegression 
lr = LogisticRegression(max_iter=1000)

from sklearn.naive_bayes import GaussianNB
g=GaussianNB()



train_df = pd.read_csv('/Users/koreynishimoto/Desktop/Titanic/'
                   +'train.csv', index_col=False)

y_col = 'Survived'

train_df['Sex'] = train_df['Sex'].map({'male': 1, 'female': 0})



####################### Makes the cabin number irrelevent but uses the floor as the delineator
train_df['Cabin'] = train_df['Cabin'].fillna('none')


train_df['Cabin'] = train_df['Cabin'].apply(lambda x: '1' if 'A' in x else x)
train_df['Cabin'] = train_df['Cabin'].apply(lambda x: '2' if 'B' in x else x)
train_df['Cabin'] = train_df['Cabin'].apply(lambda x: '3' if 'C' in x else x)
train_df['Cabin'] = train_df['Cabin'].apply(lambda x: '4' if 'D' in x else x)
train_df['Cabin'] = train_df['Cabin'].apply(lambda x: '5' if 'E' in x else x)
train_df['Cabin'] = train_df['Cabin'].apply(lambda x: '6' if 'F' in x else x)
train_df['Cabin'] = train_df['Cabin'].apply(lambda x: '7' if 'G' in x else x)
train_df['Cabin'] = train_df['Cabin'].apply(lambda x: '0' if 'none' in x else x)
train_df['Cabin'] = train_df['Cabin'].apply(lambda x: '0' if 'T' in x else x)

#######################



'''
#ignores the floor and uses the cabin number


train_df['Cabin'] = train_df['Cabin'].str.replace('A', '', regex=True)
train_df['Cabin'] = train_df['Cabin'].str.replace('B', '', regex=True)
train_df['Cabin'] = train_df['Cabin'].str.replace('C', '', regex=True)
train_df['Cabin'] = train_df['Cabin'].str.replace('D', '', regex=True)
train_df['Cabin'] = train_df['Cabin'].str.replace('E', '', regex=True)
train_df['Cabin'] = train_df['Cabin'].str.replace('F', '', regex=True)
train_df['Cabin'] = train_df['Cabin'].str.replace('G', '', regex=True)
train_df['Cabin'] = train_df['Cabin'].str.replace('T', '', regex=True)
train_df['Cabin'] = train_df['Cabin'].str.replace(' ', '', regex=True)
train_df['Cabin'] = train_df['Cabin'].replace(r'^\s*$', np.NaN, regex=True)

#remove white space for items like c28 c35 c23 and replaces NAN as 0

train_df['Cabin'] = train_df['Cabin'].str.replace(' .*','', regex=True)

train_df['Cabin'] = train_df['Cabin'].fillna(0)
train_df['Cabin'] = train_df['Cabin'].astype(int)

train_df['Cabin 0-75'] = (train_df['Cabin']<=75).astype(int)
train_df['Cabin 76-150'] = ( (train_df['Cabin']<=150) & (train_df['Cabin']>=76) ).astype(int)
'''







train_df[['Embarked_C', 'Embarked_Q', 'Embarked_S', 'Embarked_nan']] = pd.get_dummies(train_df.Embarked.astype(str), prefix='Embarked')

train_df = train_df.drop(columns=['Name', 'PassengerId', 'Ticket', 'Embarked'])

X_cols = [x for x in train_df if x != y_col]

#train_df['Age'] = train_df['Age'].fillna(0)
train_df['Age'] = train_df['Age'].fillna(train_df['Age'].mean())

# Dont run this on the test dataset
# train_df is what is used to fit the model
lr.fit(train_df[X_cols], train_df[y_col])

train_df['prediction'] = lr.predict(train_df[X_cols])

print('Accuracy for logistic regression is', (train_df[y_col] == train_df['prediction']).mean())


    
'''
#################################

train_df2= train_df.drop(columns=['prediction'])

gX_cols = [x for x in train_df2 if x != y_col]

g.fit(train_df2[gX_cols], train_df2[y_col])

train_df2['Gaus prediction'] = g.predict(train_df2[gX_cols])
print('Accuracy for Naive Gaussian is', (train_df2[y_col] == train_df2['Gaus prediction']).mean())


#random forest
'''



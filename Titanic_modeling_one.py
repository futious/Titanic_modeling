"""
Created on Fri Sep 24 07:59:06 2021

@author: koreynishimoto

Some words

"""

import pandas as pd

from numpy import mean
from numpy import std

from sklearn.model_selection import GridSearchCV
from sklearn_evaluation import plot


from sklearn.linear_model import LogisticRegression 
lr = LogisticRegression(max_iter=1000,C=1)


from sklearn.naive_bayes import GaussianNB
g=GaussianNB()

from sklearn.svm import SVC 
clf = SVC(max_iter=100000, C=1000)


from sklearn.model_selection import KFold

from sklearn.model_selection import cross_val_score


from sklearn import tree
dt = tree.DecisionTreeClassifier(max_depth=10)


from signal import signal, SIGPIPE, SIG_DFL
signal(SIGPIPE,SIG_DFL) 





#################################################################################################
########################################################################
########################################################################


train_df = pd.read_csv('/Users/koreynishimoto/Desktop/Titanic/'
                   +'train.csv', index_col=False)

y_col = 'Survived'

train_df['Sex'] = train_df['Sex'].map({'male': 1, 'female': 0})


train_df[['Embarked_C', 'Embarked_Q', 'Embarked_S', 'Embarked_nan']] = pd.get_dummies(train_df.Embarked.astype(str), prefix='Embarked')

train_df = train_df.drop(columns=['Name', 'PassengerId', 'Ticket', 'Embarked'])

X_cols = [x for x in train_df if x != y_col]

##################################################################################################
########################################################################
########################################################################


#GRID TESTING

# set the paramaters that you want to test
param_grid = {'C': [0.1, 1, 10, 100,1000],  
              'max_iter': [100,200,300,400,500,600,700,800,900,1000]}

# create a grid that will test the above parameters. change first part to
# Change the type of model.
grid = GridSearchCV(lr, param_grid, refit = True, verbose = 3,n_jobs=-1) 







#################################################################################################
########################################################################
########################################################################


#Logistic regression

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









#train_df['Age'] = train_df['Age'].fillna(0)
train_df['Age'] = train_df['Age'].fillna(train_df['Age'].mean())

# Dont run this on the test dataset
# train_df is what is used to fit the model
lr.fit(train_df[X_cols], train_df[y_col])

train_df['prediction'] = lr.predict(train_df[X_cols])

print('\n Accuracy for logistic regression is', (train_df[y_col] == train_df['prediction']).mean())




###################Kfold testing

kfold = KFold(3, shuffle=True,random_state=1)

train_df4= train_df.drop(columns=['prediction'])

testX_cols = [x for x in train_df4 if x != y_col]




#K fold cross examination for Logistical regression model
# evaluate model
lr_scores = cross_val_score(lr, train_df4[testX_cols], train_df4[y_col], scoring='accuracy', cv=kfold, n_jobs=-1)

lr_test=[]
lr_test.append(lr_scores)
# report performance

print('\n Logistic Regression Accuracy: %.3f (%.3f)' % (mean(lr_scores), std(lr_scores)))

print(lr_test)



    

#################################################################################################
########################################################################
########################################################################



#SVC testing
train_df2= train_df.drop(columns=['prediction'])

gX_cols = [x for x in train_df2 if x != y_col]

clf.fit(train_df2[gX_cols], train_df2[y_col])

train_df2['SVC prediction'] = clf.predict(train_df2[gX_cols])
print('\n Accuracy for SVC is', (train_df2[y_col] == train_df2['SVC prediction']).mean())


#K fold cross examination for SVC model
clf_scores = cross_val_score(clf, train_df4[testX_cols], train_df4[y_col], scoring='accuracy', cv=kfold, n_jobs=-1)

clf_test=[]
clf_test.append(clf_scores)
# report performance

print('\n Support Vector Machines Accuracy: %.3f (%.3f)' % (mean(clf_scores), std(clf_scores)))

print(clf_test)





#################################################################################################
########################################################################
########################################################################


#Decision tree

train_df5= train_df.drop(columns=['prediction'])

dtX_cols = [x for x in train_df if x != y_col]

dt.fit(train_df5[gX_cols], train_df5[y_col])



train_df5['Decission Tree prediction'] = dt.predict(train_df5[gX_cols])
print('\n Accuracy for Decission Tree is', (train_df5[y_col] == train_df5['Decission Tree prediction']).mean())

#Decision tree cross examination for SVC model

testX_cols = [x for x in train_df5 if x != y_col]

dt_scores = cross_val_score(dt, train_df5[testX_cols], train_df5[y_col], scoring='accuracy', cv=kfold, n_jobs=-1)

dt_test=[]
dt_test.append(dt_scores)
# report performance

print('\n Decision tree Accuracy: %.3f (%.3f)' % (mean(dt_scores), std(dt_scores)))

print(dt_test)










#################################################################################################
########################################################################
########################################################################
#GRID TESTING 


#Decision trees have a random element to them and the nest found may vary over different runs.


param_grid = {'max_depth': [3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22],
              'random_state': [1,2]
              }

# create a grid that will test the above parameters. change first part to
# Change the type of model.
grid = GridSearchCV(dt, param_grid, refit = True, verbose = 3,n_jobs=-1) 





# paramater changing

train_df3= train_df.drop(columns=['prediction'])

testX_cols = [x for x in train_df3 if x != y_col]

grid.fit(train_df3[testX_cols], train_df[y_col])

train_df3['test prediction'] = grid.predict(train_df3[testX_cols])

print('\n Accuracy for test is', (train_df3[y_col] == train_df3['test prediction']).mean())





print(grid.best_score_)

#print best esetimators for all.
print(grid.best_estimator_)


plot.grid_search(grid.cv_results_, change='max_depth', kind='bar')
 

''' Use this code to find the best paramater settings. You will need to 
change the paramaters for different tests.

#print best estimators for individuals
print(grid.best_estimator_.max_iter)
print(grid.best_estimator_.C)
'''

'''    















    



# -*- coding: utf-8 -*-
"""
Created on Mon Apr  9 22:51:08 2018

@author: Ananthu
"""

import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.linear_model import LinearRegression

train_df = pd.read_csv('F:/TA/Analytics/Kaggle/Titanic/Data/train.csv')
test_df = pd.read_csv('F:/TA/Analytics/Kaggle/Titanic/Data/test.csv')

# train_df.head()
# col = train_df.columns.tolist()

# unique = []
# for item in col:
#     unique.append(train_df[item].nunique())


'''

pclass1 - cabin A(15),B(47),C(59),D(29),E(25),T(1). total - 176
pclass2 - cabin D(4),E(4),F(8). total - 16
pclass3 - cabin E(4),F(5),G(7). total - 12

'''

'''
Predicting age in test data

train_df0 = train_df
train_df0 = train_df0.drop(['Survived','PassengerId','Name','Ticket','Cabin'],axis=1)
train_df0['Sex'] = np.where(train_df0['Sex'] == 'male',1,0)
train_df0['Embarked'] = np.where(train_df0['Embarked'] == 'C',1,np.where(train_df0['Embarked'] == 'Q',2,3))
train_df0 = train_df0.dropna()

X_train0 = train_df0.loc[:,['Pclass','Sex','SibSp','Parch','Fare','Embarked']]
Y_train0 = train_df0.loc[:,['Age']]

model0 = LinearRegression()
model0.fit(X_train0,Y_train0)

test_df0 = test_df
test_df0 = test_df0.drop(['PassengerId','Name','Ticket','Cabin'],axis=1)
test_df0['Sex'] = np.where(test_df0['Sex'] == 'male',1,0)
test_df0['Embarked'] = np.where(test_df0['Embarked'] == 'C',1,np.where(test_df0['Embarked'] == 'Q',2,3))

test_df0 = test_df0[test_df0['Age'].isna()]
X_test0 = test_df0.loc[:,['Pclass','Sex','SibSp','Parch','Fare','Embarked']]

Y_test0 = model0.predict(X_test0)
test_df0['Age'] = Y_test0

nulls = test_df[pd.isnull(test_df['Age'])]
for i, ni in enumerate(nulls.index[:len(test_df0)]):
    test_df['Age'].loc[ni] = test_df0['Age'].iloc[i]

'''

train_df = train_df.drop(['PassengerId','Name','SibSp','Parch','Ticket','Fare','Cabin','Embarked'],axis=1)
train_df['Sex'] = np.where(train_df['Sex'] == 'male',1,0)
train_df = train_df.dropna()

X_train = train_df.loc[:,['Pclass','Sex','Age']]
Y_train = train_df.loc[:,['Survived']]

model = SVC()
model.fit(X_train,Y_train)

test_df = test_df.drop(['Name','SibSp','Parch','Ticket','Fare','Cabin','Embarked'],axis=1)
test_df['Sex'] = np.where(test_df['Sex'] == 'male',1,0)
test_df = test_df.dropna()

X_test = test_df.loc[:,['Pclass','Sex','Age']]

Y_test = model.predict(X_test).tolist()

final_df = pd.DataFrame(test_df['PassengerId'])
final_df['Survived'] = Y_test
final_df = final_df.set_index('PassengerId')

final_df.to_csv('Submission2.csv')

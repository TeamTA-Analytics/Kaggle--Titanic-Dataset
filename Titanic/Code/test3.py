# -*- coding: utf-8 -*-
"""
Created on Sat May  5 19:56:49 2018

@author: Ananthu
"""

a = train_df0.loc[:,['Cabin','Pclass']]

a = a.dropna()

import matplotlib.pyplot as plt

b = a.groupby('Pclass')['Cabin'].count()


class1 = a[a['Pclass'] == 1]
class2 = a[a['Pclass'] == 2]
class3 = a[a['Pclass'] == 3]

class1 = class1['Cabin'].str.split('\d+').apply(lambda x:x[0])
class1.value_counts()

'''

train_df0[train_df0['Embarked'].isna()]
train_df0['Ticket'].value_counts()
train_df0[(train_df0['Ticket'] == '347082')]
train_df0[(train_df0['Ticket'] == '1601')]
train_df0[(train_df0['Ticket'] == 'CA. 2343')]

train_df0[(train_df0['Pclass']==1)and (train_df0['Cabin']=='B\d')]
c = train_df0.dropna()
c =c[(c['Cabin'].str.startswith('B'))]

'''

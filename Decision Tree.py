#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#AI Hackathon

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('ks-projects-201801.csv')

import datetime
df['launched'] = pd.to_datetime(df['launched'])  
df['deadline'] = pd.to_datetime(df['deadline']) 
#raw_df.head()
df = df.loc[df['launched'] < datetime.date(2018,1,1)]
df = df.loc[df['launched'] > datetime.date(2009,12,31)]
df = df.loc[df['launched'] < df['deadline']]
df = df.loc[df['usd_goal_real'] > 0]
df = df.loc[df['state'].isin(['failed','canceled','suspended','successful'])]
df['period'] = (df['deadline'] - df['launched']).astype('timedelta64[D]')
d = {'successful': True, 'failed': False, 'canceled': False, 'suspended': False}
df['successful'] = df['state'].map(d)


x_orig = df.loc[:, ('category', 'period','usd_pledged_real','usd_goal_real')]
y = df.loc[:,'successful']

dummies = pd.get_dummies(x_orig.category)
x = x_orig.join(dummies)
x = x.drop(['category'], axis = 1)

from sklearn.model_selection import train_test_split
train_x, test_x, train_y, test_y = train_test_split(x, y,random_state = 0)

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

classifier = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
classifier = classifier.fit(train_x, train_y)
y_pred = classifier.predict(test_x)
accuracy = accuracy_score(test_y,y_pred)

accuracy

from sklearn.externals.six import StringIO  
from IPython.display import Image  
from sklearn.tree import export_graphviz
import pydotplus
dot_data = StringIO()
export_graphviz(classifier, out_file=dot_data,  
                filled=True, rounded=True,
                special_characters=True)
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
Image(graph.create_png())


#!/usr/bin/env python3
# -*- coding: utf-8 -*-


#Light GBM Kickstarter

import os
import pandas as pd
import seaborn as sns
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from collections import defaultdict
import datetime
import numpy as np
from sklearn.metrics import accuracy_score


#Data cleansing
df=pd.read_csv(r'ks-projects-201801.csv',parse_dates={'deadline_dt':[5],'launched_dt':[7]})
df['camp_time']=(df['deadline_dt']-df['launched_dt']).astype('timedelta64[h]')
df['deadline_mth'] = df['deadline_dt'].map(lambda x: x.month)
df['deadline_day'] = df['deadline_dt'].map(lambda x: x.day)
df['launched_mth'] = df['launched_dt'].map(lambda x: x.month)
df['launched_day'] = df['launched_dt'].map(lambda x: x.day)

#Tag outcomes
df['outcome']=0
df.loc[df['state']=='successful','outcome']=1

#Transform categorical into integers
categorical_features=[
        'category', 'main_category','currency', 'country',
        'deadline_mth', 'deadline_day', 'launched_mth', 'launched_day'
        ]
d = defaultdict(LabelEncoder)
fit = df[categorical_features].apply(lambda x: d[x.name].fit_transform(x))
for var in categorical_features:
    df[var ]=d[var ].transform(df[var ])

#Split dataset into train and oot
keep_cols=['category', 'main_category','currency', 'country','usd_goal_real', 'camp_time',
 'deadline_mth', 'deadline_day', 'launched_mth', 'launched_day',
'outcome']

oot_df = df.loc[
        (df['launched_dt'] > datetime.date(2016,12,31)) & 
        (df['state'].isin(['failed','canceled','suspended','successful']))
        ,keep_cols
]

train_df = df.loc[
        (df['launched_dt'] <= datetime.date(2016,12,31)) & 
        (df['state'].isin(['failed','canceled','suspended','successful']))
        ,keep_cols
]

train_x, test_x, train_y, test_y = train_test_split(train_df.drop(['outcome'], axis=1), train_df['outcome'], test_size=0.5, random_state=123)

d_train = lgb.Dataset(train_x, label=train_y,categorical_feature=categorical_features)
d_val = lgb.Dataset(test_x, label=test_y,categorical_feature=categorical_features)

lgb_params = {
        'boosting_type': 'gbdt',
        'objective': 'binary',
        'metric':'auc',
        'learning_rate': 0.01,
        #'is_unbalance': 'true',  #because training data is unbalance (replaced with scale_pos_weight)
        'num_leaves': 31,  # we should let it be smaller than 2^(max_depth)
        'max_depth': -1,  # -1 means no limit
        'min_child_samples': 20,  # Minimum number of data need in a child(min_data_in_leaf)
        'max_bin': 255,  # Number of bucketed bin for feature values
        'subsample': 0.6,  # Subsample ratio of the training instance.
        'subsample_freq': 0,  # frequence of subsample, <=0 means no enable
        'colsample_bytree': 0.3,  # Subsample ratio of columns when constructing each tree.
        'min_child_weight': 5,  # Minimum sum of instance weight(hessian) needed in a child(leaf)
        'subsample_for_bin': 200000,  # Number of samples for constructing bin
        'min_split_gain': 0,  # lambda_l1, lambda_l2 and min_gain_to_split to regularization
        'reg_alpha': 0,  # L1 regularization term on weights
        'reg_lambda': 0,  # L2 regularization term on weights
        'nthread': 4,
        'verbose': 0,
        'metric':'auc'
    }
rnds = 500

evals_results = {}
bst1 = lgb.train(lgb_params,d_train,valid_sets=[d_train, d_val], 
                     valid_names=['train','valid'], 
                     evals_result=evals_results, 
                     num_boost_round=3000,
                     early_stopping_rounds=100,
                     verbose_eval=10, 
                     feval=None)

pred=bst1.predict(oot_df.drop(['outcome'], axis=1),num_iteration=bst1.best_iteration)

for i in range(0,len(pred)):
    if pred[i]>=.5:       # setting threshold to .5
       pred[i]=1
    else:  
       pred[i]=0
       
y=np.array(oot_df['outcome'])

accuracy=accuracy_score(pred,y)



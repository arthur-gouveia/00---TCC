
# coding: utf-8

# # Anotações Gerais
# 
# ## Instalação do xgboost no Windows:
# https://www.ibm.com/developerworks/community/blogs/jfp/entry/Installing_XGBoost_For_Anaconda_on_Windows?lang=en

# In[5]:

import xgboost as xgb

dtrain = xgb.DMatrix('agaricus.txt.train')
dtest = xgb.DMatrix('agaricus.txt.test')


# In[26]:

# Example from https://www.ibm.com/developerworks/community/blogs/jfp/entry/Installing_XGBoost_For_Anaconda_on_Windows?lang=en

import numpy as np

data = np.random.rand(5,10) # 5 entities, each contains 10 features
label = np.random.randint(2, size=5) # binary target

param = {'bst:max_depth':2, 'bst:eta':1, 'silent':1, 'objective':'binary:logistic' }
param['nthread'] = 4
param['eval_metric'] = 'auc'
param['eta'] = 0
param['max_depth'] = 0
param['subsample'] = 0

etarange = range(0, 10, 2)
maxdepthrange = range(0, 100, 2)
subsamplerange = range(20, 50, 10)

evallist  = [(dtest,'eval'), (dtrain,'train')]

for eta in etarange:
    for maxdepth in maxdepthrange:
        for subsample in subsamplerange:
            num_round = 2
            param['eta'] = eta/10
            param['max_depth'] = maxdepth
            param['subsample'] = subsample/100
            bst = xgb.train( param, dtrain, num_round, evallist )
            preds = bst.predict(dtest)
            labels = dtest.get_label()
            error = sum(1 for i in range(len(preds)) if int(preds[i]>0.5)!=labels[i]) /float(len(preds))
            errorstr = 'error: {:1.8f}'.format(error)
            etastr = 'eta: {}'.format(eta/10)
            maxdepthstr = 'max_depth: {}'.format(maxdepth)
            subsamplestr = 'subsample: {}'.format(subsample/100)
            print ('{:<20}{:<15}{:<20}{:<20}'.format(errorstr, etastr, maxdepthstr, subsamplestr))


# In[ ]:

import numpy as np

data = np.random.rand(5,10) # 5 entities, each contains 10 features
label = np.random.randint(2, size=5) # binary target

param = {'bst:max_depth':2, 'bst:eta':1, 'silent':1, 'objective':'binary:logistic' }
param['nthread'] = 4
param['eval_metric'] = 'auc'
param['eta'] = 0
param['max_depth'] = 0
param['subsample'] = 0


etarange = range(1, 10, 2)
maxdepthrange = range(2, 100, 2)
subsamplerange = range(30, 60, 10)

evallist  = [(dtest,'eval'), (dtrain,'train')]

for eta in etarange:
    for maxdepth in maxdepthrange:
        for subsample in subsamplerange:
            num_round = 2
            param['eta'] = eta/10
            param['max_depth'] = maxdepth
            param['subsample'] = subsample/100
            bst = xgb.train( param, dtrain, num_round, evallist )
            preds = bst.predict(dtest)
            labels = dtest.get_label()
            error = sum(1 for i in range(len(preds)) if int(preds[i]>0.5)!=labels[i]) /float(len(preds))
            errorstr = 'error: {:1.8f}'.format(error)
            etastr = 'eta: {}'.format(eta/10)
            maxdepthstr = 'max_depth: {}'.format(maxdepth)
            subsamplestr = 'subsample: {}'.format(subsample/100)
            print ('{:<20}{:<15}{:<20}{:<20}'.format(errorstr, etastr, maxdepthstr, subsamplestr))


# In[25]:

import xgboost as xgb
import numpy as np

data = np.random.rand(5,10) # 5 entities, each contains 10 features
label = np.random.randint(2, size=5) # binary target

param = {'bst:max_depth':2, 'bst:eta':1, 'silent':1, 'objective':'binary:logistic' }
param['nthread'] = 4
param['eval_metric'] = 'auc'
param['eta'] = 0
param['max_depth'] = 0
param['subsample'] = 0

etarange = range(0, 10, 1)
maxdepthrange = range(0, 100, 1)
subsamplerange = range(20, 50, 5)

evallist  = [(dtest,'eval'), (dtrain,'train')]

num_round = 1
eta = 0.9
maxdepth = 2
subsample  = 0.2
param['eta'] = eta
param['max_depth'] = maxdepth
param['subsample'] = subsample
bst = xgb.train( param, dtrain, num_round, evallist )
preds = bst.predict(dtest)
labels = dtest.get_label()
error = sum(1 for i in range(len(preds)) if int(preds[i]>0.5)!=labels[i]) /float(len(preds))
errorstr = 'error: {:1.8f}'.format(error)
etastr = 'eta: {}'.format(eta)
maxdepthstr = 'max_depth: {}'.format(maxdepth)
subsamplestr = 'subsample: {}'.format(subsample)
print ('{:<20}{:<15}{:<20}{:<20}'.format(errorstr, etastr, maxdepthstr, subsamplestr))
print(preds)


# In[17]:

# Example from https://xgboost.readthedocs.io/en/latest/get_started/
import xgboost as xgb
import timeit

# read in data
dtrain = xgb.DMatrix('agaricus.txt.train')
dtest = xgb.DMatrix('agaricus.txt.test')
# specify parameters via map
param = {'max_depth':2, 'eta':1, 'silent':1, 'objective':'binary:logistic' }
param['nthread'] = 4
num_round = 2
get_ipython().magic('timeit bst = xgb.train(param, dtrain, num_round)')
# make prediction
preds = bst.predict(dtest)


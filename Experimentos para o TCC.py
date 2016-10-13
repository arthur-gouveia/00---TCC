
# coding: utf-8

# # Anotações Gerais
# 
# ## Instalação do xgboost no Windows:
# https://www.ibm.com/developerworks/community/blogs/jfp/entry/Installing_XGBoost_For_Anaconda_on_Windows?lang=en

# In[2]:

import xgboost as xgb
import numpy as np
import time

dtrain = xgb.DMatrix('agaricus.txt.train')
dtest = xgb.DMatrix('agaricus.txt.test')


# In[7]:

# Example from https://www.ibm.com/developerworks/community/blogs/jfp/entry/Installing_XGBoost_For_Anaconda_on_Windows?lang=en
import numpy as np
import time

start_time = time.time()

param = {'bst:max_depth':2, 'bst:eta':1, 'silent':1, 'objective':'binary:logistic' }
param['nthread'] = 4
param['eval_metric'] = 'auc'
param['eta'] = 0
param['max_depth'] = 0
param['subsample'] = 0

etarange = np.arange(0, 0.5, 0.2) # original: (0, 1, 0.2)
maxdepthrange = range(0, 5, 2) # original: (0, 100, 2)
subsamplerange = np.arange(0.2, 0.4, 0.1) # original (0.2, 0.5, 0.1)

evallist  = [(dtest,'eval'), (dtrain,'train')]

paramlist = [(eta, maxdepth, subsample) for eta in etarange
                                        for maxdepth in maxdepthrange
                                        for subsample in subsamplerange]
def do_the_clean_job():
    for par in paramlist:
        num_round = 2
        param['eta'] = par[0]
        param['max_depth'] = par[1]
        param['subsample'] = par[2]
        bst = xgb.train( param, dtrain, num_round, evallist )
        preds = bst.predict(dtest)
        labels = dtest.get_label()
        error = sum(1 for i in range(len(preds)) if int(preds[i]>0.5)!=labels[i]) /float(len(preds))
        errorstr = 'error: {:1.8f}'.format(error)
        etastr = 'eta: {}'.format(par[0])
        maxdepthstr = 'max_depth: {}'.format(par[1])
        subsamplestr = 'subsample: {}'.format(par[2])
        print ('{:<20}{:<15}{:<20}{:<20}'.format(errorstr, etastr, maxdepthstr, subsamplestr))

get_ipython().magic('timeit -n5 do_the_clean_job()')

print("Total time: {:5.4f}s".format(time.time() - start_time))


# In[12]:

import time
import numpy as np

start_time = time.time()

param = {'bst:max_depth':2, 'bst:eta':1, 'silent':1, 'objective':'binary:logistic' }
param['nthread'] = 4
param['eval_metric'] = 'auc'
param['eta'] = 0
param['max_depth'] = 0
param['subsample'] = 0


etarange = range(0, 5, 2) # original (0, 10, 2)
maxdepthrange = range(1, 5, 2) # original (1, 100, 2)
subsamplerange = range(20, 40, 10) # original (20, 50, 10)

evallist  = [(dtest,'eval'), (dtrain,'train')]

def do_the_job():
    for subsample in subsamplerange:
        for maxdepth in maxdepthrange:
            for eta in etarange:
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

                
get_ipython().magic('timeit -n5 do_the_job()')

print("Tempo total: {:5.4f}s".format(time.time() - start_time))


# In[11]:

import time
import numpy as np

start_time = time.time()

param = {'bst:max_depth':2, 'bst:eta':1, 'silent':1, 'objective':'binary:logistic' }
param['nthread'] = 4
param['eval_metric'] = 'auc'
param['eta'] = 0
param['max_depth'] = 0
param['subsample'] = 0


etarange = np.arange(0, 0.5, 0.2) # original (0, 10, 2)
maxdepthrange = range(1, 5, 2) # original (1, 100, 2)
subsamplerange = np.arange(0.20, 0.40, 0.10) # original (20, 50, 10)

evallist  = [(dtest,'eval'), (dtrain,'train')]

def do_the_job():
    for subsample in subsamplerange:
        for maxdepth in maxdepthrange:
            for eta in etarange:
                num_round = 2
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

                
get_ipython().magic('timeit -n5 do_the_job()')

print("Tempo total: {:5.4f}s".format(time.time() - start_time))


# In[ ]:

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


# In[ ]:

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


# # Agora tentando fazer o scipy maximizar AUC

# In[34]:

import scipy.optimize as sco
from sklearn.metrics import roc_auc_score
from scipy.optimize import maximize

param = {'bst:max_depth':2, 'bst:eta':1, 'silent':1, 'objective':'binary:logistic' }
param['nthread'] = 4
param['eval_metric'] = 'auc'
param['eta'] = 0.5
param['max_depth'] = 3
param['subsample'] = 0.5


def train_auc(parameters, traindata, testdata):
    
    evallist = [(testdata, 'eval'), (traindata, 'train')]
    num_round = 2
    bst = xgb.train(parameters, traindata, num_round, evallist)
    pred = bst.predict(dtest)
    return roc_auc_score(testdata.get_label(), pred)


def encapsulador()
auc = train_auc(param, dtrain, dtest)


# -*- coding: utf-8 -*-
"""
Created on Mon Oct 12 15:34:08 2020

@author: donchin
"""

#%% Imports
import pystan
import pickle
from hashlib import md5

def StanModel_cache(model_code, model_name=None, **kwargs):
    """Use just as you would `stan`"""
    code_hash = md5(model_code.encode('ascii')).hexdigest()
    if model_name is None:
        cache_fn = 'cached-model-{}.pkl'.format(code_hash)
    else:
        cache_fn = 'cached-{}-{}.pkl'.format(model_name, code_hash)
    try:
        sm = pickle.load(open(cache_fn, 'rb'))
    except:
        sm = pystan.StanModel(model_code=model_code)
        with open(cache_fn, 'wb') as f:
            pickle.dump(sm, f)
    else:
        print("Using cached StanModel")
    return sm

model_code = 'parameters {real y;} model {y ~ normal(0,1);}'

# with same model_code as before
data = dict(N=10, y=[0, 1, 0, 0, 0, 0, 0, 0, 0, 1])
sm = StanModel_cache(model_code=model_code)
fit = sm.sampling(data=data, n_jobs=1)
print(fit)

new_data = dict(N=6, y=[0, 0, 0, 0, 0, 1])
# the cached copy of the model will be used
sm = StanModel_cache(model_code=model_code)
fit2 = sm.sampling(data=new_data, n_jobs=1)
print(fit2)
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 12 15:34:08 2020

@author: donchin
"""

#%% Imports
import pystan
import arviz as az
import numpy as np
import pickle
from hashlib import md5

#%% Set up for stan
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

model_file = 'twoStateTanModelNoise.stan'

#%% Get data
f = open('tanDataSim.pkl','rb')
generated_data = pickle.load(f)
f.close()

def logit(p):
    return np.log(p / (1-p))

def range2LogisticMuS(m, M):
    lm = logit(m)
    lM = logit(M)
    s = (lM - lm)/2
    mu = (lM + lm)/2
    return mu, s

minA1 = 0.85
maxA1 = 0.99
muA1, sA1 = range2LogisticMuS(minA1, maxA1)

minB1 = 0.05
maxB1 = 0.3
muB1, sB1 = range2LogisticMuS(minB1, maxB1)

minLogisticPriorPChange = 0.001
maxLogisticPriorPChange = 0.2
muLogisticPriorPChange, sLogisticPriorPChange = range2LogisticMuS(
    minLogisticPriorPChange, maxLogisticPriorPChange)

# with same model_code as before
data = dict(
    N=len(generated_data["y"]), 
    sBroad = generated_data["sBroad"],
    underWeight = generated_data["underWeight"],
    priorWeight = generated_data["priorWeight"],
    p = generated_data["p"],
    y = generated_data["y"],
    
    muA1 = muA1, 
    sA1 = sA1,
    muB1 = muB1, 
    sB1 = sB1, 
    
    muLogisticPriorPChange = muLogisticPriorPChange,
    sLogisticPriorPChange = sLogisticPriorPChange,
    
    mdSPHat1 = 2, # Perturbation uncertainty should start pretty low
    sSPHat1 = 2,
    
    mdSEpsilon = 5, # Not sure what best to put here
    sSEpsilon = 10,
    
    mdSEta = 1,
    sSEta = 1,
    
    )


#%% Fit the model
with open(model_file, 'r') as file:
    model_code = file.read()
    
sm = StanModel_cache(model_code=model_code)
fit = sm.sampling(data=data, n_jobs=1, iter=2000, verbose=True, refresh=1,
                  control={'max_treedepth': 10}, chains=4)

stan_fit = {'model': sm, 'fit': fit, 'data': data, 'iter': 500}
with open('tanFitSim.pkl','wb') as f:
    pickle.dump(stan_fit, f)

#%% Load fit
f = open('tanFitSim.pkl','rb')
stan_fit = pickle.load(f)
f.close()

fit = stan_fit["fit"]
sm = stan_fit["model"]

#%% Plot diagnostics

az.plot_trace(fit, var_names=["A", "B", "sEpsilon", "sEta"])
az.plot_trace(fit, var_names=["A1", "B1"])

#%% Make az formatted data
az_data = az.from_pystan(
    posterior=fit,
    posterior_predictive="y_hat",
    observed_data=["y"],
    log_likelihood={"y": "log_lik"},
    coords={"school": np.arange(eight_school_data["J"])},
    dims={
        "theta": ["school"],
        "y": ["school"],
        "log_lik": ["school"],
        "y_hat": ["school"],
        "theta_tilde": ["school"],
    },
)

stan_data

az.plot_density(fit, var_names=["A", "B", "sEpsilon", "sEta"])

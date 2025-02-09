# -*- coding: utf-8 -*-
"""
Created on Mon Oct 12 15:34:08 2020

@author: donchin
"""

#%% Imports
import pystan
import arviz as az
import numpy as np
import xarray as xr
import pickle
from hashlib import md5
import copy

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


#%% Get data
f = open('tanDataPredErrSim.pkl','rb')
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

#%% Create data list that carries strong priors

data_strong_priors = copy.copy(data)
minA1 = 0.88
maxA1 = 0.92
muA1, sA1 = range2LogisticMuS(minA1, maxA1)

minB1 = 0.08
maxB1 = 0.12
muB1, sB1 = range2LogisticMuS(minB1, maxB1)

minLogisticPriorPChange = 0.08
maxLogisticPriorPChange = 0.12
muLogisticPriorPChange, sLogisticPriorPChange = range2LogisticMuS(
    minLogisticPriorPChange, maxLogisticPriorPChange)

data_strong_priors["muA1"] = muA1 
data_strong_priors["sA1"] = sA1
data_strong_priors["muB1"] = muB1 
data_strong_priors["sB1"] = sB1 
data_strong_priors["muLogisticPriorPChange"] = muLogisticPriorPChange  
data_strong_priors["sLogisticPriorPChange"] = sLogisticPriorPChange
data_strong_priors["mdSPHat1"] = 2
data_strong_priors["sSPHat1"] = 0.2
data_strong_priors["mdSEpsilon"] = 5
data_strong_priors["sSEpsilon"] = 2    
data_strong_priors["mdSEta"] = 1
data_strong_priors["sSEta"] = 0.2


#%% Load the model
model_file = 'tanModelPredErr.stan'
with open(model_file, 'r') as file:
    model_code = file.read()
    
sm = StanModel_cache(model_code=model_code)

#%% Fit the model with strong priors
fit = sm.sampling(data=data_strong_priors, n_jobs=1, iter=2000, verbose=True, refresh=1,
                  control={'max_treedepth': 20, 'adapt_delta': 0.99}, chains=4)

stan_fit_strong_priors = {'model': sm, 'fit': fit, 'data': data, 'iter': 500}
with open('tanFitSimStrongPriors.pkl','wb') as f:
    pickle.dump(stan_fit_strong_priors, f)

#%% Get sample with max likelihood
az_fit = az.convert_to_inference_data(fit)
lp = az_fit.sample_stats["lp"]
max_lp_index = xr.DataArray.argmax(lp, dim=("chain", "draw"))
max_lp_sample = az_fit.posterior[max_lp_index]

#%% Set init to start at max likelihood sample
def init_f():
  return {
      "A1": max_lp_sample["A1"].item(),
      "B1": max_lp_sample["B1"].item(),
      "sEpsilon": max_lp_sample["sEpsilon"].item(),
      "sEta": max_lp_sample["sEta"].item(),
      "sPHat1": max_lp_sample["sPHat1"].item(),
      "logisticPriorPChange": max_lp_sample["logisticPriorPChange"].item(),
      "x": max_lp_sample["x"].values,
      "pHat": max_lp_sample["pHat"].values,
      }


#%% Fit the model
with open(model_file, 'r') as file:
    model_code = file.read()
    
sm = StanModel_cache(model_code=model_code)
fit = sm.sampling(data=data, init=init_f,
                  n_jobs=1, iter=2000, chains=3,
                  verbose=True, refresh=1,
                  control={'max_treedepth': 20, 'adapt_delta': 0.99, }, )

stan_fit = {'model': sm, 'fit': fit, 'data': data, 'iter': 500}
with open('tanFitPredErr.pkl','wb') as f:
    pickle.dump(stan_fit, f)



#%% Load fit
f = open('tanFitPredErr.pkl','rb')
stan_fit = pickle.load(f)
f.close()

fit = stan_fit["fit"]
sm = stan_fit["model"]
data = stan_fit["data"]
#%% Plot diagnostics

az.plot_trace(fit, var_names=["A", "B", "sEpsilon", "sEta"])
az.plot_trace(fit, var_names=["A1", "B1"])
az.plot_trace(fit, var_names=["sPHat1", "priorPChange"])

#%% Make az formatted data
az_fit = az.convert_to_inference_data(fit)

#%% Plot data and state

# Pick a sample at random
fit_coords = az_fit.posterior.coords
r_chain = np.random.choice(fit_coords["chain"])
r_draw = np.random.choice(fit_coords["draw"])

r_sample = np.random


t = range(0,N)
f, (ax1,ax2) = plt.subplots(2,1, figsize=(8, 8))
ax1.plot(t, p, label='p')
ax1.plot(t, y, label='y')
ax1.plot(t, pHat, label='\hat{p}')
ax1.plot(t, x, label='x')
ax1.legend()
ax1.yaxis.set_label_text('Degrees')

ax2.plot(t, sPHat, label='sPHat')
ax2.set_title('sPHat')
ax2.xaxis.set_label_text('Trials')
ax2.yaxis.set_label_text('degrees')

f.tight_layout()
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  7 14:17:23 2020

@author: donchin
"""

data {
    # Constants
    int<lower=2> N;
    # There is an argument to be made that all of these should be parameters
    real sBroad;
    real underWeight;
    real priorWeight; 
    
    # Data
    vector[N] p, y;
    
    # Priors
    real muA1, sA1; # Priors for A1
    real muB1, sB1; # Priors for B1
    # priors for sPHat[1]
    real mdSPHat1, sSPHat1;
    real mdSEpsilon, sSepsilon;
    
}

# gamRa = @(m,v) (m + sqrt((m)^2 + 4*v))/(2*v);
# gamSh = @(m,v) 1 + m*gamRa(m,v); 

transformed data {
    real<lower=0> raSPHat1 (mdSPHat1 + sqrt(mdSPHat1^2 + 4*sSPHat1^2)/(2*sSPHat1^2));
    real<lower=0> shSPHat1 1+mdSPHat1*raSPHat1;
    real<lower=0> raSEpsilon (mdSEpsilon + sqrt(mdSEpsilon^2 + 4*sSEpsilon^2)/(2*sSEpsilon^2));
    real<lower=0> shSPHat1 1+mdSEpsilon*raSEpsilon;
    
}

parameters {
    real A1, B1;
    real<lower=0> sEta;
    vector[N] pHat, x, eta, epsilon;
    real logisticPriorPChange;
}

transformed parameters {
    real<lower=0, upper=1> priorPChange = inv_logit(logisticPriorPChange);
    real logPriorPChange = log(priorPChange);
    real logPriorPNoChange = log(1 - priorPChange);
    
    real<lower=0, upper=1> A = inv_logit(A1);
    real<lower=0, upper=1> B = inv_logit(B1);
}

functions {
    real<lower=0, upper=1> pChangeFn(real p1, real p2) {
        # num = p2^underWeight*priorPChange;
        real logNum = underWeight*lP2 + logPriorPChange;
        
        # denom = p2^underWeight*priorPChange + p1^underWeight*(1-priorPChange);
        real logDenom = num + log(1 + exp( underWeight*lP1 + logPriorPNoChange - num ));

        return exp(logNum - logDenom);
    }
    
    real<lower=0> sPUpdate(real sPHat, real sEpsilon, real y, real sBroad) {
        real pPHat = normal_lpdf(y | 0, sqrt(sPHat^2+sEpsilon^2));
        real pBroad = normal_lpdf(y | 0, sqrt(sBroad^2+sEpsilon^2));
        real pChange = pChangeFn(pPHat, pBroad);
        
        real sPPosterior = sqrt( (priorWeight+1) / (1/sEpsilon^2 + priorWeight/sPHat^2) );
        return pChange*sBroad + (1-pChange)*sPPosterior;
    }
    
    real pHatUpdate(real pHat, real y, real sPHat, real sEpsilon) {
        real precPHat = 1/sPHat^2;
        real precY = 1/sEpsilon^2;
        return pHat - y*precY / (precY+precPHat);
    }    
}



model {
    vector<lower=0>[N] sPHat;

    A1 ~ normal(muA1, sA1);
    B1 ~ normal(muB1, sB1);
    sPHat[1] ~ gamma(shSPHat1, raSPHat1);
    sEpsilon ~ gamma(shSEpsilon, raSEpsilon);
    logisticPriorPChange ~ normal(muLogisticPriorPChange, sLogisticPriorPChange);
    
    pHat[1] ~ normal(0, sPHat[1]);
    x[1] ~ normal(0, sEta);
    epsilon[1] ~ normal(0, sEpsilon);
    predErr[1] = pHat[1] - p[1] + epsilon[1];
    y[1] = x[1]+predErr[1];
    eta[1] ~ normal(0, sEta);
    
    for (t in 2:N) {
        epsilon[t] ~ normal(0, sEpsilon);
        predErr[t] = pHat[t] - p[t] + epsilon[t];
        y[t] = x[t] + predErr[t];
        
        sPHat[t] = sPUpdate(sPHat[t-1], sEpsilon, y[t], pBroad);
        
        eta[t] ~ normal(0, sEta);
        if (t < N) {
            pHat[t+1] ~ pHatUpdate(pHat[t], y[t], sPHat[t], sEpsilon);
            x[t+1] = A*x[t] - B*y[t] + eta[t];
        }
    }
}



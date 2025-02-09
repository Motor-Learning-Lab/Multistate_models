functions {
    real pChangeFn(real lP1, real lP2, 
                                     real lPriorPChange, real lPriorPNoChange, real underWeight) {
        real logTerm = underWeight*(lP1-lP2) + lPriorPNoChange - lPriorPChange;
        return exp(1 / (1+logTerm));
    }
    
    real sPUpdate(real sPHat, real sEpsilon, real y, real sBroad, 
                        real priorWeight, real lPriorPChange, real lPriorPNoChange, 
                        real underWeight) {
        real lPPHat = normal_lpdf(y | 0, sqrt(sPHat^2+sEpsilon^2));
        real lPBroad = normal_lpdf(y | 0, sqrt(sBroad^2+sEpsilon^2));
        real pChange = pChangeFn(lPPHat, lPBroad, lPriorPChange, lPriorPNoChange, underWeight);
        
        real sPPosterior = sqrt( (priorWeight+1) / (1/sEpsilon^2 + priorWeight/sPHat^2) );
        return pChange*sBroad + (1-pChange)*sPPosterior;
    }
    
    real pHatUpdate(real pHat, real y, real sPHat, real sEpsilon) {
        real precPHat = 1/sPHat^2;
        real precY = 1/sEpsilon^2;
        return pHat - y*precY / (precY+precPHat);
    }    
    
    real gamMdStd2Rate(real md, real s) {
         return (md + sqrt(md*md + 4*s*s))/(2*s*s);
    }
    
    real gamMdStd2Shape(real md, real s) {
        return 1+md*gamMdStd2Rate(md, s);
    }
}

data {
    // Constants
    int<lower=2> N;
    // There is an argument to be made that all of these should be parameters
    real sBroad;
    real underWeight;
    real priorWeight; 
    
    // Data
    vector[N] p;
    vector[N] y;
    
    // Priors
    real muA1; // Logistic space
    real<lower=0> sA1; 
    real muB1; // Logistic space
    real<lower=0> sB1;
    real<lower=0> mdSPHat1; // Mode and standard deivation
    real<lower=0> sSPHat1;  // These get conversted to Shape and Rate
    real<lower=0> mdSEpsilon;
    real<lower=0> sSEpsilon;
    real<lower=0> mdSEta;
    real<lower=0> sSEta;
    real muLogisticPriorPChange;
    real<lower=0> sLogisticPriorPChange;
}

transformed data {
    real<lower=0> raSPHat1;
    real<lower=0> shSPHat1;
    real<lower=0> raSEpsilon;
    real<lower=0> shSEpsilon; 
    real<lower=0> raSEta;
    real<lower=0> shSEta;    
    raSPHat1 = gamMdStd2Rate(mdSPHat1, sSPHat1);
    shSPHat1 = gamMdStd2Shape(mdSPHat1, sSPHat1);
    raSEpsilon = gamMdStd2Rate(mdSEpsilon, sSEpsilon);
    shSEpsilon = gamMdStd2Shape(mdSEpsilon, sSEpsilon);
    raSEta = gamMdStd2Rate(mdSEta, sSEta);
    shSEta = gamMdStd2Shape(mdSEta, sSEta);
}


parameters {
    real A1;
    real B1;
    real<lower=0> sEpsilon;
    real<lower=0> sEta;
    real<lower=0> sPHat1;
    real logisticPriorPChange;
    real x[N];
    real pHat[N];
}

transformed parameters {
    real<lower=0, upper=1> priorPChange = inv_logit(logisticPriorPChange);
    real lPriorPChange = log(priorPChange);
    real lPriorPNoChange = log(1 - priorPChange);
    
    real<lower=0, upper=1> A = inv_logit(A1);
    real<lower=0, upper=1> B = inv_logit(B1);
}


model {
    real pHatMu[N];
    real sPHat[N];
    real predErr[N];

    A1 ~ normal(muA1, sA1);
    B1 ~ normal(muB1, sB1);
    sPHat1 ~ gamma(shSPHat1, raSPHat1);
    sPHat[1] = sSPHat1;
    pHatMu[1] = 0;
    pHatMu[2] = 0;
    sEpsilon ~ gamma(shSEpsilon, raSEpsilon);
    sEta ~ gamma(shSEta, raSEta);
    logisticPriorPChange ~ normal(muLogisticPriorPChange, sLogisticPriorPChange);
    
    pHat[1] ~ normal(pHatMu[1], sPHat[1]);
    pHat[2] ~ normal(pHatMu[1], sPHat[1]);
    
    x[1] ~ normal(0, sEta);
    y[1] ~ normal(x[1] + pHat[1] - p[1], sEpsilon);
    predErr[1] = x[1] + p[1]; 
    x[2] ~ normal(A*x[1] - B*predErr[1], sEta);

    for (t in 2:N) {
        predErr[t] = x[t] + p[t];
        y[t] ~ normal(x[t] + pHat[t] - p[t], sEpsilon);
        
        sPHat[t] = sPUpdate(sPHat[t-1], sEpsilon, y[t], sBroad, priorWeight, 
                        lPriorPChange, lPriorPNoChange, underWeight);
        if (t < N) {
            pHatMu[t+1] = pHatUpdate(pHat[t], y[t], sPHat[t], sEpsilon);
            pHat[t+1] ~ normal(pHatMu[t+1], sPHat[t]);
            x[t+1] ~ normal(A*x[t] - B*predErr[t], sEta);
        }
    }
}



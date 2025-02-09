functions {
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
    real muAf1; // Logistic space
    real<lower=0> sAf1; 
    real muBf1; // Logistic space
    real<lower=0> sBf1;
    real<lower=0> mdSEpsilon;
    real<lower=0> sSEpsilon;
    real<lower=0> mdSEta;
    real<lower=0> sSEta;
    real<lower=0> mdSEtaF;
    real<lower=0> sSEtaF;
}

transformed data {
    real<lower=0> raSEpsilon;
    real<lower=0> shSEpsilon; 
    real<lower=0> raSEta;
    real<lower=0> shSEta;    
    real<lower=0> raSEtaF;
    real<lower=0> shSEtaF;    
    raSEpsilon = gamMdStd2Rate(mdSEpsilon, sSEpsilon);
    shSEpsilon = gamMdStd2Shape(mdSEpsilon, sSEpsilon);
    raSEta = gamMdStd2Rate(mdSEta, sSEta);
    shSEta = gamMdStd2Shape(mdSEta, sSEta);
    raSEtaF = gamMdStd2Rate(mdSEtaF, sSEtaF);
    shSEtaF = gamMdStd2Shape(mdSEtaF, sSEtaF);
}


parameters {
    real A1;
    real B1;
    real Af1;
    real Bf1;
    real<lower=0> sEpsilon;
    real<lower=0> sEta;
    real<lower=0> sEtaF;
    real x[N];
    real x_f[N];
}

transformed parameters {
    real<lower=0, upper=1> A = inv_logit(A1);
    real<lower=0, upper=1> B = inv_logit(B1);
    real<lower=0, upper=1> Af = inv_logit(Af1);
    real<lower=0, upper=1> Bf = inv_logit(Bf1);
}


model {
    A1 ~ normal(muA1, sA1);
    B1 ~ normal(muB1, sB1);
    Af1 ~ normal(muAf1, sAf1);
    Bf1 ~ normal(muBf1, sBf1);
    
    sEpsilon ~ gamma(shSEpsilon, raSEpsilon);
    sEta ~ gamma(shSEta, raSEta);
    sEtaF ~ gamma(shSEtaF, raSEtaF);
        
    x[1] ~ normal(0, sEta);
    x_f[1] ~ normal(0, sEtaF);
    y[1] ~ normal(x[1] + x_f[1] + p[1], sEpsilon);
    x[2] ~ normal(A*x[1] - B*y[1], sEta);
    x_f[2] ~ normal(Af*x[1] - Bf*y[1], sEta);
    for (t in 2:N) {
        y[t] ~ normal(x[t] + x_f[t] + p[t], sEpsilon);        
        if (t < N) {
            x[t+1] ~ normal(A*x[t] - B*y[t], sEta);
            x_f[t+1] ~ normal(A*x_f[t] - B*y[t], sEtaF);
        }
    }
}



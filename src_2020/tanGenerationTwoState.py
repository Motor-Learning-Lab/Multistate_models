#%% Imports
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
import pickle

#%% Data and parameters
# Data
p = np.concatenate((np.zeros(80), np.full(150,60)))
N = len(p);

A = 0.98
B = 0.2

# Parameters
sEta = 0.4
sPHat1 = 0.4
sEpsilon = 3
sBroad = 45
underWeight = 0.4
priorPChange = 0.05
priorWeight = 500

lPriorPChange = np.log(priorPChange)
lPriorPNoChange = np.log(1-priorPChange)


#%% Functions to update sPHat and pHat

# The math for this equation is worked out here: https://www.mathcha.io/editor/BJP5VuBDtnqu5k3joeu5kJ89XskBnokJTz6wQnm
def pChangeFn(lP1, lP2, underWeight=underWeight, priorPChange=priorPChange):
    lPriorPChange = np.log(priorPChange)
    lPriorPNoChange = np.log(1-priorPChange)
    logTerm = underWeight*(lP1-lP2) + lPriorPNoChange-lPriorPChange
    return 1 / (1 + np.exp(logTerm))


def sPUpdate(sPHat, sEpsilon, y, sBroad, underWeight=underWeight, priorPChange=priorPChange, priorWeight=priorWeight):
    lPPHat = norm.logpdf(y, 0, np.sqrt(sPHat**2+sEpsilon**2))
    lPBroad = norm.logpdf(y, 0, np.sqrt(sBroad**2+sEpsilon**2))
    pChange = pChangeFn(lPPHat, lPBroad, underWeight=underWeight, priorPChange=priorPChange)
    
    sPPosterior = np.sqrt( (priorWeight+1) / (1/sEpsilon**2 + priorWeight/sPHat**2) )
    return pChange*sBroad + (1-pChange)*sPPosterior;


def pHatUpdate(pHat, y, sPHat, sEpsilon):
    precPHat = 1/sPHat**2
    precY = 1/sEpsilon**2
    return pHat - y*precY / (precY + precPHat)

#%% Initialization
x = np.zeros(N)
y = np.zeros(N)
epsilon = np.zeros(N)
sPHat = np.zeros(N)
pHat = np.zeros(N)
predErr = np.zeros(N)
eta = np.zeros(N)

#%% Data generation
sPHat[0] = sPHat1
pHat[0] = norm.rvs(0,sPHat[0])

x[0] = norm.rvs(0, sEta)
epsilon[0] = norm.rvs(0, sEpsilon)
predErr[0] = pHat[0] - p[0] + epsilon[0]
y[0] = x[0] + predErr[0]
eta[0] = norm.rvs(0, sEta)

for t in range(1,N):
    epsilon[t] = norm.rvs(0, sEpsilon)
    predErr[t] = pHat[t] - p[t] + epsilon[t];
    y[t] = x[t] + predErr[t]; 

    sPHat[t] = sPUpdate(sPHat[t-1], sEpsilon, y[t], sBroad);     
    eta[t] = norm.rvs(0, sEta);
    if t < N-1:
        pHat[t+1] = pHatUpdate(pHat[t], y[t], sPHat[t], sEpsilon)
        x[t+1] = A*x[t] - B*y[t] + eta[t]

#%% Plot data
t = range(0,N)
plt.figure()
plt.plot(t, p, label='p')
plt.plot(t, y, label='y')
plt.plot(t, pHat, label='\hat{p}')
plt.plot(t, x, label='x')
plt.legend()

#%% Save data
generated_data = {'y': y, 'p': p, 
                  'priorWeight': priorWeight, 'underWeight': underWeight, 'sBroad': sBroad,
                  'A': A, 'B': B, 
                  'sEta': sEta, 'sEpsilon': sEpsilon, 'priorPChange': priorPChange}
f = open('tanDataSim.pkl','wb')
pickle.dump(generated_data, f)
f.close()

#%% Generate effects of parameters of pHat and sPHat
y_vals = np.linspace(0, sBroad/2, num=50)
sPHatY = list(map(lambda y : sPUpdate(sPHat1, sEpsilon, y, sBroad), y_vals))
pHatY = list(map(lambda y, sPHat : pHatUpdate(sPHat1, y, sPHat, sEpsilon), y_vals, sPHatY))

plt.figure()
plt.plot(y_vals, pHatY, label='pHat')
plt.plot(y_vals, sPHatY, label='sPHat')


#%% Loop over effects for underweight
y_vals = np.linspace(0, sBroad, num=50)
underweight_vals = np.linspace(0, 0.7, num=10)

parameter_text =  """sEpsilon = {:.2f}
sBroad = {:.1f}
sPHat[t] = {:.2f}
priorPChange = {:.4f}
priorWeight = {:.0f}""".format(
sEpsilon, sBroad, sPHat1, priorPChange, priorWeight
)

f, axs = plt.subplots(5,2, sharex='all', sharey='all', figsize=(10,10))

for i, u in enumerate(underweight_vals):
    sPHatY = list(map(lambda y : sPUpdate(sPHat1, sEpsilon, y, sBroad, underWeight=u), y_vals))
    pHatY = list(map(lambda y, sPHat : pHatUpdate(sPHat1, y, sPHat, sEpsilon), y_vals, sPHatY))
    
    ax = axs.flatten()[i]
    # ax.plot(y_vals, pHatY, label='pHat')
    ax.plot(y_vals, sPHatY, label='sPHat')
    ax.set_title("Underweight: {:.2f}".format(u))

# axs[4,1].legend()
axs[4,0].xaxis.set_label_text('y[t]')
axs[4,0].yaxis.set_label_text('sPHat[t+1]')
                    
axs[4,0].text(0, 40, parameter_text, verticalalignment='top')

f.tight_layout()

#%% Loop over effects for priorPChange
y_vals = np.linspace(0, sBroad/2, num=50)
priorPChange_vals = np.linspace(0, 0.1, num=10)

parameter_text =  """sEpsilon = {:.2f}
sBroad = {:.2f}
sPHat[t] = {:.2f}
underWeight = {:.2f}
priorWeight = {:.2f}""".format(
sEpsilon, sBroad, sPHat1, underWeight, priorWeight
)

f, axs = plt.subplots(5,2, sharex='all', sharey='all', figsize=(10,10))

for i, p in enumerate(priorPChange_vals):
    sPHatY = list(map(lambda y : sPUpdate(sPHat1, sEpsilon, y, sBroad, priorPChange=p), y_vals))
    
    ax = axs.flatten()[i]
    # ax.plot(y_vals, pHatY, label='pHat')
    ax.plot(y_vals, sPHatY, label='sPHat')
    ax.set_title("priorPChange: {:.3f}".format(p))

# axs[4,1].legend()
axs[4,0].xaxis.set_label_text('y[t]')
axs[4,0].yaxis.set_label_text('sPHat[t+1]')
                    
axs[4,0].text(0, 40, parameter_text, verticalalignment='top')

f.tight_layout()

#%% Imports
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
import pickle
import sys

#%% Data and parameters
# Data
p = np.concatenate((np.zeros(80), np.full(150, 60), np.zeros(80)))
# p = np.zeros(310)
# p_choices = [-60, -40, -20, 0, 20, 40, 60]
# p = np.concatenate((np.random.choice(p_choices, 80), np.full(150, 60), np.zeros(80)))

N = len(p)

# Parameters
A = 0.99
B = 0.05
Af = 0.9
Bf = 0.3

sEta = 0.4
sEtaF = 0.4
sEpsilon = 3


#%% Initialization
x = np.zeros(N)
x_f = np.zeros(N)
y = np.zeros(N)

eta = np.zeros(N)
eta_f = np.zeros(N)
epsilon = np.zeros(N)

#%% Data generation

x[0] = norm.rvs(0, sEta)
x_f[0] = norm.rvs(0,sEtaF)
epsilon[0] = norm.rvs(0, sEpsilon)
y[0] = x[0] + x_f[0] + epsilon[0]
eta[0] = norm.rvs(0, sEta)
eta_f[0] = norm.rvs(0, sEtaF)

x[1] = A*x[0] + B*y[0] + eta[0]
x_f[1] = Af*x_f[0] + Bf*y[0] + eta_f[0]

for t in range(1,N):
    epsilon[t] = norm.rvs(0, sEpsilon)
    y[t] = x[t] + x_f[t] + p[t] + epsilon[t] 

    eta[t] = norm.rvs(0, sEta)
    eta_f[t] = norm.rvs(0, sEtaF)
    if t < N-1:
        x[t+1] = A*x[t] - B*y[t] + eta[t]
        x_f[t+1] = Af*x_f[t] - Bf*y[t] + eta_f[t] 

#%% Plot data
t = range(0,N)
f, (ax1) = plt.subplots(1,1, figsize=(8, 8))
ax1.plot(t, p, label=r'$p$')
ax1.plot(t, y, label=r'$y$')
ax1.plot(t, -x_f, label=r'$-x_f$')
ax1.plot(t, -x, label=r'$-x$')
ax1.plot(t, -x-x_f, label=r'$-x-x_f$')
ax1.legend()
ax1.yaxis.set_label_text('Degrees')

f.tight_layout()

#%% Save data
generated_data = {'y': y, 'p': p, 'x': x, 'x_f': x_f, 
                  'A': A, 'B': B, 'Af': Af, 'Bf': Bf, 
                  'sEta': sEta, 'sEtaF': sEtaF, 'sEpsilon': sEpsilon, 
                  }
f = open('smithDataSim.pkl','wb')
pickle.dump(generated_data, f)
f.close()

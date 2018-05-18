
import getpass as gp
name = gp.getuser()

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import sys
sys.path.append('/Users/%s/OneDrive/Master Thesis/Data/Analysis_Skripts/Library/' %name)
from Functions import *
from sklearn.covariance import ledoit_wolf as LW, oas, shrunk_covariance

freq = 'M'
years = 5

returns, rf_rate, market, estLength, nAssets = get_Data(freq, years) 


#Index dates for dataframe
datesPF = returns.index.values[(estLength-1):(len(returns.index)-1)] 

#Index dates for dataframe
datesImpl = returns.index.values[(estLength):(len(returns.index))]

 #Column names for dataframe
indices = returns.columns.values 

#initialize monthly return matrix with each asset after implementation
retAssets = np.empty((len(returns.index)-estLength,nAssets)) 

#initialize monthly return vector of Portfolio after implementation
retPF = np.empty((len(returns.index)-estLength,1)) 

#initialize matrix for Garlappi Wang portfolio weights
PFRPdyn = np.empty((len(returns.index)-estLength,nAssets)) 

#initialize historical return vector for month after implementation of strategy
histRet = np.empty((1,nAssets))





'''Calculate the return of a Risk Parity Portfolio'''
from scipy.optimize import minimize


'''set constraints for optimization'''
cons = ({'type' : 'eq', 'fun' : weight_constraint },
        {'type': 'ineq', 'fun': long_only_constraint})


'''set input parameters for optimization'''
x_t = np.ones(nAssets) / nAssets #equal risk contribution target vector
w_0 = rand_weights(nAssets) #initial weights from which to start opimization
    

for n in range(0,(len(returns.index)-estLength)):
    '''loop in order to calculate the risk parity portfolio in each period'''
    df_estimation = returns[n:estLength+n]
    varCov = varCovar(df_estimation) #variance covariance matrix
    res = minimize(risk_objective,
               w_0,
               args = [varCov, x_t],
               method = 'SLSQP',
               constraints = cons,
               options = {'ftol': 1e-12, 'maxiter' : 45, 'disp' : False})
    w_RP = np.array(res.x)
    PFRPdyn[n,:] = w_RP.T
    histRet = returns.iloc[(estLength + n)] 
    retAssets[n,:] = w_RP.T * (np.exp(histRet)-1)
    retPF[n,:] = retAssets[n,:].sum()
    
df_retPF = pd.DataFrame(retPF, index = datesImpl)    
RPPortfolios = pd.DataFrame(PFRPdyn, index = datesPF, columns = indices) 

os.chdir('/Users/%s/OneDrive/Master Thesis/Data/Portfolios/' %name)
RPPortfolios.to_csv('RPPortfolios%s.csv' %freq)










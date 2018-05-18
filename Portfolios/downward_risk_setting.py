
import getpass as gp
name = gp.getuser()

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import sys
sys.path.append('/Users/%s/OneDrive/Master Thesis/Data/Analysis_Skripts/Library/' %name)
from Functions import *
from scipy.optimize import minimize


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
retPF_LPM = np.empty((len(returns.index)-estLength,1)) 
#initialize matrix for Garlappi Wang portfolio weights
LPM_pfs_dyn = np.empty((len(returns.index)-estLength,nAssets)) 
#initialize historical return vector for month after implementation of strategy
histRet = np.empty((1,nAssets))



global exp_ret_chosen_LPM
global exprets_LPM
exp_ret_chosen_LPM = 0.02/12 

for n in range(0,(len(returns.index)-estLength)):
    '''loop in order to calculate the risk parity portfolio in each period'''
    df_estimation = returns[n:estLength+n]
    varCov = varCovariance(df_estimation) #variance covariance matrix
    
    exprets = df_estimation.mean()
    w0 = np.random.rand(nAssets)
    cons = ({'type' : 'eq', 'fun' : weight_constraint},
                    {'type' : 'eq', 'fun' : expected_return_constraint_test})
    
    optimization = minimize(LPM_PF_optimization, 
                            w0, 
                            args = [exp_ret_chosen, returns],
                            method = 'SLSQP', 
                            constraints = cons,
                            options={'ftol': 1e-12, 'maxiter' : 45, 'disp' : False})
    
    PF_weights_LPM = np.array(optimization.x)
    LPM_pfs_dyn[n,:] = PF_weights_LPM.T
    histRet_out_of_sample = returns.iloc[(estLength + n)] 
    retAssets[n,:] = w_RP.T * (np.exp(histRet_out_of_sample)-1)
    retPF_LPM[n,:] = retAssets[n,:].sum()
    
df_retPF = pd.DataFrame(retPF_LPM, index = datesImpl)    
Portfolios_weights = pd.DataFrame(LPM_pfs_dyn, index = datesPF, columns = indices)     
Portfolios_weights.to_csv('Portfolios_downward_risk_setting_%s.csv' %freq)


global exp_ret_chosen_LPM
global exprets_LPM
exp_ret_chosen_LPM = 0.02 / 12
exprets_LPM = np.array(returns.mean())
w0 = np.random.rand(nAssets)
cons = ({'type' : 'eq', 'fun' : weight_constraint},
                {'type' : 'eq', 'fun' : expected_return_constraint_LPM})
def expected_return_constraint_LPM(x):
    if len(x.shape)==1:
        x = np.asmatrix(x).T
    constraint = np.dot(x.T, exprets_LPM) - exp_ret_chosen_LPM
    return float(constraint)
def LPM_PF_optimization(x, args):

    returns = args[1]
    nAssets = returns.shape[1]
    L_matrix = np.zeros((nAssets, nAssets),float)
    '''create the LPM for comovements in several assets to add to the rest of the matrix'''
    for i in range(nAssets):
        for j in range(nAssets):
            if isinstance(returns, pd.DataFrame):
                L_matrix[i,j] = CO_LowerPartialMoments(returns[indices[i]],returns[indices[j]])
            elif isinstance(returns, np.ndarray):
                L_matrix[i,j] = CO_LowerPartialMoments(returns[:,i],returns[:,j])
    
    return PF_variance(x, L_matrix)
test = expected_return_constraint_LPM(np.random.rand(nAssets))
optimization = minimize(LPM_PF_optimization, 
                        w0, 
                        args = [exp_ret_chosen_LPM, returns],
                        method = 'SLSQP', 
                        constraints = cons,
                        options={'ftol': 1e-12, 'maxiter' : 400, 'disp' : True})

weights_optimization = np.asmatrix(optimization.x).T


# WITH CONSTRAINTS OF NO MORE THAN LEVERAGING / SHORTING 2 TIMES

global exp_ret_chosen
global exprets
exp_ret_chosen = 0.02 / 12
exprets = returns.mean()
w0 = np.random.rand(nAssets)
cons = ({'type' : 'eq', 'fun' : weight_constraint},
                {'type' : 'eq', 'fun' : expected_return_constraint_test})
bnds = ((-2, 2),(-2, 2),(-2, 2),(-2, 2),(-2, 2),(-2, 2),(-2, 2),(-2, 2),)

optimization_no_leverage = minimize(LPM_PF_optimization, 
                        w0, 
                        args = [exp_ret_chosen, returns],
                        method = 'SLSQP', 
                        bounds = bnds,
                        constraints = cons,
                        options={'ftol': 1e-12, 'maxiter' : 400, 'disp' : True})

w_opt_No_Double_Weights = np.array(optimization_no_leverage.x)

###############################################################################



CLASSIC_matrix = varCovariance(returns)
OAS_matrix = cov_robust(returns)

L_matrix_1 = LPM_matrix(returns)
L_matrix_2 = LPM_matrix(returns, target = 0.05)
L_matrix_3 = LPM_matrix(returns, target = 0.08)

sigma_CLASSIC = st_dev_MV_pf(exprets, CLASSIC_matrix)
sigma_OAS = st_dev_MV_pf(exprets, OAS_matrix)

sigma_LPM_1 = st_dev_MV_pf(exprets, L_matrix_1)
sigma_LPM_2 = st_dev_MV_pf(exprets, L_matrix_2)
sigma_LPM_3 = st_dev_MV_pf(exprets, L_matrix_3)

plt.figure(figsize=(10,8))
plt.plot(sigma_LPM_1, x, label='LPM, target = 0%')
plt.plot(sigma_LPM_2, x, label='LPM, target = 5%')
plt.plot(sigma_LPM_3, x, label='LPM, target = 8%')
plt.legend()
plt.savefig('comparizon_between_frontiers.svg')
plt.show()





L_matrix_1 = LPM_matrix(returns, target = 0.05)
L_matrix_approx = CO_LowerPartialMoments_approximation(returns, target = 0.05)

L_matrix_1_two = LPM_matrix(returns, target = 0.02)
L_matrix_approx_two = CO_LowerPartialMoments_approximation(returns, target = 0.02)

L_matrix_1_zero = LPM_matrix(returns)
L_matrix_approx_zero = CO_LowerPartialMoments_approximation(returns)


sigma_LPM_1 = st_dev_MV_pf(exprets, L_matrix_1)
sigma_LPM_approx = st_dev_MV_pf(exprets, L_matrix_approx)

sigma_LPM_1_two = st_dev_MV_pf(exprets, L_matrix_1_two)
sigma_LPM_approx_two = st_dev_MV_pf(exprets, L_matrix_approx_two)

sigma_LPM_1_zero = st_dev_MV_pf(exprets, L_matrix_1_zero)
sigma_LPM_approx_zero = st_dev_MV_pf(exprets, L_matrix_approx_zero)


plt.figure(figsize=(10,8))
plt.plot(sigma_LPM_1, x, label='LPM, target = 5%%')
plt.plot(sigma_LPM_approx, x, label='LPM approximation, target = 5%')

plt.plot(sigma_LPM_1_two, x, label='LPM, target = 2%%')
plt.plot(sigma_LPM_approx_two, x, label='LPM approximation, target = 2%')

plt.plot(sigma_LPM_1_zero, x, label='LPM, target = 0%%')
plt.plot(sigma_LPM_approx_zero, x, label='LPM approximation, target = 0%')

plt.legend()
plt.savefig('comparizon_between_approximation_versions.svg')
plt.show()





#
#
#plt.figure(figsize = (10, 10))
#plt.plot(sigma_CLASSIC, x, label='Classic MV')
#plt.plot(sigma_OAS, x, label='OAS estimator')
#plt.plot(sigma_LPM, x, label='LPM estimator')
#plt.legend()
#plt.savefig('comparizon_between_frontiers.svg')
#plt.show()
#    
#
#
#
#
#'''Calculate characteristics of the tangency portfolio'''
#TrueMeanRet = float(np.dot(trueWeightsSR.T, np.array(meanRet)))
#TrueVolPF = np.sqrt((varC * TrueMeanRet ** 2 - 2 * varB * TrueMeanRet + varA) / varD)
#
#
#'''Calculate Capital Market Line'''
#slope =(TrueMeanRet - rf * multiplier) / (TrueVolPF)
#tangencyLineX = np.array([0, TrueVolPF, 3 * TrueVolPF]) 
#tangencyLineY = np.array([rf * multiplier, TrueMeanRet, 3 * TrueVolPF * slope + rf * multiplier])
#
#
#
#'''Plot All Data'''
#plt.figure(figsize = (10,5))
#plt.ylim(0., 0.015)
#plt.xlim(0.0, 0.008)
#plt.scatter(TrueMuSigmaMatrixSR[:,1], TrueMuSigmaMatrixSR[:,0], s = 7, c = 'teal')
#plt.scatter(FalseMuSigmaMatrixSR[:,1], FalseMuSigmaMatrixSR[:,0], s = 7, c = 'orange')
#plt.scatter(TrueVolPF, TrueMeanRet, s = 30)
#plt.scatter(0, rf * multiplier, s = 30)
#plt.plot(sigma, x, c = 'red')
#plt.plot(tangencyLineX, tangencyLineY, c = 'grey')
##plt.savefig('diego_scatter_truefalse%s.svg'%MCSize)
##plt.savefig('diego_scatter_truefalse%s.png'%MCSize, dpi=500)
#
#
#
#
#
#

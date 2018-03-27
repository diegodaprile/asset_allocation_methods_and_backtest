# -*- coding: utf-8 -*-
"""
Created on Tue Mar  6 15:47:24 2018
"""

import getpass as gp
name = gp.getuser()

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import sys
sys.path.append('/path_to_append/' %name)

os.chdir("/Users/%s/OneDrive/Master Thesis/Data" %name)
from Functions import *



################################ Data Import ##################################



xl = pd.ExcelFile('30_Industry_Portfolios.xlsx')
returns = xl.parse('Returns_M', index_col = 'Code')

'''True Parameters of the whole dataset (30 Assets)'''
meanRet = np.asmatrix(np.mean(returns, axis = 0)).T
varCovar = np.asmatrix(np.cov(returns.T, ddof = 1))


'''Define number of draws and Seed in Monte Carlo'''
MCSize = 10000
np.random.seed(110693)   


'''Define Parameters for estimation'''
rf_annual = 0.04       # risk free rate
rf = rf_annual / 12

gamma = 1       # risk aversion parameter
epsilon = 1     # ambiguity aversion parameter


nAssets = 5     # number of assets to be considered
   
estLength = 60  # length of estimation period


'''True Parameters for 1 <= n <= 30 Assets'''
meanRet = meanRet[:nAssets]
varCovar = varCovar[:nAssets,:nAssets]


choleskyMat = np.linalg.cholesky(varCovar)      # Cholesky Matrix of varCovar

'''Calculate the true weights and true portfolio characteristics'''
truePi = tangWeights(np.array(meanRet), varCovar, rf_annual, gamma)

trueMu = float(np.dot(truePi.T, np.array(meanRet)))
trueSigma = float(np.sqrt(np.dot(truePi.T,np.dot(varCovar, truePi))))
trueUtility = utility_MV(trueMu, trueSigma, gamma)


'''example of list for saving the utility of method "i" '''
#utility_i = []
#
#counter = []
#for n in range(MCSize):
#    '''assign random values to random value matrix'''
#    randomMat = np.random.normal(0.,1.,(estLength, nAssets))
#    '''induce correlation to the random values by multiplying those with the Cholesky Decomposition'''
#    corrRandomMat = np.dot(randomMat, choleskyMat.T) 
#    '''simulate correlated returns over 60 months for nAssets'''
#    corrReturns = np.array(meanRet.T) +  np.array(corrRandomMat)
#    estMu = np.asmatrix(np.mean(corrReturns, axis = 0)).T
#    estSigma = np.asmatrix(np.cov(corrReturns.T, ddof=1))
#    
#    # Insert all the asset allocation methods here and calculate utility based on the estimated weights and true parameters
#    counter.append(n)
#    
#    minvar = minVarPF_backtest(corrReturns, estSigma)
#    # method calculates the portfolio characteristics (mu_pf, sigma_pf) and utility 
#    utility_i.append(utility(meanRet, varCovar, estimatedWeights, gamma))
#    
#    maxsharpe = maxSRPF1(estMu, estSigma, rf)
#    


utility_i_min_var = []
utility_i_max_sharpe = []
utility_i_GWPF = []
utility_i_RP_pf = []
utility_i_LPM_pf = []
utility_i_over_N_pf = []
utility_i_hierarchical_pf = []
counter = []

for n in range(MCSize):
    '''assign random values to random value matrix'''
    randomMat = np.random.normal(0.,1.,(estLength, nAssets))
    '''induce correlation to the random values by multiplying those with the Cholesky Decomposition'''
    corrRandomMat = np.dot(randomMat, choleskyMat.T) 
    '''simulate correlated returns over 60 months for nAssets'''
    corrReturns = np.array(meanRet.T) +  np.array(corrRandomMat)
    estMu = np.asmatrix(np.mean(corrReturns, axis = 0)).T
    estSigma = np.asmatrix(np.cov(corrReturns.T, ddof=1))
    
    
    counter.append(n)
    # Insert all the asset allocation methods here and calculate utility based on the estimated weights and true parameters
    
    #minimum variance portfolio
    #calculate allocation according to estimated parameters in the simulation
    minvar = minVarPF1(estSigma)
    #calculate exp ret and stdev of portfolio, with true parameters, but with the weights calculated before
    meanRet_minVar = PF_return(minvar, meanRet)
    sigma_minVar = np.sqrt(PF_variance(minvar, varCovar))
    # method calculates the portfolio characteristics (mu_pf, sigma_pf) and utility 
    utility_i_min_var.append(utility_MV(meanRet_minVar, sigma_minVar, gamma))

    #maximums sharpe ratio portfolio
    #calculate allocation according to estimated parameters in the simulation
    maxsharpe = maxSRPF1(estMu, estSigma, rf)
    #calculate exp ret and stdev of portfolio, with true parameters, but with the weights calculated before
    meanRet_maxsharpe = PF_return(maxsharpe, meanRet)
    sigma_maxsharpe = np.sqrt(PF_variance(maxsharpe, varCovar))
    # method calculates the portfolio characteristics (mu_pf, sigma_pf) and utility 
    utility_i_max_sharpe.append(utility_MV(meanRet_maxsharpe, sigma_maxsharpe, gamma))
#    maxsharpe = maxSRPF1(estMu, estSigma, rf)
#    utility_i_max_sharpe.append(utility(meanRet_minVar, varCovar_minVar, minvar, gamma))

    #garlappi wang portfolio
    #calculate allocation according to estimated parameters in the simulation
    GWPF = GWweights1(corrReturns, estMu, estSigma, epsilon, gamma)
    #calculate exp ret and stdev of portfolio, with true parameters, but with the weights calculated before
    meanRet_GWPF = PF_return(GWPF, meanRet)
    sigma_GWPF = np.sqrt(PF_variance(GWPF, varCovar))
    # method calculates the portfolio characteristics (mu_pf, sigma_pf) and utility 
    utility_i_GWPF.append(utility_MV(meanRet_GWPF, sigma_GWPF, gamma))

#    
#    # Risk Parity Portfolio
#    #calculate allocation according to estimated parameters in the simulation
#    RP_pf = riskParity(estSigma)
#    #calculate exp ret and stdev of portfolio, with true parameters, but with the weights calculated before
#    meanRet_RP_pf = PF_return(RP_pf, meanRet)
#    sigma_RP_pf = np.sqrt(PF_variance(RP_pf, varCovar))
#    # method calculates the portfolio characteristics (mu_pf, sigma_pf) and utility 
#    utility_i_RP_pf.append(utility_MV(meanRet_RP_pf, sigma_RP_pf, gamma))
#    
#    
#    #lower partial moments portfolio
#    LPM_pf = lpm_port(estMu, corrReturns)
#    meanRet_LPM_pf = PF_return(LPM_pf, meanRet)
#    sigma_LPM_pf = np.sqrt(PF_variance(LPM_pf, varCovar))
#    # method calculates the portfolio characteristics (mu_pf, sigma_pf) and utility 
#    utility_i_LPM_pf.append(utility_MV(meanRet_LPM_pf, sigma_LPM_pf, gamma))
    
    # 1 over N
    over_N_pf = np.asmatrix([1 / nAssets for i in range(nAssets)]).T
    #calculate exp ret and stdev of portfolio, with true parameters, but with the weights calculated before
    meanRet_over_N_pf = PF_return(over_N_pf, meanRet)
    sigma_over_N_pf = np.sqrt(PF_variance(over_N_pf, varCovar))
    # method calculates the portfolio characteristics (mu_pf, sigma_pf) and utility 
    utility_i_over_N_pf.append(utility_MV(meanRet_over_N_pf, sigma_over_N_pf, gamma))
    

    # hierarchical clustering portfolio
    #calculate allocation according to estimated parameters in the simulation
    hierarchical_pf = getHRP(estSigma)
    #calculate exp ret and stdev of portfolio, with true parameters, but with the weights calculated before
    meanRet_hierarchical_pf = PF_return(hierarchical_pf, meanRet)
    sigma_hierarchical_pf = np.sqrt(PF_variance(hierarchical_pf, varCovar))
    # method calculates the portfolio characteristics (mu_pf, sigma_pf) and utility 
    utility_i_hierarchical_pf.append(utility_MV(meanRet_hierarchical_pf, sigma_hierarchical_pf, gamma))
    
    
    
# calculate expected utility loss for asset allocation : minimum variance
expected_utility_loss__minvar = trueUtility - np.mean(winsorize(utility_i_min_var, 0.05))

# calculate expected utility loss for asset allocation : maximum sharpe ratio
expected_utility_loss_maxsharpe = trueUtility - np.mean(winsorize(utility_i_max_sharpe, 0.05))   

# calculate expected utility loss for asset allocation : Garlappi wang portfolio
expected_utility_loss_GWPF = trueUtility - np.mean(winsorize(utility_i_GWPF, 0.05))   

## calculate expected utility loss for asset allocation : Risk Parity Portfolio
#expected_utility_loss_RP_pf = trueUtility - np.mean(winsorize(utility_i_RP_pf, 0.05))   
#
## calculate expected utility loss for asset allocation : Lower Partial Moments Portfolio
#expected_utility_loss_LPM_pf = trueUtility - np.mean(winsorize(utility_i_LPM_pf, 0.05))   

# calculate expected utility loss for asset allocation : 1 onver N portfolio
expected_utility_loss_over_N_pf = trueUtility - np.mean(winsorize(utility_i_over_N_pf, 0.05))   

# calculate expected utility loss for asset allocation : hierarchical clustering
expected_utility_loss_hierarchical_pf = trueUtility - np.mean(winsorize(utility_i_hierarchical_pf, 0.05))   

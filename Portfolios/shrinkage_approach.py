
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
years = 10

returns, rf_rate, market, estLength, nAssets = get_Data(freq, years) # years of estimation


#Index dates for dataframe
datesPF = returns.index.values[(estLength-1):(len(returns.index)-1)] 

#Index dates for dataframe
datesImpl = returns.index.values[(estLength):(len(returns.index))]

 #Column names for dataframe
indices = returns.columns.values 

#initialize monthly return matrix with each asset after implementation
retAssetsLW = np.zeros((len(returns.index)-estLength,nAssets)) 
retAssetsOAS = np.zeros((len(returns.index)-estLength,nAssets)) 

#initialize monthly return vector of Portfolio after implementation
retPF_LW = np.zeros((len(returns.index)-estLength,1)) 
retPF_OAS = np.zeros((len(returns.index)-estLength,1)) 

#initialize matrix for Garlappi Wang portfolio weights
MPT_LW = np.zeros((len(returns.index)-estLength,nAssets)) 
MPT_OAS = np.zeros((len(returns.index)-estLength,nAssets)) 
#initialize historical return vector for month after implementation of strategy
histRet = np.zeros((1,nAssets))




for n in range(0,(len(returns.index)-estLength)):
    '''loop in order to calculate the efficient portfolios in each period'''
    df_estimation = returns[n:estLength+n]
    meanRet = df_estimation.mean()
    histRet = returns.iloc[(estLength + n)] 
    
    #identify the correct risk free rates, convert them in monthly figures, and put them in df format
    rf_ann = rf_rate.iloc[estLength + n - 1]
    if freq == "M":
        rf = (1. + rf_ann) ** (1. / 12) - 1
    elif freq == "W":
        rf = (1. + rf_ann) ** (1. / 52) - 1
    elif freq == "D":
        rf = (1. + rf_ann) ** (1. / 365) - 1

    '''Calculation of optimal portfolio based on Ledoit Wolf Shrinkage'''
    shrunkVarCovarLW = LW(df_estimation, assume_centered = True)[0]
    maxSlopePF = np.array([float(x) for x in maxSRPF_noshort(meanRet, shrunkVarCovarLW, rf)])
#    maxSlopePF =  maxSlopePF1(meanRet, shrunkVarCovarLW)
    MPT_LW[n,:] = maxSlopePF
    #problem to solve: maxSlopePF appears to be a row vector, i adapted the following line
    retAssetsLW[n,:] = np.dot(maxSlopePF, (np.exp(histRet)-1))
    retPF_LW[n,:] = retAssetsLW[n,:].sum()
    
#    '''Calculation of optimal portfolio based on OAS Shrinkage'''
#    shrunkVarCovarOAS = oas(df_estimation, assume_centered = True)[0]
#    maxSlopePF =    1 / var_B(meanRet, shrunkVarCovarOAS) * \
#                    np.dot(mat_inv(shrunkVarCovarOAS), meanRet)
#    MPT_OAS[n,:] = maxSlopePF
#    retAssetsOAS[n,:] = np.dot(maxSlopePF, (np.exp(histRet)-1))
#    retPF_OAS[n,:] = retAssetsOAS[n,:].sum()

os.chdir('/Users/%s/OneDrive/Master Thesis/Data/Portfolios/no_short_sale/' %name)

'''Reformat the data into dataframe and assign row and column indices'''
df_MPT_LW = pd.DataFrame(MPT_LW, index = datesPF, columns = indices) 
#df_MPT_OAS = pd.DataFrame(MPT_OAS, index = datesPF, columns = indices) 


df_MPT_LW.to_csv('LWPortfolios{}{}.csv'.format(freq, years))
#df_MPT_OAS.to_csv('OASPortfolios{}{}.csv'.format(freq, years))

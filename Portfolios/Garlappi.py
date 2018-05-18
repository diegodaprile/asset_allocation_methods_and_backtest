
from getpass import getuser as gu
name = gu()

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import sys
sys.path.append('/Users/%s/OneDrive/Master Thesis/Data/Analysis_Skripts/Library/' %name)
from Functions import *

from scipy.stats import trim_mean, kurtosis
from scipy.stats.mstats import mode, gmean, hmean



freq = 'M'
years = [4, 5, 6, 8, 10]

#set parameters of Garlappi Wang Model
epsilon = [0.50, 1.00, 1.50, 2.00, 3.00]
gamma_list = [0.5, 1, 1.5, 2, 3]

for y in years:    
    returns, rf_rate, market, estLength, nAssets = get_Data(freq, y)
    
    
    #Index dates for dataframe
    datesPF = returns.index.values[(estLength-1):(len(returns.index)-1)] 
    
    #Index dates for dataframe
    datesImpl = returns.index.values[(estLength):(len(returns.index))]
    
     #Column names for dataframe
    indices = returns.columns.values 
    
    #initialize monthly return matrix with each asset after implementation
    retGW = np.zeros((len(returns.index)-estLength,nAssets)) 
    
    #initialize monthly return vector of Portfolio after implementation
    retPFGW = np.zeros((len(returns.index)-estLength,1)) 
    
    #initialize matrix for Garlappi Wang portfolio weights
    GWPFdyn = np.zeros((len(returns.index)-estLength,nAssets)) 
    
    #initialize historical return vector for month after implementation of strategy
    histRet = np.zeros((1,nAssets))

    
#    GWweights_noshort(returns, meanRet, varCovar, epsilon, gamma)
    for gamma in gamma_list:
        for eps in epsilon:
            for n in range(0,(len(returns.index)-estLength)):
                '''loop in order to calculate the efficient portfolios in each period'''
                df_estimation = returns[n:estLength+n]
                meanRet = np.asmatrix(np.mean(df_estimation)).T
                varCovar = np.cov(df_estimation.T, ddof=1)
                
                #identify the correct risk free rates, convert them in monthly figures, and put them in df format
                rf_ann = np.array([float(x) for x in rf_rate[n:estLength+n].values])
                if freq == "M":
                    rf = np.array([(1. + i) ** (1. / 12) - 1 for i in rf_ann])
                elif freq == "W":
                    rf = np.array([(1. + i) ** (1. / 52) - 1 for i in rf_ann])
                elif freq == "D":
                    rf = np.array([(1. + i) ** (1. / 365) - 1 for i in rf_ann])
                #convert back to dataframe
            #    rf = pd.DataFrame(rf, index = rf_rate.iloc[n-1:estLength+n-1].index, columns = ["rf"])
            
                rets = df_estimation.values
                
                gw = GWweights1(rets, meanRet, varCovar, eps, gamma)
    #            gw = GWweights1(exrets, meanRet, varCovar, eps, gamma)
                GWPFdyn[n,:] = np.asmatrix(gw).T
            
            #format data as dataframe with dates and column names    
            GWPortfolios = pd.DataFrame(GWPFdyn, index = datesPF, columns = indices) 
            os.chdir('/Users/%s/OneDrive/Master Thesis/Data/Portfolios/' %name)
            GWPortfolios.to_csv('GUWPortfoliosM{}-gamma{:3.1f}_epsilon{:3.1f}.csv'.format(y, gamma, eps))

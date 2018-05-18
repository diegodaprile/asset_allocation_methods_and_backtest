
import getpass as gp
global name 
name = gp.getuser()

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import sys
sys.path.append('/Users/%s/OneDrive/Master Thesis/Data/Analysis_Skripts/Library/' %name)
from Functions import *


gamma_list = [0.5, 1.0, 1.5, 2.0, 3.0]

freq = 'M'
years = 10

returns, rf_rate, market, estLength, nAssets = get_Data(freq, years) # years of estimation


for gamma in gamma_list:
    
    datesPF = returns.index.values[(estLength-1):(len(returns.index)-1)] #Index dates for dataframe
    datesImpl = returns.index.values[(estLength):(len(returns.index))] #Index dates for dataframe
    indices = returns.columns.values #Column names for dataframe
    
    
    #calculate implementation of adjusting strategy over time
    oneVector = np.asmatrix(np.ones(nAssets)).T #creates a one vector for portfolio calculation
    retAM = np.zeros((len(returns.index)-estLength,nAssets)) #monthly return of each asset after implementation
    retPFM = np.zeros((len(returns.index)-estLength,1)) #monthly return of Portfolio after implementation
    
    one_fund_mean_var_PFdyn = np.zeros((len(returns.index)-estLength,nAssets)) #initializes matrix for max slope portfolio
    histRet = np.zeros((1,nAssets))
    
    for n in range(0,(len(returns.index)-estLength)):
        
        '''loop in order to calculate the efficient portfolios in each period'''
        df_estimation = returns[n:estLength+n]
        '''extract the estimation data from the dataset'''
        meanRet = np.array(np.mean(df_estimation))
        '''calculate mean returns of the estimation dataset'''
        estSigma = np.cov(df_estimation.T)
    
        '''calculate the max slope portfolio and assign the output vector to the matrix of portfolio weights'''
    
        one_fund = meanVarPF_one_fund_noshort(meanRet, estSigma, gamma)
        one_fund_mean_var_PFdyn[n,:] = one_fund.T       
    
    
    MSPortfolios_one_fund = pd.DataFrame(one_fund_mean_var_PFdyn, index = datesPF, columns = indices) #format data as dataframe with dates and column names
    
    checkoneMS = MSPortfolios_one_fund.sum(axis=1) #check if weights sum to 1
    
    
    
    os.chdir("/Users/%s/OneDrive/Master Thesis/Data/Portfolios/no_short_sale/" %name)
    MSPortfolios_one_fund.to_csv('MeanVar_1F_PFM{}-gamma{:3.1f}.csv'.format(years, gamma))
    


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


freq = 'M'
gamma_list = [0.5, 1.0, 1.5, 2.0, 3.0]
year = 10


for gamma in gamma_list:
    returns, rf_rate, market, estLength, nAssets = get_Data(freq, year) # years of estimation

    datesPF = returns.index.values[(estLength-1):(len(returns.index)-1)] #Index dates for dataframe
    datesImpl = returns.index.values[(estLength):(len(returns.index))] #Index dates for dataframe
    indices = returns.columns.values #Column names for dataframe
    
    #calculate implementation of adjusting strategy over time
    
    oneVector = np.asmatrix(np.ones(nAssets)).T #creates a one vector for portfolio calculation
    retAM = np.zeros((len(returns.index)-estLength,nAssets)) #monthly return of each asset after implementation
    retPFM = np.zeros((len(returns.index)-estLength,1)) #monthly return of Portfolio after implementation
    
    histRet = np.zeros((1,nAssets))
    
    ourGarlappi_PFdyn = np.zeros((len(returns.index)-estLength,nAssets)) #initializes matrix for min var portfolio
    
    
    limit = (len(returns.index)-estLength)
    
    for n in range(0,limit):
        df_estimation = returns[n:estLength+n]
        '''extract the estimation data from the dataset'''
        meanRet = np.asmatrix(np.mean(df_estimation)).T
        '''calculate mean returns of the estimation dataset'''
        estSigma = np.cov(df_estimation.T)
        
        market_estimation = market[n:estLength+n]
        
        
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
        
        ''' NOTE: the portfolios sum to one '''
        varyingEpsilon = varying_epsilon(rets, market_estimation.values, rf)
        ourGarlappi = GWweights_noshort(rets, meanRet, estSigma, varyingEpsilon, gamma)
    
        if ourGarlappi.sum() == 10 * len(meanRet):
            ourGarlappi_PFdyn[n,:] = ourGarlappi_PFdyn[n-1,:]
        else:
            ourGarlappi_PFdyn[n,:] = np.array([float(x) for x in ourGarlappi])


    ourGarlappiPortfolios = pd.DataFrame(ourGarlappi_PFdyn, index = datesPF, columns = indices)
    #checkoneourGarlappiPortfolios = ourGarlappiPortfolios.sum(axis=1) #check if weights sum to 1
    os.chdir("/Users/%s/OneDrive/Master Thesis/Data/Portfolios/no_short_sale/" %name)
    ourGarlappiPortfolios.to_csv('ourGarlPortfoM{}-gamma{:3.1f}.csv'.format(year, gamma))
        
        

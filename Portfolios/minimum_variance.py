
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
years = 8

returns, rf_rate, market, estLength, nAssets = get_Data(freq, years) # years of estimation



datesPF = returns.index.values[(estLength-1):(len(returns.index)-1)] #Index dates for dataframe
datesImpl = returns.index.values[(estLength):(len(returns.index))] #Index dates for dataframe
indices = returns.columns.values #Column names for dataframe

#calculate implementation of adjusting strategy over time

oneVector = np.asmatrix(np.ones(nAssets)).T #creates a one vector for portfolio calculation
retAM = np.zeros((len(returns.index)-estLength,nAssets)) #monthly return of each asset after implementation
retPFM = np.zeros((len(returns.index)-estLength,1)) #monthly return of Portfolio after implementation

histRet = np.zeros((1,nAssets))

minvar_PFdyn = np.zeros((len(returns.index)-estLength,nAssets)) #initializes matrix for min var portfolio

gamma = 1

for n in range(0,(len(returns.index)-estLength)):
    '''loop in order to calculate the efficient portfolios in each period'''
    df_estimation = returns[n:estLength+n]
    '''extract the estimation data from the dataset'''
    meanRet = np.asmatrix(np.mean(df_estimation)).T
    '''calculate mean returns of the estimation dataset'''
    estSigma = np.cov(df_estimation.T)
    
    ''' NOTE: the portfolios sum to one '''
    minvar = np.array([float(x) for x in minVarPF_noshort(estSigma)])
    minvar_PFdyn[n,:] = minvar

    rf_ann = rf_rate.iloc[estLength + n - 1]
    if freq == "M":
        rf = (1. + rf_ann) ** (1. / 12) - 1
    elif freq == "W":
        rf = (1. + rf_ann) ** (1. / 52) - 1
    elif freq == "D":
        rf = (1. + rf_ann) ** (1. / 365) - 1
    
    histRet = np.array(returns.iloc[(estLength + n)])
    retAM[n,:] = np.array(np.multiply(minvar, (np.exp(np.asmatrix(histRet))-1)))
    retPFM[n,:] = retAM[n,:].sum()

    
minvar_PFdynPortfolios = pd.DataFrame(minvar_PFdyn, index = datesPF, columns = indices) #format data as dataframe with dates and column names
checkoneminvar = minvar_PFdynPortfolios.sum(axis=1) #check if weights sum to 1

os.chdir("/Users/%s/OneDrive/Master Thesis/Data/Portfolios/no_short_sale/" %name)
minvar_PFdynPortfolios.to_csv('minvarPortfolios{}{}.csv'.format(freq, years))

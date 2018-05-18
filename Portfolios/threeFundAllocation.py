

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
years = 4

returns, rf_rate, market, estLength, nAssets = get_Data(freq, years) # years of estimation


datesPF = returns.index.values[(estLength-1):(len(returns.index)-1)] #Index dates for dataframe
datesImpl = returns.index.values[(estLength):(len(returns.index))] #Index dates for dataframe
indices = returns.columns.values #Column names for dataframe

#calculate implementation of adjusting strategy over time

oneVector = np.asmatrix(np.ones(nAssets)).T #creates a one vector for portfolio calculation
retAM = np.zeros((len(returns.index)-estLength,nAssets)) #monthly return of each asset after implementation
retPFM = np.zeros((len(returns.index)-estLength,1)) #monthly return of Portfolio after implementation

histRet = np.zeros((1,nAssets))

threeFund_PFdyn = np.zeros((len(returns.index)-estLength,nAssets)) #initializes matrix for min var portfolio

gamma = 1

for n in range(0,(len(returns.index)-estLength)):
    '''loop in order to calculate the efficient portfolios in each period'''
    df_estimation = returns[n:estLength+n]
    '''extract the estimation data from the dataset'''
    meanRet = np.asmatrix(np.mean(df_estimation)).T
    '''calculate mean returns of the estimation dataset'''
    estSigma = np.cov(df_estimation.T)
    
    rf_ann = np.array([float(x) for x in rf_rate[n:estLength+n].values])
    if freq == "M":
        rf = np.array([(1. + i) ** (1. / 12) - 1 for i in rf_ann])
    elif freq == "W":
        rf = np.array([(1. + i) ** (1. / 52) - 1 for i in rf_ann])
    elif freq == "D":
        rf = np.array([(1. + i) ** (1. / 365) - 1 for i in rf_ann])
    #convert back to dataframe
#    rf = pd.DataFrame(rf, index = rf_rate.iloc[n-1:estLength+n-1].index, columns = ["rf"])

    exrets = df_estimation.values - np.asmatrix(rf).T
    meanRet_exc = np.asmatrix(np.mean(exrets, axis = 0)).T
    threeFund = threeFundSeparationEMP(meanRet_exc, estSigma, estLength, gamma)
    threeFund = np.array([float(x) for x in threeFund])

    threeFund_PFdyn[n,:] = threeFund

threeFundPortfolios = pd.DataFrame(threeFund_PFdyn, index = datesPF, columns = indices) #format data as dataframe with dates and column names
checkonethreeFund = threeFundPortfolios.sum(axis=1) #check if weights sum to 1

os.chdir("/Users/%s/OneDrive/Master Thesis/Data/Portfolios/" %name)
threeFundPortfolios.to_csv('threeFundPortfolios{}{}.csv'.format(freq, years))

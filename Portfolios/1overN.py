
import getpass as gp
name = gp.getuser()

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import sys
sys.path.append('/Users/%s/OneDrive/Master Thesis/Data/Analysis_Skripts/Library/' %name)
from Functions import *

freq = 'M'
years = 5

get_Data(freq, years)


#Index dates for dataframe
datesPF = returns.index.values[(estLength-1):(len(returns.index)-1)] 

#Index dates for dataframe
datesImpl = returns.index.values[(estLength):(len(returns.index))]

 #Column names for dataframe
indices = returns.columns.values 

#initialize monthly return matrix with each asset after implementation
retAssets = np.zeros((len(returns.index)-estLength,nAssets)) 

#initialize monthly return vector of Portfolio after implementation
retPF = np.zeros((len(returns.index)-estLength,1)) 

#initialize matrix for Garlappi Wang portfolio weights
PFdyn = np.zeros((len(returns.index)-estLength,nAssets)) 

#initialize historical return vector for month after implementation of strategy
histRet = np.zeros((1,nAssets))

#initialize periodical trading volume vector
tradingVolume = np.zeros((len(returns.index)-estLength,1)) 

tradingVolumeAssets = np.zeros((len(returns.index)-estLength,nAssets))


'''Calculate the return of a 1/N Portfolio'''

for n in range(0,(len(returns.index)-estLength)):
    '''loop in order to calculate the efficient portfolios in each period'''
    histRet = returns.iloc[(estLength + n )] 
    pf = np.ones(nAssets) / nAssets
    PFdyn[n,:] = pf.T
    retAssets[n,:] = pf.T * (np.exp(histRet)-1)
    retPF[n,:] = retAssets[n,:].sum()
    '''Calculate trading volume / unit invested'''
    pf_EndOfPeriod = pf.T * (np.exp(histRet))
    tradingVolumeAssets[n,:] = (pf.T-pf_EndOfPeriod.T)*pf_EndOfPeriod.sum()
    tradingVolume[n,:] = abs(tradingVolumeAssets[n,:]).sum()
    
df_oneNPortfolios = pd.DataFrame(PFdyn, index = datesPF, columns = indices) 
sumTradingVolume = tradingVolume.sum()

df_TradingVolumeAssets = pd.DataFrame(tradingVolumeAssets, index = datesPF, columns = indices)

df_oneNPortfolios.to_csv('1NPortfolios%s.csv' %freq)
df_TradingVolumeAssets.to_csv('1NPortfolios%s_TradingVolume.csv' %freq)

positiveValues = np.array(sum(n > 0 for n in tradingVolumeAssets))



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


minVarPFdyn = np.zeros((len(returns.index)-estLength,nAssets)) #initializes matrix for min var portfolio
maxSlopePFdyn = np.zeros((len(returns.index)-estLength,nAssets)) #initializes matrix for max slope portfolio
maxSharpePFdyn = np.zeros((len(returns.index)-estLength,nAssets)) #initializes matrix for max slope portfolio
histRet = np.zeros((1,nAssets))

for n in range(0,(len(returns.index)-estLength)):
    '''loop in order to calculate the efficient portfolios in each period'''
    df_estimation = returns[n:estLength+n]
    '''extract the estimation data from the dataset'''
    meanRet = np.array(np.mean(df_estimation))
    '''calculate mean returns of the estimation dataset'''
    estSigma = np.cov(df_estimation.T)
    '''calculate the variance covariance matrix and its inverse'''
    MVPF = np.array([float(x) for x in minVarPF_noshort(estSigma)])
    '''calculate the min var portfolio and assign the output vector to the matrix of portfolio weights'''
    minVarPFdyn[n,:] = MVPF
    

    '''calculate the max slope portfolio and assign the output vector to the matrix of portfolio weights'''
    rf_ann = rf_rate.iloc[estLength + n - 1]
    if freq == "M":
        rf = (1. + rf_ann) ** (1. / 12) - 1
    elif freq == "W":
        rf = (1. + rf_ann) ** (1. / 52) - 1
    elif freq == "D":
        rf = (1. + rf_ann) ** (1. / 365) - 1
    
    maxSloPF = np.array([float(x) for x in maxSRPF_noshort(meanRet, estSigma, rf)])
    maxSlopePFdyn[n,:] = maxSloPF    
    '''calculate the max sharpe ratio portfolio and assign the output vector to the matrix of portfolio weights'''

    histRet = np.array(returns.iloc[(estLength + n)])
    retAM[n,:] = np.array(np.multiply(maxSloPF, (np.exp(np.asmatrix(histRet))-1)))
    retPFM[n,:] = retAM[n,:].sum()

    
    
MVPortfolios = pd.DataFrame(minVarPFdyn, index = datesPF, columns = indices) #format data as dataframe with dates and column names
MSPortfolios = pd.DataFrame(maxSlopePFdyn, index = datesPF, columns = indices) #format data as dataframe with dates and column names
df_retPFM = pd.DataFrame(retPFM, index = datesImpl, columns = ["PFret"])

#MVPortfolios.to_csv('MVPortfolios%s%sY.csv' %freq, str(years))
#MSPortfolios.to_csv('MSPortfolios%s%sY.csv' %freq, str(years))

os.chdir("/Users/%s/OneDrive/Master Thesis/Data/Portfolios/no_short_sale" %name)
MVPortfolios.to_csv('MVPortfolios{}{}.csv'.format(freq, years))
MSPortfolios.to_csv('MSPortfolios{}{}.csv'.format(freq, years))






############################ Tangency Portfolio ###############################

oneVector = np.ones(nAssets) #creates a one vector for portfolio calculation
tangPFs = np.zeros((len(returns.index)-estLength,nAssets)) #initializes matrix for min var portfolio
tangPFs_cara = np.zeros((len(returns.index)-estLength,nAssets)) #initializes matrix for min var portfolio

histRet = np.zeros((1,nAssets))

gamma = 4.

count = 0

for n in range(0,(len(returns.index)-estLength)):
    '''loop in order to calculate the efficient portfolios in each period'''
    df_estimation = returns[n:estLength + n]
    '''extract the estimation data from the dataset'''
    meanRet = np.array(df_estimation.mean())
    '''calculate mean returns of the estimation dataset'''
    estSigma = np.cov(df_estimation.T)
    rf_ann = rf_rate.iloc[(estLength + n - 1, 1)]
    if freq == "M":
        rf = (1. + rf_ann) ** (1. / 12) - 1
    elif freq == "W":
        rf = (1. + rf_ann) ** (1. / 52) - 1
    elif freq == "D":
        rf = (1. + rf_ann) ** (1. / 365) - 1
    tangPF = maxSRPF1(meanRet, estSigma, rf)
    tangPFs[n,:] = tangPF 
    tangWeight = tangWeights(meanRet, estSigma, rf, gamma)
    tangPFs_cara[n,:] = tangWeight
    histRet = np.array(returns.iloc[(estLength + n)])
    retAM[n,:] = np.array(np.multiply(tangPF, histRet))
    retPFM[n,:] = retAM[n,:].sum()


df_tanPFs = pd.DataFrame(tangPFs, index = datesPF, columns = indices)

MSPortfolios.to_csv('TanPortfolios{}{}Y.csv'.format(freq, years))



########### Tangency Portfolio further analysis of extreme values #############

list_df_estimation = []
list_varCovar = []
listMeanRet = []

Skewness = []
Kurtosis = []
TailRatio = []

dates_analysis = []
for n in range(0,(len(returns.index)-estLength)):
    '''loop in order to calculate the efficient portfolios in each period'''
    df_estimation = returns[n:estLength + n]

    '''extract the estimation data from the dataset'''
    meanRet = np.array(df_estimation.mean())
    '''calculate mean returns of the estimation dataset'''
    estSigma = np.cov(df_estimation.T)
    rf_ann = rf_rate.iloc[(estLength + n - 1, 1)]
    if freq == "M":
        rf = (1. + rf_ann) ** (1. / 12) - 1
    elif freq == "W":
        rf = (1. + rf_ann) ** (1. / 52) - 1
    elif freq == "D":
        rf = (1. + rf_ann) ** (1. / 365) - 1
    tangPF = maxSRPF1(meanRet, estSigma, rf)
    index = datetime.date((df_estimation.index[estLength-1]))
    dates_analysis.append(index)
    Skewness.append(df_estimation.skew())
    Kurtosis.append(df_estimation.kurtosis())
    TailRatio.append([tail_ratio(df_estimation[indices[i]])for i in range(nAssets)])
    
    if (np.max(tangPF) or abs(np.min(tangPF))) > 10:
        globals() ["df_estimation_"+str(index)] = df_estimation
        globals() ["varCovar_"+str(index)] = estSigma
        globals() ["meanRet_"+str(index)] = meanRet
        list_df_estimation.append(str("df_estimation_"+str(index)))
        list_varCovar.append(str("varCovar_"+str(index)))
        listMeanRet.append(str("meanRet_"+str(index)))
        count += 1
    tangPFs[n,:] = tangPF 
    tangWeight = tangWeights(meanRet, estSigma, rf, gamma)
    tangPFs_cara[n,:] = tangWeight
    histRet = np.array(returns.iloc[(estLength + n)])
    retAM[n,:] = np.array(np.multiply(tangPF, histRet))
    retPFM[n,:] = retAM[n,:].sum()


Skewness = pd.DataFrame(Skewness, index = dates_analysis, columns = indices) 
Kurtosis = pd.DataFrame(Kurtosis, index = dates_analysis, columns = indices)
TailRatio = pd.DataFrame(TailRatio, index = dates_analysis, columns = indices)


######################################################################################################


#dictionaries that contain data   
dict_df_estimation = {}
for i in list_df_estimation:
        dict_df_estimation[i].append(globals()[i])
        
        
indexDF = pd.DataFrame(dict_df_estimation.keys())
indexDF.sort_values(by=0,inplace=True)

dict_df_estimation_sorted = {}
for i,name in enumerate(indexDF[0]):
    print(i,name)
    dict_df_estimation_sorted[i] = dict_df_estimation[name][0].copy()
    

f, axs = plt.subplots(len(list_df_estimation),1, figsize = (10,200)) 
for i,key in enumerate(dict_df_estimation_sorted):
    dict_df_estimation_sorted[i].plot(ax=axs[i], s = 5)
    axs[i].legend().set_visible(False)

plt.tight_layout()
plt.savefig('extremeWeightReturns.svg')        
plt.show()    


    
    

df_tanPFs = pd.DataFrame(tangPFs, index = datesImpl, columns = indices)
df_retPFM = pd.Series(retPFM[:,0], index = datesImpl)
plot_monthly_returns_heatmap(df_retPFM)

plot_weights( datesPF, tangPFs, "SR")


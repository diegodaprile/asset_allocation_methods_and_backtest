#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  1 11:47:26 2018

@author: DiegoCarlo
"""

##############################Libraries########################################

import getpass as gp
name = gp.getuser()

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import sys
sys.path.append('/Users/%s/OneDrive/Master Thesis/Data/Analysis_Skripts/Library/' %name)
from Functions import *
from support_functions_portfolios import *
from sklearn.covariance import ledoit_wolf as LW, oas,  shrunk_covariance

from scipy.spatial.distance import pdist


import scipy.cluster.hierarchy as sch
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.cluster.hierarchy import cophenet
from scipy.cluster.hierarchy import fcluster
import random
from scipy.spatial.distance import pdist
##########################Variables and Data###################################


freq = 'M'
years = 5

get_Data(freq, years)

##############################Relevant Methods######################################


#to do 1
def getIVP(cov, **kargs):
    # Compute the inverse-variance portfolio
    ivp = 1. / np.diag(cov)
    ivp /= ivp.sum()
    return ivp

#to do 2
def getClusterVar(cov,cItems):
    # Compute variance per cluster
    cov = pd.DataFrame(cov)
    cov_=cov.loc[cItems,cItems] # matrix slice
    w_=getIVP(cov_).reshape(-1,1)
    cVar=np.dot(np.dot(w_.T,cov_),w_)[0,0]
    return cVar


def getQuasiDiag(link):
    # Sort clustered items by distance
    link = link.astype(int)
    sortIx = pd.Series([link[-1, 0], link[-1, 1]])
    numItems = link[-1, 3]  # number of original items
    while sortIx.max() >= numItems:
        sortIx.index = range(0, sortIx.shape[0] * 2, 2)  # make space
        df0 = sortIx[sortIx >= numItems]  # find clusters
        i = df0.index
        j = df0.values - numItems
        sortIx[i] = link[j, 0]  # item 1
        df0 = pd.Series(link[j, 1], index=i + 1)
        sortIx = sortIx.append(df0)  # item 2
        sortIx = sortIx.sort_index()  # re-sort
        sortIx.index = range(sortIx.shape[0])  # re-index

    return sortIx.tolist()


def getRecBipart(cov, sortIx):
    # Compute HRP alloc
    w = pd.Series(1, index=sortIx)
    cItems = [sortIx]  # initialize all items in one cluster
    while len(cItems) > 0:
        cItems = [i[j:k] for i in cItems for j, k in ((0, len(i) / 2),
                                                      (len(i) / 2, len(i))) if len(i) > 1]  # bi-section
        for i in xrange(0, len(cItems), 2):  # parse in pairs
            cItems0 = cItems[i]  # cluster 1
            cItems1 = cItems[i + 1]  # cluster 2
            cVar0 = getClusterVar(cov, cItems0)
            cVar1 = getClusterVar(cov, cItems1)
            alpha = 1 - cVar0 / (cVar0 + cVar1)
            w[cItems0] *= alpha  # weight 1
            w[cItems1] *= 1 - alpha  # weight 2
    return w


def correlDist(corr):
    # A distance matrix based on correlation, where 0<=d[i,j]<=1
    # This is a proper distance metric
    dist = ((1 - corr) / 2.)**.5  # distance matrix
    where_are_NaNs = np.isnan(dist)
    dist[where_are_NaNs] = 0;
    return dist


def generateData(nObs, sLength, size0, size1, mu0, sigma0, sigma1F):
    # Time series of correlated variables
    # 1) generate random uncorrelated data
    x = np.random.normal(mu0, sigma0, size=(nObs, size0))
    # each row is a variable
    # 2) create correlation between the variables
    cols = [random.randint(0, size0 - 1) for i in xrange(size1)]
    y = x[:, cols] + np.random.normal(0, sigma0 * sigma1F, size=(nObs, len(cols)))
    x = np.append(x, y, axis=1)
    # 3) add common random shock
    point = np.random.randint(sLength, nObs - 1, size=2)
    x[np.ix_(point, [cols[0], size0])] = np.array([[-.5, -.5], [2, 2]])
    # 4) add specific random shock
    point = np.random.randint(sLength, nObs - 1, size=2)

    x[point, cols[-1]] = np.array([-.5, 2])

    return x, cols


def getHRP(cov, corr):
    # Construct a hierarchical portfolio
    corr, cov = pd.DataFrame(corr), pd.DataFrame(cov)
    dist = correlDist(corr)
    link = sch.linkage(dist, 'single')
    sortIx = getQuasiDiag(link)
    sortIx = corr.index[sortIx].tolist()
    # recover labels
    hrp = getRecBipart(cov, sortIx)

    return hrp.sort_index()


def getCLA(cov, **kargs):
    # Compute CLA's minimum variance portfolio
    mean = np.arange(cov.shape[0]).reshape(-1, 1)
    # Not used by C portf
    lB = np.zeros(mean.shape)
    uB = np.ones(mean.shape)
    cla = CLA(mean, cov, lB, uB)
    cla.solve()
    return cla.w[-1].flatten()


def plotCorrMatrix(path,corr,labels=None):
# Heatmap of the correlation matrix
    if labels is None:labels=[]
    plt.pcolor(corr)
    plt.colorbar() 
    plt.yticks(np.arange(.5,corr.shape[0]+.5),labels) 
    plt.xticks(np.arange(.5,corr.shape[0]+.5),labels) 
    plt.savefig(path)
    plt.clf();plt.close() # reset pylab
    return


##############################Application######################################


#Index dates for dataframe
datesPF = returns.index.values[(estLength-1):(len(returns.index)-1)] 

#Index dates for dataframe
datesImpl = returns.index.values[(estLength):(len(returns.index))]

 #Column names for dataframe
indices = returns.columns.values 

#initialize monthly return matrix with each asset after implementation
retAssetsHRP = np.zeros((len(returns.index)-estLength,nAssets)) 

#initialize monthly return vector of Portfolio after implementation
retPF_HRP = np.zeros((len(returns.index)-estLength,1)) 

#initialize matrix for Garlappi Wang portfolio weights
HRP_PF = np.zeros((len(returns.index)-estLength,nAssets)) 

#initialize historical return vector for month after implementation of strategy
histRet = np.zeros((1,nAssets))



for n in range(0,(len(returns.index)-estLength)):
    '''loop in order to calculate the efficient portfolios in each period'''
    df_estimation = returns[n:estLength+n]
    covar = oas(df_estimation, assume_centered = True)[0]
    correl = cov2cor(covar)
    histRet = returns.iloc[(estLength + n)] 
    #    dist = correlDist(correl)
    #    link = sch.linkage(dist, 'single')
    #    sortCorrel = getQuasiDiag(link)
    #    HRP = getRecBipart(pd.DataFrame(covar, sortCorrel)
    HRP = getHRP(covar,correl)
    HRP_PF[n,:] = np.array(HRP.T)
    '''Calculation of optimal portfolio returns based on Hierarchical Risk Parity'''
    retAssetsHRP[n,:] = np.array(HRP.T) * (np.exp(histRet)-1)
    retPF_HRP[n,:] = retAssetsHRP[n,:].sum()

HRPPortfolios = pd.DataFrame(HRP_PF, index = datesPF, columns = indices) #format data as dataframe with dates and column names

HRPPortfolios.to_csv('HRPPortfolios%s.csv' %freq)


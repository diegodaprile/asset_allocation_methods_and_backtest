# -*- coding: utf-8 -*-
"""
Created on Fri Mar 23 22:19:04 2018
"""

import getpass as gp
global name 
name = gp.getuser()
import os
import sys
sys.path.append('/Users/%s/OneDrive/Master Thesis/Data/Analysis_Skripts/Library/' %name)
from datetime import datetime
from datetime import timedelta
from dateutil.relativedelta import relativedelta

from Functions import *

def backtester_NEW2(start, end, file, save = False, gamma = 1):

    os.chdir("/Users/%s/OneDrive/Master Thesis/Data" %name)
    years = int(file[-5:-4])
    freq = 'M'
    returns, rf_rate, market, estLength, nAssets = get_Data(freq, years) # years of estimation
    
    os.chdir("/Users/%s/OneDrive/Master Thesis/Data/Portfolios/" %name)
    df_weights = convertCSV_toDataframe(file)

    datesformat = "%Y-%m-%d"
    initial_date_backtest = datetime.strptime(start, datesformat)
    final_date_backtest = datetime.strptime(end, datesformat)
    
    windows_returns = returns.loc[initial_date_backtest : final_date_backtest]
    windows_weights = df_weights.loc[initial_date_backtest - relativedelta(months = 1) : final_date_backtest - relativedelta(months = 1)]
    nAssets = len(windows_returns.columns)
    market_window = market.loc[initial_date_backtest : final_date_backtest]
    '''important that the weights are selected in the period right before the initial date to backtest'''

    
    histRet = np.zeros((1, nAssets))
    retAssets = np.zeros((len(windows_returns.index), nAssets))
    retPF = np.zeros((len(windows_returns.index), 1))    
    turnover_pf = np.zeros((len(windows_returns.index), 1))    
    
    for n in range(len(windows_returns.index)):
        
        rf_ann_window = rf_rate.loc[initial_date_backtest : final_date_backtest][rf_rate.columns[1]]
        rf_estimation = rf_rate.loc[initial_date_backtest - relativedelta(months = 60) : initial_date_backtest, :][rf_rate.columns[1]]
        rf_rate_end_period = rf_rate.loc[final_date_backtest][1]
        if freq == "M":
            rf_vect = np.array([(1. + i) ** (1. / 12) - 1 for i in rf_ann_window])
            rf_estimation = np.array([(1. + i) ** (1. / 12) - 1 for i in rf_estimation])
            rf_rate_end_period = (1+ rf_rate_end_period) ** (1. / 12) - 1
        elif freq == "W":
            rf_vect = np.array([(1. + i) ** (1. / 52) - 1 for i in rf_ann_window])
            rf_estimation = np.array([(1. + i) ** (1. / 52) - 1 for i in rf_estimation])
            rf_rate_end_period = (1+ rf_rate_end_period) ** (1. / 52) - 1
        elif freq == "D":
            rf_vect = np.array([(1. + i) ** (1. / 365) - 1 for i in rf_ann_window])
            rf_estimation = np.array([(1. + i) ** (1. / 365) - 1 for i in rf_estimation])            
            rf_rate_end_period = (1+ rf_rate_end_period) ** (1. / 365) - 1
        
        
        weights_selected = np.array(df_weights.loc[initial_date_backtest + relativedelta(months = n - 1 ) ])
        weights_next_period = np.array(df_weights.loc[initial_date_backtest + relativedelta(months = n ) ])
        
        histRet = np.array(windows_returns.loc[initial_date_backtest + relativedelta(months = n)])
        #note the return of the assets are discrete returns
        
        #calculate the amount to invest in the risky asset
        months_back = 12 * years
        returns_est_past = returns.loc[initial_date_backtest - relativedelta(months = months_back) : initial_date_backtest]
        meanRet = returns_est_past.values.mean(axis=0)
        estSigma = np.asmatrix(np.cov(returns_est_past.T, ddof=1))
        
        pf_ret_ex_ante = PF_return(weights_selected, meanRet)
        pf_sigma_ex_ante = np.sqrt(PF_variance(weights_selected, estSigma))
        
        rf = rf_estimation.mean()
        w_risky_portfolio = weight_risky_assets(pf_ret_ex_ante, rf, pf_sigma_ex_ante, gamma)
        pf_total_return = PF_return_risky_rf(w_risky_portfolio, weights_selected, meanRet, rf)
        pf_total_sigma = PF_sigma_risky_rf(w_risky_portfolio, weights_selected, estSigma) 

        retPF[n,:] = pf_total_return
        turnover_pf[n,:] = turnover(weights_selected, histRet, weights_next_period)
        
    
    
    sharpeRatio_assets = np.array([SharpeRatio(retAssets[:,i], rf_vect) for i in range(retAssets.shape[1])])
    sharpeRatio_portfolio = SharpeRatio(retPF, rf_vect)

    #discrete return for the whole period
    
    treynorRatio_portfolio = TreynorRatio_Diego(retPF, market_window, rf_vect)
    LPM = LowerPartialMoments(retPF)
    drawdown = DrawDown(retPF, time_span = estLength)
    max_drowdown = MaximumDrawDown(retPF)
    avg_drawdown = AverageDrawDown(retPF)
    turnover_pf_sum = turnover_pf.sum() 
    
    
    retPF_cumulative_end_period = np.prod( 1 + np.array(retPF) ) - 1
    omegaRatio = OmegaRatio(retPF_cumulative_end_period, retPF, rf_rate_end_period, target=0)
    
    sortinoRatio = SortinoRatio(retPF_cumulative_end_period, retPF, rf_rate_end_period, target=0)
    retPF_1 = np.array([float(retPF[i]) for i in range(len(retPF))])
    VaR = ValueAtRisk(retPF_1, alpha = 0.05)
    ES = ExpectedShortfall(retPF_1, alpha = 0.05)
    
    performance = {}
    performance["start date"] = initial_date_backtest
    performance["end date"] = final_date_backtest
    performance["SHARPE"] = sharpeRatio_portfolio
    performance["TREYNOR"] = treynorRatio_portfolio
    performance["Turnover"] = turnover_pf_sum
    performance["LOWER PARTIAL MOMENT"] = LPM    
    performance["DRAWDOWN"] = drawdown       
    performance["MAX DRAWDOWN"] = max_drowdown       
    performance["AVG DRAWDOWN"] =  avg_drawdown
    performance["OMEGA"] =  omegaRatio
    performance["SORTINO"] =  sortinoRatio
    performance["OMEGA"] =  omegaRatio
    performance["VAR"] =  VaR
    performance["ES"] =  ES
    
    retAssets = pd.DataFrame(retAssets, index = windows_returns.index, columns = windows_returns.columns.values)
    retPF = pd.Series(retPF.reshape(len(retPF)), index = windows_returns.index)
    
    if save:
        filename_bt = datetime.today().strftime("%Y-%m-%d-%H-%M-%S")
        if not os.path.exists('/Users/{}/OneDrive/Master Thesis/Data/Portfolios/backtesting/{}'.format(name, filename_bt)):
            os.makedirs('/Users/{}/OneDrive/Master Thesis/Data/Portfolios/backtesting/{}'.format(name, filename_bt))
    
        os.chdir('/Users/{}/OneDrive/Master Thesis/Data/Portfolios/backtesting/{}'.format(name, filename_bt))
        retAssets.to_csv('return_asset_backtest_{}_{}_{}Y_{}_{}_.csv'.format(file[:-4], freq, str(years), start, end))    
        retPF.to_csv('portfolio_return_backtest_{}_{}_{}Y_{}_{}_.csv'.format(file[:-4], freq, str(years), start, end))    
    for key, value in performance.items():print("{} : {}".format(key,value))
    if save:
        performance_output = pd.Series(performance, name='Performance Indicators')
        performance_output.to_csv('portfolio_performance_{}_{}_{}Y_{}_{}_.csv'.format(file[:-4], freq, str(years), start, end))
    return  retAssets, retPF, performance


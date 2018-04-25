#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 23 22:18:19 2018

@author: DiegoCarlo
"""


import getpass as gp
global name 
name = gp.getuser()
import os
import pandas as pd
import sys
sys.path.append('/Users/%s/OneDrive/Master Thesis/Data/Analysis_Skripts/Library/' %name)
from Functions import *
from backtest_Diego import backtester_NEW2 as BACKTEST

years_list = [4, 5, 6, 8, 10]
#years_list = [4]
epsilon_list = [0.50, 1.00, 1.50, 2.00, 3.00]  
#epsilon_list = [2.00]
gamma_list = [0.5, 1.0, 1.5, 2.0, 3.0]
#gamma_list = [2.0]

start = "1980-01-01"

end = "2018-01-01"


GUW = ['GUWPortfoliosM{}-gamma{:3.1f}_epsilon{:3.1f}.csv'.format(year, gamma, eps)
            for year in years_list for gamma in gamma_list for eps in epsilon_list ]
GUW_ext = ['ourGarlPortfoM{}-gamma{:3.1f}.csv'.format(year, gamma)
            for year in years_list for gamma in gamma_list]
LPM = ['LPMPortfoliosM{}.csv'.format(year) for year in years_list]
MS = ['MSPortfoliosM{}.csv'.format(year) for year in years_list]
LW = ['LWPortfoliosM{}.csv'.format(year) for year in years_list]
#OverN = ['one_over_N.csv']
MV = ['minvarPortfoliosM{}.csv'.format(year) for year in years_list]
RP = ['RPPortfoliosM{}.csv'.format(year) for year in years_list]
HRP = ['hierarchicalPortfoliosM{}.csv'.format(year) for year in years_list]
ThreeFund = ['threeFundPortfoliosM{}.csv'.format(year) for year in years_list]


one_fund_portfolios = ['MeanVar_1F_PFM{}-gamma{:3.1f}.csv'.format(year, gamma)
        for year in years_list for gamma in gamma_list]


#portfolios = LPM + MS + LW  + MV + RP + HRP  + ThreeFund 
portfoliosGARLAPPI = GUW + GUW_ext + one_fund_portfolios

#portfoliosGARLAPPI = one_fund_portfolios

#portfolios = LW
#portfoliosGARLAPPI = GUW_ext

#'''create all the dataframe to store the results'''
#columns = ["portfolio name","Gamma","Info Estimation", "Certainty Equivalent", "Avg Ret","St Dev","Sharpe","VaR", "LPM","Turnover","Drawdown","Epsilon","Initial Date","Final Date"]
#rows = [i for i in range(len(portfolios))]
#final_dataframe = pd.DataFrame(index = rows, columns = columns)


columns = ["portfolio name","Gamma","Info Estimation", "Certainty Equivalent", "Avg Ret","St Dev","Sharpe","VaR", "LPM","Turnover","Drawdown","Epsilon","Initial Date","Final Date"]
rows2 = [i for i in range(len(portfoliosGARLAPPI))]
final_dataframe2 = pd.DataFrame(index = rows2, columns = columns)

    
'''loop for garlappi portfolios that for construction already imply a gamma'''
  

for n, i in enumerate(portfoliosGARLAPPI):
    
    if portfoliosGARLAPPI[n][14] == str(1):
        gamma = float(portfoliosGARLAPPI[n][22:25])
    else:
        gamma = float(portfoliosGARLAPPI[n][21:24])
        
    if portfoliosGARLAPPI[n][:3]=="GUW":
        retAssets, retPF, performance, file, gamma_output, average_pf_return, st_dev_pf, sharpeRatio_pf, turnover_pf_sum, LPM, drawdown, VaR, initial_date_backtest, final_date_backtest, epsilon, info_estimation, CE = BACKTEST(start, end, i, gamma, save = True)       
    else:
        retAssets, retPF, performance, file, gamma_output, average_pf_return, st_dev_pf, sharpeRatio_pf, turnover_pf_sum, LPM, drawdown, VaR, initial_date_backtest, final_date_backtest, info_estimation, CE  = BACKTEST(start, end, i, gamma, save = True)       
    
    to_print = (n + 1) / len(portfoliosGARLAPPI) * 100
    print("{:05.2f}%".format(to_print))

    if portfoliosGARLAPPI[n][:3]=="GUW":  
        final_dataframe2.set_value(n, columns[0], file)
        final_dataframe2.set_value(n, columns[1], gamma_output)
        final_dataframe2.set_value(n, columns[2], info_estimation)
        final_dataframe2.set_value(n, columns[3], CE)            
        final_dataframe2.set_value(n, columns[4], average_pf_return)
        final_dataframe2.set_value(n, columns[5], st_dev_pf)
        final_dataframe2.set_value(n, columns[6], sharpeRatio_pf)
        final_dataframe2.set_value(n, columns[7], VaR)
        final_dataframe2.set_value(n, columns[8], LPM)                          
        final_dataframe2.set_value(n, columns[9], turnover_pf_sum)
        final_dataframe2.set_value(n, columns[10], drawdown)
            
        epsilon = float(portfoliosGARLAPPI[n][-7:-4])
        final_dataframe2.set_value(n, columns[11], epsilon)
        final_dataframe2.set_value(n, columns[12], initial_date_backtest)
        final_dataframe2.set_value(n, columns[13], final_date_backtest)
    else:
        final_dataframe2.set_value(n, columns[0], file)
        final_dataframe2.set_value(n, columns[1], gamma_output)
        final_dataframe2.set_value(n, columns[2], info_estimation)
        final_dataframe2.set_value(n, columns[3], CE)                
        final_dataframe2.set_value(n, columns[4], average_pf_return)
        final_dataframe2.set_value(n, columns[5], st_dev_pf)
        final_dataframe2.set_value(n, columns[6], sharpeRatio_pf)
        final_dataframe2.set_value(n, columns[7], VaR)
        final_dataframe2.set_value(n, columns[8], LPM)                          
        final_dataframe2.set_value(n, columns[9], turnover_pf_sum)
        final_dataframe2.set_value(n, columns[10], drawdown)
        final_dataframe2.set_value(n, columns[12], initial_date_backtest)
        final_dataframe2.set_value(n, columns[13], final_date_backtest)    


folder_parent = "start_{}".format(initial_date_backtest.year)
folder_n = datetime.today().strftime("%Y-%m-%d-%H")
if not os.path.exists('/Users/{}/OneDrive/Master Thesis/Data/Analysis_Skripts/Backtest Analysis/{}/{}'.format(name, folder_parent, folder_n)):
    os.makedirs('/Users/{}/OneDrive/Master Thesis/Data/Analysis_Skripts/Backtest Analysis/{}/{}'.format(name, folder_parent, folder_n))

os.chdir('/Users/{}/OneDrive/Master Thesis/Data/Analysis_Skripts/Backtest Analysis/{}/{}'.format(name, folder_parent, folder_n))      
#final_dataframe2.to_csv('BT_result_GUW.csv') 
final_dataframe2.to_csv('BT_1Fund.csv') 






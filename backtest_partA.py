
import getpass as gp
global name 
name = gp.getuser()
import os
import pandas as pd
import sys
sys.path.append('/Users/%s/OneDrive/Master Thesis/Data/Analysis_Skripts/Library/' %name)
from Functions import *
from backtest import backtester_NEW2 as BACKTEST

#from backtest_control import backtester_NEW2 as BACKTEST

years_list = [4, 5, 6, 8, 10]
#years_list = [4]
epsilon_list = [0.50, 1.00, 1.50, 2.00, 3.00]  
#epsilon_list = [1.00]
gamma_list = [0.5, 1.0, 1.5, 2.0, 3.0]
#gamma_list = [1.0]

start = "1980-01-01"
end = "2018-01-01"


GUW = ['GUWPortfoliosM{}-gamma{:3.1f}_epsilon{:3.1f}.csv'.format(year, gamma, eps)
            for year in years_list for gamma in gamma_list for eps in epsilon_list ]
GUW_ext = ['ourGarlPortfoM{}-gamma{:3.1f}.csv'.format(year, gamma)
            for year in years_list for gamma in gamma_list]
LPM = ['LPM_Portfol_6.2%_M{}.csv'.format(year) for year in years_list]
MS = ['MSPortfoliosM{}.csv'.format(year) for year in years_list]
LW = ['LWPortfoliosM{}.csv'.format(year) for year in years_list]
#OverN = ['one_over_N.csv']
minvar = ['minvarPortfoliosM{}.csv'.format(year) for year in years_list]
RP = ['RPPortfoliosM{}.csv'.format(year) for year in years_list]
HRP = ['hierarchicalPortfoliosM{}.csv'.format(year) for year in years_list]
ThreeFund = ['threeFundPortM{}-gamma{:3.1f}.csv'.format(year, gamma) 
                for year in years_list for gamma in gamma_list]


meanVar_one_f = ['MeanVar_1F_PFM{}-gamma{:3.1f}.csv'.format(year, gamma)
        for year in years_list for gamma in gamma_list]

 
#gamma_portfolios = meanVar_one_f + GUW + GUW_ext + ThreeFund
gamma_portfolios = meanVar_one_f + GUW + GUW_ext + ThreeFund
benchmark_portfolios = minvar  + RP + LPM + HRP

#Twofund_portfolios = MS + LW 


""" CHOOSE WHICH ONE TO BACKTEST"""

portfolios = gamma_portfolios + benchmark_portfolios


columns = ["portfolio name","Gamma","Info Estimation", "Certainty Equivalent", "Avg Ret","St Dev","Sharpe","VaR", "LPM","Turnover","Drawdown","Epsilon","Initial Date","Final Date"]
rows2 = [i for i in range(len(portfolios))]
final_dataframe2 = pd.DataFrame(index = rows2, columns = columns)

    
'''loop for garlappi portfolios that for construction already imply a gamma'''


for n, i in enumerate(portfolios):
            
    if portfolios[n][:3]=="GUW":
        retAssets, retPF, performance, file, gamma_GUW, average_pf_return, st_dev_pf, sharpeRatio_pf, turnover_pf_sum, LPM, drawdown, VaR, initial_date_backtest, final_date_backtest, epsilon, info_estimation, CE = BACKTEST(start, end, i, gamma_presence = True, save = True)       
        
        final_dataframe2.set_value(n, columns[0], file)
        final_dataframe2.set_value(n, columns[1], gamma_GUW)
        final_dataframe2.set_value(n, columns[2], info_estimation)
        final_dataframe2.set_value(n, columns[3], CE)            
        final_dataframe2.set_value(n, columns[4], average_pf_return)
        final_dataframe2.set_value(n, columns[5], st_dev_pf)
        final_dataframe2.set_value(n, columns[6], sharpeRatio_pf)
        final_dataframe2.set_value(n, columns[7], VaR)
        final_dataframe2.set_value(n, columns[8], LPM)                          
        final_dataframe2.set_value(n, columns[9], turnover_pf_sum)
        final_dataframe2.set_value(n, columns[10], drawdown)
        epsilon = float(portfolios[n][-7:-4])
        final_dataframe2.set_value(n, columns[11], epsilon)
        final_dataframe2.set_value(n, columns[12], initial_date_backtest)
        final_dataframe2.set_value(n, columns[13], final_date_backtest)
        
    elif portfolios[n][:3]=="Mea" or portfolios[n][:3]=="our" or portfolios[n][:3]=="thr":
        retAssets, retPF, performance, file, gamma_output, average_pf_return, st_dev_pf, sharpeRatio_pf, turnover_pf_sum, LPM, drawdown, VaR, initial_date_backtest, final_date_backtest, info_estimation, CE  = BACKTEST(start, end, i, gamma_presence = True, save = True)       
        
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
    
    else:
        retAssets, retPF, performance, file, average_pf_return, st_dev_pf, sharpeRatio_pf, turnover_pf_sum, LPM, drawdown, VaR, initial_date_backtest, final_date_backtest, info_estimation, CE  = BACKTEST(start, end, i, gamma_presence = False, save = True)       
        
        final_dataframe2.set_value(n, columns[0], file)
#        final_dataframe2.set_value(n, columns[1], gamma_output)
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

    to_print = (n + 1) / len(portfolios) * 100
    print("{:05.2f}%".format(to_print))


        
folder_parent = "start_{}".format(initial_date_backtest.year)
folder_n = datetime.today().strftime("%Y-%m-%d-%H")
if not os.path.exists('/Users/{}/OneDrive/Master Thesis/Data/Analysis_Skripts/Backtest Analysis/{}/{}'.format(name, folder_parent, folder_n)):
    os.makedirs('/Users/{}/OneDrive/Master Thesis/Data/Analysis_Skripts/Backtest Analysis/{}/{}'.format(name, folder_parent, folder_n))

os.chdir('/Users/{}/OneDrive/Master Thesis/Data/Analysis_Skripts/Backtest Analysis/{}/{}'.format(name, folder_parent, folder_n))      
final_dataframe2.to_csv('BT_result_1FundPF_TEST.csv') 
#final_dataframe2.to_csv('BT_result_threeFund.csv') 
#final_dataframe2.to_csv('BT_result_our_Garlappi.csv') 




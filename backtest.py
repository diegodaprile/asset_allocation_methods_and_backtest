
import getpass as gp
global name 
name = gp.getuser()
import os
import sys
sys.path.append('/Users/%s/OneDrive/Master Thesis/Data/Analysis_Skripts/Library/' %name)
from datetime import datetime
from datetime import timedelta
from dateutil.relativedelta import relativedelta
from openpyxl import load_workbook
from openpyxl import Workbook
from Functions import *


def backtester_NEW2(start, end, file, gamma_presence = False, save = False):

    os.chdir("/Users/%s/OneDrive/Master Thesis/Data" %name)

    """ obtain the year and the gamma (for gamma-dependent portfolios) """             

    if file[:3]=="GUW" or file[:3]=="our" or file[:3]=="Mea":
        if int(file[14]) == 1:
            years = 10
        else:
            years = int(file[14])
    
    elif file[:3]=="thr":
        if int(file[14]) == 1:
            years = 10
        else:
            years = int(file[14])
    else:
        if int(file[-5]) == 0:
            years = 10
        else:
            years = int(file[-5])

    if gamma_presence:
        if file[:3]=="GUW":
            if file[14] == str(1):
                gamma = float(file[22:25])
            else:
                gamma = float(file[21:24])
        else:
            gamma = float(file[-7:-4])
        
    
    freq = 'M'
    returns, rf_rate, market, estLength, nAssets = get_Data(freq, years) # years of estimation
    
    os.chdir("/Users/%s/OneDrive/Master Thesis/Data/Portfolios" %name)
#    df_weights = convertCSV_toDataframe(file)
    df_weights = convertCSV_toDataframeTEST(file)
    
    datesformat = "%Y-%m-%d"
    initial_date_backtest = datetime.strptime(start, datesformat)
    final_date_backtest = datetime.strptime(end, datesformat)
    
    windows_returns = returns.loc[initial_date_backtest + relativedelta(months = 1)  : final_date_backtest + relativedelta(months = 1)]
#    windows_weights = df_weights.loc[initial_date_backtest - relativedelta(months = 1) : final_date_backtest - relativedelta(months = 1)]
    nAssets = len(windows_returns.columns)
    market_window = market.loc[initial_date_backtest : final_date_backtest]
    '''important that the weights are selected in the period right before the initial date to backtest'''

    
    histRet = np.zeros((1, nAssets))
    retPF = np.zeros((len(windows_returns.index), 1))    
    turnover_pf = np.zeros((len(windows_returns.index), 1))    
    w_risky_vector = np.zeros((len(windows_returns.index), 1))    
    
    total_w_old = np.asmatrix(np.zeros(nAssets)).T
    
    
    for n in range(len(windows_returns.index)):
        
        retAssets = windows_returns
        rf_ann_window = np.asarray(rf_rate.loc[initial_date_backtest : final_date_backtest + relativedelta(months = 1)])
        rf_estimation = np.asarray(rf_rate.loc[initial_date_backtest - relativedelta(months = estLength) : initial_date_backtest, :])
        rf_rate_end_period = float(rf_rate.loc[final_date_backtest])

        rf_vect = np.array([(1. + i) ** (1. / 12) - 1 for i in rf_ann_window])
        rf_estimation = np.array([(1. + i) ** (1. / 12) - 1 for i in rf_estimation])
        rf_rate_end_period = (1+ rf_rate_end_period) ** (1. / 12) - 1
        rf = float(rf_vect[n])
        
        weights_selected = np.array(df_weights.loc[initial_date_backtest + relativedelta(months = n)])

   
        histRet = np.array(windows_returns.loc[initial_date_backtest + relativedelta(months = n + 1)])
        #note the return of the assets are discrete returns
        
        #calculate the amount to invest in the risky asset
        months_back = 12 * years
        returns_est_past = returns.loc[initial_date_backtest - relativedelta(months = months_back - 1) : initial_date_backtest]
        meanRet = returns_est_past.values.mean(axis=0)
        estSigma = np.asmatrix(np.cov(returns_est_past.T, ddof=1))

        w_risky_portfolio = weights_selected.sum()
        
        if file[:3]=="thr":            
            ret_risky_part = PF_return(weights_selected, histRet)
            w_rf = 1 - float(weights_selected.sum())
            ret_rf_part = w_rf * rf    
            pf_total_return = float(ret_risky_part + ret_rf_part)            
        else:
            pf_total_return = PF_return(weights_selected, histRet)
            
        w_risky_vector[n,:] = w_risky_portfolio
        retPF[n,:] = pf_total_return
        
        w_now = w_risky_portfolio * weights_selected
        if n > 0:
            total_w_old = w_old * weights_selected_old

        turnover_pf[n,:] = turnover(total_w_old, histRet, w_now)

        w_old = w_risky_portfolio
        weights_selected_old = weights_selected
    
    #sharpe ratio
    sharpeRatio_assets = np.array([SharpeRatio(retAssets.iloc[:,1], rf_vect[1:]) for i in range(retAssets.shape[1])])
    sharpeRatio_portfolio = SharpeRatio(retPF, rf_vect[1:])
    
    #certainty equivalent for the strategy
    if gamma_presence:
        CE = utility_MV(float(np.mean(retPF)), float(np.std(retPF, ddof=1)), gamma)
    else:
        gamma_for_CER = 1
        CE = utility_MV(float(np.mean(retPF)), float(np.std(retPF, ddof=1)), gamma_for_CER)
    #discrete return for the whole period
    
#    treynorRatio_portfolio = TreynorRatio_Diego(retPF, market_window, rf_vect)
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
    if file[:3]=="GUW":
        epsilon = float(file[-7:-4])
        performance["epsilon"] = epsilon
    performance["average return"] = np.mean(retPF)
    performance["standard deviation"] = np.std(retPF, ddof=1)
    if gamma_presence:        
        performance["Certainty Equivalent"] = CE
    performance["INFO"] = file
    performance["start date"] = initial_date_backtest
    performance["end date"] = final_date_backtest
    if gamma_presence:        
        performance["gamma"] = gamma
    performance["SHARPE"] = sharpeRatio_portfolio
#    performance["TREYNOR"] = treynorRatio_portfolio
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
    
    performance_output = pd.Series(performance).to_frame(name='Performance Indicators')
    retAssets = pd.DataFrame(retAssets, index = windows_returns.index, columns = windows_returns.columns.values)
    retPF = pd.Series(retPF.reshape(len(retPF)), index = windows_returns.index).to_frame(name = "PF returns")
    w_risky_vector = pd.Series(w_risky_vector.reshape(len(w_risky_vector)), index = windows_returns.index).to_frame(name = "Omega_risky")
    turnover_pf = pd.Series(turnover_pf.reshape(len(turnover_pf)), index = windows_returns.index).to_frame(name = "Turnover")
    #calculates the rolling sharpe ratio
    roll_sharpe = rollingSharpe(retPF, rf_vect[1:], roll_window = estLength)
    #calculates the rolling VaR
    roll_VaR = rollingVar(retPF, roll_window = estLength)
    
    if save:
        location = "start_{}".format(initial_date_backtest.year)
        folder_simul = "BT_{}_{}_{}".format(file[:-4], start, end)
        if gamma_presence:
            filename_bt = "{}_gamma-{}".format(datetime.today().strftime("%Y-%m-%d-%H"), gamma)
        else:
            filename_bt = "{}".format(datetime.today().strftime("%Y-%m-%d-%H"))
             
        if not os.path.exists('/Users/{}/OneDrive/Master Thesis/Data/Portfolios/backtesting/no_short_sale/{}/{}'.format(name, location, folder_simul)):
            os.makedirs('/Users/{}/OneDrive/Master Thesis/Data/Portfolios/backtesting/no_short_sale/{}/{}'.format(name, location, folder_simul))
    
        os.chdir('/Users/{}/OneDrive/Master Thesis/Data/Portfolios/backtesting/no_short_sale/{}/{}'.format(name, location, folder_simul))

#        old way of saving in cvs
#        retAssets.to_csv('ret_assets_{}_{}_{}Y_{}_{}_.csv'.format(file[:-4], freq, str(years), start, end))    
#        retPF.to_csv('PF_rets_{}_{}_{}Y_{}_{}_.csv'.format(file[:-4], freq, str(years), start, end))    
#        performance_output.to_csv('PF_performance_{}_{}_{}Y_{}_{}_.csv'.format(file[:-4], freq, str(years), start, end))

#        write to excel file
        writer = pd.ExcelWriter('/Users/{}/OneDrive/Master Thesis/Data/Portfolios/backtesting/no_short_sale/{}/{}/{}.xlsx'.format(name, location, folder_simul, filename_bt))
        performance_output.to_excel(writer,  sheet_name = 'Performance Indicators')
        roll_sharpe.to_excel(writer, sheet_name = "Rolling Sharpe")
        roll_VaR.to_excel(writer, sheet_name = "Rolling VaR")
        turnover_pf.to_excel(writer, sheet_name = "Turnover")
        w_risky_vector.to_excel(writer, sheet_name = "Amount_in_Risky_Asset")
        retPF.to_excel(writer,  sheet_name = 'PF_returns')
        retAssets.to_excel(writer,  sheet_name='asset_returns')
        writer.save()

#    print()
#    for key, value in performance.items():
#        print("{} : {}".format(key,value))
#        print()
#        
#    print()

    if file[:3]=="GUW":
        info_estimation = freq + str(years)
        return  retAssets, retPF, performance, file, gamma, float(np.mean(retPF)), float(np.std(retPF, ddof=1)), sharpeRatio_portfolio, turnover_pf_sum, LPM, drawdown, VaR, initial_date_backtest, final_date_backtest, epsilon, info_estimation, CE                     

    elif (file[:3] != "GUW" and gamma_presence == True):
        info_estimation = freq + str(years)
        return  retAssets, retPF, performance, file, gamma, float(np.mean(retPF)), float(np.std(retPF, ddof=1)), sharpeRatio_portfolio, turnover_pf_sum, LPM, drawdown, VaR, initial_date_backtest, final_date_backtest, info_estimation, CE
    else:
        info_estimation = freq + str(years)
        return  retAssets, retPF, performance, file, float(np.mean(retPF)), float(np.std(retPF, ddof=1)), sharpeRatio_portfolio, turnover_pf_sum, LPM, drawdown, VaR, initial_date_backtest, final_date_backtest, info_estimation, CE




'''to use only with tangency and tangency ledoit'''


def backtester_withGamma(start, end, file, gamma, save = False):

    os.chdir("/Users/%s/OneDrive/Master Thesis/Data" %name)

    """ obtain the year and the gamma (for gamma-dependent portfolios) """             

    if file[:3]=="GUW" or file[:3]=="our" or file[:3]=="Mea":
        if int(file[14]) == 1:
            years = 10
        else:
            years = int(file[14])
    else:
        if int(file[-5]) == 0:
            years = 10
        else:
            years = int(file[-5])


    freq = 'M'
    returns, rf_rate, market, estLength, nAssets = get_Data(freq, years) # years of estimation
    
    os.chdir("/Users/%s/OneDrive/Master Thesis/Data/Portfolios/" %name)
#    df_weights = convertCSV_toDataframe(file)
    df_weights = convertCSV_toDataframeTEST(file)
    
    datesformat = "%Y-%m-%d"
    initial_date_backtest = datetime.strptime(start, datesformat)
    final_date_backtest = datetime.strptime(end, datesformat)
    
    windows_returns = returns.loc[initial_date_backtest + relativedelta(months = 1)  : final_date_backtest + relativedelta(months = 1)]
#    windows_weights = df_weights.loc[initial_date_backtest - relativedelta(months = 1) : final_date_backtest - relativedelta(months = 1)]
    nAssets = len(windows_returns.columns)
    market_window = market.loc[initial_date_backtest : final_date_backtest]
    '''important that the weights are selected in the period right before the initial date to backtest'''

    
    histRet = np.zeros((1, nAssets))
    retPF = np.zeros((len(windows_returns.index), 1))    
    turnover_pf = np.zeros((len(windows_returns.index), 1))    
    w_risky_vector = np.zeros((len(windows_returns.index), 1))    
    
    total_w_old = np.asmatrix(np.zeros(nAssets)).T
    
    
    for n in range(len(windows_returns.index)):
        
        retAssets = windows_returns
        rf_ann_window = np.asarray(rf_rate.loc[initial_date_backtest : final_date_backtest + relativedelta(months = 1)])
        rf_estimation = np.asarray(rf_rate.loc[initial_date_backtest - relativedelta(months = estLength) : initial_date_backtest, :])
        rf_rate_end_period = float(rf_rate.loc[final_date_backtest])

        rf_vect = np.array([(1. + i) ** (1. / 12) - 1 for i in rf_ann_window])
        rf_estimation = np.array([(1. + i) ** (1. / 12) - 1 for i in rf_estimation])
        rf_rate_end_period = (1+ rf_rate_end_period) ** (1. / 12) - 1
        

        
        weights_selected = np.array(df_weights.loc[initial_date_backtest + relativedelta(months = n)])

        
        histRet = np.array(windows_returns.loc[initial_date_backtest + relativedelta(months = n + 1)])
        #note the return of the assets are discrete returns
        
        #calculate the amount to invest in the risky asset
        months_back = 12 * years
        returns_est_past = returns.loc[initial_date_backtest - relativedelta(months = months_back - 1) : initial_date_backtest]
        meanRet = returns_est_past.values.mean(axis=0)
        estSigma = np.asmatrix(np.cov(returns_est_past.T, ddof=1))
        

        pf_ret_ex_ante = PF_return(weights_selected, meanRet)
        pf_sigma_ex_ante = np.sqrt(PF_variance(weights_selected, estSigma))
        rf = float(rf_vect[n])
        
        w_risky_portfolio = weight_risky_assets(pf_ret_ex_ante, rf, pf_sigma_ex_ante, gamma)
        pf_total_return = PF_return_risky_rf(w_risky_portfolio, weights_selected, histRet, rf)

        w_risky_vector[n,:] = w_risky_portfolio
        retPF[n,:] = pf_total_return
        
        w_now = w_risky_portfolio * weights_selected
        if n > 0:
            total_w_old = w_old * weights_selected_old

        turnover_pf[n,:] = turnover(total_w_old, histRet, w_now)

        w_old = w_risky_portfolio
        weights_selected_old = weights_selected
    
    #sharpe ratio
    sharpeRatio_assets = np.array([SharpeRatio(retAssets.iloc[:,1], rf_vect[1:]) for i in range(retAssets.shape[1])])
    sharpeRatio_portfolio = SharpeRatio(retPF, rf_vect[1:])
    
    #certainty equivalent for the strategy
    CE = utility_MV(float(np.mean(retPF)), float(np.std(retPF, ddof=1)), gamma)
    #discrete return for the whole period
    
#    treynorRatio_portfolio = TreynorRatio_Diego(retPF, market_window, rf_vect)
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
    if file[:3]=="GUW":
        epsilon = float(file[-7:-4])
        performance["epsilon"] = epsilon
    performance["average return"] = np.mean(retPF)
    performance["standard deviation"] = np.std(retPF, ddof=1)        
    performance["Certainty Equivalent"] = CE
    performance["INFO"] = file
    performance["start date"] = initial_date_backtest
    performance["end date"] = final_date_backtest
    performance["gamma"] = gamma
    performance["SHARPE"] = sharpeRatio_portfolio
#    performance["TREYNOR"] = treynorRatio_portfolio
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
    
    performance_output = pd.Series(performance).to_frame(name='Performance Indicators')
    retAssets = pd.DataFrame(retAssets, index = windows_returns.index, columns = windows_returns.columns.values)
    retPF = pd.Series(retPF.reshape(len(retPF)), index = windows_returns.index).to_frame(name = "PF returns")
    w_risky_vector = pd.Series(w_risky_vector.reshape(len(w_risky_vector)), index = windows_returns.index).to_frame(name = "Omega_risky")
    turnover_pf = pd.Series(turnover_pf.reshape(len(turnover_pf)), index = windows_returns.index).to_frame(name = "Turnover")
    #calculates the rolling sharpe ratio
    roll_sharpe = rollingSharpe(retPF, rf_vect[1:], roll_window = estLength)
    #calculates the rolling VaR
    roll_VaR = rollingVar(retPF, roll_window = estLength)
    
    if save:
        location = "start_{}".format(initial_date_backtest.year)
        folder_simul = "BT_{}_{}_{}".format(file[:-4], start, end)
        filename_bt = "{}_gamma-{}".format(datetime.today().strftime("%Y-%m-%d-%H"), gamma)
        if not os.path.exists('/Users/{}/OneDrive/Master Thesis/Data/Portfolios/backtesting/no_short_sale/{}/{}'.format(name, location, folder_simul)):
            os.makedirs('/Users/{}/OneDrive/Master Thesis/Data/Portfolios/backtesting/no_short_sale/{}/{}'.format(name, location, folder_simul))
    
        os.chdir('/Users/{}/OneDrive/Master Thesis/Data/Portfolios/backtesting/no_short_sale/{}/{}'.format(name, location, folder_simul))

#        old way of saving in cvs
#        retAssets.to_csv('ret_assets_{}_{}_{}Y_{}_{}_.csv'.format(file[:-4], freq, str(years), start, end))    
#        retPF.to_csv('PF_rets_{}_{}_{}Y_{}_{}_.csv'.format(file[:-4], freq, str(years), start, end))    
#        performance_output.to_csv('PF_performance_{}_{}_{}Y_{}_{}_.csv'.format(file[:-4], freq, str(years), start, end))

#        write to excel file
        writer = pd.ExcelWriter('/Users/{}/OneDrive/Master Thesis/Data/Portfolios/backtesting/no_short_sale/{}/{}/{}.xlsx'.format(name, location, folder_simul, filename_bt))
        performance_output.to_excel(writer,  sheet_name = 'Performance Indicators')
        roll_sharpe.to_excel(writer, sheet_name = "Rolling Sharpe")
        roll_VaR.to_excel(writer, sheet_name = "Rolling VaR")
        turnover_pf.to_excel(writer, sheet_name = "Turnover")
        w_risky_vector.to_excel(writer, sheet_name = "Amount_in_Risky_Asset")
        retPF.to_excel(writer,  sheet_name = 'PF_returns')
        retAssets.to_excel(writer,  sheet_name='asset_returns')
        writer.save()

#    print()
#    for key, value in performance.items():
#        print("{} : {}".format(key,value))
#        print()
#        
#    print()
    info_estimation = freq + str(years)

    return  retAssets, retPF, performance, file, gamma, float(np.mean(retPF)), float(np.std(retPF, ddof=1)), sharpeRatio_portfolio, turnover_pf_sum, LPM, drawdown, VaR, initial_date_backtest, final_date_backtest, info_estimation, CE

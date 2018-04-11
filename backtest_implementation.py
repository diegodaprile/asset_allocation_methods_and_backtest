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
#import pandas as pd
#import numpy as np
#import matplotlib.pyplot as plt
import sys
sys.path.append('/Users/%s/OneDrive/Master Thesis/Data/Analysis_Skripts/Library/' %name)
from Functions import *
#IMPORT THE BACKTEST MODULE
from backtest_Diego import backtester_NEW2 as BACKTEST


start = "1990-03-12"
end = "1996-03-12"
file = 'minvarPortfoliosM8.csv'
gamma = 1

retAssets, retPF, performance = BACKTEST(start, end, file, True)


#check the output in the following folder: '/Users/name/OneDrive/Master Thesis/Data/Portfolios/backtesting/date of backtestin.

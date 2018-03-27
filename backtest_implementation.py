# -*- coding: utf-8 -*-
"""
Created on Fri Mar 23 22:18:19 2018
"""


import getpass as gp
global name 
name = gp.getuser()
import os
#import pandas as pd
#import numpy as np
#import matplotlib.pyplot as plt
import sys
sys.path.append('/Users/path_to_append/' %name)
from Functions import *
#IMPORT THE BACKTEST MODULE
from backtest import backtester_NEW2 as BACKTEST


freq = 'M'
years = 5

START, END, FILE = "1980-03-12", "1990-03-12", 'MVPortfoliosM5Y.csv'

retAssets, retPF, performance = BACKTEST(start = START, end = END, file = FILE)


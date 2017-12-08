#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 24 11:47:13 2017

@author: jan
"""

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

m ='[2 lags][100 epochs][512 batch][5 neurons][l1 0.00,l2 0.00][l1 0.00,l2 0.00][l1 0.00,l2 0.00][0.0010 lr][0.0010 lrd][0.20 do][normalize]_equity_curve'
d = 'duka'

def annualised_sharpe(returns, periods_in_year):
    return np.sqrt(periods_in_year) * returns.mean() / returns.std()

def annual_return(equity_curve, periods_in_year):
    return equity_curve.values[-1]**(periods_in_year/len(equity_curve))-1

periods_in_year = 526311#/10#252
dataset = pd.read_csv('/Users/jan/Desktop/floyd_ec/%s/%s.csv' %(d,m), index_col=[0])
#dataset = dataset[:-200000]
transaction_cost = 0#(0.00002)
for i in [0, 0.25, 0.33, 0.5, 0.66, 0.75, 1]:
        #Plot compounded and non-compounded equity curves 

    dataset['signal_%.2f_sigma' %i]= np.sign(dataset['prediction'][dataset['prediction'].abs()>i*dataset['prediction'].std()])
    dataset['signal_%.2f_sigma' %i] = dataset['signal_%.2f_sigma' %i].ffill()
    dataset['signal_%.2f_sigma' %i]=np.sign(dataset['signal_%.2f_sigma' %i].rolling(1).mean())
    dataset['signal_%.2f_sigma' %i]=dataset['signal_%.2f_sigma' %i].fillna(0)
    dataset['trade_%.2f_sigma' %i]= (dataset['signal_%.2f_sigma' %i].shift(1)!=dataset['signal_%.2f_sigma' %i]).astype(int)
    dataset['trade_result_%.2f_sigma' %i]=dataset['return']*dataset['signal_%.2f_sigma' %i]
    dataset['trade_result_%.2f_sigma' %i] = (dataset['trade_result_%.2f_sigma' %i]+1) * (1.0-transaction_cost*dataset['trade_%.2f_sigma' %i]) -1
    dataset['equity_curve_%.2f_sigma' %i]=(dataset['trade_result_%.2f_sigma' %i]+1).cumprod()
    dataset['noncomp_curve_%.2f_sigma' %i]=(dataset['trade_result_%.2f_sigma' %i]).cumsum()        
    dataset['correct_prediction_%.2f_sigma' %i]=(dataset['signal_%.2f_sigma' %i][dataset['signal_%.2f_sigma' %i]!=0]==np.sign(dataset['return'][dataset['signal_%.2f_sigma' %i]!=0])).astype(int)

for i in [0, 0.25, 0.33, 0.5, 0.66, 0.75, 1]:
     
    dataset['equity_curve_%.2f_sigma' %i].plot()
    plt.title('equity_curve at %.2f_sigma ' %(i), fontsize=4)
    plt.ylabel('equity value')
    plt.xlabel('period')
    plt.show()
    dataset['noncomp_curve_%.2f_sigma' %i].plot()
    plt.title('non-compounding_profit_curve at %.2f_sigma' %(i), fontsize=4)
    plt.ylabel('equity value')
    plt.xlabel('period')
    plt.show()
    # Does the model have a long or short bias?
    percent_betting_up = dataset['signal_%.2f_sigma' %i][dataset['signal_%.2f_sigma' %i]>0].sum()/len(dataset['signal_%.2f_sigma' %i])#[dataset['signal_%.2f_sigma' %i]!=0])
    percent_betting_down = -dataset['signal_%.2f_sigma' %i][dataset['signal_%.2f_sigma' %i]<0].sum()/len(dataset['signal_%.2f_sigma' %i])#[dataset['signal_%.2f_sigma' %i]!=0])
    out_of_market = 1.00 - (percent_betting_up + percent_betting_down)
    print('percentage of periods betting up %.2f_sigma : ' %(i)+str(percent_betting_up*100)+' %'
          +'; percentage of periods betting down: %.2f_sigma  ' %i+str(percent_betting_down*100)+' %'
          +'; percentage of periods staying out of the market: %.2f_sigma  ' %i+str(out_of_market*100)+' %')
    #How many trades were there
    total_trades = dataset['trade_%.2f_sigma' %i].sum()
    print('There were %s total trades for %.2f_sigma.' %(total_trades, i))
    print('The annualised_sharpe for %.2f_sigma. is: %.2f.' %(i, annualised_sharpe(dataset['trade_result_%.2f_sigma' %i], periods_in_year)))
    print('The CAGR for %.2f_sigma. is: %.2f percent.' %(i, annual_return(dataset['equity_curve_%.2f_sigma' %i],periods_in_year)*100))
# For reference, plot the asset price curve
((dataset['return']+1).cumprod()).plot()
plt.title('Asset price series')
plt.ylabel('price')
plt.xlabel('period')
plt.show() 
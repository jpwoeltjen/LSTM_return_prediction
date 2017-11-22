import pandas as pd
from math import sqrt
import numpy as np

# Simulate 20 years worth of daily ohlcv and day of week data
def ohlcv_random_walk(annual_drift, annual_sd, years):
    length = 252*years+1 
    daily_drift = (annual_drift+1)**(1/252)-1
    daily_sd =annual_sd/sqrt(252)
    
    sim_data = pd.DataFrame()
    # 'open' follows a rondom walk (possibly with drift) and specified daily_sd, truncated to 4 decimal places
    sim_data['open']=np.random.normal(daily_drift, daily_sd, length)
    sim_data['open'] = round((sim_data['open']+1).cumprod(), 4)
    #high and low are just open +- half the dialy_sd
    sim_data['high'] = round(sim_data['open'] + 0.5* daily_sd, 4)
    sim_data['low'] = round(sim_data['open'] - 0.5* daily_sd, 4)
    #close is half way between open(t) and open(t+1)
    sim_data['close'] = round(0.5*sim_data['open'] + 0.5*sim_data['open'].shift(-1), 4)
    sim_data['volume'] = np.random.normal(100000, 10000, length)
    sim_data['volume'] = round(sim_data['volume'], 0)
    sim_data['day_of_business_week'] = sim_data.index%5 
    sim_data.dropna(inplace=True)
    return sim_data

sim_data = ohlcv_random_walk(0.0, 0.1, 400)  
sim_data.to_csv('data/rw.csv', header = False, index=True, encoding='utf-8')
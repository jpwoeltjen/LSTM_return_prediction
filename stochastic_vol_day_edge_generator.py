import numpy as np
import pandas as pd 
from math import sqrt
import os 


def stochastic_vol_day_edge_generator(annual_drift, annual_sd, years, edge_probability, edge_intesity):
    '''
    This function generates a random walk on which a layer of stochastic edges
    is superimposed. In particular, every Monday there's an 'edge_probability' chance that the next 
    open-to-open return is in expectation 'edge_intensity' standard deviations
    above (below) the mean if the volume the day before is above (below) the volume
    2 days ago. It returns a DataFrame with the time series and the expected profit and sharpe ratio p.a. from the edge (only).
    '''
    length = 252*years+1 
    daily_drift = (annual_drift+1)**(1/252)-1
    daily_sd =annual_sd/sqrt(252)
    vol_sd = 10000
    
    sim_data = pd.DataFrame(columns=['open','high','low','close','volume','day_of_business_week','random','up_edge','down_edge'])
    sim_data['open']=np.random.normal(daily_drift, daily_sd, length)
    sim_data['volume'] = np.random.normal(100000, vol_sd, length)
    sim_data['volume'] = round(sim_data['volume'], 0)
    sim_data['day_of_business_week'] = sim_data.index%5
    
    #create temporarily a column of random numbers to specify the probability of the edge
    sim_data['random']=np.random.uniform(0, 1, length)
    sim_data['up_edge'] = ((sim_data['random']<edge_probability)&(sim_data['volume'].shift(2)>sim_data['volume'].shift(3))&
            (sim_data['day_of_business_week']==0)).astype(int)
    sim_data['down_edge'] = ((sim_data['random']<edge_probability)&(sim_data['volume'].shift(2)<sim_data['volume'].shift(3))&
            (sim_data['day_of_business_week']==0)).astype(int)
    sim_data['open']=sim_data['open']+edge_intesity*daily_sd*sim_data['up_edge']
    sim_data['open']=sim_data['open']-edge_intesity*daily_sd*sim_data['down_edge']
    emp_sd = sim_data['open'].std()
    sim_data['open'] = round((sim_data['open']+1).cumprod(),4)
    sim_data['high'] = round(sim_data['open'] + 0.5* daily_sd, 4)
    sim_data['low'] = round(sim_data['open'] - 0.5* daily_sd,4)
    sim_data['close'] = 0.5*sim_data['open'] + 0.5*sim_data['open'].shift(-1)
    #compute expected profit p.a.
    expected_edge_profit_pa = (1.0+daily_sd*edge_intesity)**((sim_data['up_edge'].sum()+sim_data['down_edge'].sum())/years)-1.0
    expected_edge_sharpe_ratio = sqrt(252)*((1.0+expected_edge_profit_pa)**(1/252)-1.0)/emp_sd 

    sim_data=sim_data.drop(['random','up_edge','down_edge'], 1)
    sim_data.dropna(inplace=True)
    return sim_data, expected_edge_profit_pa, expected_edge_sharpe_ratio

sim_data, expected_profit_pa, expected_edge_sharpe_ratio= stochastic_vol_day_edge_generator(0.0, 0.1, 30, 0.55, 1)  
sim_data.to_csv('data/ae1.csv', header = False, index=True, encoding='utf-8')
print ('expected_profit_pa: %.2f percent; expected_edge_sharpe_ratio %.2f' % (expected_profit_pa*100, expected_edge_sharpe_ratio))
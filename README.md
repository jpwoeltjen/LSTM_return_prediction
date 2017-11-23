# LSTM Neural Network for Asset Return Prediction

This project implements a LSTM artificial neural network for asset return prediction framework. Using the Keras API, a LSTM NN is used to predict the next open-to-open return. For testing, the features used are simulated past ohlcv, and date. The simulated data are a random walk (rw) on which a layer of artificial edges is superimposed. The test for the framework is whether or not these edges can successfully be exploited. For live trading, any data, which the trader perceives to have predictive power, such as ohlcv, day of week, minute, sentiment, and fundamental data could be used. The features must be properly encoded by the user. For ordinal features the label encoder within the encode(values, columns) function can be used. Just specify the columns. Categorical features should first be label encoded and then one-hot encoded. From the one-hot encoded matrix, one column can be dropped because of perfect multicollinearity. The current configuration allows the user to input .csv files in the format:  date,open,high,low,close,volume,day_of_week,other,…

The series is split up into training, validation, an test subsections.
The model is fit on the training data. A grid search for the most effective hyper-parameters is then performed. The models are evaluated on the validation set. The best hyper-parameters are selected and the model is tested on the test set. By setting only_give_performance_of_best_model = False, trading statistics and equity curves are computed for all models. Selecting the best model manually after this step would overfit the test set though.So only the model returned by the system should be considered. By setting only_give_performance_of_best_model = True, trading statistics and equity curves are computed only for the best model. This reduces the temptation to overfit the test set. 

To reduce over-fitting dropout, weight regularization, learning rate decay, and early stopping is implemented.

Resources I used:

https://machinelearningmastery.com/use-weight-regularization-lstm-networks-time-series-forecasting/
https://github.com/jaungiers/LSTM-Neural-Network-for-Time-Series-Prediction

Raschka, Python Machine Learning

1. Test

The model is tested whether it has look-ahead bias. Look-ahead bias refers to the bug that the model has somehow -- often this is not obvious -- access to future data. If this was so, the model would of course do very well in backtesting. But it would fall apart in live trading. It is absolutely imperative that this bug doesn't exist or all further work would be worse than useless. 
Fortunately there is a simple test. Run the model over purely random data and see if it produces positive returns. If it does something is wrong. For that I simulated a random walk over 400 years. (I want to be sure.) 
As can be seen the model does not pick up a pattern. The return series itself is a random walk. The annualized Sharpe ratio is 0.07 over the test set. 
![alt text](https://github.com/jpwoeltjen/LSTM_return_prediction/blob/master/floyd_lstm_output/rw/%5B2%20lags%5D%5B100%20epochs%5D%5B512%20batch%5D%5B5%20neurons%5D%5Bl1%200.00%2Cl2%200.00%5D%5Bl1%200.00%2Cl2%200.00%5D%5Bl1%200.00%2Cl2%200.00%5D%5B0.0010%20lr%5D%5B0.0010%20lrd%5D%5B0.20%20do%5D%5Bnormalize%5D_equity_curveequity_curve_at_0.00_sigma.png "random walk")

Next similar rws are tested but with stochastic rules layered on top.
The datasets are generated with the generator functions with arguments: (annual_drift, annual_sd, years, edge_probability, edge_intesity). These functions generate random walks on which layers of stochastic edges are superimposed. These rules are getting more and more difficult for the model to detect.


2. Test

Now we want to test whether the model is actually able to find market inefficiencies. 
For ae1.csv the following function is called with the following parameters: stochastic_vol_day_edge_generator(0.0, 0.1, 30, 0.55, 1). Every Monday there's an 'edge_probability' chance that the next open-to-open return is in expectation 'edge_intensity' standard deviations above (below) the mean if the volume the day before is above (below) the volume two days ago. It returns a DataFrame with the time series and the expected profit and Sharpe ratio p.a. from the edge (only). The resulting expected profit is 18.8 percent p.a. The expected annualized Sharpe ratio 1.6. The Sharpe ratio is still quite high and I expect the model to exploit it without much difficulty. The results can be seen in results/ae1.	
![alt text](https://github.com/jpwoeltjen/LSTM_return_prediction/blob/master/results/ae1/equity_curve_at_0.00_sigma.png "ae1")

For ae2.csv the following function is called with the following parameters: stochastic_vol_mean_reversion_generator(0.0, 0.1, 60, 0.55, 0.2). This function generates a random walk on which a layer of stochastic edges is superimposed. In particular, every day there's an 'edge_probability' chance that the next open-to-open return is in expectation 'edge_intensity' standard deviations above (below) the mean if the open the day before is below (above) the open 2 days ago and if the volume the day before is above (below) the volume 2 days ago. The expected CAGR is ~9 percent. The expected Sharpe ratio is ~0.9. The edge is quite different from a1.csv in that the edge is much more frequent (everyday vs Monday) but also much smaller. The model is trained on 20 years of data and tested on 40 years. I choose such a long testing period because the edge is very small in relation to the standard deviation of the rw. Testing on a shorter period would give results with high variance because the direction of the rw is more significant than the edge. It would also be beneficial to train on a longer period than 20 years, this might not be feasible in reality though. As can be seen the model picks up the edge. 
![alt text](https://github.com/jpwoeltjen/LSTM_return_prediction/blob/master/results/ae2_try1/equity_curve_at_0.00_sigma.png "ae2")


For ae3.csv the following function is called with the following parameters: stochastic_mean_reversion_generator(0.0, 0.1, 60, 0.55, 0.05).
This function generates a random walk on which a layer of stochastic edges is superimposed. In particular, every day there's an 'edge_probability' chance that the next open-to-open return is in expectation 'edge_intensity' standard deviations above (below) the mean if the open the day before is below (above) the open 2 days ago. The expected CAGR ~4.5 percent. The expected Sharpe ratio ~0.4.
This edge is even more frequent and much smaller than ae2. The expected move on any day when this edge occurs is only 0.05 standard deviations. 
The model is trained on 20 years of data and tested on 40 years. I choose such a long testing period because the edge is very small in relation to the standard deviation of the rw. Testing on a shorter period would give results with very high variance because the direction of the rw is more significant than the edge. It would also be beneficial to train on a longer period than 20 years, this might not be feasible in reality though. As can be seen the model picks up the edge. 

![alt text](https://github.com/jpwoeltjen/LSTM_return_prediction/blob/master/results/ae3_try1/equity_curve_at_0.00_sigma.png "ae3")


For ae4.csv the following function is called with the following parameters: stochastic_mean_reversion_generator_lf(0.0, 0.1, 30, 0.9, 1).
This simulation is similar to ae3 but the edge occurs less frequent while having roughly the same expected Sharpe ratio (and thus the move is larger). The model picks up the edge. 
![alt text](https://github.com/jpwoeltjen/LSTM_return_prediction/blob/master/results/ae4_try1/equity_curve_at_0.00_sigma.png "ae4")


3. Test

The model is tested on real data. For this test I obtained minute resolution data for EURUSD from 2011 to 2016 from http://www.histdata.com. I trained this model for 100 epochs on FloydHub with the Titan K80 GPU. This took about an hour. The result is an astonishing **5.03 Sharpe** and **60% CAGR** for this **single asset**. But this assumes no transaction costs and mid-price execution. These costs would of course be substantial and in its unrefined form, the model wouldn't be profitable. There are however ways to reduce trading activity and increase the predictive power of the model. 

![alt text](https://github.com/jpwoeltjen/LSTM_return_prediction/blob/master/floyd_lstm_output/eurusd_1m_2011-2016/%5B2%20lags%5D%5B100%20epochs%5D%5B512%20batch%5D%5B5%20neurons%5D%5Bl1%200.00%2Cl2%200.00%5D%5Bl1%200.00%2Cl2%200.00%5D%5Bl1%200.00%2Cl2%200.00%5D%5B0.0010%20lr%5D%5B0.0010%20lrd%5D%5B0.20%20do%5D%5Bnormalize%5D_equity_curveequity_curve_at_0.00_sigma.png "EURUSD_1M")







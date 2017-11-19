readme.txt

This project implements a LSTM artificial neural network for asset return prediction framework. Using the Keras API, a LSTM NN is used to predict the next open-to-open return. For testing, the features used are simulated past ohlcv, and date. The simulated data are a random walk (rw) on which a layer of artificial edges is superimposed. The test for the framework is whether or not these edges can successfully be exploited. For live trading, any data, which the trader perceives to have predictive power, such as ohlcv, day of week, minute, sentiment, and fundamental data could be used. The features must be properly encoded by the user. For ordinal features the label encoder within the encode(values, columns) function can be used. Just specify the columns. Categorical features should first be label encoded and then one-hot encoded. From the one-hot encoded matrix, one column can be dropped because of perfect multicollinearity. The current configuration allows the user to input csv files in the format:  date,open,high,low,close,volume,day_of_week,other,…

An unpredictable rw is simulated.
Stochastic rules are layered on top of this rw to create small patterns which the model is supposed to exploit.
These patterns are small enough such that other frameworks I tested couldn't pick them up.

The series is split up into training, validation, an test subsections.
The model is fit on the training data. A grid search for the most effective hyper-parameters is then performed. The models are evaluated on the validation set. The best hyper-parameters are selected and the model is tested on the test set. 

To reduce over-fitting dropout, weight regularization, learning rate decay of the Adam optimizer of 0.01, and early stopping is implemented.

Resources I used:
https://machinelearningmastery.com/use-weight-regularization-lstm-networks-time-series-forecasting/
https://github.com/jaungiers/LSTM-Neural-Network-for-Time-Series-Prediction
Raschka, Python Machine Learning

As a first test, the model is run over the rw.csv file. This is purely random data. If the model produces consistently positive results on this dataset there is a look-ahead-bias present. Somehow future information are available to the model. This doesn’t seem to be the case here. 

Secondly a similar rw is tested but with stochastic rules layered on top.
The dataset is generated with the generator functions with arguments: (annual_drift, annual_sd, years, edge_probability, edge_intesity) function. This function generates a random walk on which a layer of stochastic edges is superimposed.
For ae1.csv the following function is called with the following parameters: stochastic_vol_day_edge_generator(0.0, 0.1, 30, 0.55, 1). Every Monday there's an 'edge_probability' chance that the next open-to-open return is in expectation 'edge_intensity' standard deviations above (below) the mean if the volume the day before is above (below) the volume two days ago. It returns a DataFrame with the time series and the expected profit and Sharpe ratio p.a. from the edge (only). The resulting expected profit p.a. is 18.8 percent. The expected Sharpe ratio 1.6. This Sharpe ratio is still quite high and I expect the model to exploit it without much difficulty. The results can be seen in results/ae1.	

For ae2.csv the following function is called with the following parameters: stochastic_vol_mean_reversion_generator(0.0, 0.1, 30, 0.55, 0.2). This function generates a random walk on which a layer of stochastic edges is superimposed. In particular, every day there's an 'edge_probability' chance that the next open-to-open return is in expectation 'edge_intensity' standard deviations above (below) the mean if the open the day before is below (above) the open 2 days ago and if the volume the day before is above (below) the volume 2 days ago. The expected CAGR 8.84 percent. The expected Sharpe ratio 0.84. The edge is quite different from a1.csv in that the edge is much more frequent (everyday vs Monday) but also much smaller. 
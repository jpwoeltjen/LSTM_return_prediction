readme.txt

This project implements a LSTM Artificial Neural Network for asset return prediction framework. Using the Keras API, a LSTM NN is used to predict the next open-to-open return. For testing, the features used are simulated past returns, volume, and date. The simulated data are a random walk (rw) on which a layer of artificial edges is imposed. The test for the framework is whether or not these edges can successfully be exploited. For live trading, any data, which the trader perceives to have predictive power, such as ohlcv, day of week, minute, sentiment, and fundamental data could be used. The features must be properly encoded by the user. For ordinal features the label encoder within the encode(values, columns) function can be used. Just specify the columns. Categorical features should first be label encoded and then one-hot encoded. From the one-hot encoded matrix, one column can be dropped because of perfect multicollinearity. The current configuration allows the user to input csv files in there format:  date,open,high,low,close,volume,day_of_week,other,…

Data are simulated. These data are an unpredictable rw.
Probabilistic rules are layered on top of this rw to create small patterns which the model is supposed to exploit.
These patterns are small enough such that other frameworks I tested couldn't pick them up.

The series is split up into training, validation, an test subsections.
The model is fit on the training data. A grid search for the most effective hyper-parameters is then performed. The models are evaluated on the validation set. The best hyper-parametrs are selected and the model is tested on the test set. 

To reduce over-fitting dropout, weight regularization, and early stopping is implemented.

Resources I used:
https://machinelearningmastery.com/use-weight-regularization-lstm-networks-time-series-forecasting/
https://github.com/jaungiers/LSTM-Neural-Network-for-Time-Series-Prediction
Raschka, Python Machine Learning

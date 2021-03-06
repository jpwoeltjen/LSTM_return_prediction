
import numpy as np
import pandas as pd
from math import sqrt
from numpy import concatenate
from pandas import concat
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers.core import Dropout
from sklearn.preprocessing import StandardScaler
from keras.callbacks import EarlyStopping
from sklearn.pipeline import make_pipeline
from keras.regularizers import L1L2
import matplotlib.pyplot as plt
from keras import optimizers

np.random.seed(777)


def multivariate_ts_to_supervised_extra_lag(data, n_in=1, n_out=1, dropna=True):
    """
    Convert series to supervised learning problem and respect the fact that you can't tade 
    on the past open but only on the next open. The holding period is assumed to be open to open.
    Alteratively, you may want to hold from open to close. 
    """
    n_vars = 1 if type(data) is list else data.shape[1]
    df = pd.DataFrame(data)
    cols, names = list(), list()
    # exlude most recent data which, for trading, is not yet available
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
    
    # forecast sequence (t+n_out)
    cols.append(df[0].shift(-n_out))
    names.append('return')
    agg = concat(cols, axis=1)
    agg.columns = names
    agg['return'] +=1
    agg['return'] = agg['return'].rolling(n_out).apply(np.prod)-1
#    agg['return'] = agg['return'].rolling(n_out).sum()
    agg=agg[agg.index%(n_out)==0]
    if dropna:
        agg.dropna(inplace=True)
        
    # print(agg)    
    return agg



def get_returns(data, columns=[1,2,3,4], dropna=True):
    """
    Create new DataFrame with ohlc converted into returns and other columns left unchanged.
    """
    data_returns= data.copy(deep=True)
    for i in columns:
        data_returns[i]=data_returns[i].pct_change()
        if dropna:
            data_returns.dropna(inplace=True)
    return data_returns        


def encode(values, columns=[5]):
    encoded_values =np.array(values)
    le = LabelEncoder()
    for i in columns:
        encoded_values[:,i] = le.fit_transform(values[:,i])
    return encoded_values    
        


def scale(values, train_pct, method='normalize'):
    '''
    normalize or stadardize
    '''
    scaled_values=np.array(values)
# fit on training set so that no information leaks into test set.
    if method=='normalize':
        scaler = MinMaxScaler(feature_range=(-1, 1))
        scaler = scaler.fit(scaled_values[:int(train_pct*len(scaled_values))])
        scaled_values = scaler.transform(scaled_values)
    
    if method=='standardize':
    # Alternatively, you may want to stadardize instead. This method assumes that the variables
    # are gaussian, however. Empirically, for returns this doesn't hold.
        scaler = StandardScaler()
        scaler = scaler.fit(scaled_values[:int(train_pct*len(scaled_values))])
        scaled_values = scaler.transform(scaled_values)
        
    return scaler, scaled_values#

def invert_scale(scaler, y):
    inverted = np.array(y)
    inverted = scaler.inverse_transform(inverted)
    return inverted


    
def fit_lstm(model, train, val_X, val_y, batch, n_epochs, n_neurons, layers, lstm_layers, lags, n_features, breg, kreg, rreg, lr, lrd, do):
    n_obs = lags * n_features
    train_X, train_y = train[:, :n_obs], train[:, -1]
    train_X = train_X.reshape((train_X.shape[0], lags, n_features))
    
    # design network
    if model == None:
        model = Sequential()
        if lstm_layers == 1:
            model.add(LSTM(n_neurons, activation='sigmoid', inner_activation='sigmoid', input_shape=(train_X.shape[1], train_X.shape[2]), bias_regularizer=breg, kernel_regularizer=kreg, recurrent_regularizer=rreg, recurrent_dropout=0.0))#, return_sequences=True))
            model.add(Dropout(do))
        elif lstm_layers >1:
            model.add(LSTM(n_neurons, activation='sigmoid', inner_activation='sigmoid', input_shape=(train_X.shape[1], train_X.shape[2]), bias_regularizer=breg, kernel_regularizer=kreg, recurrent_regularizer=rreg, recurrent_dropout=0.0, return_sequences=True))
            for i in range(lstm_layers-2):
                # You may add further layers
                model.add(LSTM(n_neurons, return_sequences=True))
                model.add(Dropout(do))
            model.add(LSTM(n_neurons))
            model.add(Dropout(do))

        for i in range(layers):    
            model.add(Dense(n_neurons, activation='sigmoid'))
            model.add(Dropout(do))   
        
        model.add(Dropout(do))
        model.add(Dense(1))
        
    adam = optimizers.Adam(lr=lr, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=lrd)
    nadam = optimizers.Nadam(lr=lr, beta_1=0.9, beta_2=0.999, epsilon=1e-08, schedule_decay=0.004)
    model.compile(loss='mean_squared_error', optimizer=nadam)

    history = model.fit(train_X, train_y, epochs=n_epochs,
                  validation_data=(val_X, val_y),
                  batch_size=batch, 
                  verbose=2, shuffle=False, 
#                  callbacks=[EarlyStopping(monitor='val_loss', patience=100, verbose=2, mode='auto')]
                    )
    return model, history

def validate(model, dataset, train_pct, val_pct, lags, n_repeats, n_epochs, batch, n_neurons, layers, lstm_layers, n_features, breg, kreg, rreg, lr, lrd, do, scaling_method, p_out):
    n_obs = lags * n_features
    dataset_returns = pd.DataFrame(dataset)
    dataset_returns = get_returns(dataset_returns)#, columns=[1,2,3,4,9,10,11,12,13,14,15])
    values = dataset_returns.values
    values_encoded = values#encode(values)
    reframed = multivariate_ts_to_supervised_extra_lag(values_encoded, lags, p_out)
    reframed_values=reframed.values
    scaler, scaled = scale(reframed_values, train_pct, scaling_method)
    
    reframed_values = scaled.astype('float32')
    train, val = reframed_values[:int(train_pct*len(reframed)), :] , reframed_values[int(train_pct*len(reframed)):int((train_pct+(1-train_pct)*val_pct)*len(reframed)), :]
    # split into input and outputs
    val_X, val_y = val[:, :n_obs], val[:, -1]
    # reshape input to be 3D [samples, timesteps, features]
    val_X_reshaped = val_X.reshape((val_X.shape[0], lags, n_features))
    # run r times
    error_scores = list()
    train_loss = pd.DataFrame()
    val_loss = pd.DataFrame()
    for r in range(n_repeats):
        # fit the model
        lstm_model, history = fit_lstm(model, train, val_X_reshaped, val_y, batch, n_epochs, n_neurons, layers, lstm_layers, lags, n_features, breg, kreg, rreg, lr, lrd, do)
        # forecast val dataset
        yhat = lstm_model.predict(val_X_reshaped, batch_size=batch)
        # invert scaling
        invert_array = concatenate((val_X[:, :],yhat), axis=1)
        invert_array = invert_scale(scaler, invert_array)
        yhat_inverted = invert_array[:,-1]
        
        val_y_reshaped = val_y.reshape((len(val_y), 1))
        invert_array = concatenate((val_X[:, :],val_y_reshaped), axis=1)
        invert_array = invert_scale(scaler, invert_array)
        y_inverted = invert_array[:,-1]            

        # report performance
        rmse = sqrt(mean_squared_error(y_inverted, yhat_inverted))
        print('%d) Validation RMSE: %.3f' % (r+1, rmse))
        error_scores.append(rmse)
        train_loss[r]=(history.history['loss'])
        val_loss[r]=(history.history['val_loss'])
        #return error_scores for boxplot, train_loss/val_loss to identify over-fitting, and the model for later OOS testing
    return error_scores,train_loss.mean(axis=1), val_loss.mean(axis=1), lstm_model, lags, batch, scaling_method

def out_of_sample_test(dataset, train_pct, val_pct, lags, batch,  n_features,  model, scaling_method, p_out):
    n_obs = lags * n_features
    dataset_returns = pd.DataFrame(dataset)
    dataset_returns = get_returns(dataset_returns)#, columns=[1,2,3,4,9,10,11,12,13,14,15])
    values = dataset_returns.values
    values_encoded = values#encode(values)
    reframed = multivariate_ts_to_supervised_extra_lag(values_encoded, lags, p_out)
    reframed_values=reframed.values
    scaler, scaled = scale(reframed_values, train_pct, scaling_method)
    
    reframed_values = scaled.astype('float32')
    test = reframed_values[int((train_pct+(1-train_pct)*val_pct)*len(reframed)):, :]
    # print(test)
    test_X= test[:, :n_obs]
    # reshape input to be 3D [samples, timesteps, features]
    test_X_reshaped = test_X.reshape((test_X.shape[0], lags, n_features))
    yhat = model.predict(test_X_reshaped, batch_size=batch)
    # invert scaling
    invert_array = concatenate((test_X[:, :],yhat), axis=1)
    invert_array = invert_scale(scaler, invert_array)
    yhat_inverted = invert_array[:,-1]

    # print(yhat_inverted.shape)
    
    test_y = test[:, -1]  
    test_y_reshaped = test_y.reshape((len(test_y), 1))
    invert_array = concatenate((test_X,test_y_reshaped), axis=1)
    invert_array = invert_scale(scaler, invert_array)
    y_inverted = invert_array[:,-1]
    # print(y_inverted.shape)





    output_df=pd.DataFrame()
    output_df['prediction'] = pd.Series(yhat_inverted)
    output_df['return']=pd.Series(y_inverted)#reframed['return']
    # print(output_df)
    # dataset_returns[1] +=1
    # dataset_returns[1] = dataset_returns[1].rolling(p_out).apply(np.prod)-1
    # dataset_returns[1] = dataset_returns[1][dataset_returns[1].index%(p_out)==0]
    # dataset_returns['prediction']=pd.Series(yhat_inverted, index=dataset_returns.index[-len(yhat_inverted):])###
    # reframed['prediction'] = pd.Series(yhat_inverted, index=dataset_returns.index[-len(yhat_inverted):])
    # return dataset_returns with OOS predictions
    return output_df

def equity_curve(dataset, m, periods_in_year, plot, threshold = [0, 0.25, 0.5, 1 ,2]):
    # Define the threshold as a percentage of the standard deviation of return predictions. The lower the threshold the more often the strategy is in the market.
    dataset.dropna(inplace=True)
    
    for i in threshold:
        dataset['signal_%.2f_sigma' %i]= np.sign(dataset['prediction'][dataset['prediction'].abs()>i*dataset['prediction'].std()])
        dataset['signal_%.2f_sigma' %i]=dataset['signal_%.2f_sigma' %i].fillna(0)
        dataset['trade_result_%.2f_sigma' %i]=dataset['return']*dataset['signal_%.2f_sigma' %i]
        dataset['equity_curve_%.2f_sigma' %i]=(dataset['trade_result_%.2f_sigma' %i]+1).cumprod()
        dataset['noncomp_curve_%.2f_sigma' %i]=(dataset['trade_result_%.2f_sigma' %i]).cumsum()        
        dataset['correct_prediction_%.2f_sigma' %i]=(dataset['signal_%.2f_sigma' %i][dataset['signal_%.2f_sigma' %i]!=0]==np.sign(dataset['return'][dataset['signal_%.2f_sigma' %i]!=0])).astype(int)


    print('::::::::FOR MODEL: %s:::::::' %m)
    for i in threshold:
        #If there are any trades at all, calculate some statistics.
        if (len(dataset['correct_prediction_%.2f_sigma' %i].dropna()))>0:

            pct_correct = sum(dataset['correct_prediction_%.2f_sigma' %i].dropna())/len(dataset['correct_prediction_%.2f_sigma' %i].dropna())
            print('Percent correct %.2f_sigma: ' %i + str((pct_correct)*100)+" %")
            #Plot compounded and non-compounded equity curves 
            if plot:
                dataset['equity_curve_%.2f_sigma' %i].plot()
                plt.title('equity_curve at %.2f_sigma %s' %(i,m), fontsize=4)
                plt.ylabel('equity value')
                plt.xlabel('period')
                plt.savefig('%sequity_curve_at_%.2f_sigma.png' %(m,i))
                plt.close()
                dataset['noncomp_curve_%.2f_sigma' %i].plot()
                plt.title('non-compounding_profit_curve at %.2f_sigma %s' %(i,m), fontsize=4)
                plt.ylabel('equity value')
                plt.xlabel('period')
                plt.savefig('%snon-compounding_profit_curve_at_%.2f_sigma.png' %(m,i))
                plt.close()
            # Does the model have a long or short bias?
            percent_betting_up = dataset['signal_%.2f_sigma' %i][dataset['signal_%.2f_sigma' %i]>0].sum()/len(dataset['signal_%.2f_sigma' %i])#[dataset['signal_%.2f_sigma' %i]!=0])
            percent_betting_down = -dataset['signal_%.2f_sigma' %i][dataset['signal_%.2f_sigma' %i]<0].sum()/len(dataset['signal_%.2f_sigma' %i])#[dataset['signal_%.2f_sigma' %i]!=0])
            out_of_market = 1.00 - (percent_betting_up + percent_betting_down)
            print('percentage of periods betting up %.2f_sigma : ' %(i)+str(percent_betting_up*100)+' %'
                  +'; percentage of periods betting down: %.2f_sigma  ' %i+str(percent_betting_down*100)+' %'
                  +'; percentage of periods staying out of the market: %.2f_sigma  ' %i+str(out_of_market*100)+' %')
            #How many trades were there
            dataset['trade_%.2f_sigma' %i]= (dataset['signal_%.2f_sigma' %i].shift(1)!=dataset['signal_%.2f_sigma' %i]).astype(int)
            total_trades = dataset['trade_%.2f_sigma' %i].sum()
            print('There were %s total trades for %.2f_sigma.' %(total_trades, i))
            print('The annualised_sharpe for %.2f_sigma. is: %.2f.' %(i, annualised_sharpe(dataset['trade_result_%.2f_sigma' %i], periods_in_year)))
            print('The CAGR for %.2f_sigma. is: %.2f percent.' %(i, annual_return(dataset['equity_curve_%.2f_sigma' %i],periods_in_year)*100))
    # For reference, plot the asset price curve
    if plot:
        ((dataset['return']+1).cumprod()).plot()
        plt.title('Asset price series')
        plt.ylabel('price')
        plt.xlabel('period')
        plt.savefig('Asset_price_series.png')
        plt.close() 
    return dataset

def annualised_sharpe(returns, periods_in_year):
    '''
    Assumes daily returns are supplied. If not change periods in year.
    '''
    
    # periods_in_year = 368751#252
    return np.sqrt(periods_in_year) * returns.mean() / returns.std()

def annual_return(equity_curve, periods_in_year):
    # periods_in_year = 368751#252
    return equity_curve.values[-1]**(periods_in_year/len(equity_curve))-1
    

        
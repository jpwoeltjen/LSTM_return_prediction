
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.regularizers import L1L2
import lstm
def run():
# Load dataset
    dataset = pd.read_csv('/Users/jan/Desktop/genotick/target/SPXUSD_artificial_edge/tiny_edge_on_date/SPXUSD_artificial_edge.csv', header=None, index_col=0)
    dataset.reset_index(drop=True, inplace=True)
    #Define hyperparameters
    train_pct = 0.5 #as a percentage of the whole dataset length
    val_pct = 0.25 #pct of test set used for validation ( NOT as a percentage of the whole dataset length)
    n_lags = 2 #How many past periods should the model have direct access to. Analogous to window length in, for example, a moving average model. 
    n_features = len(dataset.columns) # How many features are there?
    n_repeats = 1 # How many runs per configuration to calculate means and boxplot.
    n_epochs = [2] # List of epochs. For specific value, for example 300, set to [300].
    n_batch = 100 # Batch size, the smaller the longer the time to run.
    n_neurons = [5]#,5,10,50] # List of number of neurons. 
    regularizers = [L1L2(l1=0.0, l2=0.0)]#, L1L2(l1=0.01, l2=0.0), L1L2(l1=0.0, l2=0.01), L1L2(l1=0.01, l2=0.01)]
    threshold = [0, 0.25, 0.5, 1 ,2]
    results = pd.DataFrame()
    train_loss = pd.DataFrame()
    val_loss = pd.DataFrame()
    models = {}
    for reg in regularizers:
        name_reg = ('l1 %.2f,l2 %.2f' % (reg.l1, reg.l2))
        for neurons in n_neurons:
            name_neurons = (' [%i neurons]' % (neurons))
            name = name_reg + name_neurons
            for epochs in n_epochs:
                name_epochs = (' [%i epochs]' % (epochs))
                name = name_reg + name_neurons + name_epochs
                results[name], train_loss[name], val_loss[name], models[name] = lstm.validate(dataset, train_pct, val_pct, n_lags, n_repeats, epochs, n_batch, neurons, n_features, reg)
    # summarize results
    print(results.describe())
    # save boxplot
    results.boxplot()
    plt.savefig('validation_boxplot.png')
    plt.close()
    train_loss.plot()
    plt.title('Train_loss') 
    plt.savefig('Train_loss.png')
    plt.close()
    val_loss.plot()
    plt.title('Val_loss')
    plt.savefig('Val_loss.png')
    plt.close()
    #Identify and select the best model.
    best_model_name = results.mean(axis=0).idxmin(axis=1)
    print(' Selecting model %s based on smallest mean of validation RMSE. Out of: %s' % (best_model_name, models.keys))
    best_model = models[best_model_name]
    out_of_sample_dataset = lstm.out_of_sample_test(dataset, train_pct, val_pct, n_lags, n_batch,  n_features,  best_model)
    equity_curve_data = lstm.equity_curve(out_of_sample_dataset, threshold)
    equity_curve_data.to_csv('equity_curve.csv',header = True, index=True, encoding='utf-8')

    return results, train_loss, val_loss, best_model, equity_curve_data
   
results, train_loss, val_loss, model, equity_curve_data = run()




import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.regularizers import L1L2
import lstm
import os 

def run():
# Load dataset
    directory = os.path.dirname(os.path.abspath(__file__))
    #whats the name of the file that contains the data; the data should be in the data directory otherwise give the complete path in path
    rel_path ='data/ae3.csv'
    path = os.path.join(directory, rel_path)
    #format: date,open,high,low,close,volume,other, NO HEADER, format of date doesn't matter because it's removed in the next step anyway
    dataset = pd.read_csv(path, header=None, index_col=0)
    #remove index
    dataset.reset_index(drop=True, inplace=True)
    #Define hyperparameters
    train_pct = 0.7 #as a percentage of the whole dataset length
    val_pct = 0.1 #pct of test set used for validation ( NOT as a percentage of the whole dataset length)
    n_lags = 2 #How many past periods should the model have direct access to. Analogous to window length in, for example, a moving average model. 
    n_features = len(dataset.columns) # How many features are there?
    n_repeats = 1 # How many runs per configuration to calculate means and boxplot.
    n_epochs = [200] # List of epochs. For specific value, for example 300, set to [300].
    n_batch = 100 # Batch size, the smaller the longer the time to run.
    n_neurons = [5]#,5,10,50] # List of number of neurons. 
    regularizers = [L1L2(l1=0.0, l2=0.0)]#[L1L2(l1=0.0, l2=0.0), L1L2(l1=0.01, l2=0.0), L1L2(l1=0.0, l2=0.01), L1L2(l1=0.01, l2=0.01)]
    threshold = [0, 0.25, 0.5, 1 ,2]
    results = pd.DataFrame()
    train_loss = pd.DataFrame()
    val_loss = pd.DataFrame()
    models = {}
    #loop over all hyperparameters you want to test
    for reg in regularizers:
        name_reg = ('l1 %.2f,l2 %.2f' % (reg.l1, reg.l2))
        for neurons in n_neurons:
            name_neurons = (' [%i neurons]' % (neurons))
            name = name_reg + name_neurons
            for epochs in n_epochs:
                name_epochs = (' [%i epochs]' % (epochs))
                name = name_reg + name_neurons + name_epochs
                results[name], train_loss[name], val_loss[name], models[name] = lstm.validate(dataset, train_pct, val_pct, n_lags, n_repeats, epochs, n_batch, neurons, n_features, reg)
    
    #Identify and select the best model.
    best_model_name = results.mean(axis=0).idxmin(axis=1)
    print(' Selecting model %s based on smallest mean of validation RMSE. Out of: %s' % (best_model_name, models.keys()))
    best_model = models[best_model_name]
    
    #Create output directory if it doesn't already exists and cd to it
    filename = os.path.basename(path)
    if filename.endswith('.csv'):
        filename = filename[:-4]
    output_path = str(directory+'/results/'+filename)

    if os.path.exists(output_path)==False:
        os.makedirs(output_path)
    os.chdir(output_path)

    # summarize results of all models
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

    out_of_sample_dataset = lstm.out_of_sample_test(dataset, train_pct, val_pct, n_lags, n_batch,  n_features,  best_model)
    equity_curve_data = lstm.equity_curve(out_of_sample_dataset, threshold)
    equity_curve_data.to_csv('equity_curve.csv',header = True, index=True, encoding='utf-8')

    return results, train_loss, val_loss, best_model, equity_curve_data
   
results, train_loss, val_loss, model, equity_curve_data = run()



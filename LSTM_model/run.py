
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.regularizers import L1L2
import lstm
import os 

def run():
# Load dataset
    # directory = os.path.dirname(os.path.abspath(__file__))
    #whats the name of the file that contains the data; the data should be in the data directory otherwise give the complete path in path
    # rel_path ='data/rw.csv'
    # path = os.path.join(directory, rel_path)
    path ='/data/EURUSD_10M_2014-2016.csv'
    # path = '/Users/jan/Documents/deep_learning/LSTM2/data/EURUSD_10M_2011-2016.csv'
    output_dir ='/output/'#output directory. primarily needed if using floydhub. if plot = True this will be overwritten 
    #format: date,open,high,low,close,volume,other, NO HEADER, format of date doesn't matter because it's removed in the next step anyway
    dataset = pd.read_csv(path, header=None, index_col=0)
    #remove index
    dataset.reset_index(drop=True, inplace=True)
    periods_in_year = 31607#for annualization of Sharpe and CAGR
    #Define hyperparameters
    train_pct = 0.5 #as a percentage of the whole dataset length
    val_pct = 0.1 #pct of test set used for validation ( NOT as a percentage of the whole dataset length)
    n_lags = [2]#How many past periods should the model have direct access to. Analogous to window length in, for example, a moving average model. 
    n_features = len(dataset.columns) # How many features are there?
    n_repeats = 1 # How many runs per configuration to calculate means and boxplot.
    n_epochs = [100] # List of epochs. For specific value, for example 300, set to [300].
    n_batch = [512] # Batch size, the smaller the longer the time to run.
    n_neurons = [5]#,5,10,50] # List of number of neurons. 
    bias_regularizers = [L1L2(l1=0.0, l2=0.0)]#[L1L2(l1=0.0, l2=0.0), L1L2(l1=0.01, l2=0.0), L1L2(l1=0.0, l2=0.01), L1L2(l1=0.01, l2=0.01)]
    kernel_regularizers = [L1L2(l1=0.00, l2=0.0)]#, L1L2(l1=0.0, l2=0.01)]#[L1L2(l1=0.0, l2=0.0), L1L2(l1=0.01, l2=0.0), L1L2(l1=0.0, l2=0.01), L1L2(l1=0.01, l2=0.01)]
    recurrent_regularizers = [L1L2(l1=0.0, l2=0.0)]#[L1L2(l1=0.0, l2=0.0), L1L2(l1=0.01, l2=0.0), L1L2(l1=0.0, l2=0.01), L1L2(l1=0.01, l2=0.01)]
    learning_rates = [0.001]#, 0.001, 0.005]
    learning_rate_decay = [0.001]#, 0.001, 0.005]
    dropout = [0.2]
    threshold = [0, 0.25, 0.5, 1 ,2]
    scaling_method = ['normalize']#,'standardize']
    only_give_performance_of_best_model = False #if False, trading performance of all models is computed but only best model is returned; if true, only trading performance of best model is computed
    p_out = 1 #how many periods out you want to perdict
    plot=False

    results = pd.DataFrame()
    train_loss = pd.DataFrame()
    val_loss = pd.DataFrame()
    models = {}
    opt_lags = {}
    opt_batch = {}
    opt_sm = {}

    #loop over all hyperparameters you want to test

    for lags in n_lags:
        name_lags = ('[%i lags]' % (lags))
        for epochs in n_epochs:
            name_epochs = ('[%i epochs]' % (epochs))
            for batch in n_batch:
                name_batch = ('[%i batch]' % (batch))
                for neurons in n_neurons:
                    name_neurons = ('[%i neurons]' % (neurons))
                    for breg in bias_regularizers:
                        name_breg = ('[l1 %.2f,l2 %.2f]' % (breg.l1, breg.l2))
                        for kreg in kernel_regularizers:
                            name_kreg = ('[l1 %.2f,l2 %.2f]' % (kreg.l1, kreg.l2))
                            for rreg in recurrent_regularizers:
                                name_rreg = ('[l1 %.2f,l2 %.2f]' % (rreg.l1, rreg.l2))
                                for lr in learning_rates:
                                    name_lr = ('[%.4f lr]' % (lr))
                                    for lrd in learning_rate_decay:
                                        name_lrd = ('[%.4f lrd]' % (lrd))
                                        for do in dropout:
                                            name_do = ('[%.2f do]' % (do))
                                            for sm in scaling_method:
                                                name_sm = ('[%s]' % (sm))
                                                name = (name_lags + name_epochs + name_batch + name_neurons + name_breg  + name_kreg + 
                                                name_rreg + name_lr + name_lrd + name_do) + name_sm
                                                results[name], train_loss[name], val_loss[name], models[name], opt_lags[name], opt_batch[name], opt_sm[name]= lstm.validate(dataset, 
                                                train_pct, val_pct, lags, n_repeats, epochs, batch, neurons, n_features, breg, kreg, rreg, lr, lrd, do, sm, p_out)
                
    #Identify and select the best model.
    best_model_name = results.mean(axis=0).idxmin(axis=1)
    print(' Selecting model %s based on smallest mean of validation RMSE. Out of: %s' % (best_model_name, models.keys()))
    best_model = models[best_model_name]
    

    if plot:
        output_dir=''
        #Create output directory if it doesn't already exists and cd to it
        filename = os.path.basename(path)
        if filename.endswith('.csv'):
            filename = filename[:-4]
        output_path = str(directory+'/results/'+filename)

        if os.path.exists(output_path)==False:
            os.makedirs(output_path)
        os.chdir(output_path)

        
        # save boxplot
        results.boxplot(fontsize=4, rot=5)
        plt.title('Boxplot RMSE')
        plt.savefig('validation_boxplot.png')
        plt.close()
        train_loss.plot()
        plt.title('Train_loss') 
        plt.legend(loc=2, prop={'size': 6})
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.savefig('Train_loss.png')
        plt.close()
        val_loss.plot()
        plt.title('Val_loss')
        plt.legend(loc=2, prop={'size': 6})
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.savefig('Val_loss.png')
        plt.close()

    # summarize results of all models
    print(results.describe())

    if only_give_performance_of_best_model ==True:
        models_keys = best_model_name
    else:
        models_keys = models.keys()
    for m in models_keys:
        out_of_sample_dataset = lstm.out_of_sample_test(dataset, train_pct, val_pct, opt_lags[m], opt_batch[m],  n_features,  models[m], opt_sm[m], p_out)
        equity_curve_data = lstm.equity_curve(out_of_sample_dataset, m, periods_in_year, plot, threshold)
        equity_curve_data.to_csv('%s%s_equity_curve.csv' %(output_dir,m), header = True, index=True, encoding='utf-8')

    return results, train_loss, val_loss, best_model
   
results, train_loss, val_loss, best_model = run()



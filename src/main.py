import pickle
import os.path
import numpy             			  as     np
import pandas                         as     pd
import matplotlib.pyplot              as     plt
import csv
#import plotly.graph_objects           as     go
from   plotly.subplots                import make_subplots
from   tensorflow                     import keras
from   features.build_features        import load_data, filt_EMG, env_EMG
from   features.build_features    	  import bin_output, make_NEUmat, bin_input
from   models.utilities               import getTrSet, input_NN
from   models.metrics          		  import get_R2, get_rho2
from   models.train_and_predict_model import LSTMDecoder
from   pathlib import Path
#from   bayes_opt         import BayesianOptimization

fname = Path("src/data/trial_data_0411.mat")
print(fname.name)
# prints "raw_data.txt"
print(fname.suffix)
# prints "txt"
print(fname.stem)
# prints "raw_data"
if not fname.exists():
    print("Oops, file doesn't exist!")
else:
    print("Yay, the file exists!")

EMGr, spike_times, electrode, fs_tdt, fs_neuron, T, labels, base_trials, stim_params = load_data(fname, only_baseline = True)

plt.figure(figsize=(15,12))
for m in range(EMGr[1].shape[1]):
        plt.subplot(4,2,m+1)
        plt.plot(EMGr[0][:,m])
        plt.title(str(labels[m]))
plt.rcParams.update({'font.size': 10})

for i in range(len(EMGr)):
    x   = EMGr[i]
df = pd.DataFrame(data=x, columns=([str(labels[0]), str(labels[1]), str(labels[2]), str(labels[3]), str(labels[4]), str(labels[5]), str(labels[6]), str(labels[7])]))

df['time'] = df.index / fs_tdt
df = df.set_index(['time'])
df.plot(kind='line', subplots=True, layout=(4,2), figsize=(15,12), 
        use_index=True, title='Muscle Activity of Session 1',
        legend=True, xlabel='time [sec]', ylabel='volts', fontsize=10)


#Preprocess data to get the matrices of EMG envelope and of firing rates
bin_size = 0.05 #In seconds
overlap  = 0.03 #In seconds
EMGbin   = []
FRmat    = []
for i in range(len(EMGr)):
    #Filter EMG
    EMGf   = filt_EMG(EMGr[i], filt_ind = [0, 1, 2, 3, 4])
    #Extract EMG envelope
    EMGenv = env_EMG(EMGf, fs_tdt)
    #Bin the EMG envelope
    EMGbin_tmp = bin_output(EMGenv, int(bin_size*fs_tdt), int(overlap*fs_tdt))
    
    #Make matrix of firings (time on rows neurons on columns, ij = 1 if neuron j
    #fired at instant i)
    NEUmat     = make_NEUmat(spike_times[i], electrode[i], int(float(T[i])*fs_neuron))
    #Make the firing rate matrix
    FRmat_tmp  = bin_input(NEUmat, int(bin_size*fs_neuron), int(overlap*fs_neuron))

    #Check if the number of time bins is coherent
    n = min([FRmat_tmp.shape[0], EMGbin_tmp.shape[0]])
    EMGbin = EMGbin + [EMGbin_tmp[0:n,:]]
    FRmat  = FRmat + [FRmat_tmp[0:n,:]]

plt.figure(figsize=(15,10))
for m in range(EMGenv.shape[1]):
        plt.subplot(4,2,m+1)
        plt.plot(EMGenv[:,m])
        #plt.legend(('raw EMG'), loc = 'upper right')
        plt.title(str(labels[m]))
plt.rcParams.update({'font.size': 10})
plt.show()    

#del EMGf, EMGenv, EMGbin_tmp, NEUmat, FRmat_tmp, EMGr, spike_times, electrode

#%%Test decoding performance using Cross Validation
bins_before = 20
bins_after  = 20
R2_lstm     = []
rho2_lstm   = []
best_params = []
#Load previously saved best parameters
fname = Path("src/models/best_params.pckl")
print(fname.name)
# prints "raw_data.txt"
print(fname.suffix)
# prints "txt"
print(fname.stem)
# prints "raw_data"
if not fname.exists():
    print("Oops, file doesn't exist!")
else:
    print("Yay, the file exists!")
f           = open(fname,'rb')
best_params = pickle.load(f)
f.close()
for ts_val_ind in range(len(EMGbin)):
    print('\nFold '+str(ts_val_ind+1)+' of '+str(len(EMGbin))+'...\n')
    
    #Divide the dataset in training, validation and test set
    Z_tr       = getTrSet(FRmat, ts_val_ind)
    Z_val      = FRmat[ts_val_ind][:int(FRmat[ts_val_ind].shape[0]/2),:] #The val set is the 1st half of the trial indicated by ts_val_ind
    Z_ts       = FRmat[ts_val_ind][-int(FRmat[ts_val_ind].shape[0]/2):,:] #The test set is the 2nd half of the trial indicated by ts_val_ind
    X_tr       = getTrSet(EMGbin, ts_val_ind)
    X_val      = EMGbin[ts_val_ind][:int(FRmat[ts_val_ind].shape[0]/2),:] #The val set is the 1st half of the trial indicated by ts_val_ind
    X_ts       = EMGbin[ts_val_ind][-int(FRmat[ts_val_ind].shape[0]/2):,:] #The test set is the 2nd half of the trial indicated by ts_val_ind
        
    #Normalize outputs
    Xmin       = np.min(X_tr,axis=0)
    Xmax       = np.max(X_tr,axis=0)
    X_tr       = (X_tr-Xmin)/(Xmax-Xmin)
    X_val      = (X_val-Xmin)/(Xmax-Xmin)
    X_ts       = (X_ts-Xmin)/(Xmax-Xmin)
    
    #Format data for Neural Networks
    Z_tr        = input_NN(Z_tr, bins_before, bins_after)
    Z_ts        = input_NN(Z_ts, bins_before, bins_after)
    Z_val       = input_NN(Z_val, bins_before, bins_after)
    X_tr        = X_tr[bins_before:X_tr.shape[0]-bins_after,:]
    X_ts        = X_ts[bins_before:X_ts.shape[0]-bins_after,:]
    X_val       = X_val[bins_before:X_val.shape[0]-bins_after,:]
        
    #LSTM Network with Bayesian Optimization (uncomment below if you want to do bayesian optimization)
#    def lstm_evaluate(units, dropout, lr): #Function that evaluates the network for the given hyperparameters
#        units       = int(units)
#        dropout     = float(dropout)
#        lr          = float(lr)
#        
#        lstm  = LSTMDecoder(units=units, dropout=dropout, lr=lr, num_epochs = 10, verbose=0, poly = False)
#        lstm.fit(Z_tr, X_tr)
#        X_out = lstm.predict(Z_val)
#        
#        return np.mean(get_R2(X_val, X_out))
#    
#    BO = BayesianOptimization(lstm_evaluate, {'units': (50, 150), 'dropout': (0, 0.2), 'lr': (0.0001, 0.003)}, verbose = 1)
#    BO.maximize(init_points = 20, n_iter = 20)
#    best_params = best_params + [BO.res['max']['max_params']]
        
    units       = int(best_params[ts_val_ind]['units'])
    dropout     = float(best_params[ts_val_ind]['dropout'])
    lr          = float(best_params[ts_val_ind]['lr'])
        
    lstm       = LSTMDecoder(units=units, dropout=dropout, lr=lr, num_epochs = 100, poly = False)
    lstm.fit(Z_tr, X_tr, validation_data = (Z_val, X_val), patience = 5)
    X_out_lstm = lstm.predict(Z_ts)
    R2_lstm    = R2_lstm + [get_R2(X_ts, X_out_lstm)]
    rho2_lstm  = rho2_lstm + [get_rho2(X_ts, X_out_lstm)]
    print('\nR2 = ' + str(R2_lstm[ts_val_ind]))
    plt.figure(figsize=(15,12))
    for m in range(X_ts.shape[1]):
        plt.subplot(5,2,m+1)
        plt.plot(X_ts[:,m])
        plt.plot(X_out_lstm[:,m])
        plt.legend(('Network Output','Target EMG'), loc = 'upper right')
        plt.title(str(labels[m]) + ' LSTM')
    plt.rcParams.update({'font.size': 10})
    plt.show()
            
    del units, dropout, lr, lstm, Z_tr, X_tr, Z_val, X_val, X_out_lstm
#%%Print mean (across folds) rho2 and R2
print('\n\nLSTM \n'+'-'*20)
print('\nrho2 = ' + str(np.mean(rho2_lstm, axis = 0)))
print('\nR2 = ' + str(np.mean(R2_lstm, axis = 0)))

#%%Save the best parameters for each fold
f2 = open('src/models/best_params.pckl', 'wb')
pickle.dump(best_params, f2)
f2.close()



#%%Go from Offline to Online Decoding
Neural = FRmat.copy()
EMG = EMGbin.copy()

n_steps=50

EMGdata = np.concatenate((EMG[0], EMG[1], EMG[2], EMG[3]), axis=0)
EMGdata = EMGdata[:,4]

def generate_time_series(iterable, n_steps=1):
    l = len(iterable)
    for ndx in range(0, l, n_steps):
        yield iterable[ndx:min(ndx + n_steps, l)]

seriesEMG = []
for x in generate_time_series(EMGdata, n_steps + 10):
    seriesEMG.append(x)
seriesEMG=np.stack(seriesEMG[0:-1])
seriesEMG = seriesEMG[..., np.newaxis].astype(np.float32)

NEUROdata = np.concatenate((Neural[0], Neural[1], Neural[2], Neural[3]), axis=0)
NEUROdata = NEUROdata[:,4]

seriesNEURO = []
for x in generate_time_series(NEUROdata, n_steps + 10):
    seriesNEURO.append(x)
seriesNEURO=np.stack(seriesNEURO[0:-1])
seriesNEURO = seriesNEURO[..., np.newaxis].astype(np.float32)

X_train, y_train = seriesNEURO[:50, :n_steps], seriesEMG[:50, -10:, 0]
X_valid, y_valid = seriesNEURO[50:80, :n_steps], seriesEMG[50:80, -10:, 0]
X_test, y_test = seriesNEURO[80:130, :n_steps], seriesEMG[80:130, -10:, 0]

Y = np.empty((154, n_steps, 10))
for step_ahead in range(1, 10 +1):
  Y[:, :, step_ahead - 1] = seriesEMG[:, step_ahead:step_ahead + n_steps, 0]

Y_train = Y[:50]
Y_valid = Y[50:80]
Y_test = Y[80:130]  

model = keras.models.Sequential([
                                 keras.layers.SimpleRNN(20, return_sequences=True, input_shape=[None, 1]),
                                 keras.layers.SimpleRNN(20, return_sequences=True),
                                 keras.layers.TimeDistributed(keras.layers.Dense(10)),
])

model.summary()

def last_time_step_mse(Y_true, Y_pred):
  return keras.metrics.mean_squared_error(Y_true[:, -1], Y_pred[:, -1])

optimizer = keras.optimizers.Adam(lr=0.01) 

model.compile(loss='mse',
              optimizer=optimizer,
              metrics=[last_time_step_mse]
              )

history = model.fit(X_train, Y_train, epochs=20,
                    validation_data=(X_valid, Y_valid))

history.params

history.epoch

pd.DataFrame(history.history).plot(figsize=(8,5))
plt.grid(True)
plt.gca().set_ylim(0, 0.5)
plt.show()
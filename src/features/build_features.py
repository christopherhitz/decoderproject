from   scipy             import io, signal
import numpy             as     np

#########################LOAD BLACKROCK AND TDT FILES##########################
def load_data(fname, only_baseline = False):
    
    f = io.loadmat(fname)
    
    #Separate different data and adjust their size
    emg         = np.squeeze(f['emg'])
    
    spike_times = np.squeeze(f['spike_times'])
    for i in range(spike_times.shape[0]):
        spike_times[i] = np.squeeze(spike_times[i])
        
    electrode   = np.squeeze(f['electrode'])
    for i in range(electrode.shape[0]):
        electrode[i] = np.squeeze(electrode[i])
        
    fs_tdt      = np.squeeze(f['fs_tdt'])
    fs_neuron   = np.squeeze(f['fs_neuron'])
    
    T           = np.squeeze(f['T'])
    for i in range(T.shape[0]):
        T[i] = np.squeeze(T[i])
        
    labels      = np.squeeze(f['labels'])
    for i in range(labels.shape[0]):
        labels[i] = np.squeeze(labels[i])
        
    base_trials = int(f['base_trials'])
    
    if not only_baseline:
        stim_params = np.squeeze(f['stim_params'])
        for i in range(stim_params.shape[0]):
            stim_params[i] = np.squeeze(stim_params[i])
    else:
        stim_params = None
        
    return emg, spike_times, electrode, fs_tdt, fs_neuron, T, labels, base_trials, stim_params



##############################FILTER EMG#######################################
def filt_EMG(emg, noise_th = 2, h_win_l = 1, filt_ind = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]):
    """
    Function that filters the emg to remove the noise peaks
    
    Inputs
    -------
    emg      : The signal to be filtered (time on rows and muscles on columns)
    noise_th : The threshold (multiplied by the std) above which the signal is 
               considered noise. Default is 2
    h_win_l  : Half of the length of the window of the zeroed signal around the 
               noise. Default is 1
    filt_ind : index of the muscles to be filtered. Default are shoulder,biceps
               and triceps
    
    Output
    -------
    EMGf : the filtered EMG (time on rows and muscles on columns)
    """
    EMGf = np.copy(emg)
    
    for c in filt_ind:
        #Check for the instants in which the threshold is passed
        r_index = np.where(abs(emg[:,c]) > np.mean(abs(emg[:,c])) + noise_th*np.std(abs(emg[:,c])))
        #Go to the next muscle if there is no threshold passing
        if not r_index[0].size > 0:
            continue
        #Substitute noise values with 0
        for r in r_index[0]:
            if r >= h_win_l and r+h_win_l < EMGf.shape[0]:
                EMGf[r-h_win_l:r+h_win_l+1,c] = 0
            elif r < h_win_l:
                EMGf[0:r+h_win_l+1,c] = 0
            elif r+h_win_l >= EMGf.shape[0]:
                EMGf[r-h_win_l:-1,c] = 0
#    EMGf = emg            
    return EMGf
        


#################################EMG ENVELOPE##################################
def env_EMG(emg, fs):
    """
    Function that construct the envelope of the EMG signal
    
    Inputs
    -------
    emg : The signal to be enveloped (time on rows and muscles on columns)
    fs  : Sampling frequency of the signal
    
    Output
    -------
    EMGenv : The envelope of the input signal (time on rows and muscles on columns)
    """
    EMGenv = np.copy(emg)
    
    #Remove line noise
    cof_50 = np.array([49, 51])
    Wn_50  = 2*cof_50/fs
    Wn_50[Wn_50 >= 1] = 0.99
    [B50, A50] = signal.butter(3, Wn_50, 'bandstop') #third order bandstop Butterworth filter
    EMGenv     = signal.filtfilt(B50, A50, EMGenv, axis = 0)
    
    #BandPass filtering
    cof_1 = np.array([80, 500])
    Wn_1  = 2*cof_1/fs
    Wn_1[Wn_1 >= 1] = 0.99
    [B1, A1] = signal.butter(3, Wn_1, 'bandpass') #third order bandpass Butterworth filter
    EMGenv   = signal.filtfilt(B1, A1, EMGenv, axis = 0)
    
    #Rectify
    EMGenv = abs(EMGenv)
    
    #LowPass filtering
    cof_2 = np.array([10])
    Wn_2  = 2*cof_2/fs
    Wn_2[Wn_2 >= 1] = 0.99
    [B2, A2] = signal.butter(3, Wn_2, 'lowpass') #third order lowpass Butterworth filter
    EMGenv   = signal.filtfilt(B2, A2, EMGenv, axis = 0)
    
    return EMGenv



#################################BIN EMG#######################################
def bin_output(emg, bin_size, overlap = 0):
    """
    Function that bins the emg with the desired bin size
    
    Inputs
    -------
    emg      : The signal to be binned (time on rows and muscles on columns)
    bin_size : size of time bins (in samples)
    overlap  : number of samples that overlap in adiacent bins. Default = 0
    
    Output
    -------
    EMGbin : The binned data (bins on rows and muscles on columns)
    """
    win_d  = bin_size - overlap
    n_bins = int(((emg.shape[0]-bin_size)/win_d)+1)
    EMGbin = np.empty([n_bins, emg.shape[1]])
    for i in range(n_bins):
        EMGbin[i,:] = np.mean(emg[i*win_d:i*win_d+bin_size,:], axis = 0)
        
    return EMGbin



############################MAKE MATRIX OF FIRINGS#############################
def make_NEUmat(times, elec, T, N = 96):
    """
    Function that builds the matrix of firings (element ij is 1 if neuron j
    fired at instant i, 0 otherwise)
    
    Inputs
    -------
    times : 1D array containing firing times
    elec  : 1D array containing the identity of firing neuron
    T     : Duration of acquisition (in samples)
    N     : Number of neurons. Default is 96
    
    Output
    -------
    NEUmat : matrix of firings
    """
    NEUmat = np.zeros((T,N))
    times  = times[elec<=N] #Sometimes there is a 129th channel (error)
    elec   = elec[elec<=N]
    boolArray = times<=T # Sometimes there are firing times outside the duration of the aquisition
    temp = times[boolArray] # ...
    times = temp # ...
    temp = elec[boolArray] # ...
    elec = temp # ...
    times[times == 0] = 1 #I can't do 0 - 1 and use it as an index in the line below
    NEUmat[times-1,elec-1] = 1
    
    return NEUmat



########################MAKE FIRING RATE MATRIX################################
def bin_input(neu, bin_size, overlap = 0):
    """
    Function that bins the firing matrix with the desired bin size
    
    Inputs
    -------
    neu      : The firing matrix (time on rows and neurons on columns)
    bin_size : size of time bins (in samples)
    overlap  : number of samples that overlap in adiacent bins. Default = 0
    
    Output
    -------
    FRmat : The binned firings (bins on rows and neurons on columns)
    """
    win_d  = bin_size - overlap
    n_bins = int(((neu.shape[0]-bin_size)/win_d)+1)
    FRmat  = np.empty([n_bins, neu.shape[1]])
    for i in range(n_bins):
        FRmat[i,:] = np.sum(neu[i*win_d:i*win_d+bin_size,:], axis = 0)
        
    return FRmat
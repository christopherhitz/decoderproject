from   scipy                 import io, signal
import numpy                 as     np
from   sklearn.decomposition import FastICA, NMF
from   models.metrics               import get_R2
#####################GET TRAINING DATA GIVEN THE PARTITION#####################
def getTrSet(alist, ts_val_ind = None):
    """
    Function that takes the list of trials and returns the training set for
    Cross Validation
    
    Inputs
    -------
    alist  : List in which data from trials are stored. Every element of the list
             has time on rows and neurons or muscles on columns
    ts_ind : Index of the element in the list that is the TEST set for the 
             current partition. Default = None says that all alist is taken as
             training set
             
    Output
    -------
    TrSet  : The Training Set with time on rows and neurons or muscles on
             columns
    """
    #Get the size of the output and initialize it
    if not isinstance(ts_val_ind,list):
        t = [alist[k].shape[0] for k in range(len(alist)) if k!=ts_val_ind]
    else:
        t = [alist[k].shape[0] for k in range(len(alist)) if (k!=ts_val_ind[0] and k!=ts_val_ind[1])]
    T     = sum(t)
    TrSet = np.zeros((T,alist[0].shape[1])) 
    
    #Fill TrSet
    k = 0
    for tr in range(len(alist)):
        if not isinstance(ts_val_ind,list):
            if tr == ts_val_ind:
                continue
        else:
            if tr == ts_val_ind[0] or tr == ts_val_ind[1]:
                continue
        r_end = k + alist[tr].shape[0]
        TrSet[k:r_end,:] = alist[tr]
        k = r_end
        
    return TrSet    



########################FORMAT INPUT LINEAR FILTER#############################
def input_linear_filter(Z, nBins):
    """
    Function that formats the Firing Rate matrix for the Linear Filter
    
    Inputs
    -------
    Z     : Matrix of binned Firing Rates (bins on rows and neurons on columns)
    nBins : Number of bins to give as input to the decoder (one corresponds to
            the output time and the others are before)
    
    Output
    -------
    Zformat : Input for the filter
    """
    Zformat = np.zeros([Z.shape[0]-nBins+1, Z.shape[1]*nBins])
    k = 0
    for b in range(nBins, Z.shape[0]):
        Zformat[k,:] = np.reshape(Z[b-nBins:b,:],(1,Zformat.shape[1]))
        k += 1
        
    return Zformat



#########################FORMAT INPUT NEURAL NETWORKS##########################
def input_NN(Z, bins_before, bins_after):
    """
    Function that formats the Firing Rate matrix for the neural networks
    
    Inputs
    -------
    Z           : Matrix of binned Firing Rates (bins on rows and neurons on 
                  columns)
    bins_before : Number of bins preceding the current time step to give as 
                  input to the decoder
    bins_after  : Number of bins following the current time step to give as 
                  input to the decoder
    
    Output
    -------
    Zformat : Input for the NNs. 3D numpy array with dimensions
              [n_samples, nBins, n_neurons]
    """
    nBins   = bins_before + bins_after + 1 #The current time bin is always included
    Zformat = np.zeros([Z.shape[0]-nBins+1, nBins, Z.shape[1]])
    for k in range(Zformat.shape[0]):
        Zformat[k,:,:] = Z[k:k+nBins,:]
    
    return Zformat



############################SMOOTH FIRING RATE#################################
def smoothFR(Z, w_len = 3, sigma = 1):
    """
    Function that smooths the Firing Rate Matrix using a gaussian window
    
    Inputs
    -------
    Z     : the matrix to be smoothed (time*neurons)
    w_len : length of the gaussian window
    sigma : standard deviation of the gaussian window
    
    Output
    -------
    Z_smooth : the smoothed matrix
    """
    w        = signal.gaussian(w_len, sigma)
    Z_smooth = np.copy(Z)
    for n in range(Z.shape[1]):
        Z_smooth[:,n] = np.convolve(Z[:,n],w,mode='same')
    return Z_smooth
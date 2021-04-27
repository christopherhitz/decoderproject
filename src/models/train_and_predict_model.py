import numpy         as np
from   numpy.linalg  import pinv as pinv
from   numpy.linalg  import inv  as inv
from   keras.models  import Model, Sequential, load_model
from   keras.layers  import Input, Dense, LSTM, Dropout, BatchNormalization
import keras.backend as K
from   keras         import optimizers as kopt
from   keras         import regularizers, callbacks
import tensorflow    as tf

#######################LINEAR FILTER WITH NON LINEARITY########################
class LinearFilter():
    
    def __init__(self, degree = 3):
        self.degree = degree #Degree of the polynomial fit
        
        
    def fit(self, Z_tr, X_tr):
        #Add bias and convert to np.matrix
        Z_bias = np.matrix(np.c_[Z_tr, np.ones((Z_tr.shape[0], 1))])
        X      = np.matrix(X_tr)
        #Least square solution
        H      = pinv(Z_bias)*X_tr
        #Polynomial fitting
        X_out  = Z_bias*H
        X_out  = np.array(X_out)
        p      = []
        for r in range(X_out.shape[1]):
            p =  p + [np.polyfit(X_out[:,r], X[:,r], self.degree)]
        params     = [H, p]
        self.model = params
        
        
    def predict(self, Z_ts):
        #Extract model parameters
        H, p = self.model
        #Add bias and convert to np.matrix
        Z_bias = np.matrix(np.c_[Z_ts, np.ones((Z_ts.shape[0], 1))])
        #Regression
        X      = Z_bias*H
        X      = np.array(X)
        #Apply the polynomial
        X_predict = []
        for r in range(X.shape[1]):
            X_predict = X_predict + [np.polyval(p[r], X[:,r])]
        return np.array(X_predict).T
    
    
    
####################KALMAN FILTER WITH POLYNOMIAL FITTING######################
class KalmanFilter():
    
    def __init__(self, C=1, degree = 3): #C = Parameter that divides the W matrix, 
        self.C      = C                  #introduced by Glacer et al.
        self.degree = degree
        
    
    def fit(self, Z_tr, X_tr):
        #Learn the Gaussian Linear Model
        M  = Z_tr.shape[0]
        Z  = np.matrix(Z_tr.T)
        X  = np.matrix(X_tr.T)
        X1 = X[:,1:]
        X2 = X[:,:M-1]
        
        A = X1*X2.T*inv(X2*X2.T)
        W = (1/(M-1))*(X1*X1.T - A*X2*X1.T)/self.C
        H = Z*X.T*(inv(X*X.T))
        Q = (Z*Z.T - H*X*Z.T)/M
        
        #Apply the Kalman algorithm to the training set
        x0    = np.mean(X, axis = 1)
        P0    = np.matrix(np.eye(X.shape[0]))
        X_out = []
        x_old = x0
        P_old = P0
        for k in range(M):
            z     = Z[:,k]
            x_    = A*x_old
            P_    = A*P_old*A.T + W
            K     = P_*H.T*(inv(H*P_*H.T+Q))
            X_out = X_out + [x_+K*(z-H*x_)]
            P     = (np.matrix(np.eye(P0.shape[0])) - K*H)*P_
            x_old = X_out[k]
            P_old = P
            
        #Fit the polynomial
        X_out = np.array(X_out)
        X     = np.array(X)
        p = []
        for k in range(X_out.shape[1]):
            p = p + [np.polyfit(np.squeeze(X_out[:,k]), X[k,:], self.degree)]
        params     = [A, W, H, Q, x0, P0, p]
        self.model = params
        
        
    def predict(self, Z_ts):
        #Extract model parameters
        A, W, H, Q, x0, P0, p = self.model
        #Apply the Kalman algorithm to the training set
        Z     = np.matrix(Z_ts.T)
        x_old = x0
        P_old = P0
        X_out = []
        for k in range(Z.shape[1]):
            z     = Z[:,k]
            x_    = A*x_old
            P_    = A*P_old*A.T + W
            K     = P_*H.T*inv(H*P_*H.T+Q)
            X_out = X_out + [x_+K*(z-H*x_)]
            P     = (np.matrix(np.eye(P0.shape[0])) - K*H)*P_
            x_old = X_out[k]
            P_old = P 
            
        #Apply the polynomial
        X_out     = np.array(X_out)
        X_predict = []
        for k in range(A.shape[0]):
            X_predict = X_predict + [np.polyval(p[k], np.squeeze(X_out[:,k]))]
        return np.array(X_predict).T
    
    

#################### LONG SHORT TERM MEMORY (LSTM) DECODER ####################
class LSTMDecoder():
    """
    Class for the gated recurrent unit (GRU) decoder
    Parameters
    ----------
    units      : integer, optional, default 400
        Number of hidden units in each layer
    dropout    : decimal, optional, default 0
        Proportion of units that get dropped out
    num_epochs : integer, optional, default 10
        Number of epochs used for training
    verbose    : binary, optional, default=0
        Whether to show progress of the fit after each epoch
    """
    def __init__(self, units = 128, dropout = 0, rec_dropout = 0, num_epochs = 20, lr = 0.001, batch_size = 64, poly = True, verbose=1):
         self.units       = units
         self.dropout     = dropout         
         self.rec_dropout = rec_dropout
         self.num_epochs  = num_epochs         
         self.lr          = lr
         self.batch_size  = batch_size
         self.verbose     = verbose
         self.poly        = poly

    def fit(self, X_train, y_train, validation_data = None, patience = 5):
        """
        Train LSTM Decoder
        Parameters
        ----------
        X_train : numpy 3d array of shape [n_samples,n_time_bins,n_neurons]
            This is the neural data.
            See example file for an example of how to format the neural data correctly
        y_train : numpy 2d array of shape [n_samples, n_outputs]
            This is the outputs that are being predicted
        """
        model = Sequential() #Declare model
        #Add recurrent layer
        model.add(LSTM(self.units,input_shape=(X_train.shape[1],X_train.shape[2]),
                       dropout=self.dropout,recurrent_dropout=self.rec_dropout)) #Within recurrent layer, include dropout
        if self.dropout!=0: 
            model.add(Dropout(self.dropout)) #Dropout some units (recurrent layer output units)
        #Add dense connections to output layer
        model.add(Dense(y_train.shape[1], activation='sigmoid'))
        #Fit model (and set fitting parameters)
        model.compile(loss='mae',optimizer=kopt.RMSprop(lr = self.lr),metrics=['accuracy']) #Set loss function and optimizer
        model.fit(X_train,y_train,epochs=self.num_epochs,batch_size=self.batch_size,
                  verbose=self.verbose,validation_data=validation_data,
                  callbacks=[callbacks.EarlyStopping(patience=patience)]) #Fit the model
        #Fit the polynomial
        y_out   = model.predict(X_train)
        y_out   = np.array(y_out)
        y_train = np.array(y_train)
        if self.poly:
            p = []
            for k in range(y_out.shape[1]):
                p = p + [np.polyfit(np.squeeze(y_out[:,k]), y_train[:,k], 5)]
            self.p = p
        self.model = model

    def predict(self, X_test):
        """
        Predict outcomes using trained LSTM Decoder
        Parameters
        ----------
        X_test : numpy 3d array of shape [n_samples,n_time_bins,n_neurons]
            This is the neural data being used to predict outputs.
        Returns
        -------
        y_test_predicted : numpy 2d array of shape [n_samples,n_outputs]
            The predicted outputs
        """
        y_out = self.model.predict(X_test) #Make predictions
        #Apply the polynomial
        y_out = np.array(y_out)
        if self.poly:
            y_predict = []
            for k in range(y_out.shape[1]):
                y_predict = y_predict + [np.polyval(self.p[k], np.squeeze(y_out[:,k]))]
            return np.array(y_predict).T
        else:
            return y_out
    
    

#####################################MLP#######################################
class MLP():
    def __init__(self, units = 30, verbose = 1):
         #If "units" is an integer, put it in the form of a vector
        try: #Check if it's a vector
            units[0]
        except: #If it's not a vector, create a vector of the number of units for each layer
            units=[units]
        self.units   = units
        self.verbose = verbose
        
        
    def fit(self, Z, X):
        model = Sequential()
        for i in range(len(self.units)):
            model.add(Dense(self.units[i], activation = 'relu'))
            
        model.add(Dense(X.shape[1]))        
        model.compile(loss = 'mae', optimizer = 'adam', metrics=['accuracy'])        
        model.fit(Z,X,verbose = self.verbose)        
        self.model = model
        
        
    def predict(self, Z_ts):        
        Z_out = self.model.predict(Z_ts)        
        return Z_out



############################### AUTO ENCODER ##################################
class AutoEncoder():

    def __init__(self, encoding_dimension = 10):
        self.ed  = encoding_dimension

    def fit(self, Z):
        Z_tr    = Z.reshape((len(Z), np.prod(Z.shape[1:])))
        inputae = Input(shape = (Z_tr.shape[1],))
        #Encoder layers
        encoded = Dense(4*self.ed*Z.shape[1], activation = 'relu')(inputae)
        encoded = Dense(2*self.ed*Z.shape[1], activation = 'relu')(encoded)
        encoded = Dense(self.ed*Z.shape[1], activation = 'relu')(encoded)
        #Decoder Layers
        decoded = Dense(2*self.ed*Z.shape[1], activation = 'relu')(encoded)
        decoded = Dense(4*self.ed*Z.shape[1], activation = 'relu')(decoded)
        decoded = Dense(Z_tr.shape[1], activation = 'sigmoid')(decoded)
        
        autoencoder = Model(inputae, decoded)
        autoencoder.summary()
        encoder     = Model(inputae, encoded)
        autoencoder.compile(optimizer = 'rmsprop', loss = 'mae')
        autoencoder.fit(Z_tr, Z_tr, epochs = 50, batch_size = 256, verbose = 1)
        self.encoder = encoder
        
    def predict(self, Z):
        Z_ts = Z.reshape((len(Z), np.prod(Z.shape[1:])))
        Z_en = self.encoder.predict(Z_ts)
        return Z_en.reshape((Z.shape[0],Z.shape[1],self.ed))
    


################################ BMI 2 ########################################
class BMI2():
    """
    BMI composed by an Autoencoder followed by a LSTM network and a linear filter
    """
    
    def __init__(self, input_shape, output_shape, encoding_dimension = 10, units = 128, dropout = 0, 
                 rec_dropout = 0, num_epochs_lstm = 20, lr = 0.001, batch_size = 128,
                 model_name = None, verbose=1):
        self.inp_sh      = input_shape #3D FR matrix shape
        self.out_sh      = output_shape #EMGbin matrix shape
        self.ed          = encoding_dimension
        self.units       = units
        self.dropout     = dropout         
        self.rec_dropout = rec_dropout
        self.nel         = num_epochs_lstm         
        self.lr          = lr
        self.batch_size  = batch_size
        self.verbose     = verbose
        self.model_name  = model_name #List: first element is the filepath of the Encoder, the second is the file path of the LSTM net
        
        if model_name is not None:
            self.encoder = load_model(model_name[0])
            self.lstm    = load_model(model_name[1])
        else:
            #Define AutoEncoder
            inputae = Input(shape = (input_shape[1],input_shape[2],))
            #Encoder layers
            encoded = Dense(4*self.ed, activation = 'elu')(inputae)
            encoded = BatchNormalization()(encoded)
            encoded = Dense(2*self.ed, activation = 'elu')(encoded)
            encoded = BatchNormalization()(encoded)
            encoded = Dense(self.ed, activation = 'elu')(encoded)
            #Decoder Layers
            decoded = Dense(2*self.ed, activation = 'elu')(encoded)
            decoded = BatchNormalization()(decoded)
            decoded = Dense(4*self.ed, activation = 'elu')(decoded)
            decoded = BatchNormalization()(decoded)
            decoded = Dense(input_shape[2])(decoded)
            
            self.autoencoder = Model(inputae, decoded)
            self.autoencoder.summary()
            self.encoder     = Model(inputae, encoded)
            self.autoencoder.compile(optimizer = 'rmsprop', loss = 'mae')
            
            #Define LSTM Network
            lstm = Sequential() #Declare model
            #Add recurrent layer
            lstm.add(LSTM(self.units,input_shape=(input_shape[1],encoding_dimension),
                          dropout=self.dropout,recurrent_dropout=self.rec_dropout,
                          kernel_regularizer=regularizers.l2(0.01))) #Within recurrent layer, include dropout
            if self.dropout!=0: lstm.add(Dropout(self.dropout)) #Dropout some units (recurrent layer output units)
            #Add dense connections to output layer
            lstm.add(Dense(output_shape[1]))
            #Set fitting parameters
            lstm.compile(loss='mae',optimizer=kopt.rmsprop(lr = self.lr),metrics=['accuracy']) #Set loss function and optimizer
            
            self.lstm = lstm
        
    def fit(self, Z, X):
        #Fit the AutoEncoder
        Z_tr = Z
        self.autoencoder.fit(Z_tr, Z_tr, epochs = 50, batch_size = 256, verbose = self.verbose)  
        #Get the latent dimension of the training set
        Z_latent = self.encoder.predict(Z_tr)
        #Fit the LSTM network
        self.lstm.fit(Z_latent,X,epochs=self.nel,batch_size=self.batch_size,
                      verbose=self.verbose) #Fit the model
        #Fit the polynomial
        X_out = self.lstm.predict(Z_latent)
        X_out = np.array(X_out)
        X     = np.array(X)
        p = []
        for k in range(X_out.shape[1]):
            p = p + [np.polyfit(np.squeeze(X_out[:,k]), X[:,k], 3)]
        self.p = p
        
        #Save models
        self.encoder.save('Encoder.h5')
        self.lstm.save('LSTM.h5')
        #Save autoencoder for ADAN initialization
        self.autoencoder.save('AE.h5')
        
    def predict(self, Z):
        Z_ts  = Z
        Z_en  = self.encoder.predict(Z_ts)
        X_out = self.lstm.predict(Z_en)
        #Apply the polynomial
        X_out     = np.array(X_out)
        X_predict = []
        for k in range(X_out.shape[1]):
            X_predict = X_predict + [np.polyval(self.p[k], np.squeeze(X_out[:,k]))]
        return np.array(X_predict).T
    
    

################# ADVERSARIAL DOMAIN ADAPTATION NETWORK 2 #####################
class ADAN2():
    def __init__(self, train_shape, test_shape, AEfilename = 'AE.h5', batch_size = 256):
        self.batch_size = batch_size
        self.tr_shape   = train_shape #3D FR matrix shape
        self.ts_shape   = test_shape  #3D FR matrix shape
        
        #Define the Aligner
#        al_input     = Input(shape = (test_shape[1],test_shape[2],))
#        z_hidden     = Dense(test_shape[2], activation = 'elu')(al_input)
#        z_hidden     = BatchNormalization()(z_hidden)
#        z_hidden     = Dense(test_shape[2], activation = 'elu')(z_hidden)
#        z_hidden     = BatchNormalization()(z_hidden)
#        z_hidden     = Dense(test_shape[2], activation = 'elu')(z_hidden)
#        z_hidden     = BatchNormalization()(z_hidden)
#        z_hidden     = Dense(test_shape[2], activation = 'elu')(z_hidden)
#        z_hidden     = BatchNormalization()(z_hidden)
#        z_hidden     = Dense(test_shape[2], activation = 'elu')(z_hidden)
#        z_hidden     = BatchNormalization()(z_hidden)
#        al_out       = Dense(test_shape[2])(z_hidden)
#        self.aligner = Model(al_input, al_out)
        self.aligner = load_model(AEfilename, compile = False)
        self.aligner.name = 'Aligner'
        self.aligner.summary()
        #Load the AE of the BMI as the Discriminator and replace its input layer
        self.discriminator = load_model(AEfilename)
        self.discriminator.name = 'Discriminator'
        self.discriminator.compile(optimizer = kopt.sgd(lr = 0.001), loss = self.loss_discriminator)
        self.discriminator.summary()
        #Define the ADAN
#        adan_out  = self.discriminator(self.aligner(al_input))
#        self.adan = Model(al_input, adan_out)
#        self.adan = Model(self.aligner,adan_out)
        self.adan = Sequential()
        self.adan.add(self.aligner)
        self.adan.add(self.discriminator)
        self.discriminator.trainable = False #When training the adan, train only the aligner part
        self.adan.compile(optimizer  = kopt.sgd(lr = 0.001), loss = self.loss_aligner)
        self.adan.summary()
        
    def loss_aligner(self, y_true, y_pred):
        muk = K.mean(K.abs(y_true - y_pred))
        return muk
#        muk = K.mean(K.sum(K.abs(y_true - y_pred),axis=0))
#        return muk
    
    def loss_discriminator(self, y_true, y_pred):
        ind0 = np.arange(0,self.batch_size,1)
        indk = np.arange(self.batch_size,self.batch_size*2,1)
        mu   = K.mean(K.abs(tf.subtract(y_true,y_pred)), axis = (1,2))
        mu0  = K.gather(mu,ind0)
        muk  = K.gather(mu,indk)
        mu0  = K.mean(mu0)
        muk  = K.mean(muk)
#        y0t = K.gather(y_true,ind0)
#        y0p = K.gather(y_pred,ind0)
#        ykt = K.gather(y_true,indk)
#        ykp = K.gather(y_pred,indk)
#        mu0 = K.mean(K.sum(K.abs(tf.subtract(y0t,y0p)),axis=0))
#        muk = K.mean(K.sum(K.abs(tf.subtract(ykt,ykp)),axis=0))
        return tf.subtract(mu0,muk)

    def fit(self, Z0, Zk, num_epochs = 1000):
        for epoch in range(num_epochs):
            print('Training epoch '+str(epoch+1)+' of '+str(num_epochs)+'...')
            #Form Z_tr_discr and Z_tr_adan
            #Select a random batch of data
            idx0  = np.random.randint(0, Z0.shape[0] - self.batch_size)
            idxk  = np.random.randint(0, Zk.shape[0] - self.batch_size)
            Z0tr = Z0[idx0:idx0+self.batch_size,:,:]
            Zktr = Zk[idxk:idxk+self.batch_size,:,:]
            Zk_al = self.aligner.predict(Zktr)
            Z_tr_discr = np.concatenate((Z0tr, Zk_al))
            Z_tr_adan  = Zktr
            #Train the discriminator
            lossd = self.discriminator.train_on_batch(Z_tr_discr, Z_tr_discr)
            #Train the aligner
            lossa = self.adan.train_on_batch(Z_tr_adan, Z_tr_adan)
            print('Loss Discriminator = '+str(lossd))
            print('Loss Aligner       = '+str(lossa)+'\n')
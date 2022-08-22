# -*- coding: utf-8 -*-
"""
File: nlpca.py
Author: Christian Camilo Rosero Rodriguez
Email: christian.camilo.rosero@correounivalle.edu.co
Version: 1.6

Description: NLPCA - nonlinear PCA - Nonlinear principal component analysis
             based on an autoassociative neural network 
            -Hierarchical nonlinear PCA (NLPCA) with standard bottleneck architecture

Reference: Scholz and Vigario. Proceedings ESANN, 2002
          www.nlpca.org
          Author: Matthias Scholz
"""

import os
import keras
import numpy as np
import pandas as pd

from scipy.sparse import issparse

import operator
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

from joblib import Parallel, delayed

try:
    import cPickle as pickle
except:
    import pickle

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils import check_random_state
from sklearn.utils.validation import check_array, assert_all_finite, check_is_fitted


import tensorflow as tf
from tensorflow import keras
from tensorflow import GradientTape
from tensorflow.keras import layers
from tensorflow.keras import backend as K
from tensorflow.keras import regularizers
from keras.wrappers.scikit_learn import KerasClassifier

class NLPCA(TransformerMixin, BaseEstimator):
  
  def __init__(self, n_components=None, *, max_iteration=100, batch=None, verbose=0, opti_algoritm='sgd',
               pre_pca=False,n_pre_pca=None,pre_unit_variance=False,units_per_Hidden_layer=None,
               weight_initialization='linear',weight_decay=True,weight_decay_coefficient=0.01,pre_scaling=True, 
               scaling_factor=None,function_activation_linear='linear', function_activation_nonlinear='tanh',
               random_state_1=None,random_state_2=None, callbacks=False, callbacks_path='/', n_jobs=None):
    """ nonlinear PCA
    # Arguments
      
      max_iteration                  - Maximum number of iterations from the network. Split in epochs of 100 
      batch                          - Integer or None. Number of samples per gradient update. 
                                       If unspecified, batch_size will default to 32. Do not 
                                       specify the batch_size if your data is in the form of 
                                       datasets, generators, or keras.utils.Sequence instances 
                                       (since they generate batches).
      verbose                        - 0, 1, or 2. Verbosity mode. 0 = silent partial/progress for 100 epoch, 
                                       1 = progress bar complet, 2 = one line per epoch. Note that the progress 
                                       bar is not particularly useful when logged to a file, so verbose=2 is 
                                       recommended when not running interactively (eg, in a production environment).
      opti_algoritm                  - String (name of optimizer) or optimizer instance from network.
                                       Default: 'adam'
      n_components                   - number of requested nonlinear components
      pre_pca                        - {'True','False'} default:'False', PCA preprocessing, the first n 
                                        components are used, n is the number of output units
      n_pre_pca                      - the first n components for pre_pca
                                        default: 0.1/max(std(data),[],2)
      pre_unit_variance              - {'True','False'} default:'False', unit variance normalization
      units_per_Hidden_layer         - number of neurons in each hidden layer, does not include the input,
                                        output and bottleneck layers.
                                        default: 1 layer with 2+(2*k) neurons, being k number of requested
                                        nonlinear components
      weight_initialization          - default: 'None' for weight initialization random
                                        alternative: 'linear' for weight initialization linear
      weight_decay                   - 'True': weight_decay is on (default)
                                      'False' : weight_decay is off
      weight_decay_coefficient       - value between 0 and 1, default: 0.01
      pre_scaling                    - True: limit the max std in the data set to keep the network 
                                        in the linear range at begin
                                        default: set std to 0.1, if no 'scaling_factor' is specified
      scaling_factor                 - 0.xx: Scaling multiplier
      
      function_activation_linear     - linear activation function of linear layers
                                        default: 'linear'
      function_activation_nonlinear  - nonlinear activation function of nonlinear layers
                                       default: 'tanh'
      random_state                   - Controls the shuffling applied to the data before applying the split
                                       arrays or matrices into random train and test subsets. Pass an int 
                                       for reproducible output across multiple function calls.
      callbacks                      - If is True used List of callbacks to apply during training
      callbacks_path                 - Path File of callbacks to apply during training
    """
    #initialization of input arguments and global variables
  
    self.n_components = n_components
    self.max_iteration = max_iteration
    self.batch = batch
    self.verbose = verbose
    self.opti_algoritm = opti_algoritm
    self.pre_pca = pre_pca
    self.n_pre_pca = n_pre_pca                     
    self.pre_unit_variance = pre_unit_variance
    self.units_per_Hidden_layer = units_per_Hidden_layer
    self.weight_initialization = weight_initialization
    self.weight_decay = weight_decay
    self.weight_decay_coefficient = weight_decay_coefficient
    self.pre_scaling = pre_scaling
    self.scaling_factor = scaling_factor
    self.function_activation_linear = function_activation_linear
    self.function_activation_nonlinear = function_activation_nonlinear
    self.random_state_1 = random_state_1
    self.random_state_2 = random_state_2 
    self.callbacks = callbacks
    self.callbacks_path = callbacks_path
    self.n_jobs = n_jobs
    
  def _more_tags(self):
        return {'non_deterministic': True}
  

  def _syntax_model(self,data_in,k,units_per_Hidden_layer=None):

    # here the model layers are created from the data dimension and the units per hidden layer

    input_dim = data_in.shape[1]

    if units_per_Hidden_layer:
      #No vacio
      if type(units_per_Hidden_layer) is int:
        units_per_Hidden_layer = [units_per_Hidden_layer]
      Hidden_layer = len(units_per_Hidden_layer)
    else:
      #Vacio
      units_per_Hidden_layer = list([2+(2*k)])
      Hidden_layer=1
    
    encoded_layer = list(np.zeros(Hidden_layer))
    decoded_layer = list(np.zeros(Hidden_layer))
    

    input_layer = keras.Input(shape=(input_dim,)) #Capa de Entrada

    for l in range(Hidden_layer):#Codificación
      if l==0:
        encoded_layer[l] = layers.Dense(units=units_per_Hidden_layer[l], activation=self.function_activation_nonlinear, kernel_regularizer=regularizers.l2(self.weight_decay_coefficient*0.5))(input_layer) #Capa de codificacción
      else:
        encoded_layer[l] = layers.Dense(units=units_per_Hidden_layer[l], activation=self.function_activation_nonlinear, kernel_regularizer=regularizers.l2(self.weight_decay_coefficient*0.5))(encoded_layer[l-1]) #Capas de codificacción concadenadas

    pca_layer = layers.Dense(units=k, activation=self.function_activation_linear, kernel_regularizer=regularizers.l2(self.weight_decay_coefficient*0.5))(encoded_layer[-1])#Capa de cuello de botella conectada a la ultima capa oculta del codificador
    subnet_layer = layers.Lambda(lambda x: x * self._mask)(pca_layer) #Capa de máscara conectada al capa de Cuello de Botella
    #subnet_layer = layers.pca_layer = layers.Dense(units=k, activation=self.function_activation_linear, kernel_regularizer=regularizers.l2(self.weight_decay_coefficient*0.5))(pca_layer)

    
    units_per_Hidden_layer.reverse()# Inversión de las unidades por capa
    
    for l in range(Hidden_layer):#Decodificacion
      if l==0:
        decoded_layer[l] = layers.Dense(units=units_per_Hidden_layer[l], activation=self.function_activation_nonlinear, kernel_regularizer=regularizers.l2(self.weight_decay_coefficient*0.5))(subnet_layer) #Capa de Decodificación conectada a la mascara de subred
      else:
        decoded_layer[l] = layers.Dense(units=units_per_Hidden_layer[l], activation=self.function_activation_nonlinear, kernel_regularizer=regularizers.l2(self.weight_decay_coefficient*0.5))(decoded_layer[l-1]) #Capas de Decodificacción concadenadas
    
    output_layer = layers.Dense(units=input_dim, activation=self.function_activation_linear, kernel_regularizer=regularizers.l2(self.weight_decay_coefficient*0.5))(decoded_layer[-1])#Capa de Salida conectada a la ultima capa oculta del Decodificador
    
    #Modelos de autorncoder y encoder_
    autoencoder_ = keras.Model(input_layer, output_layer)
    return autoencoder_

  def _set_weights_linear(self):

    # % w = set_weights_linear(tmp)  
    # % 
    # % set weights by random matrixes and there pseudo inverse matrices
    # % such that 
    # %   W{4}*W{3}*W{2}*W{1}=I
    # % and
    # %   x=W{4}*W{3}*W{2}*W{1}*x
    # %
    # % if the data x are scaled to a very small std like 0.1
    # % a nonlinear network is working in the linear range
    # % and hence it will give nearly the input as output
    # %
    # % - the unnecessary weights in the bottleneck-layer are pruned
    # % - bias weigths are added and set to zero 

    #NOTE:???
    #WARNING: !!!!! this can only be used with pca preprocessing !!!!??????? 

    net_layers = len(self._network.layers) # number of layers
    index_mask_layer = int((len(self._network.layers)/2))#index mask layer
    layer=[]
    for i in range(1, net_layers):
      if i == index_mask_layer:
        continue
      layer.append(self._network.get_layer(index=i))#layers
    
    weights = list(np.zeros(len(layer)))
    
    weights_rand = check_random_state(self.random_state_1)

    weights[0] = weights_rand.rand(self.data_train_.shape[1],layer[0].get_config().get('units'))-0.5 #firts weights lyer
    weights[-1] = np.linalg.pinv(weights[0]) #end weights layer

    for l in range(1, int((len(layer)/2))):

      weights[l] = np.linalg.pinv(weights[l-1])[:,0:layer[l].get_config().get('units')] 
      weights[-(l+1)] = np.linalg.pinv(weights[l])

    #normalize frobenius layers encoder_
    for l in range(int((len(layer)/2)-1),0,-1):
      c=(np.linalg.norm(weights[l], ord='fro')**2/np.linalg.norm(weights[l-1], ord='fro')**2)**(1/4)
      weights[l-1] = c*weights[l-1]
      weights[l] = 1/c*weights[l]

    #normalize frobenius layers decoder_
    for l in range(int((len(layer))-1),int(len(layer)/2),-1):
      c=(np.linalg.norm(weights[l], ord='fro')**2/np.linalg.norm(weights[l-1], ord='fro')**2)**(1/4)
      weights[l-1] = c*weights[l-1]
      weights[l] = 1/c*weights[l]

    #normalize frobenius layers firts an ending
    c=(np.linalg.norm(weights[-1], ord='fro')**2/np.linalg.norm(weights[0], ord='fro')**2)**(1/4)
    for l in range(int(len(layer)/2)):
      weights[l] = c*weights[l]
    for l in range(int(len(layer)/2),len(layer)):
      weights[l] = 1/c*weights[l]

    # c=(np.linalg.norm(weights_2, ord='fro')**2/np.linalg.norm(weights_1, ord='fro')**2)**(1/4)
    # weights_1 = c*weights_1
    # weights_2 = 1/c*weights_2
    # c=(np.linalg.norm(weights_4, ord='fro')**2/np.linalg.norm(weights_3, ord='fro')**2)**(1/4)
    # weights_3 = c*weights_3
    # weights_4 = 1/c*weights_4
    # c=(np.linalg.norm(weights_4, ord='fro')**2/np.linalg.norm(weights_1, ord='fro')**2)**(1/4)
    # weights_1 = c*weights_1
    # weights_2 = c*weights_2
    # weights_3 = 1/c*weights_3
    # weights_4 = 1/c*weights_4

    
    #Asigned weights
    for l in range(len(layer)):
      value = layer[l].get_weights()
      value[0] = weights[l]
      layer[l].set_weights(value)


    # value = layer_1.get_weights()
    # value[0] = weights_1
    # layer_1.set_weights(value)
    # value = layer_2.get_weights()
    # value[0] = weights_2
    # layer_2.set_weights(value)
    # value = layer_3.get_weights()
    # value[0] = weights_3
    # layer_3.set_weights(value)
    # value = layer_4.get_weights()
    # value[0] = weights_4
    # layer_4.set_weights(value)

  def _hierarchic_idx(self,k):
    #k = int(k)
    #Crear la mascara para las subredes
    #Example (4 components): idx =
    #
    #     1     1     1     1     0
    #     0     1     1     1     1
    #     0     0     1     1     1
    #     0     0     0     1     1
    #k=4
    idx=np.zeros((k,(k+1)))

    for i in range(k):
      idx[0:i+1,i] = 1

    if k>1:
      idx[1:k,-1] = 1 # for zero mean in component one
    
    return idx

  def _loss(self, model, x, y, training):
    # training=training is needed only if there are layers with different
    # behavior during training versus inference (e.g. Dropout).
    loss_object = keras.losses.MSE
    y_ = model(x, training=training)
    E = K.sum(loss_object(y_true=y, y_pred=y_))
    #print(E)
    return E

  def _hierarchical_error(self,input_layer, output_layer):
    Ex=[]
    for i in range(self.n_components_+1):
      newValue=np.array([self._idx[:,i]])
      K.set_value(self._mask,newValue)
      E = self._loss(self._network, self.data_train_, self.data_train_, training=False)
      Ex.append(E)

    # Eh=np.array(Ex)
    # Eh=Eh*self._Coeficientes

    # Eh0 = tf.concat(Ex, axis=0)
    # Etotal0 = sum(Eh0)
    
    Eh = tf.stack(Ex, axis=0)
    Eh=Eh*self._Coeficientes
    Etotal = tf.reduce_sum(Eh, 0)

    # print("Etotal")
    # print(Etotal0)
    # print(Etotal)
    return Etotal

  def _sort_component(self):

    E=np.zeros(self.n_components_)
    newValue_mask = np.array([np.append(1, np.zeros(self.n_components_-1))])
    for i in range(self.n_components_):
      K.set_value(self._mask,newValue_mask)
      Epattern = self._loss(self._network, self.data_train_, self.data_train_, training=False)
      #print(Epattern)
      E[i]=Epattern
      newValue_mask = np.roll(newValue_mask, 1) 

    index_pca_layer = int((len(self._network.layers)/2)-1)
    layer_pca = self._network.get_layer(index=index_pca_layer)
    weights_pca = layer_pca.get_weights()[0]
    bias_pca = layer_pca.get_weights()[1]
    weights_pca = weights_pca.T
    value_pca = layer_pca.get_weights()

    E_component = dict(zip(range(1,self.n_components_+1),zip(E,zip(weights_pca, bias_pca))))
    E_component_sort = sorted(E_component.items(), key=operator.itemgetter(0,1))
    
    new_weights_pca = []
    for e in range(self.n_components_):
      new_weights_pca.append(E_component_sort[e][1][1][0])
    new_weights_pca = np.array(new_weights_pca).T

    new_bias_pca = []
    for e in range(self.n_components_):
      new_bias_pca.append(E_component_sort[e][1][1][1])
    new_bias_pca = np.array(new_bias_pca)

    new_value_pca =[]
    new_value_pca.append(new_weights_pca)
    new_value_pca.append(new_bias_pca)
    layer_pca.set_weights(new_value_pca)
  
  def _train(self):

    self._histories_loss=[]
    self._histories_val_loss=[]
    self._histories_cosine_similarity=[]
    self._histories_val_cosine_similarity=[]

    self.data_train_ = shuffle(self.data_train_, random_state=0)

    tf.executing_eagerly()
    
    self._network.compile(optimizer=self.opti_algoritm, loss=self._hierarchical_error, metrics='cosine_similarity', run_eagerly=True)

    checkpoint_path = self.callbacks_path + "training/cp-{epoch:04d}.ckpt"
    checkpoint_dir = os.path.dirname(checkpoint_path)

    # Create a callback that saves the model's weights every 100 epochs
    if self.callbacks == True:
      cp_callback = keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                      save_weights_only=True,
                                                      verbose=1,
                                                      period=100)
      callback_cp = [cp_callback]

      # Save the weights using the `checkpoint_path` format
      self._network.save_weights(checkpoint_path.format(epoch=0))

    else:
      callback_cp = None

    
    shuffle_seed = 10
    self.data_train_ = tf.random.shuffle(self.data_train_, seed=shuffle_seed)
    
    for e in range(int(self.max_iteration/100)):
      #Entrenamiento
      if e == 0:
        print("Epoch nlpca {:d}/{:d}\n 10/10 [==============================]".format((e)*100,self.max_iteration))
        if self.callbacks == True:
          latest = tf.train.latest_checkpoint(checkpoint_dir)
          self._network.load_weights(latest)

      
      history = self._network.fit(self.data_train_, self.data_train_,
                          epochs=10,
                          batch_size=self.batch,
                          shuffle=True,
                          validation_split=0.1,
                          verbose = self.verbose, callbacks=callback_cp)
    
      print("Epoch {:d}/{:d}\n 10/10 [==============================] - loss: {:.5f} - similarity: {:.2f}% - val_loss: {:.5f} - val_similarity: {:.2f}%".format((e+1)*100,self.max_iteration,history.history['loss'][-1],history.history['cosine_similarity'][-1]*100,history.history['val_loss'][-1],history.history['val_cosine_similarity'][-1]*100))

      self._sort_component()

      self._histories_loss.extend(history.history['loss'])
      self._histories_val_loss.extend(history.history['val_loss'])
      self._histories_cosine_similarity.extend(history.history['cosine_similarity'])
      self._histories_val_cosine_similarity.extend(history.history['val_cosine_similarity'])

      if self.callbacks == True:
        latest = tf.train.latest_checkpoint(checkpoint_dir)

    self.plot_history()
      
  def plot_history(self):
    #Plot del costo 
    plt.plot(self._histories_loss)
    plt.plot(self._histories_val_loss)
    plt.title('model train vs validation loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper right')
    plt.show()

    plt.plot(self._histories_cosine_similarity)
    plt.plot(self._histories_val_cosine_similarity)
    plt.title('model train vs validation similitary')
    plt.ylabel('similitary')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper right')
    plt.show()

  def _validate_inputs(self, X):
        # Things we don't want to allow until we've tested them:
        # - Sparse inputs
        # - Multiclass outputs (e.g., more than 2 classes in `y`)
        # - Non-finite inputs
        # - Complex inputs

        if isinstance(X, pd.DataFrame):
          X = X.to_numpy()

        X = check_array(X, accept_sparse=False, allow_nd=False)

        assert_all_finite(X)

        if np.any(np.iscomplex(X)):
            raise ValueError("Complex data not supported")
        if np.issubdtype(X.dtype, np.object_):
            try:
                X = X.astype(float)
            except (TypeError, ValueError):
                raise ValueError("argument must be a string.* number")

        return (X)
  
  def fit_transform(self, X, y=None, **params):
        """Fit the model from data in X and transform X.
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training vector, where n_samples in the number of samples
            and n_features is the number of features.
        Returns
        -------
        X_new : array-like, shape (n_samples, n_components)
        """
        self.fit(X,y)

        # no need to use the kernel to transform X, use shortcut expression
        X_transformed = self.transform(X)

        return X_transformed


  def transform(self,X, y=None):
    """Transform X.
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
        Returns
        -------
        X_new : array-like, shape (n_samples, n_components)
        """

    check_is_fitted(self)

    _encoder = tf.keras.Sequential.from_config(self.config_encoder_)
    _encoder.set_weights(self.weights_encoder_)

    data_get = self._validate_inputs(X)
    if self.pre_pca == True:
      data_get = self.data_pca.transform(data_get)
    if self.pre_unit_variance == True:
      data_get = self.scaler.transform(data_get)
    if self.pre_scaling == True:
      data_get = data_get * self.scaling_factor_ 
    pc_get = _encoder.predict(data_get) #extracting components 'pc' from new o train data
    return pc_get
    
  def inverse_transform(self,X):
    """Transform X back to original space.
        Parameters
        ----------
        X : array-like, shape (n_samples, n_components)
        Returns
        -------
        X_new : array-like, shape (n_samples, n_features)"""

    #X = self._validate_inputs(X)

    _decoder = tf.keras.Sequential.from_config(self.config_decoder_)
    _decoder.set_weights(self.weights_decoder_)

    data_get = _decoder.predict(X)
    if self.pre_scaling == True:
      data_get = data_get / self.scaling_factor_
    if self.pre_unit_variance == True:
      data_get = self.scaler.inverse_transform(data_get)
    if self.pre_pca == True:
      data_get = self.data_pca.inverse_transform(data_get)
    return  data_get #generating data from new component values 'pc' o reconstruction of train data

  def fit(self,X,y=None):

    """Fit the model from data in X.
        Parameters
        ----------
        
        X : array-like, shape (n_samples, n_features)
            Training vector, where n_samples in the number of samples
            and n_features is the number of features.
            columns are variables/attributes (e.g. genes) and 
            rows are samples/observations (e.g. conditions)
        Returns
        -------
        self : object
            Returns the instance itself.
        """
    if issparse(X):
      raise TypeError('NLPCA does not support sparse input. See ')
    
    X = self._validate_inputs(X)

    if self.n_components is None:
      self.n_components_ = min(np.array(X).shape) - 1

    else:
      self.n_components_ = abs(self.n_components)
    
    
    self.data_train_, self.data_test_ = train_test_split(X, test_size=0.25, random_state=self.random_state_2)
    self.data_train_ = np.array(self.data_train_)
    self.data_test_ = np.array(self.data_test_)
    
    
    #prepare data

    if self.pre_pca == True:
      self.data_pca = PCA(n_components=self.n_pre_pca)
      self.data_pca.fit(self.data_train_)
      self.data_train_ = self.data_pca.transform(self.data_train_)
  
    if self.pre_unit_variance == True:
      self.scaler = StandardScaler()
      self.scaler.fit(self.data_train_)
      self.data_train_ = self.scaler.transform(self.data_train_)

    if self.pre_scaling == True:
      if self.scaling_factor is None:
        self.scaling_factor_ = 0.1/max(np.std(self.data_train_, axis=1))
      else:
        self.scaling_factor_ = self.scaling_factor
      self.data_train_ = (self.data_train_) * (self.scaling_factor_)

    #Global variables for subnets
    self._idx = self._hierarchic_idx(self.n_components_)# idx como mascara para apagar las neuronas del cuello de botella
    self._mask = K.variable([np.ones(self.n_components_)]) #create a var with length k and 2D shape - mascara inicial
    self._Coeficientes=np.append(np.ones(self.n_components_),0.01)

    if self.weight_decay == False:
      self.weight_decay_coefficient=0
    
    self._network = self._syntax_model(self.data_train_, self.n_components_, self.units_per_Hidden_layer)

    
    if self.weight_initialization == 'linear':
      self._set_weights_linear()


    if self.n_jobs != None:
      Parallel(n_jobs=self.n_jobs)(delayed(self._train)() for _ in range(self.n_jobs))
    else:
      self._train()
    #self._train()

    net_layers = len(self._network.layers) # number of layers
    index_mask_layer = int((len(self._network.layers)/2))#index mask layer
    layer=[]
    for i in range(1, net_layers):
      if i == index_mask_layer:
        continue
      layer.append(self._network.get_layer(index=i))#layers
    
    self.encoder_ = keras.Sequential()
    self.decoder_ = keras.Sequential()
    self.network_ = keras.Sequential()
    
    #self._network.summary()
    #encoder to encode encoded data (PCA) from training data
    input_dim = self.data_train_.shape[1]
    encoder_input = keras.Input(shape=(input_dim,))
    self.encoder_.add(encoder_input)
    for l in range(int((len(layer)/2))):
      self.encoder_.add(self._network.layers[l+1])
    #self.encoder_.summary()

    #decoder_ to restore original data (PCA) from pca data
    decoder_input = keras.Input(shape=(self.n_components_,))
    self.decoder_.add(decoder_input)
    for l in range(int(len(layer)/2)+1,len(layer)+1):
      self.decoder_.add(self._network.layers[l+1])
    #self.decoder_.summary()
    
    #Create new network from save
    self.network_.add(encoder_input)
    for l in range(len(layer)+1):
      if l == int((len(layer)/2)):
        continue 
      self.network_.add(self._network.layers[l+1])
    #self.network_.summary()

    self.config_network_ = self.network_.get_config()
    self.config_encoder_ = self.encoder_.get_config()
    self.config_decoder_ = self.decoder_.get_config()

    self.weights_network_ = self.network_.get_weights()
    self.weights_encoder_ = self.encoder_.get_weights()
    self.weights_decoder_ = self.decoder_.get_weights()

    self.get_variance(self.data_test_)

    del self._network
    del self.encoder_
    del self.decoder_
    del self.network_
    # del self._idx
    # del self._mask 
    # del net_layers
    # del index_mask_layer
    # del layer

    return self


  def get_variance(self, X):
    # Estimate explained variance of each nonlinear component

    total_variance=sum(np.var(X, axis=0))
    #print(total_variance)

    _pc=self.transform(X)
    _evals=np.zeros(self.n_components_)

    data_recon_total=self.inverse_transform(_pc)
    self.explained_variance_=sum(np.var(data_recon_total, axis=0))
    #print(self.explained_variance_)

    percentVar_recon=(self.explained_variance_/total_variance)*100;
    percentVar_recon=(np.round(percentVar_recon*100))/100
    print('Total Explained variance for nonlinear PC:',percentVar_recon,'%')

    for i in range(self.n_components_):
      _pcx=np.zeros(_pc.shape)
      _pcx[:,i] = _pc[:,i] #only PC_i, set remaining PC's to zero
      data_recon=self.inverse_transform(_pcx)
      _evals[i]=sum(np.var(data_recon, axis=0)) 

    #print(evals)
    self.explained_variance_ratio_=(_evals/total_variance)*100;
    #print(percentVar)
    self.explained_variance_ratio_=(np.round(self.explained_variance_ratio_*100))/100
    #print(percentVar)
    print('Explained variance (see: net.variance)\n')
    for i in range(self.n_components_):
      print('nonlinear PC ',i+1,': ',self.explained_variance_ratio_[i],'%')

  def save(self, path_name, mode=None):

    _network = tf.keras.Sequential.from_config(self.config_network_)
    _network.set_weights(self.weights_network_)

    if mode == None:
      #!mkdir -p saved_model
      _network.save(path_name)
    if mode == 'h5':
      _network.save(path_name + '.h5')
    if mode == 'SavedModel':
      keras.experimental.export_saved_model(_network, path_name)
    if mode == 'Pickle':
      with open(path_name + '.pkl', 'wb') as fid:
        pickle.dump(gnb, fid)

  def load(self, path_name, mode=None):
    
    if mode == None:
      _network = keras.models.load_model(path_name)
    if mode == 'h5':
      _network = keras.models.load_model(path_name + '.h5')
    if mode == 'SavedModel':
      _network = keras.experimental.load_from_saved_model(path_name)
    if mode == 'Pickle':
      with open(path_name + '.pkl', 'rb') as fid:
        _network = pickle.load(fid)

    layer = len(_network.layers)

    # self.network_.summary()
    # print(layer)
    
    self.encoder_ = keras.Sequential()
    input_dim = self.data_train_.shape[1]
    encoder_input = keras.Input(shape=(input_dim,))
    self.encoder_.add(encoder_input)
    for l in range(int(layer/2)):
      self.encoder_.add(_network.layers[l])
    
    #self.encoder_.summary()

    self.decoder_ = keras.Sequential()
    decoder_input = keras.Input(shape=(self.n_components_,))
    self.decoder_.add(decoder_input)
    for l in range(int(layer/2),layer):
      self.decoder_.add(_network.layers[l])
      
    #self.decoder_.summary()

    self.config_network_ = self.network_.get_config()
    self.config_encoder_ = self.encoder_.get_config()
    self.config_decoder_ = self.decoder_.get_config()

    self.weights_network_ = self.network_.get_weights()
    self.weights_encoder_ = self.encoder_.get_weights()
    self.weights_decoder_ = self.decoder_.get_weights()
  
  # def config(self, net=None):
  #   self.config_network_ = net.get_config()

  def set_config(self, config):
    self.config_network_ = config

  # def weights(self, net=None):
  #   self.weights_network_ = net.get_weights()

  def set_weights(self, weights):
    self.weights_network_ = weights

  # def histories(self):
  #   return self.histories_loss, self.histories_val_loss, self.histories_cosine_similarity, self.histories_val_cosine_similarity
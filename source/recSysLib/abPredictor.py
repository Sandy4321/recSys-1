#Author: Stefano Cereda
import theano.tensor as T
import theano
import numpy as np
from lasagne.layers import InputLayer, GaussianNoiseLayer, DenseLayer
from lasagne.regularization import l2, regularize_network_params
import lasagne
import scipy
from sklearn.model_selection import train_test_split
from netflix_reader import  NetflixReader
import joblib
import random
import time
from theano import sparse
import pickle

USE_LINEAR_MODEL = True
USE_ENTIRE_FEATURES = True
NUM_HIDDEN_UNITS = 30
NUM_FEATURES = 9
if USE_ENTIRE_FEATURES:
    NUM_FEATURES = 4701

GAUSSIAN_NOISE_SIGMA = 0.
LEARNING_RATE = 0.001
L2_LAMBDA = 0.1

NUM_EPOCHS = 20
BATCH_SIZE = 1000
VAL_PERCENTAGE = 0.1
RND_NULL_SIM = 0.02 #percentage of null similarities to add

BASE_FILE = "../../datasources/ab/"
if USE_ENTIRE_FEATURES:
    BASE_FILE = "../../datasources/ab_2/"
AB_FILE_X_TRAIN = BASE_FILE + "X_train.pkl"
AB_FILE_X_VAL = BASE_FILE + "X_val.pkl"
AB_FILE_Y_TRAIN = BASE_FILE + "Y_train.npy"
AB_FILE_Y_VAL = BASE_FILE + "Y_val.npy"
FILE = BASE_FILE + "ab_model.npz"
SLIM_FILE = "../../datasources/slimW_0.1_10.npz"


class abPredictor:
    def __init__(self):
        #define network
        self._input_var = T.matrix('inputs')
        self._target_var = T.vector('targets')
        self._sigma = theano.shared(np.float32(GAUSSIAN_NOISE_SIGMA))
        print("Creating network")
        if USE_LINEAR_MODEL:
            self._net = self._define_ab_network_linear(self._input_var, NUM_FEATURES, self._sigma)
        else:
            self._net = self._define_ab_network_mlp(self._input_var, NUM_FEATURES, self._sigma, NUM_HIDDEN_UNITS)
        
        #load old parameters to continue training
        try:
            with np.load(FILE) as f:
                param_values = [f['arr_%d' % i] for i in range(len(f.files))]
            lasagne.layers.set_all_param_values(self._net, param_values)
        except:
            print("Old parameters not found, starting from scratch")
        
        #compile functions
        self._lr = theano.shared(np.float32(LEARNING_RATE))
        self._l2 = theano.shared(np.float32(L2_LAMBDA))
        print("Creating functions")
        (self._train_fn, self._val_fn, self._predict_fn) = self._create_learning_functions(self._net, self._input_var,
                                                                         self._target_var, self._lr, self._l2)
        
        #load the dataset
        print("Loading data")
        try:
            with open(AB_FILE_X_TRAIN, 'rb') as infile:
                self._X_train = pickle.load(infile).astype(np.float32)
            with open(AB_FILE_X_VAL, 'rb') as infile:
                self._X_val = pickle.load(infile).astype(np.float32)
            self._y_train = np.load(AB_FILE_Y_TRAIN).flatten().astype(np.float32)
            self._y_val = np.load(AB_FILE_Y_VAL).flatten().astype(np.float32)
            self._print_data_dim()
        except:
            print("Data not found. Creating dataset")
            self._create_dataset()
            with open(AB_FILE_X_TRAIN, 'rb') as infile:
                self._X_train = pickle.load(infile).astype(np.float32)
            with open(AB_FILE_X_VAL, 'rb') as infile:
                self._X_val = pickle.load(infile).astype(np.float32)
            self._y_train = np.load(AB_FILE_Y_TRAIN).flatten().astype(np.float32)
            self._y_val = np.load(AB_FILE_Y_VAL).flatten().astype(np.float32)
            self._print_data_dim()


   ###PRINT FIMENSION OF DATA VECTORS###
    def _print_data_dim(self):
        print("X Train: " + str(self._X_train.shape))
        print("y Train: " + str(self._y_train.shape))
        print("X Validation: " + str(self._X_val.shape))
        print("y Validation: " + str(self._y_val.shape))

        
        
    
    ###NETWORK DEFINITION###
    def _define_ab_network_linear(self, input_var, num_features, sigma):
        net = {}
        net['input'] = InputLayer(shape=(BATCH_SIZE, num_features), input_var=input_var)
        net['noise'] = GaussianNoiseLayer(net['input'], sigma=sigma)
        net['out'] = DenseLayer(net['noise'], num_units=1, nonlinearity=lasagne.nonlinearities.linear, W=lasagne.init.Constant(0.0))
        return net['out']
    
    def _define_ab_network_mlp(self, input_var, num_features, sigma, num_hidden):
        net = {}
        net ['input'] = InputLayer(shape=(None, num_features), input_var=input_var)
        net['noise'] = GaussianNoiseLayer(net['input'], sigma=sigma)
        net['hidden'] = DenseLayer(net['noise'], num_units=num_hidden, nonlinearity=lasagne.nonlinearities.sigmoid)
        net['out'] = DenseLayer(net['hidden'], num_units=1, nonlinearity=lasagne.nonlinearities.rectify)
        return net['out']

    
    
    ###COMPILE LEARNNG FUNCTIONS###
    def _create_learning_functions(self, network, input_var, target_var, lr, l2_lambda):
        params = lasagne.layers.get_all_params(network, trainable=True)
        
        out = lasagne.layers.get_output(network)
        test_out = lasagne.layers.get_output(network, deterministic=True)
        
        loss = lasagne.objectives.squared_error(out, target_var)
        l2_loss = l2_lambda * regularize_network_params(network, l2)
        loss = loss.mean() + l2_loss
        
        test_loss = lasagne.objectives.squared_error(test_out, target_var)
        test_loss = test_loss.mean() + l2_loss
        
        updates = lasagne.updates.adam(loss, params, lr)
        
        train_fn = theano.function([input_var, target_var], [loss, l2_loss], updates=updates)
        val_fn = theano.function([input_var, target_var], [test_loss, l2_loss])
        predict_fn = theano.function([input_var], test_out)
        
        return (train_fn, val_fn, predict_fn)
            
            
    ###TRAIN THE NETWORK###
    def _train_network(self, network, train_fn, val_fn, X_train, y_train, X_val, y_val, num_epochs, verbose = False, save_all=True):        
        # Launch the training loop.
        # We iterate over epochs:
        for epoch in range(num_epochs):        
            # In each epoch, we do a full pass over the training data:
            start_time = time.time()
            train_err = train_err_l2 = train_batches = 0
            for batch in self._iterate_minibatches(X_train, y_train, BATCH_SIZE, shuffle= True):
                inputs, targets = batch
                e,e2 = train_fn(inputs.toarray().astype(np.float32), targets)
                train_err += e
                train_err_l2 += e2
                train_batches += 1
            train_err /= train_batches
            train_err_l2 /= train_batches
            
           # And a full pass over the validation data:
            val_err = val_err_l2 = val_batches = 0
            for batch in self._iterate_minibatches(X_val, y_val, BATCH_SIZE, shuffle = False):
                inputs, targets = batch
                e,e2 = val_fn(inputs.toarray().astype(np.float32), targets)
                val_err += e
                val_err_l2 += e2
                val_batches += 1
            val_err_l2 /= val_batches
            val_err /= val_batches

            # Then we print the results for this epoch:
            if verbose:
                print("Epoch {} of {} took {:.3f}s".format(
                    epoch + 1, num_epochs, time.time() - start_time))
                print("  training loss:\t\t{}".format(train_err))
                print("  validation loss:\t\t{}".format(val_err))
                print("  train/val loss:\t\t{}".format(train_err/val_err))
                
            #And save the partial model
            if save_all:
                FILE_TMP = FILE + "_partial/" + str(epoch) + ".npz"
                np.savez(FILE_TMP, *lasagne.layers.get_all_param_values(network))
        
        return network, train_err, val_err, train_err_l2, val_err_l2
    
    
    
    def _iterate_minibatches(self, i, t, batchsize, shuffle=False):        
        if shuffle:
            indices = np.arange(i.shape[0])
            np.random.shuffle(indices)
            
        for start_idx in range(0, i.shape[0] - batchsize + 1, batchsize):
            if shuffle:
                excerpt = indices[start_idx:start_idx + batchsize]
            else:
                excerpt = slice(start_idx, start_idx + batchsize)
            yield i[excerpt], t[excerpt]
            
            
        
    ###EXPLORE HYPERPARAMETERS###
    def explore_hyperparameters(self):
        network = self._net
        X_train = self._X_train
        y_train = self._y_train
        X_val = self._X_val
        y_val = self._y_val

        #save starting parameters
        params = lasagne.layers.get_all_param_values(network)

        #get initial values
        val_err = val_err_l2 = val_batches = 0
        for batch in self._iterate_minibatches(X_val, y_val, BATCH_SIZE, shuffle = False):
            inputs, targets = batch
            e,e2 = self._val_fn(inputs.toarray().astype(np.float32), targets)
            val_err += e
            val_err_l2 += e2
            val_batches += 1
        val_err_l2 /= val_batches
        val_err /= val_batches
        print("Initial  valid. loss:\t\t%.8f" %(val_err))


        print("LR\t\tL2\t\tGS\t\tTrain Loss\tVal Loss\tTrain - l2\tVal - l2")
        for i in range(0,11):
            #self._lr.set_value(10**i)#np.float32(random.uniform(0.001, 0.0001)))
            #self._l2.set_value(10**i)#np.float32(random.uniform(0.00000001, 0.00000100)))
            self._sigma.set_value(i/10)#np.float32(random.uniform(0.,0.05)))
            
            network, train_loss, val_loss, t_l2, v_l2 = self._train_network(network, self._train_fn, self._val_fn,
                                                           X_train, y_train, X_val, y_val,
                                                           NUM_EPOCHS, verbose = False, save_all=True)
            
            print("%.8f\t%.8f\t%.8f\t%.8f\t%.8f\t%.8f\t%.8f" % (self._lr.get_value(), self._l2.get_value(), self._sigma.get_value(), train_loss, val_loss, train_loss-t_l2, val_loss-v_l2))
        
            #save the models	
            FILE_TMP = FILE + "_partial/" + str(i) + ".npz"
            np.savez(FILE_TMP, *lasagne.layers.get_all_param_values(network))

            #reset the network
            lasagne.layers.set_all_param_values(network, params)
        
        
        
    ###FIT THE NETWORK###
    def fit_network(self):
        network = self._net
        X_train = self._X_train
        y_train = self._y_train
        X_val = self._X_val
        y_val = self._y_val
        network = self._train_network(network, self._train_fn, self._val_fn, X_train, y_train, X_val, y_val, NUM_EPOCHS, verbose=True, save_all=True)[0]
        np.savez(FILE, *lasagne.layers.get_all_param_values(network))
        
        
        
    ###CREATE THE DATASET###
    def _create_dataset(self):
        #load the slim matrix and the netflix dataset
        weight_matrix = joblib.load(SLIM_FILE)
        num_items = weight_matrix.shape[0]
        print("Loaded weight matrix (y)")
        rdr = NetflixReader()
        print("Loaded products matrix (x)")
        
        #get all the couple of items with a non zero similarity
        idx = weight_matrix.nonzero()
        print("We have %d couples of items to compute:" %(len(idx[0])))

        Xs = list()
        ys = list()
        counter = 0
        
        if USE_ENTIRE_FEATURES:
            s = rdr._similarity_features
        else:
            s = rdr._similarity

        for (i1,i2) in zip(idx[0], idx[1]):
            #the weight matrix is not sym. We just insert two times the same Xs with different ys
            #which is equal to optimize for the mean of the two
            #TODO
            #WARNING USING _SIMILARITY INSTEAD OF _GET_SIMILARITY
            x = s(i1,i2).toarray().flatten()
            y = weight_matrix[i1,i2]
            
            Xs.append(x)
            ys.append([y])

            counter +=1
            if (counter % 10000 == 0):
                print(counter)
        
        #Now randomly add some couples with null similarity
        for i1 in range(num_items):
            print(i1)
            for i2 in range(i1+1, num_items):
                if weight_matrix[i1,i2] != 0:
                    continue
                if weight_matrix[i2,i1] != 0:
                    continue
                if random.random() < RND_NULL_SIM:
                    x = s(i1,i2).toarray().flatten()
                    y = 0.0
                    Xs.append(x)
                    ys.append(y)

 
        #take out some samples for validation
        X_train, X_val, y_train, y_val= train_test_split(Xs, ys, test_size=VAL_PERCENTAGE, random_state=1)
        
        with open(AB_FILE_X_TRAIN, 'wb') as outfile:
            pickle.dump(scipy.sparse.csc_matrix(X_train, dtype=np.float32), outfile, pickle.HIGHEST_PROTOCOL)
        with open(AB_FILE_X_VAL, 'wb') as outfile:
            pickle.dump(scipy.sparse.csc_matrix(X_val, dtype=np.float32), outfile, pickle.HIGHEST_PROTOCOL)
        np.save(AB_FILE_Y_TRAIN, np.array(y_train))
        np.save(AB_FILE_Y_VAL, np.array(y_val))
        
        
        
    ###COMPUTE THE SIMILARITY MATRIX USING THE NETWORK###
    #TODO
    def _compute_sim_matrix(self, list_of_items):
        rdr = NetflixReader()        
        weight_matrix = joblib.load(SLIM_FILE)
        n_items = weight_matrix.shape[0]
        matrix = np.zeros((n_items,n_items), dtype=np.float32)
        
        print("Computing products")
        products = list()
        indices = list()
        for movieId1 in list_of_items:
            for movieId2 in list_of_items:
                if movieId2 <= movieId1:
                    continue
                products.append(rdr._similarity_features(movieId1, movieId2).toarray().flatten().astype(np.float32))
                indices.append((movieId1, movieId2))
        
        similarities = self._predict_fn(np.array(products))
        
        for (idx, sim) in zip(indices, similarities):
            matrix[idx[0], idx[1]] = sim
            matrix[idx[1], idx[0]] = sim
        
        return scipy.sparse.csc_matrix(matrix)
        
    
if __name__ == '__main__':
    a = abPredictor()
    #a.fit_network()

    b = a._compute_sim_matrix(range(0,10))
    print(b)
    print(np.sum(np.abs(b)))

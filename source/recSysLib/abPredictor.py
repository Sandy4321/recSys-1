#Author: Stefano Cereda
import theano.tensor as T
import theano
import numpy as np
from lasagne.layers import InputLayer, GaussianNoiseLayer, DenseLayer
from lasagne.regularization import l2, regularize_network_params
import lasagne
import scipy
from netflix_reader import  NetflixReader
import multiprocessing


NUM_FEATURES = 9
GAUSSIAN_NOISE_SIGMA = 0.
LEARNING_RATE = 0.001
L2_LAMBDA = 0.1

NUM_EPOCHS = 1000

VAL_PERCENTAGE = 0.1

BASE_FILE = "../datasources/ab/"
AB_FILE_X_TRAIN = BASE_FILE + "X_train.npy"
AB_FILE_X_VAL = BASE_FILE + "X_val.npy"
AB_FILE_Y_TRAIN = BASE_FILE + "Y_train.npy"
AB_FILE_Y_VAL = BASE_FILE + "Y_val.npy"
FILE = BASE_FILE + "ab_model.npz"


class abPredictor:
    def __init__(self):
        #define network
        self._input_var = T.matrix('inputs')
        self._target_var = T.matrix('targets')
        self._sigma = theano.shared(np.float32(GAUSSIAN_NOISE_SIGMA))
        print("Creating network")
        self._net = self._define_ab_network_linear(self._input_var, NUM_FEATURES, self._sigma)
        
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
            self._X_train = np.load(AB_FILE_X_TRAIN)
            self._X_val = np.load(AB_FILE_X_VAL)
            self._y_train = np.load(AB_FILE_Y_TRAIN)
            self._y_val = np.load(AB_FILE_Y_VAL)
        except:
            print("Data not found. Creating dataset")
            self._create_dataset()
            self._X_train = np.load(AB_FILE_X_TRAIN)
            self._X_val = np.load(AB_FILE_X_VAL)
            self._y_train = np.load(AB_FILE_Y_TRAIN)
            self._y_val = np.load(AB_FILE_Y_VAL)
        
        
    
    ###NETWORK DEFINITION###
    def _define_ab_network_linear(self, input_var, num_features, sigma):
        net = {}
        net['input'] = InputLayer(shape=(None, num_features), input_var=input_var)
        net['noise'] = GaussianNoiseLayer(net['input'], sigma=sigma)
        net['out'] = DenseLayer(net['noise'], num_units=1, nonlinearity=lasagne.nonlinearities.rectify)
        
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
        predict_fn = theano.function([input_var], [test_out])
        
        return (train_fn, val_fn, predict_fn)
            
            
    ###TRAIN THE NETWORK###
    def _train_network(self, network, train_fn, val_fn, X_train, y_train, X_val, y_val, num_epochs, verbose = False, save_all=True):
        import time
        
        # Launch the training loop.
        # We iterate over epochs:
        for epoch in range(num_epochs):        
            # In each epoch, we do a full pass over the training data:
            start_time = time.time()
            train_err, train_err_l2 = train_fn(X_train, y_train)

            # And a full pass over the validation data:
            val_err, val_err_l2 = val_fn(X_train, y_train)

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
        
        return network, train_err, val_err
    
    
    
    ###EXPLORE HYPERPARAMETERS###
    def explore_hyperparameters(self, X_train, y_train, X_val, y_val):
        network = self._net
        #save starting parameters
        params = lasagne.layers.get_all_param_values(network)

        #get initial values
        t_loss, t_l2 = val_fn(X_val, y_val)     
        print("Initial  valid. loss:\t\t{:.6f}".format(t_loss))


        print("LR\t\tL2\t\tGS\tTrain Loss\tVal Loss")
        for i in range(10):
            #learning_rate.set_value(np.float32(random.uniform(0.001, 0.0001)))
            #l2_lambda.set_value(np.float32(random.uniform(0.00000001, 0.00000100)))
            gaussian_noise_sigma.set_value(np.float32(random.uniform(0.,0.05)))
            
            network, train_loss, val_loss = _train_network(network, self._train_fn, self._val_fn,
                                                           X_train, y_train, X_val, y_val,
                                                           NUM_EPOCHS, verbose = False, save_all=True)
            
            print("%.8f\t%.8f\t%.8f\t%.8f\t%.8f" % (learning_rate.get_value(), l2_lambda.get_value(), gaussian_noise_sigma.get_value(), train_loss, val_loss))
        
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
    #TODO: REFACTOR
    def _create_dataset(self):
        #load the slim matrix and the netflix dataset
        weight_matrix = np.load('../../datasources/slim_W01.npz')
        from netflix_reader import  NetflixReader
        rdr = NetflixReader()
        from sklearn.cross_validation import train_test_split
        
        #get all the couple of items with a non zero similarity
        idx = weight_matrix.nonzero()
        
        Xs = list()
        ys = list()
        
        for (i1,i2) in zip(idx[0], idx[1]):
            x = rdr.similarity(i1,i2)
            y = weight_matrix[i1,i2]
            
            Xs.append(x)
            ys.append([y])
        
        #take out some samples for validation
        X_train, X_val, y_train, y_val= train_test_split(Xs, ys, test_size=VAL_PERCENTAGE, random_state=1)
        
        #save
        np.save(AB_FILE_X_TRAIN, np.array(X_train, dtype=np.float32))
        np.save(AB_FILE_X_VAL, np.array(X_val, dtype=np.float32))
        np.save(AB_FILE_Y_TRAIN, np.array(y_train, dtype=np.float32))
        np.save(AB_FILE_Y_VAL, np.array(y_val, dtype=np.float32))
        
        
        
    ###COMPUTE THE SIMILARITY MATRIX USING THE NETWORK###
    def _compute_sim_matrix(self):
        rdr = NetflixReader()
        matrix = scipy.sparse.csc_matrix((6489,6489), dtype=np.float32)
        
        #our similarity measure is symmetric
        pool = multiprocessing.Pool(processes = multiprocessing.cpu_count())
        
        print("Computing products")
        features_products=pool.map(rdr.similarity_tuple, ((i,j) for i in range(1,50) for j in range(i+1,50)))
        print("Computing similarities")
        similarities = self._predict_fn(np.array(features_products, dtype = np.float32))
        print(similarities.shape)
        
    
if __name__ == '__main__':
    a = abPredictor()
    a._compute_sim_matrix()

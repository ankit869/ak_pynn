'''
This is the complete code of MultiLayer Neural Network

Developed by: 
    - Ankit Kohli (Student at Delhi University)
    - ankitkohli181@gmail.com (mail)

Have fun!

'''
import numpy as np
import random
import time
import math
from ak_pynn.utils import *
from tqdm import tqdm
from ak_pynn.nnv import NNV
import matplotlib.pyplot as plt
from numpy.random import default_rng

class instance_variables:
    def __init__(self):
        super(instance_variables, self).__init__()
        self.weights = []
        self.bias = []
        self.Vdb = []
        self.Vdw = []
        self.Mdw = []
        self.Mdb = []
        self.derivatives_w = []
        self.derivatives_b = []
        self.layers = []
        self.B_gamma = []
        self.B_beta = []
        self.B_dGamma = []
        self.B_dBeta = []
        self.B_mov_avg=[]


class Weight_Initalizer(instance_variables):
    def __init__(self):
        super(Weight_Initalizer, self).__init__()
        self.weight_initializers_method = {
            'random_uniform': self.random_uniform,
            'random_normal': self.random_normal,
            'glorot_uniform': self.glorot_uniform,
            'glorot_normal': self.glorot_normal,
            'he_uniform': self.he_uniform,
            'he_normal': self.he_normal
        }

    def random_uniform(self, seed=None, args=dict()):
        minval = -0.05
        maxval = 0.05
        for key, value in args.items():
            if (key == 'minval'):
                minval = value
            elif (key == 'maxval'):
                maxval = value
            elif (key == 'seed'):
                np.random.seed(seed)

        for i in range(1, len(self.layers)):
            self.weights[i-1] = np.random.uniform(minval, maxval, size=(self.layers[i]['nodes'], self.layers[i-1]['nodes']))

    def random_normal(self, args=dict()):
        for key, value in args.items():
            if (key == 'seed'):
                np.random.seed(value)
        for i in range(1, len(self.layers)):
            self.weights[i-1] = np.random.randn(self.layers[i]['nodes'], self.layers[i-1]['nodes'])

    def glorot_uniform(self, args=dict()):
        for key, value in args.items():
            if (key == 'seed'):
                np.random.seed(value)
        for i in range(1, len(self.layers)):
            limit = np.sqrt(6 / (self.layers[i]['nodes'] + self.layers[i-1]['nodes']))
            vals = np.random.uniform(-limit, limit,size=(self.layers[i]['nodes'], self.layers[i-1]['nodes']))
            self.weights[i-1] = vals

    def glorot_normal(self, args=dict()):
        for key, value in args.items():
            if (key == 'seed'):
                np.random.seed(value)
        for i in range(1, len(self.layers)):
            limit = np.sqrt(2 / (self.layers[i]['nodes'] + self.layers[i-1]['nodes']))
            vals = np.random.randn(self.layers[i]['nodes'], self.layers[i-1]['nodes'])*limit
            self.weights[i-1] = vals

    def he_uniform(self, seed=None, args=dict()):
        for key, value in args.items():
            if (key == 'seed'):
                np.random.seed(value)
        for i in range(1, len(self.layers)):
            limit = np.sqrt(6 / (self.layers[i-1]['nodes']))
            vals = np.random.uniform(-limit, limit,size=(self.layers[i]['nodes'], self.layers[i-1]['nodes']))
            self.weights[i-1] = vals

    def he_normal(self, args=dict()):
        for key, value in args.items():
            if (key == 'seed'):
                np.random.seed(value)
        for i in range(1, len(self.layers)):
            vals = np.random.randn(self.layers[i]['nodes'], self.layers[i-1]['nodes']) * np.sqrt(2/(self.layers[i-1]['nodes']))
            self.weights[i-1] = vals


class Optimizers(Weight_Initalizer):
    def __init__(self):
        super(Optimizers, self).__init__()
        self.epsilon = 1e-07
        self.momentum = 0.9
        self.mov_avg_momentum = 0.9
        self.beta = 0.9
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.optimizer = None
        self.optimizer_function = {
            'momentum': self.Momentum,
            'gradient_descent': self.Gradient_descent,
            'AdaGrad': self.AdaGrad,
            'RMSprop': self.RMSprop,
            'Adam': self.Adam
        }

    def get_gradients(self, layer):
        derivatives_w = None
        derivatives_b = None
        derivatives_w = self.derivatives_w[layer]

        if (self.layers[layer+1]['L2_norm'] is not None):
            derivatives_w += (2*self.layers[layer+1]['L2_norm']*self.weights[layer])

        if (self.layers[layer+1]['L1_norm'] is not None):
            derivatives_w += (self.layers[layer+1]['L1_norm']*np.sign(self.weights[layer]))

        derivatives_b = self.derivatives_b[layer]
        return derivatives_w, derivatives_b

    def Gradient_descent(self, learningRate=0.001):
        for i in range(len(self.layers)-1):
            dw, db = self.get_gradients(i)
            self.weights[i] -= (dw * learningRate)
            self.bias[i] -= (db * learningRate)

            if self.layers[i]['batch_norm'] is True:
                self.B_gamma[i] -= (self.B_dGamma[i]*learningRate)
                self.B_beta[i] -= (self.B_dBeta[i]*learningRate)

    def Momentum(self, learningRate=0.001):
        for i in range(len(self.layers)-1):
            dw, db = self.get_gradients(i)
            self.Vdw[i] = (self.momentum*self.Vdw[i]) + (dw*learningRate)
            self.Vdb[i] = (self.momentum*self.Vdb[i]) + (db*learningRate)
            self.weights[i] -= self.Vdw[i]
            self.bias[i] -= self.Vdb[i]

            if self.layers[i]['batch_norm'] is True:
                self.B_gamma[i] -= (self.B_dGamma[i]*learningRate)
                self.B_beta[i] -= (self.B_dBeta[i]*learningRate)

    def AdaGrad(self, learningRate=0.001):
        for i in range(len(self.layers)-1):
            dw, db = self.get_gradients(i)
            self.Vdw[i] = self.Vdw[i]+(dw**2)
            self.Vdb[i] = self.Vdb[i]+(db**2)
            self.weights[i] -= (learningRate *(dw/np.sqrt(self.Vdw[i]+self.epsilon)))
            self.bias[i] -= (learningRate *(db/np.sqrt(self.Vdb[i]+self.epsilon)))

            if self.layers[i]['batch_norm'] is True:
                self.B_gamma[i] -= (self.B_dGamma[i]*learningRate)
                self.B_beta[i] -= (self.B_dBeta[i]*learningRate)

    def RMSprop(self, learningRate=0.001):
        for i in range(len(self.layers)-1):
            dw, db = self.get_gradients(i)
            self.Vdw[i] = self.beta*self.Vdw[i]+(1-self.beta)*(dw**2)
            self.Vdb[i] = self.beta*self.Vdb[i]+(1-self.beta)*(db**2)
            self.weights[i] -= (learningRate *(dw/np.sqrt(self.Vdw[i]+self.epsilon)))
            self.bias[i] -= (learningRate *(db/np.sqrt(self.Vdb[i]+self.epsilon)))

            if self.layers[i]['batch_norm'] is True:
                self.B_gamma[i] -= (self.B_dGamma[i]*learningRate)
                self.B_beta[i] -= (self.B_dBeta[i]*learningRate)

    def Adam(self, learningRate=0.001):
        for i in range(len(self.layers)-1):
            dw, db = self.get_gradients(i)
            self.Mdw[i] = self.beta1*self.Mdw[i]+(1-self.beta1)*dw
            self.Vdw[i] = self.beta2*self.Vdw[i]+(1-self.beta2)*(dw**2)
            m_dw = self.Mdw[i]/(1-self.beta1)
            v_dw = self.Vdw[i]/(1-self.beta2)
            self.weights[i] -= (learningRate*(m_dw/np.sqrt(v_dw+self.epsilon)))

            self.Mdb[i] = self.beta1*self.Mdb[i]+(1-self.beta1)*db
            self.Vdb[i] = self.beta2*self.Vdb[i]+(1-self.beta2)*(db**2)
            m_db = self.Mdb[i]/(1-self.beta1)
            v_db = self.Vdb[i]/(1-self.beta2)
            self.bias[i] -= (learningRate*(m_db/np.sqrt(v_db+self.epsilon)))

            if self.layers[i]['batch_norm'] is True:
                self.B_gamma[i] -= (self.B_dGamma[i]*learningRate)
                self.B_beta[i] -= (self.B_dBeta[i]*learningRate)


class MLP(Optimizers):
    def __init__(self):
        super(MLP, self).__init__()
        self.history = {'Losses': [], 'Scores':[],'Val_Losses':[],'Val_Scores':[], 'Weights': [], 'Biases': []}
        self.loss_functions = {
            "mae": mae,
            "mse": mse,
            "binary_cross_entropy": binary_cross_entropy,
            "categorical_cross_entropy": categorical_cross_entropy
        }
        self.loss_function_grads = {
            "mae": MAE_grad,
            "mse": MSE_grad,
            "binary_cross_entropy": BCE_grad,
            "categorical_cross_entropy": CCE_grad
        }
        self.activation_functions = {
            "sigmoid": sigmoid,
            "softplus": softplus,
            "softmax": softmax,
            "relu": relu,
            "leaky_relu": leaky_relu,
            "elu": elu,
            "tanh": tanh,
            "linear": linear
        }
    
    def validation_split(self,x,y,ratio=0.2):
        m=np.ceil(x.shape[0]*ratio)
        rng = default_rng()
        idx=rng.choice(x.shape[0], size=int(m), replace=False)
        test_idx=np.isin(np.arange(x.shape[0]),idx)
        train_idx=~np.isin(np.arange(x.shape[0]),idx)
        x_test,y_test=x[test_idx],y[test_idx]
        x_train,y_train=x[np.where(train_idx)],y[train_idx]
        return x_train,x_test,y_train,y_test

    def get_classification_confusions(self,y_true, y_pred):
        num_classes = y_true.shape[1]

        if y_pred.shape[1] == 1:  
            y_pred = np.where(y_pred >= 0.5,1,0)
        else:  
            y_pred_binary = np.argmax(y_pred, axis=1)
            y_pred = np.eye(num_classes)[y_pred_binary]

        true_positives = np.zeros(num_classes)
        true_negatives = np.zeros(num_classes)
        false_positives = np.zeros(num_classes)
        false_negatives = np.zeros(num_classes)
        
        for i in range(num_classes):
            true_positives[i] = np.sum((y_true[:, i] == 1) & (y_pred[:, i] == 1))
            true_negatives[i] = np.sum((y_true[:, i] == 0) & (y_pred[:, i] == 0))
            false_positives[i] = np.sum((y_true[:, i] == 0) & (y_pred[:, i] == 1))
            false_negatives[i] = np.sum((y_true[:, i] == 1) & (y_pred[:, i] == 0))
        
        return true_positives, true_negatives, false_positives, false_negatives

    def score(self, y_true, y_pred, metric='mse', multi_class=False):
        import numpy as np
        if(metric=='mse'):
            return np.mean(np.mean((y_true - y_pred) ** 2,axis=0))

        if(metric=='mae'):
            return np.mean(np.mean(np.abs(y_true - y_pred),axis=0))

        if(metric=='r2'):
            SS_r = self.score(y_true,y_pred,metric='mse')
            SS_m = self.score(y_true,np.mean(y_true,axis=0),metric='mse')
            r2=1-(SS_r/SS_m)
            return r2

        if (metric=='accuracy'):
            tp,tn,fp,fn=self.get_classification_confusions(y_true, y_pred)
            total=np.clip(np.sum(tp)+np.sum(tn)+np.sum(fp)+np.sum(fn),self.epsilon,np.inf)
            return (np.sum(tp)+np.sum(tn))/total
            
        if (metric=='precision'):
            tp,tn,fp,fn=self.get_classification_confusions(y_true, y_pred)
            total=np.clip((np.sum(tp)+np.sum(fp)),self.epsilon,np.inf)
            return np.sum(tp)/total

        if (metric=='recall'):
            tp,tn,fp,fn=self.get_classification_confusions(y_true, y_pred)
            total=np.clip((np.sum(tp)+np.sum(fn)),self.epsilon,np.inf)
            return np.sum(tp)/total

        if (metric=='f1_score'):
            pr = self.score(y_true, y_pred, metric='precision')
            rc = self.score(y_true, y_pred, metric='recall')
            return (2*pr*rc)/np.clip(pr+rc,self.epsilon,np.inf)

        if (metric=='macro_precision'):
            tp,tn,fp,fn=self.get_classification_confusions(y_true, y_pred)
            precision= tp/np.clip(tp+fp,self.epsilon,np.inf)
            return np.mean(precision)

        if (metric=='weighted_precision'):
            tp,tn,fp,fn=self.get_classification_confusions(y_true, y_pred)
            precision= tp/np.clip(tp+fp,self.epsilon,np.inf)
            class_frequencies = np.sum(y_true, axis=0) / np.sum(y_true)
            return np.sum(precision * class_frequencies)

        if (metric=='macro_recall'):
            tp,tn,fp,fn=self.get_classification_confusions(y_true, y_pred)
            recall= tp/np.clip(tp+fn,self.epsilon,np.inf)
            return np.mean(recall)

        if (metric=='weighted_recall'):
            tp,tn,fp,fn=self.get_classification_confusions(y_true, y_pred)
            recall= tp/np.clip(tp+fn,self.epsilon,np.inf)
            class_frequencies = np.sum(y_true, axis=0) / np.sum(y_true)
            return np.sum(recall * class_frequencies)

        if (metric=='macro_f1_score'):
            tp,tn,fp,fn=self.get_classification_confusions(y_true, y_pred)
            pr = tp/np.clip(tp+fp,self.epsilon,np.inf)
            rc = tp/np.clip(tp+fn,self.epsilon,np.inf)
            f1 = 2*((pr*rc)/(pr+rc))
            return np.mean(f1)

        if (metric=='weighted_f1_score'):
            tp,tn,fp,fn=self.get_classification_confusions(y_true, y_pred)
            class_frequencies = np.sum(y_true, axis=0) / np.sum(y_true)
            wpr = (tp/(tp+fp))*class_frequencies
            wrc = (tp/(tp+fn))*class_frequencies
            wf1 = 2*((wpr*wrc)/(wpr+wrc))
            return np.sum(wf1)

    def show_summary(self):
        print(f'''
        {'( MODEL SUMMARY )'.center(65)}
        
        ===================================================================
        {"Layer".center(20)}{"Activation".center(15)}{"Output Shape".center(15)}{"Params".center(15)}
        ===================================================================''')
        total_sum = 0
        nontrainable_sum = 0

        for i in range(len(self.layers)):
            total_sum += self.layers[i]['params']
            print(f'''
        {(self.layers[i]['type']).center(20)}{str(self.layers[i]['activation_function']).center(15)}{str(self.layers[i]['output_shape']).center(15)}{str(int(self.layers[i]['params'])).center(15)}
        -------------------------------------------------------------------''')

            if self.layers[i]['batch_norm'] is True:
                total_sum += self.layers[i]['b_params']
                nontrainable_sum += (self.layers[i]['b_params']/2)
                print(f'''
        {('BatchNormalization').center(20)}{str(None).center(15)}{str(self.layers[i]['output_shape']).center(15)}{str(int(self.layers[i]['b_params'])).center(15)}
        -------------------------------------------------------------------''')

            if self.layers[i]['dropouts'] is True:
                print(f'''
        {('Dropout').center(20)}{str(None).center(15)}{str(self.layers[i]['output_shape']).center(15)}{str(0).center(15)}
        -------------------------------------------------------------------''')

        print(f'''
        ===================================================================

        Total Params  - {int(total_sum)}
        Trainable Params  - {int(total_sum-nontrainable_sum)}
        Non-Trainable Params  - {int(nontrainable_sum)}
        ___________________________________________________________________
        ''')

    def add_layer(self, nodes=3, activation_function='linear', input_layer=False, output_layer=False, dropouts=False, batch_norm=False, drop_rate=0.2, L1_norm=None, L2_norm=None, **kwargs):
        if (input_layer is True):
            self.n_inputs = nodes
            self.layers.append({'nodes': nodes, 'activation_function': 'linear', 'dropouts': dropouts, 'drop_rate': drop_rate,
                               'regularizer': None, 'batch_norm': batch_norm, 'type': 'Input', 'output_shape': (None, nodes), 'params': 0,'b_params':0})
        elif (output_layer is True):
            self.n_outputs = nodes
            self.layers.append({'nodes': nodes, 'activation_function': activation_function, 'dropouts': False, 'drop_rate': None,
                               'L1_norm': L1_norm, 'L2_norm': L2_norm, 'batch_norm': False, 'type': 'Output', 'output_shape': (None, nodes), 'params': 0,'b_params':0})
        else:
            self.layers.append({'nodes': nodes, 'activation_function': activation_function, 'dropouts': dropouts, 'drop_rate': drop_rate,
                               'L1_norm': L1_norm, 'L2_norm': L2_norm, 'batch_norm': batch_norm, 'type': 'Dense', 'output_shape': (None, nodes), 'params': 0,'b_params':0})

    def compile_model(self, loss_function='mse', weight_initializer='glorot_uniform', optimizer="RMSprop", show_summary=True,metrics=['mse'], **kwargs):
        self.loss_func = loss_function
        self.metrics = metrics
        if self.optimizer is None:
            self.optimizer = optimizer

        self.weight_initializer = weight_initializer
        for i in range(1, len(self.layers)):
            self.Vdw.append(np.zeros((self.layers[i]['nodes'], self.layers[i-1]['nodes'])))
            self.Mdw.append(np.zeros((self.layers[i]['nodes'], self.layers[i-1]['nodes'])))
            self.weights.append(np.random.rand(self.layers[i]['nodes'], self.layers[i-1]['nodes']))
            self.derivatives_w.append(np.zeros((self.layers[i]['nodes'], self.layers[i-1]['nodes'])))
            self.bias.append(np.zeros(self.layers[i]['nodes']))
            self.Vdb.append(np.zeros(self.layers[i]['nodes']))
            self.Mdb.append(np.zeros(self.layers[i]['nodes']))
            self.derivatives_b.append(np.zeros(self.layers[i]['nodes']))
            self.B_gamma.append(np.ones(self.layers[i-1]['nodes']))
            self.B_beta.append(np.zeros(self.layers[i-1]['nodes']))
            self.B_dGamma.append(np.zeros(self.layers[i-1]['nodes']))
            self.B_dBeta.append(np.zeros(self.layers[i-1]['nodes']))
            self.B_mov_avg.append({'layer':i-1,'mean':0,'var':0})

            if (self.layers[i-1]['batch_norm'] == True):
                self.layers[i-1]['b_params'] += self.layers[i-1]['nodes']*4

            if i > 0:
                self.layers[i]['params'] += self.layers[i]['nodes']

            self.layers[i]['params'] += (self.layers[i]['nodes']*self.layers[i-1]['nodes'])

        self.weight_initializers_method[self.weight_initializer](kwargs)

        for key, value in kwargs.items():
            if (key == 'momentum'):
                self.momentum = value
            elif (key == 'mov_avg_momentum'):
                self.mov_avg_momentum = value
            elif (key == 'epsilon'):
                self.epsilon = value
            elif (key == 'beta'):
                self.beta = value
            elif (key == 'beta1'):
                self.beta1 = value
            elif (key == 'beta2'):
                self.beta2 = value

        if (show_summary):
            self.show_summary()

    def apply_dropouts(self, layer):
        drop_size = np.ceil(self.layers[layer]['drop_rate']*self.layers[layer]['nodes'])
        dropout_nodes = np.zeros(self.layers[layer]['nodes'], dtype=bool)
        node_id = random.sample(range(self.layers[layer]['nodes']), int(drop_size))
        for j in node_id:
            dropout_nodes[j] = True
        return dropout_nodes

    def set_optimizer(self, optimizer, **kwargs):
        self.optimizer = optimizer
        for key, value in kwargs.items():
            if (key == 'momentum'):
                self.momentum = value
            elif (key == 'mov_avg_momentum'):
                self.mov_avg_momentum = value
            elif (key == 'epsilon'):
                self.epsilon = value
            elif (key == 'beta'):
                self.beta = value
            elif (key == 'beta1'):
                self.beta1 = value
            elif (key == 'beta2'):
                self.beta2 = value

    def check_encoding(self, X):
        return ((X.sum(axis=1)-np.ones(X.shape[0])).sum() == 0)

    def batchnorm_forward(self,x, gamma, beta):
        N,D = x.shape
        mu = 1./N * np.sum(x, axis = 0)
        xmu = x - mu
        sq = xmu ** 2
        var = 1./N * np.sum(sq, axis = 0)
        sqrtvar = np.sqrt(var + self.epsilon)
        ivar = 1./sqrtvar
        xhat = xmu * ivar
        gammax = gamma * xhat
        out = gammax + beta
        cache = (xhat,gamma,xmu,ivar,sqrtvar,mu,var)
        return out, cache

    def batchnorm_backward(self,dout, cache,running_mean,running_var):
        xhat,gamma,xmu,ivar,sqrtvar,mu,var = cache
        N,D = dout.shape
        dbeta = np.sum(dout, axis=0)
        dgammax = dout #not necessary, but more understandable
        dgamma = np.sum(dgammax*xhat, axis=0)
        dxhat = dgammax * gamma
        divar = np.sum(dxhat*xmu, axis=0)
        dxmu1 = dxhat * ivar
        dsqrtvar = -1./(sqrtvar**2) * divar
        dvar = 0.5 * 1./np.sqrt(var+self.epsilon) * dsqrtvar
        dsq = 1./N * np.ones((N,D)) * dvar
        dxmu2 = 2 * xmu * dsq
        dx1 = (dxmu1 + dxmu2)
        dmu = -1 * np.sum(dxmu1+dxmu2, axis=0)
        dx2 = 1./N * np.ones((N,D)) * dmu
        dx = dx1 + dx2

        running_mean = self.mov_avg_momentum * running_mean + (1 - self.mov_avg_momentum) * mu
        running_var = self.mov_avg_momentum * running_var + (1 - self.mov_avg_momentum) * var

        return dx, dgamma, dbeta, running_mean, running_var

    def forward_propagate(self, x):
        """
        Performs forward propagation on Batch input
        Input: x- ndarray (Input data)

        Returns: List - activations, node_values
        """
        activations = []
        node_values = []
        B_cache = []
        for j in range(len(self.layers)):
            activations.append(np.zeros((len(x), self.layers[j]['nodes'])))
            node_values.append(np.zeros((len(x), self.layers[j]['nodes'])))
            B_cache.append(None)

        activations[0] = x
        node_values[0] = x

        if self.layers[0]['batch_norm'] is True:
            node_values[0],B_cache[0] = self.batchnorm_forward(node_values[0],self.B_gamma[0],self.B_beta[0])
            activations[0]=node_values[0]

        if (self.layers[0]['dropouts'] == True):
            drop_nodes = self.apply_dropouts(0)
            activations[0][:, np.where(drop_nodes)] = 0

        for i in range(1, len(self.layers)):
            node_values[i] = np.dot(activations[i-1], self.weights[i-1].T)+self.bias[i-1]

            if self.layers[i]['batch_norm'] is True:
                node_values[i],B_cache[i] = self.batchnorm_forward(node_values[i],self.B_gamma[i],self.B_beta[i])

            activations[i] = self.activation_functions[self.layers[i]['activation_function']](node_values[i])

            if (self.layers[i]['dropouts'] == True):
                drop_nodes = self.apply_dropouts(i)
                activations[i][:, np.where(drop_nodes)] = 0

        return activations, node_values,B_cache

    def back_propagate(self, y, activations, node_values,B_cache):
        p = activations[len(self.layers)-1]
        """
        Performs Back propagation on Batch input

        Input: y- targets (N, k) ndarray (N: no. of samples, k: no. of output nodes)
               activations - list of activations of all layers
               node_values - list of node values of all layers
               B_cache - tuple of cache for batch_norm_layers

        Returns: None

        """

        error = float('inf')
        error = self.loss_function_grads[self.loss_func](y, p)

        for i in reversed(range(len(self.derivatives_w))):
            delta_w = None
            func_name = self.layers[i+1]['activation_function']
            activation_func = self.activation_functions[func_name]
            if (func_name == "softmax"):
                delta_w = softmaxTimesVector(error, activation_func(node_values[i+1], derive=True))
            else:
                delta_w = error*activation_func(node_values[i+1], derive=True)
            self.derivatives_w[i] = np.dot(delta_w.T, activations[i])/self.batch_size
            self.derivatives_b[i] = np.sum(delta_w, axis=0)/self.batch_size

            error = np.dot(delta_w, self.weights[i])

            if self.layers[i]['batch_norm'] is True:
                func_name = self.layers[i]['activation_function']
                activation_func = self.activation_functions[func_name]
                error, self.B_dGamma[i], self.B_dBeta[i],running_mean,running_var= self.batchnorm_backward(error,B_cache[i],self.B_mov_avg[i]['mean'],self.B_mov_avg[i]['var'])
                self.B_mov_avg[i]['mean']=running_mean
                self.B_mov_avg[i]['var']=running_var

    def fit(self, x, y, learning_rate=0.001, epochs=50, batch_size=32, verbose=False, early_stopping=False, patience=2, shuffle=False,validation_split=None):
        total_time = 0
        patience_count = 0
        self.batch_size = batch_size
        self.isMultiClass=False
        if len(y.shape) == 1:
            y = y.reshape(-1, 1)
        else:
            self.isMultiClass=True
            if self.check_encoding(y) is False:
                print("ERROR: please use one-hot-encoded targets")
                return

        if verbose is False:
            epoch_range = tqdm(range(1, epochs+1), desc="Training progress :")
        else:
            epoch_range = range(1, epochs+1)

        for i in epoch_range:
            t = Timer()
            t.start()
            sum_errors = 0
            sum_scores = np.zeros(len(self.metrics))
            if shuffle is True:
                shuffled_indices = np.random.permutation(x.shape[0])
                x = x[shuffled_indices]
                y = y[shuffled_indices]
            
            X_train,Y_train=x,y

            if validation_split is not None:
                X_train,X_test,Y_train,Y_test=self.validation_split(x,y,validation_split)

            x_batches = np.array_split(X_train, math.ceil(X_train.shape[0]/batch_size))
            y_batches = np.array_split(Y_train, math.ceil(Y_train.shape[0]/batch_size))
            n_batches = len(x_batches)
            metric_str=""

            if verbose is True:
                batch_range = tqdm(range(1, n_batches+1), desc=f"EPOCH {i} ")
            else:
                batch_range = range(1, n_batches+1)

            for b in batch_range:
                X_batch = x_batches[b-1]
                Y_batch = y_batches[b-1]
                if shuffle is True:
                    shuffled_indices = np.random.permutation(X_batch.shape[0])
                    X_batch = X_batch[shuffled_indices]
                    Y_batch = Y_batch[shuffled_indices]
                
                activations, node_values,B_cache = self.forward_propagate(X_batch)
                self.back_propagate(Y_batch, activations, node_values,B_cache)
                self.optimizer_function[self.optimizer](learning_rate)

                batch_loss = self.loss_functions[self.loss_func](Y_batch, activations[len(self.layers)-1])
                batch_scores=np.zeros(len(self.metrics))
                for i in range(len(self.metrics)):
                    batch_scores[i]=(self.score(Y_batch, activations[len(self.layers)-1],metric=self.metrics[i],multi_class=self.isMultiClass))
                
                sum_errors += batch_loss
                sum_scores += batch_scores
                b_metric_str=""
                for i in range(len(self.metrics)):
                    b_metric_str+= f" - {self.metrics[i]}: {sum_scores[i]/b:.5f}"
                
                if verbose is True:
                    batch_range.set_postfix_str(f"Loss: {sum_errors/b:.5f}"+b_metric_str)
                
                if b==n_batches:
                    for i in range(len(self.metrics)):
                        metric_str+= f" - {self.metrics[i]}: {sum_scores[i]/b:.5f}"

                    if validation_split is not None:
                        Y_pred=self.predict(X_test)
                        batch_val_scores=np.zeros(len(self.metrics))
                        for i in range(len(self.metrics)):
                            batch_val_scores[i]=(self.score(Y_test,Y_pred,metric=self.metrics[i],multi_class=self.isMultiClass))
                        val_loss=self.loss_functions[self.loss_func](Y_test,Y_pred)
                    
                        self.history['Val_Losses'].append(val_loss)
                        self.history['Val_Scores'].append(batch_val_scores)
                    
                        val_metric_str=f" - val_loss: {val_loss:.5f}"
                        for i in range(len(self.metrics)):
                            val_metric_str+= f" - val_{self.metrics[i]}: {batch_val_scores[i]:.5f}"

                        if verbose is True:
                            batch_range.set_postfix_str(f"Loss: {sum_errors/b:.5f}"+b_metric_str+val_metric_str)

                        metric_str+=val_metric_str

                                
            elapse_time = t.stop()
            total_time += elapse_time
            loss = sum_errors/n_batches
            score = sum_scores/n_batches
            
            if verbose is False:
                epoch_range.set_postfix_str(f"Loss: {loss:.5f}"+metric_str)

            if (len(self.history['Losses']) > 1 and loss <= self.history['Losses'][-1]):
                patience_count = 0

            if (early_stopping == True and len(self.history['Losses']) > 1 and loss > self.history['Losses'][-1]):
                patience_count += 1
                if (patience_count >= patience):
                    print(
                        "\n<==================(EARLY STOPPING AT --> EPOCH {})====================> ".format(i))
                    break

            self.history['Losses'].append(loss)
            self.history['Scores'].append(score)
            self.history['Weights'].append(self.weights)
            self.history['Biases'].append(self.bias)
            final_loss=self.history['Losses'][-1]
            final_score={}
            for i in range(len(self.metrics)):
                final_score[self.metrics[i]]=np.round_(self.history['Scores'][-1],decimals=8)[i]

        print(f"\nMinimised Loss : {final_loss:.8f}, Training metrics : {final_score} ")
        print(
            f"\nTraining complete!! , Average Elapse-Time (per epoch) : {(total_time/epochs):.5f} seconds")
        print(
            "========================================================================= :)")
        return self.history['Losses']

    def batchnorm_inference(self, X, gamma, beta, running_mean, running_var):
        xhat = (X - running_mean) / np.sqrt(running_var + self.epsilon)
        out = gamma * xhat + beta
        return out

    def predict(self, x):
        outputs = []
        values = x
        if (self.layers[0]['batch_norm'] == True):
            values=self.batchnorm_inference(values,self.B_gamma[0],self.B_beta[0],self.B_mov_avg[0]['mean'],self.B_mov_avg[0]['var'])

        for i in range(1, len(self.layers)):

            if (self.layers[i-1]['dropouts'] == True):
                wgt = self.weights[i-1] * (1-self.layers[i-1]['drop_rate'])
                z = np.dot(values, wgt.T)+self.bias[i-1]
            else:
                z = np.dot(values, self.weights[i-1].T)+self.bias[i-1]
            
            if (self.layers[i]['batch_norm'] == True):
                z=self.batchnorm_inference(z,self.B_gamma[i],self.B_beta[i],self.B_mov_avg[i]['mean'],self.B_mov_avg[i]['var'])
        
            values = self.activation_functions[self.layers[i]['activation_function']](z)

        outputs.append(values)
        return np.array(outputs).reshape(-1, self.n_outputs)
    
    def predict_scores(self, x_test, y_test,metrics=['mse']):
        y_pred = self.predict(x_test)
        scores={}
        for i in metrics:
            scores[i]=(self.score(y_test, y_pred,metric=i))
        return scores
        
    def visualize(self,max_num_nodes_visible=10, spacing_layer=60, figsize=(14,10),font_size=17, filename='neural_network_visualization.png'):
        plt.rcParams["figure.figsize"] = figsize
        layersList=[]
        for layer in self.layers:
            if layer['type']=='Input':
                layersList.append({"title":f"{layer['type']}\n( n={str(layer['nodes'])} )", "units": layer['nodes'], "color": "royalblue","fully_connect":True})
                if layer['batch_norm']==True:
                    layersList.append({"title":f"Batch_Norm\n( n={str(layer['nodes'])} )", "units": layer['nodes'], "color": "lightpink","fully_connect":False,"spacing_layer":spacing_layer/2})
                if layer['dropouts']==True:
                    layersList.append({"title":f"Dropout\n( n={str(layer['nodes'])} )", "units": layer['nodes'], "color": "lightpink","fully_connect":False,"spacing_layer":spacing_layer/2})

            elif layer['type']=='Output':
                layersList.append({"title":f"{layer['type']}\n( {layer['activation_function']} )\n( n={str(layer['nodes'])} )", "units": layer['nodes'],"color": "lightcoral","fully_connect":True})
            else:
                layersList.append({"title":f"{layer['type']}\n( {layer['activation_function']} )\n( n={str(layer['nodes'])} )", "units": layer['nodes'],"color": "violet","fully_connect":True})
                if layer['batch_norm']==True:
                    layersList.append({"title":f"Batch_Norm\n( n={str(layer['nodes'])} )", "units": layer['nodes'], "color": "lightpink","fully_connect":False,"spacing_layer":spacing_layer/2})
                if layer['dropouts']==True:
                    layersList.append({"title":f"Dropout\n( n={str(layer['nodes'])} )", "units": layer['nodes'], "color": "lightpink","fully_connect":False,"spacing_layer":spacing_layer/2})

 
        NNV(layersList,max_num_nodes_visible=max_num_nodes_visible, node_radius=5, spacing_layer=spacing_layer, font_size=font_size).render(save_to_file=filename)

        plt.rcParams["figure.figsize"] = (6,4)
        plt.show()
    

class TimerError(Exception):
    """A custom exception used to report errors in use of Timer class"""


class Timer:
    def __init__(self):
        self._start_time = None

    def start(self):
        """Start a new timer"""
        if self._start_time is not None:
            raise TimerError(f"Timer is running. Use .stop() to stop it")

        self._start_time = time.perf_counter()

    def stop(self, show_elapsed=False):
        """Stop the timer, and report the elapsed time"""
        if self._start_time is None:
            raise TimerError(f"Timer is not running. Use .start() to start it")

        elapsed_time = time.perf_counter() - self._start_time
        self._start_time = None
        if show_elapsed:
            print(f"Elapsed time: {elapsed_time:0.4f} seconds")
        return elapsed_time


''' References

https://towardsdatascience.com/batch-norm-explained-visually-how-it-works-and-why-neural-networks-need-it-b18919692739
https://kratzert.github.io/2016/02/12/understanding-the-gradient-flow-through-the-batch-normalization-layer.html
https://medium.com/intuitionmath/how-to-implement-the-softmax-derivative-independently-from-any-loss-function-ae6d44363a9d
https://stackoverflow.com/questions/42670274/how-to-calculate-fan-in-and-fan-out-in-xavier-initialization-for-neural-networks
https://e2eml.school/softmax.html#:~:text=Backpropagation,respect%20to%20its%20output%20values
https://medium.com/intuitionmath/how-to-implement-the-softmax-derivative-independently-from-any-loss-function-ae6d44363a9d
https://deepnotes.io/softmax-crossentropy
https://github.com/ryanchesler/NN-Plot

'''
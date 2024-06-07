import math
import numpy as np
import pandas as pd
from nnfs.datasets import spiral_data
import itertools
from sklearn.model_selection import train_test_split
import tqdm
import sklearn.datasets

class Pipeline():
    '''A improved version of sklearn.Pipeline, can use a list of functions to process a variable'''
    def __init__(self, data, methods_params:dict):
        self.data = data
        self.methods_params = methods_params
        self.methods = methods_params.keys()
        self.params = methods_params.values()
        
    def transform(self):
        '''returns the output of the process'''
        d = self.data
        for func,params in self.methods_params.items():
            if params == None or params == '':
                try:
                    d = func(d)
                except Exception as e:
                    print(f"{func}, error message: {e.args}")
                    
            else:
                try:
                    d = func(d, *params)
                except Exception as e:
                    print(f"{func}, error message: {e.args}")
        return d



   class ReLU():
    '''ReLU activation function. The function is the identity map when the input>0
    otherwise it outputs 0
    '''
    
    def fp(self, x):
        self.outputs = np.maximum(x, 0)
        self.inputs = x
        return self.outputs

    def deriv_ReLU(x):
        if x > 0:
            return 1
        elif x == 0:
            return 0
        
    def bp(self, dvalues):
        '''the derivative is simply 0 or 1, depend on input'''
        self.dinputs = dvalues.copy()
        self.dinputs[self.inputs <= 0] = 0
        
        
        
        
        
class Leaky_ReLU(ReLU):
    '''Improve version of ReLU. When the input is smaller than 0, the function is still linear but has 
        a very small slope.
        '''
    def Leaky_Relu(self, x):
        self.outputs = np.maximum(x, 0.01*x)
        self.inputs = x
        return self.outputs

    def deriv_Leaky_ReLU(x):
        if x > 0:
            return 1
        elif x == 0:
            return 0
        else:
            return 0.01
        
    def bp(self, dvalues):
        '''Same as ReLU, the slope is taken as 0.01 by default'''
        self.dinputs = dvalues.copy()
        self.dinputs[self.inputs <= 0] = -0.01
        
        
class Sigmoid():
    '''given by e^x/1+e^x, outputs a probability between 0 and 1'''
    def fp(x):
        return 1/((np.e)**(-x)+1)


    def deriv_sigmoid(x):
        return Sigmoid(x)*(1-Sigmoid(x))
    
    def bp(self, dvalue):
        '''not applicable for this project'''
        pass


class Softmax:
    '''the extension of Sigmoid function to higher dimensions, 
    outputs a vector of probabilities that add up to 1
    '''
    def fp(self, inputs:np.array):
        self.inputs = inputs

        exp_values = np.exp(inputs - np.max(inputs, axis=1,
                        keepdims=True))

        probabilities = exp_values/np.sum(exp_values, axis=1, keepdims=True)
        self.outputs = probabilities
        return self.outputs
    
    
        
    
    def bp(self, dvalues):
        '''The backpropagation of Softmax, given as a Jacobian matrix.'''
        self.dinputs = np.empty_like(dvalues)
        
        for index, (single_output, single_dvalue) in enumerate(zip(self.outputs, dvalues)):
            single_output = single_output.reshape(1,-1)
        
            jacobian_matrix = np.dot(np.identity(len(single_output)), single_output)-np.dot(single_output, single_output.T)
            
            self.dinputs[index] = np.dot(jacobian_matrix, single_dvalue)
                                 
        
        

class CrossEntropyLoss():
    def calculate(self, inputs, desired, threshold = 1e-07):
        '''∑ desired * log(predicted)   calculate the loss for discrete data'''
        
        
        inputs_clipped = np.clip(inputs, threshold, 1-threshold)
        
        samples = len(inputs)
        
        if len(desired.shape) == 1:
            correct_confidences = inputs_clipped[range(samples),
                   desired] 
       
        elif len(desired.shape) == 2:
            correct_confidences = np.sum(inputs_clipped*desired, axis = 1)
            
       
        losses = -np.log(correct_confidences)
        overall_loss = np.mean(losses)
        return overall_loss
    
    def bp(self, dvalues, y_true):
        '''returns the dderivative of loss'''
        labels = len(dvalues[0])
        
        if len(y_true.shape) == 1:
            y_true = np.eye(labels)[y_true]
            
        self.dinputs = -y_true/dvalues
        self.dinputs = self.dinputs/len(dvalues)
        
        return self.dinputs
            
            

class CrossEntropyLoss_SoftMax():
    '''combining Cross entropy loss with softmax function, because softmax is always put at the last layer'''
    def __init__(self):
        self.activation = Softmax()
        self.loss = CrossEntropyLoss()
        
        
    def fp(self, inputs, ytrue):
        self.outputs = self.activation.fp(inputs)
        return self.loss.calculate(self.outputs, ytrue)
        
    def bp(self, dvalues, ytrue):
        ''' There's a short cut if we use softmax on the last layer.
        The derivative of the combined loss is just predicted-actual'''
        sample = len(dvalues)
        
        if ytrue.shape == 2:
            ytrue = np.argmax(ytrue, axis = 1)
            
        self.dinputs = dvalues.copy()
        self.dinputs[range(sample), ytrue] -= 1
        self.dinputs = self.dinputs/sample
        return self.dinputs


class DenseLayer(object):
    '''fully connected layer: every neuron in two layers are connected'''
    def __init__(self, num_inputs, num_neurons):
        self.weights = 0.01*np.random.randn(num_inputs, num_neurons)
        self.bias = np.zeros((1,num_neurons))
        
        
    def fw_progagation(self, inputs):
        '''forward propagation without activation function'''
        
        self.inputs  = inputs
        self.outputs = np.dot(inputs,self.weights)+self.bias
        return self.outputs

    def bw_progagation(self, dvalues):
         '''backward propagation without activation function'''
        self.dweights = np.dot(self.inputs.T, dvalues) 
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True) 
        self.dinputs = np.dot(dvalues, self.weights.T)
        


def Accuracy(inputs, targets):
    '''accuracy of a forward pass'''
    targets = np.array(targets)
    prediction = np.argmax(inputs, axis = 1)
    if len(targets.shape) == 2:
        targets = np.argmax(targets, axis = 1)
    acc = np.mean(prediction == targets)
    return acc


class Optimizer_SGD:
    '''classic stochastic gradient descent'''
    def __init__(self, learning_rate = 1):
        self.learning_rate = learning_rate
        
    def update_params(self, layer):
        '''gradient descend for weights and biases'''
        
        layer.weights -= self.learning_rate*layer.dweights
        layer.bias -= self.learning_rate*layer.dbiases




class Optimizer_adagrad:
    '''update learning rate based on past gradients'''

    def __init__(self,learning_rate = 1., decay = 1e-4,epsilon = 1e-7):
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iteration = 0
        self.epsilon = 1e-7
        
    def pre_update_params(self):
        '''learning rate decreases as iteration increases'''
        if self.decay:
            self.current_learning_rate = self.learning_rate *(1. / (1. + self.decay * self.iteration))
        
    def update_params(self, layer):
        '''past gradient information is stored in cache, represented by the squared sum of past gradients
            a constant epsilon is used to tweak the graidient
        '''
        if not hasattr(layer, "weight_cache"):
            layer.weight_cache = np.zeros_like(layer.weights)
            layer.bias_cache = np.zeros_like(layer.bias)
        layer.weight_cache += layer.dweights**2
        layer.bias_cache += layer.dbiases**2
        layer.weights += -self.current_learning_rate * layer.dweights / (np.sqrt(layer.weight_cache) + self.epsilon)
        
        
        layer.bias += -self.current_learning_rate * layer.dbiases / (np.sqrt(layer.bias_cache) + self.epsilon)

    def update_iteration(self):
        self.iteration += 1



df_train = pd.read_csv("sevens_and_nines_train.csv")
df_test = pd.read_csv("sevens_and_nines_test.csv")
df_train["target"] = df_train["target"].replace({7:0, 9:1})


X_df = np.array(df_train.drop(["target"], axis = 1))
y_df = np.array(df_train["target"])
X_train, X_valid, Y_train, Y_valid = train_test_split(X_df, y_df, shuffle=True, test_size=0.3)

# A hard coded working network
Loss = CrossEntropyLoss_SoftMax()
relu = Leaky_ReLU()
softmax = Softmax()
 

ds1 = DenseLayer(64,64)
ds2 = DenseLayer(64,2)
#ds3 =  DenseLayer(3,3)

X,y = spiral_data(samples = 100, classes=3)

optimizer = Optimizer_adagrad()



for epoch in range(100):
    
    ds1.fw_progagation(X_train) 
    relu.fp(ds1.outputs)
    ds2.fw_progagation(relu.outputs)
   
    #ds3.fw_progagation(ds2.outputs)
    loss = Loss.fp(ds2.outputs,Y_train)
    
    if len(y.shape) == 2:
        y = np.argmax(y, axis=1)

    accuracy = Accuracy(Loss.outputs, Y_train)
    
    
    if not epoch % 1: print(f'epoch: {epoch}, ' +f'acc: {accuracy:.3f}, ' + f'loss: {loss:.3f}'
                              +f'lr: {optimizer.current_learning_rate}')
    
    Loss.bp(Loss.outputs, Y_train)
    #ds3.bw_progagation(Loss.dinputs)
    
    
    ds2.bw_progagation(Loss.dinputs)
    relu.bp(ds2.dinputs)
    ds1.bw_progagation(relu.dinputs)
    
    optimizer.pre_update_params()
    optimizer.update_params(ds2)
    optimizer.update_params(ds1)
    optimizer.update_iteration()
   
    
    
    
    
   












class ANN():
    '''class for constructing neural networks'''
    def __init__(self, layers:list, optimizer = Optimizer_adagrad()):
        '''layers = [(layer1, activation1), (layer1, activation1),...]
        all layers and activations should be instances of corresponding classes
        '''
        self.layers = layers
        self.optimizer = optimizer
    
    
    def forward(self, X, y):
        '''define a forward pass'''
        self.Loss = CrossEntropyLoss_SoftMax()
        
        try:
            self.layers = list(itertools.chain(*self.layers))
        except TypeError:
            pass
        
        self.layers[0].fw_progagation(X)
        for c in range(1, len(self.layers)):
            if hasattr(self.layers[c], "fw_progagation"):
                self.layers[c].fw_progagation(self.layers[c-1].outputs)
                
            else:
                self.layers[c].fp(self.layers[c-1].outputs)
        tot_outputs = self.layers[-1].outputs
        self.loss = self.Loss.fp(tot_outputs,y)
        self.tot_outputs = tot_outputs
        self.accuracy = Accuracy(self.Loss.outputs, y)
        
        
        return self.accuracy, self.loss
    
    
    def backward(self, X, y):
        '''define a back propagation
        the order of it is exactly the opposite 
        
        '''
        self.layers.reverse()
        self.Loss.bp(self.Loss.outputs, y)
        self.layers[0].bw_progagation(self.Loss.dinputs)
        for c in range(1, len(self.layers)):
            if hasattr(self.layers[c], "bw_progagation"):
                self.layers[c].bw_progagation(self.layers[c-1].dinputs)
                
            else:
                self.layers[c].bp(self.layers[c-1].dinputs)
        self.layers.reverse()
    
    
    def fit(self, X, y , epochs = 100):
        '''forward and backward propagation. epochs is the number of iterations of the cycle
            it has a loading bar!
        '''
        for epoch in tqdm.tqdm(range(epochs), desc ="Calculating..."):
            fw_params = self.forward(X,y)
            if not epoch % (epochs//100): 
                print(f'epoch: {epoch}, ' +f'acc: {fw_params[0]:.3f}, ' + f'loss: {fw_params[1]:.3f}')
            self.backward(X,y)
            
            self.optimizer.pre_update_params()
            for c in range(len(self.layers)):
                if isinstance(self.layers[c], DenseLayer):
                    self.optimizer.update_params(self.layers[c])   
                    
            self.optimizer.update_iteration()
            
            
    def evaluate(self, X):
        '''A complete forward propagation based on the updated wights and biases'''
        self.layers[0].fw_progagation(X)
        for c in range(1, len(self.layers)):
            if hasattr(self.layers[c], "fw_progagation"):
                self.layers[c].fw_progagation(self.layers[c-1].outputs)
                
            else:
                self.layers[c].fp(self.layers[c-1].outputs)
        tot_outputs = self.layers[-1].outputs
        return tot_outputs
        
        
nn = ANN([(DenseLayer(64,64), Leaky_ReLU()), 
          (DenseLayer(64,32), Leaky_ReLU()),
          (DenseLayer(32,32), ),
          (DenseLayer(32,10), )])


# test on a digit recognition dataset


digits_data = sklearn.datasets.load_digits()
digits_data_Y = digits_data.target
digits_data_X = digits_data.data
X_train, X_valid, Y_train, Y_valid = train_test_split(digits_data_X, digits_data_Y, shuffle=True, test_size=0.3)

nn2 = ANN([(DenseLayer(64,64), ReLU()), 
          (DenseLayer(64,32), ReLU()),
          (DenseLayer(32,32), ReLU()),
          (DenseLayer(32,10), )])
nn2.fit(X_train, Y_train, epochs = 400)
out = nn2.evaluate(X_valid)
Accuracy(out, Y_valid)















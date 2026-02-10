'''
Self Implemented Multivariable Linear Regression Algorithm 
''' 

import numpy as np
import matplotlib.pyplot as plt

class MultivariableLinearRegression():
    def __init__(self,num_features,lr):
        self.weights = np.zeros(num_features)
        self.b = 0
        self.lr = lr
        self.normalize_parameters = None
    
    '''
    Function that trains the machine learning model an {epochs} amount of times to improve predictions.
    '''
    def train(self,inputs,targets,epochs):
        for epoch in range(epochs):
            yhat = [] # F_wb

            for i in range(inputs.shape[0]):
                yhat.append(np.dot(inputs[i],self.weights) + self.b)


            loss = self.compute_loss(yhat,targets)
            print(f'Epoch: {epoch+1}, Current Loss: {loss:.4f}')

            self.weights,self.b = self.gradient_descent(inputs,targets,yhat)
        return yhat
    
    '''
    Function that calculates the loss (J_wb) through a modified mean squared error formula.
    '''
    def compute_loss(self,yhat,y):
        J_wb = 0
        for i in range(y.shape[0]):
            J_wb += (yhat[i]-y[i])**2
        return J_wb/(2*y.shape[0]) 

    '''
    Function that calculates the change in weight values through batch gradient descent.
    '''
    def gradient_descent(self,inputs,targets,yhat):

        weights = self.weights.copy()
        b = self.b

        b_derv = 0
        for i in range(inputs.shape[0]):
            b_derv += (yhat[i]-targets[i])

        b_derv /= (inputs.shape[0])
        b -= (self.lr*b_derv)

        for j in range(len(self.weights)): 
            w_derv = 0 

            for i in range(inputs.shape[0]):
                w_derv += np.dot((yhat[i]-targets[i]),inputs[i][j])

            w_derv /= (inputs.shape[0])
            weights[j] -= (self.lr*w_derv)

        return weights,b
    
    ''' 
    Function that uses the mean and standard deviation to represent it on a smaller range of values.
    '''
    def normalize_data(self, inputs):
        inputs = np.asarray(inputs,dtype=float)
        # Calculates mean and standard deviation for future data to adhere to same conversion rules.
        if self.normalize_parameters is None:
            mean = np.mean(inputs, axis=0)
            std = np.std(inputs, axis=0)

            std[std == 0] = 1

            self.normalize_parameters = (mean, std)
            return (inputs - mean) / std
        else:
            mean,std = self.normalize_parameters
            return (inputs - mean) / std
    
    '''
    Function to evaluate one pair of data by normalizing the data and returning its prediction (f_wb).
    '''
    def eval(self,inputs):
        inputs = self.normalize_data(inputs)
        return (np.dot(inputs,self.weights) + self.b)
    
if __name__ == "__main__":
    # Testing Multi Variable Linear Regression
    X = np.array(
        [[1, 1],
        [2, 1],
        [3, 1],
        [5, 1]])
        
    Y = np.array([17, 27, 37, 57])

    model = MultivariableLinearRegression(num_features=X.shape[1], lr=0.1)
    X_norm = model.normalize_data(X)

    y_pred = model.train(X_norm, Y, epochs=100)
    print(f"\nResults\n - Evaluation Test: Predicted: {model.eval([4,1]):2f}, Actual: 47")

    plt.scatter(Y, y_pred, c="purple")
    plt.plot([Y.min(), Y.max()], [Y.min(), Y.max()], 'b--')
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.show()

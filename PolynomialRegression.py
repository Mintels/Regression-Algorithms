'''
Self Implemented Polynomial Regression Algorithm
(Using a machine learning model similar to the one used in Multivariable Linear Regression, but with the addition of polynomial features.)
'''

import numpy as np
import matplotlib.pyplot as plt

class PolynomialRegression():
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
            if (epoch+1) % (epochs/10) == 0:
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
    # Univariable data with polynomial relationship

    x_raw = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

    # Feature engineering: Create polynomial features [x, x^2]
    x_new = np.column_stack([x_raw, x_raw**2, x_raw**3, x_raw**4, x_raw**5])

    # Predctions based off the polynomial = 3x^2 + 2x + 5
    y = 3*x_raw**2 + 2*x_raw + 5 + np.random.randn(10)*2

    model = PolynomialRegression(num_features=x_new.shape[1], lr=0.3)

    x_norm = model.normalize_data(x_new)

    y_pred = model.train(x_norm, y, epochs=100000)

    plt.scatter(x_raw,y,c="red") 
    plt.plot(x_raw,y_pred,c="green")
    plt.xlabel('Input Values')
    plt.ylabel('Predictions')
    plt.show()
'''
Self Implemented Univariable Linear Regression Algorithm
'''

import numpy as np
import matplotlib.pyplot as plt


class UnivariableLinearRegression():
    def __init__(self,lr):
        self.weight1 = 0 # w_0
        self.weight2 = 0 # b
        self.lr = lr

    '''
    Function that trains the machine learning model an {epochs} amount of times to improve predictions.
    '''
    def train(self,x,y,epochs):
        for i in range(epochs):
            yhat = [] # F_wb
            for j in range(x.shape[0]):
                yhat.append(self.weight1*x[j] + self.weight2)
            loss = self.compute_loss(yhat,y)
            print(f'Epoch: {i+1}, Current Loss: {loss:.4f}')
            self.weight1,self.weight2 = self.graident_descent(x,y,yhat,self.weight1,self.weight2,self.lr)
            
        return yhat
    
    '''
    Function that calculates the loss (J_wb) through a modified mean squared error formula.
    '''
    def compute_loss(self,yhat,y):
        J_wb = 0
        for i in range(y.shape[0]):
            J_wb += ((yhat[i]-y[i])**2)
        return J_wb/(2*y.shape[0]) 
    
    '''
    Function that calculates the change in weight values through batch gradient descent.
    '''
    def graident_descent(self,x,y,yhat,w,b,lr):
        w_derv,b_derv = 0,0 
        for i in range(x.shape[0]):
            w_derv +=  ((yhat[i]-y[i])*x[i])
            b_derv += (yhat[i]-y[i])
        w_derv /= (x.shape[0])
        b_derv /= (x.shape[0])
        return w-(lr*w_derv),b-(lr*b_derv)
        
    def eval(self,x):
        return self.weight1*x + self.weight2
    

if __name__ == "__main__":
    # Testing One Variable Linear Regression
    X = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=float)
    Y = np.array([3.2, 5.1, 7.3, 9.2, 11.1, 12.9, 15.3, 17.0, 19.4, 20.8], dtype=float)

    plt.scatter(X,Y,c="red") 

    model = UnivariableLinearRegression(lr=0.04)
    predictions = model.train(X,Y,epochs=1000)
    plt.plot(X,predictions,c="green")
    plt.show()

'''
Self-implemented Logistic Regression model using Sigmoids Function
'''

import numpy as np
import matplotlib.pyplot as plt


class LogisticRegression():
    def __init__(self,num_features,lr, reg):
        self.weights = np.zeros(num_features)
        self.b = 0
        self.lr = lr # Learning rate for gradient descent.
        self.reg = reg # Regularization parameter to prevent overfitting.
        self.normalize_parameters = None # To store mean and std for normalization of future data.
    '''
    Function that trains the machine learning model an {epochs} amount of times to improve predictions.
    '''
    def train(self,inputs: np.ndarray,targets: np.ndarray, epochs: int) -> np.ndarray:

        m = inputs.shape[0] # Number of training examples

        for epoch in range(1,epochs+1):

            cost, yhat = self.compute_cost(inputs,targets)

            self.weights, self.b = self.gradient_descent(inputs,targets,yhat)

            if epoch % 100 == 0:
                print(f"Epoch {epoch}, Cost: {cost:3f}")
        

        return yhat

    '''
    Function that computes the cost of the model using maximum likelihood.
    '''

    def compute_cost(self,inputs: np.ndarray,y: np.ndarray) -> tuple[float, np.ndarray]:

        cost = 0
        regularization = 0 
        m = inputs.shape[0]
        n = len(self.weights)

        predictions = np.zeros(m)

        for i in range(m):

            z = np.dot(inputs[i],self.weights) + self.b # The Linear combination of weights and inputs (w*x + b)

            yhat = (1 / (1 + np.exp(-z))) # Sigmoid Functon Equation

            predictions[i] = yhat

            cost -= y[i] * np.log(yhat) + (1 - y[i]) * np.log(1 - yhat)

        
        for j in range(n):
            regularization += self.weights[j] ** 2 
        regularization = (self.reg / (2 * m))
            
        cost = (cost / m)  + regularization

        return cost, predictions 

    '''
    Function that performs gradient descent to update the weights and bias of the model.
    '''
    def gradient_descent(self, inputs: np.ndarray, targets: np.ndarray, yhat: np.ndarray) -> tuple[np.ndarray, int]:

        weights = self.weights.copy()
        b = self.b

        m = inputs.shape[0]
        diff = yhat - targets

        b_derv = 0
        b_derv = np.mean(diff)
        b -= (self.lr*b_derv)

        for j in range(len(self.weights)): 
            w_derv = 0 

            for i in range(m):
                w_derv += np.dot((diff[i]),inputs[i][j])

            w_derv /= (m)
            w_derv += (self.reg * weights[j]) / m
            weights[j] -= (self.lr*w_derv)

        
        return weights,b
    
    ''' 
    Function that uses the mean and standard deviation to represent it on a smaller range of values.
    '''
    def normalize_data(self, inputs: np.ndarray) -> np.ndarray:
        inputs = np.asarray(inputs,dtype=float)
        # Calculates mean and standard deviation for future data to adhere to same conversion rules.
        if self.normalize_parameters is None:

            mean = np.mean(inputs, axis=0)
            std = np.std(inputs, axis=0)

            std[std == 0] = 1

            self.normalize_parameters = (mean, std)

        else:

            mean,std = self.normalize_parameters
        return (inputs - mean) / std
    
    '''
    Function that predicts the output for a given input using the trained model.
    '''
    def predict(self,inputs: np.ndarray, y: np.ndarray) -> np.ndarray:

        z = np.dot(inputs, self.weights) + self.b

        return 1 / (1 + np.exp(-z))

    
if __name__ == "__main__":

    
    # Dataset Creation

    np.random.seed(42)
    n = 300
    X1 = np.random.uniform(-3, 3, n)
    X2 = np.random.uniform(-3, 3, n)
    Y = (X1 + X2 + np.random.normal(0, 1, n) > 0).astype(int)

    X = np.column_stack((X1, X2))

    # Data Manipulation and Training
    
    model = LogisticRegression(num_features=X.shape[1], lr=0.1, reg=1)
    X_norm = model.normalize_data(X)
    results = model.train(X_norm, Y, epochs=800)



    '''
    Visualization of the Decision Boundary for Self Implementation
    '''
    
    for i in range(X1.shape[0]):
        if Y[i] == 1: 
            plt.scatter(X1[i], X2[i], color='red', marker="x")
        else:
            plt.scatter(X1[i], X2[i], color='green', marker="o")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.title("Decision Boundary of Logistic Regression")


    x1_line = np.linspace(-3.5, 3.5, 100)
    x2_line = -(model.weights[0] * x1_line + model.b) / model.weights[1]

    plt.plot(x1_line, x2_line, color='blue', linewidth=2, label='Decision Boundary')

    '''
    Scikit-learn Implementation Demonstration
    '''
    from sklearn.linear_model import LogisticRegression
    print("\nScikit-learn Implementation:\n")
    model = LogisticRegression() 
    model.fit(X,Y)

    print(model.predict(X))


    plt.show()

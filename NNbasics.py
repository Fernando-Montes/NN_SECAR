import numpy as np
import matplotlib.pyplot as plt

def initialize_parameters(n_x, n_h1, n_h2, n_y):
    np.random.seed(2)     
    W1 = np.random.randn(n_h1, n_x)*10.- 5.
    b1 = np.zeros((n_h1,1))
    W2 = np.random.randn(n_h2, n_h1)*6. - 3.
    b2 = np.zeros((n_h2,1)) 
    W3 = np.random.randn(n_y, n_h2)*8. - 4.
    b3 = np.zeros((n_y,1)) 
    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2,
                  "W3": W3,
                  "b3": b3}
    return parameters

def forward_propagation(X, parameters):
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]
    W3 = parameters["W3"]
    b3 = parameters["b3"]
    Z1 = np.dot(W1, X) + b1
    A1 = sigmoid(Z1)
    Z2 = np.dot(W2, A1) + b2
    A2 = sigmoid(Z2)
    Z3 = np.dot(W3, A2) + b3
    A3 = sigmoid(Z3)

    cache = {"Z1": Z1,
             "A1": A1,
             "Z2": Z2,
             "A2": A2,
             "Z3": Z3,
             "A3": A3}
    
    #return A3, cache
    return Z3, cache

#def compute_cost(A3, Y, parameters):
def compute_cost(Z3, Y, parameters):
    m = Y.shape[1]
    #cost = 1/(2*m)*np.dot(Y-A3,(Y-A3).T) 
    cost = 1/(2*m)*np.dot(Y-Z3,(Y-Z3).T) 
    cost = np.squeeze(cost)     # makes sure cost is the dimension we expect. 
                                # E.g., turns [[17]] into 17 
    return cost

def backward_propagation(parameters, cache, X, Y):
    m = Y.shape[1]
    W1 = parameters["W1"]
    W2 = parameters["W2"]
    W3 = parameters["W3"]
    A1 = cache["A1"]
    A2 = cache["A2"]
    A3 = cache["A3"]
    Z3 = cache["Z3"]
    # Backward propagation: calculate dW1, db1, dW2, db2, dW3, db3. 
    #dZ3= np.multiply(np.multiply(A3 - Y, A3),(1-A3))   # logistic regression
    dZ3= Z3 - Y    
    dW3 = 1/m*np.dot(dZ3,A2.T)
    db3 = 1/m*np.sum(dZ3, axis=1, keepdims=True)
    dZ2 = np.multiply(np.dot(W3.T,dZ3),A2-np.power(A2,2))
    dW2 = 1/m*np.dot(dZ2,A1.T)
    db2 = 1/m*np.sum(dZ2, axis=1, keepdims=True)
    dZ1 = np.multiply(np.dot(W2.T,dZ2),A1-np.power(A1,2))
    dW1 = 1/m*np.dot(dZ1,X.T)
    db1 = 1/m*np.sum(dZ1, axis=1, keepdims=True)
    grads = {"dW1": dW1,
             "db1": db1,
             "dW2": dW2,
             "db2": db2,
             "dW3": dW3,
             "db3": db3}
    return grads

def update_parameters(parameters, grads, learning_rate = 1.2):
    # Retrieve each parameter from the dictionary "parameters"
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]
    W3 = parameters["W3"]
    b3 = parameters["b3"]
    # Retrieve each gradient from the dictionary "grads"
    dW1 = grads["dW1"]
    db1 = grads["db1"]
    dW2 = grads["dW2"]
    db2 = grads["db2"]
    dW3 = grads["dW3"]
    db3 = grads["db3"] 
    # Update rule for each parameter
    W1 = W1 - learning_rate*dW1
    b1 = b1 - learning_rate*db1
    W2 = W2 - learning_rate*dW2
    b2 = b2 - learning_rate*db2
    W3 = W3 - learning_rate*dW3
    b3 = b3 - learning_rate*db3
    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2,
                  "W3": W3,
                  "b3": b3}
    
    return parameters

def nn_model(X, Y, n_h1, n_h2, maxNumIter = 1000000, costLim = 0.02, print_cost=False, learning_rate = 1.2, guessPar = None):
    np.random.seed(3)
    n_x = 7
    n_y = 1
    # Initialize parameters, then retrieve W1, b1, W2, b2. Inputs: "n_x, n_h, n_y". Outputs = "W1, b1, W2, b2, parameters".
    if guessPar == None:
        parameters = initialize_parameters(n_x, n_h1, n_h2, n_y)
    else: 
        parameters = guessPar
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]
    W3 = parameters["W3"]
    b3 = parameters["b3"]

    # Forward propagation. Inputs: "X, parameters". Outputs: "A3, cache".
    #A3, cache = forward_propagation(X, parameters)
    Z3, cache = forward_propagation(X, parameters)        
    # Cost function. Inputs: "A2, Y, parameters". Outputs: "cost".
    #cost = compute_cost(A3, Y, parameters)
    cost = compute_cost(Z3, Y, parameters)

    # Loop (gradient descent)
    #for i in range(0, num_iterations):
    i = 0
    while (cost > costLim and i < maxNumIter and learning_rate > 0.0001):
        # Backpropagation. Inputs: "parameters, cache, X, Y". Outputs: "grads".
        grads = backward_propagation(parameters, cache, X, Y)
        # Gradient descent parameter update. Inputs: "parameters, grads". Outputs: "parameters".
        old_parameters = parameters
        parameters = update_parameters(parameters, grads, learning_rate)
        # Print the cost every 5000 iterations
        #if print_cost and i % 100000 == 0:
        #    print ("Cost after iteration %i: %f , learning_rate = %f" %(i, cost, learning_rate))
        # Forward propagation. Inputs: "X, parameters". Outputs: "A3, cache".
        Z3, cache = forward_propagation(X, parameters)        
        # Cost function. Inputs: "A2, Y, parameters". Outputs: "cost".
        old_cost = cost
        cost = compute_cost(Z3, Y, parameters)
        if old_cost < cost: 
            parameters = old_parameters
            learning_rate = learning_rate/2 
        i = i + 1
    print ("Final cost after iteration %i: %f %f" %(i, cost, learning_rate))

    return parameters

def predict(parameters, X):
    A3, cache = forward_propagation(X, parameters)
    return A3

def sigmoid(x):
    s = 1/(1+np.exp(-x))
    return s

    


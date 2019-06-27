import numpy as np
import random
from scipy.stats import norm
from NNbasics import predict

# Calculates uncertainty/sigma using gaussians centered on sampled points
def sig(x_sampled, x, theta):
    tmp = np.asarray([1 - np.exp( -1/theta**2*np.dot(k-x_sampled[0], k-x_sampled[0]) ) for k in x])
    for i in range(1, x_sampled.shape[0]):
        tmp = tmp - [np.exp(      -1/theta**2*np.dot(k-x_sampled[i], k-x_sampled[i]) ) for k in x]  
    tmp = np.asarray([ np.maximum(0.001,k) for k in tmp ])
    return tmp

# Returns probability of improvement
def PI(mean, fxmax, sigma, eps):
    return norm.cdf((mean-fxmax-eps)/sigma)
	
# Sample the PI over phase space and returns next point to try
def samplePS(x, fxmax, parameters, qNom, eps, theta, num_points) :
    x_sample = np.asarray([[random.uniform(qNom[i]*0.8, qNom[i]*1.2) for i in range(7)] for j in range(num_points)]) # Randonmly sample from a uniform distribution
    dim = x_sample.shape[0]
    predY = predict(parameters, x_sample.reshape(7,dim))  # Make predictions of the mean using trained NN
    predY = predY.reshape(-1)
    sigma = sig(x, x_sample, theta)            # Estimate uncertainty/sigma of the mean
    PIx = PI(predY, fxmax, sigma, eps)  # Find the PI of the sample points
    xPImax = np.argmax(PIx)             # Find the arg of the max PI value
    return( x_sample[xPImax] )          # Return the point that has the max PI value


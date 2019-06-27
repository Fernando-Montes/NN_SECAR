import sys, math
import os, shutil, signal
import subprocess as commands
import re
import random
import numpy as np
import matplotlib.pyplot as plt
import multiprocessing

from cosy import cosyrun
from NNbasics import *
from utilities import *

# Hyper-parameters
theta = 0.0016 # sigma: how well we know neighbor resolutions vs fields
eps = 0.5 # Acquisition function (probability/expectation of improvement) parameter
num_points = 100000 # Number of points to sample
n_h1 = 5 # hidden units layer 1
n_h2 = 10 # hidden units layer 1
costLim = 0.00001  # Cost limit to optimize NN
learning_rate = 1.2 # Gradient descent parameter

numSim = int(sys.argv[1]) # Number of simulations
num_steps = int(sys.argv[2]) # Number of steps to do calculation

# Nominal quad fields at pole tip
qNom = [-0.39773, 0.217880+0.001472, 0.242643-0.0005+0.000729, -0.24501-0.002549, 0.1112810+0.00111, 0.181721-0.000093+0.00010-0.000096, -0.0301435+0.0001215] 

def optim(initial, index):
    parameters = None
    x = initial # Starting point
    y = np.asarray([cosyrun(x,index)])
    x = np.asarray([x])
    for i in range(0, num_steps):
        # Neural network preparation
        dim = x.shape[0]
        parameters = nn_model(x.reshape(7,dim), y.reshape(1,len(y)), n_h1 = n_h1, n_h2 = n_h2, costLim = costLim, print_cost=True, learning_rate = learning_rate, guessPar = parameters)
    
        tmp1 = samplePS(x, np.max(y), parameters, qNom, eps, theta, num_points)  # next point to try
        tmp2 = cosyrun(tmp1,index)
        x = np.concatenate ( (x, [tmp1]) , axis=0)
        y = np.concatenate ( (y, [tmp2]) , axis=None)
        #print(i, x, y)
        #print(parameters)
    return 0

# Removing files from older runs
cmd = 'rm -f temp*.pdf'
failure, output = commands.getstatusoutput(cmd)
cmd = 'rm -f simpleOptimization*.pdf'
failure, output = commands.getstatusoutput(cmd)
cmd = 'rm -f results*.txt'
failure, output = commands.getstatusoutput(cmd)

pool = multiprocessing.Pool() # Take as many processes as possible	
for index in range(0, numSim):
    initial = np.asarray([random.uniform(qNom[i]*0.8, qNom[i]*1.2) for i in range(7)]) 
    pool.apply_async( optim, [initial,index] )
pool.close()
pool.join()



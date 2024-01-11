# -*- coding: utf-8 -*-
"""
Created on Sun Jan 10 16:51:31 2021

@author: Narmin Ghaffari Laleh
"""
from scipy.optimize import differential_evolution
from scipy.integrate import odeint
import numpy as np
import warnings
from scipy.integrate import solve_ivp

##############################################################################
# FUNCTIONS TO FIT THE DATA
##############################################################################

def Select_Fucntion(functionName):
    if functionName == 'Exponential':
        def fitfunc(t, alpha, beta, v0):     
            def myode(dim, t):
                return (alpha - beta)*dim        
            Ca0 = v0
            Casol = odeint(myode, Ca0, t)
            return Casol[:,0]
    elif functionName == 'Logistic' :
        def fitfunc(t, alpha, beta, v0):     
            def myode(dim, t):
                return alpha*dim * (1 - (dim/beta))     
            Ca0 = v0
            Casol = odeint(myode, Ca0, t)
            return Casol[:,0]
    elif functionName == 'HyperLogistic' :
        def fitfunc(t, alpha, beta, p, v0):     
            def myode(dim, t):
                return (alpha / beta)* (dim **(1-p))*((beta-dim)**(1+p))    
                #return alpha * dim*(2*np.exp(-p * t)/ 1 + np.exp(-p * t))
            Ca0 = v0
            Casol = odeint(myode, Ca0, t)
            return Casol[:,0]
    elif functionName == 'ClassicBertalanffy' :
        def fitfunc(t, alpha, beta, v0):     
            def myode(dim, t):
                return (alpha * (dim**(2/3))) - (beta*dim)     # Classic Bertalanffy,
            Ca0 = v0
            Casol = odeint(myode, Ca0, t)
            return Casol[:,0]
    elif functionName == 'GeneralBertalanffy' :
        def fitfunc(t, alpha, beta, lamda, v0):             
            def myode(dim, t):
                return (alpha * (dim**lamda)) - (beta*dim)    # General Bertalanffy
            Ca0 = v0
            Casol = odeint(myode, Ca0, t)
            return Casol[:,0]        
    elif functionName == 'Gompertz' :
        def fitfunc(t, alpha, beta, v0):     
            def myode(dim, t): #C, added a bracket, was previously: dim*(beta - alpha* np.log(dim))
                # print(f"alpha {alpha}, beta: {beta}, dim: {dim}")
                # print(f"orginial return: {dim*(beta - alpha* np.log(dim))}")
                # print(f"new return: {dim * (beta - alpha * np.log(dim))}")
                return  dim*(beta - (alpha* np.log(dim)))    # Gompertz
            Ca0 = v0
            Casol = odeint(myode, Ca0, t)
            return Casol[:,0]
    elif functionName == 'GeneralGompertz' :
        def fitfunc(t, alpha, beta, lamda, v0):     
            def myode(dim, t):
                return (dim ** lamda)*(beta-(alpha*np.log(dim)) )   # General Gompertz
            Ca0 = v0
            Casol = odeint(myode, Ca0, t)
            return Casol[:,0]
    elif functionName == 'Bernoulli' :
        def fitfunc(t, alpha, beta, lamda, v0):     
            def myode(dim, t):
                return (alpha * dim) - beta *(dim ** lamda)    # Bernoulli
            Ca0 = v0
            Casol = odeint(myode, Ca0, t)
            return Casol[:,0]
    return fitfunc

##############################################################################
    
def sumOfSquaredError(parameterTuple):
    warnings.filterwarnings("ignore") 
    val = fitFunc(time, *parameterTuple)
    return np.sum((dimension - val) ** 2.0) 

##############################################################################
        
def generate_Initial_Parameters_genetic(ff, k, boundry, t, d, seed = 23, strategy = 'best1bin'):
    global fitFunc
    fitFunc= ff
    global time
    time = t
    global dimension
    dimension = d
    
    parameterBounds = []
    for i in range(k):
        parameterBounds.append(boundry)
    result = differential_evolution(sumOfSquaredError, parameterBounds, seed = seed, strategy = strategy)
    
    return result.x    
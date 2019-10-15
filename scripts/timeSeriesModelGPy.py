# -*- coding: utf-8 -*-
"""
Created on Tue Oct 15 10:05:39 2019

@author: engs1602
"""
from timeSeriesModelGPyUtils import findCI
import numpy as np
import GPy
import matplotlib.pyplot as pp

class timeSeriesModelGPy(object):
    '''
    Trains a GP model for a time-series data:
    
    Based on GPy toolbox. https://github.com/SheffieldML/GPy
    '''
    def __init__(self, kernel = GPy.kern.Matern32(input_dim=1), num_restarts=10, pValue = 0.05, mu = [], cov = [], var = [], xCIleft = [], xCIright = []):
        '''
        Initializes parameters for timeSeriesModelGPy
        @param kernel: GPy kernel
        @param num_restarts: num of restarts during GP optimization
        @param pValue: pValue to compute statistical tetst
        @param mu: GP estimated mean
        @param cov: GP estimated covariace
        @param var: GP estimated variance
        @param xCIleft: GP estimated confidence interval lower bound
        @param xCIright: GP estimated confidence interval upper bound
        '''
        self.kernel = kernel
        self.num_restarts = num_restarts
        self.pValue = pValue
        self.mu = mu 
        self.cov = cov
        self.var = var
        self.xCIleft = xCIleft
        self.xCIright = xCIright
                        
    def train_GP(self,x,y):
        '''
        Trains GP
        @param x: features (numpy array)
        @param y: labels (numpy array)        
        '''
        # Parameter estimation via MLE
        #model = GPy.models.GPRegression(x,y1,kernel1)
        # Parameter estimation via VB        
        model = GPy.models.GPVariationalGaussianApproximation(x,y,self.kernel,GPy.likelihoods.Gaussian())
        fig = model.plot()
        # Options to constrain bounds of the parameters
        #GaussNoiseVarBounds = [1e-3,5]
        #model.Gaussian_noise.constrain_bounded(GaussNoiseVarBounds[0],GaussNoiseVarBounds[1])
        #model.optimize(messages=True)
        #fig = model.plot()
        # Multiple re-starts
        model.optimize_restarts(self.num_restarts)
        fig = model.plot()
        # New plotting style to show density 
        #fig = model.plot(plot_density=True)
        
        #fig, ax = pp.subplots(1,1,figsize=(13,5))
        #model1.plot_f(ax=ax)
        #model1.plot_data(ax=ax)
        #model1.plot_errorbars_trainset(ax=ax, alpha=1)
        #fig.tight_layout()
        #pb.grid()
        
        # Regenerate this plot to verify
        self.mu = model.posterior.mean
        self.cov = model.posterior.covariance
        self.var = np.diag(self.cov)
        
        # Compute CI for plotting
        Nsamples = x.shape[0]
        xCIleft,xCIright = findCI( self.pValue, self.mu.reshape((Nsamples,)), np.sqrt(self.var).reshape((Nsamples,)) )
        self.xCIleft = xCIleft
        self.xCIright = xCIright
        
        return model            
        
    def plot_trained_GP(self,x,y,xYr=[],pValue=0.05):
        '''
        Plot trained GP
        @param xYr: By default xYr = x (numpy array)
        @param x: features (numpy array)
        @param y: labels (numpy array)        
        '''       
        # If years not provided, instead of using years, plot using feature space
        if xYr==[]:
            xYr = x
        Nsamples = x.shape[0]
        ax1 = pp.plot(figsize=(10,6))
        ax1 = pp.plot(xYr,y,'.',color='#1f77b4')
        ax1 = pp.plot(xYr,self.mu,color='#1f77b4')
        ax1 = pp.fill_between(xYr.reshape((Nsamples,)), self.xCIleft, self.xCIright, color='#1f77b4', alpha=.25)
        #ax1 = pp.plot([1989,1989],[min(self.xCIleft),max(self.xCIright)],'-.r',Linewidth=2)
        
    
        
        

# -*- coding: utf-8 -*-
"""
Created on Tue Oct 15 10:05:39 2019

@author: engs1602
"""
from timeSeriesModelGPyUtils import findCI
import numpy as np
import GPy
import matplotlib.pyplot as pp
from scipy.stats import chi2

class timeSeriesModelGPyDiff(object):
    '''
    Computes the difference in two GP models.
    
    GP model trained based on GPy toolbox. https://github.com/SheffieldML/GPy
    Difference in GP models based on paper: http://proceedings.mlr.press/v38/benavoli15.pdf
    '''
    def __init__(self, pValue = 0.05, X=[], Y1=[], Y2=[], deltaMuPost = [], deltaCovPost = [], xCIleft = [], xCIright = []):
        '''
        Initializes parameters for timeSeriesModelGPyDiff
        @param pValue: pValue for computing statistical test
        @param deltaMuPost: Difference in the GP estimated mean vectors
        @param deltaCovPost: Difference in the GP estimated covariance matrices
        @param xCIleft: Confidence interval of the estimated difference (lower bound)
        @param xCIright: Confidence interval of the estimated difference (upper bound)
        '''
        self.pValue = pValue
        self.X = X 
        self.Y1 = Y1
        self.Y2 = Y2
        self.deltaMuPost = deltaMuPost 
        self.deltaCovPost = deltaCovPost
        self.xCIleft = xCIleft
        self.xCIright = xCIright
                        
    def diff_GP_models(self,model1,model2):
        '''
        Trains GP
        @param x: features (numpy array)
        @param y: labels (numpy array)        
        '''
        # Assign parameters based on the models
        self.X = model1.X
        self.Y1 = model1.Y
        self.Y2 = model1.Y
        
        Nsamples = model1.X.shape[0]
        # Difference in two models, here Inward-Outward
        # p(x|N(m,S))
        deltaMuPost = model2.posterior.mean.reshape((Nsamples,)) - model1.posterior.mean.reshape((Nsamples,))
        deltaCovPost = model1.posterior.covariance + model2.posterior.covariance
        
        # Regenerate this plot to verify
        self.deltaMuPost = deltaMuPost
        self.deltaCovPost = deltaCovPost
        
        # Compute CI for plotting
        xCIleft,xCIright = findCI( self.pValue, self.deltaMuPost, np.sqrt(np.diag(self.deltaCovPost)) )
        self.xCIleft = xCIleft
        self.xCIright = xCIright        
        
    def plot_diff_GP_models(self,xYr=[],pValue=0.05):
        '''
        Plot trained GP
        @param xYr: By default xYr = x (numpy array)
        @param x: features (numpy array)
        @param y: labels (numpy array)        
        '''       
        # If years not provided, instead of using years, plot using feature space
        if xYr==[]:
            xYr = self.X
        Nsamples = xYr.shape[0]
        ax4 = pp.plot(xYr.reshape((Nsamples,)),np.zeros(Nsamples),':k')
        ax4 = pp.plot(xYr,self.deltaMuPost,'k')
        ax4 = pp.fill_between(xYr.reshape((xYr.shape[0],)), self.xCIleft, self.xCIright, color='k', alpha=.25)
        labels = ('Zero-vector','Mean Difference','99% CI')
        ax4 = pp.legend(labels,loc='lower right', shadow=True, fontsize='x-large')
        ax4 = pp.plot([1989,1989],[np.min(self.xCIleft),np.max(self.xCIright)],'--r',Linewidth=3)
        ax4 = pp.title('Difference in InOut-OutIn Movement Rates', fontsize='x-large')
        
    def statTest(self,model1,model2,pValue):
        '''
        Stat Sig Test
        Use posterior
        '''
        # If new pValue provided, use that one
        if pValue:
            self.pValue=pValue
        Nsamples = model1.posterior.mean.shape[0]
        deltaMuPost = model1.posterior.mean.reshape((Nsamples,)) - model2.posterior.mean.reshape((Nsamples,))
        deltaCovPost = model1.posterior.covariance + model2.posterior.covariance
        
        xCIleftB,xCIrightB = findCI( pValue, deltaMuPost, np.sqrt(np.diag(deltaCovPost)) )
        testStat = np.matmul(np.matmul(deltaMuPost,np.linalg.inv(deltaCovPost)),deltaMuPost.T)
        eigVal, eigVec = np.linalg.eig(deltaCovPost)
        eigValNorm = eigVal/np.sum(eigVal) 
        df = np.sum(eigValNorm>0)
        pValueCalc = 2*(1-chi2.cdf(testStat,df))
        
        return testStat,df,pValueCalc
        
        

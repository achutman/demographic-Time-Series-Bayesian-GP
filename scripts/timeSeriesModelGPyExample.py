# -*- coding: utf-8 -*-
"""
Created on Tue Oct 15 11:13:40 2019

@author: engs1602

"""
#cd C:\Users\engs1602\research\codes\github\scripts\demographicTimeSeriesBayesianGPy
from timeSeriesModelGPy import timeSeriesModelGPy
from timeSeriesModelGPyDiff import timeSeriesModelGPyDiff
from timeSeriesModelGPyUtils import concatenate_GP_outputs_Pre_Post
from timeSeriesModelGPyUtils import plot_GP_models_Pre_Post
from timeSeriesModelGPyUtils import plot_GP_diff_Pre_Post

import pandas as pd
import GPy
#import pylab as pb
#from sklearn.preprocessing import StandardScaler

# Load data
data = pd.read_csv(r'C:\Users\engs1602\research\codes\github\demographicTimeSeriesBayesianGPy\data\intCountsIoOiNormPerYr.csv')
data = data.set_index('year')

# Plot data
# Outward movement rates (already normalized by mid-year pop by embankment)
data['intIoPer1000InPop'].plot()
# Inward movement rates (already normalized by mid-year pop by embankment)
data['intOiPer1000OutPop'].plot()

## Data pre-processing
# (already normalized by mid-year pop by embankment)
# Remove nan row
data = data.dropna(axis=0)

# Assume [1983,2014] = [0,32]
Nsamples = data.shape[0]
xYr = data.index.values.reshape((Nsamples,1))
# For a GP model: features
x = xYr-xYr[0]
# For a GP model: labels = Outward movement rates 
y1 = data['intIoPer1000InPop'].values.reshape((Nsamples,1))
# For a GP model: labels = Inward movement rates
y2 = data['intOiPer1000OutPop'].values.reshape((Nsamples,1))

# Choose GP parameters, e.g. kernel, noise variance, etc.
kernel = GPy.kern.Matern32(input_dim=x.shape[1])

# PRE vs. POST embankment
# Assume PRE = [1983,1989] 
# Assume POST = [1990,2014] 

# PRE embankment
# Model Outward movement rates with independent GPs
model1Pre = timeSeriesModelGPy(kernel)
model1GpPre = model1Pre.train_GP(x[:7],y1[:7])
model1Pre.plot_trained_GP(x[:7],y1[:7],xYr[:7])
# Model Inward movement rates with independent GPs
model2Pre = timeSeriesModelGPy(kernel)
model2GpPre = model2Pre.train_GP(x[:7],y2[:7])
model2Pre.plot_trained_GP(x[:7],y2[:7],xYr[:7])
# Based on the trained GPs, compute differences in the two time-series
# Here, Inward-Outward
model12DiffPre = timeSeriesModelGPyDiff()
model12DiffPre.diff_GP_models(model1GpPre,model2GpPre)
model12DiffPre.plot_diff_GP_models(xYr[:7])
testStatPre,dfPre,pValueCalcPre = model12DiffPre.statTest(model1GpPre,model2GpPre,.05)
print('testStat=%.2f \t df=%.2d \t pValueCalc=%.2d'%(testStatPre,dfPre,pValueCalcPre))

# POST embankment
# Model inward to outward movement rates with independent GPs
model1Post = timeSeriesModelGPy(kernel)
model1GpPost = model1Post.train_GP(x[7:],y1[7:])
model1Post.plot_trained_GP(x[7:],y1[7:],xYr[7:])
# Model outward to inward movement rates with independent GPs
model2Post = timeSeriesModelGPy(kernel)
model2GpPost = model2Post.train_GP(x[7:],y2[7:])
model2Post.plot_trained_GP(x[7:],y2[7:],xYr[7:])
# Based on the trained GPs, compute differences in the two time-series
model12DiffPost = timeSeriesModelGPyDiff()
model12DiffPost.diff_GP_models(model1GpPost,model2GpPost)
model12DiffPost.plot_diff_GP_models(xYr[7:])
testStatPost,dfPost,pValueCalcPost = model12DiffPost.statTest(model1GpPost,model2GpPost,.05)
print('testStat=%.2f \t df=%.2d \t pValueCalc=%.2d'%(testStatPost,dfPost,pValueCalcPost))

# Concatenate PRE vs POST outputs
# Inward to Outward vs. Outward to Inward
dataOut = concatenate_GP_outputs_Pre_Post(data,model1Pre,model2Pre,model1Post,model2Post,model12DiffPre,model12DiffPost)

# Plot independent GPs PRE vs POST simultaneously
plot_GP_models_Pre_Post(dataOut)

# Plot difference in GPs PRE vs POST simultaneously
plot_GP_diff_Pre_Post(dataOut)

# Save outputs
#pathToSave = r'C:\Users\engs1602\research\meetings\REACH\20180321EmbankmentOutIn\min\dataPlots'
#data.to_csv(os.path.join(pathToSave,'GPvarIoModelOutputsPrePostEmbank.csv'))

# Save outputs                         
# Save a dictionary into a pickle file.
#import pickle
## Io/Oi
#fileName = os.path.join(pathToSave,'GPvarModelIopreEmbankPickle.out')
#pickle.dump( model1OutPre.model, open( fileName, "wb" ) )
#fileName = os.path.join(pathToSave,'GPvarModeOipreEmbankPickle.out')
#pickle.dump( model2OutPre.model, open( fileName, "wb" ) )
#fileName = os.path.join(pathToSave,'GPvarModelIopostEmbankPickle.out')
#pickle.dump( model1OutPost.model, open( fileName, "wb" ) )
#fileName = os.path.join(pathToSave,'GPvarModeOipostEmbankPickle.out')
#pickle.dump( model2OutPost.model, open( fileName, "wb" ) )

#model1Load = pickle.load( open( fileName, "rb" ) )

# Test stats
statTest = pd.DataFrame({'testStat':[testStatPre,testStatPost],
                         'df':[dfPre,dfPost],
                         'pValueCalc':[pValueCalcPre,pValueCalcPost]},
                        index=['Pre','Post'])
#statTest.to_csv(os.path.join(pathToSave,'GPvarIoOiStatTestPrePostEmbank.csv'))







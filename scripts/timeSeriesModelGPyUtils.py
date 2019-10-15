# -*- coding: utf-8 -*-
"""
Created on Tue Oct 15 12:17:35 2019

Additional modules used in other scripts.

@author: engs1602
"""
from scipy import stats
import matplotlib.pyplot as pp
from numpy import concatenate
from numpy import zeros

def findCI(p,Nmean,Nsigma):    
    xxCIleft = stats.norm.ppf(p/2,Nmean,Nsigma)
    xxCIright = stats.norm.ppf(1-p/2,Nmean,Nsigma)
    return xxCIleft,xxCIright

def concatenate_GP_outputs_Pre_Post(data,m1Pre,m2Pre,m1Post,m2Post,mDiffPre,mDiffPost):
    '''
    Concatenate GP outputs for Pre and Post embankment periods
    '''
    data['muIo'] = concatenate((m1Pre.mu,m1Post.mu),axis=0)
    data['varIo'] = concatenate((m1Pre.var,m1Post.var),axis=0)
    data['muxCIleftIo'] = concatenate((m1Pre.xCIleft,m1Post.xCIleft),axis=0)
    data['muxCIrightIo'] = concatenate((m1Pre.xCIright,m1Post.xCIright),axis=0)
    data['muOi'] = concatenate((m2Pre.mu,m2Post.mu),axis=0)
    data['varOi'] = concatenate((m2Pre.var,m2Post.var),axis=0)
    data['muxCIleftOi'] = concatenate((m2Pre.xCIleft,m2Post.xCIleft),axis=0)
    data['muxCIrightOi'] = concatenate((m2Pre.xCIright,m2Post.xCIright),axis=0)
    data['deltaMu'] = concatenate((mDiffPre.deltaMuPost,mDiffPost.deltaMuPost),axis=0)
    data['deltaMuxCIleft'] = concatenate((mDiffPre.xCIleft,mDiffPost.xCIleft),axis=0)
    data['deltaMuxCIright'] = concatenate((mDiffPre.xCIright,mDiffPost.xCIright),axis=0)  
    
    return data

def plot_GP_models_Pre_Post(data):
    '''
    Plot both simulatneously
    Io/Oi
    '''
    Nsamples = data.shape[0]
    ax3 = pp.plot(data.index,data['intIoPer1000InPop'],'.',color='#1f77b4')
    ax3 = pp.plot(data.index,data['muIo'],color='#1f77b4')
    ax3 = pp.fill_between(data.index.values.reshape((Nsamples,)), data['muxCIleftIo'], data['muxCIrightIo'], color='#1f77b4', alpha=.25)
    ax3 = pp.plot(data.index,data['intOiPer1000OutPop'],'.',color='#ff7f0e')
    ax3 = pp.plot(data.index,data['muOi'],color='#ff7f0e')
    ax3 = pp.fill_between(data.index.values.reshape((Nsamples,)), data['muxCIleftOi'], data['muxCIrightOi'], color='#ff7f0e', alpha=.25)                                        
    labels = ('Outward','Mean','Inward','Mean','99% CI','99% CI')
    ax3 = pp.legend(labels,loc='upper right', shadow=True, fontsize='large')
    yAll = concatenate((data['muxCIleftOi'],data['muxCIrightOi']),axis=0)
    ax3 = pp.plot([1989,1989],[min(yAll),max(yAll)],'--r',Linewidth=3)
    ax3 = pp.title('Outward/Inward Movement per 1000 mid-year pop by embank',fontsize='x-large')

def plot_GP_diff_Pre_Post(data):
    '''
    Plot both simulatneously
    Io/Oi
    '''
    Nsamples = data.shape[0]
    ax4 = pp.plot(data.index.values.reshape((Nsamples,)),zeros(Nsamples),':k')
    ax4 = pp.plot(data.index,data['deltaMu'],'k')
    ax4 = pp.fill_between(data.index.values.reshape((Nsamples,)), data['deltaMuxCIleft'], data['deltaMuxCIright'], color='k', alpha=.25)
    #ax4 = pp.xlim([xYr[0][0],xYr[-1][0]])     
    labels = ('Zero-vector','Mean Difference','99% CI')
    ax4 = pp.legend(labels,loc='upper right', shadow=True, fontsize='x-large')
    ax4 = pp.plot([1989,1989],[min(data['deltaMuxCIleft']),max(data['deltaMuxCIright'])],'--r',Linewidth=3)
    ax4 = pp.title('Difference in Inward-Outward Movement Rates', fontsize='x-large')

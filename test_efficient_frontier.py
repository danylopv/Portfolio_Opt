#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  7 17:15:16 2021

@author: danilo
"""

import numpy as np
import numpy.ma as ma
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import math
from scipy.optimize import LinearConstraint

# Here goes a brief description

# stock = ['TSLA','PFE','MSFT','KO','GOOGL','AMZN','MCD','BA','C','FDX']
stock = ['TSLA','AMZN','PFE']


stock_df = yf.download(stock, 
                      start='2019-01-01', 
                      end='2020-12-31', 
                      progress=False)

T=len(stock_df)
t=np.arange(T)

# plt.plot(t,stock_df.Close.TSLA)
# plt.plot(t,stock_df.Close.PFE)

stock_close = stock_df.Close
# Returns calculation
retns = ((stock_close-stock_close.shift(1))/stock_close.shift(1)).iloc[1:]
# retns = (np.log(stock_close/stock_close.shift(1))).iloc[1:] #log_ret

ret_mean = retns.mean()
ret_cov = retns.cov()
ret_corr = retns.corr()

weights = np.ones(len(stock))/len(stock)



def exp_return(ret_mean, weights):
    er = sum(ret_mean.values * weights.T)
    return(er)

def var_return(ret_cov,weights):
    vr = sum(weights * ret_cov.values.dot(weights))
    return(vr)

def efficient_frontier(ret_mean,ret_cov,N):
    m=np.zeros(N)
    std=np.zeros(N)
    l=len(ret_mean)
    gen=np.zeros([N,l])
    for i in range(N):
        generator = np.random.rand(len(ret_mean))
        rdm_weights = generator/sum(generator)
        gen[i,:]=rdm_weights
        m[i]=exp_return(ret_mean,rdm_weights)
        std[i]=math.sqrt(var_return(ret_cov,rdm_weights))
    res=(std,m,gen)
    return(res)
    
def plot_efficient_frontier(ret_mean,ret_cov,N=10000):    
    (std,m,gen)=efficient_frontier(ret_mean,ret_cov,N)

    fig,ax = plt.subplots()
    sc=plt.scatter(std,m)


    annot = ax.annotate("", xy=(0,0), xytext=(20,20),textcoords="offset points",
                    bbox=dict(boxstyle="round", fc="w"),
                    arrowprops=dict(arrowstyle="->"))
    annot.set_visible(False)

    def update_annot(ind):

        pos = sc.get_offsets()[ind["ind"][0]]
        annot.xy = pos
        text = "{}".format(" ".join(list(map(str,gen[ind["ind"]]))))
        annot.set_text(text)
       #annot.get_bbox_patch().set_facecolor(cmap(norm(c[ind["ind"][0]])))
        annot.get_bbox_patch().set_alpha(0.4)


    def hover(event):
        vis = annot.get_visible()
        if event.inaxes == ax:
            cont, ind = sc.contains(event)
            if cont:
                update_annot(ind)
                annot.set_visible(True)
                fig.canvas.draw_idle()
            else:
                if vis:
                    annot.set_visible(False)
                    fig.canvas.draw_idle()

    fig.canvas.mpl_connect("motion_notify_event", hover)


    plt.title("Efficient Frontier")
    plt.xlabel("std")
    plt.ylabel("mean")
    plt.grid()
    plt.show()


    
def get_optimum(ret_mean,ret_cov,N,risk):
    (std,m,gen)=efficient_frontier(ret_mean,ret_cov,N)
    loc1=np.logical_and((0.999*risk<=std),(std<=1.001*risk))
    optimum=max(m[loc1])
    loc2=(m==optimum)
    W=gen[loc2,:]
    s=summary(ret_mean,ret_cov,W)
    res=(s,optimum)
    return(res)
    
def summary(ret_mean,ret_cov,W=np.ones(len(ret_mean))):
    n=len(ret_mean)
    w=pd.Series(index=ret_mean.index.values.tolist(),dtype='float64')
    stdev=pd.Series(index=ret_mean.index.values.tolist(),dtype='float64')
    for i in range(n):
        stdev.iloc[i]=math.sqrt(ret_cov.iloc[i,i])
        w.iloc[i]=W[0][i]
    df=pd.DataFrame({'mean':ret_mean,'std':stdev,'weights':w})
    return(df)
<<<<<<< HEAD
=======

plot_efficient_frontier(ret_mean,ret_cov,500)
>>>>>>> 6dad0caec0e796668c3faf0d8ca19f2d4dd81248

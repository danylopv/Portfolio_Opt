#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  7 17:15:16 2021

@author: danilo
"""

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import math
from scipy.optimize import minimize
from scipy.optimize import Bounds

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

risk=(ret_cov.iloc[0,0]+ret_cov.iloc[1,1]+ret_cov.iloc[2,2])/3

weights = np.ones(len(stock))/len(stock)


def efficient_frontier(ret_mean,ret_cov,N):
    '''Sampling the convex envelope efficient frontier'''
    m=np.zeros(N)
    std=np.zeros(N)
    l=len(ret_mean)
    gen=np.zeros([N,l])
    for i in range(N):
        generator = np.random.rand(len(ret_mean))
        rdm_weights = generator/sum(generator)
        gen[i,:]=rdm_weights
        m[i]=get_exp_return(ret_mean,rdm_weights)
        std[i]=math.sqrt(get_exp_risk(ret_cov,rdm_weights))
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

def get_optimal_return(ret_mean,ret_cov,risk):
    '''Get the optimal portfolio weights provided a risk upper bound'''
    N = len(ret_mean)
    w0 = np.tile(1/N,N)
    objective = lambda w: -get_exp_return(ret_mean,w)
    ineq_cons = {'type': 'ineq',
             'fun' : lambda w: risk-get_exp_risk(ret_cov,w)}
    eq_cons = {'type': 'eq',
               'fun' : lambda w: sum(w)-1}
    bounds = Bounds(np.tile(0,N).tolist(), np.tile(1,N).tolist())
    opts = {'ftol':1e-9,'disp':True}
    opt = minimize(objective,w0,method='SLSQP',
                   constraints=[eq_cons, ineq_cons], 
                   options=opts, bounds=bounds)
    return(opt.x)

def get_optimal_risk(ret_mean,ret_cov,ret):
    '''Get the optimal portfolio weights provided a return lower bound'''
    N = len(ret_mean)
    w0 = np.tile(1/N,N)
    objective = lambda w: get_exp_risk(ret_cov,w)
    ineq_cons = {'type': 'ineq',
             'fun' : lambda w: get_exp_return(ret_mean,w)-ret}
    eq_cons = {'type': 'eq',
               'fun' : lambda w: sum(w)-1}
    bounds = Bounds(np.tile(0,N).tolist(), np.tile(1,N).tolist())
    opts = {'ftol':1e-9,'disp':True}
    opt = minimize(objective,w0,method='SLSQP',
                   constraints=[eq_cons, ineq_cons], 
                   options=opts, bounds=bounds)
    return(opt.x)
    
def get_exp_return(ret_mean,w):
    return(sum(ret_mean*w))

def get_exp_risk(ret_cov,w):
    return(sum(w*ret_cov.dot(w)))

def summary(ret_mean,ret_cov,W=np.ones(len(ret_mean))):
    n=len(ret_mean)
    w=pd.Series(index=ret_mean.index.values.tolist(),dtype='float64')
    stdev=pd.Series(index=ret_mean.index.values.tolist(),dtype='float64')
    for i in range(n):
        stdev.iloc[i]=math.sqrt(ret_cov.iloc[i,i])
        w.iloc[i]=W[0][i]
    df=pd.DataFrame({'mean':ret_mean,'std':stdev,'weights':w})
    return(df)


plot_efficient_frontier(ret_mean,ret_cov,500)
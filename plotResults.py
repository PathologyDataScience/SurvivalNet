# -*- coding: utf-8 -*-
"""
Created on Sat Apr  2 17:24:55 2016

@author: Safoora
"""
import os
import matplotlib.pyplot as plt
import cPickle
import numpy as np

def plotresults(path = '', msr = 'ci', colori = 0, n = 500):
    markers = ['o', '*', '^', 'v', 'x', 's', '<', '>','d', 'p', 'h', '+']   
    colors = ['r', 'b', 'g', 'm', 'c', 'y', 'k', 'w', '#56B4E9', '#A60628', '#8C0900', '#7A68A6']        
    
    loaded_objects = np.empty([1, n]);
    paths = []    
    for f in os.listdir(path):
        if (f.endswith(msr) and 'do.5' in f):
            print f
            paths.append(f)
            
    for i in range(len(paths)):
        if i > 0:
            break        
        f = file(path + paths[i], 'rb')
        #print f                    
        loaded_objects[i] = (cPickle.load(f))
        f.close
    mean = np.mean(loaded_objects, 0)
    std = np.std(loaded_objects, 0)
     
    plt.plot(range(len(mean)), mean, color=colors[colori], marker=markers[colori], lw=2, ms=5, mfc = colors[colori], markevery = 5)
    plt.fill_between(range(len(mean)), mean-std, mean+std,color = colors[colori], alpha = .3)


if __name__ == '__main__':
    #path = os.path.join(os.getcwd(), './results/Brain_P_results/relu/Apr3/')
    #msr = 'ci_tst'
    #plotresults(path, msr, 0, 100)
 
    path = os.path.join(os.getcwd(), './results/')
    msr = 'trn_ci.txt'
    plotresults(path, msr, 2, 500)

    msr = 'tst_ci.txt'
    plotresults(path, msr, 0, 500)

 #   path = os.path.join(os.getcwd(), '../results/Brain_P_results/Relu/Jan/')
 #   msr = 'ci'
 #   plotresults(path, msr, 2, 50)
   
    #plt.plot(range(50), np.ones(50) * .84 , color='c', marker='s', lw=2, ms=5, mfc = 'c', markevery = 5)     

    plt.legend(['trn_cost', 'tst_cost'], loc=4, prop={'size':12});
    #plt.ylim([.50, .9])
    plt.savefig('tf_fig_1-500-.5.pdf')

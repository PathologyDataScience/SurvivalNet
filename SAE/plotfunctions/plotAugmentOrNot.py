# -*- coding: utf-8 -*-
"""
Created on Tue Oct 27 11:33:05 2015

@author: syouse3
"""

import os
import matplotlib.pyplot as plt
import cPickle

def plotresults():
    #measure = 'lpl'
    msr = 'lpl'
    path = '/home/syouse3/git/Survivalnet/SAE/results/LUAD-Oct26/AugmentOrNot/'        
    paths = []    
    for f in os.listdir(path):
        if f.endswith(msr):
            if 'auTrue' in f:
                print f
                paths.append(f)

    markers = ['o', '*', '^', '.', 'v', 'x', 's', 'd', '<', '>',  'p', 'h', '+']   
    colors = ['r', 'b', 'g', 'm', 'c', 'y', 'k', 'w']    
    for i in range(len(paths) -1):
        f = file(path + paths[i], 'rb')
        loaded_objects = []
        loaded_objects = (cPickle.load(f))
        f.close()
        plt.plot(range(len(loaded_objects)), loaded_objects, c=colors[i], marker='s', lw=2, ms=4, mfc=colors[i])
    plt.legend([]);
    #plt.ylim([-500, 0]);
    #plt.ylim([.50, .80]);
    #plt.xlim([0, 100])
    paths = []    
    for f in os.listdir(path):
        if f.endswith(msr):
            if 'auFalse' in f:
                print f
                paths.append(f)

    for i in range(len(paths) -1):
        f = file(path + paths[i], 'rb')
        loaded_objects = []
        loaded_objects = (cPickle.load(f))
        f.close()
        plt.plot(range(len(loaded_objects)), loaded_objects, c=colors[i], marker='o', lw=2, ms=4, mfc=colors[i])
    plt.legend(['2*60 -aug', '6*60 -aug','2*60','6*60'], loc=2,prop={'size':6})
    if msr == 'lpl':    
        plt.ylim([-1000, 0])
    else:
        plt.ylim([.40, .80])
    #plt.show()
    plt.savefig('augOrNot' + msr + '.png', format='png')
    plt.savefig('augOrNot' + msr + '.eps', format='eps')
if __name__ == '__main__':
    plotresults()

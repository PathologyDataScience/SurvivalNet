# -*- coding: utf-8 -*-
"""
Created on Tue Oct 20 12:53:54 2015

@author: syouse3
"""
import os
import matplotlib.pyplot as plt
import cPickle

def plotresults():
    #measure = 'lpl'
    msr = 'lpl'
    path = '/home/syouse3/git/Survivalnet/SAE/results/LUAD-Oct26/DropOutFractions/'        
    paths = []    
    for f in os.listdir(path):
        if f.endswith(msr):
            print f
            paths.append(f)

    markers = ['o', '*', '^', '.', 'v', 'x', 's', 'd', '<', '>',  'p', 'h', '+']   
    colors = ['r', 'b', 'g', 'm', 'c', 'y', 'k', 'w']    
    for i in range(len(paths)):
        f = file(path + paths[i], 'rb')
        loaded_objects = []
        loaded_objects = (cPickle.load(f))
        f.close()
        plt.plot(range(len(loaded_objects)), loaded_objects, c=colors[i], marker=markers[i], lw=2, ms=4, mfc=colors[i])
    plt.legend(['22*200 do.0','22*200 do.1', '22*200 do.3', '22*200 do.5', '22*200 do.7'], loc=2, prop={'size':6});

    #plt.show()
    plt.savefig('doRatesonDeepModel' + msr + '.png', format='png')
    plt.savefig('doRatesonDeepModel' + msr + '.eps', format='eps')
    
if __name__ == '__main__':
    plotresults()

# -*- coding: utf-8 -*-
"""
Created on Tue Oct 20 12:53:54 2015

@author: syouse3
"""
import os
import matplotlib.pyplot as plt
import cPickle

def plotresults():
    msr = 'ci'
    path = '/home/syouse3/git/Survivalnet/SAE/results/LUAD-Oct26/FinalExperiment/'        
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
        print 'max %d    start %d    end %d' % (max(loaded_objects), loaded_objects(0), loaded_objects(len(loaded_objects) - 1))
        f.close()
        plt.plot(range(len(loaded_objects)), loaded_objects, c=colors[i], marker=markers[i], lw=2, ms=4, mfc=colors[i])
        plt.legend(['6 layers', '10 layers','14 layers','18 layers','22 layers'], loc=2, prop={'size':6});
    if msr == 'lpl':
        a = 1
        #plt.ylim([-500, 0])
    else:
        plt.ylim([.40, .80])
    #plt.show()
    plt.savefig('final' + msr + '.png', format='png')
    plt.savefig('final' + msr + '.eps', format='eps')

if __name__ == '__main__':
    plotresults()

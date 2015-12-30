# -*- coding: utf-8 -*-
"""
Created on Tue Oct 20 12:53:54 2015

@author: syouse3
"""
import os
import matplotlib.pyplot as plt
import cPickle
import numpy as np

def plotresults():
    msr = 'ci'
    path = '/Users/Ayine/pySurv/LUSC_results/sigmoid/'        
    paths = []    
    for f in os.listdir(path):
        if ('nl3' in f and f.endswith(msr)):
            print f
            paths.append(f)

    markers = ['o', '*', '^', 'v', 'x', 's', '<', '>','d', 'p', 'h', '+']   
    colors = ['r', 'b', 'g', 'm', 'c', 'y', 'k', 'w', '#56B4E9', '#A60628', '#8C0900', '#7A68A6']    
    for i in range(len(paths)):
        if i > 11:
            break        
        f = file(path + paths[i], 'rb')
        #print f                    
        loaded_objects = []
        loaded_objects = (cPickle.load(f))
        #print 'max %d    start %d    end %d' % (max(loaded_objects), loaded_objects(0), loaded_objects(len(loaded_objects) - 1))
        f.close()
        #print i
        #plt.plot(range(len(loaded_objects)), loaded_objects, color='g', marker=markers[i], lw=5, ms=5, mfc = 'g')
              
        if 'nl10' in paths[i]:        
            plt.plot(range(len(loaded_objects)), loaded_objects, color='r', marker=markers[i], lw=2, ms=5, mfc = 'r', markevery = 5)
        elif 'nl1' in paths[i]:
#        elif i == 1 or i == 2 or i == 3:
            plt.plot(range(len(loaded_objects)), loaded_objects, color='b', marker=markers[i], lw=2, ms=5, mfc = 'b', markevery = 5)
#        elif i == 4 or i == 5 or i == 6:
        elif 'nl5' in paths[i]:
            plt.plot(range(len(loaded_objects)), loaded_objects, color='g', marker=markers[i], lw=2, ms=5, mfc = 'g', markevery = 5)
        else:
            plt.plot(range(len(loaded_objects)), loaded_objects, color='c', marker=markers[i], lw=2, ms=5, mfc = 'c', markevery = 5)
#        #plt.show()
        plt.plot(range(len(loaded_objects)), np.ones(500) * .49 , color='r', marker=markers[i + 1], lw=2, ms=5, mfc = 'r', markevery = 5)        
    #plt.legend(['cox', '1*150','1*300','1*5', '1*50', '10*150','10*300', '10*5', '10*50','5*150', '5*300', '5*5', '5*50'], loc=4, prop={'size':12});
    #plt.legend(paths, loc=4, prop={'size':12});
    #plt.legend(['lr .0001', 'lr .0001', 'lr .001', 'lr .001','lr .01', 'lr .01'], loc=3, prop={'size':6});
    #plt.legend([ 'nn dropout 0', 'nn dropout 0.1','nn dropout 0.3', 'nn dropout 0.5', 'nn dropout 0.7','nn dropout 0.9'], loc=4, prop={'size':12});
    #plt.legend(['training', 'testing'], loc=4, prop={'size':12});
    #plt.legend(['cox', '1*150', '300', '1*5', '1*50', '1*150 PRE-TRAIN', '300', '1*5 PRE-TRAIN', '1*50 PRE-TRAIN'], loc=4, prop={'size':12});
    plt.legend(['3*83 NN', 'cox'], loc=4, prop={'size':12})   
    #if msr == 'lpl':
     #   a = 1
    #plt.ylim([-240, - 190])
    #else:
        #plt.ylim([.40, .80])
    plt.show()
    #plt.savefig('final' + msr + '.png', format='png')
    #plt.savefig('final' + msr + '.eps', format='eps')

if __name__ == '__main__':
    plotresults()

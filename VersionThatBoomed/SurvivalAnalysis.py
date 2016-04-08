# -*- coding: utf-8 -*-
"""
Created on Wed Mar  9 11:00:38 2016

@author: Ayine
"""
import numpy as np

class SurvivalAnalysis(object):
    """ This class contains methods used in survival analysis
    """
    def calc_at_risk(self, X, T, O):
        """
        Calculate the at risk group of all patients. For every patient i, this
        function returns the index of the first patient who died after i, after
        sorting the patients w.r.t. time of death.
        Refer to the definition of
        Cox proportional hazards log likelihood for details: https://goo.gl/k4TsEM
        
        Parameters
        ----------
        X: numpy.ndarray
           m*n matrix of expression data
        T: numpy.ndarray
           m sized vector of time of death
        O: numpy.ndarray
           m sized vector of observed status (1 - censoring status)

        Returns
        -------
        X: numpy.ndarray
           m*n matrix of expression data sorted w.r.t time of death
        T: numpy.ndarray
           m sized sorted vector of time of death
        O: numpy.ndarray
           m sized vector of observed status sorted w.r.t time of death
        at_risk: numpy.ndarray
           m sized vector of starting index of risk groups
        """
        tmp = list(T)
        T = np.asarray(tmp).astype('float64')
        order = np.argsort(T)
        sorted_T = T[order]
        at_risk = np.asarray([list(sorted_T).index(x) for x in sorted_T]).astype('int32')
        T = np.asarray(sorted_T)
        O = O[order]
        X = X[order]
        return X, T, O, at_risk
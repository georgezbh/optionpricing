'Autocallable pricing module'
'with Monte Carlo simulation to price'

__author__ = 'George Zhao'


import math

import statistics

import random

import numpy as np

import matplotlib.pyplot as plt

from scipy.stats import norm


class Autocall_CCBN(object):  'Autocallable contingent coupon barrier notes'

    def __init__(self, coupon, principal, call_level, barrier, strike, fixing_schedule):

    self._coupon = coupon

    self._principal = principal

    self._call_level = call_level  

    self._barrier = barrier

    self._strike = _strike

    self._fixing_schedule = fixing_schedule


    self._npaths_mc = 10000
    self._nsteps_mc = 300
    self._rnd_seed = 10000

    self._delta_bump=1
    self._delta_bump_is_percent=True
    self._gamma_bump=1
    self._gamma_bump_is_percent=True
    self._vega_bump=0.01
    self._vega_bump_is_percent=False
    self._theta_bump=1/365
    self._vanna_dvega = False
    self._rho_bump = 0.001

    self._spot_minimum = 10e-6


    def gen_fixings(self,spot,vol,r,div):

        if spot<=0:
            spot = self._spot_minimum

        s_fix =[]

        t_fix = self._fixing_schedule

        for index in range(len(self._fixing_schedule)):

            if index ==0:

                t = t_fix[index]

                sigma = math.sqrt(t)

                fix = spot * math.exp((r-div-0.5*vol**2)*t+vol*random.gauss(0,sigma))

            else:

                t = t_fix[index] - t_fix[index-1]

                sigma = math.sqrt(t)

                fix = s_fix[index-1] * math.exp((r-div-0.5*vol**2)*t+vol*random.gauss(0,sigma))

            s_fix.append(fix)

        return s_fix

    
    def pricer(self,spot,vol,rate,div,debug=False):

        n_paths = self._npaths_mc
        rndseed = self._rnd_seed

        if rndseed is not None:
            random.seed(rndseed)

        T= self._fixing_schedule[-1] 

        K = self._strike

        C = self._call_level

        coupon = self._coupon

        B = self._barrier

        N = self._principal

        t_fix = self._fixing_schedule

        n_fix = len(t_fix)  # total number of fixings is n_fix

        payoff_by_path = []

        for i in range(n_paths):
            
            s_fix = self.gen_fixings(spot,vol,rate,div)

            payoff = 0 

            for j in range(n_fix):

                df = math.exp(-rate * t_fix[j])

                if s_fix[j] < C:

                    if s_fix[j] >= B:

                        payoff = df * N * coupon + payoff    
                
                    elif s_fix[j] < B:

                        pass

                elif s_fix[j] >= C:

                    payoff = df * N * coupon + payoff  

                    break



            
            payoff_by_path.append(payoff)


        














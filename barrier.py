#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'Barrier option pricing module'
'with PDE method'

__author__ = 'George Zhao'


import math

import statistics

import random

import numpy as np

import matplotlib.pyplot as plt

from scipy.stats import norm

class Barrier(object):
    
    def __init__(self, barrier, barrier_type, strike, cp, maturity, quantity):

        self._barrier = barrier
        self._barrier_type = barrier_type
        self._strike = strike
        self._cp = cp
        self._maturity = maturity
        self._quantity = quantity

        # default parameters

        self._delta_bump=0.1
        self._delta_bump_is_percent=True
        self._gamma_bump=0.1
        self._gamma_bump_is_percent=True
        self._vega_bump=0.01
        self._vega_bump_is_percent=False
        self._theta_bump=1/365
        self._rho_bump = 0.01
        self._rho_bump_is_percent=False

        self._vanna_dvega = False

        self._tsteps_pde = 100
        self._ssteps_pde = 200
        self._spot_max_factor_pde = 5

        self._spot_minimum = 10e-6

        def forward(self,spot,rate_c,rate_a):
        
            fwd=math.exp((rate_c-rate_a)*self._maturity)*spot

            return fwd

        def pde(self,spot,vol,rate_c,rate_a,greeks='pl'):

            Q = self._quantity
            
            exp = math.exp

            if spot<=0:

                spot = self._spot_minimum

            if self._cp.lower()=='call' or self._cp.lower()=='c':

                cp='c'

            else:

                cp='p'

            s_steps = self._ssteps_pde

            t_steps = self._tsteps_pde

            T = self._maturity

            dt = T / t_steps

            K = self._strike

            B = self._barrier

            B_type = self._barrier_type

            spot_max = K * self._spot_max_factor_pde

            if B_type == 'UO' and spot >=B:

                pl = 0

            elif B_type == 'DO' and spot <=B:

                pl = 0

            elif B_type == 'UI' and spot < B:

                pl = 0

            elif B_type == 'DI' and spot > B:

                pl = 0

            else:

                if spot > spot_max: # if the spot is outside the grids

                    ds = spot_max / s_steps

                    fwd= self.forward(spot,rate_c,rate_a)

                    fwd_shift = spot * math.exp((rate_c - rate_a) * (T-dt))

                    spot_up = spot + ds
                    spot_down = spot -ds

                    fwd_up = self.forward(spot_up,rate_c,rate_a)
                    fwd_down = self.forward(spot_down,rate_c,rate_a)

                    if cp=='c':

                        pl = max(fwd-K,0) * math.exp(-rate_c*T)

                        pl_up = max(fwd_up-K,0) * math.exp(-rate_c*T)
                        pl_down = max(fwd_down-K,0) * math.exp(-rate_c*T)

                        pl_shift = max(fwd_shift-K,0) * math.exp(-rate_c*(T-dt))

                    else:

                        pl=max(K-fwd,0) * math.exp(-rate_c*T)
                        pl_up=max(K-fwd_up,0) * math.exp(-rate_c*T)
                        pl_down=max(K-fwd_down,0) * math.exp(-rate_c*T)
                        pl_shift = max(K-fwd_shift,0) * math.exp(-rate_c*(T-dt))
                    
                    delta = (pl_up - pl_down) / (2*ds)
                    gamma = (pl_up-2*pl + pl_down) / (ds**2)
                    theta = (pl_shift - pl)/ dt * self._theta_bump

                else:   # if the spot is within the grids
        
                    ds = spot_max / s_steps

                    onnode = True

                    x = spot/ds

                    x0 = int(x)
                    x1 = x0+1

                    if x1-x <= 10e-8:
                        spot_index = x1   # spot falls on x1
                    elif x-x0 <= 10e-8:
                        spot_index = x0   # spot falls on x0
                    else:
                        onnode=False 

                    spot_rng = np.linspace(0,spot_max,s_steps+1)
        #############################################################################################################################

                    grid= np.zeros((s_steps+1,t_steps+1))

                    for i in range(s_steps+1):  # boundry condition at T: for payoff corresponds to each spot prices at maturity


                        if B_type == 'UO' and spot_rng[i] >=B:

                            grid[i,t_steps] = 0

                        elif B_type == 'UI' and spot_rng[i] < B:

                            grid[i,t_steps] = 0

                        elif B_type == 'DO' and spot_rng[i] <= B:

                            grid[i,t_steps] = 0
                        
                        elif B_type == 'DI' and spot_rng[i] > B:

                            grid[i,t_steps] = 0

                        else:

                            if cp=='c':

                                grid[i,t_steps] = max (spot_rng[i]-K,0) 

                            else:
                                grid[i,t_steps] = max (K-spot_rng[i],0) 
            

                    for j in range(t_steps): # boundry condition at spot =0 and spot = s_max

                        DF_t =  exp(-rate_c*(T-j*dt))

                        F_t = spot_rng[s_steps] * exp((rate_c-rate_a) * (T-j*dt))

                        if cp=='c':

                            grid[0,j] = 0

                            grid[s_steps,j] = max(F_t - K,0) * DF_t
                        
                        else:

                            grid[0,j] = max(K,0)* DF_t

                            grid[s_steps,j] = 0

                        if B_type == 'UO' and spot_rng[s_steps] >=B:

                            grid[s_steps,j] = 0

                        elif B_type == 'UI' and spot_rng[s_steps] < B:

                            grid[s_steps,j] = 0

                            grid[0,j] = 0

                        elif B_type == 'DO' and spot_rng[s_steps]<= B:

                            grid[s_steps,j] = 0
                            grid[0,j] = 0
                        
                        elif B_type == 'DI' and spot_rng[s_steps] > B:

                            grid[s_steps,j] = 0
                        





                        

                    for t in range(t_steps-1,-1,-1):  # from t=t_step-1 to t=0

                        A = np.zeros((s_steps-1,s_steps-1))

                        B = np.zeros((1,s_steps-1))

                        for i in range(1,s_steps):   # index from 1 to s_steps-1

                            if i==1 and cp=='c':  # Van Neuman boundary condition=0 and secondary order condition=0 for call

                                a=0
                                b=-1/dt-rate_c
                                c=0
                                d=-1/dt

                            elif i==1 and cp=='p': # Van Neuman boundary condition=-exp(-q*T) and secondary order condition=0 for call
                                a=0
                                b=-1/dt-rate_c
                                c=0
                                d=-1/dt

                            else:   

                                a = 0.5* (vol**2) * (i**2) - 0.5 * (rate_c-rate_a)*i
                                b = -1/dt - (vol**2)*(i**2)- rate_c
                                c = 0.5 * (vol**2) * (i**2) + 0.5* (rate_c - rate_a)*i
                                d =- 1/ dt
                            
                            # construct matrix A and B in AX=B
                            if i == 1:
                                if cp=='c':
                                    B[0,i-1] = d * grid[i,t+1] -  a * grid[i-1,t]  
                                else:
                                    B[0,i-1] = d * grid[i,t+1] -  a * grid[i-1,t]  +(rate_c-rate_a) * spot_rng[i] * math.exp(-rate_a*(T-t*dt))

                                A[i-1,i]=c
            
                            elif i == s_steps-1:

                                B[0,i-1]=d*grid[i,t+1]-c*grid[i+1,t]

                                A[i-1,i-2] =a

                            else:

                                B[0,i-1]=d*grid[i,t+1]

                                A[i-1,i-2] =a
                                A[i-1,i]=c

                            A[i-1,i-1]=b

                        V = np.linalg.solve(A,B.T)

                        if style == 'a':

                            for i in range(s_steps-1):

                                if cp == 'c':

                                    V[i] = max(V[i], spot_rng[i+1]-K)

                                else:

                                    V[i] = max(V[i], K-spot_rng[i+1])

                        grid[1:s_steps,t] = V[:,0]

                    # plt.figure()
                    # plt.plot(spot_rng,grid[0:s_steps+1,0] ,label='pde test')
                    # plt.show()

        #########################################################################################################

                    if onnode == True : # spot falls on node

                        if spot_index == 0:

                            y_up = grid[spot_index+2,0]

                            y = grid[spot_index+1,0]

                            y_down = grid[spot_index,0]

                            pl = y_down

                            delta = (y - y_down) /ds

                        elif spot_index == s_steps:

                            y_up = grid[spot_index,0]

                            y = grid[spot_index-1,0]

                            y_down = grid[spot_index-2,0]

                            pl = y_up

                            delta = (y_up - y)/ds

                        else:

                            y_up = grid[spot_index+1,0]

                            y = grid[spot_index,0]

                            y_down = grid[spot_index-1,0]

                            pl = y

                            delta = (y_up - y_down)/(ds*2)

                        pl_shift = grid[spot_index,1]

                        gamma = (y_up -2*y + y_down)/(ds**2)

                        theta = (pl_shift-pl) / dt * self._theta_bump
                
                    else:  # spot falls in between nodes

                        pl0 = grid[x0,0]
                        pl1 = grid[x0+1,0]

                        pl0_shift = grid[x0,1]
                        pl1_shift = grid[x0+1,1]
                        
                        if x0 == 0:

                            if cp == 'c':

                                delta0 = 0

                            else:

                                delta0 = - math.exp(-rate_a*T)

                            # delta0 = (grid[x0+1,0]- grid[x0,0])/ds
                            delta1 = (grid[x0+2,0]- grid[x0,0])/(2*ds)

                            gamma0 = 0 
                            gamma1 = (grid[x0+2,0] - 2*grid[x0+1,0] + grid[x0,0]) / (ds**2)

                        elif x0 == s_steps -1:

                            delta0 = (grid[x0+1,0]- grid[x0-1,0])/(2*ds)
                            # delta1 = (grid[x0+1,0]- grid[x0,0])/(2*ds)

                            if cp == 'c':

                                delta1 = math.exp(-rate_a*T)

                            else:

                                delta1 = 0

                            gamma0 = (grid[x0+1,0] - 2*grid[x0,0] + grid[x0-1,0]) / (ds**2)
                            gamma1 = 0

                        else:

                            delta0 = (grid[x0+1,0]- grid[x0-1,0])/(2*ds)
                            delta1 = (grid[x0+2,0]- grid[x0,0])/(2*ds)
                            
                            gamma0 = (grid[x0+1,0] - 2*grid[x0,0] + grid[x0-1,0]) / (ds**2)
                            gamma1 = (grid[x0+2,0] - 2*grid[x0+1,0] + grid[x0,0]) / (ds**2)

                        gamma =gamma0 + (gamma1 - gamma0) * (x-x0)

                        delta = delta0+(x-x0)*(delta1-delta0)   # using naive interpolation
                        #delta = delta0 + gamma * (spot - spot_rng[x0])  # delta from Taylor expansion using gamma value
                        
                        pl = pl0 + (x-x0)* (pl1-pl0) 
                        # pl = pl0 + delta * (spot - spot_rng[x0]) + 0.5 * gamma * (spot - spot_rng[x0])**2

                        pl_shift = pl0_shift + (x-x0)* (pl1_shift-pl0_shift)

                        theta = (pl_shift - pl)/dt * self._theta_bump



        
            






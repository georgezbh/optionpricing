#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'Barrier option pricing module'
'with PDE methods (explicit and implicit) and Monte Carlo simulation method'

__author__ = 'George Zhao'

from math import sqrt, exp, log

import numpy as np

import matplotlib.pyplot as plt

from scipy.stats import norm
from scipy.interpolate import interp1d

from multiprocessing import Process, Pool
import os


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
        self._spot_max_factor_pde = 3
        self._spot_minimum = 10e-6
        self._displayprogress = True

        self._npaths_mc = 1000
        self._nsteps_mc = 300
        self._rnd_seed = 12345

        self._default_model = self.pde


    def forward(self,spot_array,rate_c,rate_a):
    
        fwd_array = exp((rate_c-rate_a)*self._maturity) * spot_array

        return fwd_array


    def bsm(self,spot_array,vol,rate_c,rate_a,greeks='mv'):

        if self._displayprogress == True: print('Black-Scholes Model \n')

        spot_array[spot_array<=0]=self._spot_minimum
      
        fwd_array = self.forward(spot_array,rate_c,rate_a)

        if self._cp.lower() == 'call' or self._cp.lower()=='c':

            cp = 'c'

        else:

            cp = 'p'

        Q = self._quantity
        T = self._maturity
        K = self._strike
        Df = exp(-rate_c * T)
        Cf = exp(-rate_a * T)

        N=norm.cdf     # standard normal distribution cumulative function
        N_pdf = norm.pdf

        d1 = (np.log(fwd_array/K)+0.5*vol**2*T)/(vol * sqrt(T)) 
        d2 = d1 - vol * sqrt(T)

        if greeks.lower() =='delta':

            call_delta = N(d1) * Cf
            put_delta = (N(d1)-1) * Cf

            if cp =='c':

                return call_delta * Q

            else:

                return put_delta * Q

        elif greeks.lower()=='gamma':

            gamma = Cf * N_pdf(d1) / (spot_array * vol * sqrt(T))

            return gamma * Q

        elif greeks.lower()=='vega':

            vega = spot_array * N_pdf(d1) * sqrt(T) * Cf / 100

            return vega * Q
        
        elif greeks.lower()=='theta':

            call_theta = (-spot_array * N_pdf(d1) * vol * Cf /(2*sqrt(T)) - rate_c * K * Cf * N(d2) + rate_a * spot_array * Cf* N(d1)) * self._theta_bump
            put_theta = (-spot_array * N_pdf(d1) * vol * Cf /(2*sqrt(T)) + rate_c * K * Cf * N(-d2)-rate_a * spot_array * Cf * N(-d1)) * self._theta_bump

            if cp =='c':

                return call_theta * Q 

            else:

                return put_theta * Q

        elif greeks.lower()=='rho':

            call_rho = K * T * Df * N(d2) / 100
            put_rho = -K * T * Df * N(-d2) / 100

            if cp =='c':

                return call_rho * Q

            else:

                return put_rho * Q

        else:

            call_price = Df * (N(d1) * fwd_array - N(d2) * K) 
            put_price = Df * (N(-d2) * K - N(-d1) * fwd_array)

            if cp =='c':

                return call_price * Q

            else:

                return put_price * Q
                

    def pde(self,spot,vol,rate_c,rate_a,greeks='mv'): 

        if self._displayprogress == True: print('PDE(implicit) \n')

        Q = self._quantity
        
        if self._cp.lower() in ('call','c'):  # payoff factor, if call then 1 else if put then -1

            cp = 1     

        else:

            cp = -1

        s_steps = self._ssteps_pde

        t_steps = self._tsteps_pde

        T = self._maturity

        dt = T / t_steps

        K = self._strike

        B_level = self._barrier

        B_type = self._barrier_type

        spot[spot<=0] = self._spot_minimum

        if B_type in ('UO','UI'):

            spot_max = B_level
        
        else:

            spot_max = max(K,B_level) * self._spot_max_factor_pde  # to make sure strike and barrier are both within lattice

        Df = exp(-rate_c * T)  # discount factor from maturity
    
        ds = spot_max / s_steps

        spot_rng = np.linspace(0,spot_max,s_steps+1)   # spot array from S = 0 to S = spot_max

        grid= np.zeros((s_steps+1,t_steps+1))

        p_T = np.maximum(cp * (spot_rng - K),0) # vanilla payoff at time T

        if B_type in ['UI','UO']:

            f = 1 * (spot_rng < B_level)  # for up & out option, f=1 only if spot < Barrier

        elif B_type in ['DI','DO']:

            f = 1 * (spot_rng > B_level)  # for down & out option, f=1 only if spot > Barrier

        grid[:,-1] = p_T * f # boundry condition at T: for payoff corresponds to each spot prices at maturity

        time_rng = np.arange(t_steps) * dt

        DF_t = np.exp(-rate_c * time_rng) # discount factor array, size t_steps x 1

        F_t = spot_max * np.exp((rate_c-rate_a)*(T-time_rng))  # forward price along the spot upper boundary, size t_steps x 1

        grid[0,:t_steps] = max(cp * (0-K),0) * DF_t # boundry condition for spot = 0 along each time step

        g_temp = np.maximum(cp * (F_t - K),0)

        grid[s_steps,:t_steps] = g_temp * DF_t # boundry condition for spot = spot_max along each time step

        if (B_type in ['UI','UO']):

            grid[s_steps,:] = 0
        
        elif (B_type in ['DI','DO']):

            grid[0,:]=0
        
        for t in range(t_steps-1,-1,-1):  # from t=t_step-1 to t=0

            if (B_type in ['UI','UO']):

                i_uo = max(np.where(spot_rng < B_level)[0])
                A = np.zeros((i_uo,i_uo))
                B = np.zeros((1,i_uo))

                i_start = 1
                i_end = i_uo+1

            elif (B_type in ['DI','DO']):

                i_do = min(np.where(spot_rng > B_level)[0])
                A = np.zeros((s_steps-i_do,s_steps-i_do))
                B = np.zeros((1,s_steps-i_do))

                i_start = i_do
                i_end = s_steps

            for i in range(i_start,i_end):   # index from 1 to s_steps-1

                a = 0.5* (vol**2) * (i**2) - 0.5 * (rate_c-rate_a) * i
                b = -1/dt - (vol**2)*(i**2)- rate_c

                c = 0.5 * (vol**2) * (i**2) + 0.5 * (rate_c - rate_a) * i
                d = - 1/ dt
                
                # construct matrix A and B in AX=B
                if i == i_start:

                    B[0,i-i_start] = d * grid[i,t+1] -  a * grid[i-1,t]  
            
                    A[i-i_start,i-i_start+1]=c


                elif i == i_end-1:

                    B[0,i-i_start]=d*grid[i,t+1]-c*grid[i+1,t]

                    A[i-i_start,i-i_start-1] =a

                else:

                    B[0,i-i_start]=d*grid[i,t+1]

                    A[i-i_start,i-i_start-1] = a
                    A[i-i_start,i-i_start+1] = c

                A[i-i_start,i-i_start] = b

            V_t = np.linalg.solve(A,B.T)

            grid[i_start:i_end,t] = V_t[:,0]

                # plt.figure()
                # plt.plot(spot_rng,grid[0:s_steps+1,0] ,label='pde test')
                # plt.show()

        V_t0 = grid[:,0]
        V_t1 = grid[:,1]
        x = spot
        x_u = spot + ds
        x_d = np.maximum(spot - ds,0)

        if B_type in ('UI','UO'):

            f_t0 = interp1d(spot_rng,V_t0,kind='linear',bounds_error=False,fill_value=0)

            f_t1 = interp1d(spot_rng,V_t1,kind='linear',bounds_error=False,fill_value=0)

        elif B_type in ('DI', 'DO'):

            f_t0 = interp1d(spot_rng,V_t0,kind='linear',fill_value='extrapolate')

            f_t1 = interp1d(spot_rng,V_t1,kind='linear',fill_value='extrapolate')   

        mv_out = f_t0(x) * Q

        BSM = self.bsm

        if greeks.lower() in ('pl','mv'):

            if B_type in ['UI','DI']: 
                
                mv = BSM(spot,vol,rate_c,rate_a,'mv') - mv_out

            else:
                
                mv = mv_out

            return mv
        
        elif greeks.lower() == 'delta':

            delta_out = (f_t0(x_u) - f_t0(x_d)) / (x_u - x_d) * Q

            if B_type in ['UI','DI']: 
                
                delta = BSM(spot,vol,rate_c,rate_a,'delta') - delta_out

            else:
                
                delta = delta_out

            return delta

        elif greeks.lower() == 'gamma':

            gamma_out = (f_t0(x_u) - 2 * f_t0(x) + f_t0(x_d)) / ds**2 * Q

            if B_type in ['UI','DI']: 
                
                gamma = BSM(spot,vol,rate_c,rate_a,'gamma') - gamma_out

            else:
                
                gamma = gamma_out

            return gamma

        elif greeks.lower() == 'theta':

            theta_out = (f_t1(x) - f_t0(x))/dt * self._theta_bump * Q

            if B_type in ['UI','DI']: 
                
                theta = BSM(spot,vol,rate_c,rate_a,'theta') - theta_out

            else:
                
                theta = theta_out

            return theta

 
    def pde2(self,spot,vol,rate_c,rate_a,greeks='mv'):   # PDE using explicit finite differece method

        if self._displayprogress == True: print('PDE(explicit) \n')

        Q = self._quantity

        K = self._strike

        B_level = self._barrier

        B_type = self._barrier_type
        
        if spot<=0:

            spot = self._spot_minimum

        if self._cp.lower()=='call' or self._cp.lower()=='c':  # payoff factor, if call then 1 else if put then -1

            cp = 1     

        else:

            cp = -1

        s_steps = self._ssteps_pde

        t_steps = self._tsteps_pde

        T = self._maturity

        # t_steps_min = int(T*(1+vol**2*s_steps**2))

        # t_steps = max(t_steps, t_steps_min)

        dt = T / t_steps

        if B_type in ('UI', 'UO'):

            spot_max = B_level

            s_steps_max = int(spot_max / (vol*B_level) * sqrt(1/dt-rate_c))
        
        else:

            spot_max = max(K,B_level) * self._spot_max_factor_pde  # to make sure strike and barrier are both within lattice

            s_steps_max = int(sqrt(1/vol**2 *(1/dt - rate_c)))

        s_steps = min(s_steps_max, s_steps)

        print(s_steps)

        # if s_steps > s_steps_max:

        #     print('Unstable condition, max spot step is set of %d' % s_steps_max)
        #     s_steps = s_steps_max

        Df = exp(-rate_c * T)  # discount factor from maturity

        if (B_type in ['UO','UI']) and spot >= B_level:   # compute up&out barrier option price in case of knock out,  up&in price then can be derived from it

            mv_out = 0
            delta_out =0
            gamma_out = 0
            theta_out=0

        elif (B_type in ['DO','DI']) and spot <= B_level:

            mv_out = 0
            delta_out =0
            gamma_out = 0
            theta_out=0

        else:  # if spot does not knock barrier for an out option

            if spot > spot_max: # if the spot is outside the grids

                ds = spot_max / s_steps

                fwd= self.forward(spot,rate_c,rate_a)

                fwd_shift = spot * exp((rate_c - rate_a) * (T-dt))

                spot_up = spot + ds
                spot_down = spot -ds

                fwd_up = self.forward(spot_up,rate_c,rate_a)
                fwd_down = self.forward(spot_down,rate_c,rate_a)

                mv_out = max(cp*(fwd-K),0) * Df
                mv_up = max(cp*(fwd_up-K),0) * Df
                mv_down = max(cp*(fwd_down-K),0) * Df
                mv_shift = max(cp*(fwd_shift-K),0) * exp(-rate_c * (T-dt))
                
                delta_out = (mv_up - mv_down) / (2*ds)
                gamma_out = (mv_up-2 * mv_out + mv_down) / (ds**2)
                theta_out = (mv_shift - mv_out)/ dt * self._theta_bump

            else:   # if the spot is within the grids
    
                ds = spot_max / s_steps

                onnode = True

                x = spot/ds  # index of spot on the grids

                x0 = int(x)
                x1 = x0+1

                if x1-x <= 10e-8:

                    spot_index = x1   # spot falls on x1

                elif x-x0 <= 10e-8:

                    spot_index = x0   # spot falls on x0

                else:

                    onnode=False # spot falls in between nodes

                spot_rng = np.linspace(0,spot_max,s_steps+1)   # spot list from S = 0 to S = spot_max

    #############################################################################################################################

                grid= np.zeros((s_steps+1,t_steps+1))

                p_T = np.maximum(cp * (spot_rng - K),0) # vanilla payoff at time T

                if B_type in ['UI','UO']:

                    f = 1 * (spot_rng < B_level)  # for up & out option, f=1 only if spot < Barrier

                elif B_type in ['DI','DO']:

                    f = 1 * (spot_rng > B_level)  # for down & out option, f=1 only if spot > Barrier

                grid[:,-1] = p_T * f # boundry condition at T: for payoff corresponds to each spot prices at maturity

                time_rng = np.arange(t_steps) * dt

                DF_t = np.exp(-rate_c * time_rng) # discount factor array, size t_steps x 1

                F_t = spot_max * np.exp((rate_c-rate_a)*(T-time_rng))  # forward price along the spot upper boundary, size t_steps x 1

                grid[0,:t_steps] = max(cp * (0-K),0) * DF_t # boundry condition for spot = 0 along each time step

                g_temp = np.maximum(cp * (F_t - K),0)

                grid[s_steps,:t_steps] = g_temp * DF_t # boundry condition for spot = spot_max along each time step

                if (B_type in ['UI','UO']):

                    grid[s_steps,:] = 0
                
                elif (B_type in ['DI','DO']):

                    grid[0,:]=0
                
                for t in range(t_steps-1,-1,-1):  # from t=t_step-1 to t=0
                
                    if (B_type in ['UI','UO']):

                        i_uo = max(np.where(spot_rng < B_level)[0])
                        i_start = 1
                        i_end = i_uo+1

                    elif (B_type in ['DI','DO']):

                        i_do = min(np.where(spot_rng > B_level)[0])
                        i_start = i_do
                        i_end = s_steps

                    # for i in range(i_start,i_end):   # index from 1 to s_steps-1

                    S_i = spot_rng[i_start:i_end]

                    a = 0.5 * dt/ds *(vol**2 * S_i**2/ds -(rate_c-rate_a)*S_i)

                    b = 1 - vol**2 * S_i**2 / ds**2 * dt - rate_c * dt

                    if np.min(b)<0:
                        print('Warning!!! Unstable setting!  b= %.2f' % np.min(b) )

                    c = 0.5 * ( vol**2 * S_i **2 / ds**2 + (rate_c-rate_a) * S_i / ds) * dt

                    V_minus_1 =  grid[(i_start-1):(i_end-1),t+1]

                    V_0_1 = grid[i_start:i_end,t+1]

                    V_plus_1 = grid[(i_start+1):(i_end+1),t+1]

                    V_0_0 = a * V_minus_1 + b * V_0_1 + c * V_plus_1

                    grid[i_start:i_end,t] = V_0_0

                # plt.figure()
                # plt.plot(spot_rng,grid[0:s_steps+1,0] ,label='pde test')
                # plt.show()

    #########################################################################################################

                if onnode == True : # spot falls on node

                    if spot_index == 0:

                        y_up = grid[spot_index+2,0]

                        y = grid[spot_index+1,0]

                        y_down = grid[spot_index,0]

                        mv_out = y_down

                        delta_out = (y - y_down) /ds

                    elif spot_index == s_steps:

                        y_up = grid[spot_index,0]

                        y = grid[spot_index-1,0]

                        y_down = grid[spot_index-2,0]

                        mv_out = y_up

                        delta_out = (y_up - y)/ds

                    else:

                        y_up = grid[spot_index+1,0]

                        y = grid[spot_index,0]

                        y_down = grid[spot_index-1,0]

                        mv_out = y

                        delta_out = (y_up - y_down)/(ds*2)

                    mv_out_shift = grid[spot_index,1]

                    gamma_out = (y_up -2*y + y_down)/(ds**2)

                    theta_out = (mv_out_shift-mv_out) / dt * self._theta_bump
            
                else:  # spot falls in between nodes

                    mv0 = grid[x0,0]
                    mv1 = grid[x0+1,0]

                    mv0_shift = grid[x0,1]
                    mv1_shift = grid[x0+1,1]
                    
                    if x0 == 0:

                        if cp == 'c':

                            delta0 = 0

                        else:

                            delta0 = - exp(-rate_a*T)

                        # delta0 = (grid[x0+1,0]- grid[x0,0])/ds
                        delta1 = (grid[x0+2,0]- grid[x0,0])/(2*ds)

                        gamma0 = 0 
                        gamma1 = (grid[x0+2,0] - 2*grid[x0+1,0] + grid[x0,0]) / (ds**2)

                    elif x0 == s_steps -1:

                        delta0 = (grid[x0+1,0]- grid[x0-1,0])/(2*ds)
                        # delta1 = (grid[x0+1,0]- grid[x0,0])/(2*ds)

                        if cp == 'c':

                            delta1 = exp(-rate_a*T)

                        else:

                            delta1 = 0

                        gamma0 = (grid[x0+1,0] - 2*grid[x0,0] + grid[x0-1,0]) / (ds**2)
                        gamma1 = 0

                    else:

                        delta0 = (grid[x0+1,0]- grid[x0-1,0])/(2*ds)
                        delta1 = (grid[x0+2,0]- grid[x0,0])/(2*ds)
                        
                        gamma0 = (grid[x0+1,0] - 2*grid[x0,0] + grid[x0-1,0]) / (ds**2)
                        gamma1 = (grid[x0+2,0] - 2*grid[x0+1,0] + grid[x0,0]) / (ds**2)

                    gamma_out =gamma0 + (gamma1 - gamma0) * (x-x0)

                    delta_out = delta0+(x-x0)*(delta1-delta0)   # using naive interpolation
                    #delta = delta0 + gamma * (spot - spot_rng[x0])  # delta from Taylor expansion using gamma value
                    
                    mv_out = mv0 + (x-x0)* (mv1-mv0) 
                    # pl = pl0 + delta * (spot - spot_rng[x0]) + 0.5 * gamma * (spot - spot_rng[x0])**2

                    mv_out_shift = mv0_shift + (x-x0)* (mv1_shift-mv0_shift)

                    theta_out = (mv_out_shift - mv_out)/dt * self._theta_bump

        if B_type in ['UI','DI']: 

            BSM = self.bsm

            delta = BSM(spot,vol,rate_c,rate_a,'delta') - delta_out * Q
            gamma = BSM(spot,vol,rate_c,rate_a,'gamma') - gamma_out * Q
            theta = BSM(spot,vol,rate_c,rate_a,'theta') - theta_out * Q
            mv = BSM(spot,vol,rate_c,rate_a,'mv') - mv_out * Q

        else:

            delta = delta_out * Q
            gamma = gamma_out * Q
            theta = theta_out * Q
            mv =  mv_out * Q

        if greeks.lower() == 'delta':

            return delta
            
        elif greeks.lower() == 'gamma':

            return gamma

        elif greeks.lower() == 'theta':

            return theta

        else:

            return mv


    def mc(self,spot,vol,rate_c,rate_a):

        if self._displayprogress == True: 
            
            print('Monte Carlo simulation \n')

            print('spot= %.3f' % spot)

        Q = self._quantity

        spot=max(self._spot_minimum,spot)

        n_path = self._npaths_mc
        n_time = self._nsteps_mc
        T = self._maturity
        dt = T / n_time
        K = self._strike
        B_level = self._barrier
        B_type = self._barrier_type

        if self._cp.lower() in('call','c'): 

            cp = 1

        else:

            cp = -1

        D = exp(-rate_c * T)

        sigma = vol * sqrt(dt)

        mu = (rate_c - rate_a) * dt 

        np.random.seed(self._rnd_seed)

        # mv = np.zeros(n)

        #i=0

        #for s in spot:

        rr= np.random.normal(mu,sigma,(n_path,n_time))
        #rr= np.random.normal(0,sqrt(dt),(n_paths,t_steps))*vol + mu

        S = np.zeros((n_path,n_time+1)) 

        S[:,0] = spot 

        for t in range(n_time):

            #S[:,t+1] = S[:,t] * np.exp(rr[:,t])
            S[:,t+1] =S[:,t] * (1+rr[:,t])

        S_max = np.max(S,axis=1)  # max spot of each path
        S_min = np.min(S,axis=1)  # min spot of each path

        if B_type == 'UO':

            f = 1 * (S_max<B_level)

        elif B_type == 'UI':

            f = 1 * (S_max>=B_level)
        
        elif B_type == 'DO':

            f = 1 * (S_min>B_level)

        elif B_type == 'DI':

            f = 1 * (S_min<=B_level)

        S_T = S[:,-1]

        p = (S_T - K)*cp
        
        p = np.maximum(p,0) * D * f

        mv = p.mean()*Q

            #mv[i] = p.mean()*Q

            #i=i+1

        return mv  


    def mc_vector(self,spot_array,vol,rate_c,rate_a):

        if self._displayprogress == True: print('Monte Carlo simulation(vector version) \n')

        Q = self._quantity

        n_spot = spot_array.size

        spot_array[spot_array<=0] = self._spot_minimum

        n_path = self._npaths_mc
        n_time = self._nsteps_mc

        T = self._maturity
        dt = T / n_time
        K = self._strike
        B_level = self._barrier
        B_type = self._barrier_type

        if self._cp.lower() in('call','c'): 

            cp = 1

        else:

            cp = -1

        D = exp(-rate_c * T)

        sigma = vol * sqrt(dt)

        mu = (rate_c - rate_a) * dt 

        np.random.seed(self._rnd_seed)

        # d_rtr= np.random.normal(mu,sigma,(n_path,n_spot,n_time+1))

        S0 = np.array([spot_array,]*n_path)

        S_max = S0
        S_min = S0

        for t in range(n_time):

            if self._displayprogress == True and t%10 ==0: print(t)

            d_rtr= np.random.normal(mu,sigma,(n_path,n_spot))

            S1 =S0 * (1+d_rtr)

            S_max = np.maximum(S_max,S1)
            S_min = np.minimum(S_min,S1)

            S0 = S1

        if B_type == 'UO':

            f = 1 * (S_max<B_level)

        elif B_type == 'UI':

            f = 1 * (S_max>=B_level)
        
        elif B_type == 'DO':

            f = 1 * (S_min>B_level)

        elif B_type == 'DI':

            f = 1 * (S_min<=B_level)

        S_T = S1

        p_T = np.maximum(cp * (S_T - K),0) * f

        df_payoff = p_T * D

        mv_array = df_payoff.mean(axis=0) * Q

        return mv_array


    def delta(self,spot,vol,rate_c,rate_a,model_alt=None,mvcache=None):

        if self._displayprogress == True: print('Delta \n')

        spot[spot<=0]=self._spot_minimum

        if model_alt is None:

            model = self._default_model
        else:
            model = model_alt

        if model in (self.pde, self.pde2): 

            delta_value = self.pde(spot,vol,rate_c,rate_a,'delta')

        elif model == self.bsm:

            delta_value = self.bsm(spot,vol,rate_c,rate_a,'delta')

        else:

            if mvcache is None:      

                mv = model(spot,vol,rate_c,rate_a)

            else:

                mv = mvcache

            bumpvalue = self._delta_bump

            if self._delta_bump_is_percent == True:

                spot_up = spot * (1+bumpvalue/100)
            
            else:                                 

                spot_up = spot + bumpvalue
            
            mv_up = model(spot_up,vol,rate_c,rate_a)

            delta_value = (mv_up - mv)/(spot_up-spot)
        
        return delta_value
    

    def gamma(self,spot,vol,rate_c,rate_a,model_alt=None, mvcache=None):

        if self._displayprogress == True: print('Gamma \n')

        spot[spot<=0]=self._spot_minimum

        if model_alt is None:

            model = self._default_model
        else:
            model = model_alt
        
        if model in (self.pde, self.pde2):

            gamma_value = self.pde(spot,vol,rate_c,rate_a,'gamma')

        elif model == self.bsm:

            gamma_value = self.bsm(spot,vol,rate_c,rate_a,'gamma')

        else:
            if mvcache is None:    
                mv = model(spot,vol,rate_c,rate_a)
            else:
                mv = mvcache

            bumpvalue = self._gamma_bump

            if self._gamma_bump_is_percent == True:

                spot_up = spot * (1+bumpvalue/100)
                spot_down = spot * (1-bumpvalue/100)
            
            else:                                 # then the bumpvalue is absolute

                spot_up = spot + bumpvalue
                spot_down = max(spot - bumpvalue,0)
            
            mv_up = model(spot_up,vol,rate_c,rate_a)
            mv_down = model(spot_down,vol,rate_c,rate_a)

            gamma_value = (mv_up + mv_down -2 * mv)/((spot_up-spot)**2)

        return gamma_value

    def vega(self,spot,vol,rate_c,rate_a,model_alt=None, mvcache=None):

        if self._displayprogress == True: print('Vega \n')

        if model_alt is None:

            model = self._default_model
        else:

            model = model_alt

        spot[spot<=0]=self._spot_minimum

        if model == self.bsm:

            vega_value = self.bsm(spot,vol,rate_c,rate_a,'vega')

        else: 

            if mvcache is None:

                mv = model(spot,vol,rate_c,rate_a)

            else:

                mv = mvcache

            bumpvalue=self._vega_bump

            if self._vega_bump_is_percent == True:

                vol_up = vol * (1+bumpvalue/100)
                
            else:                                 # then the bumpvalue is absolute

                vol_up = vol + bumpvalue
                
            mv_up = model(spot,vol_up,rate_c,rate_a)

            vega_value = (mv_up- mv)/(vol_up-vol) / 100
        
        return vega_value


    def rho(self,spot,vol,rate_c,rate_a,model_alt=None,mvcache=None):

        if self._displayprogress == True: print('Rho \n')

        if model_alt is None:

            model = self._default_model
        else:
            model = model_alt

        spot[spot<=0]=self._spot_minimum

        if model == self.bsm:

            rho_value = self.bsm(spot,vol,rate_c,rate_a,'rho')

        else:

            if mvcache is None:

                mv = model(spot,vol,rate_c,rate_a)

            else:

                mv = mvcache

            bumpvalue=self._rho_bump

            if self._rho_bump_is_percent == True:

                rate_c_up = rate_c * (1+bumpvalue/100)
                
            else:                                 # then the bumpvalue is absolute

                rate_c_up = rate_c + bumpvalue
                
            mv_up = model(spot,vol,rate_c_up,rate_a)

            rho_value = (mv_up- mv)/(rate_c_up-rate_c) / 100
        
        return rho_value

    def theta(self,spot,vol,rate_c,rate_a,model_alt=None,mvcache=None):

        if self._displayprogress == True: print('Theta \n')

        if model_alt is None:

            model = self._default_model
        else:
            model = model_alt

        spot[spot<=0]=self._spot_minimum

        if model == self.bsm:

            theta_value = self.bsm(spot,vol,rate_c,rate_a,'theta')

        elif model in (self.pde, self.pde2):

            theta_value = self.pde(spot,vol,rate_c,rate_a,'theta')
        
        else:

            if mvcache is None:

                mv = model(spot,vol,rate_c,rate_a)

            else:

                mv = mvcache

            bumpvalue = self._theta_bump

            self._maturity= self._maturity - bumpvalue

            mv_shift = model(spot,vol,rate_c,rate_a)

            theta_value = (mv_shift- mv) / bumpvalue * 1/365

            self._maturity= self._maturity + bumpvalue
        
        return theta_value


    def volga(self,spot,vol,rate_c,rate_a,model_alt=None,vegacache=None):

        if self._displayprogress == True: print('Volga \n')

        if model_alt is None:

            model = self._default_model
        else:
            model = model_alt

        spot[spot<=0]=self._spot_minimum

        bumpvalue=self._vega_bump

        if vegacache is None:

            vega = self.vega(spot,vol,rate_c,rate_a,model)

        else:

            vega = vegacache

        if self._vega_bump_is_percent == True:

            vol_up = vol * (1+bumpvalue/100)

            # vol_down = vol - (1+bumpvalue/100)
            
        else:                                 # then the bumpvalue is absolute

            vol_up = vol + bumpvalue

            # vol_down = max(vol - bumpvalue,0)

        vega_up = self.vega(spot,vol_up,rate_c,rate_a,model)
        # vega_down = self.vega(spot,vol_down,rate_c,rate_a,model)

        volga_value = (vega_up - vega)/(vol_up-vol)

        return volga_value


    def vanna(self,spot,vol,rate_c,rate_a,model_alt=None,vegacache=None,deltacache=None):

        if self._displayprogress == True: print('Vanna \n')

        if model_alt is None:

            model = self._default_model
        else:
            model = model_alt

        spot[spot<=0]=self._spot_minimum

        if self._vanna_dvega == True:

            if vegacache is None:

                vega = self.vega(spot,vol,rate_c,rate_a,model)
            
            else:

                vega = vegacache

            bumpvalue=self._delta_bump

            if self._delta_bump_is_percent == True:

                spot_up = spot * (1+bumpvalue/100)
            
            else:                                 # then the bumpvalue is absolute

                spot_up = spot + bumpvalue

            vega_up = self.vega(spot_up,vol,rate_c,rate_a,model)

            vanna_value = (vega_up - vega)/(spot_up-vol)

        else:

            if deltacache is None:

                delta = self.delta(spot,vol,rate_c,rate_a,model)

            else:

                delta = deltacache
            
            bumpvalue=self._vega_bump

            if self._vega_bump_is_percent == True:

                vol_up = vol * (1+bumpvalue/100)
            
            else:                                 # then the bumpvalue is absolute

                vol_up = vol + bumpvalue

            delta_up = self.delta(spot,vol_up,rate_c,rate_a,model)

            vanna_value = (delta_up-delta) / (vol_up-vol)

        return vanna_value


    def spot_ladder(self, spot_array, vol , rate_c , rate_a , model1=None, model2=None):

        s = spot_array

        if model1 is None:
            model1 = self._default_model

        mv = model1(s,vol,rate_c,rate_a)
        
        print('Parent process %s.' % os.getpid())

        delta=self.delta(s,vol,rate_c,rate_a,model1,mvcache=mv)
        gamma=self.gamma(s,vol,rate_c,rate_a,model1,mvcache=mv)
        vega=self.vega(s,vol,rate_c,rate_a,model1,mvcache=mv)
        theta=self.theta(s,vol,rate_c,rate_a,model1,mvcache=mv)
        rho=self.rho(s,vol,rate_c,rate_a,model1,mvcache=mv)
        vanna=self.vanna(s,vol,rate_c,rate_a,model1,vegacache=vega,deltacache=delta)
        volga=self.volga(s,vol,rate_c,rate_a,model1,vegacache=vega)
        
        if model2 is not None:
            
            mv2 =model2(s,vol,rate_c,rate_a)
            delta2 = self.delta(s,vol,rate_c,rate_a,model2,mvcache=mv2)
            gamma2 = self.gamma(s,vol,rate_c,rate_a,model2,mvcache=mv2)
            vega2 = self.vega(s,vol,rate_c,rate_a,model2,mvcache=mv2)
            theta2 = self.theta(s,vol,rate_c,rate_a,model2,mvcache=mv2)
            volga2 = self.volga(s,vol,rate_c,rate_a,model2,vegacache=vega2)
            vanna2 = self.vanna(s,vol,rate_c,rate_a,model2,vegacache=vega2,deltacache=delta2)
            rho2 = self.rho(s,vol,rate_c,rate_a,model2,mvcache=mv2)

        fig, ax = plt.subplots(ncols=4, nrows=2, figsize=(14,9))

        ax[0,0].set_title("MV")
        ax[0,1].set_title("Delta")
        ax[0,2].set_title("Gamma")
        ax[0,3].set_title("Theta")
        ax[1,0].set_title("Rho")
        ax[1,1].set_title("Vega")
        ax[1,2].set_title("Volga")
        ax[1,3].set_title("Vanna")       

        if model2 is None:

            ax[0,0].plot(s,mv,label='MV:model1')  
            ax[0,1].plot(s,delta,label='Delta:model1')
            ax[0,2].plot(s,gamma,label='Gamma:model1')
            ax[0,3].plot(s,theta,label='Theta:model1')
            ax[1,0].plot(s,rho,label='Rho:model1')
            ax[1,1].plot(s,vega,label='Vega:model1')
            ax[1,2].plot(s,volga,label='Volga:model1')
            ax[1,3].plot(s,vanna,label='Vanna:model1')
        
        else:

            ax[0,0].plot(s,mv,label='MV:model1')  
            ax[0,1].plot(s,delta,label='Delta:model1')
            ax[0,2].plot(s,gamma,label='Gamma:model1')
            ax[0,3].plot(s,theta,label='Theta:model1')
            ax[1,0].plot(s,rho,label='Rho:model1')
            ax[1,1].plot(s,vega,label='Vega:model1')
            ax[1,2].plot(s,volga,label='Volga:model1')
            ax[1,3].plot(s,vanna,label='Vanna:model1')
        
            ax[0,0].plot(s,mv2,label='MV:model2')  
            ax[0,1].plot(s,delta2,label='Delta:model2')
            ax[0,2].plot(s,gamma2,label='Gamma:model2')
            ax[0,3].plot(s,theta2,label='Theta:model2')
            ax[1,0].plot(s,rho2,label='Rho:model2')
            ax[1,1].plot(s,vega2,label='Vega:model2')
            ax[1,2].plot(s,volga2,label='Volga:model2')
            ax[1,3].plot(s,vanna2,label='Vanna:model2')

        ax[0,0].legend()
        ax[0,1].legend()
        ax[0,2].legend()
        ax[0,3].legend()
        ax[1,0].legend()
        ax[1,1].legend()
        ax[1,2].legend()
        ax[1,3].legend()
        
        #plt.legend(loc=0)

        fig.suptitle('Show me the speed')
        plt.show()

        return None

    def spot_ladder_light(self, spot_array, vol , rate_c , rate_a , model1=None, model2=None):

        s = spot_array

        if model1 is None:
            model1 = self._default_model

        mv = model1(s,vol,rate_c,rate_a)
   
        if model2 is not None:
            
            mv2 =model2(s,vol,rate_c,rate_a)


        plt.figure()
 

        if model2 is None:

            plt.plot(s,mv,label='MV:model1')
  
        
        else:

            plt.plot(s,mv,label='MV:model1')
            plt.plot(s,mv2,label='MV:model2')

        
        plt.legend(loc=0)

        plt.show()

        return None


def main_barrier():

    vol=0.2
    T=1
    K =50
    rate=5/100
    div=3/100
    quantity = 1
    cp= 'call'
    B = 80
    Btype ='UO'

    op = Barrier(B,Btype,K,cp,T,quantity)

    PDE= op.pde
    PDE2= op.pde2
    BSM = op.bsm
    #MC =np.vectorize(op.mc)
    MC = op.mc_vector


    op._npaths_mc = 50000
    op._nsteps_mc = 10000
    op._rnd_seed = 10000

    op._tsteps_pde = 500
    op._ssteps_pde = 500

    op._delta_bump=1
    op._gamma_bump=1
    op._vega_bump=0.01
    op._theta_bump=1/365
    op._rho_bump = 0.01
    op._displayprogress = False

    spot_array= np.linspace(0,100,100)

    op.spot_ladder_light(spot_array,vol,rate,div,PDE,MC)

if __name__ =='__main__':

    main_barrier()
                        


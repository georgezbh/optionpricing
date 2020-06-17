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

import vanilla

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
        self._displayprogress = True

        self._npaths_mc = 1000
        self._nsteps_mc = 300
        self._rnd_seed = 12345

        self._default_model = self.pde


    def forward(self,spot,rate_c,rate_a):
    
        fwd=math.exp((rate_c-rate_a)*self._maturity)*spot

        return fwd

    def bsm(self,spot,vol,rate_c,rate_a,greeks='pl'):

        if spot<=0:

            spot = self._spot_minimum
      
        fwd = self.forward(spot,rate_c,rate_a)

        if self._cp.lower() == 'call' or self._cp.lower()=='c':

            cp = 'c'

        else:

            cp = 'p'

        Q = self._quantity

        T = self._maturity
        K = self._strike
        D = math.exp(-rate_c*T)

        N=norm.cdf     # standard normal distribution cumulative function
        N_pdf = norm.pdf

        d1 = (math.log(fwd/K)+0.5*vol**2*T)/(vol*math.sqrt(T)) 
        d2 = d1-vol*math.sqrt(T)

        if greeks.lower() =='delta':

            call_delta = N(d1) * math.exp(-rate_a*T)
            put_delta = (N(d1)-1) * math.exp(-rate_a*T)

            if cp =='c':

                 return call_delta * Q
            else:

                return put_delta * Q

        elif greeks.lower()=='gamma':

            gamma = math.exp(-rate_a*T)*N_pdf(d1) / (spot*vol*math.sqrt(T))

            return gamma * Q

        elif greeks.lower()=='vega':

            vega = spot*N_pdf(d1)*math.sqrt(T) * math.exp(-rate_a*T) / 100

            return vega * Q
        
        elif greeks.lower()=='theta':

            call_theta = (-spot*N_pdf(d1)*vol*math.exp(-rate_a*T) /(2*math.sqrt(T)) - rate_c*K*math.exp(-rate_c*T)*N(d2)+rate_a*spot*math.exp(-rate_a*T)*N(d1))*self._theta_bump
            put_theta = (-spot*N_pdf(d1)*vol*math.exp(-rate_a*T) /(2*math.sqrt(T)) + rate_c*K*math.exp(-rate_c*T)*N(-d2)-rate_a*spot*math.exp(-rate_a*T)*N(-d1))*self._theta_bump

            if cp =='c':

                 return call_theta * Q 
            else:

                return put_theta * Q

        elif greeks.lower()=='rho':

            call_rho = K*T*math.exp(-rate_c*T) * N(d2) / 100
            put_rho = -K*T*math.exp(-rate_c*T) * N(-d2) / 100

            if cp =='c':

                 return call_rho * Q
            else:

                return put_rho * Q

        else:

            call_price = D*(N(d1)*fwd - N(d2)*K) 
            put_price = D*(N(-d2)*K-N(-d1)*fwd)

            if cp =='c':

                return call_price * Q
            else:

                return put_price * Q
                

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

        B_level = self._barrier

        B_type = self._barrier_type

        spot_max = max(K,B_level) * self._spot_max_factor_pde

        if (B_type in ['UO','UI']) and spot >= B_level:

            pl_out = 0
            delta_out =0
            gamma_out = 0
            theta_out=0

        elif (B_type in ['DO','DI']) and spot <= B_level:

            pl_out = 0
            delta_out =0
            gamma_out = 0
            theta_out=0

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

                    pl_out = max(fwd-K,0) * math.exp(-rate_c*T)
                    pl_up = max(fwd_up-K,0) * math.exp(-rate_c*T)
                    pl_down = max(fwd_down-K,0) * math.exp(-rate_c*T)
                    pl_shift = max(fwd_shift-K,0) * math.exp(-rate_c*(T-dt))

                else:

                    pl_out = max(K-fwd,0) * math.exp(-rate_c*T)
                    pl_up = max(K-fwd_up,0) * math.exp(-rate_c*T)
                    pl_down = max(K-fwd_down,0) * math.exp(-rate_c*T)
                    pl_shift = max(K-fwd_shift,0) * math.exp(-rate_c*(T-dt))
                
                delta_out = (pl_up - pl_down) / (2*ds)
                gamma_out = (pl_up-2*pl + pl_down) / (ds**2)
                theta_out = (pl_shift - pl)/ dt * self._theta_bump

            else:   # if the spot is within the grids
    
                ds = spot_max / s_steps

                onnode = True

                x = spot/ds  # coordinate of spot on the grids

                x0 = int(x)
                x1 = x0+1

                if x1-x <= 10e-8:
                    spot_index = x1   # spot falls on x1
                elif x-x0 <= 10e-8:
                    spot_index = x0   # spot falls on x0
                else:
                    onnode=False # spot falls in between nodes

                spot_rng = np.linspace(0,spot_max,s_steps+1)
    #############################################################################################################################

                grid= np.zeros((s_steps+1,t_steps+1))

                for i in range(s_steps+1):  # boundry condition at T: for payoff corresponds to each spot prices at maturity

                    if (B_type in ['UI','UO']) and spot_rng[i] >=B_level:

                        grid[i,t_steps] = 0

                    elif (B_type in ['DI','DO']) and spot_rng[i] <= B_level:

                        grid[i,t_steps] = 0
                    
                    else:

                        if cp=='c':

                            grid[i,t_steps] = max (spot_rng[i]-K,0) 

                        else:
                            grid[i,t_steps] = max (K-spot_rng[i],0)        

                for j in range(t_steps): # boundry condition at spot =0 and spot = s_max

                    DF_t = exp(-rate_c*(T-j*dt))

                    F_t = spot_rng[s_steps] * exp((rate_c-rate_a) * (T-j*dt))

                    if cp=='c':

                        grid[0,j] = 0

                        grid[s_steps,j] = max(F_t - K,0) * DF_t
                    
                    else:

                        grid[0,j] = max(K,0)* DF_t

                        grid[s_steps,j] = 0

                    if (B_type in ['UI','UO']) and spot_rng[s_steps] >=B_level:

                        grid[s_steps,j] = 0

                    elif (B_type in ['DI','DO']) and spot_rng[s_steps]<= B_level:

                        grid[s_steps,j] = 0
                        grid[0,j] = 0
                
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

                        pl_out = y_down

                        delta_out = (y - y_down) /ds

                    elif spot_index == s_steps:

                        y_up = grid[spot_index,0]

                        y = grid[spot_index-1,0]

                        y_down = grid[spot_index-2,0]

                        pl_out = y_up

                        delta_out = (y_up - y)/ds

                    else:

                        y_up = grid[spot_index+1,0]

                        y = grid[spot_index,0]

                        y_down = grid[spot_index-1,0]

                        pl_out = y

                        delta_out = (y_up - y_down)/(ds*2)

                    pl_out_shift = grid[spot_index,1]

                    gamma_out = (y_up -2*y + y_down)/(ds**2)

                    theta_out = (pl_out_shift-pl_out) / dt * self._theta_bump
            
                else:  # spot falls in between nodes

                    pl0 = grid[x0,0]
                    pl1 = grid[x0+1,0]

                    pl0_shift = grid[x0,1]
                    pl1_shift = grid[x0+1,1]
                    
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

                    gamma_out =gamma0 + (gamma1 - gamma0) * (x-x0)

                    delta_out = delta0+(x-x0)*(delta1-delta0)   # using naive interpolation
                    #delta = delta0 + gamma * (spot - spot_rng[x0])  # delta from Taylor expansion using gamma value
                    
                    pl_out = pl0 + (x-x0)* (pl1-pl0) 
                    # pl = pl0 + delta * (spot - spot_rng[x0]) + 0.5 * gamma * (spot - spot_rng[x0])**2

                    pl_out_shift = pl0_shift + (x-x0)* (pl1_shift-pl0_shift)

                    theta_out = (pl_out_shift - pl_out)/dt * self._theta_bump

        if B_type in ['UI','DI']: 

            BSM = self.bsm

            delta = BSM(spot,vol,rate_c,rate_a,'delta') - delta_out * Q
            gamma = BSM(spot,vol,rate_c,rate_a,'gamma') - gamma_out * Q
            theta = BSM(spot,vol,rate_c,rate_a,'theta') - theta_out * Q
            pl = BSM(spot,vol,rate_c,rate_a,'pl') - pl_out * Q

        else:

            delta = delta_out * Q
            gamma = gamma_out * Q
            theta = theta_out * Q
            pl =  pl_out * Q

        if greeks.lower() == 'delta':

            return delta
            
        elif greeks.lower() == 'gamma':

            return gamma

        elif greeks.lower() == 'theta':

            return theta

        else:

            return pl

    def mc(self,spot,vol,rate_c,rate_a):

        Q = self._quantity

        spot = max(spot,self._spot_minimum)

        n_paths = self._npaths_mc
        n_steps = self._nsteps_mc
        T = self._maturity
        dt = T / n_steps
        K = self._strike
        B_level = self._barrier
        B_type = self._barrier_type

        knockout = False
        knockin = False

        S = spot

        c = []

        p = []

        D = math.exp(-rate_c * T)

        sigma = math.sqrt(dt)

        mu = (rate_c - rate_a) * dt 

        np.random.seed(self._rnd_seed)

        rr= np.random.normal(mu,sigma,(n_paths,n_steps))

        S = np.zeros((n_paths,))
 

        for i in range(n_paths):

            for j in range(n_steps):

                if B_type == 'UO' and S >= B_level:

                    knockout = True

                    break

                elif B_type == 'DO' and S <= B_level:

                    knockout = True

                    break
                
                elif B_type == 'UI' and S >= B_level:

                    knockin = True

                elif B_type == 'DI' and S <= B_level:

                    knockin = True
                
                else: 

                    pass

                dS = S*((rate_c-rate_a) * dt + vol * random.gauss(0,sigma))

                S= S + dS
            
            if B_type == 'UO' and S >= B_level:

                knockout = True

            elif B_type == 'DO' and S <= B_level:

                knockout = True
            
            elif B_type == 'UI' and S >= B_level:

                knockin = True

            elif B_type == 'DI' and S <= B_level:

                knockin = True

            if B_type in ['UO', 'DO']:

                if knockout == True:

                    call_payoff = 0
                    put_payoff = 0
                
                else:

                    call_payoff = D * max(S-K , 0)
                    put_payoff = D* max(K-S , 0)

            elif B_type in ['UI','DI']:

                if knockin == True:

                    call_payoff= D * max(S-K , 0)
                    put_payoff = D* max(K-S , 0)
                
                else:

                    call_payoff = 0
                    put_payoff = 0
            
            else:

                call_payoff = 0
                put_payoff = 0

            c.append(call_payoff)
            p.append(put_payoff)

        call_price = statistics.mean(c) * Q
        put_price = statistics.mean(p) * Q

        if self._cp =='call':

            return call_price
        
        elif self._cp =='put':

            return put_price
        
        else:

            return -999999


    def delta(self,spot,vol,rate_c,rate_a,model_alt=None):

        if spot<=0:

            spot = self._spot_minimum

        if model_alt is None:

            model = self._default_model
        else:
            model = model_alt


        if model == self.pde: 

            delta_value = self.pde(spot,vol,rate_c,rate_a,'delta')

        elif model == self.bsm:

            delta_value = self.bsm(spot,vol,rate_c,rate_a,'delta')

        else:       

            price = model(spot,vol,rate_c,rate_a)

            bumpvalue = self._delta_bump

            if self._delta_bump_is_percent == True:

                spot_up = spot * (1+bumpvalue/100)
            
            else:                                 

                spot_up = spot + bumpvalue
            
            price_up = model(spot_up,vol,rate_c,rate_a)

            delta_value = (price_up - price)/(spot_up-spot)
        
        return delta_value
    

    def gamma(self,spot,vol,rate_c,rate_a,model_alt=None):

        if spot<=0:

            spot = self._spot_minimum

        if model_alt is None:

            model = self._default_model
        else:
            model = model_alt
        
        if model == self.pde:

            gamma_value = self.pde(spot,vol,rate_c,rate_a,'gamma')

        elif model == self.bsm:

            gamma_value = self.bsm(spot,vol,rate_c,rate_a,'gamma')

        else:    
            price = model(spot,vol,rate_c,rate_a)

            bumpvalue = self._gamma_bump

            if self._gamma_bump_is_percent == True:

                spot_up = spot * (1+bumpvalue/100)
                spot_down = spot * (1-bumpvalue/100)
            
            else:                                 # then the bumpvalue is absolute

                spot_up = spot + bumpvalue
                spot_down = max(spot - bumpvalue,0)
            
            price_up = model(spot_up,vol,rate_c,rate_a)
            price_down = model(spot_down,vol,rate_c,rate_a)

            gamma_value = (price_up + price_down -2*price)/((spot_up-spot)**2)

        return gamma_value

    def vega(self,spot,vol,rate_c,rate_a,model_alt=None):

        if model_alt is None:

            model = self._default_model
        else:
            model = model_alt

        if spot<=0:

            spot = self._spot_minimum

        if model == self.bsm:

            vega_value = self.bsm(spot,vol,rate_c,rate_a,'vega')

        else: 

            price = model(spot,vol,rate_c,rate_a)

            bumpvalue=self._vega_bump

            if self._vega_bump_is_percent == True:

                vol_up = vol * (1+bumpvalue/100)
                
            else:                                 # then the bumpvalue is absolute

                vol_up = vol + bumpvalue
                
            price_up = model(spot,vol_up,rate_c,rate_a)

            vega_value = (price_up- price)/(vol_up-vol) / 100
        
        return vega_value

    def rho(self,spot,vol,rate_c,rate_a,model_alt=None):

        if model_alt is None:

            model = self._default_model
        else:
            model = model_alt

        if spot<=0:

            spot = self._spot_minimum

        if model == self.bsm:

            rho_value = self.bsm(spot,vol,rate_c,rate_a,'rho')

        else:

            price = model(spot,vol,rate_c,rate_a)

            bumpvalue=self._rho_bump

            if self._rho_bump_is_percent == True:

                rate_c_up = rate_c * (1+bumpvalue/100)
                
            else:                                 # then the bumpvalue is absolute

                rate_c_up = rate_c + bumpvalue
                
            price_up = model(spot,vol,rate_c_up,rate_a)

            rho_value = (price_up- price)/(rate_c_up-rate_c) / 100
        
        return rho_value
    
    def theta(self,spot,vol,rate_c,rate_a,model_alt=None):

        if model_alt is None:

            model = self._default_model
        else:
            model = model_alt

        if spot<=0:

            spot = self._spot_minimum

        if model == self.bsm:

            theta_value = self.bsm(spot,vol,rate_c,rate_a,'theta')

        elif model == self.pde:

            theta_value = self.pde(spot,vol,rate_c,rate_a,'theta')
        
        else:

            price = model(spot,vol,rate_c,rate_a)

            bumpvalue = self._theta_bump

            self._maturity= self._maturity - bumpvalue

            price_shift = model(spot,vol,rate_c,rate_a)

            theta_value = (price_shift- price)

            self._maturity= self._maturity + bumpvalue
        
        return theta_value

    
    def volga(self,spot,vol,rate_c,rate_a,model_alt=None):

        if model_alt is None:

            model = self._default_model
        else:
            model = model_alt

        if spot<=0:

            spot = self._spot_minimum

        bumpvalue=self._vega_bump

        # vega = self.vega(spot,vol,rate_c,rate_a,model)

        if self._vega_bump_is_percent == True:

            vol_up = vol * (1+bumpvalue/100)

            vol_down = vol - (1+bumpvalue/100)
            
        else:                                 # then the bumpvalue is absolute

            vol_up = vol + bumpvalue

            vol_down = max(vol - bumpvalue,0)

        vega_up = self.vega(spot,vol_up,rate_c,rate_a,model)
        vega_down = self.vega(spot,vol_down,rate_c,rate_a,model)

        volga_value = (vega_up - vega_down)/(vol_up-vol_down)

        return volga_value


    def vanna(self,spot,vol,rate_c,rate_a,model_alt=None):

        if model_alt is None:

            model = self._default_model
        else:
            model = model_alt

        if spot<=0:

            spot = self._spot_minimum

        if self._vanna_dvega == True:

            vega = self.vega(spot,vol,rate_c,rate_a,model)

            bumpvalue=self._delta_bump

            if self._delta_bump_is_percent == True:

                spot_up = spot * (1+bumpvalue/100)
            
            else:                                 # then the bumpvalue is absolute

                spot_up = spot + bumpvalue

            vega_up = self.vega(spot_up,vol,rate_c,rate_a,model)

            vanna_value = (vega_up - vega)/(spot_up-vol)

        else:

            delta = self.delta(spot,vol,rate_c,rate_a,model)
            
            bumpvalue=self._vega_bump

            if self._vega_bump_is_percent == True:

                vol_up = vol * (1+bumpvalue/100)
            
            else:                                 # then the bumpvalue is absolute

                vol_up = vol + bumpvalue

            delta_up = self.delta(spot,vol_up,rate_c,rate_a,model)

            vanna_value = (delta_up-delta) / (vol_up-vol)

        return vanna_value

    
    def spot_ladder(self, spot_list,vol,rate_c,rate_a,model1=None, model2=None):

        pl =[]
        delta=[]
        gamma=[]
        vega=[]
        theta=[]
        rho=[]
        vanna=[]
        volga=[]

        if model1 is None:
            model1 = self._default_model

        if model2 is not None:
            
            pl2 =[]
            delta2=[]
            gamma2=[]
            vega2=[]
            theta2=[]
            rho2=[]
            vanna2=[]
            volga2=[]

        i = 0

        for s in spot_list:

            if self._displayprogress == True:

                n=len(spot_list)
                progress= int(i/n*100)
                print('Spot = %f, in progress %d complete' % (s, progress))

            pl_value= model1(s,vol,rate_c,rate_a)
            delta_value = self.delta(s,vol,rate_c,rate_a,model1)
            gamma_value = self.gamma(s,vol,rate_c,rate_a,model1)
            vega_value = self.vega(s,vol,rate_c,rate_a,model1)
            theta_value = self.theta(s,vol,rate_c,rate_a,model1)
            volga_value = self.volga(s,vol,rate_c,rate_a,model1)
            vanna_value = self.vanna(s,vol,rate_c,rate_a,model1)
            rho_value = self.rho(s,vol,rate_c,rate_a,model1)

            pl.append(pl_value)
            delta.append(delta_value)
            gamma.append(gamma_value)
            vega.append(vega_value)
            theta.append(theta_value)
            volga.append(volga_value)
            vanna.append(vanna_value)
            rho.append(rho_value)

                
            if model2 is not None:

                pl_value2 = model2(s,vol,rate_c,rate_a)
                delta_value2 = self.delta(s,vol,rate_c,rate_a,model2)
                gamma_value2 = self.gamma(s,vol,rate_c,rate_a,model2)
                vega_value2 = self.vega(s,vol,rate_c,rate_a,model2)
                theta_value2 = self.theta(s,vol,rate_c,rate_a,model2)
                volga_value2 = self.volga(s,vol,rate_c,rate_a,model2)
                vanna_value2 = self.vanna(s,vol,rate_c,rate_a,model2)
                rho_value2 = self.rho(s,vol,rate_c,rate_a,model2)

                pl2.append(pl_value2)
                delta2.append(delta_value2)
                gamma2.append(gamma_value2)
                vega2.append(vega_value2)
                theta2.append(theta_value2)
                volga2.append(volga_value2)
                vanna2.append(vanna_value2)
                rho2.append(rho_value2)
            
            i=i+1


        fig, ax = plt.subplots(ncols=4, nrows=2, figsize=(14,9))

        ax[0,0].set_title("P&L")
        ax[0,1].set_title("Delta")
        ax[0,2].set_title("Gamma")
        ax[0,3].set_title("Theta")
        ax[1,0].set_title("Rho")
        ax[1,1].set_title("Vega")
        ax[1,2].set_title("Volga")
        ax[1,3].set_title("Vanna")       

        if model2 is None:

            ax[0,0].plot(spot_list,pl,label='P&L:model1')  
            ax[0,1].plot(spot_list,delta,label='Delta:model1')
            ax[0,2].plot(spot_list,gamma,label='Gamma:model1')
            ax[0,3].plot(spot_list,theta,label='Theta:model1')
            ax[1,0].plot(spot_list,rho,label='Rho:model1')
            ax[1,1].plot(spot_list,vega,label='Vega:model1')
            ax[1,2].plot(spot_list,volga,label='Volga:model1')
            ax[1,3].plot(spot_list,vanna,label='Vanna:model1')
        
        else:

            ax[0,0].plot(spot_list,pl,label='P&L:model1')  
            ax[0,1].plot(spot_list,delta,label='Delta:model1')
            ax[0,2].plot(spot_list,gamma,label='Gamma:model1')
            ax[0,3].plot(spot_list,theta,label='Theta:model1')
            ax[1,0].plot(spot_list,rho,label='Rho:model1')
            ax[1,1].plot(spot_list,vega,label='Vega:model1')
            ax[1,2].plot(spot_list,volga,label='Volga:model1')
            ax[1,3].plot(spot_list,vanna,label='Vanna:model1')
        
            ax[0,0].plot(spot_list,pl2,label='P&L:model2')  
            ax[0,1].plot(spot_list,delta2,label='Delta:model2')
            ax[0,2].plot(spot_list,gamma2,label='Gamma:model2')
            ax[0,3].plot(spot_list,theta2,label='Theta:model2')
            ax[1,0].plot(spot_list,rho2,label='Rho:model2')
            ax[1,1].plot(spot_list,vega2,label='Vega:model2')
            ax[1,2].plot(spot_list,volga2,label='Volga:model2')
            ax[1,3].plot(spot_list,vanna2,label='Vanna:model2')

        ax[0,0].legend()
        ax[0,1].legend()
        ax[0,2].legend()
        ax[0,3].legend()
        ax[1,0].legend()
        ax[1,1].legend()
        ax[1,2].legend()
        ax[1,3].legend()
        
        #plt.legend(loc=0)

        fig.suptitle('Barrier Option Greeks')
        plt.show()

        return None


def main_barrier():

    spot = 100
    vol=0.3
    T=2
    K =100
    rate=0
    div=0
    quantity = 1
    cp='put'
    B = 70
    Btype ='DO'

    op = Barrier(B,Btype,K,cp,T,quantity)

    PDE= op.pde
    BSM = op.bsm
    MC =op.mc

    op._npaths_mc = 5000
    op._nsteps_mc = 200
    op._rnd_seed = 12345

    spot_list= np.linspace(50,150,201)

    op.spot_ladder(spot_list,vol,rate,div,PDE,BSM)

if __name__ =='__main__':

    main_barrier()
                        


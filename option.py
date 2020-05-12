'option class pricing module'
'with Black Scholes, Cox Tree, Monte Carlo and PDE pricing methods'

__author__ = 'George Zhao'


import math

import statistics

import random

import numpy as np

import matplotlib.pyplot as plt

from scipy.stats import norm


class Option(object):

    __slots__ = ('_underlying', '_asset_class')

    def __init__(self, underlying, asset_class):

        self._underlying = underlying
        self._asset_class = asset_class


class Vanilla(Option):
    
    def __init__(self, underlying, asset_class, maturity, strike, style, cp, quantity):

        super().__init__(underlying,asset_class)

        self._maturity = maturity

        self._strike = strike

        self._style = style

        if cp.lower() == 'call' or cp.lower() == 'put':

            self._cp = cp
        
        else:

            raise ValueError('Option type is not correct, please input either call or put!')

        self._quantity=quantity

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
    
        self._npaths_mc = 1000
        self._nsteps_mc = 300
        self._rnd_seed = 12345
        self._nsteps_crr = 101

        self._tsteps_pde = 100
        self._ssteps_pde = 200
        self._spot_max_factor_pde = 3

        self._spot_minimum = 10e-6

        self._displayprogress = False

        if self._style.lower() == 'e':

            self._default_model = self.pricing_bsm
        
        else:

            self._default_model = self.pricing_crr


    def intrinsicvalue(self,spot):

        intrinsicvalue_c = max(spot-self._strike,0)

        intrinsicvalue_p = max(self._strike - spot, 0)

        if self._cp.lower() == 'call':

            return intrinsicvalue_c

        elif self._cp.lower() == 'put':

            return intrinsicvalue_p
        
        else:

            return None

    def forward(self,spot,rate_c,rate_a):
        
        fwd=math.exp((rate_c-rate_a)*self._maturity)*spot

        return fwd

    def pricing_bsm(self,spot,vol,rate_c,rate_a,greeks='pl'):

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

            vega = spot*N_pdf(d1)*math.sqrt(T) * math.exp(-rate_a*T)

            return vega * Q
        
        elif greeks.lower()=='theta':

            call_theta = (-spot*N_pdf(d1)*vol*math.exp(-rate_a*T) /(2*math.sqrt(T)) - rate_c*K*math.exp(-rate_c*T)*N(d2)+rate_a*spot*math.exp(-rate_a*T)*N(d1))*self._theta_bump
            put_theta = (-spot*N_pdf(d1)*vol*math.exp(-rate_a*T) /(2*math.sqrt(T)) + rate_c*K*math.exp(-rate_c*T)*N(-d2)-rate_a*spot*math.exp(-rate_a*T)*N(-d1))*self._theta_bump

            if cp =='c':

                 return call_theta * Q 
            else:

                return put_theta * Q

        elif greeks.lower()=='rho':

            call_rho = K*T*math.exp(-rate_c*T) * N(d2)
            put_rho = -K*T*math.exp(-rate_c*T) * N(-d2)

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

    def delta(self,spot,vol,rate_c,rate_a,model_alt=None):

        if spot<=0:

            spot = self._spot_minimum

        if model_alt is None:

            model = self._default_model
        else:
            model = model_alt

        if model == self.pricing_crr:

            delta_value = self.pricing_crr(spot,vol,rate_c,rate_a,'delta')
        
        elif model == self.pricing_pde: 

            delta_value = self.pricing_pde(spot,vol,rate_c,rate_a,'delta')

        elif model == self.pricing_pde_alt: 

            delta_value = self.pricing_pde_alt(spot,vol,rate_c,rate_a,'delta')

        elif model == self.pricing_bsm:

            delta_value = self.pricing_bsm(spot,vol,rate_c,rate_a,'delta')

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

        if model == self.pricing_crr:

            gamma_value = self.pricing_crr(spot,vol,rate_c,rate_a,'gamma')
        
        elif model == self.pricing_pde:

            gamma_value = self.pricing_pde(spot,vol,rate_c,rate_a,'gamma')

        elif model == self.pricing_pde_alt: 

            gamma_value = self.pricing_pde_alt(spot,vol,rate_c,rate_a,'gamma')

        elif model == self.pricing_bsm:

            gamma_value = self.pricing_bsm(spot,vol,rate_c,rate_a,'gamma')

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

        if model == self.pricing_bsm:

            vega_value = self.pricing_bsm(spot,vol,rate_c,rate_a,'vega')

        else: 

            price = model(spot,vol,rate_c,rate_a)

            bumpvalue=self._vega_bump

            if self._vega_bump_is_percent == True:

                vol_up = vol * (1+bumpvalue/100)
                
            else:                                 # then the bumpvalue is absolute

                vol_up = vol + bumpvalue
                
            price_up = model(spot,vol_up,rate_c,rate_a)

            vega_value = (price_up- price)/(vol_up-vol)
        
        return vega_value

    def rho(self,spot,vol,rate_c,rate_a,model_alt=None):

        if model_alt is None:

            model = self._default_model
        else:
            model = model_alt

        if spot<=0:

            spot = self._spot_minimum

        if model == self.pricing_bsm:

            rho_value = self.pricing_bsm(spot,vol,rate_c,rate_a,'rho')

        else:

            price = model(spot,vol,rate_c,rate_a)

            bumpvalue=self._rho_bump

            if self._rho_bump_is_percent == True:

                rate_c_up = rate_c * (1+bumpvalue/100)
                
            else:                                 # then the bumpvalue is absolute

                rate_c_up = rate_c + bumpvalue
                
            price_up = model(spot,vol,rate_c_up,rate_a)

            rho_value = (price_up- price)/(rate_c_up-rate_c)
        
        return rho_value
    
    def theta(self,spot,vol,rate_c,rate_a,model_alt=None):

        if model_alt is None:

            model = self._default_model
        else:
            model = model_alt

        if spot<=0:

            spot = self._spot_minimum

        if model == self.pricing_bsm:

            theta_value = self.pricing_bsm(spot,vol,rate_c,rate_a,'theta')
        
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

        vega = self.vega(spot,vol,rate_c,rate_a,model)

        bumpvalue=self._vega_bump

        if self._vega_bump_is_percent == True:

            vol_up = vol * (1+bumpvalue/100)
            
        else:                                 # then the bumpvalue is absolute

            vol_up = vol + bumpvalue

        vega_up = self.vega(spot,vol_up,rate_c,rate_a,model)

        volga_value = (vega_up - vega)/(vol_up-vol)

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


    
    @staticmethod
    def gen_one_path(T,spot,vol,rate_c,rate_a,n_steps):

        if spot<=0:

            spot = 10e-6

        dt = T / n_steps
        S=[spot]
        sigma = math.sqrt(dt)

        for _ in range(n_steps):

            dS = S[-1]*((rate_c-rate_a) * dt + vol * random.gauss(0,sigma))
            S.append(S[-1]+dS)

        return S      # the size of S should be n_steps + 1 

    @staticmethod
    def sim_spot(T,spot,vol,rate_c,rate_a,n_paths):

        if spot<=0:

            spot = 10e-6

        sigma = math.sqrt(T)

        diffuse = np.random.normal(0,vol*sigma,n_paths)

        s=[]

        for i in range(n_paths):

            s_i = spot * math.exp((rate_c-rate_a-0.5*vol**2)*T + diffuse[i])

            s.append(s_i)

        return s      # the size of s should be n_paths


    def pricing_mc(self,spot,vol,rate_c,rate_a):

        Q = self._quantity

        if spot<=0:

            spot = self._spot_minimum

        if self._rnd_seed > 0:
            random.seed(self._rnd_seed)

        n_paths = self._npaths_mc
        n_steps = self._nsteps_mc

        T = self._maturity

        K = self._strike

        D = math.exp(-rate_c * T)

        c =[]

        p =[]

        for i in range(n_paths):

            s = self.gen_one_path(T,spot,vol,rate_c,rate_a,n_steps)

            c.append(D * max(s[-1]-K , 0))

            p.append(D* max(K-s[-1] , 0))

        call_price = statistics.mean(c) * Q
        put_price = statistics.mean(p) * Q

        if self._cp =='call':

            return call_price
        
        elif self._cp =='put':

            return put_price

        else:

            return None


    def pricing_mc_fast(self,spot,vol,rate_c,rate_a):

        Q = self._quantity

        if spot<=0:

            spot = self._spot_minimum

        if self._rnd_seed > 0:
            np.random.seed(self._rnd_seed)

        n_paths = self._npaths_mc

        T = self._maturity

        K = self._strike

        D = math.exp(-rate_c * T)

        c =[]

        p =[]

        s = self.sim_spot(T,spot,vol,rate_c,rate_a,n_paths)

        for i in range(n_paths):

            c.append(D * max(s[i]-K , 0))

            p.append(D * max(K-s[i] , 0))

        call_price = statistics.mean(c) * Q
        put_price = statistics.mean(p) * Q

        if self._cp =='call':

            return call_price
        
        elif self._cp =='put':

            return put_price

        else:

            return None


    def pricing_crr(self,spot,vol,rate_c,rate_a,greeks='pl'):  # pricing option using binomial tree, time steps is n_steps

        Q = self._quantity

        if spot<=0:

            spot = self._spot_minimum
        
        n_steps = self._nsteps_crr

        n_steps = max(n_steps,3)

        if rate_c != rate_a :

            n_steps = max(n_steps, self._maturity*(rate_c-rate_a)**2/vol**2)

            # print(n_steps)

        dt = self._maturity / (n_steps-1)

        K = self._strike

        spot_tree = np.zeros((2*n_steps-1, n_steps), dtype=float)

        call_tree = np.zeros((2*n_steps-1, n_steps), dtype=float)
        put_tree = np.zeros((2*n_steps-1, n_steps), dtype=float)

        u = math.exp(vol * math.sqrt(dt))

        d = 1 / u

        p = (math.exp((rate_c-rate_a) * dt) -d) / (u-d)

        D = math.exp(-rate_c*dt)

        for i in range(n_steps-1): # from 0 to n_steps-2

            if i == 0:

                spot_tree[n_steps-1,0] = spot

                spot_tree[n_steps-1-1,1] = spot * u

                spot_tree[n_steps-1+1,1] = spot * d

            else:

                for j in range(n_steps-1-i,n_steps-1+i+1,2): # from n_steps-1-i to n_steps-1+i(include), step 2

                    spot_tree[j-1,i+1] = spot_tree[j,i] * u

                    spot_tree[j+1,i+1] = spot_tree[j,i] * d

        for i in range(n_steps-1,-1,-1): # from n_steps-1 to 0

            for j in range(n_steps-1-i,n_steps-1+i+1,2):

                if i == n_steps-1:

                    call_tree[j,i] = max(spot_tree[j,i]-K,0)
                    put_tree[j,i] = max(K-spot_tree[j,i],0)

                else:

                    if self._style == 'a' or self._style =='A':  # if the option is american style

                        call_iv = max(spot_tree[j,i] - K,0) 
                        put_iv =  max(K - spot_tree[j,i],0)

                        call_tree[j,i] = max(D * (p*call_tree[j-1,i+1]+(1-p)*call_tree[j+1,i+1]), call_iv)
                        put_tree[j,i] = max(D * (p*put_tree[j-1,i+1]+(1-p)*put_tree[j+1,i+1]), put_iv)

                        # if put_iv == put_tree[j,i]:
                        #     print('exercise on %i step %i node' % (i,j))
                    
                    else:         # if the option is european style

                        call_tree[j,i] = D * (p*call_tree[j-1,i+1]+(1-p)*call_tree[j+1,i+1])
                        put_tree[j,i] = D * (p*put_tree[j-1,i+1]+(1-p)*put_tree[j+1,i+1])
                    
        call_price = call_tree[n_steps-1,0] * Q
        put_price= put_tree[n_steps-1,0] * Q

        call_delta = (call_tree[n_steps-2,1]-call_tree[n_steps,1])/(spot_tree[n_steps-2,1]-spot_tree[n_steps,1]) * Q
        put_delta = (put_tree[n_steps-2,1]-put_tree[n_steps,1])/(spot_tree[n_steps-2,1]-spot_tree[n_steps,1]) * Q

        call_delta_up =(call_tree[n_steps-3,2]-call_tree[n_steps-1,2])/(spot_tree[n_steps-3,2]-spot_tree[n_steps-1,2])
        call_delta_down =(call_tree[n_steps-1,2]-call_tree[n_steps+1,2])/(spot_tree[n_steps-1,2]-spot_tree[n_steps+1,2])
        put_delta_up =(put_tree[n_steps-3,2]-put_tree[n_steps-1,2])/(spot_tree[n_steps-3,2]-spot_tree[n_steps-1,2])
        put_delta_down =(put_tree[n_steps-1,2]-put_tree[n_steps+1,2])/(spot_tree[n_steps-1,2]-spot_tree[n_steps+1,2])

        call_gamma =  (call_delta_up - call_delta_down)/(spot_tree[n_steps-2,1]-spot_tree[n_steps,1]) * Q
        put_gamma =  (put_delta_up - put_delta_down)/(spot_tree[n_steps-2,1]-spot_tree[n_steps,1]) * Q

        if greeks.lower()=='pl':

            if self._cp =='call':

                return call_price
            
            else:

                return put_price

        elif greeks.lower()=='delta':

            if self._cp =='call':

                return call_delta
            
            else:

                return put_delta

        elif greeks.lower()=='gamma':

            if self._cp == 'call':

                return call_gamma
            
            else:

                return put_gamma

        else:

            return None


    def pricing_pde(self,spot,vol,rate_c,rate_a,greeks='pl'):

        if spot<=0:

            spot = self._spot_minimum

        Q = self._quantity

        if self._cp.lower()=='call' or self._cp.lower()=='c':

            cp='c'

        else:

            cp='p'
        
        if self._style.lower() == 'a' or self._style.lower()=='american':

            style = 'a'

        else:

            style = 'e'

        s_steps = self._ssteps_pde

        t_steps = self._tsteps_pde

        T = self._maturity

        dt = T / t_steps

        K = self._strike

        spot_max = K * self._spot_max_factor_pde

        if spot > spot_max: # if the spot is outside the grids

            fwd= self.forward(spot,rate_c,rate_a)

            if cp=='c':

                pl = max(fwd-K,0) * math.exp(-rate_c*T)

                if style =='a':
                    
                    pl = max (spot - K,pl)
                
                delta = 1


            else:

                pl=max(K-fwd,0) * math.exp(-rate_c*T)

                if style =='a':
                    
                    pl = max (K-spot,pl)

                delta = 0

            gamma = 0


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

                if cp=='c':

                    grid[i,t_steps] = max (spot_rng[i]-K,0) 

                else:
                    grid[i,t_steps] = max (K-spot_rng[i],0) 
    

            for j in range(t_steps): # boundry condition at spot =0 and spot = s_max

                DF_t =  math.exp(-rate_c*(T-j*dt))

                F_t = spot_rng[s_steps] * math.exp((rate_c-rate_a) * (T-j*dt))

                if cp=='c':

                    grid[0,j] = 0

                    if style =='a':
                        grid[s_steps,j] = max(spot_rng[s_steps] - K,0)
                    else:
                        grid[s_steps,j] = max(F_t - K,0) * DF_t
                
                else:
                    if style =='a':
                        grid[0,j] = max(K,0)
                    else:
                        grid[0,j] = max(K,0)* DF_t

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

                gamma = (y_up -2*y + y_down)/(ds**2)

                    
            else:  # spot falls in between nodes

                pl0 = grid[x0,0]
                pl1 = grid[x0+1,0]
                
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

        
        if greeks.lower() == 'delta':

            return delta * Q
            

        elif greeks.lower() == 'gamma':

            return gamma * Q

        else:

            return pl * Q


    def pricing_pde_alt(self,spot,vol,rate_c,rate_a,greeks='pl'):   # PDE with enhanced grids density around spot

        if spot<=0:

            spot = self._spot_minimum

        Q = self._quantity

        if self._cp.lower()=='call' or self._cp.lower()=='c':

            cp='c'

        else:

            cp='p'
        
        if self._style.lower() == 'a' or self._style.lower()=='american':

            style = 'a'

        else:

            style = 'e'

        s_steps = self._ssteps_pde

        t_steps = self._tsteps_pde

        T = self._maturity

        dt = T / t_steps

        K = self._strike

        spot_max = K * self._spot_max_factor_pde

        if spot > spot_max: # if the spot is outside the grids

            fwd= self.forward(spot,rate_c,rate_a)

            if cp=='c':

                pl = max(fwd-K,0) * math.exp(-rate_c*T)

                delta = math.exp(-rate_a*T) * 1

                if style =='a':
                    
                    pl = max (spot - K,pl)
                
            else:

                pl=max(K-fwd,0) * math.exp(-rate_c*T)

                if style =='a':
                    
                    pl = max (K-spot,pl)

                delta = 0

            gamma = 0


        else:   # if the spot is within the grids
        
            ds = spot_max / s_steps

            x = spot/ds

            x0 = int(x)

            x1 = x0+1

            if x1-x<=10e-8:

                 x0 = x1

            spot_rng = np.linspace(0,spot_max,s_steps+1)

            n = s_steps
#############################################################################################################################

            if abs(x-x0) > 10e-6:  # if spot falls in between nodes

                ds_local = 0.5 * min(spot-spot_rng[x0],spot_rng[x0+1]-spot)

                spot_pre= spot -ds_local

                spot_aft = spot + ds_local

                spot_rng=np.insert(spot_rng,x0+1,[spot_pre,spot,spot_aft])

                spot_index = x0+2

                n = s_steps+3  # update the number of spot steps, 3 steps are added

            else:  # spot just falls on grid nodes

                ds_local = 0.5 * ds
                
                if x0 == 0:  # spot falls on node 0

                    spot_aft = 0 + ds_local

                    spot_rng = np.insert(spot_rng,x0+1,spot_aft)

                    spot_index = x0

                    n = s_steps + 1

                elif x0 == s_steps: # spot falls on last node s_max

                    spot_pre = spot_max - ds_local

                    spot_rng = np.insert(spot_rng,x0,spot_pre)

                    spot_index = x0 + 1

                    n = s_steps + 1

                else: # spot falls on other node

                    spot_pre = spot - ds_local

                    spot_aft = spot + ds_local

                    spot_rng = np.insert(spot_rng,x0,spot_pre)

                    spot_index = x0+1

                    spot_rng = np.insert(spot_rng,spot_index+1,spot_aft)

                    n = s_steps + 2

            grid= np.zeros((n+1,t_steps+1))

            for i in range(n+1):  # boundry condition at T: for payoff corresponds to each spot prices at maturity

                if cp=='c':

                    grid[i,t_steps] = max (spot_rng[i]-K,0) 

                else:
                    grid[i,t_steps] = max (K-spot_rng[i],0) 

            
            for j in range(t_steps): # boundry condition at spot =0 and spot = s_max

                DF_t =  math.exp(-rate_c*(T-j*dt))

                F_t = spot_rng[n] * math.exp((rate_c-rate_a) * (T-j*dt))

                if cp=='c':   # if the option is a call

                    grid[0,j] = 0

                    if style =='a':
                        grid[n,j] = max(spot_rng[n] - K,0)
                    else:
                        grid[n,j] = max(F_t - K,0) * DF_t
                
                else:   # if the option is a put
                    if style =='a':
                        grid[0,j] = max(K,0)
                    else:
                        grid[0,j] = max(K,0)* DF_t

                    grid[n,j] = 0
            
            for t in range(t_steps-1,-1,-1):  # from t=t_step-1 to t=0

                A = np.zeros((n-1,n-1))  

                B = np.zeros((1,n-1))

                for i in range(1,n):   # index from 1 to n-1

                    s_i = spot_rng[i]  
                    ds_up = spot_rng[i+1]-s_i
                    ds_down = s_i-spot_rng[i-1]

                    a = 0.5* (vol**2) * (s_i**2) /(ds_down **2)- (rate_c-rate_a)*s_i / (ds_up+ds_down)
                    b = -1/dt - 0.5 * (vol**2)*(s_i**2)/ds_down*(1/ds_up + 1/ds_down) - rate_c
                    c = 0.5 * (vol**2) * (s_i**2)/(ds_up * ds_down) + (rate_c - rate_a)*s_i/(ds_up + ds_down)
                    d =- 1/ dt
                    
                    # construct matrix A and B in AX=B
                    if i == 1:

                        B[0,i-1] = d * grid[i,t+1] -  a * grid[i-1,t]  

                        A[i-1,i]=c
    
                    elif i == n-1:

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

                grid[1:n,t] = V[:,0]

#########################################################################################################
        # plt.figure()
        # plt.scatter(spot_rng,grid[0:n+1,0], label='test')
        # plt.show()

        if spot_index == 0:

            pl = grid[spot_index,0]
            pl_up = grid[spot_index+1,0]
            ds_up = spot_rng[spot_index+1]-spot_rng[spot_index]
            delta = (pl_up-pl) / ds_up

            gamma = 0

        elif spot_index == n:

            pl = grid[spot_index,0]
            pl_down = grid[spot_index-1,0]
            ds_down =  spot_rng[spot_index]-spot_rng[spot_index-1]
            delta = (pl - pl_down)/ds_down

            gamma = 0

        else:

            pl = grid[spot_index,0]
            pl_up = grid[spot_index+1,0]
            pl_down = grid[spot_index-1,0]

            ds_up = spot_rng[spot_index+1]-spot_rng[spot_index]
            ds_down =  spot_rng[spot_index]-spot_rng[spot_index-1]

            delta = (pl_up-pl_down)/(ds_up+ds_down)
            delta_up = (pl_up - pl)/ds_up
            delta_down = (pl - pl_down)/ ds_down

            gamma = (delta_up - delta_down)/ds_down

        
        if greeks.lower() == 'delta':

            return delta * Q
            

        elif greeks.lower() == 'gamma':

            return gamma * Q

        else:

            return pl * Q

 

    def spot_ladder(self, spot_start, spot_end, spot_step,vol,rate_c,rate_a,greeks,model_alt=None, showdiff=False):

        spot_rng = np.arange(spot_start,spot_end,spot_step)

        greeks_value =[]

        model = self._default_model

        if model_alt is not None:
            
            greeks_value_alt =[]

            greeks_value_diff = []

        for s in spot_rng:

            if self._displayprogress == True:

                n=int((spot_end-spot_start)/spot_step)
                i=int((s-spot_start)/spot_step)
                progress= int(i/n*100)
                print('Spot = %f, in progress %d complete' % (s, progress))
           
            if greeks.lower() == 'pl' or greeks.lower() == 'mv':

                value = model(s,vol,rate_c,rate_a)
                
                if model_alt is not None:

                    value_alt =model_alt(s,vol,rate_c,rate_a)
                
            elif greeks.lower()=='delta':
                
                value = self.delta(s,vol,rate_c,rate_a)

                if model_alt is not None:

                    value_alt =self.delta(s,vol,rate_c,rate_a,model_alt)

            elif greeks.lower()=='gamma':
                
                value = self.gamma(s,vol,rate_c,rate_a)

                if model_alt is not None:

                    value_alt =self.gamma(s,vol,rate_c,rate_a,model_alt)

            elif greeks.lower()=='vega':
                
                value = self.vega(s,vol,rate_c,rate_a)

                if model_alt is not None:

                    value_alt =self.vega(s,vol,rate_c,rate_a,model_alt)

            elif greeks.lower()=='theta':
                
                value = self.theta(s,vol,rate_c,rate_a)

                if model_alt is not None:

                    value_alt =self.theta(s,vol,rate_c,rate_a,model_alt)

            elif greeks.lower()=='volga':

                value = self.volga(s,vol,rate_c,rate_a)

                if model_alt is not None:

                    value_alt =self.volga(s,vol,rate_c,rate_a,model_alt)
            
            elif greeks.lower()=='vanna':

                value = self.vanna(s,vol,rate_c,rate_a)

                if model_alt is not None:

                    value_alt =self.vanna(s,vol,rate_c,rate_a,model_alt)

            elif greeks.lower()=='rho':

                value = self.rho(s,vol,rate_c,rate_a)

                if model_alt is not None:

                    value_alt =self.rho(s,vol,rate_c,rate_a,model_alt)
          
            greeks_value.append(value)

            if self._displayprogress == True:

                print('%s by Model1: %.10f' % (greeks.lower(), value))

            if model_alt is not None:

                value_diff = value_alt - value

                # value_diff_per = (value_diff)/value * 100

                greeks_value_alt.append(value_alt)

                greeks_value_diff.append(value_diff)

                if self._displayprogress == True:

                    print('%s by Model2: %.10f' % (greeks.lower(), value_alt))
        
        y = greeks_value

        if model_alt is not None:

            y_alt = greeks_value_alt

            y_diff = greeks_value_diff
            
        y_label = greeks.lower()
        #y_reg = np.polyfit(spot_rng,y,6)
        #y_fit = np.polyval(y_reg,spot_rng)      

        plt.figure(figsize=(14,10))

        if showdiff == False:
            plt.plot(spot_rng,y,label=y_label+': model1')
            #plt.plot(spot_rng,y_fit,label=y_label+' fit curve')

        if model_alt is not None:

            #y_reg_alt = np.polyfit(spot_rng,y_alt,6)
            #y_fit_alt = np.polyval(y_reg_alt,spot_rng)

            if showdiff == False:   
                plt.plot(spot_rng,y_alt,label=y_label + ': model2')
                # plt.plot(spot_rng,y_fit_alt,label=y_label+' fit curve'+'-model 2')
            else:
                plt.plot(spot_rng,y_diff,label=y_label + ': model2-model1')

        plt.xlabel('spot')
        plt.ylabel(y_label)
        plt.legend(loc='best')
        plt.title('Vanilla option Greeks') 
        plt.show()

        return None

                        
def main_vanilla():

    underlying='spy'
    assetclass='EQD'
    spot=50
    vol=0.3
    T=1
    K =50
    rate_usd=0.005
    div_spy=0.02
    quantity = 1
    cp='put'

    op = Vanilla(underlying,assetclass,T,K,'E',cp,quantity)

    op._nsteps_crr=100
    op._npaths_mc=10000
    op._nsteps_mc=200
    op._rnd_seed=10000
    op._vega_bump_is_percent = False
    op._vega_bump=0.05

    op._rnd_seed = 6666

    op._delta_bump=1


    op._spot_max_factor_pde=5

    op._ssteps_pde=200
    op._tsteps_pde=100

    op._displayprogress = True

    op.spot_ladder(1,150,1,vol,rate_usd,div_spy,'vega',op.pricing_pde,False)

if __name__ =='__main__':

    main_vanilla()
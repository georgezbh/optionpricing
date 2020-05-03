'option class pricing module'

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
        self._vega_bump=0.1
        self._vega_bump_is_percent=False
        self._theta_bump=1/365

        self._vanna_dvega = False
    
        self._npaths_mc = 1000
        self._nsteps_mc = 300
        self._rnd_seed = 12345
        self._nsteps_crr = 101

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

    def pricing_bsm(self,spot,vol,rate_c,rate_a):

        if spot<=0:
            spot=0.0001
        
        fwd = self.forward(spot,rate_c,rate_a)

        Q = self._quantity

        T = self._maturity
        K = self._strike
        D = math.exp(-rate_c*T)

        N=norm.cdf     # standard normal distribution cumulative function

        d1 = (math.log(fwd/K)+0.5*vol**2*T)/(vol*math.sqrt(T)) 
        d2 = d1-vol*math.sqrt(T)

        call_price = D*(N(d1)*fwd - N(d2)*K) * Q

        put_price = D*(N(-d2)*K-N(-d1)*fwd) * Q

        if self._cp == 'call':

            return call_price
        
        elif self._cp =='put':

            return put_price
        
        else:

            return None

    def delta(self,spot,vol,rate_c,rate_a,model_alt=None):

        if spot<=0:
            spot=0.0001

        if model_alt is None:

            model = self._default_model
        else:
            model = model_alt

        if model == self.pricing_crr:

            delta_value = self.pricing_crr(spot,vol,rate_c,rate_a,'delta')

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
            spot=0.0001

        if model_alt is None:

            model = self._default_model
        else:
            model = model_alt

        if model == self.pricing_crr:

            gamma_value = self.pricing_crr(spot,vol,rate_c,rate_a,'gamma')

        else:    
            price = model(spot,vol,rate_c,rate_a)

            bumpvalue = self._gamma_bump

            if self._gamma_bump_is_percent == True:

                spot_up = spot * (1+bumpvalue/100)
                spot_down = max(spot * (1-bumpvalue/100),0.000000001)
            
            else:                                 # then the bumpvalue is absolute

                spot_up = spot + bumpvalue
                spot_down = max(spot - bumpvalue,0.000000001)
            
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
            spot=0.0001

        price = model(spot,vol,rate_c,rate_a)

        bumpvalue=self._vega_bump

        if self._vega_bump_is_percent == True:

            vol_up = vol * (1+bumpvalue/100)
            
        else:                                 # then the bumpvalue is absolute

            vol_up = vol + bumpvalue
            
        price_up = model(spot,vol_up,rate_c,rate_a)

        vega_value = (price_up- price)/(vol_up-vol)
        
        return vega_value
    
    def theta(self,spot,vol,rate_c,rate_a,model_alt=None):

        if model_alt is None:

            model = self._default_model
        else:
            model = model_alt

        if spot<=0:
            spot=0.0001

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
            spot=0.0001

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
            spot=0.0001

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
            spot=0.0001

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
            spot=0.0001

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
            spot=0.0001

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


    def pricing_mc2(self,spot,vol,rate_c,rate_a):

        Q = self._quantity

        if spot<=0:
            spot=0.0001

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
            spot=0.0001
        
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


    def spot_ladder(self, spot_start, spot_end, spot_step,vol,rate_c,rate_a,greeks,model_alt=None):

        spot_rng = np.arange(spot_start,spot_end,spot_step)

        greeks_value =[]

        model = self._default_model

        if model_alt is not None:
            
            greeks_value_alt =[]

        for s in spot_rng:

            n=(spot_end-spot_start)/spot_step -1
            i=(s-spot_start)/spot_step
            progress= int(i/n*100)

            print('Spot = %f, in progress %d complete' % (s, progress))
           
            if greeks.lower() == 'pl':

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
          
            greeks_value.append(value)

            print('Model1:',value)

            if model_alt is not None:
                greeks_value_alt.append(value_alt)

                print('Model2:',value_alt)
        
        y = greeks_value

        if model_alt is not None:

            y_alt = greeks_value_alt

            
        y_label = greeks.lower()
        y_reg = np.polyfit(spot_rng,y,6)
        y_fit = np.polyval(y_reg,spot_rng)      

        
        plt.figure()
        plt.plot(spot_rng,y,label=y_label)

        #plt.plot(spot_rng,y_fit,label=y_label+' fit curve')

        if model_alt is not None:

            y_reg_alt = np.polyfit(spot_rng,y_alt,6)
            y_fit_alt = np.polyval(y_reg_alt,spot_rng)   

            plt.plot(spot_rng,y_alt,label=y_label + '---model 2')
            #plt.plot(spot_rng,y_fit_alt,label=y_label+' fit curve'+'-model 2')

        plt.xlabel('spot')
        plt.ylabel(y_label)
        plt.legend(loc=4)
        plt.title('Vanilla option') 
        plt.show()

        return None

                        
def main_vanilla():

    underlying='spy'
    assetclass='EQD'
    spot=50
    vol=0.3
    T=0.5
    K = 50
    rate_usd=0.01
    div_spy=0.0
    quantity = 100
    cp='call'

    op = Vanilla(underlying,assetclass,T,K,'e',cp,quantity)

    op._nsteps_crr=300
    op._npaths_mc=100000
    op._nsteps_mc=200
    op._rnd_seed=6666
    op._vega_bump_is_percent = False

    # op._vanna_dvega = True

    #opmc2=op.pricing_mc2(spot,vol,rate_usd,div_spy)
    #print(opmc2)
    op.spot_ladder(0,150,2,vol,rate_usd,div_spy,'theta',op.pricing_mc2)


if __name__ =='__main__':

    main_vanilla()
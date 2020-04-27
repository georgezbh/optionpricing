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
    
    def __init__(self, underlying, asset_class, maturity, strike, style, quantity):

        super().__init__(underlying,asset_class)

        self._maturity = maturity

        self._strike = strike

        self._style = style

        self._quantity=quantity
    
        self._delta_bump=0.1
        self._delta_bump_is_percent=True
        self._gamma_bump=0.1
        self._gamma_bump_is_percent=True
        self._vega_bump=0.1
        self._vega_bump_is_percent=False
        self._theta_bump=1/365
    
    
        self._npaths_mc = 1000
        self._nsteps_mc = 300
        self._rnd_seed = 6666
        self._nsteps_crr = 101


    def intrinsicvalue(self,spot):

        intrinsicvalue_c = max(spot-self._strike,0)

        intrinsicvalue_p = max(self._strike - spot, 0)

        return {'call': intrinsicvalue_c, 'put': intrinsicvalue_p}
    

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
        
        return {'call':call_price,'put':put_price}

    def delta_fd(self,spot,vol,rate_c,rate_a,model):

        if spot<=0:
            spot=0.0001

        Q = self._quantity

        price = model(spot,vol,rate_c,rate_a)

        bumpvalue = self._delta_bump

        if self._delta_bump_is_percent == True:

            spot_up = spot * (1+bumpvalue/100)
            
        else:                                 

            spot_up = spot + bumpvalue
            
        price_up = model(spot_up,vol,rate_c,rate_a)

        delta_call = (price_up['call']- price['call'])/(spot_up-spot) * Q

        delta_put = (price_up['put'] - price['put'])/(spot_up-spot) * Q
        
        return {'call': delta_call, 'put': delta_put}


    def gamma_fd(self,spot,vol,rate_c,rate_a,model):

        Q = self._quantity

        if spot<=0:
            spot=0.0001

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

        gamma_call = (price_up['call'] + price_down['call']-2*price['call'])/((spot_up-spot)**2) * Q

        gamma_put = (price_up['put'] + price_down['put']-2*price['put'])/((spot_up-spot)**2) * Q

        return {'call':gamma_call,'put':gamma_put}

    def vega_fd(self,spot,vol,rate_c,rate_a,model):

        Q = self._quantity

        if spot<=0:
            spot=0.0001

        price = model(spot,vol,rate_c,rate_a)

        bumpvalue=self._vega_bump

        if self._vega_bump_is_percent == True:

            vol_up = vol * (1+bumpvalue/100)
            
        else:                                 # then the bumpvalue is absolute

            vol_up = vol + bumpvalue
            
        price_up = model(spot,vol_up,rate_c,rate_a)

        vega_call = (price_up['call']- price['call'])/(vol_up-vol) * Q

        vega_put = (price_up['put'] - price['put'])/(vol_up-vol) * Q

        
        return {'call': vega_call, 'put': vega_put}
    
    def theta_fd(self,spot,vol,rate_c,rate_a,model):

        Q = self._quantity

        if spot<=0:
            spot=0.0001

        price = model(spot,vol,rate_c,rate_a)

        bumpvalue = self._theta_bump

        self._maturity= self._maturity - bumpvalue

        price_shift = model(spot,vol,rate_c,rate_a)

        theta_call = (price_shift['call']- price['call'])*Q

        theta_put = (price_shift['put'] - price['put'])*Q

        self._maturity= self._maturity + bumpvalue
        
        return {'call': theta_call, 'put': theta_put}
    
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

        return S


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

        return {'call':call_price, 'put':put_price}


    def pricing_crr(self,spot,vol,rate_c,rate_a):  # pricing option using binomial tree, time steps is n_steps

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

        return {'call':call_price,'put':put_price,'call delta':call_delta,'put delta':put_delta,'call gamma':call_gamma,'put gamma':put_gamma}
        #return put_tree


    def spot_ladder(self, spot_start, spot_end, spot_step,vol,rate_c,rate_a,greeks_fn,model_alt=None):

        spot_rng = np.arange(spot_start,spot_end,spot_step)

        c_greeks_value =[]
        p_greeks_value =[]

        if model_alt is not None:
            c_greeks_value_alt =[]
            p_greeks_value_alt =[]

        for s in spot_rng:

            n=(spot_end-spot_start)/spot_step -1
            i=(s-spot_start)/spot_step
            progress= int(i/n*100)
           
            print('Spot = %f, in progress %d complete' % (s, progress))

            if self._style.lower() == 'a':

                pricing_model = self.pricing_crr

            else:

                pricing_model = self.pricing_bsm

            if greeks_fn.lower() == 'pl':

                c_value = pricing_model(s,vol,rate_c,rate_a)['call']
                p_value = pricing_model(s,vol,rate_c,rate_a)['put']
                
                if model_alt is not None:

                    c_value_alt =model_alt(s,vol,rate_c,rate_a)['call']
                    p_value_alt =model_alt(s,vol,rate_c,rate_a)['put']
                
            elif greeks_fn.lower()=='delta':
                
                if self._style.lower() == 'a':

                    c_value = pricing_model(s,vol,rate_c,rate_a)['call delta']
                    p_value = pricing_model(s,vol,rate_c,rate_a)['put delta']

                else:

                    c_value = self.delta_fd(s,vol,rate_c,rate_a,pricing_model)['call']
                    p_value = self.delta_fd(s,vol,rate_c,rate_a,pricing_model)['put']

                if model_alt is not None:

                    c_value_alt =self.delta_fd(s,vol,rate_c,rate_a,model_alt)['call']
                    p_value_alt =self.delta_fd(s,vol,rate_c,rate_a,model_alt)['put']

            elif greeks_fn.lower()=='gamma':
                
                if self._style.lower() == 'a':

                    c_value = pricing_model(s,vol,rate_c,rate_a)['call gamma']
                    p_value = pricing_model(s,vol,rate_c,rate_a)['put gamma']

                else:

                    c_value = self.gamma_fd(s,vol,rate_c,rate_a,pricing_model)['call']
                    p_value = self.gamma_fd(s,vol,rate_c,rate_a,pricing_model)['put']

                if model_alt is not None:

                    c_value_alt =self.gamma_fd(s,vol,rate_c,rate_a,model_alt)['call']
                    p_value_alt =self.gamma_fd(s,vol,rate_c,rate_a,model_alt)['put']

            elif greeks_fn.lower()=='vega':
                
                c_value = self.vega_fd(s,vol,rate_c,rate_a,pricing_model)['call']
                p_value = self.vega_fd(s,vol,rate_c,rate_a,pricing_model)['put']

                if model_alt is not None:

                    c_value_alt =self.vega_fd(s,vol,rate_c,rate_a,model_alt)['call']
                    p_value_alt =self.vega_fd(s,vol,rate_c,rate_a,model_alt)['put']

            elif greeks_fn.lower()=='theta':
                
                c_value = self.theta_fd(s,vol,rate_c,rate_a,pricing_model)['call']
                p_value = self.theta_fd(s,vol,rate_c,rate_a,pricing_model)['put']

                if model_alt is not None:

                    c_value_alt =self.theta_fd(s,vol,rate_c,rate_a,model_alt)['call']
                    p_value_alt =self.theta_fd(s,vol,rate_c,rate_a,model_alt)['put']
            

            c_greeks_value.append(c_value)
            p_greeks_value.append(p_value)

            if model_alt is not None:
                c_greeks_value_alt.append(c_value_alt)
                p_greeks_value_alt.append(p_value_alt)
        
        y_c = c_greeks_value
        y_p = p_greeks_value

        if model_alt is not None:

            y_c_alt = c_greeks_value_alt
            y_p_alt = p_greeks_value_alt
            
        y_label = greeks_fn.lower()
        y_reg_c = np.polyfit(spot_rng,y_c,6)
        y_reg_p = np.polyfit(spot_rng,y_p,6)    
        y_fit_c = np.polyval(y_reg_c,spot_rng)
        y_fit_p = np.polyval(y_reg_p,spot_rng)      

        
        plt.figure()
        plt.plot(spot_rng,y_c,label='call '+ y_label)
        plt.plot(spot_rng,y_p,label='put '+ y_label)
        #plt.plot(spot_rng,y_fit_c,label='call '+ y_label+' fit curve')
        #plt.plot(spot_rng,y_fit_p,label='put '+ y_label+' fit curve')

        if model_alt is not None:

            y_reg_c_alt = np.polyfit(spot_rng,y_c_alt,6)
            y_reg_p_alt = np.polyfit(spot_rng,y_p_alt,6)    
            y_fit_c_alt = np.polyval(y_reg_c_alt,spot_rng)
            y_fit_p_alt = np.polyval(y_reg_p_alt,spot_rng)   

            plt.plot(spot_rng,y_c_alt,label='call '+ y_label + '---model 2')
            plt.plot(spot_rng,y_p_alt,label='put '+ y_label + '---model 2')
            #plt.plot(spot_rng,y_fit_c_alt,label='call '+ y_label+' fit curve'+'-model 2')
            #plt.plot(spot_rng,y_fit_p_alt,label='put '+ y_label+' fit curve'+'-model 2')  

        plt.xlabel('spot')
        plt.ylabel(y_label)
        plt.legend(loc=4)
        plt.title('Vanilla option') 
        plt.show()

        return None



class TARF(Option):

    def __init__(self, underlying, asset_class, strike, barrier,barrier_type, target, gear, notional, fixing_schedule ):

        super().__init__(underlying,asset_class)

        self._strike=strike
        self._barrier=barrier
        self._barrier_type=barrier_type
        self._target=target
        self._gear=gear
        self._notional=notional
        self._fixing_schedule = fixing_schedule


        self._npaths_mc = 10000
        self._nsteps_mc = 300
        self._rnd_seed = 6666

        self._delta_bump=1
        self._delta_bump_is_percent=True
        self._gamma_bump=1
        self._gamma_bump_is_percent=True
        self._vega_bump=0.01
        self._vega_bump_is_percent=False
        self._theta_bump=1/365

    
    def gen_one_path(self,spot,vol,rate_d,rate_f):  # simulate and return the fixing spots along fixing schedule

        if spot<=0:
            spot=0.0001

        n_steps = self._nsteps_mc

        T=self._fixing_schedule[-1] 

        dt = T / n_steps
        s=[spot]
        sigma = math.sqrt(dt)

        for _ in range(n_steps):

            ds= s[-1]*((rate_d-rate_f) * dt + vol * random.gauss(0,sigma))
            s.append(s[-1]+ds)

        # print (s)
        
        s_fix = []

        for one_fix_time in self._fixing_schedule:

            one_fix_index = int(one_fix_time/dt+0.5)

            s_fix.append(s[one_fix_index])

        return s_fix

    def pricing_mc(self,spot,vol,rate_d,rate_f,debug=False):

        n_steps = self._nsteps_mc
        n_paths = self._npaths_mc
        rndseed = self._rnd_seed

        if rndseed is not None:
            random.seed(rndseed)

        T= self._fixing_schedule[-1] 

        K = self._strike

        TIV = self._target

        N = self._notional

        B = self._barrier

        G = self._gear

        n_fix = len(self._fixing_schedule)  # total number of fixings is n_fix

        stat_fix = [0] * n_fix  # create an array n_fix *1  to store statistics of knock out event

        t_fix = self._fixing_schedule

        payoff_by_path = []


        for i in range(n_paths):

            s_fix=self.gen_one_path(spot,vol,rate_d,rate_f) 

            # print (s_fix)  

            civ = 0

            payoff = 0 

            for j in range(n_fix):

                df = math.exp(-rate_f * t_fix[j])

                if civ < TIV:
                    
                    if s_fix[j] <= K:

                        if (civ + max(K-s_fix[j],0)) < TIV:

                            payoff = payoff + df * N * (K - s_fix[j]) / s_fix[j]

                        else:      # knock out event happens

                            payoff = payoff + df * (TIV - civ) * N / s_fix[j]

                            civ = TIV

                            stat_fix[j] = stat_fix[j] + 1

                            # print('knock out at fixings %i, spot is %.4f !' % (j+1, s_fix[j]))

                            break
                
                    elif s_fix[j] > K and s_fix[j] <= B:

                        civ = civ

                        payoff = payoff
                
                    elif s_fix[j] > B:

                        payoff = payoff - df * N * G * (s_fix[j] - K)
            
            payoff_by_path.append(payoff)

        price = statistics.mean(payoff_by_path)

        if debug == True:

            print('TARF price = %.4f' % price)

            path_no_knock = n_paths - sum(stat_fix)

            percent_no_knock = path_no_knock / n_paths *100

            print('Knock out distribution:')

            for i in range(n_fix):

                stat_fix[i] = stat_fix[i] / n_paths * 100

                print('%.2f%%' % stat_fix[i], end=', ')
            
            print ('Percentage never knock out: %.2f%%. \n' % percent_no_knock)

        return price



    def delta(self,spot,vol,rate_d,rate_f,model):

        if spot<=0:

            spot = 0.0001

        bumpvalue = self._delta_bump

        is_bump_percent = self._delta_bump_is_percent

        price = model(spot,vol,rate_d,rate_f)

        if is_bump_percent == True:

            spot_up = spot *(1+bumpvalue/100)

        else:

            spot_up = spot + bumpvalue

        price_up = model(spot_up,vol,rate_d,rate_f)

        delta_fd = (price_up - price) / (spot_up - spot)

        return delta_fd


    def gamma(self,spot,vol,rate_d,rate_f,model):

        if spot<=0:

            spot = 0.0001

        bumpvalue = self._gamma_bump

        is_bump_percent = self._gamma_bump_is_percent

        price = model(spot,vol,rate_d,rate_f)

        if is_bump_percent == True:

            spot_up = spot *(1+bumpvalue/100)
            spot_down = spot * (1-bumpvalue/100)

        else:

            spot_up = spot + bumpvalue
            spot_down = spot - bumpvalue

        price_up = model(spot_up,vol,rate_d,rate_f)
        price_down = model(spot_down,vol,rate_d,rate_f)

        gamma_fd = (price_up + price_down - 2*price) / ((spot_up - spot)**2)

        return gamma_fd

    def vega(self,spot,vol,rate_d,rate_f,model):

        if spot<=0:
            spot=0.0001

        price = model(spot,vol,rate_d,rate_f)

        bumpvalue=self._vega_bump

        if self._vega_bump_is_percent == True:

            vol_up = vol * (1+bumpvalue/100)
            
        else:                                 

            vol_up = vol + bumpvalue
            
        price_up = model(spot,vol_up,rate_d,rate_f)

        vega_fd = (price_up - price) / (vol_up - vol)
        
        return vega_fd

    def theta(self,spot,vol,rate_d,rate_f,model):

        if spot<=0:
            spot=0.0001

        price = model(spot,vol,rate_d,rate_f)

        bumpvalue = self._theta_bump

        old_fixing_time = self._fixing_schedule

        new_fixing_time = []

        for fix_time in old_fixing_time:

            new_time = fix_time - bumpvalue

            if new_time > 0:

                new_fixing_time.append(new_time)

        self._fixing_schedule = new_fixing_time

        price_shift = model(spot,vol,rate_d,rate_f)

        theta_fd= price_shift - price

        self._fixing_schedule = old_fixing_time
        
        return theta_fd

    
    def spot_ladder(self,spot_start,spot_end,spot_step,vol,rate_d,rate_f,target_fn,fit_df=6):


        spot_rng = np.arange(spot_start,spot_end,spot_step)

        n=(spot_end-spot_start+spot_step)/spot_step

        # print('total steps is: %d' % n)

        y = []

        for s in spot_rng:
  
            i=(s-spot_start)/spot_step+1

            progress= int(i/n * 100)

            #if progress % 10 ==0:
            print('Spot = %f, in progress %d complete' % (s, progress))

            if target_fn.lower() == 'pl':

                PL = self.pricing_mc(s,vol,rate_d,rate_f,True)

                y.append(PL)

            elif target_fn.lower() == 'delta':

                delta = self.delta(s,vol,rate_d,rate_f,self.pricing_mc)

                y.append(delta)

            elif  target_fn.lower() == 'gamma':

                gamma = self.gamma(s,vol,rate_d,rate_f,self.pricing_mc)

                y.append(gamma)

            elif target_fn.lower() == 'vega':

                vega = self.vega(s,vol,rate_d,rate_f,self.pricing_mc)

                y.append(vega)
            
            elif target_fn.lower() == 'theta':

                theta = self.theta(s,vol,rate_d,rate_f,self.pricing_mc)

                y.append(theta)


        y_label = target_fn.lower()
        reg = np.polyfit(spot_rng,y,fit_df)  # returns the fit curve polynominal function, order is fit_df
        y_fit = np.polyval(reg,spot_rng)   # return the y value array given polynominal function and x value array

        # print(y_fit)
        
        plt.figure()
        plt.scatter(spot_rng,y,c='b',label=y_label)
        plt.plot(spot_rng,y_fit,c='r',label=y_label+' fit curve')
        plt.xlabel('spot')
        plt.ylabel(y_label)
        plt.legend(loc=4)
        plt.title('Target redemption forward') 
        plt.show()

                

            
 
        
def main_vanilla():

    underlying='spy'
    assetclass='EQD'
    spot=50
    vol=0.3
    T=0.5
    K = 50
    rate_usd=0.01
    div_spy=0.03
    quantity = 100

    op = Vanilla(underlying,assetclass,T,K,'a',quantity)

    op._nsteps_crr=300
    op._npaths_mc=5000
    op._nsteps_mc=100
    op._rnd_seed=54321

    op.spot_ladder(5,100,5,vol,rate_usd,div_spy,'delta',op.pricing_bsm)




def main_tarf():

    underlying = 'USDCNH'
    assetclass='FX'
    spot = 7.0730
    vol = 0.05
    rate_cnh = 0.015
    rate_usd = 0.01
    notional = 1
    strike = 7.28
    barrier = 7.38
    barrier_type = 'KI'
    target = 0.2
    gear =2
    fixings_schedule = []

    for i in range(9):

        fixing_time = (i+1)  * 15/365

        fixings_schedule.append(fixing_time)

    # print(fixings_schedule)


    tarf1 = TARF(underlying,assetclass,strike,barrier,barrier_type,target,gear,notional,fixings_schedule)

    tarf1._nsteps_mc =10*3

    #price = tarf1.pricing_mc(spot,vol,rate_cnh,rate_usd,270,10000,None)
    #print(price)

    tarf1._npaths_mc = 5000
    tarf1._nsteps_mc = 270
    tarf1._vega_bump = 0.001
    tarf1._gamma_bump=1
    tarf1._gamma_bump_is_percent=True
    tarf1._rnd_seed = 10000

    tarf1.spot_ladder(6.6,8.0,0.02,vol,rate_cnh,rate_usd,'theta',6)

if __name__ =='__main__':

    main_vanilla()
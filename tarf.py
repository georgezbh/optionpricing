'TARF pricing module'

__author__ = 'George Zhao'


import math

import statistics

import random

import numpy as np

import matplotlib.pyplot as plt

from scipy.stats import norm

from option import Option


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

    main_tarf()
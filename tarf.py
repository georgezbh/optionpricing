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

    
    def gen_one_path(self,spot,vol,rate_d,rate_f):  # simulate and return the fixing spots along fixing schedule

        if spot<=0:
            spot=self._spot_minimum

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
    
    def gen_fixings(self,spot,vol,rate_d,rate_f):

        if spot<=0:
            spot = self._spot_minimum

        s_fix =[]

        t_fix = self._fixing_schedule

        for index in range(len(self._fixing_schedule)):


            if index ==0:

                t = t_fix[index]

                sigma = math.sqrt(t)

                fix = spot * math.exp((rate_d-rate_f-0.5*vol**2)*t+vol*random.gauss(0,sigma))

            else:

                t = t_fix[index] - t_fix[index-1]

                sigma = math.sqrt(t)

                fix = s_fix[index-1] * math.exp((rate_d-rate_f-0.5*vol**2)*t+vol*random.gauss(0,sigma))

            s_fix.append(fix)

        return s_fix


    def mc(self,spot,vol,rate_d,rate_f,debug=False):

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

            #s_fix=self.gen_one_path(spot,vol,rate_d,rate_f) 
            
            s_fix = self.gen_fixings(spot,vol,rate_d,rate_f)
            # print (s_fix)  

            civ = 0

            payoff = 0 

            for j in range(n_fix):

                df = math.exp(-rate_f * t_fix[j])

                if civ < TIV:
                    
                    if s_fix[j] <= K:

                        if (civ + max(K-s_fix[j],0)) < TIV:

                            payoff = payoff + df * N * (K - s_fix[j]) / s_fix[j]

                            civ = civ + max(K-s_fix[j],0)

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

            print('Knocked out distribution:')

            for i in range(n_fix):

                stat_fix[i] = stat_fix[i] / n_paths * 100

                print('%.2f%%' % stat_fix[i], end=', ')
            
            print ('Percentage never knocked out: %.2f%%. \n' % percent_no_knock)

        return price


    def delta(self,spot,vol,rate_d,rate_f,model):

        if spot<=0:

            spot = self._spot_minimum

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

    def vanna(self,spot,vol,rate_d,rate_f,model):

        if spot<=0:
            spot=0.0001

        if self._vanna_dvega == False:

            delta = self.delta(spot,vol,rate_d,rate_f,model)

            bumpvalue = self._vega_bump

            if self._vega_bump_is_percent == True:

                vol_up = vol * (1+bumpvalue/100)
            
            else:                                 

                vol_up = vol + bumpvalue

            delta_up = self.delta(spot,vol_up,rate_d,rate_f,model)

            vanna_value = (delta_up - delta)/(vol_up-vol)

        else:

            vega = self.vega(spot,vol,rate_d,rate_f,model)

            bumpvalue = self._delta_bump

            if self._delta_bump_is_percent == True:

                spot_up = spot *(1+bumpvalue/100)

            else:

                spot_up = spot + bumpvalue

            vega_up = self.vega(spot,vol,rate_d,rate_f,model)

            vanna_value = (vega_up - vega)/(spot_up-spot)

        return vanna_value

    def volga(self,spot,vol,rate_d,rate_f,model):

        if spot<=0:
            spot=self._spot_minimum
        
        vega = self.vega(spot,vol,rate_d,rate_f,model)

        bumpvalue = self._vega_bump

        if self._vega_bump_is_percent == True:

            vol_up = vol * (1+bumpvalue/100)
            
        else:                                 

            vol_up = vol + bumpvalue

        vega_up = self.vega(spot,vol_up,rate_d,rate_f,model)

        volga_value = (vega_up - vega)/(vol_up-vol)

        return volga_value

    def rho(self,spot,vol,rate_d,rate_f,model):

        if spot<=0:

            spot = self._spot_minimum

        bumpvalue = self._rho_bump

        price = model(spot,vol,rate_d,rate_f)

        rate_d_up = rate_d + bumpvalue

        price_up = model(spot,vol,rate_d_up,rate_f)

        rho_d = (price_up - price) / (rate_d_up - rate_d) / 100

        return rho_d

    def rhof(self,spot,vol,rate_d,rate_f,model):

        if spot<=0:

            spot = self._spot_minimum

        bumpvalue = self._rho_bump

        price = model(spot,vol,rate_d,rate_f)

        rate_f_up = rate_f + bumpvalue

        price_up = model(spot,vol,rate_d,rate_f_up)

        rho_f = (price_up - price) / (rate_f_up - rate_f) / 100

        return rho_f

    
    
    def spot_ladder(self,spot_list, vol, rate_d, rate_f):

        n=len(spot_list)

        pl = []
        delta=[]
        gamma=[]
        vega=[]
        theta=[]
        rho=[]
        rhof=[]
        volga=[]
        vanna=[]

        i = 0

        for s in spot_list:
  
            progress= int(i/n * 100)

            print('Spot = %f, in progress %d complete' % (s, progress))

            y_pl= self.mc(s,vol,rate_d,rate_f,True)

            pl.append(y_pl)

            y_delta = self.delta(s,vol,rate_d,rate_f,self.mc)

            delta.append(y_delta)

            y_gamma = self.gamma(s,vol,rate_d,rate_f,self.mc)

            gamma.append(y_gamma)

            y_vega = self.vega(s,vol,rate_d,rate_f,self.mc)

            vega.append(y_vega)

            y_rho = self.rho(s,vol,rate_d,rate_f,self.mc)

            rho.append(y_rho)

            y_rhof = self.rhof(s,vol,rate_d,rate_f,self.mc)

            rhof.append(y_rhof)
        
            y_theta = self.theta(s,vol,rate_d,rate_f,self.mc)

            theta.append(y_theta)

            y_vanna = self.vanna(s,vol,rate_d,rate_f,self.mc)

            vanna.append(y_vanna)

            y_volga = self.volga(s,vol,rate_d,rate_f,self.mc)

            volga.append(y_volga)

            i= i + 1

        fig, ax = plt.subplots(ncols=4, nrows=2, figsize=(14,9))

        ax[0,0].set_title("P&L")
        ax[0,1].set_title("Delta")
        ax[0,2].set_title("Gamma")
        ax[0,3].set_title("Theta")
        ax[1,0].set_title("Rho")
        ax[1,1].set_title("Vega")
        ax[1,2].set_title("Volga")
        ax[1,3].set_title("Vanna")

        ax[0,0].plot(spot_list,pl,label='P&L')  
        ax[0,1].plot(spot_list,delta,label='Delta')
        ax[0,2].plot(spot_list,gamma,label='Gamma')
        ax[0,3].plot(spot_list,theta,label='Theta')
        ax[1,0].plot(spot_list,rho,label='Rho')
        ax[1,0].plot(spot_list,rhof,label='Rhof')
        ax[1,1].plot(spot_list,vega,label='Vega')
        ax[1,2].plot(spot_list,volga,label='Volga')
        ax[1,3].plot(spot_list,vanna,label='Vanna')

        ax[0,0].legend()
        ax[0,1].legend()
        ax[0,2].legend()
        ax[0,3].legend()
        ax[1,0].legend()
        ax[1,1].legend()
        ax[1,2].legend()
        ax[1,3].legend()
        
        #plt.legend(loc=0)

        fig.suptitle('Target Redemption Forward')
        plt.show()       
 
        # reg = np.polyfit(spot_list,pl,fit_df)  # returns the fit curve polynominal function, order is fit_df
        # pl_fit = np.polyval(reg,spot_list)   # return the y value array given polynominal function and x value array
        return None


def main_tarf():

    underlying = 'USDCNH'
    assetclass='FX'
    spot = 7.0730
    vol = 0.05
    rate_cnh = 0.015
    rate_usd = 0.01
    notional = -1
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

    tarf1._nsteps_mc =300
    tarf1._npaths_mc = 30000
    tarf1._nsteps_mc = 1000
    tarf1._vega_bump = 0.01
    tarf1._gamma_bump=0.1
    tarf1._gamma_bump_is_percent=True
    tarf1._rnd_seed = 10000

    spot_list= np.arange(6.6,8,0.02)

    tarf1.spot_ladder(spot_list,vol,rate_cnh,rate_usd)

if __name__ =='__main__':

    main_tarf()
class Option(object):

    def __init__(self, otype, style):
        self.otype = otype
        self.style= style

    def print_info(self):
        print('The option is an %s %s' % (self.style, self.otype))

import math

def cdf_sn(x):  # return cumulative distribution fucntion of standard normal distribution
    return (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0


class Vanilla(object):

    def __init__(self,otype,spot,vol,rate,strike,maturity): 

        self.otype=otype
        self.spot=spot
        self.vol=vol
        self.rate=rate
        self.strike=strike
        self.maturity=maturity

    def forward(self):
        fwd = math.exp(self.rate * self.maturity) * self.spot
        return fwd

    def print_forward(self):  # method of computing forward price
        print(self.forward())

    def pricing_bs(self): # method of option pricing using Blach-Scholes formula

        S = self.spot
        K = self.strike
        T = self.maturity
        r = self.rate
        sigma = self.vol

        d_1 = (math.log(S/K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))

        d_2 = d_1 - sigma * math.sqrt (T)

        if str.lower(self.otype) == 'call':

            price = cdf_sn(d_1) * S - cdf_sn(d_2) * K * math.exp(-r * T)

        elif str.lower(self.otype) == 'put':

            price = cdf_sn(-d_2) * K * math.exp(-r * T) - cdf_sn(-d_1) * S
        else:

            price = 'option type is not valid'

        return price

import random


def gen_one_path(n_steps,spot,rate_d,rate_f,vol,maturity):
        
        T = maturity
        dt = T / n_steps
        S=[spot]
        rd=rate_d
        rf=rate_f
        sigma = math.sqrt(dt)

        for _ in range(n_steps):

            dS = S[-1]*((rd-rf) * dt + vol * random.gauss(0,sigma))
            S.append(S[-1]+dS)

        return S


class TRF(object):  # Target Redemption Forward class  


    def __init__(self,otype,spot,vol,rate_d,rate_f,strike,maturity,n_steps,*fixings): 

        self.otype=otype
        self.spot=spot
        self.vol=vol
        self.rate_f=rate_f
        self.rate_d=rate_d
        self.strike=strike
        self.maturity=maturity
        self.n_steps=n_steps

    # spot dynamic is dS/S =  (rd-rf) * dt + vol * dw

    
            

    

if __name__ == "__main__":

    S=gen_one_path(100,0.66,0.03,0.02,0.1, 1)

    print(S)
    #option1 = Vanilla('put', 50, 0.3, 0.02, 51.01006700133779, 1)
    #option2 = Vanilla('call', 50, 0.3, 0.02, 51.01006700133779, 1)

    #print(option1)

    #print(option1.pricing_bs()-option2.pricing_bs())
    
    #print(option1.forward())
    #print(option1.forward())

    
    

    





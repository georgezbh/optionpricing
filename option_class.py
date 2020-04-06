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

    

if __name__ == "__main__":


    option1 = Vanilla('put', 50, 0.3, 0.02, 51.01006700133779, 1)
    option2 = Vanilla('call', 50, 0.3, 0.02, 51.01006700133779, 1)

    print(option1)

    print(option1.pricing_bs()-option2.pricing_bs())
    
    #print(option1.forward())
    #print(option1.forward())

    
    

    





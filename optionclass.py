'option class pricing module'

__author__ = 'George Zhao'


import math

from statistics import NormalDist

class Option(object):

    __slots__ = ('_underlying', '_maturity', '_strike')

    def __init__(self, underlying, maturity):

        self._underlying = underlying
        self._maturity = maturity

    def payoff(self,spot):

        payoff_c = max(spot-self._strike,0)

        payoff_p = max(self._strike - spot, 0)

        return [payoff_c, payoff_p]


class Vanilla(Option):
    
    def __init__(self, underlying, maturity, strike, style):

        super().__init__(underlying,maturity)

        self._strike = strike

        self._style = style

    
    @property
    def spot(self):
        return self._spot
    
    @spot.setter
    def spot(self,spot):
        if not isinstance(spot,(float,int)):
            raise ValueError('spot must be a number!')
        if spot <= 0:
            raise ValueError('spot must be greater than 0!')
        self._spot = spot

    @property
    def vol(self):
        return self._vol
    
    @spot.setter
    def vol(self,vol):
        if not isinstance(vol,float):
            raise ValueError('vol must be a float!')
        if vol <= 0:
            raise ValueError('vol must be greater than 0!')
        self._vol = vol
    

    def forward(self,spot,rate_c,rate_a):
        
        fwd=math.exp((rate_c-rate_a)*self._maturity)*spot

        return fwd

    def pricing_bsm(self,spot,vol,rate_c,rate_a,style='E'):

        if style != 'E' and style !='e':
            raise ValueError('Black-Scholes equation can only be used for European vanilla option!')
        
        fwd = self.forward(spot,rate_c,rate_a)

        T = self._maturity
        K = self._strike
        D = math.exp(-rate_c*T)

        N=NormalDist().cdf

        d1 = (math.log(fwd/K)+0.5*vol**2*T)/(vol*math.sqrt(T)) 
        d2 = d1-vol*math.sqrt(T)

        call_price = D*(N(d1)*fwd - N(d2)*K)

        put_price = D*(N(-d2)*K-N(-d1)*fwd)
        
        return [call_price, put_price]
    

def main():

    underlying='spy'
    spot=286.72
    vol=0.3
    T=1
    K = 283.8670883325615

    div_spy=0.02
    rate_usd=0.01 

    spy_vanilla= Vanilla(underlying,T,K,'E')

    print('Intrinsic value of the option is %r' % spy_vanilla.payoff(spot))

    spy_fwd=spy_vanilla.forward(spot,rate_usd,div_spy)
    spy_vanilla_price = spy_vanilla.pricing_bsm(spot,vol,rate_usd,div_spy)
  
    print('%.1f year forward price of %s is %.4f' % (T, underlying,spy_fwd))

    print('The call and put price on %s is %r' % (underlying,spy_vanilla_price))


if __name__ =='__main__':

    main()
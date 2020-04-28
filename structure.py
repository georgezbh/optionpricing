from option import *

class Structure(Option):

    def __init__(self, underlying, asset_class, option_array):

        super().__init__(underlying,asset_class)

        self._underlying = underlying

        self._option_array = option_array

    def pricing(self,spot,vol,rate_c,rate_a):

        options = self._option_array

        mv=0

        for op in options:

            if op._style == 'a':

                model = op.pricing_crr

            else:

                model = op.pricing_bsm

            opvalue = model(spot,vol,rate_c,rate_a)['call']

            print(opvalue)

            mv = mv + opvalue
        
        return mv



def main_structure():

    underlying='spy'
    assetclass='EQD'
    spot=50
    vol=0.3
    T1=0.5
    K1 = 50
    K2 = 55
    rate_usd=0.01
    div=0.03
    q1 = 100
    q2 = -100

    op1 = Vanilla(underlying,assetclass,T1,K1,'a',q1)
    op2 = Vanilla(underlying,assetclass,T1,K2,'a',q2)

    s1 = Structure(underlying,[op1,op2])

    s1_value = s1.pricing(spot,vol,rate_usd,div)

    print(s1_value)


if __name__ =='__main__':

    main_structure()














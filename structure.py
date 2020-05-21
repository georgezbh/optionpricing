from option import *

class Structure(Option):

    def __init__(self, underlying, asset_class, option_array):

        super().__init__(underlying,asset_class)

        self._option_array = option_array


    def pricer(self,spot,vol,rate_c,rate_a,greeks='pl'):

        options = self._option_array

        struc_greeks_value =0

        for op in options:

            model = op._default_model

            if greeks.lower() == 'pl':

                value = model(spot,vol,rate_c,rate_a)
            
            elif greeks.lower() =='delta':

                value = op.delta(spot,vol,rate_c,rate_a)

            elif greeks.lower() =='gamma':

                value = op.gamma(spot,vol,rate_c,rate_a)

            elif greeks.lower()=='vega':

                value = op.vega(spot,vol,rate_c,rate_a)

            elif greeks.lower()=='theta':

                value = op.theta(spot,vol,rate_c,rate_a)

            else:

                value = 0

            struc_greeks_value = struc_greeks_value + value
        
        return struc_greeks_value 

    
    def spot_ladder(self, spot_start, spot_end, spot_step,vol,rate_c,rate_a,greeks):

        spot_rng = np.arange(spot_start,spot_end,spot_step)

        greeks_value =[]

        for s in spot_rng:

            n=(spot_end-spot_start)/spot_step -1
            i=(s-spot_start)/spot_step
            progress= int(i/n*100)

            print('Spot = %f, in progress %d complete' % (s, progress))

            value = self.pricer(s,vol,rate_c,rate_a,greeks)

            greeks_value.append(value)
        
        y = greeks_value
        y_label = greeks.lower()
        y_reg = np.polyfit(spot_rng,y,6)
        y_fit = np.polyval(y_reg,spot_rng)

        plt.figure()
        plt.plot(spot_rng,y,label=y_label)

        plt.xlabel('spot')
        plt.ylabel(y_label)
        plt.legend(loc=4)
        plt.title('Structure') 
        plt.show()



        



def main_structure():

    underlying='spy'
    assetclass='EQD'
    spot=50
    vol=0.3
    T1=1/365
    T2=0.1
    K1 = 48
    K2 = 52
    K3 = 80
    K4 = 30
    rate_usd=0.01
    div=0.02
    q1 = 10000
    q2 = -10000
    q3 = -100
    q4 = -100


    op1 = Vanilla(underlying,assetclass,T1,K1,'e','call',q1)
    op2 = Vanilla(underlying,assetclass,T1,K2,'e','call',q2)
    op3 = Vanilla(underlying,assetclass,T1,K3,'e','call',q3)
    op4 = Vanilla(underlying,assetclass,T1,K4,'e','put',q4)

    s1 = Structure(underlying,assetclass,[op1,op2])



    s1.spot_ladder(1,100,1,vol,rate_usd,div,'theta')



if __name__ =='__main__':

    main_structure()














import numpy as np
from numba import double, int32

spec_band = [
    ('price_0', double),
    ('actual_mean', double),
    ('actual_vol', double),
    ('interest_rate', double),
    ('expire_date', double),
    ('strike_price', double),
    ('transaction_cost', double),
    ('bandwidth', double),
    ('n_paths', int32),
    ('n_steps', int32),
    ('delta_t',double)
]


# base class
class Option_fixed_bandwidth():
    def __init__(self, price_0, actual_mean, actual_vol, interest_rate, expire_date, strike_price, transaction_cost,bandwidth, n_paths,n_steps=250):
        self.price_0 = price_0
        # some papers do not assume that interest rate= drift of the process so i've added this as an argument
        self.actual_mean = actual_mean

        self.actual_vol = actual_vol

        self.interest_rate = interest_rate

        self.expire_date = expire_date

        self.strike_price = strike_price

        self.transaction_cost = transaction_cost

        self.n_paths = n_paths

        self.bandwidth=bandwidth

        self.n_steps = n_steps

        self.delta_t = 1/n_steps

    # methods to be implemented for each specific option type/strategy to be tested
    def delta(self, s, t):
        pass

    def pricing(self, s, t):
        pass

    def exercise(self, s):
        pass

    def replicating_error(self):
        '''
            Some notes about this function:
            -I've assumed continuos interest (some of the papers use anualized instead: one has to do the necessary transformations)
            -I've also written with the conventions that are used to hedge a short option (i.e positive delta, negative payoff at exercise and positive cf at t=0 from the sold option
            one has to do the apropriate changes in those methods (i.e. change the signal of each method) in order to replicate a long option);
        '''

        # hedging error
        error = np.empty(self.n_paths)
        # 1+interest in delta_t time interval
        compound = np.exp(self.interest_rate*self.delta_t)

        for j in range(0, self.n_paths):
            # this values are kept in an array in order to make debbuing easier for now
            path = np.empty(self.n_steps+1)
            # number of stocks
            d = np.empty(self.n_steps)
            #value of rf bank account
            q = np.empty(self.n_steps)

            path[0] = self.price_0
            # quantity of stocks- equal to the delta of the option
            d[0] = self.delta(path[0], 0)
            # rf bank account at t=0- revenue of selling the option
            q[0] = self.pricing(path[0], 0)-d[0]*path[0] - \
                np.abs(d[0])*path[0]*self.transaction_cost

            for i in range(1, self.n_steps):
                # genarate next price using the gbm formula
                path[i] = path[i-1]*np.exp((self.actual_mean-0.5*self.actual_vol**2) *
                                           self.delta_t + self.actual_vol*np.sqrt(self.delta_t)*np.random.normal())

            
                delta=self.delta(path[i],   self.delta_t * i)
                #portfolio's delta
                portfolio_delta=d[i-1]-delta

                #if the portfolio's delta is over the deffined bandwidth, short enough stock for the portfolio delta to get near de deffined bandwidth
                if(portfolio_delta>self.bandwidth):
                    d[i]=delta+self.bandwidth
                #if the portfolio's delta is over the deffined bandwidth, buy enough stock for the portfolio delta to get near de deffined bandwidth
                elif(portfolio_delta<-self.bandwidth): 
                    d[i]=delta-self.bandwidth
                else:
                    d[i]=d[i-1]
                
                # rf bank account- (last value+interest) + cost/revenue of buying stocks + TC of buying stocks
                q[i] = q[i-1]*compound - (d[i]-d[i-1])*path[i] - \
                    np.abs(d[i]-d[i-1])*path[i]*self.transaction_cost

            # genarate final price of the stock
            path[self.n_steps] = path[self.n_steps-1]*np.exp(
                (self.actual_mean-0.5*self.actual_vol**2)*self.delta_t+self.actual_vol*np.sqrt(self.delta_t)*np.random.normal())

            # Any of the papers described how transaction cost are computed at expire date, so I've assumed for now that arent TC at expire
            # hedging error=last value in bank account + interest + last quantity of stocks* new price - payoff of the option
            error[j] = q[self.n_steps-1]*compound+d[self.n_steps-1] * \
                path[self.n_steps] - self.exercise(path[self.n_steps])

        return error


def risk_return_fixed_bandwidth(price_0, actual_mean, actual_vol, interest_rate, expire_date, strike_price, transaction_cost, n_paths,Option_type,risk_measure,*args):
    #function to test the risk return profile of the replicating strategy for different bandwidths
    list = np.empty(40)
    list_risk = np.empty(40)

    for i in range(1, 41):

        bandwidth = float(i/100)

        option = Option_type(
            price_0, actual_mean, actual_vol, interest_rate, expire_date, strike_price, transaction_cost, bandwidth, n_paths)
        
        result = option.replicating_error()


        list[i-1] = np.mean(result)
        list_risk[i-1] = risk_measure(result,*args)
        print(bandwidth)

    return list, list_risk
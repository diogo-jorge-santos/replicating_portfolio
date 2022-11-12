
from .replicating_barrier import Option_barrier_in, spec_barrier
from .misc import normcdf
import numpy as np
from numba.experimental import jitclass
from numba import njit, prange


@jitclass(spec_barrier)
class Down_and_in_put(Option_barrier_in):
    # NOTE: strike price>barrier

    def bs_call(self, s, t, x):
        '''
        European call price using black shoules

        Input:
        s: spot price
        t: time to maturity
        x: stike price

        '''
        d1 = (np.log(s / x) + (self.interest_rate + (self.actual_vol**2) / 2)
              * (self.expire_date - t)) / (self.actual_vol * np.sqrt(self.expire_date - t))

        d2 = d1 - self.actual_vol * np.sqrt(self.expire_date - t)

        c0 = s * normcdf(d1) - x * \
            np.exp(-self.interest_rate * (self.expire_date - t)) * normcdf(d2)

        return c0

    def bs_put(self, s, t, x):
        '''
        European put price using black shoules

        Input:
        s: spot price
        t: time to maturity
        x: stike price

        '''

        d1 = (np.log(s / x) + (self.interest_rate + (self.actual_vol**2) / 2)
              * (self.expire_date - t)) / (self.actual_vol * np.sqrt(self.expire_date - t))

        d2 = d1 - self.actual_vol * np.sqrt(self.expire_date - t)

        p0 = x * np.exp(-self.interest_rate * (self.expire_date - t)) * \
            normcdf(-d2) - s * normcdf(-d1)

        return p0

    def d2(self, s, t, x):
        '''
        d2's value from european call price using black shoules

        s: spot price
        t: time to maturity
        x: stike price
        '''
        d1 = (np.log(s / x) + (self.interest_rate + (self.actual_vol**2) / 2) *
              (self.expire_date - t)) / (self.actual_vol * np.sqrt(self.expire_date - t))

        d2 = d1 - self.actual_vol * np.sqrt(self.expire_date - t)

        return d2

    def pricing(self, s, t, barrier_in):
        '''
        Down_and_in_put price, under the assumptions of Merton (1973)
        s: spot price
        t: time to maturity

        ref: pdf
        '''
        if barrier_in or s < self.barrier_price:
            return self.bs_put(s, t, self.strike_price)
        else:
            mu = self.interest_rate - 0.5 * self.actual_vol**2

            price_1 = ((self.barrier_price / s)**(2 * mu / (self.actual_vol**2))) * (self.bs_call((self.barrier_price**2) / s, t, self.strike_price) -
                                                                                     self.bs_call((self.barrier_price**2) / s, t, self.barrier_price) + (self.strike_price - self.barrier_price) *
                                                                                     np.exp(-self.interest_rate * (self.expire_date - t)) * normcdf(self.d2(self.barrier_price, t, s)))

            price_2 = self.bs_put(s, t, self.barrier_price) + (self.strike_price - self.barrier_price) *\
                np.exp(-self.interest_rate * (self.expire_date - t)) * \
                normcdf(-self.d2(s, t, self.barrier_price))

            return price_1 + price_2

    def delta(self, s, t, barrier_in):
        '''
        Delta of the option using finite diference
        s: spot price
        t: time to maturity
        '''
        h = 10**-8

        return (self.pricing(s + h, t, barrier_in) -
                self.pricing(s - h, t, barrier_in)) / (2 * h)

    def delta_t_t(self,s,barrier_in):

        if self.exercise(s,barrier_in)>0.0:
            return -1
        else:
            return 0

    def gamma(self, s, t, barrier_in):
        '''
        Gamma of the option using finite diference
        s: spot price
        t: time to maturity
        '''
        h = 10**-4
        # to remove discontinuity
        if s > self.barrier_price:
            return (self.pricing(s + 2 * h, t, barrier_in) + self.pricing(s,
                    t, barrier_in) - 2 * self.pricing(s + h, t, barrier_in)) / (h**2)
        else:
            return (self.pricing(s, t, barrier_in) + self.pricing(s - 2 * h,
                    t, barrier_in) - 2 * self.pricing(s - h, t, barrier_in)) / (h**2)

    def barrier(self, s):
        '''
        Function to check if spot is past the barrier
        s: spot price
        '''
        return s < self.barrier_price

    def exercise(self, s, barrier_in):
        '''
        Function to check the price of the option at expire
        s: spot price
        barrier_in: if the spot price has past the barrier
        '''
        if barrier_in or s < self.barrier_price:
            return max(self.strike_price - s, 0)
        else:
            return 0.0


@njit(parallel=True)
def risk_return_barrier_put(
        price_0,
        actual_mean,
        actual_vol,
        interest_rate,
        expire_date,
        strike_price,
        barrier_price,
        transaction_cost,
):
    '''
    Funtion to obtain in parallel the hedging cost and slippage of
    hedging the option with N=1,...,200 hedging intervals for a Down_and_in_put

    Input:
    price_0: spot price at t=0
    actual_mean: spot price mean
    actual_vol: volatility of spot price
    interest_rate: continuous compounded interet rate
    expire_date
    strike_price
    barrier_price
    transaction_cost: half of the relative bid-offer spread in percentage of underlying price

    Output:
    list,list_risk: mean and standard deviation of the Pnl of hedging the option with N=1,...,201
    list_cost, list_cost_risk: mean and standard deviation of hedging cost  with with N=1,...,201
    list_slipage, list_slipage_risk: mean and standard deviation of hedging slippage  with with N=1,...,201

    '''

    list = np.zeros(200)
    list_risk = np.zeros(200)

    list_cost = np.zeros(200)
    list_cost_risk = np.zeros(200)

    list_slipage = np.zeros(200)
    list_slipage_risk = np.zeros(200)

    for i in prange(200):

        delta_t = int(i + 1)
        print(delta_t)

        option = Down_and_in_put(
            price_0,
            actual_mean,
            actual_vol,
            interest_rate,
            expire_date,
            strike_price,
            barrier_price,
            transaction_cost,
            10001,
            delta_t,
            100000)

        result_cost, result_slipage, barrier_in_arr = option.replicating_error()
        list[i] = (np.mean(result_cost + result_slipage))
        list_risk[i] = (np.std(result_cost + result_slipage))

        list_cost[i] = (np.mean(result_cost))
        list_cost_risk[i] = (np.std(result_cost))

        list_slipage[i] = (np.mean(result_slipage))
        list_slipage_risk[i] = (np.std(result_slipage))

    return list, list_risk, list_cost, list_cost_risk, list_slipage, list_slipage_risk


@njit(parallel=True)
def gamma_test_barrier_put(

        price_0,
        actual_mean,
        actual_vol,
        interest_rate,
        expire_date,
        strike_price,
        barrier_price,
        transaction_cost,
):
    '''
    Funtion to obtain in parallel absolute sum of gamma and the square sum of gammas
    to compute the closed form solutions over with N=1,...,200 hedging intervals for a Down_and_in_put


    Input:
    price_0: spot price at t=0
    actual_mean: spot price mean
    actual_vol: volatility of spot price
    interest_rate: continuous compounded interet rate
    expire_date
    strike_price
    barrier_price
    transaction_cost: half of the relative bid-offer spread in percentage of underlying price

    Output:
    list_abs: array of the sum of absolute gammas with N=1,...,201
    list_square: array of the sum of the square of gammas with N=1,...,201

    '''

    list_abs = np.zeros(200)
    list_square = np.zeros(200)

    for i in prange(200):

        delta_t = int(i + 1)

        option = Down_and_in_put(
            price_0,
            actual_mean,
            actual_vol,
            interest_rate,
            expire_date,
            strike_price,
            barrier_price,
            transaction_cost,
            10001,
            delta_t,
            100000)

        result_abs, result_square = option.test_gamma()

        list_abs[i] = result_abs
        list_square[i] = result_square

    return list_abs, list_square


'''

import matplotlib.pyplot as plt

option_call=Down_and_in_put(100,0.05,0.25,0.05,1,100,70,0.002,1000,100,1000)

c , s =option_call.replicating_error()
plt.hist(c+s)
plt.show()


option_put=Down_and_in_put(100,0.05,0.25,0.05,1,100,70,0.00,1000,17,200000)

cost_mean,cost_std,slippage_mean,slippage_std=risk_return_barrier_call(100,0.05,0.25,0.05,1,100,200,0.002)

plt.plot(cost_mean)
plt.show()

plt.plot(cost_std)
plt.show()

plt.plot(slippage_mean)
plt.show()

plt.plot(slippage_std)
plt.show()
'''
'''
option_put=Down_and_in_put(100,0.05,0.25,0.05,1,100,70,0.002,1000,100,1000)


cost,slippage=option_put.replicating_error()
plt.hist(cost)
plt.show()
print(np.mean(cost))
print(np.std(cost))
plt.hist(slippage)
plt.show()
print(np.mean(slippage))
print(np.std(slippage))
'''

'''
cost,slippage=option_put.replicating_error()

print(np.mean(slippage))
print(np.std(slippage))
plt.hist(slippage)
plt.show()
'''
'''

h=0.0001

print(option_call.pricing(100,0))
print(option_call.bs_call(100,0,100))

print((option_call.bs_call(100+h,0,100)-option_call.bs_call(100-h,0,100))/(2*h))
print(option_call.delta(100,0))
option_put=Down_and_in_put(100,0.05,0.25,0.05,1,100,99.99999999,0.0,1000,1,2)
h=0.0001

print(option_put.pricing(100,0,False))
print(option_put.bs_put(100,0,100))

print((option_put.bs_put(100+h,0,100)-option_put.bs_put(100-h,0,100))/(2*h))
print(option_put.delta(100,0,False))



import matplotlib.pyplot as plt
option_call=Up_and_out_call(100,0.05,0.25,0.05,1,100,150,0.0,10000,10000,1000)




cost,slippage=option_call.replicating_error()
print(np.std(slippage))
print(np.mean(slippage))
plt.hist(slippage)
plt.show()


from class_seperate import Short_european_call_black_sholes_fixed_time
option_call=Short_european_call_black_sholes_fixed_time(100,0.05,0.20,0.05,1,100,0.0,10000,10000)
cost,slippage=option_call.replicating_error()
print(np.std(slippage))
print(np.mean(slippage))
plt.hist(slippage)
plt.show()
'''

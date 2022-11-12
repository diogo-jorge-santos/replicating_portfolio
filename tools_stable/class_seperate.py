from .replicating_seperate import Option_fixed_time_sep, spec_time
from .misc import normcdf, normpdf
from numba.experimental import jitclass
from numba import njit, prange
import numpy as np


@jitclass(spec_time)
class Short_european_call_black_sholes_fixed_time_sep(Option_fixed_time_sep):

    def delta(self, s, t):
        '''
        Delta of the option using the closed form solution
        s: spot price
        t: time to maturity
        '''
        d1 = (np.log(s / self.strike_price) + (self.interest_rate + (self.actual_vol**2) / 2)
              * (self.expire_date - t)) / (self.actual_vol * np.sqrt(self.expire_date - t))

        return normcdf(d1)
    def delta_t_t(self,s):

        if self.exercise(s)>0.0:
            return 1
        else:
            return 0

    def pricing(self, s, t):
        '''
        Up_and_out_call price, under the assumptions of Merton (1973)
        s: spot price
        t: time to maturity
        '''
        d1 = (np.log(s / self.strike_price) + (self.interest_rate + (self.actual_vol**2) / 2)
              * (self.expire_date - t)) / (self.actual_vol * np.sqrt(self.expire_date - t))

        d2 = d1 - self.actual_vol * np.sqrt(self.expire_date - t)

        c0 = s * normcdf(d1) - self.strike_price * \
            np.exp(-self.interest_rate * (self.expire_date - t)) * normcdf(d2)

        return c0

    def exercise(self, s):
        '''
        Function to check the price of the option at expire
        s: spot price
        '''
        return max(s - self.strike_price, 0)

    def gamma(self, s, t):
        '''
        Gamma of the option using closed form solution
        s: spot price
        t: time to maturity
        '''
        d1 = (np.log(s / self.strike_price) + (self.interest_rate + (self.actual_vol**2) / 2)
              * (self.expire_date - t)) / (self.actual_vol * np.sqrt(self.expire_date - t))

        return normpdf(d1) / (s * self.actual_vol *
                              np.sqrt(self.expire_date - t))


@njit(parallel=True)
def risk_return_fixed_time_sep(
        price_0,
        actual_mean,
        actual_vol,
        interest_rate,
        expire_date,
        strike_price,
        transaction_cost,
        n_paths,
):
    '''
    Funtion to obtain in parallel the hedging cost and slippage of
    hedging the option with N=1,...,200 hedging intervals for a european call

    Input:
    price_0: spot price at t=0
    actual_mean: spot price mean
    actual_vol: volatility of spot price
    interest_rate: continuous compounded interet rate
    expire_date
    strike_price
    transaction_cost: half of the relative bid-offer spread in percentage of underlying price
    n_paths: number of paths to simulate

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

        n_steps = int(i + 1)

        option = Short_european_call_black_sholes_fixed_time_sep(
            price_0,
            actual_mean,
            actual_vol,
            interest_rate,
            expire_date,
            strike_price,
            transaction_cost,
            n_steps,
            n_paths)

        result_cost, result_slipage = option.replicating_error()

        list[i] = (np.mean(result_cost + result_slipage))
        list_risk[i] = (np.std(result_cost + result_slipage))

        list_cost[i] = (np.mean(result_cost))
        list_cost_risk[i] = (np.std(result_cost))

        list_slipage[i] = (np.mean(result_slipage))
        list_slipage_risk[i] = (np.std(result_slipage))

    return list, list_risk, list_cost, list_cost_risk, list_slipage, list_slipage_risk


@njit(parallel=True)
def gamma_test_fixed_time(
        price_0,
        actual_mean,
        actual_vol,
        interest_rate,
        expire_date,
        strike_price,
        transaction_cost,
        n_paths,
):
    '''
    Function to obtain in parallel absolute sum of gamma and the square sum of gammas
    to compute the closed form solutions over with N=1,...,200 hedging intervals for a european call


    Input:
    price_0: spot price at t=0
    actual_mean: spot price mean
    actual_vol: volatility of spot price
    interest_rate: continuous compounded interet rate
    expire_date
    strike_price
    barrier_price
    transaction_cost: half of the relative bid-offer spread in percentage of underlying price
    n_paths: number of paths to simulate

    Output:
    list_abs: array of the sum of absolute gammas with N=1,...,201
    list_square: array of the sum of the square of gammas with N=1,...,201

    '''
    # function to test the risk return profile of the replicating strategy for
    # different time intervals
    list_abs = np.zeros(200)
    list_square = np.zeros(200)

    for i in prange(200):

        n_steps = int(i + 1)

        option = Short_european_call_black_sholes_fixed_time_sep(
            price_0,
            actual_mean,
            actual_vol,
            interest_rate,
            expire_date,
            strike_price,
            transaction_cost,
            n_steps,
            n_paths)

        result_abs, result_square = option.test_gamma()

        list_abs[i] = result_abs
        list_square[i] = result_square

    return list_abs, list_square

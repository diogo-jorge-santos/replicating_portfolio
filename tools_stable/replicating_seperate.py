import numpy as np
from numba import double, int32
import matplotlib.pyplot as plt
spec_time = [
    ('price_0', double),
    ('actual_mean', double),
    ('actual_vol', double),
    ('interest_rate', double),
    ('expire_date', double),
    ('strike_price', double),
    ('transaction_cost', double),
    ('n_steps', int32),
    ('n_paths', int32),
    ('delta_t', double),
]


# base class
class Option_fixed_time_sep():

    def __init__(
            self,
            price_0,
            actual_mean,
            actual_vol,
            interest_rate,
            expire_date,
            strike_price,
            transaction_cost,
            n_steps,
            n_paths):
        '''
        Base class to test european style options

        Inputs:
        price_0: spot price at t=0
        actual_mean: spot price mean
        actual_vol: volatility of spot price
        interest_rate: continuous compounded interet rate
        expire_date
        strike_price
        transaction_cost: half of the relative bid-offer spread in percentage of underlying price
        n_steps: number of hedging events+1
        n_paths: number of sample paths to test
        '''
        self.price_0 = price_0

        self.actual_mean = actual_mean

        self.actual_vol = actual_vol

        self.interest_rate = interest_rate

        self.expire_date = expire_date

        self.strike_price = strike_price

        self.transaction_cost = transaction_cost

        self.n_steps = n_steps

        self.n_paths = n_paths

        self.delta_t = expire_date / n_steps

    # methods to be implemented for each specific option type/strategy to be
    # tested
    def delta(self, s, t):
        pass

    def delta_t_t(self,s):
        pass

    def pricing(self, s, t):
        pass

    def exercise(self, s):
        pass

    def gamma(self, s, t):
        pass



    def avg_cash_gamma(self):

        ladder = np.array([60, 80, 90, 95, 100, 105, 110,
                          120, 140]) * self.strike_price / 100
        sum = 0.0
        for i in range(len(ladder)):
            sum += np.abs(self.gamma(ladder[i], 0))
        sum = sum / len(ladder) * self.price_0**2

        return sum

    def replicating_error(self):
        '''
        Function used to estimate the distribution of the hedging cost and slippage of a deffined option

        Output:
        error_cost: array of hedging cost for each generated sample path
        error_slipage:array of hedging slippage for each generated sample path

        '''

        # hedging error
        error_slipage = np.empty(self.n_paths)
        error_cost = np.empty(self.n_paths)

        # 1+interest in delta_t time interval
        compound = np.exp(self.interest_rate * self.delta_t)

        path = np.empty(self.n_steps + 1)
        d = np.empty(self.n_steps)

        non_tc = np.empty(self.n_steps)
        tc = np.empty(self.n_steps)

        path[0] = self.price_0
        # quantity of stocks- equal to the delta of the option
        d[0] = self.delta(path[0], 0)
        # rf bank account at t=0- revenue of selling the option

        non_tc[0] = self.pricing(path[0], 0) - d[0] * path[0]

        # assuming no tc at t=0
        tc[0] = 0

        for j in range(0, self.n_paths):
            # this values are kept in an array in order to make debbuing easier
            # for now

            for i in range(1, self.n_steps):
                # genarate next price using the gbm formula
                path[i] = path[i - 1] * np.exp((self.actual_mean - 0.5 * self.actual_vol**2) *
                                               self.delta_t + self.actual_vol * np.sqrt(self.delta_t) * np.random.normal())
                # quantity of stocks- equal to the delta of the option
                d[i] = self.delta(path[i], self.delta_t * i)
                # rf bank account- (last value+interest) + cost/revenue of
                # buying stocks + TC of buying stocks

                tc[i] = tc[i - 1] * compound - \
                    np.abs(d[i] - d[i - 1]) * path[i] * self.transaction_cost

                non_tc[i] = non_tc[i - 1] * compound - \
                    (d[i] - d[i - 1]) * path[i]

            # genarate final price of the stock
            path[self.n_steps] = path[self.n_steps - 1] * np.exp(
                (self.actual_mean - 0.5 * self.actual_vol**2) * self.delta_t + self.actual_vol * np.sqrt(self.delta_t) * np.random.normal())

            # Any of the papers described how transaction cost are computed at expire date, so I've assumed for now that arent TC at expire
            # hedging error=last value in bank account + interest + last
            # quantity of stocks* new price - payoff of the option

            error_slipage[j] = (non_tc[self.n_steps - 1] * compound + d[self.n_steps - 1] * path[self.n_steps] -
                                self.exercise(path[self.n_steps])) * np.exp(-self.interest_rate * self.expire_date)

            error_cost[j] = (tc[self.n_steps - 1] * compound - np.abs(self.delta_t_t(path[self.n_steps]) - d[self.n_steps - 1]) * path[self.n_steps] * self.transaction_cost) * \
                np.exp(-self.interest_rate * self.expire_date)

        return error_cost, error_slipage

    def test_gamma(self):
        '''
        Funtion used to estimate the sum of gammas used in the closed form solutions

        output:
        np.sum(expected_abs)- sum of absolute gammas
        np.sum(expected_power)-sum of squared gammas


        '''

        expected_abs = np.empty(self.n_steps)
        expected_power = np.empty(self.n_steps)

        # t=0 in the provided formulas
        path = np.ones(self.n_paths) * self.price_0
        gamma_array = np.empty(self.n_paths)
        for j in range(0, self.n_paths):
            gamma_array[j] = self.gamma(
                path[j], (0) * self.delta_t) * (path[j]**2)

        expected_abs[0] = np.mean(
            np.abs(np.exp(-self.interest_rate * self.delta_t * (0)) * gamma_array))

        expected_power[0] = np.mean(
            (np.exp(-self.interest_rate * self.delta_t * (0)) * gamma_array)**2)

        for i in range(1, self.n_steps):
            for j in range(0, self.n_paths):
                path[j] = path[j] * np.exp((self.actual_mean - 0.5 * self.actual_vol**2) *
                                           self.delta_t + self.actual_vol * np.sqrt(self.delta_t) * np.random.normal())

                gamma_array[j] = self.gamma(
                    path[j], (i) * self.delta_t) * (path[j]**2)

            expected_abs[i] = np.mean(
                np.abs(np.exp(-self.interest_rate * self.delta_t * (i)) * gamma_array))

            expected_power[i] = np.mean(
                (np.exp(-self.interest_rate * self.delta_t * (i)) * gamma_array)**2)

        return np.mean(expected_abs), np.mean(expected_power)

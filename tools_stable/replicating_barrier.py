
import numpy as np
from numba import double, int32
import matplotlib.pyplot as plt
spec_barrier = [
    ('price_0', double),
    ('actual_mean', double),
    ('actual_vol', double),
    ('interest_rate', double),
    ('expire_date', double),
    ('strike_price', double),
    ('transaction_cost', double),
    ('barrier_price', double),
    ('min_euler', int32),
    ('n_steps', int32),
    ('n_paths', int32),
    ('delta_t', double),
    ('n_steps_euler', int32),
    ('euler_dt', double),
    ('factor', int32)
]


class Option_barrier_in():
    def __init__(
            self,
            price_0,
            actual_mean,
            actual_vol,
            interest_rate,
            expire_date,
            strike_price,
            barrier_price,
            transaction_cost,
            min_euler,
            n_steps,
            n_paths,
    ):
        '''
        Base class to test barrier options

        Inputs:
        price_0: spot price at t=0
        actual_mean: spot price mean
        actual_vol: volatility of spot price
        interest_rate: continuous compounded interet rate
        expire_date
        strike_price
        barrier_price
        transaction_cost: half of the relative bid-offer spread in percentage of underlying price
        min_euler: minimum of instances to check the barrier condition
        n_steps: number of hedging events
        n_paths: number of sample paths to test
        '''
        self.price_0 = price_0

        self.actual_mean = actual_mean

        self.actual_vol = actual_vol

        self.interest_rate = interest_rate

        self.expire_date = expire_date

        self.strike_price = strike_price

        self.barrier_price = barrier_price

        self.transaction_cost = transaction_cost

        self.n_steps = n_steps

        self.n_paths = n_paths

        self.delta_t = expire_date / n_steps

        self.min_euler = min_euler

        self.min_euler = min_euler

        # number of "barrier checks" between hedging events (t=0 inclusive)
        self.factor = 1
        # number of hedging events
        self.n_steps_euler = self.n_steps

        while True:
            if(self.n_steps_euler > self.min_euler * self.expire_date):
                break
            else:
                self.n_steps_euler = self.n_steps_euler + self.n_steps
                self.factor = self.factor + 1

        self.euler_dt = self.expire_date / self.n_steps_euler

    def generate_sample_path(self, s_0):

        path = np.zeros(self.n_steps_euler + 1)
        path[0] = s_0

        for i in range(1, self.n_steps_euler + 1):

            path[i] = path[i - 1] * np.exp((self.actual_mean - 0.5 * self.actual_vol**2)
                                           * self.euler_dt + self.actual_vol * np.sqrt(self.euler_dt) * np.random.normal())

        return path

    # methods to be implemented for each specific option type/strategy to be
    # tested
    def pricing(self, s, t, barrier_in):
        pass

    def delta(self, s, t, barrier_in):
        pass
    
    def delta_t_t(self,s,barrier_in):
        pass

    def barrier(self, s):
        pass

    def exercise(self, s, barrier_in):
        pass

    def gamma(self, s, t, barrier_in):
        pass

    def avg_cash_gamma(self):

        ladder = np.array([60, 80, 90, 95, 100, 105, 110,
                          120, 140]) * self.strike_price / 100
        sum = 0.0
        for i in range(len(ladder)):
            sum += np.abs(self.gamma(ladder[i], 0, self.barrier(ladder[i])))
        sum = sum / len(ladder) * self.price_0**2

        return sum

    def replicating_error(self):
        '''
        Function used to estimate the distribution of the hedging cost and slippage of a deffined option

        Output:
        error_cost: array of hedging cost for each generated sample path
        error_slipage:array of hedging slippage for each generated sample path
        barrier_in_arr:array that indicates if the sample path got over the barrier


        '''

        # hedging error
        error_slipage = np.zeros(self.n_paths)
        error_cost = np.zeros(self.n_paths)
        barrier_in_arr = np.full(self.n_paths, False)

        # 1+interest in delta_t time interval
        compound = np.exp(self.interest_rate * self.delta_t)

        

        barrier = self.barrier
        delta = self.delta
        exercise = self.exercise

        for j in range(0, self.n_paths):
            d = np.zeros(self.n_steps)

            non_tc = np.zeros(self.n_steps)
            tc = np.zeros(self.n_steps)

            d[0] = self.delta(self.price_0, 0, False)
            non_tc[0] = self.pricing(self.price_0, 0, False) - d[0] * self.price_0
            tc[0] = 0
            # this values are kept in an array in order to make debbuing easier
            # for now
            path = self.generate_sample_path(self.price_0)
            barrier_in = False
            i_index = 0
            for i in range(1, self.n_steps_euler):

                if(barrier(path[i])):
                    # if barrier is passed delta and exercise function will
                    # change to replicate Black-shoules
                    barrier_in = True
                    barrier_in_arr[j] = True
                if(i % self.factor == 0):
                    i_index = i_index + 1

                    # quantity of stocks- equal to the delta of the option
                    d[i_index] = delta(
                        path[i], self.delta_t * i_index, barrier_in)
                    # rf bank account- (last value+interest) + cost/revenue of
                    # buying stocks + TC of buying stocks

                    tc[i_index] = tc[i_index - 1] * compound - \
                        np.abs(d[i_index] - d[i_index - 1]) * \
                        path[i] * self.transaction_cost

                    non_tc[i_index] = non_tc[i_index - 1] * compound - \
                        (d[i_index] - d[i_index - 1]) * path[i]

            if barrier(path[self.n_steps_euler]):
                barrier_in = True
                barrier_in_arr[j] = True

            # hedging error=last value in bank account + interest + last
            # quantity of stocks* new price - payoff of the option

            error_slipage[j] = (non_tc[self.n_steps - 1] * compound + d[self.n_steps - 1] * path[self.n_steps_euler] -
                                exercise(path[self.n_steps_euler], barrier_in)) * np.exp(-self.interest_rate * self.expire_date)

            error_cost[j] = (tc[self.n_steps - 1] * compound - np.abs(self.delta_t_t(path[self.n_steps_euler],barrier_in) - d[self.n_steps - 1]) * path[self.n_steps_euler] * self.transaction_cost) * \
                np.exp(-self.interest_rate * self.expire_date)
        return error_cost, error_slipage, barrier_in_arr

    def test_gamma(self):
        '''
        Funtion used to estimate the sum of gammas used in the closed form solutions

        output:
        np.sum(expected_abs)- sum of absolute gammas
        np.sum(expected_power)-sum of squared gammas
        '''

        expected_abs = np.zeros(self.n_steps)
        expected_power = np.zeros(self.n_steps)
        gamma_array = np.zeros((self.n_paths, self.n_steps))
        # t=0 in the provided formulas
        
        gamma = self.gamma
        barrier = self.barrier
        for j in range(0, self.n_paths):
            path = self.generate_sample_path(self.price_0)
            barrier_in = False
            i_index = 0
            for i in range(0, self.n_steps_euler):
                if(barrier(path[i])):
                    # if barrier is passed- start using BS gammas
                    barrier_in = True
                if(i % self.factor == 0):
        

                    gamma_array[j, i_index] = gamma(
                        path[i], self.delta_t * i_index, barrier_in) * path[i]**2
                    i_index = i_index + 1
        for i in range(0, self.n_steps):

            expected_abs[i] = np.mean(
                np.abs(np.exp(-self.interest_rate * self.delta_t * (i+1)) * gamma_array[:, i]))

            expected_power[i] = np.mean(
                (np.exp(-self.interest_rate * self.delta_t * (i+1)) * gamma_array[:, i])**2)

        return np.mean(expected_abs), np.mean(expected_power)

    def test_gamma_aux(self):
        '''
        Funtion used to estimate the sum of gammas used in the closed form solutions

        output:
        np.sum(expected_abs)- sum of absolute gammas
        np.sum(expected_power)-sum of squared gammas
        '''

        expected_abs = np.zeros(self.n_steps)
        expected_power = np.zeros(self.n_steps)
        sd_abs = np.zeros(self.n_steps)
        sd_power = np.zeros(self.n_steps)
        gamma_array = np.zeros((self.n_paths, self.n_steps))
        # t=0 in the provided formulas
        
        gamma = self.gamma
        barrier = self.barrier
        for j in range(0, self.n_paths):
            path = self.generate_sample_path(self.price_0)
            barrier_in = False
            i_index = 0
            for i in range(0, self.n_steps_euler):
                if(barrier(path[i])):
                    # if barrier is passed- start using BS gammas
                    barrier_in = True
                if(i % self.factor == 0):
        

                    gamma_array[j, i_index] = gamma(
                        path[i], self.delta_t * i_index, barrier_in) * path[i]**2
                    i_index = i_index + 1
        for i in range(0, self.n_steps):

            expected_abs[i] = np.mean(
                np.abs(np.exp(-self.interest_rate * self.delta_t * i) * gamma_array[:, i]))
            
            sd_abs[i] = np.std(
                np.abs(np.exp(-self.interest_rate * self.delta_t * i) * gamma_array[:, i]))

            expected_power[i] = np.mean(
                (np.exp(-self.interest_rate * self.delta_t * i) * gamma_array[:, i])**2)

            sd_power[i] = np.std(
                (np.exp(-self.interest_rate * self.delta_t * i) * gamma_array[:, i])**2)

        return expected_abs,expected_power,sd_abs,sd_power
'''
def gamma(self, s, t):
        pass

    def test_gamma(self):
        pass
'''

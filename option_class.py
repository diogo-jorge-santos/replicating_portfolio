import numpy as np
from numba.experimental import jitclass
from numba import njit, prange, double
from tools_stable.misc import normcdf
from tools_stable.replicating_fixed_time import Option_fixed_time, spec

# hedging of a short european call, using black-scholes strategy and price
@jitclass(spec)
class Short_european_call_black_sholes(Option_fixed_time):

    def delta(self, s, t):
        d1 = (np.log(s/self.strike_price) + (self.interest_rate + (self.actual_vol**2)/2)
              * (self.expire_date-t)) / (self.actual_vol*np.sqrt(self.expire_date-t))

        return normcdf(d1)

    def pricing(self, s, t):
        d1 = (np.log(s/self.strike_price) + (self.interest_rate + (self.actual_vol**2)/2)
              * (self.expire_date-t)) / (self.actual_vol*np.sqrt(self.expire_date-t))

        d2 = d1-self.actual_vol*np.sqrt(self.expire_date-t)

        c0 = s*normcdf(d1)-self.strike_price * \
            np.exp(-self.interest_rate*(self.expire_date-t))*normcdf(d2)

        return c0

    def exercise(self, s):
        return max(s-self.strike_price, 0)


spec1 = [('modified_vol', double)]

# hedging of a short european call, using black-scholes price and leland strategy
@jitclass(spec+spec1)
class Short_european_call_leland(Option_fixed_time):

    __init__A = Option_fixed_time.__init__

    def __init__(self, price_0, actual_mean, actual_vol, interest_rate, expire_date, strike_price, transaction_cost, n_steps, n_paths):

        self.__init__A(price_0, actual_mean, actual_vol, interest_rate,
                       expire_date, strike_price, transaction_cost, n_steps, n_paths)
        # its needed to add the modified volatility to the class objects
        self.modified_vol = self.actual_vol * \
            np.sqrt(1 + (self.transaction_cost/self.actual_vol)
                    * np.sqrt(8 / (self.delta_t*np.pi)))

    def delta(self, s, t):
        d1 = (np.log(s/self.strike_price) + (self.interest_rate + (self.modified_vol**2)/2)
              * (self.expire_date-t)) / (self.modified_vol*np.sqrt(self.expire_date-t))

        return normcdf(d1)

    def pricing(self, s, t):
        d1 = (np.log(s/self.strike_price) + (self.interest_rate + (self.actual_vol**2)/2)
              * (self.expire_date-t)) / (self.actual_vol*np.sqrt(self.expire_date-t))

        d2 = d1-self.actual_vol*np.sqrt(self.expire_date-t)

        c0 = s*normcdf(d1)-self.strike_price * \
            np.exp(-self.interest_rate*(self.expire_date-t))*normcdf(d2)

        return c0

    def exercise(self, s):
        return max(s-self.strike_price, 0)


@njit(parallel=True)
def risk_return_parallel(price_0, actual_mean, actual_vol, interest_rate, expire_date, strike_price, transaction_cost, n_paths):
    list = np.empty(40)
    list_sd = np.empty(40)

    list_1 = np.empty(40)
    list_1_sd = np.empty(40)

    for i in prange(1, 41):

        delta_t = int(i*5)

        option = Short_european_call_black_sholes(
            price_0, actual_mean, actual_vol, interest_rate, expire_date, strike_price, transaction_cost, delta_t, n_paths)
        option1 = Short_european_call_leland(
            price_0, actual_mean, actual_vol, interest_rate, expire_date, strike_price, transaction_cost, delta_t, n_paths)

        result = option.replicating_error()

        result1 = option1.replicating_error()

        list[i-1] = (np.mean(result))
        list_sd[i-1] = (np.std(result))
        list_1[i-1] = (np.mean(result1))
        list_1_sd[i-1] = (np.std(result1))
        print(delta_t)

    return list, list_sd, list_1, list_1_sd


def risk_return_stable(price_0, actual_mean, actual_vol, interest_rate, expire_date, strike_price, transaction_cost, n_paths):
    list = np.empty(40)
    list_sd = np.empty(40)

    list_1 = np.empty(40)
    list_1_sd = np.empty(40)

    for i in prange(1, 41):

        delta_t = int(i*5)

        option = Short_european_call_black_sholes(
            price_0, actual_mean, actual_vol, interest_rate, expire_date, strike_price, transaction_cost, delta_t, n_paths)
        option1 = Short_european_call_leland(
            price_0, actual_mean, actual_vol, interest_rate, expire_date, strike_price, transaction_cost, delta_t, n_paths)

        result = option.replicating_error()

        result1 = option1.replicating_error()

        list[i-1] = (np.mean(result))
        list_sd[i-1] = (np.std(result))
        list_1[i-1] = (np.mean(result1))
        list_1_sd[i-1] = (np.std(result1))
        print(delta_t)

    return list, list_sd, list_1, list_1_sd

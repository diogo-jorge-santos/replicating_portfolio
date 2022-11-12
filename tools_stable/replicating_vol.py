
import matplotlib.pyplot as plt
import numpy as np
from numba import double, int32

spec_vol = [
    ('price_spot', double),
    ('price_vol', double),
    ('actual_mean', double),
    ('interest_rate', double),
    ('expire_date', double),
    ('strike_price', double),
    ('transaction_cost_1', double),
    ('transaction_cost_2', double),
    ('euler', int32),
    ('delta_euler', double),
    ('n_spot', int32),
    ('delta_spot', double),
    ('factor_spot', int32),
    ('n_vol', int32),
    ('delta_vol', double),
    ('factor_vol', int32),
    ('n_paths', int32)

]


# base class
class Option_vol():

    def __init__(
            self,
            price_spot,
            price_vol,
            actual_mean,
            interest_rate,
            expire_date,
            strike_price,
            transaction_cost_1,
            transaction_cost_2,
            euler,
            n_spot,
            n_vol,
            n_paths):
        '''
        Base class to test european options with stocastic volatlity

        Inputs:
        price_0: spot price at t=0
        actual_mean: spot price mean
        actual_vol: volatility of spot price
        interest_rate: continuous compounded interet rate
        expire_date
        strike_price
        barrier_price
        transaction_cost_1: half of the relative bid-offer spread in percentage of underlying price
        transaction_cost_2,
        euler: size of sample path to be computed
        n_spot: number of hedging events to hedge delta + 1
        n_vol: number of hedging events to hedge vega + 1
        n_paths: number of sample paths to test
        '''

        self.price_spot = price_spot

        self.price_vol = price_vol
        # some papers do not assume that interest rate= drift of the process so
        # i've added this as an argument
        self.actual_mean = actual_mean

        self.interest_rate = interest_rate

        self.expire_date = expire_date

        self.strike_price = strike_price

        self.transaction_cost_1 = transaction_cost_1

        self.transaction_cost_2 = transaction_cost_2

        self.euler = euler
        self.delta_euler = self.expire_date / euler

        # NOTE: n hedging events-> n_steps=n+1
        self.n_spot = n_spot
        self.delta_spot = self.expire_date / n_spot
        # sample path must be divisable by number of hedging events
        assert(euler % n_spot == 0)
        self.factor_spot = round(euler / n_spot)

        self.n_vol = n_vol
        self.delta_vol = self.expire_date / n_vol
        assert(euler % n_vol == 0)
        self.factor_vol = round(euler / n_vol)

        self.n_paths = n_paths

    # methods to be implemented for each specific option type/strategy to be
    # tested
    def generate_sample_path(self, s_0, sigma_0):
        pass

    def estimate(self):
        pass

    def exercise(self, s):
        pass

    def pricing(self, s, sigma, t):
        pass

    def delta(self, s, sigma, t):
        pass
    def delta_t_t(self,s):
        pass

    def vega(self, s, sigma, t):
        pass

    def gamma(self, s, sigma, t):
        '''
         gamma of the option using finite difference
        s: spot price
        sigma: volatility
        t: time to maturity
        '''
        H = 10**-4
        return (self.pricing(s + H, sigma, t) + self.pricing(s - H,
                sigma, t) - 2 * self.pricing(s, sigma, t)) / (H**2)
        # *s**2

    def vanna(self, s, sigma, t):
        '''
         vanna of the option using finite difference
        s: spot price
        sigma: volatility
        t: time to maturity
        '''
        H = 10**-4
        return (self.pricing(s + H, sigma + H, t) - self.pricing(s - H, sigma + H, t) -
                self.pricing(s + H, sigma - H, t) + self.pricing(s - H, sigma - H, t)) / (4 * H**2)

    def volga(self, s, sigma, t):
        '''
         volga of the option using finite difference
        s: spot price
        sigma: volatility
        t: time to maturity
        '''
        H = 10**-4
        volga_aux = (self.pricing(s, sigma + H, t) + self.pricing(s,
                     sigma - H, t) - 2 * self.pricing(s, sigma, t)) / (H**2)
        return volga_aux

    def replicating_error(self):
        '''
        Function used to estimate the distribution of the hedging cost and slippage of a deffined option

        Output:
        error_cost_1: array of hedging cost of delta for each generated sample path
        error_cost_2: array of hedging cost of vega for each generated sample path
        error_slipage:array of hedging slippage for each generated sample path

        '''
        # hedging error
        slippage = np.zeros(self.n_paths)
        cost_1 = np.zeros(self.n_paths)
        cost_2 = np.zeros(self.n_paths)

        # 1+interest in delta_t time interval

        d = np.zeros(self.n_spot)
        d_vol = np.zeros(self.n_vol)

        q_spot = np.zeros(self.n_spot)
        q_vol = np.zeros(self.n_vol)
        tc_1 = np.zeros(self.n_spot)
        tc_2 = np.zeros(self.n_vol)

        price_interest = self.pricing(self.price_spot, self.price_vol, 0)
        compound_spot = np.exp(self.delta_spot * self.interest_rate)
        compound_vol = np.exp(self.delta_vol * self.interest_rate)

        # rf bank account at t=0- revenue of selling the option

        for j in range(0, self.n_paths):

            path, sigma = self.generate_sample_path(
                self.price_spot, self.price_vol)

            d[0] = self.delta(path[0], sigma[0], 0.0)
            d_vol[0] = self.vega(path[0], sigma[0], 0.0)
            # seperate bank accounts for each greek
            # although it is not possible to compute seperate slippages
            q_spot[0] = -d[0] * path[0]
            q_vol[0] = -d_vol[0] * sigma[0]

            tc_1[0] = 0.0
            tc_2[0] = 0.0

            # for loop to hedge delta
            for i in range(1, self.n_spot):

                d[i] = self.delta(path[i * self.factor_spot],
                                  sigma[i * self.factor_spot], self.delta_spot * i)
                # rf bank account- (last value+interest) + cost/revenue of
                # buying stocks + TC of buying stocks

                q_spot[i] = q_spot[i - 1] * compound_spot - \
                    (d[i] - d[i - 1]) * path[i * self.factor_spot]

                tc_1[i] = tc_1[i - 1] * compound_spot - \
                    np.abs(d[i] - d[i - 1]) * \
                    path[i * self.factor_spot] * self.transaction_cost_1

            # for loop to hedge vega
            for i in range(1, self.n_vol):

                # quantity of stocks- equal to the delta of the option

                d_vol[i] = self.vega(
                    path[i * self.factor_vol], sigma[i * self.factor_vol], self.delta_vol * i)

                # rf bank account- (last value+interest) + cost/revenue of
                # buying stocks + TC of buying stocks

                q_vol[i] = q_vol[i - 1] * compound_vol - \
                    (d_vol[i] - d_vol[i - 1]) * sigma[i * self.factor_vol]

                tc_2[i] = tc_2[i - 1] * compound_vol - np.abs(
                    d_vol[i] - d_vol[i - 1]) * sigma[i * self.factor_vol] * self.transaction_cost_2

            # slippage= both bank accounts+ value of hedging instruments +
            # (compounded) option price at t=0- payoff of the option

            slippage[j] = (q_spot[self.n_spot - 1] * compound_spot + q_vol[self.n_vol - 1] * compound_vol +
                           d[self.n_spot - 1] * path[self.euler] +
                           d_vol[self.n_vol - 1] * sigma[self.euler]
                           - self.exercise(path[self.euler])) * np.exp(-self.interest_rate * self.expire_date) +\
                price_interest

            cost_1[j] = (tc_1[self.n_spot - 1] * compound_spot - np.abs(self.delta_t_t(path[self.euler]) - d[self.n_spot - 1]) * path[self.euler] * self.transaction_cost_1) * \
                np.exp(-self.interest_rate * self.expire_date)
            cost_2[j] = (tc_2[self.n_vol - 1] * compound_vol- np.abs(0 - d_vol[self.n_vol - 1]) * sigma[self.euler] * self.transaction_cost_2) * \
                np.exp(-self.interest_rate * self.expire_date)

        return cost_1, cost_2, slippage

    def test_gamma(self, sd_s, sd_sigma, corr):
        '''
        Funtion used to estimate the sum of gammas used in the closed form solutions

        Relies on input of an estiamte of the volatility of spot and sigma and an estimate of correlation
        Can be obtain using the function estimate

        output:
        e_combined_gamma_vanna_a-combined gamma vanna to compute hedging cost of delta
        e_combined_gamma_vanna_q-combined gamma vanna to compute hedging slippage of delta
        e_combined_volga_vanna_a-combined volga vanna to compute hedgign cost of vega
        e_combined_volga_vanna_q-combined volga vanna to compute hedging slippage of vega


        '''

        e_combined_gamma_vanna_a = np.zeros(self.n_spot)

        e_combined_gamma_vanna_q = np.zeros(self.n_spot)

        e_combined_volga_vanna_a = np.zeros(self.n_vol)

        e_combined_volga_vanna_q = np.zeros(self.n_vol)

        combined_gamma_vanna_a = np.zeros((self.n_paths, self.n_spot))
        combined_gamma_vanna_q = np.zeros((self.n_paths, self.n_spot))

        combined_volga_vanna_a = np.zeros((self.n_paths, self.n_vol))
        combined_volga_vanna_q = np.zeros((self.n_paths, self.n_vol))

        # t=0 in the provided formulas

        for j in range(0, self.n_paths):
            path, sigma = self.generate_sample_path(
                self.price_spot, self.price_vol)
            i_spot = 0
            i_vol = 0
            for i in range(0, self.euler):
                if(i % self.factor_spot == 0 and i_spot<self.n_spot):
                    cash_gamma = self.gamma(
                        path[i], sigma[i], self.delta_spot * i_spot) * path[i]**2
                    cash_vanna = self.vanna(
                        path[i], sigma[i], self.delta_spot * i_spot) * path[i] * sigma[i]

                    combined_gamma_vanna_a[j, i_spot] = sd_s**2 * cash_gamma**2 + \
                        sd_sigma**2 * cash_vanna**2 + 2 * corr * \
                        sd_s * sd_sigma * cash_gamma * cash_vanna
                    combined_gamma_vanna_q[j, i_spot] = sd_s**2 * cash_gamma**2 + 0.5 * (
                        1 + corr**2) * sd_sigma**2 * cash_vanna**2 + 2 * corr * sd_s * sd_sigma * cash_gamma * cash_vanna
                    i_spot += 1
                if(i % self.factor_vol == 0 and i_vol<self.n_vol):
                    cash_volga = self.volga(
                        path[i], sigma[i], self.delta_vol * i_vol) * sigma[i]**2
                    cash_vanna = self.vanna(
                        path[i], sigma[i], self.delta_vol * i_vol) * path[i] * sigma[i]

                    combined_volga_vanna_a[j, i_vol] = sd_s**2 * cash_vanna**2 + \
                        sd_sigma**2 * cash_volga**2 + 2 * corr * \
                        sd_s * sd_sigma * cash_volga * cash_vanna
                    combined_volga_vanna_q[j, i_vol] = 0.5 * (
                        1 + corr**2) * sd_s**2 * cash_vanna**2 + sd_sigma**2 * cash_volga**2 + 2 * corr * sd_s * sd_sigma * cash_volga * cash_vanna
                    i_vol += 1

        for i in range(0, self.n_spot):

            e_combined_gamma_vanna_a[i] = np.mean(
                np.abs(np.exp(-self.interest_rate * self.delta_spot * (i+1)) * np.sqrt(combined_gamma_vanna_a[:, i])))

            e_combined_gamma_vanna_q[i] = np.mean(
                (np.exp(-self.interest_rate * self.delta_spot * (i+1)) * combined_gamma_vanna_q[:, i]))

        for i in range(0, self.n_vol):
            e_combined_volga_vanna_a[i] = np.mean(
                np.exp(-self.interest_rate * self.delta_vol * (i+1)) * np.sqrt(combined_volga_vanna_a[:, i]))

            e_combined_volga_vanna_q[i] = np.mean(
                (np.exp(-self.interest_rate * self.delta_vol * (i+1)) * combined_volga_vanna_q[:, i]))

        return np.mean(e_combined_gamma_vanna_a), np.mean(e_combined_gamma_vanna_q), np.mean(
            e_combined_volga_vanna_a), np.mean(e_combined_volga_vanna_q)

    def t0_greeks(self, sd_s, sd_sigma, corr):
        '''
        Funtion used to estimate the sum of gammas used in the closed form solutions, using t_0 aproximations

        Relies on input of an estiamte of the volatility of spot and sigma and an estimate of correlation
        Can be obtain using the function estimate

        output:
        e_combined_gamma_vanna_a-combined gamma vanna to compute hedging cost of delta
        e_combined_gamma_vanna_q-combined gamma vanna to compute hedging slippage of delta
        e_combined_volga_vanna_a-combined volga vanna to compute hedgign cost of vega
        e_combined_volga_vanna_q-combined volga vanna to compute hedging slippage of vega


        '''
        cash_gamma = self.gamma(
            self.price_spot, self.price_vol, 0) * self.price_spot**2

        # amortized vanna and volga due to t->0: x->0
        cash_vanna = self.vanna(
            self.price_spot, self.price_vol, 0) * self.price_spot * self.price_vol / 2
        cash_volga = self.volga(
            self.price_spot, self.price_vol, 0) * self.price_vol**2 / 2

        combined_gamma_vanna_a = np.sqrt(
            sd_s**2 * cash_gamma**2 + sd_sigma**2 * cash_vanna**2 + 2 * corr * sd_s * sd_sigma * cash_gamma * cash_vanna)

        combined_gamma_vanna_q = sd_s**2 * cash_gamma**2 + 0.5 * \
            (1 + corr**2) * sd_sigma**2 * cash_vanna**2 + 2 * \
            corr * sd_s * sd_sigma * cash_gamma * cash_vanna

        combined_volga_vanna_a = np.sqrt(
            sd_s**2 * cash_vanna**2 + sd_sigma**2 * cash_volga**2 + 2 * corr * sd_s * sd_sigma * cash_volga * cash_vanna)
        combined_volga_vanna_q = 0.5 * (1 + corr**2) * sd_s**2 * cash_vanna**2 + \
            sd_sigma**2 * cash_volga**2 + 2 * corr * \
            sd_s * sd_sigma * cash_volga * cash_vanna
        return combined_gamma_vanna_a, combined_gamma_vanna_q, combined_volga_vanna_a, combined_volga_vanna_q

    def aprox_greeks(self, sd_s, sd_sigma, corr):
        '''
        Funtion used to estimate the sum of gammas used in the closed form solutions, using the aproximation formulas

        Relies on input of an estiamte of the volatility of spot and sigma and an estimate of correlation
        Can be obtain using the function estimate

        output:
        e_combined_gamma_vanna_a-combined gamma vanna to compute hedging cost of delta
        e_combined_gamma_vanna_q-combined gamma vanna to compute hedging slippage of delta
        e_combined_volga_vanna_a-combined volga vanna to compute hedgign cost of vega
        e_combined_volga_vanna_q-combined volga vanna to compute hedging slippage of vega


        '''
        cash_vanna = 0.0
        sign_cash_vanna = 0.0

        cash_volga = 0.0
        sign_cash_volga = 0.0
        ladder = np.array(
            [60, 80, 90, 95, 100, 105, 110, 120, 140]) * self.strike_price / 100

        for i in range(len(ladder)):

            sign_cash_volga += self.volga(ladder[i], self.price_vol, 0)
            cash_volga += np.abs(self.volga(ladder[i], self.price_vol, 0))

            sign_cash_vanna += self.vanna(ladder[i], self.price_vol, 0)
            cash_vanna += np.abs(self.vanna(ladder[i], self.price_vol, 0))

        cash_gamma = self.gamma(
            self.price_spot, self.price_vol, 0) * self.price_spot**2

        cash_vanna = cash_vanna / \
            len(ladder) * self.price_vol * \
            self.price_spot * np.sign(sign_cash_vanna) / 2

        cash_volga = cash_volga / len(ladder) * \
            self.price_vol**2 * np.sign(sign_cash_volga) / 2

        combined_gamma_vanna_a = np.sqrt(
            sd_s**2 * cash_gamma**2 + sd_sigma**2 * cash_vanna**2 + 2 * corr * sd_s * sd_sigma * cash_gamma * cash_vanna)

        combined_gamma_vanna_q = sd_s**2 * cash_gamma**2 + 0.5 * \
            (1 + corr**2) * sd_sigma**2 * cash_vanna**2 + 2 * \
            corr * sd_s * sd_sigma * cash_gamma * cash_vanna

        combined_volga_vanna_a = np.sqrt(
            sd_s**2 * cash_vanna**2 + sd_sigma**2 * cash_volga**2 + 2 * corr * sd_s * sd_sigma * cash_volga * cash_vanna)

        combined_volga_vanna_q = 0.5 * (1 + corr**2) * sd_s**2 * cash_vanna**2 + \
            sd_sigma**2 * cash_volga**2 + 2 * corr * \
            sd_s * sd_sigma * cash_volga * cash_vanna

        return combined_gamma_vanna_a, combined_gamma_vanna_q, combined_volga_vanna_a, combined_volga_vanna_q

    def gamma_aprox(self, sd_s, sd_sigma, rho, e_combined_gamma_vanna_a,
                    e_combined_gamma_vanna_q, e_combined_volga_vanna_a, e_combined_volga_vanna_q):
        '''
        Function that transforms the computed combined greeks in the closed form solution of the quantities

        Input: results from estiamte functions

        Output: closed form solution of the Expected value of hedging cost of delta and vega and
        stnadart deviation of  hedging slippage of delta and vega


        '''

        e_friction_s = e_combined_gamma_vanna_a * self.transaction_cost_1 * \
            np.sqrt(2 * self.expire_date * (self.n_spot) / np.pi)

        sd_slippage_s = np.sqrt(e_combined_gamma_vanna_q) * \
            sd_s * self.expire_date * np.sqrt(1 / (2 * (self.n_spot)))

        e_friction_sigma = e_combined_volga_vanna_a * self.transaction_cost_2 * \
            np.sqrt(2 * self.expire_date * (self.n_vol) / np.pi)

        sd_slippage_sigma = np.sqrt(
            e_combined_volga_vanna_q) * sd_sigma * self.expire_date * np.sqrt(1 / (2 * (self.n_vol)))

        return e_friction_s, sd_slippage_s, e_friction_sigma, sd_slippage_sigma

    def test_vanna_volga(self):
        '''
        Funtion used to test the behavior of the cash vanna and volga of the option


        output:
        cash_vanna->actualized expected value of the cash vanna in each moment defined in the sums
        cash_volga->actualized expected value of the cash volga in each moment defined in the sums


        '''

        cash_vanna = np.zeros(self.n_vol)

        cash_volga = np.zeros(self.n_vol)

        cash_volga_arr = np.zeros((self.n_paths, self.n_vol))
        cash_vanna_arr = np.zeros((self.n_paths, self.n_vol))

        # t=0 in the provided formulas

        for j in range(0, self.n_paths):
            path, sigma = self.generate_sample_path(
                self.price_spot, self.price_vol)
            i_vol = 0
            for i in range(0, self.euler):
                if(i % self.factor_vol == 0):
                    cash_volga_arr[j, i_vol] = self.volga(
                        path[i], sigma[i], self.delta_vol * i_vol) * sigma[i]**2
                    cash_vanna_arr[j, i_vol] = self.vanna(
                        path[i], sigma[i], self.delta_vol * i_vol) * path[i] * sigma[i]
                    i_vol = i_vol + 1

        for i in range(0, self.n_vol):

            cash_vanna[i] = np.mean(
                np.exp(-self.interest_rate * self.delta_vol * (i)) * cash_vanna_arr[:, i])

            cash_volga[i] = np.mean(
                (np.exp(-self.interest_rate * self.delta_vol * (i)) * cash_volga_arr[:, i]))

        return cash_vanna, cash_volga

from .replicating_vol import Option_vol, spec_vol
from .misc import W_QUAD_PLUS, PHI_QUAD_PLUS, TIME
from numba.experimental import jitclass
from numba import double, njit, prange
import numpy as np
spec_heston = spec_vol + [
    ('kappa', double),
    ('theta', double),
    ('epsilon', double),
    ('corr', double)
]


@jitclass(spec_heston)
class Vol_heston(Option_vol):

    __init__A = Option_vol.__init__

    def __init__(
            self,
            price_spot,
            price_vol,
            actual_mean,
            kappa,
            theta,
            epsilon,
            corr,
            interest_rate,
            expire_date,
            strike_price,
            transaction_cost_1,
            transaction_cost_2,
            euler,
            n_spot,
            n_vol,
            n_paths):

        self.__init__A(
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
            n_paths)

        # feller condition
        assert(2 * kappa * theta > epsilon**2)

        # same notation
        self.kappa = kappa
        self.theta = theta
        self.epsilon = epsilon
        self.corr = corr

    def generate_sample_path(self, s_0, sigma_0):
        '''
        Function to generate a sample path of the spot price and the volatility using Euler–Maruyama method

        This function used full truncation correction to avoid negative instant variances

        s_0: spot price at t=0
        sigma_0: volatility at t=0

        '''
        path = np.empty(self.euler + 1)
        sigma = np.empty(self.euler + 1)
        path[0] = s_0
        sigma[0] = sigma_0**2

        for i in range(1, self.euler + 1):
            rng_1 = np.random.normal()
            rng_2 = self.corr * rng_1 + \
                np.sqrt(1 - self.corr**2) * np.random.normal()

            path[i] = path[i - 1] * np.exp((self.actual_mean - 0.5 * max(sigma[i - 1], 0.0))
                                           * self.delta_euler + np.sqrt(self.delta_euler * max(sigma[i - 1], 0.0)) * rng_1)

            sigma[i] = sigma[i - 1] + self.kappa * (self.theta - max(sigma[i - 1], 0.0)) * self.delta_euler + \
                self.epsilon * \
                np.sqrt(max(sigma[i - 1], 0.0) * self.delta_euler) * rng_2
        # 0.001 instead of zero to not create negative volatilities when
        # computing derivatives
        return path, np.sqrt(np.maximum(sigma, 0.0002))

    def estimate(self):
        '''
        Function to estimate sigma_s , sigma_Sigma and rho to use in the closed form solutions using montecarlo
        '''
        sd_s_arr = np.zeros(self.n_paths)
        sd_sigma_arr = np.zeros(self.n_paths)
        corr_arr = np.zeros(self.n_paths)

        for i in range(self.n_paths):
            spot, sigma = self.generate_sample_path(
                self.price_spot, self.price_vol)
            sd_s_arr[i] = np.std(np.diff(np.log(spot))) * \
                np.sqrt(self.euler / self.expire_date)
            sd_sigma_arr[i] = np.std(
                np.diff(np.log(sigma))) * np.sqrt(self.euler / self.expire_date)
            corr_arr[i] = np.corrcoef(
                np.diff(np.log(spot)), np.diff(np.log(sigma)))[0, 1]

        sd_s = np.mean(sd_s_arr)
        sd_sigma = np.mean(sd_sigma_arr)
        corr = np.mean(corr_arr)

        return sd_s, sd_sigma, corr

    def exercise(self, s):
        '''
        Function to check the price of the option at expire
        s: spot price
        '''
        return max(s - self.strike_price, 0)

    def phi_1(self, s, sigma, t, phi):
        '''

        1st characteristic function for heston

        s: spot price
        sigma: volatility
        t: time to maturity
        phi:

        '''
        u = 0.5
        a = self.kappa * self.theta
        b = self.kappa - self.corr * self.epsilon

        aux = self.corr * self.epsilon * phi * 1j
        tau = self.expire_date - t

        d = np.sqrt((aux - b)**2 - self.epsilon **
                    2 * (2 * u * phi * 1j - phi**2))
        g = (b - aux + d) / (b - aux - d)

        c_f = (self.interest_rate) * phi * 1j * tau + a / (self.epsilon**2) * \
            ((b - aux + d) * tau - 2 * np.log((1 - g * np.exp(d * tau)) / (1 - g)))
        d_f = (b - aux + d) * (1 - np.exp(d * tau)) / \
            ((1 - g * np.exp(d * tau)) * (self.epsilon**2))

        return np.exp(c_f + d_f * sigma + 1j * phi * np.log(s))

    def Q1(self, s, sigma, t):
        '''

        1st integral, using a 100-point gauss leguerre quadrature

        s: spot price
        sigma: volatility
        t: time to maturity

        '''

        phi_1 = self.phi_1
        integral = 0.0
        for i in range(len(PHI_QUAD_PLUS)):
            u = PHI_QUAD_PLUS[i]
            integral = integral + \
                np.real((np.exp(u - u * 1j * np.log(self.strike_price)) *
                        (phi_1(s, sigma, t, u))) / (1j * u)) * W_QUAD_PLUS[i]

        return 0.5 + 1 / np.pi * integral

    def phi_2(self, s, sigma, t, phi):
        '''

        2nd characteristic function for heston

        s: spot price
        sigma: volatility
        t: time to maturity
        phi:

        '''

        u = -0.5
        a = self.kappa * self.theta
        b = self.kappa

        aux = self.corr * self.epsilon * phi * 1j
        tau = self.expire_date - t

        d = np.sqrt((aux - b)**2 - self.epsilon **
                    2 * (2 * u * phi * 1j - phi**2))
        g = (b - aux + d) / (b - aux - d)

        c_f = (self.interest_rate) * phi * 1j * tau + a / (self.epsilon**2) * \
            ((b - aux + d) * tau - 2 * np.log((1 - g * np.exp(d * tau)) / (1 - g)))
        d_f = (b - aux + d) / (self.epsilon**2) * \
            (1 - np.exp(d * tau)) / (1 - g * np.exp(d * tau))

        return np.exp(c_f + d_f * sigma + 1j * phi * np.log(s))

    def Q2(self, s, sigma, t):
        '''

        2nd integral, using a 100-point gauss leguerre quadrature

        s: spot price
        sigma: volatility
        t: time to maturity

        '''
        phi_2 = self.phi_2
        integral = 0.0
        for i in range(len(PHI_QUAD_PLUS)):
            u = PHI_QUAD_PLUS[i]
            integral = integral + \
                np.real((np.exp(u - u * 1j * np.log(self.strike_price)) *
                        (phi_2(s, sigma, t, u))) / (1j * u)) * W_QUAD_PLUS[i]

        return 0.5 + 1 / np.pi * integral

    def pricing(self, s, sigma, t):
        '''

        heston european call price, using semi-closed form solution

        s: spot price
        sigma: volatility
        t: time to maturity


        '''
        sigma = sigma**2
        moneyness = np.log(s / self.strike_price) / \
            np.sqrt(sigma * (self.expire_date - t))

        # asymptotic option prices for deep OTM and ITM options
        if moneyness < -15:
            return 0
        if moneyness > 15:
            return s - self.strike_price * \
                np.exp(-self.interest_rate * (self.expire_date - t))

        return s * self.Q1(s, sigma, t) - np.exp(-self.interest_rate *
                                                 (self.expire_date - t)) * self.strike_price * self.Q2(s, sigma, t)

    def delta(self, s, sigma, t):
        '''
        Delta of the option using finite difference
        s: spot price
        sigma: volatility
        t: time to maturity
        '''

        H = 10**-8
        price_1 = self.pricing(s + H, sigma, t)
        price_2 = self.pricing(s - H, sigma, t)

        return (price_1 - price_2) / (2 * H)

    def delta_t_t(self,s):

        if self.exercise(s)>0.0:
            return 1
        else:
            return 0


    def vega(self, s, sigma, t):
        '''
        Vega of the option using finite difference
        s: spot price
        sigma: volatility
        t: time to maturity
        '''

        H = 10**-8
        price_1 = self.pricing(s, sigma + H, t)
        price_2 = self.pricing(s, sigma - H, t)

        return (price_1 - price_2) / (2 * H)


@njit(parallel=True)
def risk_return_vol_heston(
    price_spot,
    price_vol,
    actual_mean,
    kappa,
    theta,
    epsilon,
    corr,
    interest_rate,
    expire_date,
    strike_price,
    transaction_cost_1,
    transaction_cost_2,
    n_paths
):
    '''
    Function to obtain in parallel the hedging cost and slippage of
    hedging the option with N=1,...,200 hedging intervals for a european call using heston model

    Input:
    price_spot: spot price at t=0
    price_vol: volatility at t=0
    kappa: rate of reversion
    theta: long term volatility^2
    epsilon: volatility of volatility^2
    corr:correlation between BM of spot and vol^2
    interest_rate: continuous compounded interet rate
    expire_date
    strike_price
    transaction_cost_1: half of the relative bid-offer spread in percentage of underlying price
    transaction_cost_2: half of the relative bid-offer spread in percentage of volatility
    n_paths: number of sample paths to estimates each option hedging error

    Output:
    list,list_risk: mean and standard deviation of the Pnl of hedging the option with N=1,...,201
    list_cost, list_cost_risk: mean and standard deviation of hedging cost  with with N=1,...,201
    list_slipage, list_slipage_risk: mean and standard deviation of hedging slippage  with with N=1,...,201


    '''
    # function to test the risk return profile of the replicating strategy for
    # different time intervals
    SIZE = len(TIME)
    list = np.zeros((SIZE, SIZE))
    list_risk = np.zeros((SIZE, SIZE))

    list_cost_1 = np.zeros((SIZE, SIZE))
    list_cost_risk_1 = np.zeros((SIZE, SIZE))

    list_cost_2 = np.zeros((SIZE, SIZE))
    list_cost_risk_2 = np.zeros((SIZE, SIZE))

    list_slipage = np.zeros((SIZE, SIZE))
    list_slipage_risk = np.zeros((SIZE, SIZE))

    for i in prange(SIZE * SIZE):

        n_spot_plus_1 = int(i / SIZE)
        n_vol_plus_1 = int(i % SIZE)
        print(TIME[n_spot_plus_1], TIME[n_vol_plus_1])
        option = Vol_heston(
            price_spot,
            price_vol,
            actual_mean,
            kappa,
            theta,
            epsilon,
            corr,
            interest_rate,
            expire_date,
            strike_price,
            transaction_cost_1,
            transaction_cost_2,
            1000,
            TIME[n_spot_plus_1],
            TIME[n_vol_plus_1],
            n_paths)

        result_cost_1, result_cost_2, result_slipage = option.replicating_error()

        list[n_spot_plus_1, n_vol_plus_1] = (
            np.mean(result_cost_1 + result_cost_2 + result_slipage))
        list_risk[n_spot_plus_1, n_vol_plus_1] = (
            np.std(result_cost_1 + result_cost_2 + result_slipage))

        list_cost_1[n_spot_plus_1, n_vol_plus_1] = (np.mean(result_cost_1))
        list_cost_risk_1[n_spot_plus_1, n_vol_plus_1] = (np.std(result_cost_1))

        list_cost_2[n_spot_plus_1, n_vol_plus_1] = (np.mean(result_cost_2))
        list_cost_risk_2[n_spot_plus_1, n_vol_plus_1] = (np.std(result_cost_2))

        list_slipage[n_spot_plus_1, n_vol_plus_1] = (np.mean(result_slipage))
        list_slipage_risk[n_spot_plus_1, n_vol_plus_1] = (
            np.std(result_slipage))

    return list, list_risk, list_cost_1, list_cost_risk_1, list_cost_2, list_cost_risk_2, list_slipage, list_slipage_risk


@njit(parallel=True)
def gamma_test_vol_heston(
        price_spot,
    price_vol,
    actual_mean,
    kappa,
    theta,
    epsilon,
    corr,
    interest_rate,
    expire_date,
    strike_price,
    transaction_cost_1,
    transaction_cost_2,
    n_paths
):
    '''
    Function to obtain in parallel absolute sum of gamma and the square sum of gammas
    to compute the closed form solutions over with N=1,...,200 hedging intervals for a european call using heston model

    Input:
    price_spot: spot price at t=0
    price_vol: volatility at t=0
    kappa: rate of reversion
    theta: long term volatility^2
    epsilon: volatility of volatility^2
    corr:correlation between BM of spot and vol^2
    interest_rate: continuous compounded interet rate
    expire_date
    strike_price
    transaction_cost_1: half of the relative bid-offer spread in percentage of underlying price
    transaction_cost_2: half of the relative bid-offer spread in percentage of volatility
    n_paths: number of sample paths to estimates each option hedging error

    Output:
    list_e_s,list_sd_s,list_e_sigma, list_sd_sigma-> results form the closed form solutions
    list_gamma_vanna_a ,list_gamma_vanna_q ,list_volga_vanna_a ,list_volga_vanna_q-> sum of gammas computed
    sd_s_coef, sd_sigma_coef, corr_coef-> estimated parameters to use in  the closed form solutions


    '''
    list_gamma_vanna_a = np.zeros(len(TIME))
    list_gamma_vanna_q = np.zeros(len(TIME))
    list_volga_vanna_a = np.zeros(len(TIME))
    list_volga_vanna_q = np.zeros(len(TIME))

    list_e_s = np.zeros(len(TIME))
    list_sd_s = np.zeros(len(TIME))

    list_e_sigma = np.zeros(len(TIME))
    list_sd_sigma = np.zeros(len(TIME))

    option_aux = Vol_heston(
        price_spot,
        price_vol,
        actual_mean,
        kappa,
        theta,
        epsilon,
        corr,
        interest_rate,
        expire_date,
        strike_price,
        transaction_cost_1,
        transaction_cost_2,
        1000,
        1,
        1,
        n_paths)

    sd_s_coef, sd_sigma_coef, corr_coef = option_aux.estimate()

    for i in prange(len(TIME)):

        print(TIME[i])
        option = Vol_heston(
            price_spot,
            price_vol,
            actual_mean,
            kappa,
            theta,
            epsilon,
            corr,
            interest_rate,
            expire_date,
            strike_price,
            transaction_cost_1,
            transaction_cost_2,
            1000,
            TIME[i],
            TIME[i],
            n_paths)

        gamma_vanna_a, gamma_vanna_q, volga_vanna_a, volga_vanna_q = option.test_gamma(
            sd_s_coef, sd_sigma_coef, corr_coef)

        e_s, sd_s, e_sigma, sd_sigma = option.gamma_aprox(
            sd_s_coef, sd_sigma_coef, corr_coef, gamma_vanna_a, gamma_vanna_q, volga_vanna_a, volga_vanna_q)

        list_gamma_vanna_a[i] = gamma_vanna_a
        list_gamma_vanna_q[i] = gamma_vanna_q

        list_volga_vanna_a[i] = volga_vanna_a
        list_volga_vanna_q[i] = volga_vanna_q

        list_e_s[i] = e_s
        list_sd_s[i] = sd_s

        list_e_sigma[i] = e_sigma
        list_sd_sigma[i] = sd_sigma

    return list_e_s, list_sd_s, list_e_sigma, list_sd_sigma, list_gamma_vanna_a, list_gamma_vanna_q, list_volga_vanna_a, list_volga_vanna_q, sd_s_coef, sd_sigma_coef, corr_coef


@njit(parallel=True)
def gamma_test_vol_heston_t(
        price_spot,
    price_vol,
    actual_mean,
    kappa,
    theta,
    epsilon,
    corr,
    interest_rate,
    expire_date,
    strike_price,
    transaction_cost_1,
    transaction_cost_2,
    n_paths,
    sd_s_coef, sd_sigma_coef, corr_coef
):
    '''
    Function to obtain in parallel absolute sum of gamma and the square sum of gammas using t_0 aproximation
    to compute the closed form solutions over with N=1,...,200 hedging intervals for a european call using heston model

    Input:
    price_spot: spot price at t=0
    price_vol: volatility at t=0
    kappa: rate of reversion
    theta: long term volatility^2
    epsilon: volatility of volatility^2
    corr:correlation between BM of spot and vol^2
    interest_rate: continuous compounded interet rate
    expire_date
    strike_price
    transaction_cost_1: half of the relative bid-offer spread in percentage of underlying price
    transaction_cost_2: half of the relative bid-offer spread in percentage of volatility
    n_paths: number of sample paths to estimates each option hedging error
    sd_s_coef, sd_sigma_coef, corr_coef-> results from the option.estimate function

    Output:
    list_e_s,list_sd_s,list_e_sigma, list_sd_sigma-> results form the closed form solutions


    '''

    list_e_s = np.zeros(len(TIME))
    list_sd_s = np.zeros(len(TIME))

    list_e_sigma = np.zeros(len(TIME))
    list_sd_sigma = np.zeros(len(TIME))

    for i in prange(len(TIME)):

        option = Vol_heston(
            price_spot,
            price_vol,
            actual_mean,
            kappa,
            theta,
            epsilon,
            corr,
            interest_rate,
            expire_date,
            strike_price,
            transaction_cost_1,
            transaction_cost_2,
            1000,
            TIME[i],
            TIME[i],
            n_paths)

        gamma_vanna_a, gamma_vanna_q, volga_vanna_a, volga_vanna_q = option.t0_greeks(
            sd_s_coef, sd_sigma_coef, corr_coef)

        e_s, sd_s, e_sigma, sd_sigma = option.gamma_aprox(
            sd_s_coef, sd_sigma_coef, corr_coef, gamma_vanna_a, gamma_vanna_q, volga_vanna_a, volga_vanna_q)

        list_e_s[i] = e_s
        list_sd_s[i] = sd_s

        list_e_sigma[i] = e_sigma
        list_sd_sigma[i] = sd_sigma

    return list_e_s, list_sd_s, list_e_sigma, list_sd_sigma


@njit(parallel=True)
def gamma_test_vol_heston_aprox(
        price_spot,
    price_vol,
    actual_mean,
    kappa,
    theta,
    epsilon,
    corr,
    interest_rate,
    expire_date,
    strike_price,
    transaction_cost_1,
    transaction_cost_2,
    n_paths,
    sd_s_coef, sd_sigma_coef, corr_coef
):
    '''
    Function to obtain in parallel absolute sum of gamma and the square sum of gammas using aproximation formulas
    to compute the closed form solutions over with N=1,...,200 hedging intervals for a european call using heston model

    Input:
    price_spot: spot price at t=0
    price_vol: volatility at t=0
    kappa: rate of reversion
    theta: long term volatility^2
    epsilon: volatility of volatility^2
    corr:correlation between BM of spot and vol^2
    interest_rate: continuous compounded interet rate
    expire_date
    strike_price
    transaction_cost_1: half of the relative bid-offer spread in percentage of underlying price
    transaction_cost_2: half of the relative bid-offer spread in percentage of volatility
    n_paths: number of sample paths to estimates each option hedging error
    sd_s_coef, sd_sigma_coef, corr_coef-> results from the option.estimate function

    Output:
    list_e_s,list_sd_s,list_e_sigma, list_sd_sigma-> results form the closed form solutions

    '''

    list_e_s = np.zeros(len(TIME))
    list_sd_s = np.zeros(len(TIME))

    list_e_sigma = np.zeros(len(TIME))
    list_sd_sigma = np.zeros(len(TIME))

    for i in prange(len(TIME)):

        option = Vol_heston(
            price_spot,
            price_vol,
            actual_mean,
            kappa,
            theta,
            epsilon,
            corr,
            interest_rate,
            expire_date,
            strike_price,
            transaction_cost_1,
            transaction_cost_2,
            1000,
            TIME[i],
            TIME[i],
            n_paths)

        gamma_vanna_a, gamma_vanna_q, volga_vanna_a, volga_vanna_q = option.aprox_greeks(
            sd_s_coef, sd_sigma_coef, corr_coef)

        e_s, sd_s, e_sigma, sd_sigma = option.gamma_aprox(
            sd_s_coef, sd_sigma_coef, corr_coef, gamma_vanna_a, gamma_vanna_q, volga_vanna_a, volga_vanna_q)

        list_e_s[i] = e_s
        list_sd_s[i] = sd_s

        list_e_sigma[i] = e_sigma
        list_sd_sigma[i] = sd_sigma

    return list_e_s, list_sd_s, list_e_sigma, list_sd_sigma

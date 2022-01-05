
from tools_stable.replicating_fixed_time import risk_return_fixed_time
from tools_stable.replicating_fixed_bandwidth import risk_return_fixed_bandwidth
from option_class import Short_european_call_black_sholes_fixed_time, Short_european_call_leland_fixed_time,Short_european_call_black_sholes_fixed_bandwidth
from tools_stable.misc import mean_CI
from tools_stable.misc import sd_CI
import numpy as np
import matplotlib.pyplot as plt


'''
Yet Another Note on the Leland’s Option
Hedging Strategy with Transaction Costs
Valeri I. Zakamouline
table 1
'''


import time

time0 = time.time()


option = Short_european_call_black_sholes_fixed_time(price_0=100.0, actual_mean=0.05, actual_vol=0.25, interest_rate=0.05,
                                          expire_date=1, strike_price=100.0, transaction_cost=0.01, n_steps=100, n_paths=200000)
result = option.replicating_error()


print(mean_CI(result))
print(sd_CI(result))


option1 = Short_european_call_leland_fixed_time(price_0=100.0, actual_mean=0.05, actual_vol=0.25, interest_rate=0.05,
                                     expire_date=1, strike_price=100.0, transaction_cost=0.01, n_steps=100, n_paths=200000)

result1 = option1.replicating_error()

print(mean_CI(result1))
print(sd_CI(result1))



'''
Yet Another Note on the Leland’s Option
Hedging Strategy with Transaction Costs
Valeri I. Zakamouline
figure 1 (a)
'''


time0 = time.time()

black_sholes_mean, black_sholes_sd=risk_return_fixed_time(
    100.0, 0.05, 0.25, 0.05, 1, 100.0, 0.01, 200000,Short_european_call_black_sholes_fixed_time,np.std)
leland_mean, leland_sd = risk_return_fixed_time(
    100.0, 0.05, 0.25, 0.05, 1, 100.0, 0.01, 200000,Short_european_call_leland_fixed_time,np.std)



print(time.time()-time0)
plt.plot(black_sholes_sd, black_sholes_mean)
plt.plot(leland_sd, leland_mean)
plt.legend(['black_sholes', 'leland'], loc='upper left')
plt.show()


'''
Efficient Analytic Approximation of the
Optimal Hedging Strategy for a European
Call Option with Transaction Costs

'''


black_sholes_mean_b, black_sholes_sd_b=risk_return_fixed_bandwidth(
    100.0, 0.05, 0.25, 0.05, 1, 100.0, 0.01, 200000,Short_european_call_black_sholes_fixed_bandwidth,np.std)
plt.plot(black_sholes_sd, black_sholes_mean)
plt.plot(leland_sd, leland_mean)
plt.plot(black_sholes_sd_b, black_sholes_mean_b)
plt.legend(['fixed_time','leland', 'fixed_bandwidth'], loc='upper left')
plt.show()

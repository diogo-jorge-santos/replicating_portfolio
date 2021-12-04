

from option_class import Short_european_call_black_sholes, Short_european_call_leland, frontier_parallel
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


option = Short_european_call_black_sholes(price_0=100.0, actual_mean=0.05, actual_vol=0.25, interest_rate=0.05,
                                          expire_date=1, strike_price=100.0, transaction_cost=0.01, n_steps=100, n_paths=200000)
result = option.replicating_error()

print(np.mean(result))
print(np.std(result))


option1 = Short_european_call_leland(price_0=100.0, actual_mean=0.05, actual_vol=0.25, interest_rate=0.05,
                                     expire_date=1, strike_price=100.0, transaction_cost=0.01, n_steps=100, n_paths=200000)

result1 = option1.replicating_error()


print(np.mean(result1))
print(np.std(result1))


print("\n")
print(time.time()-time0)


'''
Yet Another Note on the Leland’s Option
Hedging Strategy with Transaction Costs
Valeri I. Zakamouline
figure 1 (a)
'''


time0 = time.time()

black_sholes_mean, black_sholes_sd, leland_mean, leland_sd = risk_return_parallel(
    100.0, 0.05, 0.25, 0.05, 1, 100.0, 0.01, 200000)

print(time.time()-time0)
plt.plot(black_sholes_sd, black_sholes_mean)
plt.plot(leland_sd, leland_mean)
plt.legend(['black_sholes', 'leland'], loc='upper left')
plt.show()

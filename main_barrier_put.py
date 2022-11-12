import os
import time
from tools_stable.class_barrier_put import Down_and_in_put, risk_return_barrier_put, gamma_test_barrier_put
import numpy as np
import matplotlib.pyplot as plt
import os
import pickle


plt.rcParams['text.usetex'] = True

plt.rcParams.update({
    'text.latex.preamble': r'\usepackage{amsfonts}'
})

time0 = time.time()
# default
result_file = open("results/3/result_3.txt", mode="w")

if os.path.isfile("results/3/MC.pickle"):
    mean_total, sd_total, mean_cost, sd_cost, mean_slippage, sd_slippage = pickle.load(
        open("results/3/MC.pickle", "rb"))
else:
    mean_total, sd_total, mean_cost, sd_cost, mean_slippage, sd_slippage = risk_return_barrier_put(
        100.0, 0.05, 0.25, 0.05, 1, 100.0, 85, 0.01)
    pickle.dump((mean_total, sd_total, mean_cost, sd_cost,
                mean_slippage, sd_slippage), open("results/3/MC.pickle", "wb"))


####################
# tc
####################
# high tc


if os.path.isfile("results/3/MC_1.pickle"):
    mean_total_1, sd_total_1, mean_cost_1, sd_cost_1, mean_slippage_1, sd_slippage_1 = pickle.load(
        open("results/3/MC_1.pickle", "rb"))
else:
    mean_total_1, sd_total_1, mean_cost_1, sd_cost_1, mean_slippage_1, sd_slippage_1 = risk_return_barrier_put(
        100.0, 0.05, 0.25, 0.05, 1, 100.0, 85, 0.05)
    pickle.dump((mean_total_1, sd_total_1, mean_cost_1, sd_cost_1,
                mean_slippage_1, sd_slippage_1), open("results/3/MC_1.pickle", "wb"))

print("high_tc")

# low tc

if os.path.isfile("results/3/MC_2.pickle"):
    mean_total_2, sd_total_2, mean_cost_2, sd_cost_2, mean_slippage_2, sd_slippage_2 = pickle.load(
        open("results/3/MC_2.pickle", "rb"))
else:
    mean_total_2, sd_total_2, mean_cost_2, sd_cost_2, mean_slippage_2, sd_slippage_2 = risk_return_barrier_put(
        100.0, 0.05, 0.25, 0.05, 1, 100.0, 85, 0.002)
    pickle.dump((mean_total_2, sd_total_2, mean_cost_2, sd_cost_2,
                mean_slippage_2, sd_slippage_2), open("results/3/MC_2.pickle", "wb"))

print("low_tc")


plt.plot(sd_total_2, mean_total_2)
plt.plot(sd_total, mean_total)
plt.plot(sd_total_1, mean_total_1)

plt.legend(["Low TC", "Default", "High TC"], loc='upper left')
plt.xlabel("sd_total")
plt.ylabel("mean_total")
# plt.show()
plt.savefig('images/3/risk_return_k_put.png')
plt.close()

plt.plot(np.array(range(1, 201)), mean_cost_2)
plt.plot(np.array(range(1, 201)), mean_cost)
plt.plot(np.array(range(1, 201)), mean_cost_1)

plt.legend(["Low TC", "Default", "High TC"], loc='upper left')
plt.xlabel("$N_S$")
plt.ylabel("$\widehat{\mathbb{E}[Friction]}$")


plt.savefig('images/3/mc_k_put.png')
plt.close()

plt.plot(np.array(range(1, 201)), sd_cost_2)
plt.plot(np.array(range(1, 201)), sd_cost)
plt.plot(np.array(range(1, 201)), sd_cost_1)

plt.legend(["Low TC", "Default", "High TC"], loc='upper left')
plt.xlabel("$N_S$")
plt.ylabel("sd_cost")
plt.savefig('images/3/sc_k_put.png')
plt.close()

plt.plot(np.array(range(1, 201)), sd_slippage_2)
plt.plot(np.array(range(1, 201)), sd_slippage)
plt.plot(np.array(range(1, 201)), sd_slippage_1)
plt.legend(["Low TC", "Default", "High TC"], loc='upper left')
plt.xlabel("$N_S$")
plt.ylabel("$\widehat{Sd(Slippage)}$")

plt.savefig('images/3/sl_k_put.png')
plt.close()


plt.plot(np.array(range(1, 201)), mean_slippage_2)
plt.plot(np.array(range(1, 201)), mean_slippage)
plt.plot(np.array(range(1, 201)), mean_slippage_1)

plt.legend(["Low TC", "Default", "High TC"], loc='upper left')
plt.xlabel("$N_S$")
plt.ylabel("mean_slippage")
plt.savefig('images/3/ml_k_put.png')
plt.close()


plt.plot(np.array(range(1, 201)), ((sd_total_2**2 - sd_cost_2 **
         2 - sd_slippage_2**2) / (2 * sd_cost_2 * sd_slippage_2)))
plt.plot(np.array(range(1, 201)), ((sd_total**2 - sd_cost **
         2 - sd_slippage**2) / (2 * sd_cost * sd_slippage)))
plt.plot(np.array(range(1, 201)), ((sd_total_1**2 - sd_cost_1 **
         2 - sd_slippage_1**2) / (2 * sd_cost_1 * sd_slippage_1)))
plt.legend(["Low TC", "Default", "High TC"], loc='upper left')
plt.xlabel("$N_S$")
plt.ylabel("corr_coef")
plt.savefig('images/3/corr_k_put.png')
plt.close()


###############
# VOL
###############

# low vol
if os.path.isfile("results/3/MC_3.pickle"):
    mean_total_3, sd_total_3, mean_cost_3, sd_cost_3, mean_slippage_3, sd_slippage_3 = pickle.load(
        open("results/3/MC_3.pickle", "rb"))
else:
    mean_total_3, sd_total_3, mean_cost_3, sd_cost_3, mean_slippage_3, sd_slippage_3 = risk_return_barrier_put(
        100.0, 0.05, 0.1, 0.05, 1, 100.0, 85, 0.01)
    pickle.dump((mean_total_3, sd_total_3, mean_cost_3, sd_cost_3,
                mean_slippage_3, sd_slippage_3), open("results/3/MC_3.pickle", "wb"))

if os.path.isfile("results/3/MC_4.pickle"):
    mean_total_4, sd_total_4, mean_cost_4, sd_cost_4, mean_slippage_4, sd_slippage_4 = pickle.load(
        open("results/3/MC_4.pickle", "rb"))
else:
    mean_total_4, sd_total_4, mean_cost_4, sd_cost_4, mean_slippage_4, sd_slippage_4 = risk_return_barrier_put(
        100.0, 0.05, 0.5, 0.05, 1, 100.0, 85, 0.01)
    pickle.dump((mean_total_4, sd_total_4, mean_cost_4, sd_cost_4,
                mean_slippage_4, sd_slippage_4), open("results/3/MC_4.pickle", "wb"))

print("low_vol")
# high vol

print("high_vol")


plt.plot(sd_total_3, mean_total_3)
plt.plot(sd_total, mean_total)
plt.plot(sd_total_4, mean_total_4)
plt.legend(["Low vol", "Default", "High vol"], loc='upper left')


plt.xlabel("sd_total")
plt.ylabel("mean_total")
plt.savefig('images/3/risk_return_v_put.png')
plt.close()

plt.plot(np.array(range(1, 201)), mean_cost_3)
plt.plot(np.array(range(1, 201)), mean_cost)

plt.plot(np.array(range(1, 201)), mean_cost_4)
plt.legend(["Low vol", "Default", "High vol"], loc='upper left')

plt.xlabel("$N_S$")
plt.ylabel("$\widehat{\mathbb{E}[Friction]}$")

plt.savefig('images/3/mc_v_put.png')
plt.close()

plt.plot(np.array(range(1, 201)), sd_cost_3)
plt.plot(np.array(range(1, 201)), sd_cost)
plt.plot(np.array(range(1, 201)), sd_cost_4)
plt.legend(["Low vol", "Default", "High vol"], loc='upper left')

plt.xlabel("$N_S$")
plt.ylabel("sd_cost")
plt.savefig('images/3/sc_v_put.png')
plt.close()

plt.plot(np.array(range(1, 201)), mean_slippage_3)
plt.plot(np.array(range(1, 201)), mean_slippage)
plt.plot(np.array(range(1, 201)), mean_slippage_4)
plt.legend(["Low vol", "Default", "High vol"], loc='upper left')

plt.xlabel("$N_S$")
plt.ylabel("mean_slippage")
plt.savefig('images/3/ml_v_put.png')
plt.close()

plt.plot(np.array(range(1, 201)), sd_slippage_3)
plt.plot(np.array(range(1, 201)), sd_slippage)
plt.plot(np.array(range(1, 201)), sd_slippage_4)
plt.legend(["Low vol", "Default", "High vol"], loc='upper left')

plt.xlabel("$N_S$")
plt.ylabel("$\widehat{Sd(Slippage)}$")

plt.savefig('images/3/sl_v_put.png')
plt.close()


plt.plot(np.array(range(1, 201)), ((sd_total_3**2 - sd_cost_3 **
         2 - sd_slippage_3**2) / (2 * sd_cost_3 * sd_slippage_3)))
plt.plot(np.array(range(1, 201)), ((sd_total**2 - sd_cost **
         2 - sd_slippage**2) / (2 * sd_cost * sd_slippage)))
plt.plot(np.array(range(1, 201)), ((sd_total_4**2 - sd_cost_4 **
         2 - sd_slippage_4**2) / (2 * sd_cost_4 * sd_slippage_4)))
plt.legend(["Low vol", "Default", "High vol"], loc='upper left')

plt.xlabel("$N_S$")
plt.ylabel("corr_coef")
plt.savefig('images/3/corr_v_put.png')
plt.close()


###############
# BARRIER
###############

# LOW BARRIER


if os.path.isfile("results/3/MC_5.pickle"):
    mean_total_5, sd_total_5, mean_cost_5, sd_cost_5, mean_slippage_5, sd_slippage_5 = pickle.load(
        open("results/3/MC_5.pickle", "rb"))
else:
   
    mean_total_5, sd_total_5, mean_cost_5, sd_cost_5, mean_slippage_5, sd_slippage_5 = risk_return_barrier_put(
        100.0, 0.05, 0.25, 0.05, 1, 100.0, 75, 0.01)
    pickle.dump((mean_total_5, sd_total_5, mean_cost_5, sd_cost_5,
                mean_slippage_5, sd_slippage_5), open("results/3/MC_5.pickle", "wb"))
    
print("low_vol")
# HIGH BARRIER


if os.path.isfile("results/3/MC_6.pickle"):
    mean_total_6, sd_total_6, mean_cost_6, sd_cost_6, mean_slippage_6, sd_slippage_6 = pickle.load(
        open("results/3/MC_6.pickle", "rb"))
else:
   
    mean_total_6, sd_total_6, mean_cost_6, sd_cost_6, mean_slippage_6, sd_slippage_6 = risk_return_barrier_put(
        100.0, 0.05, 0.25, 0.05, 1, 100.0, 95, 0.01)
    pickle.dump((mean_total_6, sd_total_6, mean_cost_6, sd_cost_6,
                mean_slippage_6, sd_slippage_6), open("results/3/MC_6.pickle", "wb"))
   
print("high_vol")


plt.plot(sd_total_5, mean_total_5)
plt.plot(sd_total, mean_total)
plt.plot(sd_total_6, mean_total_6)
plt.legend(["H=75", "H=85", "H=95"], loc='upper left')

plt.xlabel("sd_total")
plt.ylabel("mean_total")
plt.savefig('images/3/risk_return_b_put.png')
plt.close()

plt.plot(np.array(range(1, 201)), mean_cost_5)
plt.plot(np.array(range(1, 201)), mean_cost)

plt.plot(np.array(range(1, 201)), mean_cost_6)
plt.legend(["H=75", "H=85", "H=95"], loc='upper left')
plt.xlabel("$N_S$")
plt.ylabel("$\widehat{\mathbb{E}[Friction]}$")

plt.savefig('images/3/mc_b_put.png')
plt.close()

plt.plot(np.array(range(1, 201)), sd_cost_5)
plt.plot(np.array(range(1, 201)), sd_cost)
plt.plot(np.array(range(1, 201)), sd_cost_6)
plt.legend(["H=75", "H=85", "H=95"], loc='upper left')
plt.xlabel("$N_S$")
plt.ylabel("sd_cost")
plt.savefig('images/3/sc_b_put.png')
plt.close()

plt.plot(np.array(range(1, 201)), mean_slippage_5)
plt.plot(np.array(range(1, 201)), mean_slippage)
plt.plot(np.array(range(1, 201)), mean_slippage_6)
plt.legend(["H=75", "H=85", "H=95"], loc='upper left')
plt.xlabel("$N_S$")
plt.ylabel("mean_slippage")
plt.savefig('images/3/ml_b_put.png')
plt.close()

plt.plot(np.array(range(1, 201)), sd_slippage_5)
plt.plot(np.array(range(1, 201)), sd_slippage)
plt.plot(np.array(range(1, 201)), sd_slippage_6)
plt.legend(["H=75", "H=85", "H=95"], loc='upper left')
plt.xlabel("$N_S$")
plt.ylabel("$\widehat{Sd(Slippage)}$")

plt.savefig('images/3/sl_b_put.png')
plt.close()


plt.plot(np.array(range(1, 201)), ((sd_total_5**2 - sd_cost_5 **
         2 - sd_slippage_5**2) / (2 * sd_cost_5 * sd_slippage_5)))
plt.plot(np.array(range(1, 201)), ((sd_total**2 - sd_cost **
         2 - sd_slippage**2) / (2 * sd_cost * sd_slippage)))
plt.plot(np.array(range(1, 201)), ((sd_total_6**2 - sd_cost_6 **
         2 - sd_slippage_6**2) / (2 * sd_cost_6 * sd_slippage_6)))
plt.legend(["H=75", "H=85", "H=95"], loc='upper left')
plt.xlabel("$N_S$")
plt.ylabel("corr_coef")
plt.savefig('images/3/corr_b_put.png')
plt.close()


##################
# cost_function
##################
LAMBDA = 0.52

#
plt.plot(np.array(range(1, 201)), -mean_cost + LAMBDA * sd_slippage)
plt.plot(np.array(range(1, 201)), -mean_cost + LAMBDA * sd_total)

plt.legend(["Cost slippage criterion", "PnL criterion"], loc='upper left')
plt.xlabel("$N_S$")
plt.ylabel("Value of both criteria")
plt.savefig('images/3/cost_func_put.png')
plt.close()

#
plt.plot(np.array(range(1, 201)), -mean_cost_1 + LAMBDA * sd_slippage_1)
plt.plot(np.array(range(1, 201)), -mean_cost_1 + LAMBDA * sd_total_1)

plt.legend(["Cost slippage criterion", "PnL criterion"], loc='upper left')
plt.xlabel("$N_S$")
plt.ylabel("Value of both criteria")
plt.savefig('images/3/cost_func_1_put.png')
plt.close()

#
plt.plot(np.array(range(1, 201)), -mean_cost_2 + LAMBDA * sd_slippage_2)
plt.plot(np.array(range(1, 201)), -mean_cost_2 + LAMBDA * sd_total_2)

plt.legend(["Cost slippage criterion", "PnL criterion"], loc='upper left')
plt.xlabel("$N_S$")
plt.ylabel("Value of both criteria")
plt.savefig('images/3/cost_func_2_put.png')
plt.close()

#
plt.plot(np.array(range(1, 201)), -mean_cost_3 + LAMBDA * sd_slippage_3)
plt.plot(np.array(range(1, 201)), -mean_cost_3 + LAMBDA * sd_total_3)

plt.legend(["Cost slippage criterion", "PnL criterion"], loc='upper left')
plt.xlabel("$N_S$")
plt.ylabel("Value of both criteria")
plt.savefig('images/3/cost_func_3_put.png')
plt.close()

#
plt.plot(np.array(range(1, 201)), -mean_cost_4 + LAMBDA * sd_slippage_4)
plt.plot(np.array(range(1, 201)), -mean_cost_4 + LAMBDA * sd_total_4)

plt.legend(["Cost slippage criterion", "PnL criterion"], loc='upper left')
plt.xlabel("$N_S$")
plt.ylabel("Value of both criteria")
plt.savefig('images/3/cost_func_4_put.png')
plt.close()

plt.plot(np.array(range(1, 201)), -mean_cost_5 + LAMBDA * sd_slippage_5)
plt.plot(np.array(range(1, 201)), -mean_cost_5 + LAMBDA * sd_total_5)

plt.legend(["Cost slippage criterion", "PnL criterion"], loc='upper left')
plt.xlabel("$N_S$")
plt.ylabel("Value of both criteria")
plt.savefig('images/3/cost_func_5_put.png')
plt.close()

plt.plot(np.array(range(1, 201)), -mean_cost_6 + LAMBDA * sd_slippage_6)
plt.plot(np.array(range(1, 201)), -mean_cost_6 + LAMBDA * sd_total_6)

plt.legend(["Cost slippage criterion", "PnL criterion"], loc='upper left')
plt.xlabel("$N_S$")
plt.ylabel("Value of both criteria")
plt.savefig('images/3/cost_func_6_put.png')
plt.close()


y = np.empty(7)
x = np.array([1, 2, 3, 4, 5, 6, 7])

y[0] = (np.argmin(-mean_cost + LAMBDA * sd_slippage) + 1)
y[1] = (np.argmin(-mean_cost_1 + LAMBDA * sd_slippage_1) + 1)
y[2] = (np.argmin(-mean_cost_2 + LAMBDA * sd_slippage_2) + 1)
y[3] = (np.argmin(-mean_cost_3 + LAMBDA * sd_slippage_3) + 1)
y[4] = (np.argmin(-mean_cost_4 + LAMBDA * sd_slippage_4) + 1)
y[5] = (np.argmin(-mean_cost_5 + LAMBDA * sd_slippage_5) + 1)
y[6] = (np.argmin(-mean_cost_6 + LAMBDA * sd_slippage_6) + 1)

plt.plot(x, y, marker='o', linestyle='', markersize=12, label="cost_slippage")

y_total = np.empty(7)
y_total[0] = (np.argmin(-mean_cost + LAMBDA * sd_total) + 1)
y_total[1] = (np.argmin(-mean_cost_1 + LAMBDA * sd_total_1) + 1)
y_total[2] = (np.argmin(-mean_cost_2 + LAMBDA * sd_total_2) + 1)
y_total[3] = (np.argmin(-mean_cost_3 + LAMBDA * sd_total_3) + 1)
y_total[4] = (np.argmin(-mean_cost_4 + LAMBDA * sd_total_4) + 1)
y_total[5] = (np.argmin(-mean_cost_5 + LAMBDA * sd_total_5) + 1)
y_total[6] = (np.argmin(-mean_cost_6 + LAMBDA * sd_total_6) + 1)
plt.plot(x, y_total, marker='o', linestyle='', markersize=12, label="total")

plt.xlabel("param_scenario")
plt.ylabel("optimal_N*")
plt.legend()
plt.savefig('images/3/op_dt_put.png')
plt.close()


# closed form solution

y_closed = np.empty(7)
y_gamma_0 = np.empty(7)
y_gamma_aprox = np.empty(7)

# default

option = Down_and_in_put(100.0, 0.05, 0.25, 0.05, 1,
                         100.0, 85, 0.01, 1000, 1, 87500)

if os.path.isfile("results/3/gamma.pickle"):
    gamma_abs, gamma_square = pickle.load(open("results/3/gamma.pickle", "rb"))
else:
    gamma_abs, gamma_square = gamma_test_barrier_put(
        100.0, 0.05, 0.25, 0.05, 1, 100.0, 85, 0.01)
    pickle.dump((gamma_abs, gamma_square), open(
        "results/3/gamma.pickle", "wb"))


mean_cost_ap = 0.01 * np.sqrt(2 / np.pi) * 0.25 * \
    np.sqrt(np.array(range(1, 201))) * gamma_abs

mean_cost_ap_gamma = 0.01 * np.sqrt(2 / np.pi) * 0.25 * np.sqrt(
    np.array(range(1, 201))) * np.abs(option.gamma(100, 0, False)) * 100**2

mean_cost_ap_avg_gamma = 0.01 * \
    np.sqrt(2 / np.pi) * 0.25 * np.sqrt(np.array(range(1, 201))) * \
    np.abs(option.avg_cash_gamma())

sd_slippage_ap = np.sqrt(
    (0.5 / (np.array(range(1, 201)))) * gamma_square * 0.25**4)

sd_slippage_ap_gamma = np.sqrt(
    (0.5 / (np.array(range(1, 201)))) * ((np.abs(option.gamma(100, 0, False)) * 100**2)**2) * 0.25**4)
sd_slippage_ap_avg_gamma = np.sqrt(
    (0.5 / (np.array(range(1, 201)))) * np.abs(option.avg_cash_gamma())**2 * 0.25**4)


y_closed[0] = round(LAMBDA * np.sqrt(np.pi) * 0.25 / (2 * 0.01))


plt.plot(np.array(range(1, 201)), -mean_cost)
plt.plot(np.array(range(1, 201)), mean_cost_ap)
plt.plot(np.array(range(1, 201)), mean_cost_ap_gamma)
plt.plot(np.array(range(1, 201)), mean_cost_ap_avg_gamma)

plt.xlabel("$N_S$")
plt.ylabel("$\widehat{\mathbb{E}[Friction]}$")

plt.legend(["Path MC", "Gamma MC", "Gamma", "Gamma avg"])
plt.savefig('images/3/mean_aprox_1_put.png')
plt.close()

plt.plot(np.array(range(1, 201)), sd_slippage)
plt.plot(np.array(range(1, 201)), sd_slippage_ap)
plt.plot(np.array(range(1, 201)), sd_slippage_ap_gamma)
plt.plot(np.array(range(1, 201)), sd_slippage_ap_avg_gamma)
plt.xlabel("$N_S$")
plt.ylabel("$\widehat{Sd(Slippage)}$")

plt.legend(["Path MC", "Gamma MC", "Gamma", "Gamma avg"])
plt.savefig('images/3/sd_aprox_1_put.png')
plt.close()
print("0")


print("gamma_1")
####################
# tc
####################
# high tc
option1 = Down_and_in_put(100.0, 0.05, 0.25, 0.05, 1,
                          100.0, 85, 0.05, 1000, 1, 87500)

if os.path.isfile("results/3/gamma_1.pickle"):
    gamma_abs_1, gamma_square_1 = pickle.load(
        open("results/3/gamma_1.pickle", "rb"))
else:
    gamma_abs_1, gamma_square_1 = gamma_test_barrier_put(
        100.0, 0.05, 0.25, 0.05, 1, 100.0, 85, 0.05)
    pickle.dump((gamma_abs_1, gamma_square_1),
                open("results/3/gamma_1.pickle", "wb"))


mean_cost_ap_1 = 0.05 * np.sqrt(2 / np.pi) * 0.25 * \
    np.sqrt(np.array(range(1, 201))) * gamma_abs_1

mean_cost_ap_gamma1 = 0.05 * np.sqrt(2 / np.pi) * 0.25 * np.sqrt(
    np.array(range(1, 201))) * np.abs(option1.gamma(100, 0, False)) * 100**2

mean_cost_ap_avg_gamma1 = 0.05 * \
    np.sqrt(2 / np.pi) * 0.25 * np.sqrt(np.array(range(1, 201))) * \
    np.abs(option1.avg_cash_gamma())
sd_slippage_ap_1 = np.sqrt(
    (0.5 / (np.array(range(1, 201)))) * gamma_square_1 * 0.25**4)
sd_slippage_ap_gamma1 = np.sqrt(
    (0.5 / (np.array(range(1, 201)))) * ((option1.gamma(100, 0, False) * 100**2)**2) * 0.25**4)
sd_slippage_ap_avg_gamma1 = np.sqrt(
    (0.5 / (np.array(range(1, 201)))) * np.abs(option1.avg_cash_gamma())**2 * 0.25**4)

y_closed[1] = round(LAMBDA * np.sqrt(np.pi) * 0.25 / (2 * 0.05))

plt.plot(np.array(range(1, 201)), -mean_cost_1)
plt.plot(np.array(range(1, 201)), mean_cost_ap_1)
plt.plot(np.array(range(1, 201)), mean_cost_ap_gamma1)
plt.plot(np.array(range(1, 201)), mean_cost_ap_avg_gamma1)

plt.xlabel("$N_S$")
plt.ylabel("$\widehat{\mathbb{E}[Friction]}$")

plt.legend(["Path MC", "Gamma MC", "Gamma", "Gamma avg"])
plt.savefig('images/3/mean_aprox_2_put.png')
plt.close()


plt.plot(np.array(range(1, 201)), sd_slippage_1)
plt.plot(np.array(range(1, 201)), sd_slippage_ap_1)
plt.plot(np.array(range(1, 201)), sd_slippage_ap_gamma1)
plt.plot(np.array(range(1, 201)), sd_slippage_ap_avg_gamma1)
plt.xlabel("$N_S$")
plt.ylabel("$\widehat{Sd(Slippage)}$")

plt.legend(["Path MC", "Gamma MC", "Gamma", "Gamma avg"])
plt.savefig('images/3/sd_aprox_2_put.png')
plt.close()
print("gamma_2")

# low tc
option2 = Down_and_in_put(100.0, 0.05, 0.25, 0.05, 1,
                          100.0, 85, 0.002, 1000, 1, 87500)


if os.path.isfile("results/3/gamma_2.pickle"):
    gamma_abs_2, gamma_square_2 = pickle.load(
        open("results/3/gamma_2.pickle", "rb"))
else:
    gamma_abs_2, gamma_square_2 = gamma_test_barrier_put(
        100.0, 0.05, 0.25, 0.05, 1, 100.0, 85, 0.002)
    pickle.dump((gamma_abs_2, gamma_square_2),
                open("results/3/gamma_2.pickle", "wb"))


mean_cost_ap_2 = 0.002 * np.sqrt(2 / np.pi) * 0.25 * \
    np.sqrt(np.array(range(1, 201))) * gamma_abs_2

mean_cost_ap_gamma2 = 0.002 * np.sqrt(2 / np.pi) * 0.25 * np.sqrt(
    np.array(range(1, 201))) * np.abs(option2.gamma(100, 0, False)) * 100**2
mean_cost_ap_avg_gamma2 = 0.002 * \
    np.sqrt(2 / np.pi) * 0.25 * np.sqrt(np.array(range(1, 201))) * \
    np.abs(option2.avg_cash_gamma())
sd_slippage_ap_2 = np.sqrt(
    (0.5 / (np.array(range(1, 201)))) * gamma_square_2 * 0.25**4)
sd_slippage_ap_gamma2 = np.sqrt(
    (0.5 / (np.array(range(1, 201)))) * ((np.abs(option2.gamma(100, 0, False) * 100**2))**2) * 0.25**4)
sd_slippage_ap_avg_gamma2 = np.sqrt(
    (0.5 / (np.array(range(1, 201)))) * np.abs(option2.avg_cash_gamma())**2 * 0.25**4)

y_closed[2] = round(LAMBDA * np.sqrt(np.pi) * 0.25 / (2 * 0.002))
print("gamma_3")

plt.plot(np.array(range(1, 201)), -mean_cost_2)
plt.plot(np.array(range(1, 201)), mean_cost_ap_2)
plt.plot(np.array(range(1, 201)), mean_cost_ap_gamma2)
plt.plot(np.array(range(1, 201)), mean_cost_ap_avg_gamma2)

plt.xlabel("$N_S$")
plt.ylabel("$\widehat{\mathbb{E}[Friction]}$")

plt.legend(["Path MC", "Gamma MC", "Gamma", "Gamma avg"])
plt.savefig('images/3/mean_aprox_3_put.png')
plt.close()


plt.plot(np.array(range(1, 201)), sd_slippage_2)
plt.plot(np.array(range(1, 201)), sd_slippage_ap_2)
plt.plot(np.array(range(1, 201)), sd_slippage_ap_gamma2)
plt.plot(np.array(range(1, 201)), sd_slippage_ap_avg_gamma2)
plt.xlabel("$N_S$")
plt.ylabel("$\widehat{Sd(Slippage)}$")

plt.legend(["Path MC", "Gamma MC", "Gamma", "Gamma avg"])
plt.savefig('images/3/sd_aprox_3_put.png')
plt.close()
###############
# VOL
###############
option3 = Down_and_in_put(100.0, 0.05, 0.1, 0.05, 1,
                          100.0, 85, 0.01, 1000, 1, 87500)


# low vol

if os.path.isfile("results/3/gamma_3.pickle"):
    gamma_abs_3, gamma_square_3 = pickle.load(
        open("results/3/gamma_3.pickle", "rb"))
else:
    gamma_abs_3, gamma_square_3 = gamma_test_barrier_put(
        100.0, 0.05, 0.1, 0.05, 1, 100.0, 85, 0.01)
    pickle.dump((gamma_abs_3, gamma_square_3),
                open("results/3/gamma_3.pickle", "wb"))


mean_cost_ap_3 = 0.01 * np.sqrt(2 / np.pi) * 0.1 * \
    np.sqrt(np.array(range(1, 201))) * gamma_abs_3

mean_cost_ap_gamma_3 = 0.01 * np.sqrt(2 / np.pi) * 0.1 * np.sqrt(
    np.array(range(1, 201))) * np.abs(option3.gamma(100, 0, False)) * 100**2
mean_cost_ap_avg_gamma_3 = 0.01 * \
    np.sqrt(2 / np.pi) * 0.1 * np.sqrt(np.array(range(1, 201))) * \
    np.abs(option3.avg_cash_gamma())
sd_slippage_ap_3 = np.sqrt(
    (0.5 / (np.array(range(1, 201)))) * gamma_square_3 * 0.1**4)
sd_slippage_ap_gamma_3 = np.sqrt(
    (0.5 / (np.array(range(1, 201)))) * ((np.abs(option3.gamma(100, 0, False) * 100**2))**2) * 0.1**4)
sd_slippage_ap_avg_gamma_3 = np.sqrt(
    (0.5 / (np.array(range(1, 201)))) * np.abs(option3.avg_cash_gamma())**2 * 0.1**4)

y_closed[3] = round(LAMBDA * 1 * np.sqrt(np.pi) * 0.1 / (2 * 0.01))

plt.plot(np.array(range(1, 201)), -mean_cost_3)
plt.plot(np.array(range(1, 201)), mean_cost_ap_3)
plt.plot(np.array(range(1, 201)), mean_cost_ap_gamma_3)
plt.plot(np.array(range(1, 201)), mean_cost_ap_avg_gamma_3)

plt.xlabel("$N_S$")
plt.ylabel("$\widehat{\mathbb{E}[Friction]}$")

plt.legend(["Path MC", "Gamma MC", "Gamma", "Gamma avg"])
plt.savefig('images/3/mean_aprox_4_put.png')
plt.close()

plt.plot(np.array(range(1, 201)), sd_slippage_3)
plt.plot(np.array(range(1, 201)), sd_slippage_ap_3)
plt.plot(np.array(range(1, 201)), sd_slippage_ap_gamma_3)
plt.plot(np.array(range(1, 201)), sd_slippage_ap_avg_gamma_3)
plt.xlabel("$N_S$")
plt.ylabel("$\widehat{Sd(Slippage)}$")

plt.legend(["Path MC", "Gamma MC", "Gamma", "Gamma avg"])
plt.savefig('images/3/sd_aprox_4_put.png')
plt.close()
print("4")


print("gamma_4")

# high vol
option4 = Down_and_in_put(100.0, 0.05, 0.5, 0.05, 1,
                          100.0, 85, 0.01, 1000, 1, 87500)


if os.path.isfile("results/3/gamma_4.pickle"):
    gamma_abs_4, gamma_square_4 = pickle.load(
        open("results/3/gamma_4.pickle", "rb"))
else:
    gamma_abs_4, gamma_square_4 = gamma_test_barrier_put(
        100.0, 0.05, 0.5, 0.05, 1, 100.0, 85, 0.01)
    pickle.dump((gamma_abs_4, gamma_square_4),
                open("results/3/gamma_4.pickle", "wb"))


mean_cost_ap_4 = 0.01 * np.sqrt(2 / np.pi) * 0.5 * \
    np.sqrt(np.array(range(1, 201))) * gamma_abs_4
mean_cost_ap_gamma4 = 0.01 * np.sqrt(2 / np.pi) * 0.5 * np.sqrt(
    np.array(range(1, 201))) * np.abs(option4.gamma(100, 0, False)) * 100**2
mean_cost_ap_avg_gamma4 = 0.01 * \
    np.sqrt(2 / np.pi) * 0.5 * np.sqrt(np.array(range(1, 201))) * \
    np.abs(option4.avg_cash_gamma())
sd_slippage_ap_4 = np.sqrt(
    (0.5 / (np.array(range(1, 201)))) * gamma_square_4 * 0.5**4)
sd_slippage_ap_gamma4 = np.sqrt(
    (0.5 / (np.array(range(1, 201)))) * ((np.abs(option4.gamma(100, 0, False)) * 100**2)**2) * 0.5**4)
sd_slippage_ap_avg_gamma4 = np.sqrt(
    (0.5 / (np.array(range(1, 201)))) * np.abs(option4.avg_cash_gamma())**2 * 0.5**4)

y_closed[4] = round(LAMBDA * np.sqrt(np.pi) * 0.5 / (2 * 0.01))
plt.plot(np.array(range(1, 201)), -mean_cost_4)
plt.plot(np.array(range(1, 201)), mean_cost_ap_4)
plt.plot(np.array(range(1, 201)), mean_cost_ap_gamma4)
plt.plot(np.array(range(1, 201)), mean_cost_ap_avg_gamma4)

plt.xlabel("$N_S$")
plt.ylabel("$\widehat{\mathbb{E}[Friction]}$")

plt.legend(["Path MC", "Gamma MC", "Gamma", "Gamma avg"])
plt.savefig('images/3/mean_aprox_5_put.png')
plt.close()

plt.plot(np.array(range(1, 201)), sd_slippage_4)
plt.plot(np.array(range(1, 201)), sd_slippage_ap_4)
plt.plot(np.array(range(1, 201)), sd_slippage_ap_gamma4)
plt.plot(np.array(range(1, 201)), sd_slippage_ap_avg_gamma4)
plt.xlabel("$N_S$")
plt.ylabel("$\widehat{Sd(Slippage)}$")

plt.legend(["Path_MC", "Gamma_MC", "Gamma", "Gamma_avg", "Gamma_avg_d"])
plt.savefig('images/3/sd_aprox_5_put.png')
plt.close()


print("gamma_5")

###############
# Barrier
###############

# low barrier
option5 = Down_and_in_put(100.0, 0.05, 0.25, 0.05, 1,
                          100.0, 75, 0.05, 1000, 1, 87500)


if os.path.isfile("results/3/gamma_5.pickle"):
    gamma_abs_5, gamma_square_5 = pickle.load(
        open("results/3/gamma_5.pickle", "rb"))
else:

    gamma_abs_5, gamma_square_5 = gamma_test_barrier_put(
        100.0, 0.05, 0.25, 0.05, 1, 100.0, 75, 0.01)

    pickle.dump((gamma_abs_5, gamma_square_5),
                open("results/3/gamma_5.pickle", "wb"))

mean_cost_ap_5 = 0.01 * np.sqrt(2 / np.pi) * 0.25 * \
    np.sqrt(np.array(range(1, 201))) * gamma_abs_5

mean_cost_ap_gamma_5 = 0.01 * np.sqrt(2 / np.pi) * 0.25 * np.sqrt(
    np.array(range(1, 201))) * np.abs(option5.gamma(100, 0, False)) * 100**2
mean_cost_ap_avg_gamma_5 = 0.01 * \
    np.sqrt(2 / np.pi) * 0.25 * np.sqrt(np.array(range(1, 201))) * \
    np.abs(option5.avg_cash_gamma())

sd_slippage_ap_5 = np.sqrt(
    (0.5 / (np.array(range(1, 201)))) * gamma_square_5 * 0.25**4)
sd_slippage_ap_gamma_5 = np.sqrt(
    (0.5 / (np.array(range(1, 201)))) * ((np.abs(option5.gamma(100, 0, False)) * 100**2)**2) * 0.25**4)
sd_slippage_ap_avg_gamma_5 = np.sqrt(
    (0.5 / (np.array(range(1, 201)))) * np.abs(option5.avg_cash_gamma())**2 * 0.25**4)

y_closed[5] = round(LAMBDA * 1 * np.sqrt(np.pi) * 0.25 / (2 * 0.01))



plt.plot(np.array(range(1, 201)), -mean_cost_5)
plt.plot(np.array(range(1, 201)), mean_cost_ap_5)
plt.plot(np.array(range(1, 201)), mean_cost_ap_gamma_5)
plt.plot(np.array(range(1, 201)), mean_cost_ap_avg_gamma_5)

plt.xlabel("$N_S$")
plt.ylabel("$\widehat{\mathbb{E}[Friction]}$")

plt.legend(["Path MC", "Gamma MC", "Gamma", "Gamma avg"])
plt.savefig('images/3/mean_aprox_6_put.png')
plt.close()

plt.plot(np.array(range(1, 201)), sd_slippage_5)
plt.plot(np.array(range(1, 201)), sd_slippage_ap_5)
plt.plot(np.array(range(1, 201)), sd_slippage_ap_gamma_5)
plt.plot(np.array(range(1, 201)), sd_slippage_ap_avg_gamma_5)
plt.xlabel("$N_S$")
plt.ylabel("$\widehat{Sd(Slippage)}$")

plt.legend(["Path MC", "Gamma MC", "Gamma", "Gamma avg"])
plt.savefig('images/3/sd_aprox_6_put.png')
plt.close()
print("0")


print("gamma_5")

# high barrier
option6 = Down_and_in_put(100.0, 0.05, 0.25, 0.05, 1,
                          100.0, 95, 0.05, 1000, 1, 87500)


if os.path.isfile("results/3/gamma_6.pickle"):
    gamma_abs_6, gamma_square_6 = pickle.load(
        open("results/3/gamma_6.pickle", "rb"))
else:

    gamma_abs_6, gamma_square_6 = gamma_test_barrier_put(
        100.0, 0.05, 0.25, 0.05, 1, 100.0, 95, 0.01)

    pickle.dump((gamma_abs_6, gamma_square_6),
                open("results/3/gamma_6.pickle", "wb"))


mean_cost_ap_6 = 0.01 * np.sqrt(2 / np.pi) * 0.25 * \
    np.sqrt(np.array(range(1, 201))) * gamma_abs_6

mean_cost_ap_gamma_6 = 0.01 * np.sqrt(2 / np.pi) * 0.25 * np.sqrt(
    np.array(range(1, 201))) * np.abs(option6.gamma(100, 0, False)) * 100**2
mean_cost_ap_avg_gamma_6 = 0.01 * \
    np.sqrt(2 / np.pi) * 0.25 * np.sqrt(np.array(range(1, 201))) * \
    np.abs(option6.avg_cash_gamma())

sd_slippage_ap_6 = np.sqrt(
    (0.5 / (np.array(range(1, 201)))) * gamma_square_6 * 0.25**4)
sd_slippage_ap_gamma_6 = np.sqrt(
    (0.5 / (np.array(range(1, 201)))) * ((np.abs(option6.gamma(100, 0, False)) * 100**2)**2) * 0.25**4)
sd_slippage_ap_avg_gamma_6 = np.sqrt(
    (0.5 / (np.array(range(1, 201)))) * np.abs(option6.avg_cash_gamma())**2 * 0.25**4)


y_closed[6] = round(LAMBDA * np.sqrt(np.pi) * 0.25 / (2 * 0.01))

print("gamma_6")
plt.plot(np.array(range(1, 201)), -mean_cost_6)
plt.plot(np.array(range(1, 201)), mean_cost_ap_6)
plt.plot(np.array(range(1, 201)), mean_cost_ap_gamma_6)
plt.plot(np.array(range(1, 201)), mean_cost_ap_avg_gamma_6)

plt.xlabel("$N_S$")
plt.ylabel("$\widehat{\mathbb{E}[Friction]}$")

plt.legend(["Path MC", "Gamma MC", "Gamma", "Gamma avg"])
plt.savefig('images/3/mean_aprox_7_put.png')
plt.close()

plt.plot(np.array(range(1, 201)), sd_slippage_6)
plt.plot(np.array(range(1, 201)), sd_slippage_ap_6)
plt.plot(np.array(range(1, 201)), sd_slippage_ap_gamma_6)
plt.plot(np.array(range(1, 201)), sd_slippage_ap_avg_gamma_6)
plt.xlabel("$N_S$")
plt.ylabel("$\widehat{Sd(Slippage)}$")

plt.legend(["Path MC", "Gamma MC", "Gamma", "Gamma avg"])
plt.savefig('images/3/sd_aprox_7_put.png')
plt.close()


plt.plot(np.array(range(1, 201)),  np.sqrt(gamma_square_2)/gamma_abs_2)
plt.plot(np.array(range(1, 201)), np.sqrt(gamma_square)/gamma_abs)
plt.plot(np.array(range(1, 201)),   np.sqrt(gamma_square_1)/gamma_abs_1)
plt.legend(["Low TC", "Default", "High TC"], loc='upper left')
plt.legend(["Low TC", "Default", "High TC"], loc='upper left')
plt.xlabel("$N_S$")
plt.ylabel("Value of the ratio")
plt.savefig('images/3/ratio_k_put.png')
plt.close()


plt.plot(np.array(range(1, 201)),   (np.sqrt(gamma_square_3)/gamma_abs_3))
plt.plot(np.array(range(1, 201)),   np.sqrt(gamma_square)/gamma_abs)
plt.plot(np.array(range(1, 201)),  (np.sqrt(gamma_square_4)/gamma_abs_4))
plt.legend(["Low vol", "Default", "High vol"], loc='upper left')

plt.xlabel("$N_S$")
plt.ylabel("Value of the ratio")
plt.savefig('images/3/ratio_v_put.png')
plt.close()



plt.plot(np.array(range(1, 201)),  (np.sqrt(gamma_square_5)/gamma_abs_5))
plt.plot(np.array(range(1, 201)),  (np.sqrt(gamma_square)/gamma_abs))
plt.plot(np.array(range(1, 201)),  (np.sqrt(gamma_square_6)/gamma_abs_6))
plt.legend(["H=75", "H=85", "H=95"], loc='upper left')
plt.xlabel("$N_S$")
plt.ylabel("Value of the ratio")
plt.savefig('images/3/ratio_h_put.png')
plt.close()


y_gamma_aux = np.zeros(7)

y_gamma_aux[0] = (np.argmin(mean_cost_ap + LAMBDA * sd_slippage_ap) + 1)
y_gamma_aux[1] = (np.argmin(mean_cost_ap_1 + LAMBDA * sd_slippage_ap_1) + 1)
y_gamma_aux[2] = (np.argmin(mean_cost_ap_2 + LAMBDA * sd_slippage_ap_2) + 1)
y_gamma_aux[3] = (np.argmin(mean_cost_ap_3 + LAMBDA * sd_slippage_ap_3) + 1)
y_gamma_aux[4] = (np.argmin(mean_cost_ap_4 + LAMBDA * sd_slippage_ap_4) + 1)
y_gamma_aux[5] = (np.argmin(mean_cost_ap_5 + LAMBDA * sd_slippage_ap_5) + 1)
y_gamma_aux[6] = (np.argmin(mean_cost_ap_6 + LAMBDA * sd_slippage_ap_6) + 1)
print("gamma_aux", file=result_file)
print(y_gamma_aux, file=result_file)
print(" ", file=result_file)
y_gamma = np.zeros(7)

y_gamma[0] = (np.argmin(mean_cost_ap[1:] + LAMBDA * sd_slippage_ap[1:]) + 2)
y_gamma[1] = (np.argmin(mean_cost_ap_1[1:] + LAMBDA * sd_slippage_ap_1[1:]) + 2)
y_gamma[2] = (np.argmin(mean_cost_ap_2[1:] + LAMBDA * sd_slippage_ap_2[1:]) + 2)
y_gamma[3] = (np.argmin(mean_cost_ap_3[1:] + LAMBDA * sd_slippage_ap_3[1:]) + 2)
y_gamma[4] = (np.argmin(mean_cost_ap_4[1:] + LAMBDA * sd_slippage_ap_4[1:]) + 2)
y_gamma[5] = (np.argmin(mean_cost_ap_5[1:] + LAMBDA * sd_slippage_ap_5[1:]) + 2)
y_gamma[6] = (np.argmin(mean_cost_ap_6[1:] + LAMBDA * sd_slippage_ap_6[1:]) + 2)

plt.plot(x, y_gamma, marker='o', linestyle='',
         markersize=12, label="True_gamma")

plt.plot(
    x,
    y_closed,
    marker='o',
    linestyle='',
    markersize=12,
    label="Approximation_gamma")

plt.xlabel("param_scenario")
plt.ylabel("optimal_N*")
plt.legend()
plt.savefig('images/3/op_aprox_put.png')
plt.close()


plt.plot(x, y, marker='o', linestyle='', markersize=12, label="cost_slippage")

plt.plot(x, y_total, marker='o', linestyle='', markersize=12, label="total")

plt.plot(x, y_gamma, marker='o', linestyle='',
         markersize=12, label="True_gamma")

plt.plot(
    x,
    y_closed,
    marker='o',
    linestyle='',
    markersize=12,
    label="Approximation_gamma")

plt.xlabel("param_scenario")
plt.ylabel("optimal_N*")
plt.legend()
plt.savefig('images/3/op_dt_all_put.png')
plt.close()


plt.plot(x, y, marker='o', linestyle='', markersize=12, label="cost_slippage")

plt.plot(x, y_gamma, marker='o', linestyle='',
         markersize=12, label="True_gamma")

plt.plot(
    x,
    y_closed,
    marker='o',
    linestyle='',
    markersize=12,
    label="Approximation_gamma")

plt.xlabel("param_scenario")
plt.ylabel("optimal_N*")
plt.legend()
plt.savefig('images/3/op_dt_all_total_put.png')
plt.close()


print(f"Time used:{time.time()-time0}", file=result_file)

print("Total UF", file=result_file)
print(x, file=result_file)
print(y_total, file=result_file)
print(f"{mean_cost[int(y_total[0]-1)]},{mean_cost_1[int(y_total[1]-1)]},{mean_cost_2[int(y_total[2]-1)]},\
    {mean_cost_3[int(y_total[3]-1)]},{mean_cost_4[int(y_total[4]-1)]},{mean_cost_5[int(y_total[5]-1)]},{mean_cost_6[int(y_total[6]-1)]}", file=result_file)
print(f"{sd_total[int(y_total[0]-1)]},{sd_total_1[int(y_total[1]-1)]},{sd_total_2[int(y_total[2]-1)]},\
    {sd_total_3[int(y_total[3]-1)]},{sd_total_4[int(y_total[4]-1)]},{sd_total_5[int(y_total[5]-1)]},{sd_total_6[int(y_total[6]-1)]}", file=result_file)


print("cost_slippage UF", file=result_file)
print(x, file=result_file)
print(y, file=result_file)
print(f"{mean_cost[int(y[0]-1)]},{mean_cost_1[int(y[1]-1)]},{mean_cost_2[int(y[2]-1)]},\
    {mean_cost_3[int(y[3]-1)]},{mean_cost_4[int(y[4]-1)]},{mean_cost_5[int(y[5]-1)]},{mean_cost_6[int(y[6]-1)]}", file=result_file)
print(f"{sd_slippage[int(y[0]-1)]},{sd_slippage_1[int(y[1]-1)]},{sd_slippage_2[int(y[2]-1)]},\
    {sd_slippage_3[int(y[3]-1)]},{sd_slippage_4[int(y[4]-1)]},{sd_slippage_5[int(y[5]-1)]},{sd_slippage_6[int(y[6]-1)]}", file=result_file)


print("True gamma Aprox", file=result_file)
print(x, file=result_file)
print(y_gamma, file=result_file)

print(f"{mean_cost_ap[int(y_gamma[0]-1)]},{mean_cost_ap_1[int(y_gamma[1]-1)]},{mean_cost_ap_2[int(y_gamma[2]-1)]},\
    {mean_cost_ap_3[int(y_gamma[3]-1)]},{mean_cost_ap_4[int(y_gamma[4]-1)]},{mean_cost_ap_5[int(y_gamma[6]-1)]},{mean_cost_ap_6[int(y_gamma[6]-1)]}", file=result_file)
print(f"{sd_slippage_ap[int(y_gamma[0]-1)]},{sd_slippage_ap_1[int(y_gamma[1]-1)]},{sd_slippage_ap_2[int(y_gamma[2]-1)]},\
    {sd_slippage_ap_3[int(y_gamma[3]-1)]},{sd_slippage_ap_4[int(y_gamma[4]-1)]},{sd_slippage_ap_5[int(y_gamma[5]-1)]},{sd_slippage_ap_6[int(y_gamma[6]-1)]}", file=result_file)

print("closed form t0", file=result_file)

print(x, file=result_file)
print(y_closed, file=result_file)
print(f"{mean_cost_ap_gamma[int(y_closed[0]-1)]},{mean_cost_ap_gamma1[int(y_closed[1]-1)]},{mean_cost_ap_gamma2[int(y_closed[2]-1)]},\
    {mean_cost_ap_gamma_3[int(y_closed[3]-1)]},{mean_cost_ap_gamma4[int(y_closed[4]-1)]},{mean_cost_ap_gamma_5[int(y_closed[6]-1)]},{mean_cost_ap_gamma_6[int(y_closed[6]-1)]}", file=result_file)
print(f"{sd_slippage_ap_gamma[int(y_closed[0]-1)]},{sd_slippage_ap_gamma1[int(y_closed[1]-1)]},{sd_slippage_ap_gamma2[int(y_closed[2]-1)]},\
    {sd_slippage_ap_gamma_3[int(y_closed[3]-1)]},{sd_slippage_ap_gamma4[int(y_closed[4]-1)]},{sd_slippage_ap_gamma_5[int(y_closed[5]-1)]},{sd_slippage_ap_gamma_6[int(y_closed[6]-1)]}", file=result_file)

print("closed form avg", file=result_file)

print(x, file=result_file)
print(y_closed, file=result_file)
print(f"{mean_cost_ap_avg_gamma[int(y_closed[0]-1)]},{mean_cost_ap_avg_gamma1[int(y_closed[1]-1)]},{mean_cost_ap_avg_gamma2[int(y_closed[2]-1)]},\
    {mean_cost_ap_avg_gamma_3[int(y_closed[3]-1)]},{mean_cost_ap_avg_gamma4[int(y_closed[4]-1)]},{mean_cost_ap_avg_gamma_5[int(y_closed[6]-1)]},{mean_cost_ap_avg_gamma_6[int(y_closed[6]-1)]}", file=result_file)
print(f"{sd_slippage_ap_avg_gamma[int(y_closed[0]-1)]},{sd_slippage_ap_avg_gamma1[int(y_closed[1]-1)]},{sd_slippage_ap_avg_gamma2[int(y_closed[2]-1)]},\
    {sd_slippage_ap_avg_gamma_3[int(y_closed[3]-1)]},{sd_slippage_ap_avg_gamma4[int(y_closed[4]-1)]},{sd_slippage_ap_avg_gamma_5[int(y_closed[5]-1)]},{sd_slippage_ap_avg_gamma_6[int(y_closed[6]-1)]}", file=result_file)

result_file.close()

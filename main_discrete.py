from re import S
import time
from tools_stable.class_seperate import Short_european_call_black_sholes_fixed_time_sep, risk_return_fixed_time_sep, gamma_test_fixed_time
import numpy as np
import matplotlib.pyplot as plt
import os
import pickle
N_PATHS = 100000
plt.rcParams['text.usetex'] = True
plt.rcParams.update({
    'text.latex.preamble': r'\usepackage{amsfonts}'
})

time0 = time.time()

result_file = open("results/1/result_1.txt", mode="w")
# grid_search

# default

if os.path.isfile("results/1/MC.pickle"):
    mean_total, sd_total, mean_cost, sd_cost, mean_slippage, sd_slippage = pickle.load(
        open("results/1/MC.pickle", "rb"))
else:
    mean_total, sd_total, mean_cost, sd_cost, mean_slippage, sd_slippage = risk_return_fixed_time_sep(
        100.0, 0.05, 0.25, 0.05, 1, 100.0, 0.01, N_PATHS)
    pickle.dump((mean_total, sd_total, mean_cost, sd_cost,
                mean_slippage, sd_slippage), open("results/1/MC.pickle", "wb"))


print("default")
####################
# tc
####################
# high tc


if os.path.isfile("results/1/MC_1.pickle"):
    mean_total_1, sd_total_1, mean_cost_1, sd_cost_1, mean_slippage_1, sd_slippage_1 = pickle.load(
        open("results/1/MC_1.pickle", "rb"))
else:
    mean_total_1, sd_total_1, mean_cost_1, sd_cost_1, mean_slippage_1, sd_slippage_1 = risk_return_fixed_time_sep(
        100.0, 0.05, 0.25, 0.05, 1, 100.0, 0.05, N_PATHS)
    pickle.dump((mean_total_1, sd_total_1, mean_cost_1, sd_cost_1,
                mean_slippage_1, sd_slippage_1), open("results/1/MC_1.pickle", "wb"))


print("high_tc")

# low tc
if os.path.isfile("results/1/MC_2.pickle"):
    mean_total_2, sd_total_2, mean_cost_2, sd_cost_2, mean_slippage_2, sd_slippage_2 = pickle.load(
        open("results/1/MC_2.pickle", "rb"))
else:
    mean_total_2, sd_total_2, mean_cost_2, sd_cost_2, mean_slippage_2, sd_slippage_2 = risk_return_fixed_time_sep(
        100.0, 0.05, 0.25, 0.05, 1, 100.0, 0.002, N_PATHS)
    pickle.dump((mean_total_2, sd_total_2, mean_cost_2, sd_cost_2,
                mean_slippage_2, sd_slippage_2), open("results/1/MC_2.pickle", "wb"))


print("low_tc")


plt.plot(sd_total_2, mean_total_2)
plt.plot(sd_total, mean_total)
plt.plot(sd_total_1, mean_total_1)

plt.legend(["Low TC", "Default", "High TC"], loc='upper left')
plt.xlabel("sd_total")
plt.ylabel("mean_total")
# plt.show()
plt.savefig('images/1/risk_return_k.png')
plt.close()

plt.plot(np.array(range(1, 201)), mean_cost_2)
plt.plot(np.array(range(1, 201)), mean_cost)
plt.plot(np.array(range(1, 201)), mean_cost_1)

plt.legend(["Low TC", "Default", "High TC"], loc='upper left')


plt.xlabel("$N_S$")
plt.ylabel("$\widehat{\mathbb{E}[Friction]}$")
plt.savefig('images/1/mc_k.png')
plt.close()

plt.plot(np.array(range(1, 201)), sd_cost_2)
plt.plot(np.array(range(1, 201)), sd_cost)
plt.plot(np.array(range(1, 201)), sd_cost_1)

plt.legend(["Low TC", "Default", "High TC"], loc='upper left')
plt.xlabel("$N_S$")
plt.ylabel("sd_cost")
plt.savefig('images/1/sc_k.png')
plt.close()

plt.plot(np.array(range(1, 201)), sd_slippage_2)
plt.plot(np.array(range(1, 201)), sd_slippage)
plt.plot(np.array(range(1, 201)), sd_slippage_1)
plt.legend(["Low TC", "Default", "High TC"], loc='upper left')
plt.xlabel("$N_S$")
plt.ylabel("$\widehat{Sd(Slippage)}$")

plt.savefig('images/1/sl_k.png')
plt.close()


plt.plot(np.array(range(1, 201)), mean_slippage_2)
plt.plot(np.array(range(1, 201)), mean_slippage)
plt.plot(np.array(range(1, 201)), mean_slippage_1)

plt.legend(["Low TC", "Default", "High TC"], loc='upper left')
plt.xlabel("$N_S$")
plt.ylabel("mean_slippage")
plt.savefig('images/1/ml_k.png')
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
plt.savefig('images/1/corr_k.png')
plt.close()


###############
# VOL
###############

# low vol
if os.path.isfile("results/1/MC_3.pickle"):
    mean_total_3, sd_total_3, mean_cost_3, sd_cost_3, mean_slippage_3, sd_slippage_3 = pickle.load(
        open("results/1/MC_3.pickle", "rb"))
else:
    mean_total_3, sd_total_3, mean_cost_3, sd_cost_3, mean_slippage_3, sd_slippage_3 = risk_return_fixed_time_sep(
        100.0, 0.05, 0.1, 0.05, 1, 100.0, 0.01, N_PATHS)
    pickle.dump((mean_total_3, sd_total_3, mean_cost_3, sd_cost_3,
                mean_slippage_3, sd_slippage_3), open("results/1/MC_3.pickle", "wb"))


print("low_vol")
# high vol

if os.path.isfile("results/1/MC_4.pickle"):
    mean_total_4, sd_total_4, mean_cost_4, sd_cost_4, mean_slippage_4, sd_slippage_4 = pickle.load(
        open("results/1/MC_4.pickle", "rb"))
else:
    mean_total_4, sd_total_4, mean_cost_4, sd_cost_4, mean_slippage_4, sd_slippage_4 = risk_return_fixed_time_sep(
        100.0, 0.05, 0.5, 0.05, 1, 100.0, 0.01, N_PATHS)
    pickle.dump((mean_total_4, sd_total_4, mean_cost_4, sd_cost_4,
                mean_slippage_4, sd_slippage_4), open("results/1/MC_4.pickle", "wb"))


print("high_vol")


plt.plot(sd_total_3, mean_total_3)
plt.plot(sd_total, mean_total)
plt.plot(sd_total_4, mean_total_4)
plt.legend(["Low vol", "Default", "High vol"], loc='upper left')

plt.xlabel("sd_total")
plt.ylabel("mean_total")
plt.savefig('images/1/risk_return_v.png')
plt.close()

plt.plot(np.array(range(1, 201)), mean_cost_3)
plt.plot(np.array(range(1, 201)), mean_cost)

plt.plot(np.array(range(1, 201)), mean_cost_4)
plt.legend(["Low vol", "Default", "High vol"], loc='upper left')

plt.xlabel("$N_S$")
plt.ylabel("$\widehat{\mathbb{E}[Friction]}$")

plt.savefig('images/1/mc_v.png')
plt.close()

plt.plot(np.array(range(1, 201)), sd_cost_3)
plt.plot(np.array(range(1, 201)), sd_cost)
plt.plot(np.array(range(1, 201)), sd_cost_4)
plt.legend(["Low vol", "Default", "High vol"], loc='upper left')

plt.xlabel("$N_S$")
plt.ylabel("sd_cost")
plt.savefig('images/1/sc_v.png')
plt.close()

plt.plot(np.array(range(1, 201)), mean_slippage_3)
plt.plot(np.array(range(1, 201)), mean_slippage)
plt.plot(np.array(range(1, 201)), mean_slippage_4)
plt.legend(["Low vol", "Default", "High vol"], loc='upper left')

plt.xlabel("$N_S$")
plt.ylabel("mean_slippage")
plt.savefig('images/1/ml_v.png')
plt.close()

plt.plot(np.array(range(1, 201)), sd_slippage_3)
plt.plot(np.array(range(1, 201)), sd_slippage)
plt.plot(np.array(range(1, 201)), sd_slippage_4)
plt.legend(["Low vol", "Default", "High vol"], loc='upper left')

plt.xlabel("$N_S$")
plt.ylabel("$\widehat{Sd(Slippage)}$")

plt.savefig('images/1/sl_v.png')
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
plt.savefig('images/1/corr_v.png')
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
plt.savefig('images/1/cost_func.png')
plt.close()

#
plt.plot(np.array(range(1, 201)), -mean_cost_1 + LAMBDA * sd_slippage_1)
plt.plot(np.array(range(1, 201)), -mean_cost_1 + LAMBDA * sd_total_1)

plt.legend(["Cost slippage criterion", "PnL criterion"], loc='upper left')
plt.xlabel("$N_S$")
plt.ylabel("Value of both criteria")
plt.savefig('images/1/cost_func_1.png')
plt.close()

#
plt.plot(np.array(range(1, 201)), -mean_cost_2 + LAMBDA * sd_slippage_2)
plt.plot(np.array(range(1, 201)), -mean_cost_2 + LAMBDA * sd_total_2)

plt.legend(["Cost slippage criterion", "PnL criterion"], loc='upper left')
plt.xlabel("$N_S$")
plt.ylabel("Value of both criteria")
plt.savefig('images/1/cost_func_2.png')
plt.close()

#
plt.plot(np.array(range(1, 201)), -mean_cost_3 + LAMBDA * sd_slippage_3)
plt.plot(np.array(range(1, 201)), -mean_cost_3 + LAMBDA * sd_total_3)

plt.legend(["Cost slippage criterion", "PnL criterion"], loc='upper left')
plt.xlabel("$N_S$")
plt.ylabel("Value of both criteria")
plt.savefig('images/1/cost_func_3.png')
plt.close()

#
plt.plot(np.array(range(1, 201)), -mean_cost_4 + LAMBDA * sd_slippage_4)
plt.plot(np.array(range(1, 201)), -mean_cost_4 + LAMBDA * sd_total_4)

plt.legend(["Cost slippage criterion", "PnL criterion"], loc='upper left')
plt.xlabel("$N_S$")
plt.ylabel("Value of both criteria")
plt.savefig('images/1/cost_func_4.png')
plt.close()


y = np.empty(5)
x = np.array([1, 2, 3, 4, 5])

y[0] = (np.argmin(-mean_cost + LAMBDA * sd_slippage) + 1)
y[1] = (np.argmin(-mean_cost_1 + LAMBDA * sd_slippage_1) + 1)
y[2] = (np.argmin(-mean_cost_2 + LAMBDA * sd_slippage_2) + 1)
y[3] = (np.argmin(-mean_cost_3 + LAMBDA * sd_slippage_3) + 1)
y[4] = (np.argmin(-mean_cost_4 + LAMBDA * sd_slippage_4) + 1)
plt.plot(x, y, marker='o', linestyle='', markersize=12, label="cost_slippage")

y_1 = np.empty(5)
y_1[0] = (np.argmin(-mean_cost + LAMBDA * sd_total) + 1)
y_1[1] = (np.argmin(-mean_cost_1 + LAMBDA * sd_total_1) + 1)
y_1[2] = (np.argmin(-mean_cost_2 + LAMBDA * sd_total_2) + 1)
y_1[3] = (np.argmin(-mean_cost_3 + LAMBDA * sd_total_3) + 1)
y_1[4] = (np.argmin(-mean_cost_4 + LAMBDA * sd_total_4) + 1)
plt.plot(x, y_1, marker='o', linestyle='', markersize=12, label="total")

plt.xlabel("param_scenario")
plt.ylabel("optimal_N*")
plt.legend()
plt.savefig('images/1/op_dt.png')
plt.close()


# closed form solution

y_3 = np.empty(5)
y_gamma_0 = np.empty(5)
y_gamma_aprox = np.empty(5)
option = Short_european_call_black_sholes_fixed_time_sep(
    100.0, 0.05, 0.25, 0.05, 1, 100.0, 0.01, 1, 200000)

# default


if os.path.isfile("results/1/gamma.pickle"):
    gamma_abs, gamma_square = pickle.load(open("results/1/gamma.pickle", "rb"))
else:
    gamma_abs, gamma_square = gamma_test_fixed_time(
        100.0, 0.05, 0.25, 0.05, 1, 100.0, 0.01, N_PATHS)
    pickle.dump((gamma_abs, gamma_square), open(
        "results/1/gamma.pickle", "wb"))


mean_cost_ap = 0.01 * np.sqrt(2 / np.pi) * 0.25 * \
    np.sqrt(np.array(range(1, 201))) * gamma_abs

mean_cost_ap_gamma = 0.01 * np.sqrt(2 / np.pi) * 0.25 * np.sqrt(
    np.array(range(1, 201))) * np.abs(option.gamma(100, 0)) * 100**2
mean_cost_ap_avg_gamma = 0.01 * \
    np.sqrt(2 / np.pi) * 0.25 * np.sqrt(np.array(range(1, 201))) * \
    (np.abs(option.avg_cash_gamma()))

sd_slippage_ap = np.sqrt(
    (0.5 / (np.array(range(1, 201)))) * gamma_square * 0.25**4)
sd_slippage_ap_gamma = np.sqrt(
    (0.5 / (np.array(range(1, 201)))) * ((np.abs(option.gamma(100, 0)) * 100**2)**2) * 0.25**4 * np.sqrt(np.pi / 2))
sd_slippage_ap_avg_gamma = np.sqrt(
    (0.5 / (np.array(range(1, 201)))) * (np.abs(option.avg_cash_gamma()))**2 * 0.25**4 * np.sqrt(np.pi / 2))

plt.plot(np.array(range(1, 201)), -mean_cost)
plt.plot(np.array(range(1, 201)), mean_cost_ap)
plt.plot(np.array(range(1, 201)), mean_cost_ap_gamma)
plt.plot(np.array(range(1, 201)), mean_cost_ap_avg_gamma)

plt.xlabel("$N_S$")
plt.ylabel("$\widehat{\mathbb{E}[Friction]}$")

plt.legend(["Path MC", "Gamma MC", "Gamma", "Gamma avg"])
plt.savefig('images/1/mean_aprox_1_call.png')
plt.close()

plt.plot(np.array(range(1, 201)), sd_slippage)
plt.plot(np.array(range(1, 201)), sd_slippage_ap)
plt.plot(np.array(range(1, 201)), sd_slippage_ap_gamma)
plt.plot(np.array(range(1, 201)), sd_slippage_ap_avg_gamma)
plt.xlabel("$N_S$")
plt.ylabel("$\widehat{Sd(Slippage)}$")

plt.legend(["Path MC", "Gamma MC", "Gamma", "Gamma avg"])
plt.savefig('images/1/sd_aprox_1_call.png')
plt.close()
print("0")

y_3[0] = round(LAMBDA * np.sqrt(np.pi) * 0.25 / (2 * 0.01))

print("gamma_1")
####################
# tc
####################
# high tc
option1 = Short_european_call_black_sholes_fixed_time_sep(
    100.0, 0.05, 0.25, 0.05, 1, 100.0, 0.05, 1, 200000)


if os.path.isfile("results/1/gamma_1.pickle"):
    gamma_abs_1, gamma_square_1 = pickle.load(
        open("results/1/gamma_1.pickle", "rb"))
else:
    gamma_abs_1, gamma_square_1 = gamma_test_fixed_time(
        100.0, 0.05, 0.25, 0.05, 1, 100.0, 0.05, N_PATHS)
    pickle.dump((gamma_abs_1, gamma_square_1),
                open("results/1/gamma_1.pickle", "wb"))

mean_cost_ap_1 = 0.05 * np.sqrt(2 / np.pi) * 0.25 * \
    np.sqrt(np.array(range(1, 201))) * gamma_abs_1

mean_cost_ap_gamma1 = 0.05 * np.sqrt(2 / np.pi) * 0.25 * np.sqrt(
    np.array(range(1, 201))) * np.abs(option1.gamma(100, 0)) * 100**2
mean_cost_ap_avg_gamma1 = 0.05 * \
    np.sqrt(2 / np.pi) * 0.25 * np.sqrt(np.array(range(1, 201))) * \
    np.abs(option1.avg_cash_gamma())

sd_slippage_ap_1 = np.sqrt(
    (0.5 / (np.array(range(1, 201)))) * gamma_square_1 * 0.25**4)
sd_slippage_ap_gamma1 = np.sqrt(
    (0.5 / (np.array(range(1, 201)))) * ((option1.gamma(100, 0) * 100**2)**2) * 0.25**4 * np.sqrt(np.pi / 2))
sd_slippage_ap_avg_gamma1 = np.sqrt(
    (0.5 / (np.array(range(1, 201)))) * np.abs(option1.avg_cash_gamma())**2 * 0.25**4 * np.sqrt(np.pi / 2))

plt.plot(np.array(range(1, 201)), -mean_cost_1)
plt.plot(np.array(range(1, 201)), mean_cost_ap_1)
plt.plot(np.array(range(1, 201)), mean_cost_ap_gamma1)
plt.plot(np.array(range(1, 201)), mean_cost_ap_avg_gamma1)

plt.xlabel("$N_S$")
plt.ylabel("$\widehat{\mathbb{E}[Friction]}$")

plt.legend(["Path MC", "Gamma MC", "Gamma", "Gamma avg"])
plt.savefig('images/1/mean_aprox_2_call.png')
plt.close()


plt.plot(np.array(range(1, 201)), sd_slippage_1)
plt.plot(np.array(range(1, 201)), sd_slippage_ap_1)
plt.plot(np.array(range(1, 201)), sd_slippage_ap_gamma1)
plt.plot(np.array(range(1, 201)), sd_slippage_ap_avg_gamma1)
plt.xlabel("$N_S$")
plt.ylabel("$\widehat{Sd(Slippage)}$")

plt.legend(["Path MC", "Gamma MC", "Gamma", "Gamma avg"])
plt.savefig('images/1/sd_aprox_2_call.png')
plt.close()
y_3[1] = round(LAMBDA * np.sqrt(np.pi) *0.25/ (2 * 0.05))
print("gamma_2")

# low tc
option2 = Short_european_call_black_sholes_fixed_time_sep(
    100.0, 0.05, 0.25, 0.05, 1, 100.0, 0.002, 1, 200000)

if os.path.isfile("results/1/gamma_2.pickle"):
    gamma_abs_2, gamma_square_2 = pickle.load(
        open("results/1/gamma_2.pickle", "rb"))
else:
    gamma_abs_2, gamma_square_2 = gamma_test_fixed_time(
        100.0, 0.05, 0.25, 0.05, 1, 100.0, 0.002, N_PATHS)
    pickle.dump((gamma_abs_2, gamma_square_2),
                open("results/1/gamma_2.pickle", "wb"))

mean_cost_ap_2 = 0.002 * np.sqrt(2 / np.pi) * 0.25 * \
    np.sqrt(np.array(range(1, 201))) * gamma_abs_2
mean_cost_ap_gamma2 = 0.002 * np.sqrt(2 / np.pi) * 0.25 * np.sqrt(
    np.array(range(1, 201))) * np.abs(option2.gamma(100, 0)) * 100**2
mean_cost_ap_avg_gamma2 = 0.002 * \
    np.sqrt(2 / np.pi) * 0.25 * np.sqrt(np.array(range(1, 201))) * \
    np.abs(option2.avg_cash_gamma())

sd_slippage_ap_2 = np.sqrt(
    (0.5 / (np.array(range(1, 201)))) * gamma_square_2 * 0.25**4)
sd_slippage_ap_gamma2 = np.sqrt(
    (0.5 / (np.array(range(1, 201)))) * ((np.abs(option2.gamma(100, 0) * 100**2))**2) * 0.25**4 * np.sqrt(np.pi / 2))
sd_slippage_ap_avg_gamma2 = np.sqrt(
    (0.5 / (np.array(range(1, 201)))) * np.abs(option2.avg_cash_gamma())**2 * 0.25**4 * np.sqrt(np.pi / 2))

y_3[2] = round(LAMBDA * np.sqrt(np.pi)*0.25  / (2 * 0.002))
print("gamma_3")
plt.plot(np.array(range(1, 201)), -mean_cost_2)
plt.plot(np.array(range(1, 201)), mean_cost_ap_2)
plt.plot(np.array(range(1, 201)), mean_cost_ap_gamma2)
plt.plot(np.array(range(1, 201)), mean_cost_ap_avg_gamma2)

plt.xlabel("$N_S$")
plt.ylabel("$\widehat{\mathbb{E}[Friction]}$")

plt.legend(["Path MC", "Gamma MC", "Gamma", "Gamma avg"])
plt.savefig('images/1/mean_aprox_3_call.png')
plt.close()


plt.plot(np.array(range(1, 201)), sd_slippage_2)
plt.plot(np.array(range(1, 201)), sd_slippage_ap_2)
plt.plot(np.array(range(1, 201)), sd_slippage_ap_gamma2)
plt.plot(np.array(range(1, 201)), sd_slippage_ap_avg_gamma2)
plt.xlabel("$N_S$")
plt.ylabel("$\widehat{Sd(Slippage)}$")

plt.legend(["Path MC", "Gamma MC", "Gamma", "Gamma avg"])
plt.savefig('images/1/sd_aprox_3_call.png')
plt.close()

###############
# VOL
###############

# low vol
option3 = Short_european_call_black_sholes_fixed_time_sep(
    100.0, 0.05, 0.1, 0.05, 1, 100.0, 0.01, 1, 200000)


if os.path.isfile("results/1/gamma_3.pickle"):
    gamma_abs_3, gamma_square_3 = pickle.load(
        open("results/1/gamma_3.pickle", "rb"))
else:
    gamma_abs_3, gamma_square_3 = gamma_test_fixed_time(
        100.0, 0.05, 0.1, 0.05, 1, 100.0, 0.01, N_PATHS)
    pickle.dump((gamma_abs_3, gamma_square_3),
                open("results/1/gamma_3.pickle", "wb"))

mean_cost_ap_3 = 0.01 * np.sqrt(2 / np.pi) * 0.1 * \
    np.sqrt(np.array(range(1, 201))) * gamma_abs_3
mean_cost_ap_gamma_3 = 0.01 * np.sqrt(2 / np.pi) * 0.1 * np.sqrt(
    np.array(range(1, 201))) * np.abs(option3.gamma(100, 0)) * 100**2
mean_cost_ap_avg_gamma_3 = 0.01 * \
    np.sqrt(2 / np.pi) * 0.1 * np.sqrt(np.array(range(1, 201))) * \
    np.abs(option3.avg_cash_gamma())

sd_slippage_ap_3 = np.sqrt(
    (0.5 / (np.array(range(1, 201)))) * gamma_square_3 * 0.1**4)
sd_slippage_ap_gamma_3 = np.sqrt(
    (0.5 / (np.array(range(1, 201)))) * ((np.abs(option3.gamma(100, 0) * 100**2))**2) * 0.1**4 * np.sqrt(np.pi / 2))
sd_slippage_ap_avg_gamma_3 = np.sqrt(
    (0.5 / (np.array(range(1, 201)))) * np.abs(option3.avg_cash_gamma())**2 * 0.1**4 * np.sqrt(np.pi / 2))

y_3[3] = round(LAMBDA * 1 * np.sqrt(np.pi) *0.1 / (2 * 0.01))
plt.plot(np.array(range(1, 201)), -mean_cost_3)
plt.plot(np.array(range(1, 201)), mean_cost_ap_3)
plt.plot(np.array(range(1, 201)), mean_cost_ap_gamma_3)
plt.plot(np.array(range(1, 201)), mean_cost_ap_avg_gamma_3)

plt.xlabel("$N_S$")
plt.ylabel("$\widehat{\mathbb{E}[Friction]}$")

plt.legend(["Path MC", "Gamma MC", "Gamma", "Gamma avg"])
plt.savefig('images/1/mean_aprox_4_call.png')
plt.close()

plt.plot(np.array(range(1, 201)), sd_slippage_3)
plt.plot(np.array(range(1, 201)), sd_slippage_ap_3)
plt.plot(np.array(range(1, 201)), sd_slippage_ap_gamma_3)
plt.plot(np.array(range(1, 201)), sd_slippage_ap_avg_gamma_3)
plt.xlabel("$N_S$")
plt.ylabel("$\widehat{Sd(Slippage)}$")

plt.legend(["Path MC", "Gamma MC", "Gamma", "Gamma avg"])
plt.savefig('images/1/sd_aprox_4_call.png')
plt.close()
print("gamma_4")

# high vol
option4 = Short_european_call_black_sholes_fixed_time_sep(
    100.0, 0.05, 0.5, 0.05, 1, 100.0, 0.01, 1, 200000)


if os.path.isfile("results/1/gamma_4.pickle"):
    gamma_abs_4, gamma_square_4 = pickle.load(
        open("results/1/gamma_4.pickle", "rb"))
else:
    gamma_abs_4, gamma_square_4 = gamma_test_fixed_time(
        100.0, 0.05, 0.5, 0.05, 1, 100.0, 0.01, N_PATHS)
    pickle.dump((gamma_abs_4, gamma_square_4),
                open("results/1/gamma_4.pickle", "wb"))


mean_cost_ap_4 = 0.01 * np.sqrt(2 / np.pi) * 0.5 * \
    np.sqrt(np.array(range(1, 201))) * gamma_abs_4
mean_cost_ap_gamma4 = 0.01 * np.sqrt(2 / np.pi) * 0.5 * np.sqrt(
    np.array(range(1, 201))) * np.abs(option4.gamma(100, 0)) * 100**2
mean_cost_ap_avg_gamma4 = 0.01 * \
    np.sqrt(2 / np.pi) * 0.5 * np.sqrt(np.array(range(1, 201))) * \
    np.abs(option4.avg_cash_gamma())
sd_slippage_ap_4 = np.sqrt(
    (0.5 / (np.array(range(1, 201)))) * gamma_square_4 * 0.5**4)
sd_slippage_ap_gamma4 = np.sqrt(
    (0.5 / (np.array(range(1, 201)))) * ((np.abs(option4.gamma(100, 0)) * 100**2)**2) * 0.5**4 * np.sqrt(np.pi / 2))
sd_slippage_ap_avg_gamma4 = np.sqrt(
    (0.5 / (np.array(range(1, 201)))) * np.abs(option4.avg_cash_gamma())**2 * 0.5**4 * np.sqrt(np.pi / 2))

y_3[4] = round(LAMBDA * np.sqrt(np.pi)*0.5  / (2 * 0.01))
print("gamma_5")
plt.plot(np.array(range(1, 201)), -mean_cost_4)
plt.plot(np.array(range(1, 201)), mean_cost_ap_4)
plt.plot(np.array(range(1, 201)), mean_cost_ap_gamma4)
plt.plot(np.array(range(1, 201)), mean_cost_ap_avg_gamma4)

plt.xlabel("$N_S$")
plt.ylabel("$\widehat{\mathbb{E}[Friction]}$")

plt.legend(["Path MC", "Gamma MC", "Gamma", "Gamma avg"])
plt.savefig('images/1/mean_aprox_5_call.png')
plt.close()

plt.plot(np.array(range(1, 201)), sd_slippage_4)
plt.plot(np.array(range(1, 201)), sd_slippage_ap_4)
plt.plot(np.array(range(1, 201)), sd_slippage_ap_gamma4)
plt.plot(np.array(range(1, 201)), sd_slippage_ap_avg_gamma4)
plt.xlabel("$N_S$")
plt.ylabel("$\widehat{Sd(Slippage)}$")

plt.legend(["Path_MC", "Gamma_MC", "Gamma", "Gamma_avg", "Gamma_avg_d"])
plt.savefig('images/1/sd_aprox_5_call.png')
plt.close()


plt.plot(np.array(range(1, 201)), np.sqrt(gamma_square_2)/gamma_abs_2 )
plt.plot(np.array(range(1, 201)), np.sqrt(gamma_square)/gamma_abs )
plt.plot(np.array(range(1, 201)), np.sqrt(gamma_square_1)/gamma_abs_1 )
plt.legend(["Low TC", "Default", "High TC"], loc='upper left')
plt.xlabel("$N_S$")
plt.ylabel("Value of the ratio")
plt.savefig('images/1/ratio_k.png')
plt.close()


plt.plot(np.array(range(1, 201)), (np.sqrt(gamma_square_3)/gamma_abs_3))
plt.plot(np.array(range(1, 201)), np.sqrt(gamma_square)/gamma_abs )
plt.plot(np.array(range(1, 201)),   (np.sqrt(gamma_square_4)/gamma_abs_4))
plt.legend(["Low vol", "Default", "High vol"], loc='upper left')


plt.xlabel("$N_S$")
plt.ylabel("Value of the ratio")
plt.savefig('images/1/ratio_v.png')
plt.close()


y_2 = np.empty(5)

y_2[0] = (np.argmin(mean_cost_ap + LAMBDA * sd_slippage_ap) + 1)
y_2[1] = (np.argmin(mean_cost_ap_1 + LAMBDA * sd_slippage_ap_1) + 1)
y_2[2] = (np.argmin(mean_cost_ap_2 + LAMBDA * sd_slippage_ap_2) + 1)
y_2[3] = (np.argmin(mean_cost_ap_3 + LAMBDA * sd_slippage_ap_3) + 1)
y_2[4] = (np.argmin(mean_cost_ap_4 + LAMBDA * sd_slippage_ap_4) + 1)
plt.plot(x, y_2, marker='o', linestyle='', markersize=12, label="True_gamma")

plt.plot(
    x,
    y_3,
    marker='o',
    linestyle='',
    markersize=12,
    label="Approximation_gamma")

plt.xlabel("param_scenario")
plt.ylabel("optimal_N*")
plt.legend()
plt.savefig('images/1/op_aprox.png')
plt.close()


plt.plot(x, y, marker='o', linestyle='', markersize=12, label="cost_slippage")

plt.plot(x, y_1, marker='o', linestyle='', markersize=12, label="total")

plt.plot(x, y_2, marker='o', linestyle='', markersize=12, label="True_gamma")

plt.plot(
    x,
    y_3,
    marker='o',
    linestyle='',
    markersize=12,
    label="Approximation_gamma")

plt.xlabel("param_scenario")
plt.ylabel("optimal_N*")
plt.legend()
plt.savefig('images/1/op_dt_all.png')
plt.close()


plt.plot(x, y, marker='o', linestyle='', markersize=12, label="cost_slippage")

plt.plot(x, y_2, marker='o', linestyle='', markersize=12, label="True_gamma")

plt.plot(
    x,
    y_3,
    marker='o',
    linestyle='',
    markersize=12,
    label="Approximation_gamma")

plt.xlabel("param_scenario")
plt.ylabel("optimal_N*")
plt.legend()
plt.savefig('images/1/op_dt_all_total.png')
plt.close()


print(f"Time used:{time.time()-time0}", file=result_file)
print("slippage UF", file=result_file)
print(x, file=result_file)
print(y, file=result_file)

print(
    f"{mean_cost[int(y[0]-1)]},{mean_cost_1[int(y[1]-1)]},{mean_cost_2[int(y[2]-1)]},{mean_cost_3[int(y[3]-1)]},{mean_cost_4[int(y[4]-1)]}",
    file=result_file)
print(
    f"{sd_slippage[int(y[0]-1)]},{sd_slippage_1[int(y[1]-1)]},{sd_slippage_2[int(y[2]-1)]},{sd_slippage_3[int(y[3]-1)]},{sd_slippage_4[int(y[4]-1)]}",
    file=result_file)



print("Total UF", file=result_file)
print(x, file=result_file)
print(y_1, file=result_file)


print(
    f"{mean_cost[int(y_1[0]-1)]},{mean_cost_1[int(y_1[1]-1)]},{mean_cost_2[int(y_1[2]-1)]},{mean_cost_3[int(y_1[3]-1)]},{mean_cost_4[int(y_1[4]-1)]}",
    file=result_file)
print(
    f"{sd_total[int(y_1[0]-1)]},{sd_total_1[int(y_1[1]-1)]},{sd_total_2[int(y_1[2]-1)]},{sd_total_3[int(y_1[3]-1)]},{sd_total_4[int(y_1[4]-1)]}",
    file=result_file)



print("True gamma Aprox", file=result_file)
print(x, file=result_file)
print(y_2, file=result_file)

print(
    f"{mean_cost_ap[int(y_2[0]-1)]},{mean_cost_ap_1[int(y_2[1]-1)]},{mean_cost_ap_2[int(y_2[2]-1)]},{mean_cost_ap_3[int(y_2[3]-1)]},{mean_cost_ap_4[int(y_2[4]-1)]}",
    file=result_file)
print(
    f"{sd_slippage_ap[int(y_2[0]-1)]},{sd_slippage_ap_1[int(y_2[1]-1)]},{sd_slippage_ap_2[int(y_2[2]-1)]},{sd_slippage_ap_3[int(y_2[3]-1)]},{sd_slippage_ap_4[int(y_2[4]-1)]}",
    file=result_file)


print("gamma aprox", file=result_file)
print(x, file=result_file)
print(y_3, file=result_file)

print(
    f"{mean_cost_ap[int(y_3[0]-1)]},{mean_cost_ap_1[int(y_3[1]-1)]},{mean_cost_ap_2[int(y_3[2]-1)]},{mean_cost_ap_3[int(y_3[3]-1)]},{mean_cost_ap_4[int(y_3[4]-1)]}",
    file=result_file)
print(
    f"{sd_slippage_ap[int(y_3[0]-1)]},{sd_slippage_ap_1[int(y_3[1]-1)]},{sd_slippage_ap_2[int(y_3[2]-1)]},{sd_slippage_ap_3[int(y_3[3]-1)]},{sd_slippage_ap_4[int(y_3[4]-1)]}",
    file=result_file)

print("closed form t0", file=result_file)

print(x, file=result_file)
print(y_3, file=result_file)
print(f"{mean_cost_ap_gamma[int(y_3[0]-1)]},{mean_cost_ap_gamma1[int(y_3[1]-1)]},{mean_cost_ap_gamma2[int(y_3[2]-1)]},\
    {mean_cost_ap_gamma_3[int(y_3[3]-1)]},{mean_cost_ap_gamma4[int(y_3[4]-1)]}", file=result_file)
print(f"{sd_slippage_ap_gamma[int(y_3[0]-1)]},{sd_slippage_ap_gamma1[int(y_3[1]-1)]},{sd_slippage_ap_gamma2[int(y_3[2]-1)]},\
    {sd_slippage_ap_gamma_3[int(y_3[3]-1)]},{sd_slippage_ap_gamma4[int(y_3[4]-1)]}", file=result_file)

print("closed form avg", file=result_file)

print(x, file=result_file)
print(y_3, file=result_file)
print(f"{mean_cost_ap_avg_gamma[int(y_3[0]-1)]},{mean_cost_ap_avg_gamma1[int(y_3[1]-1)]},{mean_cost_ap_avg_gamma2[int(y_3[2]-1)]},\
    {mean_cost_ap_avg_gamma_3[int(y_3[3]-1)]},{mean_cost_ap_avg_gamma4[int(y_3[4]-1)]}", file=result_file)
print(f"{sd_slippage_ap_avg_gamma[int(y_3[0]-1)]},{sd_slippage_ap_avg_gamma1[int(y_3[1]-1)]},{sd_slippage_ap_avg_gamma2[int(y_3[2]-1)]},\
    {sd_slippage_ap_avg_gamma_3[int(y_3[3]-1)]},{sd_slippage_ap_avg_gamma4[int(y_3[4]-1)]}", file=result_file)


result_file.close()

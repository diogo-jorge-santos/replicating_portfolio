import matplotlib.pyplot as plt
import numpy as np
from tools_stable.class_vol_heston import risk_return_vol_heston, gamma_test_vol_heston, gamma_test_vol_heston_t, Vol_heston, gamma_test_vol_heston_aprox
from tools_stable.misc import TIME
import pickle
import os

N_PATHS = 100000
TIME = TIME 
plt.rcParams['text.usetex'] = True

plt.rcParams.update({
    'text.latex.preamble': r'\usepackage{amsfonts}'
    
})
result_file = open("results/4/result_4.txt", mode="w")

print("one")
if os.path.isfile("results/4/MC.pickle"):
    mean, sd, cost_1_mean, cost_1_sd, cost_2_mean, cost_2_sd, slippage_mean, slippage_sd = pickle.load(
        open("results/4/MC.pickle", "rb"))
else:
    mean, sd, cost_1_mean, cost_1_sd, cost_2_mean, cost_2_sd, slippage_mean, slippage_sd = risk_return_vol_heston(
        100, np.sqrt(0.03), 0.0, 3, 0.03, 0.15, -0.7, 0.0, 1, 100.0, 0.005, 0.01, N_PATHS)

    pickle.dump((mean, sd, cost_1_mean, cost_1_sd, cost_2_mean, cost_2_sd,
                slippage_mean, slippage_sd), open("results/4/MC.pickle", "wb"))
print("two")

if os.path.isfile("results/4/MC_1.pickle"):
    mean_1, sd_1, cost_1_mean_1, cost_1_sd_1, cost_2_mean_1, cost_2_sd_1, slippage_mean_1, slippage_sd_1 = pickle.load(
        open("results/4/MC_1.pickle", "rb"))
else:
    mean_1, sd_1, cost_1_mean_1, cost_1_sd_1, cost_2_mean_1, cost_2_sd_1, slippage_mean_1, slippage_sd_1 = risk_return_vol_heston(
        100, np.sqrt(0.03), 0.0, 3, 0.03, 0.15, -0.7, 0.0, 1, 100.0, 0.01, 0.02, N_PATHS)

    pickle.dump((mean_1, sd_1, cost_1_mean_1, cost_1_sd_1, cost_2_mean_1, cost_2_sd_1,
                slippage_mean_1, slippage_sd_1), open("results/4/MC_1.pickle", "wb"))
print("three")

if os.path.isfile("results/4/MC_2.pickle"):
    mean_2, sd_2, cost_1_mean_2, cost_1_sd_2, cost_2_mean_2, cost_2_sd_2, slippage_mean_2, slippage_sd_2 = pickle.load(
        open("results/4/MC_2.pickle", "rb"))
else:
    mean_2, sd_2, cost_1_mean_2, cost_1_sd_2, cost_2_mean_2, cost_2_sd_2, slippage_mean_2, slippage_sd_2 = risk_return_vol_heston(
        100, np.sqrt(0.03), 0.0, 3, 0.03,0.15, -0.7, 0.0, 1, 100.0, 0.0025, 0.005, N_PATHS)

    pickle.dump((mean_2, sd_2, cost_1_mean_2, cost_1_sd_2, cost_2_mean_2, cost_2_sd_2,
                slippage_mean_2, slippage_sd_2), open("results/4/MC_2.pickle", "wb"))
print("four")

if os.path.isfile("results/4/MC_3.pickle"):
    mean_3, sd_3, cost_1_mean_3, cost_1_sd_3, cost_2_mean_3, cost_2_sd_3, slippage_mean_3, slippage_sd_3 = pickle.load(
        open("results/4/MC_3.pickle", "rb"))
else:
    mean_3, sd_3, cost_1_mean_3, cost_1_sd_3, cost_2_mean_3, cost_2_sd_3, slippage_mean_3, slippage_sd_3 = risk_return_vol_heston(
        100, np.sqrt(0.02), 0.0, 3,0.02, 0.1, -0.3, 0.0, 1, 100.0, 0.005, 0.01, N_PATHS)

    pickle.dump((mean_3, sd_3, cost_1_mean_3, cost_1_sd_3, cost_2_mean_3, cost_2_sd_3,
                slippage_mean_3, slippage_sd_3), open("results/4/MC_3.pickle", "wb"))

print("five")

if os.path.isfile("results/4/MC_4.pickle"):
    mean_4, sd_4, cost_1_mean_4, cost_1_sd_4, cost_2_mean_4, cost_2_sd_4, slippage_mean_4, slippage_sd_4 = pickle.load(
        open("results/4/MC_4.pickle", "rb"))
else:
    mean_4, sd_4, cost_1_mean_4, cost_1_sd_4, cost_2_mean_4, cost_2_sd_4, slippage_mean_4, slippage_sd_4 = risk_return_vol_heston(
        100, np.sqrt(0.04), 0.0, 3, 0.04, 0.2, -0.8, 0.0, 1, 100.0, 0.005, 0.01, N_PATHS)

    pickle.dump((mean_4, sd_4, cost_1_mean_4, cost_1_sd_4, cost_2_mean_4, cost_2_sd_4,
                slippage_mean_4, slippage_sd_4), open("results/4/MC_4.pickle", "wb"))


plt.plot(TIME, cost_1_mean_1[:, 0])
plt.plot(TIME, cost_1_mean[:, 0])
plt.plot(TIME, cost_1_mean_2[:, 0])
plt.legend(["k_spot=0.01", "k_spot=0.005", "k_spot=0.0025"])

plt.xlabel("$N_S$")
plt.ylabel("$\widehat{\mathbb{E}[friction^{\mathrm{Delta}}]}$")
plt.savefig('images/4/mean_cost_1.png')
plt.close()

plt.plot(TIME, cost_1_mean_3[:, 0])
plt.plot(TIME, cost_1_mean[:, 0])
plt.plot(TIME, cost_1_mean_4[:, 0])
plt.legend(["Low vol", "Default", "High vol"], loc='upper left')


plt.xlabel("$N_S$")
plt.ylabel("$\widehat{\mathbb{E}[friction^{\mathrm{Delta}}]}$")
plt.savefig('images/4/mean_cost_1_vol.png')
plt.close()


plt.plot(TIME, cost_2_mean_1[0, :])
plt.plot(TIME, cost_2_mean[0, :])
plt.plot(TIME, cost_2_mean_2[0, :])
plt.legend(["k_vol=0.02", "k_vol=0.01", "k_vol=0.005"])
plt.xlabel("$N_S$")
plt.ylabel("$\widehat{\mathbb{E}[friction^{\mathrm{Vega}}]}$")
plt.savefig('images/4/mean_cost_2.png')
plt.close()

plt.plot(TIME, cost_2_mean_3[0, :])
plt.plot(TIME, cost_2_mean[0, :])
plt.plot(TIME, cost_2_mean_4[0, :])
plt.legend(["Low vol", "Default", "High vol"], loc='upper left')

plt.xlabel("$N_S$")
plt.ylabel("$\widehat{\mathbb{E}[friction^{\mathrm{Vega}}]}$")
plt.savefig('images/4/mean_cost_2_vol.png')
plt.close()


plt.plot(TIME, slippage_mean_3[:, 2])
plt.plot(TIME, slippage_mean[:, 2])
plt.plot(TIME, slippage_mean_4[:, 2])
plt.legend(["Low vol", "Default", "High vol"], loc='upper left')

plt.xlabel("$N_S, \quad N_\Sigma=5$")
plt.ylabel("mean_sl")
plt.savefig('images/4/mean_sl_5_vol.png')
plt.close()


plt.plot(TIME, slippage_mean_3[:, 6])
plt.plot(TIME, slippage_mean[:, 6])
plt.plot(TIME, slippage_mean_4[:, 6])
plt.legend(["Low vol", "Default", "High vol"], loc='upper left')

plt.xlabel("$N_S, \quad N_\Sigma=25$")
plt.ylabel("mean_sl")
plt.savefig('images/4/mean_sl_25_vol.png')
plt.close()


plt.plot(TIME, slippage_mean_3[2, :])
plt.plot(TIME, slippage_mean[2, :])
plt.plot(TIME, slippage_mean_4[2, :])
plt.legend(["Low vol", "Default", "High vol"], loc='upper left')

plt.xlabel("$N_S, \quad N_\Sigma=5$")
plt.ylabel("mean_sl")
plt.savefig('images/4/mean_sl_5_vol_vol.png')
plt.close()


plt.plot(TIME, slippage_mean_3[6, :])
plt.plot(TIME, slippage_mean[6, :])
plt.plot(TIME, slippage_mean_4[6, :])
plt.legend(["Low vol", "Default", "High vol"], loc='upper left')

plt.xlabel("$N_S, \quad N_\Sigma=25$")
plt.ylabel("mean_sl")
plt.savefig('images/4/mean_sl_25_vol_vol.png')
plt.close()


plt.plot(TIME, cost_1_sd_1[:, 0])
plt.plot(TIME, cost_1_sd[:, 0])
plt.plot(TIME, cost_1_sd_2[:, 0])
plt.legend(["k_vol=0.02", "k_vol=0.01", "k_vol=0.005"])
plt.xlabel("$N_S$")
plt.ylabel("sd_cost_1")
plt.savefig('images/4/sd_cost_1.png')
plt.close()

plt.plot(TIME, cost_1_sd_3[:, 0])
plt.plot(TIME, cost_1_sd[:, 0])
plt.plot(TIME, cost_1_sd_4[:, 0])
plt.legend(["Low vol", "Default", "High vol"], loc='upper left')

plt.xlabel("$N_S$")
plt.ylabel("sd_cost_1")
plt.savefig('images/4/sd_cost_1_vol.png')
plt.close()

plt.plot(TIME, cost_2_sd_1[0, :])
plt.plot(TIME, cost_2_sd[0, :])
plt.plot(TIME, cost_2_sd_2[0, :])
plt.legend(["k_vol=0.02", "k_vol=0.01", "k_vol=0.005"])
plt.xlabel("$N_S$")
plt.ylabel("sd_cost_2")
plt.savefig('images/4/sd_cost_2.png')
plt.close()

plt.plot(TIME, cost_2_sd_3[0, :])
plt.plot(TIME, cost_2_sd[0, :])
plt.plot(TIME, cost_2_sd_4[0, :])
plt.legend(["Low vol", "Default", "High vol"], loc='upper left')

plt.xlabel("$N_S$")
plt.ylabel("sd_cost_2")
plt.savefig('images/4/sd_cost_2_vol.png')
plt.close()


plt.plot(TIME, slippage_sd_3[:, 2])
plt.plot(TIME, slippage_sd[:, 2])
plt.plot(TIME, slippage_sd_4[:, 2])
plt.xlabel("$N_S, \quad N_\Sigma=5$")
plt.ylabel("sd_sl")
plt.legend(["Low vol", "Default", "High vol"], loc='upper left')

plt.savefig('images/4/sd_sl_5_vol.png')
plt.close()


plt.plot(TIME, slippage_sd_3[:, 6])
plt.plot(TIME, slippage_sd[:, 6])
plt.plot(TIME, slippage_sd_4[:, 6])
plt.legend(["Low vol", "Default", "High vol"], loc='upper left')

plt.xlabel("$N_S, \quad N_\Sigma=25$")
plt.ylabel("sd_sl")
plt.savefig('images/4/sd_sl_25_vol.png')
plt.close()


plt.plot(TIME, slippage_sd_4[2, :])

plt.xlabel("N,N_spot=5")
plt.ylabel("sd_sl")
plt.savefig('images/4/sd_sl_5_vol_vol.png')
plt.close()


plt.plot(TIME, slippage_sd_4[6, :])
plt.xlabel("N,N_spot=25")
plt.ylabel("sd_sl")
plt.savefig('images/4/sd_sl_25_vol_vol.png')
plt.close()

LAMBDA = 0.52
cost_slippage_utility = -(cost_1_mean + cost_2_mean) + LAMBDA * slippage_sd
cost_slippage_utility_1 = - \
    (cost_1_mean_1 + cost_2_mean_1) + LAMBDA * slippage_sd_1
cost_slippage_utility_2 = - \
    (cost_1_mean_2 + cost_2_mean_2) + LAMBDA * slippage_sd_2
cost_slippage_utility_3 = - \
    (cost_1_mean_3 + cost_2_mean_3) + LAMBDA * slippage_sd_3
cost_slippage_utility_4 = - \
    (cost_1_mean_4 + cost_2_mean_4) + LAMBDA * slippage_sd_4

pnl_utility = -(cost_1_mean + cost_2_mean) + LAMBDA * sd
pnl_utility_1 = -(cost_1_mean_1 + cost_2_mean_1) + LAMBDA * sd_1
pnl_utility_2 = -(cost_1_mean_2 + cost_2_mean_2) + LAMBDA * sd_2
pnl_utility_3 = -(cost_1_mean_3 + cost_2_mean_3) + LAMBDA * sd_3
pnl_utility_4 = -(cost_1_mean_4 + cost_2_mean_4) + LAMBDA * sd_4

plt.plot(TIME, cost_slippage_utility[:, 2])
plt.plot(TIME, pnl_utility[:, 2])
plt.legend(["Cost slippage criterion", "PnL criterion"], loc='upper left')
plt.xlabel("$N_S, \quad N_\Sigma=5$")
plt.ylabel("Value of both criteria")
plt.savefig('images/4/PNLvs_5.png')
plt.close()

plt.plot(TIME, cost_slippage_utility[:, 6])
plt.plot(TIME, pnl_utility[:, 6])
plt.legend(["Cost slippage criterion", "PnL criterion"], loc='upper left')
plt.xlabel("$N_S, \quad N_\Sigma=25$")
plt.ylabel("Value of both criteria")
plt.savefig('images/4/PNLvs_25.png')
plt.close()

plt.plot(TIME, cost_slippage_utility_1[:, 2])
plt.plot(TIME, pnl_utility_1[:, 2])
plt.legend(["Cost slippage criterion", "PnL criterion"], loc='upper left')
plt.xlabel("$N_S, \quad N_\Sigma=5$")
plt.ylabel("Value of both criteria")
plt.savefig('images/4/PNLvs_5_1.png')
plt.close()

plt.plot(TIME, cost_slippage_utility_1[:, 6])
plt.plot(TIME, pnl_utility_1[:, 6])
plt.legend(["Cost slippage criterion", "PnL criterion"], loc='upper left')
plt.xlabel("$N_S, \quad N_\Sigma=25$")
plt.ylabel("Value of both criteria")
plt.savefig('images/4/PNLvs_25_1.png')
plt.close()

plt.plot(TIME, cost_slippage_utility_2[:, 2])
plt.plot(TIME, pnl_utility_2[:, 2])
plt.legend(["Cost slippage criterion", "PnL criterion"], loc='upper left')
plt.xlabel("$N_S, \quad N_\Sigma=5$")
plt.ylabel("Value of both criteria")
plt.savefig('images/4/PNLvs_5_2.png')
plt.close()

plt.plot(TIME, cost_slippage_utility_2[:, 6])
plt.plot(TIME, pnl_utility_2[:, 6])
plt.legend(["Cost slippage criterion", "PnL criterion"], loc='upper left')
plt.xlabel("$N_S, \quad N_\Sigma=25$")
plt.ylabel("Value of both criteria")
plt.savefig('images/4/PNLvs_25_2.png')
plt.close()

plt.plot(TIME, cost_slippage_utility_3[:, 2])
plt.plot(TIME, pnl_utility_3[:, 2])
plt.legend(["Cost slippage criterion", "PnL criterion"], loc='upper left')
plt.xlabel("$N_S, \quad N_\Sigma=5$")
plt.ylabel("Value of both criteria")
plt.savefig('images/4/PNLvs_5_3.png')
plt.close()

plt.plot(TIME, cost_slippage_utility_3[:, 6])
plt.plot(TIME, pnl_utility_3[:, 6])
plt.legend(["Cost slippage criterion", "PnL criterion"], loc='upper left')
plt.xlabel("$N_S, \quad N_\Sigma=25$")
plt.ylabel("Value of both criteria")
plt.savefig('images/4/PNLvs_25_3.png')
plt.close()

plt.plot(TIME, cost_slippage_utility_4[:, 2])
plt.plot(TIME, pnl_utility_4[:, 2])
plt.legend(["Cost slippage criterion", "PnL criterion"], loc='upper left')
plt.xlabel("$N_S, \quad N_\Sigma=5$")
plt.ylabel("Value of both criteria")
plt.savefig('images/4/PNLvs_5_4.png')
plt.close()

plt.plot(TIME, cost_slippage_utility_4[:, 6])
plt.plot(TIME, pnl_utility_4[:, 6])
plt.legend(["Cost slippage criterion", "PnL criterion"], loc='upper left')
plt.xlabel("$N_S, \quad N_\Sigma=25$")
plt.ylabel("Value of both criteria")
plt.savefig('images/4/PNLvs_25_4.png')
plt.close()


plt.plot(TIME, cost_slippage_utility_4[2, :])
plt.plot(TIME, pnl_utility_4[2, :])

plt.legend(["Cost slippage criterion", "PnL criterion"], loc='upper left')
plt.xlabel("N,N_spot=25")
plt.ylabel("Value of both criteria")
plt.savefig('images/4/PNLvs_25_4_vol_vol.png')
plt.close()

plt.plot(TIME, cost_slippage_utility_4[6, :])
plt.plot(TIME, pnl_utility_4[6, :])

plt.legend(["Cost slippage criterion", "PnL criterion"], loc='upper left')
plt.xlabel("N,N_spot=5")
plt.ylabel("Value of both criteria")
plt.savefig('images/4/PNLvs_5_4_vol_vol.png')
plt.close()



i, j = np.unravel_index(cost_slippage_utility.argmin(),
                        cost_slippage_utility.shape)
i_1, j_1 = np.unravel_index(
    cost_slippage_utility_1.argmin(), cost_slippage_utility_1.shape)
i_2, j_2 = np.unravel_index(
    cost_slippage_utility_2.argmin(), cost_slippage_utility_2.shape)
i_3, j_3 = np.unravel_index(
    cost_slippage_utility_3.argmin(), cost_slippage_utility_3.shape)
i_4, j_4 = np.unravel_index(
    cost_slippage_utility_4.argmin(), cost_slippage_utility_4.shape)

l, k = np.unravel_index(pnl_utility.argmin(), pnl_utility.shape)
l_1, m_1 = np.unravel_index(pnl_utility_1.argmin(), pnl_utility_1.shape)
l_2, m_2 = np.unravel_index(pnl_utility_2.argmin(), pnl_utility_2.shape)
l_3, m_3 = np.unravel_index(pnl_utility_3.argmin(), pnl_utility_3.shape)
l_4, m_4 = np.unravel_index(pnl_utility_4.argmin(), pnl_utility_4.shape)


print(
    f"{TIME[i]} {TIME[j]} {cost_1_mean[i,j]} {cost_2_mean[i,j]} {slippage_sd[i,j]}",
    file=result_file)
print(f"{TIME[l]} {TIME[k]} {cost_1_mean[i,j]} {cost_2_mean[i,j]} {sd[l,k]}", file=result_file)
print(" ", file=result_file)

print(
    f"{TIME[i_1]} {TIME[j_1]} {cost_1_mean_1[i_1,j_1]} {cost_2_mean_1[i_1,j_1]} {slippage_sd_1[i_1,j_1]}",
    file=result_file)
print(f"{TIME[l_1]} {TIME[m_1]} {cost_1_mean_1[l_1,m_1]} {cost_2_mean_1[l_1,m_1]} {sd_1[l_1,m_1]}",
      file=result_file)
print(" ", file=result_file)
print(
    f"{TIME[i_2]} {TIME[j_2]} {cost_1_mean_2[i_2,j_2]} {cost_2_mean_2[i_2,j_2]} {slippage_sd_2[i_2,j_2]}",
    file=result_file)
print(f"{TIME[l_2]} {TIME[m_2]} {cost_1_mean_2[l_2,m_2]} {cost_2_mean_2[l_2,m_2]} {sd_2[l_2,m_2]}",
      file=result_file)
print(" ", file=result_file)
print(
    f"{TIME[i_3]} {TIME[j_3]} {cost_1_mean_3[i_3,j_3]} {cost_2_mean_3[i_3,j_3]} {slippage_sd_3[i_3,j_3]}",
    file=result_file)
print(f"{TIME[l_3]} {TIME[m_3]} {cost_1_mean_3[l_3,m_3]} {cost_2_mean_3[l_3,m_3]} {sd_3[l_3,m_3]}",
      file=result_file)
print(" ", file=result_file)
print(
    f"{TIME[i_4]} {TIME[j_4]} {cost_1_mean_4[i_4,j_4]} {cost_2_mean_4[i_4,j_4]} {slippage_sd_4[i_4,j_4]}",
    file=result_file)
print(f"{TIME[l_4]} {TIME[m_4]} {cost_1_mean_4[l_4,m_4]} {cost_2_mean_4[l_4,m_4]} {sd_4[l_4,m_4]}",
      file=result_file)
print(" ", file=result_file)


if os.path.isfile("results/4/gamma.pickle"):
    cost_1_mean_aux, slippage_sd_1_aux, cost_2_mean_aux, slippage_sd_2_aux, list_gamma_vanna_a, list_gamma_vanna_q, list_volga_vanna_a, list_volga_vanna_q, sd_s_coef, sd_sigma_coef, corr_coef = pickle.load(
        open("results/4/gamma.pickle", "rb"))
else:
    cost_1_mean_aux, slippage_sd_1_aux, cost_2_mean_aux, slippage_sd_2_aux, list_gamma_vanna_a, list_gamma_vanna_q, list_volga_vanna_a, list_volga_vanna_q, sd_s_coef, sd_sigma_coef, corr_coef = gamma_test_vol_heston(
        100, np.sqrt(0.03), 0.0, 3, 0.03, 0.15, -0.7, 0.0, 1, 100.0, 0.005, 0.01, N_PATHS)

    pickle.dump((cost_1_mean_aux, slippage_sd_1_aux, cost_2_mean_aux, slippage_sd_2_aux, list_gamma_vanna_a, list_gamma_vanna_q,
                list_volga_vanna_a, list_volga_vanna_q, sd_s_coef, sd_sigma_coef, corr_coef), open("results/4/gamma.pickle", "wb"))


if os.path.isfile("results/4/gamma_1.pickle"):
    cost_1_mean_aux_1, slippage_sd_1_aux_1, cost_2_mean_aux_1, slippage_sd_2_aux_1, list_gamma_vanna_a_1, list_gamma_vanna_q_1, list_volga_vanna_a_1, list_volga_vanna_q_1, sd_s_coef_1, sd_sigma_coef_1, corr_coef_1 = pickle.load(
        open("results/4/gamma_1.pickle", "rb"))
else:
    cost_1_mean_aux_1, slippage_sd_1_aux_1, cost_2_mean_aux_1, slippage_sd_2_aux_1, list_gamma_vanna_a_1, list_gamma_vanna_q_1, list_volga_vanna_a_1, list_volga_vanna_q_1, sd_s_coef_1, sd_sigma_coef_1, corr_coef_1 = gamma_test_vol_heston(
        100, np.sqrt(0.03), 0.0, 3, 0.03, 0.15, -0.7, 0.0, 1, 100.0, 0.01, 0.02, N_PATHS)

    pickle.dump((cost_1_mean_aux_1, slippage_sd_1_aux_1, cost_2_mean_aux_1, slippage_sd_2_aux_1, list_gamma_vanna_a_1, list_gamma_vanna_q_1,
                list_volga_vanna_a_1, list_volga_vanna_q_1, sd_s_coef_1, sd_sigma_coef_1, corr_coef_1), open("results/4/gamma_1.pickle", "wb"))


if os.path.isfile("results/4/gamma_2.pickle"):
    cost_1_mean_aux_2, slippage_sd_1_aux_2, cost_2_mean_aux_2, slippage_sd_2_aux_2, list_gamma_vanna_a_2, list_gamma_vanna_q_2, list_volga_vanna_a_2, list_volga_vanna_q_2, sd_s_coef_2, sd_sigma_coef_2, corr_coef_2 = pickle.load(
        open("results/4/gamma_2.pickle", "rb"))
else:
    cost_1_mean_aux_2, slippage_sd_1_aux_2, cost_2_mean_aux_2, slippage_sd_2_aux_2, list_gamma_vanna_a_2, list_gamma_vanna_q_2, list_volga_vanna_a_2, list_volga_vanna_q_2, sd_s_coef_2, sd_sigma_coef_2, corr_coef_2 = gamma_test_vol_heston(
        100, np.sqrt(0.03), 0.0, 3, 0.03,0.15, -0.7, 0.0, 1, 100.0, 0.0025, 0.005, N_PATHS)

    pickle.dump((cost_1_mean_aux_2, slippage_sd_1_aux_2, cost_2_mean_aux_2, slippage_sd_2_aux_2, list_gamma_vanna_a_2, list_gamma_vanna_q_2,
                list_volga_vanna_a_2, list_volga_vanna_q_2, sd_s_coef_2, sd_sigma_coef_2, corr_coef_2), open("results/4/gamma_2.pickle", "wb"))


if os.path.isfile("results/4/gamma_3.pickle"):
    cost_1_mean_aux_3, slippage_sd_1_aux_3, cost_2_mean_aux_3, slippage_sd_2_aux_3, list_gamma_vanna_a_3, list_gamma_vanna_q_3, list_volga_vanna_a_3, list_volga_vanna_q_3, sd_s_coef_3, sd_sigma_coef_3, corr_coef_3 = pickle.load(
        open("results/4/gamma_3.pickle", "rb"))
else:
    cost_1_mean_aux_3, slippage_sd_1_aux_3, cost_2_mean_aux_3, slippage_sd_2_aux_3, list_gamma_vanna_a_3, list_gamma_vanna_q_3, list_volga_vanna_a_3, list_volga_vanna_q_3, sd_s_coef_3, sd_sigma_coef_3, corr_coef_3 = gamma_test_vol_heston(
        100, np.sqrt(0.02), 0.0, 3,0.02, 0.1, -0.3, 0.0, 1, 100.0, 0.005, 0.01, N_PATHS)

    pickle.dump((cost_1_mean_aux_3, slippage_sd_1_aux_3, cost_2_mean_aux_3, slippage_sd_2_aux_3, list_gamma_vanna_a_3, list_gamma_vanna_q_3,
                list_volga_vanna_a_3, list_volga_vanna_q_3, sd_s_coef_3, sd_sigma_coef_3, corr_coef_3), open("results/4/gamma_3.pickle", "wb"))


if os.path.isfile("results/4/gamma_4.pickle"):
    cost_1_mean_aux_4, slippage_sd_1_aux_4, cost_2_mean_aux_4, slippage_sd_2_aux_4, list_gamma_vanna_a_4, list_gamma_vanna_q_4, list_volga_vanna_a_4, list_volga_vanna_q_4, sd_s_coef_4, sd_sigma_coef_4, corr_coef_4 = pickle.load(
        open("results/4/gamma_4.pickle", "rb"))
else:
    cost_1_mean_aux_4, slippage_sd_1_aux_4, cost_2_mean_aux_4, slippage_sd_2_aux_4, list_gamma_vanna_a_4, list_gamma_vanna_q_4, list_volga_vanna_a_4, list_volga_vanna_q_4, sd_s_coef_4, sd_sigma_coef_4, corr_coef_4 = gamma_test_vol_heston(
        100, np.sqrt(0.04), 0.0, 3, 0.04, 0.2, -0.8, 0.0, 1, 100.0, 0.005, 0.01, N_PATHS)

    pickle.dump((cost_1_mean_aux_4, slippage_sd_1_aux_4, cost_2_mean_aux_4, slippage_sd_2_aux_4, list_gamma_vanna_a_4, list_gamma_vanna_q_4,
                list_volga_vanna_a_4, list_volga_vanna_q_4, sd_s_coef_4, sd_sigma_coef_4, corr_coef_4), open("results/4/gamma_4.pickle", "wb"))


cost_1_mean_aux_t, slippage_sd_1_aux_t, cost_2_mean_aux_t, slippage_sd_2_aux_t = gamma_test_vol_heston_t(
    100, np.sqrt(0.03), 0.0, 3, 0.03, 0.15, -0.7, 0.0, 1, 100.0, 0.005, 0.01, N_PATHS, sd_s_coef, sd_sigma_coef, corr_coef)

cost_1_mean_aux_t_1, slippage_sd_1_aux_t_1, cost_2_mean_aux_t_1, slippage_sd_2_aux_t_1 = gamma_test_vol_heston_t(
    100, np.sqrt(0.03), 0.0, 3, 0.03, 0.15, -0.7, 0.0, 1, 100.0, 0.01, 0.02, N_PATHS, sd_s_coef_1, sd_sigma_coef_1, corr_coef_1)

cost_1_mean_aux_t_2, slippage_sd_1_aux_t_2, cost_2_mean_aux_t_2, slippage_sd_2_aux_t_2 = gamma_test_vol_heston_t(
    100, np.sqrt(0.03), 0.0, 3, 0.03,0.15, -0.7, 0.0, 1, 100.0, 0.0025, 0.005, N_PATHS, sd_s_coef_2, sd_sigma_coef_2, corr_coef_2)

cost_1_mean_aux_t_3, slippage_sd_1_aux_t_3, cost_2_mean_aux_t_3, slippage_sd_2_aux_t_3 = gamma_test_vol_heston_t(
   100, np.sqrt(0.02), 0.0, 3,0.02, 0.1, -0.3, 0.0, 1, 100.0, 0.005, 0.01, N_PATHS, sd_s_coef_3, sd_sigma_coef_3, corr_coef_3)

cost_1_mean_aux_t_4, slippage_sd_1_aux_t_4, cost_2_mean_aux_t_4, slippage_sd_2_aux_t_4 = gamma_test_vol_heston_t(
    100, np.sqrt(0.04), 0.0, 3, 0.04, 0.2, -0.8, 0.0, 1, 100.0, 0.005, 0.01, N_PATHS, sd_s_coef_4, sd_sigma_coef_4, corr_coef_4)


cost_1_mean_aux_aprox, slippage_sd_1_aux_aprox, cost_2_mean_aux_aprox, slippage_sd_2_aux_aprox = gamma_test_vol_heston_aprox(
    100, np.sqrt(0.03), 0.0, 3, 0.03, 0.15, -0.7, 0.0, 1, 100.0, 0.005, 0.01, N_PATHS, sd_s_coef, sd_sigma_coef, corr_coef)

cost_1_mean_aux_aprox_1, slippage_sd_1_aux_aprox_1, cost_2_mean_aux_aprox_1, slippage_sd_2_aux_aprox_1 = gamma_test_vol_heston_aprox(
    100, np.sqrt(0.03), 0.0, 3, 0.03, 0.15, -0.7, 0.0, 1, 100.0, 0.01, 0.02, N_PATHS, sd_s_coef_1, sd_sigma_coef_1, corr_coef_1)

cost_1_mean_aux_aprox_2, slippage_sd_1_aux_aprox_2, cost_2_mean_aux_aprox_2, slippage_sd_2_aux_aprox_2 = gamma_test_vol_heston_aprox(
    100, np.sqrt(0.03), 0.0, 3, 0.03,0.15, -0.7, 0.0, 1, 100.0, 0.0025, 0.005, N_PATHS, sd_s_coef_2, sd_sigma_coef_2, corr_coef_2)

cost_1_mean_aux_aprox_3, slippage_sd_1_aux_aprox_3, cost_2_mean_aux_aprox_3, slippage_sd_2_aux_aprox_3 = gamma_test_vol_heston_aprox(
    100, np.sqrt(0.02), 0.0, 3,0.02, 0.1, -0.3, 0.0, 1, 100.0, 0.005, 0.01, N_PATHS, sd_s_coef_3, sd_sigma_coef_3, corr_coef_3)

cost_1_mean_aux_aprox_4, slippage_sd_1_aux_aprox_4, cost_2_mean_aux_aprox_4, slippage_sd_2_aux_aprox_4 = gamma_test_vol_heston_aprox(
    100, np.sqrt(0.04), 0.0, 3, 0.04, 0.2, -0.8, 0.0, 1, 100.0,0.005, 0.01, N_PATHS, sd_s_coef_4, sd_sigma_coef_4, corr_coef_4)


plt.plot(TIME,  np.sqrt(list_gamma_vanna_q_3)/list_gamma_vanna_a_3 )
plt.plot(TIME,  np.sqrt(list_gamma_vanna_q)/list_gamma_vanna_a )
plt.plot(TIME,  np.sqrt(list_gamma_vanna_q_4)/list_gamma_vanna_a_4 )

plt.xlabel("$N_S$")
plt.legend(["Low vol", "Default", "High vol"], loc='upper left')

plt.ylabel("Value of the ratio")
plt.savefig('images/4/racio_gamma_vanna.png')
plt.close()

plt.plot(TIME, list_gamma_vanna_a_3, marker='o')
plt.plot(TIME, list_gamma_vanna_a, marker='o')
plt.plot(TIME, list_gamma_vanna_a_4, marker='o')

plt.xlabel("$N_S$")
plt.legend(["Low vol", "Default", "High vol"], loc='upper left')

plt.ylabel("combined_gamma_vanna_a")
plt.savefig('images/4/A.png')
plt.close()

plt.plot(TIME, list_gamma_vanna_q_3, marker='o')
plt.plot(TIME, list_gamma_vanna_q, marker='o')
plt.plot(TIME, list_gamma_vanna_q_4, marker='o')

plt.xlabel("$N_S$")
plt.legend(["Low vol", "Default", "High vol"], loc='upper left')

plt.ylabel("combined_gamma_vanna_q")
plt.savefig('images/4/B.png')
plt.close()


plt.plot(TIME,  np.sqrt(list_volga_vanna_q_3)/list_volga_vanna_a_3 )
plt.plot(TIME, np.sqrt(list_volga_vanna_q)/list_volga_vanna_a )
plt.plot(TIME, np.sqrt(list_volga_vanna_q_4)/list_volga_vanna_a_4  )

plt.legend(["Low vol", "Default", "High vol"], loc='upper left')

plt.xlabel("$N_\Sigma$")
plt.ylabel("Value of the ratio")
plt.savefig('images/4/racio_volga_vanna.png')
plt.close()

plt.plot(TIME, list_volga_vanna_a_3, marker='o')
plt.plot(TIME, list_volga_vanna_a, marker='o')
plt.plot(TIME, list_volga_vanna_a_4, marker='o')

plt.legend(["Low vol", "Default", "High vol"], loc='upper left')

plt.xlabel("$N_S$")
plt.ylabel("combined_vanna_volga_a")
plt.savefig('images/4/C.png')
plt.close()

plt.plot(TIME, list_volga_vanna_q_3, marker='o')
plt.plot(TIME, list_volga_vanna_q, marker='o')
plt.plot(TIME, list_volga_vanna_q_4, marker='o')
plt.legend(["Low vol", "Default", "High vol"], loc='upper left')

plt.xlabel("$N_S$")
plt.ylabel("combined_vanna_volga_q")
plt.savefig('images/4/D.png')
plt.close()


plt.plot(TIME, -cost_1_mean[:, 0])
plt.plot(TIME, cost_1_mean_aux)
plt.plot(TIME, cost_1_mean_aux_t)
plt.plot(TIME, cost_1_mean_aux_aprox)
plt.legend(["Path MC", "Gamma MC", "Gamma", "Gamma avg"])
plt.xlabel("$N_S$")
plt.ylabel("$\widehat{\mathbb{E}[friction^{\mathrm{Delta}}]}$")
plt.savefig('images/4/cost_1_comp.png')
plt.close()

plt.plot(TIME, -cost_1_mean_1[:, 0])
plt.plot(TIME, cost_1_mean_aux_1)
plt.plot(TIME, cost_1_mean_aux_t_1)
plt.plot(TIME, cost_1_mean_aux_aprox_1)
plt.legend(["Path MC", "Gamma MC", "Gamma", "Gamma avg"])
plt.xlabel("$N_S$")
plt.ylabel("$\widehat{\mathbb{E}[friction^{\mathrm{Delta}}]}$")
plt.savefig('images/4/cost_1_comp_1.png')
plt.close()

plt.plot(TIME, -cost_1_mean_2[:, 0])
plt.plot(TIME, cost_1_mean_aux_2)
plt.plot(TIME, cost_1_mean_aux_t_2)
plt.plot(TIME, cost_1_mean_aux_aprox_2)
plt.legend(["Path MC", "Gamma MC", "Gamma", "Gamma avg"])
plt.xlabel("$N_S$")
plt.ylabel("$\widehat{\mathbb{E}[friction^{\mathrm{Delta}}]}$")
plt.savefig('images/4/cost_1_comp_2.png')
plt.close()

plt.plot(TIME, -cost_1_mean_3[:, 0])
plt.plot(TIME, cost_1_mean_aux_3)
plt.plot(TIME, cost_1_mean_aux_t_3)
plt.plot(TIME, cost_1_mean_aux_aprox_3)
plt.legend(["Path MC", "Gamma MC", "Gamma", "Gamma avg"])
plt.xlabel("$N_S$")
plt.ylabel("$\widehat{\mathbb{E}[friction^{\mathrm{Delta}}]}$")
plt.savefig('images/4/cost_1_comp_3.png')
plt.close()

plt.plot(TIME, -cost_1_mean_4[:, 0])
plt.plot(TIME, cost_1_mean_aux_4)
plt.plot(TIME, cost_1_mean_aux_t_4)
plt.plot(TIME, cost_1_mean_aux_aprox_4)
plt.legend(["Path MC", "Gamma MC", "Gamma", "Gamma avg"])
plt.xlabel("$N_S$")
plt.ylabel("$\widehat{\mathbb{E}[friction^{\mathrm{Delta}}]}$")
plt.savefig('images/4/cost_1_comp_4.png')
plt.close()

plt.plot(TIME, -cost_2_mean[0, :])
plt.plot(TIME, cost_2_mean_aux)
plt.plot(TIME, cost_2_mean_aux_t)
plt.plot(TIME, cost_2_mean_aux_aprox)
plt.legend(["Path MC", "Gamma MC", "Gamma", "Gamma avg"])
plt.xlabel("$N_\Sigma$")
plt.ylabel("$\widehat{\mathbb{E}[friction^{\mathrm{Vega}}]}$")
plt.savefig('images/4/cost_2_comp.png')
plt.close()

plt.plot(TIME, -cost_2_mean_1[0, :])
plt.plot(TIME, cost_2_mean_aux_1)
plt.plot(TIME, cost_2_mean_aux_t_1)
plt.plot(TIME, cost_2_mean_aux_aprox_1)
plt.legend(["Path MC", "Gamma MC", "Gamma", "Gamma avg"])
plt.xlabel("$N_\Sigma$")
plt.ylabel("$\widehat{\mathbb{E}[friction^{\mathrm{Vega}}]}$")
plt.savefig('images/4/cost_2_comp_1.png')
plt.close()

plt.plot(TIME, -cost_2_mean_2[0, :])
plt.plot(TIME, cost_2_mean_aux_2)
plt.plot(TIME, cost_2_mean_aux_t_2)
plt.plot(TIME, cost_2_mean_aux_aprox_2)

plt.legend(["Path MC", "Gamma MC", "Gamma", "Gamma avg"])
plt.xlabel("$N_\Sigma$")
plt.ylabel("$\widehat{\mathbb{E}[friction^{\mathrm{Vega}}]}$")
plt.savefig('images/4/cost_2_comp_2.png')
plt.close()

plt.plot(TIME, -cost_2_mean_3[0, :])
plt.plot(TIME, cost_2_mean_aux_3)
plt.plot(TIME, cost_2_mean_aux_t_3)
plt.plot(TIME, cost_2_mean_aux_aprox_3)

plt.legend(["Path MC", "Gamma MC", "Gamma", "Gamma avg"])
plt.xlabel("$N_\Sigma$")
plt.ylabel("$\widehat{\mathbb{E}[friction^{\mathrm{Vega}}]}$")
plt.savefig('images/4/cost_2_comp_3.png')
plt.close()

plt.plot(TIME, -cost_2_mean_4[0, :])
plt.plot(TIME, cost_2_mean_aux_4)
plt.plot(TIME, cost_2_mean_aux_t_4)
plt.plot(TIME, cost_2_mean_aux_aprox_4)

plt.legend(["Path MC", "Gamma MC", "Gamma", "Gamma avg"])
plt.xlabel("$N_\Sigma$")
plt.ylabel("$\widehat{\mathbb{E}[friction^{\mathrm{Vega}}]}$")
plt.savefig('images/4/cost_2_comp_4.png')
plt.close()

plt.plot(TIME, slippage_sd[:, 2])
plt.plot(TIME, np.sqrt(slippage_sd_1_aux**2 + slippage_sd_2_aux[2]**2))
plt.plot(TIME, np.sqrt(slippage_sd_1_aux_t**2 + slippage_sd_2_aux_t[2]**2))
plt.plot(
    TIME,
    np.sqrt(
        slippage_sd_1_aux_aprox**2 +
        slippage_sd_2_aux_aprox[2]**2))
plt.legend(["Path MC", "Gamma MC", "Gamma", "Gamma avg"])
plt.xlabel("$N_S, \quad N_\Sigma=5$")
plt.ylabel("$\widehat{Sd(Slippage)}$")
plt.savefig('images/4/sd_slippage_comp_5.png')
plt.close()

plt.plot(TIME, slippage_sd_1[:, 2])
plt.plot(TIME, np.sqrt(slippage_sd_1_aux_1**2 + slippage_sd_2_aux_1[2]**2))
plt.plot(TIME, np.sqrt(slippage_sd_1_aux_t_1**2 + slippage_sd_2_aux_t_1[2]**2))
plt.plot(
    TIME,
    np.sqrt(
        slippage_sd_1_aux_aprox_1**2 +
        slippage_sd_2_aux_aprox_1[2]**2))

plt.legend(["Path MC", "Gamma MC", "Gamma", "Gamma avg"])
plt.xlabel("$N_S, \quad N_\Sigma=5$")
plt.ylabel("$\widehat{Sd(Slippage)}$")
plt.savefig('images/4/sd_slippage_comp_5_1.png')
plt.close()


plt.plot(TIME, slippage_sd_2[:, 2])
plt.plot(TIME, np.sqrt(slippage_sd_1_aux_2**2 + slippage_sd_2_aux_2[2]**2))
plt.plot(TIME, np.sqrt(slippage_sd_1_aux_t_2**2 + slippage_sd_2_aux_t_2[2]**2))
plt.plot(
    TIME,
    np.sqrt(
        slippage_sd_1_aux_aprox_2**2 +
        slippage_sd_2_aux_aprox_2[2]**2))

plt.legend(["Path MC", "Gamma MC", "Gamma", "Gamma avg"])
plt.xlabel("$N_S, \quad N_\Sigma=5$")
plt.ylabel("$\widehat{Sd(Slippage)}$")
plt.savefig('images/4/sd_slippage_comp_5_2.png')
plt.close()

plt.plot(TIME, slippage_sd_3[:, 2])
plt.plot(TIME, np.sqrt(slippage_sd_1_aux_3**2 + slippage_sd_2_aux_3[2]**2))
plt.plot(TIME, np.sqrt(slippage_sd_1_aux_t_3**2 + slippage_sd_2_aux_t_3[2]**2))
plt.plot(
    TIME,
    np.sqrt(
        slippage_sd_1_aux_aprox_3**2 +
        slippage_sd_2_aux_aprox_3[2]**2))

plt.legend(["Path MC", "Gamma MC", "Gamma", "Gamma avg"])
plt.xlabel("$N_S, \quad N_\Sigma=5$")
plt.ylabel("$\widehat{Sd(Slippage)}$")
plt.savefig('images/4/sd_slippage_comp_5_3.png')
plt.close()

plt.plot(TIME, slippage_sd_4[:, 2])
plt.plot(TIME, np.sqrt(slippage_sd_1_aux_4**2 + slippage_sd_2_aux_4[2]**2))
plt.plot(TIME, np.sqrt(slippage_sd_1_aux_t_4**2 + slippage_sd_2_aux_t_4[2]**2))
plt.plot(
    TIME,
    np.sqrt(
        slippage_sd_1_aux_aprox_4**2 +
        slippage_sd_2_aux_aprox_4[2]**2))

plt.legend(["Path MC", "Gamma MC", "Gamma", "Gamma avg"])
plt.xlabel("$N_S, \quad N_\Sigma=5$")
plt.ylabel("$\widehat{Sd(Slippage)}$")
plt.savefig('images/4/sd_slippage_comp_5_4.png')
plt.close()

plt.plot(TIME, slippage_sd[:, 6])
plt.plot(TIME, np.sqrt(slippage_sd_1_aux**2 + slippage_sd_2_aux[6]**2))
plt.plot(TIME, np.sqrt(slippage_sd_1_aux_t**2 + slippage_sd_2_aux_t[6]**2))
plt.plot(
    TIME,
    np.sqrt(
        slippage_sd_1_aux_aprox**2 +
        slippage_sd_2_aux_aprox[6]**2))
plt.legend(["Path MC", "Gamma MC", "Gamma", "Gamma avg"])
plt.xlabel("$N_S, \quad N_\Sigma=25$")
plt.ylabel("$\widehat{Sd(Slippage)}$")
plt.savefig('images/4/sd_slippage_comp_25.png')
plt.close()


plt.plot(TIME, slippage_sd_1[:, 6])
plt.plot(TIME, np.sqrt(slippage_sd_1_aux_1**2 + slippage_sd_2_aux_1[6]**2))
plt.plot(TIME, np.sqrt(slippage_sd_1_aux_t_1**2 + slippage_sd_2_aux_t_1[6]**2))
plt.plot(
    TIME,
    np.sqrt(
        slippage_sd_1_aux_aprox_1**2 +
        slippage_sd_2_aux_aprox_1[6]**2))

plt.legend(["Path MC", "Gamma MC", "Gamma", "Gamma avg"])
plt.xlabel("$N_S, \quad N_\Sigma=25$")
plt.ylabel("$\widehat{Sd(Slippage)}$")
plt.savefig('images/4/sd_slippage_comp_25_1.png')
plt.close()


plt.plot(TIME, slippage_sd_2[:, 6])
plt.plot(TIME, np.sqrt(slippage_sd_1_aux_2**2 + slippage_sd_2_aux_2[6]**2))
plt.plot(TIME, np.sqrt(slippage_sd_1_aux_t_2**2 + slippage_sd_2_aux_t_2[6]**2))
plt.plot(
    TIME,
    np.sqrt(
        slippage_sd_1_aux_aprox_2**2 +
        slippage_sd_2_aux_aprox_2[6]**2))

plt.legend(["Path MC", "Gamma MC", "Gamma", "Gamma avg"])
plt.xlabel("$N_S, \quad N_\Sigma=25$")
plt.ylabel("$\widehat{Sd(Slippage)}$")
plt.savefig('images/4/sd_slippage_comp_25_2.png')
plt.close()

plt.plot(TIME, slippage_sd_3[:, 6])
plt.plot(TIME, np.sqrt(slippage_sd_1_aux_3**2 + slippage_sd_2_aux_3[6]**2))
plt.plot(TIME, np.sqrt(slippage_sd_1_aux_t_3**2 + slippage_sd_2_aux_t_3[6]**2))
plt.plot(
    TIME,
    np.sqrt(
        slippage_sd_1_aux_aprox_3**2 +
        slippage_sd_2_aux_aprox_3[6]**2))

plt.legend(["Path MC", "Gamma MC", "Gamma", "Gamma avg"])
plt.xlabel("$N_S, \quad N_\Sigma=25$")
plt.ylabel("$\widehat{Sd(Slippage)}$")
plt.savefig('images/4/sd_slippage_comp_25_3.png')
plt.close()

plt.plot(TIME, slippage_sd_4[:, 6])
plt.plot(TIME, np.sqrt(slippage_sd_1_aux_4**2 + slippage_sd_2_aux_4[6]**2))
plt.plot(TIME, np.sqrt(slippage_sd_1_aux_t_4**2 + slippage_sd_2_aux_t_4[6]**2))
plt.plot(
    TIME,
    np.sqrt(
        slippage_sd_1_aux_aprox_4**2 +
        slippage_sd_2_aux_aprox_4[6]**2))

plt.legend(["Path MC", "Gamma MC", "Gamma", "Gamma avg"])
plt.xlabel("$N_S, \quad N_\Sigma=25$")
plt.ylabel("$\widehat{Sd(Slippage)}$")
plt.savefig('images/4/sd_slippage_comp_25_4.png')
plt.close()



plt.plot(TIME, np.sqrt(slippage_sd_1_aux_4[2]**2 + slippage_sd_2_aux_4**2))
plt.xlabel("$N_S$")
plt.ylabel("$\widehat{Sd(Slippage)}$")
plt.savefig('images/4/sd_slippage_vol_vol.png')
plt.close()

plt.plot(TIME, np.sqrt(slippage_sd_1_aux_4[6]**2 + slippage_sd_2_aux_4**2))
plt.xlabel("$N_S$")
plt.ylabel("$\widehat{Sd(Slippage)}$")
plt.savefig('images/4/sd_slippage_vol_vol_25.png')
plt.close()


plt.plot(TIME, slippage_sd_2_aux_4)
plt.xlabel("$N_S$")
plt.ylabel("$\widehat{Sd(Slippage)}$")
plt.savefig('images/4/sd_slippage_vol_vol_one.png')
plt.close()


n = np.argmin(cost_1_mean_aux[1:] + LAMBDA *slippage_sd_1_aux[1:])+1
o = np.argmin(cost_2_mean_aux[1:] + LAMBDA *slippage_sd_2_aux[1:])+1

n_1 = np.argmin(cost_1_mean_aux_1[1:] + LAMBDA * slippage_sd_1_aux_1[1:])+1
o_1 = np.argmin(cost_2_mean_aux_1[1:] + LAMBDA * slippage_sd_2_aux_1[1:])+1

n_2 = np.argmin(cost_1_mean_aux_2[1:] + LAMBDA * slippage_sd_1_aux_2[1:])+1
o_2 = np.argmin(cost_2_mean_aux_2[1:] + LAMBDA * slippage_sd_2_aux_2[1:])+1

n_3 = np.argmin(cost_1_mean_aux_3[1:] + LAMBDA * slippage_sd_1_aux_3[1:])+1
o_3 = np.argmin(cost_2_mean_aux_3[1:] + LAMBDA * slippage_sd_2_aux_3[1:])+1

n_4 = np.argmin(cost_1_mean_aux_4[1:] + LAMBDA * slippage_sd_1_aux_4[1:])+1
o_4 = np.argmin(cost_2_mean_aux_4[1:] + LAMBDA * slippage_sd_2_aux_4[1:])+1

def aux(a,b):
    matrix=np.zeros((a.shape[0],a.shape[0]))
    
    for i in range(0,a.shape[0]):
        for j in range(0,a.shape[0]):
            matrix[i,j]=a[i]+b[j]
    return matrix

cost_slippage_utility_aux_aux = aux(cost_1_mean_aux, cost_2_mean_aux) + LAMBDA * np.sqrt(aux(slippage_sd_1_aux**2,slippage_sd_2_aux**2))
cost_slippage_utility_aux_1_aux  = aux(cost_1_mean_aux_1 , cost_2_mean_aux_1) + LAMBDA * np.sqrt(aux(slippage_sd_1_aux_1**2,slippage_sd_2_aux_1**2))
cost_slippage_utility_aux_2_aux  = aux(cost_1_mean_aux_2 , cost_2_mean_aux_2) + LAMBDA * np.sqrt(aux(slippage_sd_1_aux_2**2,slippage_sd_2_aux_2**2))
cost_slippage_utility_aux_3_aux  = aux(cost_1_mean_aux_3 , cost_2_mean_aux_3) + LAMBDA * np.sqrt(aux(slippage_sd_1_aux_3**2,slippage_sd_2_aux_3**2))
cost_slippage_utility_aux_4_aux  = aux(cost_1_mean_aux_4 , cost_2_mean_aux_4) + LAMBDA * np.sqrt(aux(slippage_sd_1_aux_4**2,slippage_sd_2_aux_4**2))

n_aux_aux ,o_aux_aux  = np.unravel_index(cost_slippage_utility_aux_aux.argmin(), cost_slippage_utility_aux_aux.shape)
n_1_aux_aux  ,o_1_aux_aux  = np.unravel_index(cost_slippage_utility_aux_1_aux.argmin(), cost_slippage_utility_aux_1_aux.shape)
n_2_aux_aux  ,o_2_aux_aux  = np.unravel_index(cost_slippage_utility_aux_2_aux.argmin(), cost_slippage_utility_aux_2_aux.shape)
n_3_aux_aux  ,o_3_aux_aux  = np.unravel_index(cost_slippage_utility_aux_3_aux.argmin(), cost_slippage_utility_aux_3_aux.shape)
n_4_aux_aux  ,o_4_aux_aux  = np.unravel_index(cost_slippage_utility_aux_4_aux.argmin(), cost_slippage_utility_aux_4_aux.shape)

print(
f"{TIME[n_aux_aux]} {TIME[o_aux_aux]}",
file=result_file)
print(
f"{TIME[n_1_aux_aux]} {TIME[o_1_aux_aux]}",
file=result_file)
print(
f"{TIME[n_2_aux_aux]} {TIME[o_2_aux_aux]}",
file=result_file)
print(
f"{TIME[n_3_aux_aux]} {TIME[o_3_aux_aux]}",
file=result_file)
print(
f"{TIME[n_4_aux_aux]} {TIME[o_4_aux_aux]}",
file=result_file)
   

cost_slippage_utility_aux = aux(cost_1_mean_aux[1:] , cost_2_mean_aux[1:]) + LAMBDA * np.sqrt(aux(slippage_sd_1_aux[1:]**2,slippage_sd_2_aux[1:]**2))
cost_slippage_utility_aux_1 = aux(cost_1_mean_aux_1[1:] , cost_2_mean_aux_1[1:]) + LAMBDA * np.sqrt(aux(slippage_sd_1_aux_1[1:]**2,slippage_sd_2_aux_1[1:]**2))
cost_slippage_utility_aux_2 = aux(cost_1_mean_aux_2[1:] , cost_2_mean_aux_2[1:]) + LAMBDA * np.sqrt(aux(slippage_sd_1_aux_2[1:]**2,slippage_sd_2_aux_2[1:]**2))
cost_slippage_utility_aux_3 = aux(cost_1_mean_aux_3[1:] , cost_2_mean_aux_3[1:]) + LAMBDA * np.sqrt(aux(slippage_sd_1_aux_3[1:]**2,slippage_sd_2_aux_3[1:]**2))
cost_slippage_utility_aux_4 = aux(cost_1_mean_aux_4[1:] , cost_2_mean_aux_4[1:]) + LAMBDA * np.sqrt(aux(slippage_sd_1_aux_4[1:]**2,slippage_sd_2_aux_4[1:]**2))

n_aux,o_aux = np.unravel_index(cost_slippage_utility_aux.argmin(), cost_slippage_utility_aux.shape)
n_aux,o_aux=n_aux+1,o_aux+1
n_1_aux ,o_1_aux = np.unravel_index(cost_slippage_utility_aux_1.argmin(), cost_slippage_utility_aux_1.shape)
n_1_aux ,o_1_aux=n_1_aux+1 ,o_1_aux+1
n_2_aux ,o_2_aux = np.unravel_index(cost_slippage_utility_aux_2.argmin(), cost_slippage_utility_aux_2.shape)
n_2_aux ,o_2_aux =n_2_aux +1,o_2_aux +1
n_3_aux ,o_3_aux = np.unravel_index(cost_slippage_utility_aux_3.argmin(), cost_slippage_utility_aux_3.shape)
n_3_aux ,o_3_aux=n_3_aux+1 ,o_3_aux+1
n_4_aux ,o_4_aux = np.unravel_index(cost_slippage_utility_aux_4.argmin(), cost_slippage_utility_aux_4.shape)
n_4_aux ,o_4_aux=n_4_aux+1 ,o_4_aux+1


p = np.argmin(np.abs(TIME-LAMBDA*np.sqrt(np.pi)*sd_s_coef/(2*0.005)))
q = np.argmin(np.abs(TIME-LAMBDA*np.sqrt(np.pi)*sd_sigma_coef/(2*0.01)))

p_1 = np.argmin(np.abs(TIME-LAMBDA*np.sqrt(np.pi)*sd_s_coef_1/(2*0.01)))
q_1 = np.argmin(np.abs(TIME-LAMBDA*np.sqrt(np.pi)*sd_sigma_coef_1/(2*0.02)))

p_2 = np.argmin(np.abs(TIME-LAMBDA*np.sqrt(np.pi)*sd_s_coef_2/(2*0.0025)))
q_2 = np.argmin(np.abs(TIME-LAMBDA*np.sqrt(np.pi)*sd_sigma_coef_2/(2*0.005)))

p_3 = np.argmin(np.abs(TIME-LAMBDA*np.sqrt(np.pi)*sd_s_coef_3/(2*0.005)))
q_3 = np.argmin(np.abs(TIME-LAMBDA*np.sqrt(np.pi)*sd_s_coef_3/(2*0.01)))

p_4 = np.argmin(np.abs(TIME-LAMBDA*np.sqrt(np.pi)*sd_s_coef_4/(2*0.005)))
q_4 = np.argmin(np.abs(TIME-LAMBDA*np.sqrt(np.pi)*sd_sigma_coef_4/(2*0.01)))


r = np.argmin(np.abs(TIME-LAMBDA*np.sqrt(np.pi)*sd_s_coef/(2*0.005)))
s = np.argmin(np.abs(TIME-LAMBDA*np.sqrt(np.pi)*sd_sigma_coef/(2*0.01)))

r_1 = np.argmin(np.abs(TIME-LAMBDA*np.sqrt(np.pi)*sd_s_coef_1/(2*0.01)))
s_1 = np.argmin(np.abs(TIME-LAMBDA*np.sqrt(np.pi)*sd_sigma_coef_1/(2*0.02)))

r_2 = np.argmin(np.abs(TIME-LAMBDA*np.sqrt(np.pi)*sd_s_coef_2/(2*0.0025)))
s_2 = np.argmin(np.abs(TIME-LAMBDA*np.sqrt(np.pi)*sd_sigma_coef_2/(2*0.005)))

r_3 = np.argmin(np.abs(TIME-LAMBDA*np.sqrt(np.pi)*sd_s_coef_3/(2*0.005)))
s_3 = np.argmin(np.abs(TIME-LAMBDA*np.sqrt(np.pi)*sd_s_coef_3/(2*0.01)))

r_4 = np.argmin(np.abs(TIME-LAMBDA*np.sqrt(np.pi)*sd_s_coef_4/(2*0.005)))
s_4 = np.argmin(np.abs(TIME-LAMBDA*np.sqrt(np.pi)*sd_sigma_coef_4/(2*0.01)))


print("0", file=result_file)
print(sd_s_coef, sd_sigma_coef, corr_coef, file=result_file)

print(
    f"{TIME[n]} {TIME[o]} {cost_1_mean_aux[n]} {cost_2_mean_aux[o]} {np.sqrt(slippage_sd_1_aux[n]**2+slippage_sd_2_aux[o]**2)}",
    file=result_file)

print(
    f"{TIME[n_aux]} {TIME[o_aux]} {cost_1_mean_aux[n_aux]} {cost_2_mean_aux[o_aux]} {np.sqrt(slippage_sd_1_aux[n_aux]**2+slippage_sd_2_aux[o_aux]**2)}",
    file=result_file)

print(
    f"{TIME[p]} {TIME[q]} {cost_1_mean_aux_t[p]} {cost_2_mean_aux_t[q]} {np.sqrt(slippage_sd_1_aux_t[p]**2+slippage_sd_2_aux_t[q]**2)}",
    file=result_file)
print(
    f"{TIME[r]} {TIME[s]} {cost_1_mean_aux_aprox[r]} {cost_2_mean_aux_aprox[s]} {np.sqrt(slippage_sd_1_aux_aprox[r]**2+slippage_sd_2_aux_aprox[s]**2)}",
    file=result_file)

print(
    f"{LAMBDA*np.sqrt(np.pi)*sd_s_coef/(2*0.005)} {LAMBDA*np.sqrt(np.pi)*sd_sigma_coef/(2*0.01)}",
    file=result_file)
print("1", file=result_file)
print(sd_s_coef_1, sd_sigma_coef_1, corr_coef_1, file=result_file)

print(
    f"{TIME[n_1]} {TIME[o_1]} {cost_1_mean_aux_1[n_1]} {cost_2_mean_aux_1[o_1]} {np.sqrt(slippage_sd_1_aux_1[n_1]**2+slippage_sd_2_aux_1[o_1]**2)}",
    file=result_file)

print(
    f"{TIME[n_1_aux]} {TIME[o_1_aux]} {cost_1_mean_aux_1[n_1_aux]} {cost_2_mean_aux_1[o_1_aux]} {np.sqrt(slippage_sd_1_aux_1[n_1_aux]**2+slippage_sd_2_aux_1[o_1_aux]**2)}",
    file=result_file)


print(
    f"{TIME[p_1]} {TIME[q_1]} {cost_1_mean_aux_t_1[p_1]} {cost_2_mean_aux_t_1[q_1]} {np.sqrt(slippage_sd_1_aux_t_1[p_1]**2+slippage_sd_2_aux_t_1[q_1]**2)}",
    file=result_file)
print(
    f"{TIME[r_1]} {TIME[s_1]} {cost_1_mean_aux_aprox_1[r_1]} {cost_2_mean_aux_aprox_1[s_1]} {np.sqrt(slippage_sd_1_aux_aprox_1[r_1]**2+slippage_sd_2_aux_aprox_1[s_1]**2)}",
    file=result_file)

print(
    f"{LAMBDA*np.sqrt(np.pi)*sd_s_coef_1/(2*0.01)} {LAMBDA*np.sqrt(np.pi)*sd_sigma_coef_1/(2*0.02)}",
    file=result_file)

print("2", file=result_file)
print(sd_s_coef_2, sd_sigma_coef_2, corr_coef_2, file=result_file)

print(
    f"{TIME[n_2]} {TIME[o_2]} {cost_1_mean_aux_2[n_2]} {cost_2_mean_aux_2[o_2]} {np.sqrt(slippage_sd_1_aux_2[n_2]**2+slippage_sd_2_aux_2[o_2]**2)}",
    file=result_file)

print(
    f"{TIME[n_2_aux]} {TIME[o_2_aux]} {cost_1_mean_aux_2[n_2_aux]} {cost_2_mean_aux_2[o_2_aux]} {np.sqrt(slippage_sd_1_aux_2[n_2_aux]**2+slippage_sd_2_aux_2[o_2_aux]**2)}",
    file=result_file)

print(
    f"{TIME[p_2]} {TIME[q_2]} {cost_1_mean_aux_t_2[p_2]} {cost_2_mean_aux_t_2[q_2]} {np.sqrt(slippage_sd_1_aux_t_2[p_2]**2+slippage_sd_2_aux_t_2[q_2]**2)}",
    file=result_file)

print(
    f"{TIME[r_2]} {TIME[s_2]} {cost_1_mean_aux_aprox_2[r_2]} {cost_2_mean_aux_aprox_2[s_2]} {np.sqrt(slippage_sd_1_aux_aprox_2[r_2]**2+slippage_sd_2_aux_aprox_2[s_2]**2)}",
    file=result_file)
print(
    f"{LAMBDA*np.sqrt(np.pi)*sd_s_coef_2/(2*0.0025)} {LAMBDA*np.sqrt(np.pi)*sd_sigma_coef_2/(2*0.005)}",
    file=result_file)


print("3", file=result_file)
print(sd_s_coef_3, sd_sigma_coef_3, corr_coef_3, file=result_file)



print(
    f"{TIME[n_3]} {TIME[o_3]} {cost_1_mean_aux_3[n_3]} {cost_2_mean_aux_3[o_3]} {np.sqrt(slippage_sd_1_aux_3[n_3]**2+slippage_sd_2_aux_3[o_3]**2)}",
    file=result_file)
print(
    f"{TIME[n_3_aux]} {TIME[o_3_aux]} {cost_1_mean_aux_3[n_3_aux]} {cost_2_mean_aux_3[o_3_aux]} {np.sqrt(slippage_sd_1_aux_3[n_3_aux]**2+slippage_sd_2_aux_3[o_3_aux]**2)}",
    file=result_file)
    
print(
    f"{TIME[p_3]} {TIME[q_3]} {cost_1_mean_aux_t_3[p_3]} {cost_2_mean_aux_t_3[q_3]} {np.sqrt(slippage_sd_1_aux_t_3[p_3]**2+slippage_sd_2_aux_t_3[q_3]**2)}",
    file=result_file)

print(
    f"{TIME[r_3]} {TIME[s_3]} {cost_1_mean_aux_aprox_3[r_3]} {cost_2_mean_aux_aprox_3[s_3]} {np.sqrt(slippage_sd_1_aux_aprox_3[r_3]**2+slippage_sd_2_aux_aprox_3[s_3]**2)}",
    file=result_file)

print(
    f"{LAMBDA*np.sqrt(np.pi)*sd_s_coef_3/(2*0.005)} {LAMBDA*np.sqrt(np.pi)*sd_s_coef_3/(2*0.01)}",
    file=result_file)


print("4", file=result_file)
print(sd_s_coef_4, sd_sigma_coef_4, corr_coef_4, file=result_file)


print(
    f"{TIME[n_4]} {TIME[o_4]} {cost_1_mean_aux_4[n_4]} {cost_2_mean_aux_4[o_4]} {np.sqrt(slippage_sd_1_aux_4[n_4]**2+slippage_sd_2_aux_4[o_4]**2)}",
    file=result_file)
print(
    f"{TIME[n_4_aux]} {TIME[o_4_aux]} {cost_1_mean_aux_4[n_4_aux]} {cost_2_mean_aux_4[o_4_aux]} {np.sqrt(slippage_sd_1_aux_4[n_4_aux]**2+slippage_sd_2_aux_4[o_4_aux]**2)}",
    file=result_file)

print(
    f"{TIME[p_4]} {TIME[q_4]} {cost_1_mean_aux_t_4[p_4]} {cost_2_mean_aux_t_4[q_4]} {np.sqrt(slippage_sd_1_aux_t_4[p_4]**2+slippage_sd_2_aux_t_4[q_4]**2)}",
    file=result_file)
print(
    f"{TIME[r_4]} {TIME[s_4]} {cost_1_mean_aux_aprox_4[r_4]} {cost_2_mean_aux_aprox_4[s_4]} {np.sqrt(slippage_sd_1_aux_aprox_4[r_4]**2+slippage_sd_2_aux_aprox_4[s_4]**2)}",
    file=result_file)
print(
    f"{LAMBDA*np.sqrt(np.pi)*sd_s_coef_4/(2*0.005)} {LAMBDA*np.sqrt(np.pi)*sd_sigma_coef_4/(2*0.01)}",
    file=result_file)


if os.path.isfile("results/4/greeks.pickle"):
    option = Vol_heston(
        100, np.sqrt(0.04), 0.0, 3, 0.04, 0.2, -0.8, 0.0, 1, 100.0, 0.01, 0.02, 1000, 50, 50, N_PATHS)
    cash_vanna, cash_volga = pickle.load(open("results/4/greeks.pickle", "rb"))
else:

    option = Vol_heston(
        100, np.sqrt(0.04), 0.0, 3, 0.04, 0.2, -0.8, 0.0, 1, 100.0, 0.01, 0.02, 1000, 50, 50, N_PATHS)
    cash_vanna, cash_volga = option.test_vanna_volga()
    pickle.dump(
        (cash_vanna, cash_volga), open(
            "results/4/greeks.pickle", "wb"))


if os.path.isfile("results/4/greeks_1.pickle"):
    option_1 = Vol_heston(
        100, np.sqrt(0.03), 0.0, 3, 0.03, 0.15, -0.7, 0.0, 1, 100.0, 0.02, 0.02, 1000, 50, 50, N_PATHS)
    cash_vanna_1, cash_volga_1 = pickle.load(
        open("results/4/greekS_1.pickle", "rb"))
else:

    option_1 = Vol_heston(
        100, np.sqrt(0.03), 0.0, 3, 0.03, 0.15, -0.7, 0.0, 1, 100.0, 0.02, 0.02, 1000, 50, 50, N_PATHS)
    cash_vanna_1, cash_volga_1 = option_1.test_vanna_volga()
    pickle.dump(
        (cash_vanna_1, cash_volga_1), open(
            "results/4/greeks_1.pickle", "wb"))


if os.path.isfile("results/4/greeks_2.pickle"):
    option_2 = Vol_heston(
        100, np.sqrt(0.02), 0.0, 3,0.02, 0.1, -0.3, 0.0, 1, 100.0, 0.01, 0.02, 1000, 50, 50, N_PATHS)
    cash_vanna_2, cash_volga_2 = pickle.load(
        open("results/4/greeks_2.pickle", "rb"))
else:

    option_2 = Vol_heston(
        100, np.sqrt(0.02), 0.0, 3,0.02, 0.1, -0.3, 0.0, 1, 100.0, 0.01, 0.02, 1000, 50, 50, N_PATHS)
    cash_vanna_2, cash_volga_2 = option_2.test_vanna_volga()
    pickle.dump(
        (cash_vanna_2, cash_volga_2), open(
            "results/4/greeks_2.pickle", "wb"))

cash_vanna_t0 = option.vanna(
    option.price_spot,
    option.price_vol,
    0) * option.price_vol * option.price_spot
cash_vanna_t0_1 = option_1.vanna(
    option_1.price_spot,
    option_1.price_vol,
    0) * option_1.price_vol * option_1.price_spot
cash_vanna_t0_2 = option_2.vanna(
    option_2.price_spot,
    option_2.price_vol,
    0) * option_2.price_vol * option_2.price_spot

cash_volga_t0 = option.volga(
    option.price_spot,
    option.price_vol,
    0) * option.price_vol**2
cash_volga_t0_1 = option_1.volga(
    option_1.price_spot,
    option_1.price_vol,
    0) * option_1.price_vol**2
cash_volga_t0_2 = option_2.volga(
    option_2.price_spot,
    option_2.price_vol,
    0) * option_2.price_vol**2


vanna = []
for i in range(1, 200):
    vanna.append(option.vanna(i+0.1, option.price_vol, 0.1))
plt.plot(vanna)
plt.savefig('images/4/vanna_t_0.png')
plt.close()

vanna_5 = []
for i in range(1, 200):
    vanna_5.append(option.vanna(i+0.1, option.price_vol, 0.5))
plt.plot(vanna_5)
plt.savefig('images/4/vanna_t_5.png')
plt.close()

volga = []
for i in range(1, 200):
    volga.append(option.volga(i+0.1, option.price_vol, 0.1))
plt.plot(volga)
plt.savefig('images/4/volga_t_0.png')
plt.close()


volga_5 = []
for i in range(1, 200):
    volga_5.append(option.volga(i+0.1, option.price_vol, 0.5))
plt.plot(volga_5)
plt.savefig('images/4/volga_t_5.png')
plt.close()

plt.plot(np.linspace(0,1-1/50,len(cash_vanna_2)),cash_vanna_2)
plt.plot(np.linspace(0,1-1/50,len(cash_vanna_2)),cash_vanna_1)
plt.plot(np.linspace(0,1-1/50,len(cash_vanna_2)),cash_vanna)
plt.legend(["Low vol", "Default", "High vol"], loc='upper left')
plt.xlabel("$t$")
plt.ylabel("$\mathbb{E}[\\frac{\delta^2 V}{\delta S_t \delta \Sigma_t} S_t \Sigma_t]$")
plt.savefig('images/4/cash_vanna.png')
plt.close()

plt.plot(np.linspace(0,1-1/50,len(cash_vanna_2)),cash_volga_2)
plt.plot(np.linspace(0,1-1/50,len(cash_vanna_2)),cash_volga_1)
plt.plot(np.linspace(0,1-1/50,len(cash_vanna_2)),cash_volga)
plt.legend(["Low vol", "Default", "High vol"], loc='upper left')
plt.xlabel("$t$")
plt.ylabel("$\mathbb{E}[\\frac{\delta^2 V}{\delta \Sigma_t^2} \Sigma_t^2]$")
plt.savefig('images/4/cash_volga.png')
plt.close()


print(" ", file=result_file)
print("Greeks", file=result_file)


print(
    f"vanna_high_vol: {np.mean(cash_vanna)} {cash_vanna_t0} {cash_vanna_t0/2}",
    file=result_file)
print(
    f"vanna: {np.mean(cash_vanna_1)} {cash_vanna_t0_1} {cash_vanna_t0_1/2}",
    file=result_file)
print(
    f"vanna_low_vol: {np.mean(cash_vanna_2)} {cash_vanna_t0_2} {cash_vanna_t0_2/2}",
    file=result_file)
print(" ", file=result_file)
print(
    f"volga_high_vol: {np.mean(cash_volga)} {cash_volga_t0} {cash_volga_t0/2}",
    file=result_file)
print(
    f"volga: {np.mean(cash_volga_1)} {cash_volga_t0_1} {cash_volga_t0_1/2}",
    file=result_file)
print(
    f"volga_low_vol: {np.mean(cash_volga_2)} {cash_volga_t0_2} {cash_volga_t0_2/2}",
    file=result_file)


if os.path.isfile("results/4/greeks_3.pickle"):
    option_3 = Vol_heston(
        100, np.sqrt(0.04), 0.0, 3, 0.04, 0.2, -0.8, 0.0, 10, 100.0, 0.01, 0.02, 1000, 50, 50, 50000)
    cash_vanna_3, cash_volga_3 = pickle.load(
        open("results/4/greeks_3.pickle", "rb"))
else:

    option_3 = Vol_heston(
       100, np.sqrt(0.04), 0.0, 3, 0.03, 0.15, -0.7, 0.0, 10, 100.0, 0.01, 0.02, 1000, 50, 50, N_PATHS)
    cash_vanna_3, cash_volga_3 = option_3.test_vanna_volga()
    pickle.dump(
        (cash_vanna_3, cash_volga_3), open(
            "results/4/greeks_3.pickle", "wb"))


if os.path.isfile("results/4/greeks_4.pickle"):
    option_4 = Vol_heston(
        100, np.sqrt(0.03), 0.0, 3, 0.03, 0.15, -0.7, 0.0, 10, 100.0, 0.005, 0.01, 1000, 50, 50, N_PATHS)
    cash_vanna_4, cash_volga_4 = pickle.load(
        open("results/4/greekS_4.pickle", "rb"))
else:

    option_4 = Vol_heston(
        100, np.sqrt(0.02), 0.0, 3,0.02, 0.1, -0.3, 0.0, 10, 100.0, 0.005, 0.01, 1000, 50, 50, N_PATHS)
    cash_vanna_4, cash_volga_4 = option_4.test_vanna_volga()
    pickle.dump(
        (cash_vanna_4, cash_volga_4), open(
            "results/4/greeks_4.pickle", "wb"))


if os.path.isfile("results/4/greeks_5.pickle"):
    option_5 = Vol_heston(
        100, np.sqrt(0.02), 0.0, 3,0.02, 0.1, -0.3, 0.0, 10, 100.0, 0.01, 0.02, 1000, 50, 50, N_PATHS)
    cash_vanna_5, cash_volga_5 = pickle.load(
        open("results/4/greeks_5.pickle", "rb"))
else:

    option_5 = Vol_heston(
        100, np.sqrt(0.01), 0.0, 3,0.01, 0.1, -0.3, 0.0, 10, 100.0, 0.01, 0.02, 1000, 50, 50, N_PATHS)
    cash_vanna_5, cash_volga_5 = option_5.test_vanna_volga()
    pickle.dump(
        (cash_vanna_5, cash_volga_5), open(
            "results/4/greeks_5.pickle", "wb"))

cash_vanna_t0_3 = option_3.vanna(
    option_3.price_spot,
    option_3.price_vol,
    0) * option_3.price_vol * option_3.price_spot
cash_vanna_t0_4 = option_4.vanna(
    option_4.price_spot,
    option_4.price_vol,
    0) * option_4.price_vol * option_4.price_spot
cash_vanna_t0_5 = option_5.vanna(
    option_5.price_spot,
    option_5.price_vol,
    0) * option_5.price_vol * option_5.price_spot

cash_volga_t0_3 = option_3.volga(
    option_3.price_spot,
    option_3.price_vol,
    0) * option_3.price_vol**2
cash_volga_t0_4 = option_4.volga(
    option_4.price_spot,
    option_4.price_vol,
    0) * option_4.price_vol**2
cash_volga_t0_5 = option_5.volga(
    option_5.price_spot,
    option_5.price_vol,
    0) * option_5.price_vol**2


plt.plot(cash_vanna_3)
plt.plot(cash_vanna_4)
plt.plot(cash_vanna_5)
plt.legend(["high_vol", "default", "low_vol"])
plt.xlabel("$N_S$")
plt.ylabel("cash_vanna")
plt.savefig('images/4/cash_vanna_1.png')
plt.close()

plt.plot(cash_volga_3)
plt.plot(cash_volga_4)
plt.plot(cash_volga_5)
plt.legend(["high_vol", "default", "low_vol"])
plt.xlabel("$N_S$")
plt.ylabel("cash_volga")
plt.savefig('images/4/cash_volga_1.png')
plt.close()


print(" ", file=result_file)
print("Greeks_T=10", file=result_file)


print(
    f"vanna_high_vol: {np.mean(cash_vanna_3)} {cash_vanna_t0_3} {cash_vanna_t0_3/2}",
    file=result_file)
print(
    f"vanna: {np.mean(cash_vanna_4)} {cash_vanna_t0_4} {cash_vanna_t0_4/2}",
    file=result_file)
print(
    f"vanna_low_vol: {np.mean(cash_vanna_5)} {cash_vanna_t0_5} {cash_vanna_t0_5/2}",
    file=result_file)
print(" ", file=result_file)
print(
    f"volga_high_vol: {np.mean(cash_volga_3)} {cash_volga_t0_3} {cash_volga_t0_3/2}",
    file=result_file)
print(
    f"volga: {np.mean(cash_volga_4)} {cash_volga_t0_4} {cash_volga_t0_4/2}",
    file=result_file)
print(
    f"volga_low_vol: {np.mean(cash_volga_5)} {cash_volga_t0_5} {cash_volga_t0_5/2}",
    file=result_file)


result_file.close()

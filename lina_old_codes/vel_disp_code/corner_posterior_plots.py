# Making multiple corner plots on top of each other, and single posterior plots. 

import sys, os
import numpy as np
import corner
import DM_profiles
import matplotlib.pyplot as plt
import matplotlib 
matplotlib.use('Agg')
from matplotlib import rcParams

rcParams['xtick.labelsize'] = 22
rcParams['xtick.direction'] = 'in'
rcParams['ytick.labelsize'] = 22
rcParams['ytick.direction'] = 'in'
rcParams['axes.labelsize'] = 22
rcParams['axes.titlesize'] = 22
rcParams['font.family'] = 'serif'
rcParams['font.serif'] = 'CMU Serif'
rcParams['figure.figsize'] = (10/1.2,8/1.2)
rcParams['legend.fontsize'] = 18
rcParams['text.usetex'] = True
rcParams['text.latex.preamble'] = [r"\usepackage{amsmath}"]

plots_dir =  "/tigress/lnecib/dSph_likelihoods/Plots/DM_plots/"

chains_dir_base = "/tigress/ljchang/dSph_likelihoods/RunFiles/vel_disp_code/multinest_chains/"

param_to_plot = 4 # gamma in a gNFW. This needs to be generalized
cusp = True

if cusp:
    sim_tag =  "PlumCuspIso" 
    gamma_value = 1
else:
    sim_tag = "PlumCoreIso" #
    gamma_value = 0

list_files = ['df_100_0_Plummer_gNFW', 'df_1000_0_Plummer_gNFW', 'df_10000_0_Plummer_gNFW']

plot_labels = [100, 1000, 10000]

inputs = [chains_dir_base + sim_tag + '/' + list_files[i] + '/post_equal_weights.dat' for i in range(len(list_files))]

input_data = []

for i in range(len(list_files)):
    data = np.loadtxt(inputs[i])
    input_data.append(data.T[param_to_plot])    




def make_posterior_plot(input_data = [], title = '', save_tag = '',
                        truth_line_args = {"color":"k","ls":"--","label":"Truth", "params":[gamma_value]},
                        plot_colors = ['cornflowerblue', 'firebrick' ,'forestgreen','darkgoldenrod','darkorchid'],
                        plot_labels = [], post_quantiles = [[16,84]],
                        results_args = {"color":"cornflowerblue","alpha":0.2, "nbins":20}):

    nscans = len(input_data)
    assert len(plot_colors) >= nscans, "Need to specify more colors!"
    assert len(plot_labels) == nscans, "Number of labels "+str(len(plot_labels))+" doesn't match number of scans "+str(nscans)


    
    fontsize = 16

    ax = plt.subplot(1,1,1) 
    ax.minorticks_on()
    ax.tick_params('both', length=8, width=1, which='major', direction='in', labelsize=fontsize)
    ax.tick_params('both', length=4, width=1, which='minor', direction='in', labelsize=fontsize)


    ax.axvline(x = truth_line_args["params"],ls=truth_line_args["ls"],color=truth_line_args["color"],label=truth_line_args["label"])

    # nbins = results_args["nbins"]
    gamma_min = -2
    gamma_max = 2
    gamma_step = 0.1
    bins = np.arange(gamma_min, gamma_max, gamma_step)
    # hist, bins = np.histogram(input_data[0], bins = nbins, density = 1)

    for i in range(nscans):
        histogram, edges = np.histogram(input_data[i], bins = bins)
        bin_means = (edges[1:] + edges[:-1])/2.
        # ax.step(bin_means, histogram, where = 'mid', color=plot_colors[i], label=plot_labels[i])
        ax.plot(bin_means, histogram, color=plot_colors[i], label=plot_labels[i])

    plt.legend()
    plt.tight_layout()

    out_path = plots_dir + sim_tag + "/composite" + "/single_posterior_" + save_tag

    plt.savefig(out_path + ".pdf", bbox_inches="tight")
    plt.close()




make_posterior_plot(input_data = input_data, title = r'$\gamma$', save_tag = '100_1000_10000_gamma', plot_labels = plot_labels)






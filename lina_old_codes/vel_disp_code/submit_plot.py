import os, sys
import numpy as np

batch1='''#!/bin/bash
##SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 2:00:00
#SBATCH --mem=10GB
##SBATCH --mail-type=begin
##SBATCH --mail-type=end
##SBATCH --mail-user=ljchang@princeton.edu
#SBATCH --mail-user=lnecib@caltech.edu

cd /tigress/lnecib/dSph_likelihoods/RunFiles/vel_disp_code/

'''

cusp = False 

chains_dir_base = "/tigress/ljchang/dSph_likelihoods/RunFiles/vel_disp_code/multinest_chains/"

if not os.path.exists(chains_dir_base):
	os.makedirs(chains_dir_base)

data_dir_base = "/tigress/ljchang/dSph_likelihoods/RunFiles/data/gaia_challenge/"
# plots_dir_base = "/tigress/ljchang/dSph_likelihoods/Plots/DM_plots/"
plots_dir_base = "/tigress/lnecib/dSph_likelihoods/Plots/DM_plots/"
post_dir_base = "/tigress/ljchang/dSph_likelihoods/RunFiles/vel_disp_code/DM_posteriors/"

##############################
# List of posteriors to plot #
##############################

if cusp:
	sim_tag =  "PlumCuspIso" 
else:
	sim_tag = "PlumCoreIso" #

post_to_plot =  [

				[sim_tag+"/df_100_0/rho_Menc_Plummer_NFW_fix_lp",
				sim_tag+"/df_100_0/rho_Menc_Plummer_NFW_float_lp_all"],

				[sim_tag+"/df_100_0/rho_Menc_Plummer_gNFW_fix_lp",
				sim_tag+"/df_100_0/rho_Menc_Plummer_gNFW_float_lp_all"], 

				[sim_tag+"/df_100_0/rho_Menc_Plummer_gNFW_0p02verr",
				sim_tag+"/df_100_0/rho_Menc_Plummer_gNFW_0p2verr", 
				sim_tag+"/df_100_0/rho_Menc_Plummer_gNFW",
				sim_tag+"/df_100_0/rho_Menc_Plummer_gNFW_20verr"], 

				[sim_tag+"/df_100_0/rho_Menc_Plummer_NFW_0p02verr",
				sim_tag+"/df_100_0/rho_Menc_Plummer_NFW_0p2verr", 
				sim_tag+"/df_100_0/rho_Menc_Plummer_NFW",
				sim_tag+"/df_100_0/rho_Menc_Plummer_NFW_20verr"]

				]

labels_list = [
				["'Fixed LP'", "'Float LP'"],
				["'Fixed LP'", "'Float LP'"],
				["'$\Delta v = 0.02$ km/s'", "'$\Delta v = 0.2$ km/s'", "'$\Delta v = 2$ km/s'", "'$\Delta v = 20$ km/s'"],
				["'$\Delta v = 0.02$ km/s'", "'$\Delta v = 0.2$ km/s'", "'$\Delta v = 2$ km/s'", "'$\Delta v = 20$ km/s'"]

]

# labels_list = [
# 				[0, 1],
# 				[0, 1],
# 				[0.02, 0.2, 2, 20],
# 				[0.02, 0.2, 2, 20]

# ]

cuspy_params = [np.log(0.064e9),np.log(1),1,3,1]
cored_params = [np.log(0.4e9),np.log(1),1,3,0]

if cusp:
	true_params_list = [cuspy_params for _ in range(len(post_to_plot))]
else:
	true_params_list = [cored_params for _ in range(len(post_to_plot))]

# true_params_list = [
# 					# [np.log(0.064e9),np.log(1),1,3,1] # For PlumCuspIso
# 					[np.log(0.4e9),np.log(1),1,3,1], # For PlumCoreIso
# 					[np.log(0.4e9),np.log(1),1,3,1], # For PlumCoreIso
# 					[np.log(0.4e9),np.log(1),1,3,1], # For PlumCoreIso
# 					[np.log(0.4e9),np.log(1),1,3,1] # For PlumCoreIso
# ]

# plot_titles_list = ["Plummer stars, NFW DM"]
plot_titles_list = ["NFW", "gNFW", "gNFW", "NFW"]

save_tags_list = ["df_100_0_NFW_lp", "df_100_0_gNFW_lp", "df_100_0_gNFW_deltav", "df_100_0_NFW_deltav"]

assert len(post_to_plot) == len(labels_list)
assert len(post_to_plot) == len(plot_titles_list)
assert len(post_to_plot) == len(save_tags_list)
##############################################################################

for iplot in range(len(post_to_plot)):
	post_list = post_to_plot[iplot]
	labels = labels_list[iplot]
	true_params = true_params_list[iplot]
	title = plot_titles_list[iplot]
	save_tag = save_tags_list[iplot]

	batch2 = "plots_dir="+plots_dir_base+sim_tag+"/composite/"+"\n"

	if len(title) != 0:
		batch2 = batch2+"title="+title+"\n"
	if len(save_tag) != 0:
		batch2 = batch2+"save_tag="+save_tag+"\n"

	batch3 = "python plot_interface.py --plots_dir $plots_dir "
	if len(title) != 0:
		batch3 = batch3 + "--title $title "
	if len(save_tag) != 0:
		batch3 = batch3+"--save_tag $save_tag "

	batch3 = batch3 + "--post_paths_list "
	for ipost in range(len(post_list)):
		batch3 = batch3+post_dir_base+post_list[ipost]+".npz "

	batch3 = batch3+"--plot_labels " 
	for ipost in range(len(post_list)):
		batch3 = batch3+str(labels[ipost])+" "

	batch3 = batch3+"--true_params "
	for ipost in range(len(true_params)):
		batch3 = batch3+str(true_params[ipost])+" "

	batchn = batch1+batch2+batch3
	fname = "./batch/comp_plot_"+save_tag+".batch"

	f=open(fname, "w")
	f.write(batchn)
	f.close()
	os.system("sbatch "+fname);

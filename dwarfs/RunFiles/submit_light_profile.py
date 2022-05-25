import os, sys
import numpy as np

batch1='''#!/bin/bash
#SBATCH -N 1   # node count
#SBATCH --ntasks-per-node=1
#SBATCH -t 5:00:00
#SBATCH --mem=16GB
##SBATCH --mail-type=begin
##SBATCH --mail-type=end
##SBATCH --mail-user=ljchang@princeton.edu

cd /tigress/ljchang/dSph_likelihoods/RunFiles/
		'''

#####################
# Light profile fit #
#####################

chains_dir = "/tigress/ljchang/dSph_likelihoods/RunFiles/multinest_chains/"
if not os.path.exists(chains_dir):
	os.makedirs(chains_dir)
plots_dir_base = "/tigress/ljchang/dSph_likelihoods/Plots/light_profile_plots/"
plots_extra_text = ""

# extra_text = "PlumCuspIso"
# data_folder = "/tigress/ljchang/dSph_likelihoods/RunFiles/data/gaia_challenge/gs010_bs050_rcrs025_rarcinf_cusp_0064mpc3_df/processed_samps/poiss_err/"
# has_truth = True
# true_params = [0,np.log10(0.25),2,5,0.1]

# extra_text = "PlumCoreIso"
# data_folder = "/tigress/ljchang/dSph_likelihoods/RunFiles/data/gaia_challenge/gs010_bs050_rcrs100_rarcinf_core_0400mpc3_df/processed_samps/poiss_err/"
# has_truth = True
# true_params = [0,np.log10(1),2,5,0.1]

# extra_text = "Tucana"
# data_folder = "/tigress/ljchang/dSph_likelihoods/RunFiles/data/tucana/"
# has_truth = False

###########################################################################################
nstars = 100

extra_text = "PlumCuspIso"
data_folder = "/tigress/ljchang/dSph_likelihoods/RunFiles/data/lina_mocks/PlumCuspIso/"
plots_extra_text = "/lina_mocks/"
has_truth = True

data_tag_base = "rs_025_df_"+str(nstars)+"_"
true_params = [0,np.log10(0.25),2,5,0.1]

# data_tag_base = "rs_100_df_"+str(nstars)+"_"
# true_params = [0,np.log10(1),2,5,0.1]
############################################################################################

run_scan = True
postprocess = True

# for data_tag in ["tuc_old_measurement","tuc_old_measurement_2bins"]:
for iMC in range(5):
	data_tag = "df_1000_"+str(iMC)
# for data_tag in ["df_100_0","df_100_1","df_100_2","df_100_3","df_100_4","df_100_5","df_100_6","df_100_7","df_100_8","df_100_9"]:
	# data_tag = data_tag_base+str(iMC)

	data_file_path = data_folder+data_tag+"_lp.npz"

	save_dir = chains_dir+extra_text+"/"
	quantiles_dir = data_folder+"light_profile_bestfits/"

	if not os.path.exists(save_dir):
		os.makedirs(save_dir)

	if not os.path.exists(quantiles_dir):
		os.makedirs(quantiles_dir)

	run_tag = "_1comp"
	nu_model = "Plummer"

	# run_tag = "_2comp"
	# nu_model = "Plummer2"

	# run_tag = "_Zhao"
	# nu_model = "Zhao"

	outfile_path = save_dir+data_tag+run_tag+"/"
	quantiles_path = quantiles_dir+data_tag+run_tag
	plots_dir = plots_dir_base+plots_extra_text+extra_text+"/"+data_tag+"/"

	if not os.path.exists(plots_dir):
		os.makedirs(plots_dir)

	batch2 = "data_file_path="+data_file_path+"\n"+"outfile_path="+outfile_path+"\n"+"quantiles_path="+quantiles_path+"\n"\
			+"nu_model="+nu_model+"\n"+"plots_dir="+plots_dir+"\n"\
			+"run_scan="+str(run_scan)+"\n"+"postprocess="+str(postprocess)+"\n"+"has_truth="+str(has_truth)+"\n"
	batch3 = "python scan_light_profile_interface.py --data_file_path $data_file_path --outfile_path $outfile_path --quantiles_path $quantiles_path --has_truth $has_truth "
	batch4 = "--nu_model $nu_model --plots_dir $plots_dir --run_scan $run_scan --postprocess $postprocess "
	batchn = batch1+batch2+batch3+batch4

	fname = "./batch/scan_multinest_"+data_tag+run_tag+"_"+extra_text+".batch"
	f=open(fname, "w")
	f.write(batchn)
	f.close()
	os.system("sbatch "+fname);


#######################
# sigmap observed fit #
#######################

# # extra_text = "PlumCuspIso"
# # data_folder = "/tigress/ljchang/dSph_likelihoods/RunFiles/data/gaia_challenge/gs010_bs050_rcrs025_rarcinf_cusp_0064mpc3_df/processed_samps/poiss_err/"

# # extra_text = "PlumCoreIso"
# # data_folder = "/tigress/ljchang/dSph_likelihoods/RunFiles/data/gaia_challenge/gs010_bs050_rcrs100_rarcinf_core_0400mpc3_df/processed_samps/poiss_err/"

# extra_text = "Tucana"
# data_folder = "/tigress/ljchang/dSph_likelihoods/RunFiles/data/tucana/"

# run_scan = False
# postprocess = False
# fit_sigmap = True

# nbins = 3
# deltav = 2.
# measured_errs = True

# extra_text = ""

# for data_tag in ["tuc"]:
# # for data_tag in ["df_100_0","df_100_1","df_100_2","df_100_3","df_100_4","df_100_5","df_100_6","df_100_7","df_100_8","df_100_9"]:
# 	data_file_path = data_folder+data_tag+".npz"

# 	save_dir = data_folder+"/sigmap_bestfits/"

# 	if not os.path.exists(save_dir):
# 		os.makedirs(save_dir)

# 	outfile_path = save_dir+data_tag+extra_text

# 	batch2 = "data_file_path_sigmap="+data_file_path+"\n"+"outfile_path_sigmap="+outfile_path+"\n"\
# 			+"run_scan="+str(run_scan)+"\n"+"postprocess="+str(postprocess)+"\n"+"fit_sigmap="+str(fit_sigmap)+"\n"\
# 			+"nbins="+str(nbins)+"\n"+"deltav="+str(deltav)+"\n"+"measured_errs="+str(measured_errs)+"\n"
# 	batch3 = "python scan_light_profile_interface.py --data_file_path_sigmap $data_file_path_sigmap --outfile_path_sigmap $outfile_path_sigmap "
# 	batch4 = "--run_scan $run_scan --postprocess $postprocess --fit_sigmap $fit_sigmap --nbins $nbins --deltav $deltav --measured_errs $measured_errs "
# 	batchn = batch1+batch2+batch3+batch4

# 	fname = "./batch/scan_sigmap_"+data_tag+"_"+extra_text+".batch"
# 	f=open(fname, "w")
# 	f.write(batchn)
# 	f.close()
# 	os.system("sbatch "+fname);
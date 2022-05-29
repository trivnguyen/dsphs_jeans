import os, sys

batch1='''#!/bin/bash
##SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 6:00:00
#SBATCH --mem=50GB
##SBATCH --mail-type=begin
##SBATCH --mail-type=end
#SBATCH --mail-user=ljchang@princeton.edu

cd /tigress/ljchang/dSph_likelihoods/RunFiles/vel_disp_code/

		'''

chains_dir_base = "/tigress/ljchang/dSph_likelihoods/RunFiles/vel_disp_code/multinest_chains/"

if not os.path.exists(chains_dir_base):
	os.makedirs(chains_dir_base)

data_dir_base = "/tigress/ljchang/dSph_likelihoods/RunFiles/data/gaia_challenge/"
plots_dir_base = "/tigress/ljchang/dSph_likelihoods/Plots/DM_plots/"
post_dir_base = "/tigress/ljchang/dSph_likelihoods/RunFiles/vel_disp_code/DM_posteriors/"

#################################################
# Comment out simulation tags that won't be run #
#################################################

sim_tag_list =  [
				["PlumCuspIso","gs010_bs050_rcrs025_rarcinf_cusp_0064mpc3_df"],
				["PlumCoreIso","gs010_bs050_rcrs100_rarcinf_core_0400mpc3_df"],
				# ["PlumCuspOm","gs010_bs050_rcrs010_rarc100_cusp_0064mpc3_df"],
				# ["PlumCoreOm","gs010_bs050_rcrs025_rarc100_core_0400mpc3_df"]
				]

###########################################
# Comment out data tags that won't be run #
###########################################

data_tag_list = []
for iMC in range(5):
	data_tag_list += ["df_1000_"+str(iMC)]
# data_tag_list = ["df_100_0","df_100_1","df_100_2","df_100_3","df_100_4","df_100_5","df_100_6","df_100_7","df_100_8","df_100_9"]
# data_tag_list = ["df_10000_0"]

#######################################################
# Comment out light profile models that won't be run #
#######################################################

nu_model_list = [
				["Plummer","_1comp"],
				# ["Plummer2","_2comp"],
				# ["Zhao","_Zhao"]
				]

###################################################
# Comment out DM density models that won't be run #
###################################################

dm_model_list = [
				# "BPL",
				# "BPL2",
				# "BPL3",
				# "Zhao",
				"gNFW",
				"NFW",
				# "Burkert",
				"NFWc",
				# "gNFW_trunc"
				]

#####################################
# Define global params for all runs #
#####################################

nbeta = 0
deltav = 2.

fix_breaks = False
fix_light_profile = False

load_light_profile = True

##############################################################################
# This block of code figures out which temporary index to start at 
# (to ensure that existing temporary directories don't get overwritten into) 

# dirs = os.listdir('/tigress/ljchang/dSph_likelihoods/RunFiles/temp/')
# temp_ind_base = 0
# max_existing = 0

# for i in range(len(dirs)):
#     if int(dirs[i][1:]) > max_existing:
#         max_existing = int(dirs[i][1:])

# max_existing += 1
# temp_ind = temp_ind_base + max_existing

temp_ind = 150

##############################################################################

run_scan = False
make_plots = True

# These arguments are for the plotting code
save_plots = True
save_post = True
has_truth = True

run_tag = ""

for extratext, sim_tag in sim_tag_list:
	for data_tag in data_tag_list:
		for nu_model, lp_tag in nu_model_list:
			for dm_model in dm_model_list:		

				data_file_path = data_dir_base+sim_tag+"/processed_samps/poiss_err/"+data_tag+".npz"
				light_profile_params_path = data_dir_base+sim_tag+"/processed_samps/poiss_err/light_profile_bestfits/"+data_tag+lp_tag+".npz"

				chains_dir = chains_dir_base+"/"+extratext+"/"+data_tag+"_"+nu_model+"_"+dm_model+run_tag
				sigmap_path = data_dir_base+sim_tag+"/processed_samps/poiss_err/sigmap_bestfits/"+data_tag+run_tag
				
				plots_dir = plots_dir_base+extratext+"/"+data_tag+"/"
				post_dir = post_dir_base+extratext+"/"+data_tag+"/"

				if not os.path.exists(chains_dir):
					os.makedirs(chains_dir)

				# if os.path.exists(chains_dir+"/post_equal_weights.dat"):
				# 	break

				if not os.path.exists(plots_dir):
					os.makedirs(plots_dir)

				if not os.path.exists(post_dir):
					os.makedirs(post_dir)				

				batch2 = "chains_dir="+chains_dir+"\n"+"data_file_path="+data_file_path+"\n"+"light_profile_params_path="+light_profile_params_path+"\n"+"sigmap_path="+sigmap_path+"\n"\
						+"load_light_profile="+str(load_light_profile)+"\n"+"fix_light_profile="+str(fix_light_profile)+"\n"+"nu_model="+nu_model+"\n"\
						+"dm_model="+dm_model+"\n"+"nbeta="+str(nbeta)+"\n"+"fix_breaks="+str(fix_breaks)+"\n"+"deltav="+str(deltav)+"\n"+"temp_ind="+str(temp_ind)+"\n"\
						+"run_scan="+str(run_scan)+"\n"+"make_plots="+str(make_plots)+"\n"+"plots_dir="+plots_dir+"\n"+"true_profile="+extratext+"\n"\
						+"save_plots="+str(save_plots)+"\n"+"save_post="+str(save_post)+"\n"+"has_truth="+str(has_truth)+"\n"+"post_dir="+post_dir+"\n"
				if len(run_tag) > 0:
					batch2 = batch2+"run_tag="+run_tag+"\n"
				batch3 = "python scan_vel_disp_interface.py --setup_dirs --chains_dir $chains_dir --temp_ind $temp_ind "+"\n"
				batch4 = "mpiexec -np 56 python scan_vel_disp_interface.py --run_scan $run_scan --chains_dir $chains_dir --data_file_path $data_file_path --light_profile_params_path $light_profile_params_path "
				batch5 = "--load_light_profile $load_light_profile --fix_light_profile $fix_light_profile --nu_model $nu_model --dm_model $dm_model --nbeta $nbeta --fix_breaks $fix_breaks --deltav $deltav --temp_ind $temp_ind "+"\n"
				batch6 = "python scan_vel_disp_interface.py --cleanup_dirs --chains_dir $chains_dir --temp_ind $temp_ind --make_plots $make_plots --plots_dir $plots_dir "+"\n"
				batch7 = "python scan_vel_disp_interface.py --make_plots $make_plots --save_plots $save_plots --save_post $save_post --has_truth $has_truth "
				batch8 = "--chains_dir $chains_dir --plots_dir $plots_dir --post_dir $post_dir --data_file_path $data_file_path --light_profile_params_path $light_profile_params_path --sigmap_path $sigmap_path "
				batch9 = "--load_light_profile $load_light_profile --fix_light_profile $fix_light_profile --nu_model $nu_model --dm_model $dm_model --nbeta $nbeta --fix_breaks $fix_breaks --deltav $deltav --true_profile $true_profile "
				if len(run_tag) > 0:
						batch8 = batch8+"--run_tag $run_tag "
				if run_scan and make_plots:
					batchn = batch1+batch2+batch3+batch4+batch5+batch6+batch7+batch8+batch9
					fname = "./batch/scan_plot_"+extratext+"_"+data_tag+"_"+nu_model+"_"+dm_model+"_"+str(nbeta)+run_tag+"_"+str(temp_ind)+".batch"
				elif run_scan:
					batchn = batch1+batch2+batch3+batch4+batch5+batch6
					fname = "./batch/scan_"+extratext+"_"+data_tag+"_"+nu_model+"_"+dm_model+"_"+str(nbeta)+run_tag+"_"+str(temp_ind)+".batch"
				elif make_plots:
					batchn = batch1+batch2+batch7+batch8+batch9
					fname = "./batch/plot_"+extratext+"_"+data_tag+"_"+nu_model+"_"+dm_model+"_"+str(nbeta)+run_tag+"_"+str(temp_ind)+".batch"

				f=open(fname, "w")
				f.write(batchn)
				f.close()
				os.system("sbatch "+fname);

				temp_ind += 1

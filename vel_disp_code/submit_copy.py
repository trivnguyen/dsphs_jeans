import os, sys
import numpy as np

batch1='''#!/bin/bash
#SBATCH -n 1
#SBATCH -t 6:00:00
#SBATCH --mem=50GB
#SBATCH -p physics
##SBATCH --mail-type=begin
##SBATCH --mail-type=end
#SBATCH --mail-user=ljchang@princeton.edu

cd /tigress/ljchang/dSph_likelihoods/RunFiles/vel_disp_code/

		'''

chains_dir_base = "/tigress/ljchang/dSph_likelihoods/RunFiles/vel_disp_code/clean_chains/"

if not os.path.exists(chains_dir_base):
	os.makedirs(chains_dir_base)

data_dir_base = "/tigress/ljchang/dSph_likelihoods/Generate_mocks/mocks/"
plots_dir_base = "/tigress/ljchang/dSph_likelihoods/Plots/DM_plots/clean/"
post_dir_base = "/tigress/ljchang/dSph_likelihoods/RunFiles/vel_disp_code/DM_posteriors/clean/"

#################################################
# Comment out simulation tags that won't be run #
#################################################

sim_tag_list =  [
				# ["a_1_b_3_g_1_rdm_1_iso",[64.0,1.0,1.0,3.0,1.0]],
				# ["a_1_b_3_g_1_rdm_0.2_iso",[64.0,0.2,1.0,3.0,1.0]],
				# ["a_1_b_3_g_0_rdm_1_iso",[64.,1.0,1.0,3.0,0.0]],
				["a_1_b_3_g_0_rdm_0.2_iso",[64.,0.2,1.0,3.0,0.0]],

				]

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
				#"Burkert",
				"NFWc",
				# "gNFW_trunc",
				]

#####################################
# Define global params for all runs #
#####################################

nbeta = 0
deltav_list = [
			  [0.,"_no_err","_0verr"],
			  [0.2,"del_v_0.2","_0p2verr"],
			  # [2.0,"del_v_2.0","_2verr"],
			  [5.0,"del_v_5.0","_5verr"],
			  [10.0,"del_v_10.0","_10verr"]
			  ]

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

temp_ind = 500
# 
##############################################################################

# nstars_list = [20,100,1000,10000]
nstars_list = [10000]

run_scan = False
make_plots = True

# These arguments are for the plotting code
save_plots = True
save_post = True
has_truth = True

# run_tag = "rho0_"+str(true_params_base[0])
# rs_dm = 1.0

# subsamp_tag_list = ["_low_1","_low_2","_low_5","_low_10"]
# subsamp_tag_list = ["_high_0.1","_high_0.5","_high_1"]
subsamp_tag_list = [""]

# rs_tag = "rs_1kpc"
rs_tag = ""
for extratext, true_params_base in sim_tag_list:
	for nu_model, lp_tag in nu_model_list:
		for dm_model in dm_model_list:		
			for nstars in nstars_list:
				for deltav, dvtag, dvtag_out in deltav_list:
					for subsamp_tag in subsamp_tag_list:
						for iMC in range(10,20):
						# for iMC in [19]:
							rho0 = true_params_base[0]
							rs_dm = true_params_base[1]
							rs_stars = rs_dm
							# run_tag = "_rho0_"+str(rho0)
							# run_tag = "_symm_priors"
							# run_tag = "_fix_rho0_r0"
							run_tag = ""

							##############################################################
							# LINA CHANGED NAMING SCHEME, THIS IS FOR GAMMA=0, RS=1 ONLY #
							##############################################################
							if rs_stars == 1.0:
								data_tag_base = "dist_as_2.0_bs_5.0_gs_0.1_rs_"+str(round(rs_stars))+"_dm_"+str(nstars)+"_"+str(iMC)+"_rho_"+str(rho0)
							else:
								data_tag_base = "dist_as_2.0_bs_5.0_gs_0.1_rs_"+str((rs_stars))+"_dm_"+str(nstars)+"_"+str(iMC)+"_rho_"+str(rho0)
							
							out_tag_base = "df_"+str(nstars)+"_"+str(iMC)+dvtag_out+subsamp_tag
							data_tag = data_tag_base+dvtag+subsamp_tag
							true_params = [np.log(rho0*1e6),np.log(rs_dm)]+true_params_base[2:]

							data_file_path = data_dir_base+extratext+"/"+data_tag+".npz"
							full_file_path = data_dir_base+extratext+"/"+data_tag_base+".dat"
							light_profile_params_path = data_dir_base+extratext+"/light_profile_bestfits/"+data_tag_base+lp_tag+".npz"

							chains_dir = chains_dir_base+"/"+extratext+"/"+rs_tag+"/"+out_tag_base+"_"+nu_model+"_"+dm_model+run_tag
							# chains_dir = chains_dir_base+"/"+extratext+"/"+out_tag_base+"_"+nu_model+"_"+dm_model+run_tag
							sigmap_path = data_dir_base+extratext+"/sigmap_bestfits/"+data_tag+run_tag
							
							plots_dir = plots_dir_base+extratext+"/"+rs_tag+"/"+out_tag_base+"/"
							post_dir = post_dir_base+extratext+"/"+rs_tag+"/"+out_tag_base+"/"

							if not os.path.exists(chains_dir):
								os.makedirs(chains_dir)

							if not os.path.exists(plots_dir):
								os.makedirs(plots_dir)

							if not os.path.exists(post_dir):
								os.makedirs(post_dir)		
											
							# if not os.path.exists(chains_dir+"/post_equal_weights.dat"):
							batch2 = "chains_dir="+chains_dir+"\n"+"data_file_path="+data_file_path+"\n"+"full_file_path="+full_file_path+"\n"+"light_profile_params_path="+light_profile_params_path+"\n"+"sigmap_path="+sigmap_path+"\n"\
									+"load_light_profile="+str(load_light_profile)+"\n"+"fix_light_profile="+str(fix_light_profile)+"\n"+"nu_model="+nu_model+"\n"\
									+"dm_model="+dm_model+"\n"+"nbeta="+str(nbeta)+"\n"+"fix_breaks="+str(fix_breaks)+"\n"+"deltav="+str(deltav)+"\n"+"temp_ind="+str(temp_ind)+"\n"\
									+"run_scan="+str(run_scan)+"\n"+"make_plots="+str(make_plots)+"\n"+"plots_dir="+plots_dir+"\n"+"\n"\
									+"save_plots="+str(save_plots)+"\n"+"save_post="+str(save_post)+"\n"+"has_truth="+str(has_truth)+"\n"+"post_dir="+post_dir+"\n"
							if len(run_tag) > 0:
								batch2 = batch2+"run_tag="+run_tag+"\n"
							batch3 = "python scan_vel_disp_interface_copy.py --setup_dirs --chains_dir $chains_dir --temp_ind $temp_ind "+"\n"
							batch4 = "mpiexec -np 60 python scan_vel_disp_interface_copy.py --run_scan $run_scan --chains_dir $chains_dir --data_file_path $data_file_path --full_file_path $full_file_path --light_profile_params_path $light_profile_params_path "
							batch5 = "--load_light_profile $load_light_profile --fix_light_profile $fix_light_profile --nu_model $nu_model --dm_model $dm_model --nbeta $nbeta --fix_breaks $fix_breaks --deltav $deltav --temp_ind $temp_ind "
							batch6 = "python scan_vel_disp_interface_copy.py --cleanup_dirs --chains_dir $chains_dir --temp_ind $temp_ind --make_plots $make_plots --plots_dir $plots_dir "+"\n"
							batch7 = "python scan_vel_disp_interface_copy.py --make_plots $make_plots --save_plots $save_plots --save_post $save_post --has_truth $has_truth "
							batch8 = "--chains_dir $chains_dir --plots_dir $plots_dir --post_dir $post_dir --data_file_path $data_file_path --full_file_path $full_file_path --light_profile_params_path $light_profile_params_path --sigmap_path $sigmap_path "
							batch9 = "--load_light_profile $load_light_profile --fix_light_profile $fix_light_profile --nu_model $nu_model --dm_model $dm_model --nbeta $nbeta --fix_breaks $fix_breaks --deltav $deltav "
							batch10 = "--true_params "
							for i in range(len(true_params)):
								batch10 = batch10+str(true_params[i])+" "
							batch10 = batch10+"\n"
							if len(run_tag) > 0:
									batch8 = batch8+"--run_tag $run_tag "
							if run_scan and make_plots:
								batchn = batch1+batch2+batch3+batch4+batch5+batch10+batch6+batch7+batch8+batch9+batch10
								fname = "./batch/scan_plot_"+str(temp_ind)+"_"+extratext+"_"+data_tag+"_"+nu_model+"_"+dm_model+"_"+str(nbeta)+run_tag+".batch"
							elif run_scan:
								batchn = batch1+batch2+batch3+batch4+batch5+batch10+batch6
								fname = "./batch/scan_"+str(temp_ind)+"_"+extratext+"_"+data_tag+"_"+nu_model+"_"+dm_model+"_"+str(nbeta)+run_tag+".batch"
							elif make_plots:
								batchn = batch1+batch2+batch7+batch8+batch9+batch10
								fname = "./batch/plot_"+str(temp_ind)+"_"+extratext+"_"+data_tag+"_"+nu_model+"_"+dm_model+"_"+str(nbeta)+run_tag+".batch"

							f=open(fname, "w")
							f.write(batchn)
							f.close()
							os.system("sbatch "+fname);

							temp_ind += 1

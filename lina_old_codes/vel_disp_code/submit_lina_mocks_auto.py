import os, sys
import numpy as np
sys.path.append("/tigress/lnecib/dSph_likelihoods/Generate_mocks")
import generation_functions

user = 'lnecib' # "ljchang"

nsample = int(sys.argv[1])

i_initial = 10
ntotal = 10

same_parameters = True
run_scan = False
make_plots = False 
calculate_j_factor = True

Laura_directory = True #If true it switches to the formatting of Laura's directories

# These arguments are for the plotting code
save_plots = True
save_post = True
has_truth = True



if nsample < 1000:
	n_nodes = 1
	n_cpus = 40
	time = "00:30:00"
	memory = "8GB"
elif nsample < 9999:
	n_nodes = 3
	n_cpus = 120
	time = "12:00:00"
else:
	n_nodes = 3
	n_cpus = 120
	time = "36:00:00"
	memory = "50GB"


batch1_setup='#!/bin/bash' + '\n' + \
'#SBATCH -N ' + str(n_nodes) + '\n' + \
'#SBATCH -n ' + str(n_cpus) + '\n' + \
'#SBATCH -t ' + time + '\n' + \
'#SBATCH --mem-per-cpu=2GB' + '\n' 


batch1_setup_plot='#!/bin/bash' + '\n' + \
'#SBATCH -N 1'  + '\n' + \
'#SBATCH -n 1'  + '\n' + \
'#SBATCH -t ' + "00:30:00" + '\n' + \
'#SBATCH --mem-per-cpu=2GB' + '\n' 

batch1_setup_jfac='#!/bin/bash' + '\n' + \
'#SBATCH -N 1'  + '\n' + \
'#SBATCH -n 1'  + '\n' + \
'#SBATCH -t ' + "02:30:00" + '\n' + \
'#SBATCH --mem-per-cpu=2GB' + '\n' 

batch_email= '''
#SBATCH --mail-type=begin
#SBATCH --mail-type=end
'''

batch1_packages='''
#SBATCH --mail-user=lnecib@caltech.edu

module load anaconda3 openmpi/gcc/3.1.3/64
cd /home/lnecib/.conda/envs/dwarf3/bin
source activate dwarf3

'''
if nsample > 9999:
	batch1 = batch1_setup + batch_email + batch1_packages
else:
	batch1 = batch1_setup + batch1_packages

location = "cd /tigress/" + user + "/dSph_likelihoods/RunFiles/vel_disp_code/" + "\n"

batch1+= location

batch_plot = batch1_setup_plot + batch1_packages + location
batch_jfac = batch1_setup_jfac + batch1_packages + location

if Laura_directory:
	chains_dir_base = "/tigress/" + user + "/dSph_likelihoods/RunFiles/vel_disp_code/clean_chains/"
else:
	chains_dir_base = "/tigress/" + user + "/dSph_likelihoods/RunFiles/vel_disp_code/multinest_chains/mocks/"


if not os.path.exists(chains_dir_base):
	os.makedirs(chains_dir_base)

data_dir_base = "/tigress/" + user + "/dSph_likelihoods/Generate_mocks/mocks/"
plots_dir_base = "/tigress/" + user + "/dSph_likelihoods/Plots/DM_plots/mocks/"
post_dir_base = "/tigress/" + user + "/dSph_likelihoods/RunFiles/vel_disp_code/DM_posteriors/mocks/"
run_dir_base = "/tigress/" + user + "/dSph_likelihoods/RunFiles/batch/"
if not os.path.exists(run_dir_base):
	os.makedirs(run_dir_base)

temp_directory = "/tigress/" + user + "/dSph_likelihoods/RunFiles/temp/"
if not os.path.exists(temp_directory):
	os.makedirs(temp_directory)

#################################################
# Comment out simulation tags that won't be run #
#################################################

# sim_tag_list =  [
# 				["PlumCuspIso",[1,3,1]],
# 				["PlumCoreIso",[1,3,0]],
# 				# ["PlumCuspOm","gs010_bs050_rcrs010_rarc100_cusp_0064mpc3_df"],
# 				# ["PlumCoreOm","gs010_bs050_rcrs025_rarc100_core_0400mpc3_df"]
# 				]

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
				"NFWc",
				#"Burkert",
				# "gNFW_trunc",
				]

#####################################
# Define global params for all runs #
#####################################

nbeta = 0

fix_breaks = False
fix_light_profile = False

load_light_profile = True

##############################################################################
# This block of code figures out which temporary index to start at 
# (to ensure that existing temporary directories don't get overwritten into) 

dirs = os.listdir('/tigress/' + user + '/dSph_likelihoods/RunFiles/temp/')
temp_ind_base = 0
max_existing = 0

for i in range(len(dirs)):
    if int(dirs[i][1:]) > max_existing:
        max_existing = int(dirs[i][1:])

max_existing += 1
temp_ind = temp_ind_base + max_existing

##############################################################################



### Specify alpha/beta/gamma and scale radius (in kpc) of tracer density profile

alphalight=2.
betalight=5.
gammalight=0.1

# Dark Matter

gamma_dm_list =  np.array([0, 0, 1, 1]) #,0.5, 0.5, 1.5, 1.5]) #[0, 0,
rs_list = np.array([1, 0.2, 1, 0.2]) #,1, 0.2, 1, 0.2]) #,  [1, 0.2, 
rlight_list = rs_list


if same_parameters:
	# rlight_param = 1.
	rho_param = 0.064e9
	alpha_param = 1
	beta_param = 3
	# rlight_list = [rlight_param for _ in range(len(gamma_dm_list)) ]
	rho_list = [rho_param for _ in range(len(gamma_dm_list)) ]
	alpha_dm_list = [alpha_param for _ in range(len(gamma_dm_list)) ]
	beta_dm_list = [beta_param for _ in range(len(gamma_dm_list)) ]
	
else:
	rlight_list=[1., 1., 1., 1.]#kpc
	alpha_dm_list = [1, 1, 1, 1] 
	beta_dm_list = [3, 3, 3, 3] 
	rho_list = [0.064e9, 0.064e9, 0.064e9, 0.064e9] #M_sun/kpc^3

###specify osipkov-merritt anisotropy parameter in units of tracer scale radius

ra_list = [1.e+30] #, 0.5]#kpc
anisotropy = ['' for _ in range(len(ra_list))]
for s,ra in enumerate(ra_list):
	if ra == 1.e+30:
		anisotropy[s] = 'iso'
	else:
		anisotropy[s] = str(ra)


n_distribution = 1 # -1

## Measurement Errors
errors = [2.]#[0, 0.2, 2., 5., 10.]
error_list_tag = [generation_functions.err_filename('', vel_err) for vel_err in errors ]

error_list_tag_Laura = ['2'] #['0', '0p2', '2', '5', '10']

n_extension = -4

##############################################################################
## Jfactor caluclations
distance = 50 #kpc
angle_max = 0.5 #Degrees
full_posterior = True #Calculates the J-factor for the entire posterior

##############################################################################

# rs_list = [1, 1, 1, 1] #[0.25,1]
rho0_list = np.array(rho_list) #[64,400]

run_tag = ""

# Start with no error runs
for k in range(len(alpha_dm_list)):
	print("Starting at k", k)

	true_params = [np.log(rho0_list[k]), np.log(rs_list[k]), alpha_dm_list[k], beta_dm_list[k], gamma_dm_list[k]]	

	directory = generation_functions.set_directory(rs = rs_list[k], alpha = alpha_dm_list[k], beta = beta_dm_list[k], gamma = gamma_dm_list[k], anisotropy = anisotropy[0])

	for s, err_tag in enumerate(error_list_tag):
		deltav = errors[s]
		for nu_model, lp_tag in nu_model_list:
			for dm_model in dm_model_list:		
				for iMC in range(i_initial, i_initial + ntotal):
					filename = generation_functions.get_preprocess_filename(nsample, rs = rs_list[k], alpha = alpha_dm_list[k], beta = beta_dm_list[k], gamma = gamma_dm_list[k], n_distribution = iMC, ntotal = ntotal, rlight = rlight_list[k], rho = rho_list[k], data_dir = data_dir_base)
					filename_no_extension = filename[:n_extension]

					data_tag = generation_functions.get_preprocess_filename(nsample, rs = rs_list[k], alpha = alpha_dm_list[k], beta = beta_dm_list[k], gamma = gamma_dm_list[k], n_distribution = iMC, ntotal = ntotal, rlight = rlight_list[k], rho = rho_list[k], data_dir = data_dir_base, simple = True)

					if Laura_directory:
						data_tag_chains = 'df_' + str(nsample) + '_' + str(iMC) + '_' + str(error_list_tag_Laura[s]) + 'verr' 
					else:
						data_tag_chains = data_tag+ err_tag 

					data_file_path = data_dir_base + directory + data_tag  + err_tag + ".npz"
					full_file_path = data_dir_base + directory + data_tag + ".dat"
					light_profile_params_path = data_dir_base + directory +"light_profile_bestfits/"+ data_tag + lp_tag + ".npz"

					chains_dir = chains_dir_base + directory + data_tag_chains  +  "_" + nu_model + "_" + dm_model + run_tag
					sigmap_path = data_dir_base + directory + "sigmap_bestfits/"+ data_tag  + err_tag+run_tag
					
					plots_dir = plots_dir_base + directory + data_tag  + err_tag + "/"
					post_dir = post_dir_base + directory + data_tag  + err_tag + "/"
					batch_dir = run_dir_base + directory + data_tag  + err_tag + "/"

					if not os.path.exists(chains_dir):
						os.makedirs(chains_dir)

					if not os.path.exists(plots_dir):
						os.makedirs(plots_dir)

					if not os.path.exists(post_dir):
						os.makedirs(post_dir)		

					if not os.path.exists(batch_dir):
						os.makedirs(batch_dir)
									
					if not os.path.exists(chains_dir+"/post_equal_weights.dat"):
						print("This file does not exist!!!", chains_dir+"/post_equal_weights.dat")
						batch2 = "chains_dir="+chains_dir+"\n"+"data_file_path="+data_file_path+"\n"+"full_file_path="+full_file_path+"\n"+"light_profile_params_path="+light_profile_params_path+"\n"+"sigmap_path="+sigmap_path+"\n"\
								+"load_light_profile="+str(load_light_profile)+"\n"+"fix_light_profile="+str(fix_light_profile)+"\n"+"nu_model="+nu_model+"\n"\
								+"dm_model="+dm_model+"\n"+"nbeta="+str(nbeta)+"\n"+"fix_breaks="+str(fix_breaks)+"\n"+"deltav="+str(deltav)+"\n"+"temp_ind="+str(temp_ind)+"\n"\
								+"run_scan="+str(run_scan)+"\n"+"make_plots="+str(make_plots)+"\n"+"plots_dir="+plots_dir+"\n"+"\n"\
								+"save_plots="+str(save_plots)+"\n"+"save_post="+str(save_post)+"\n"+"has_truth="+str(has_truth)+"\n"+"post_dir="+post_dir+"\n" \
								+"temp_directory=" + temp_directory + "\n"
						if len(run_tag) > 0:
							batch2 = batch2+"run_tag="+run_tag+"\n"
						batch3 = "python scan_vel_disp_interface.py --setup_dirs --chains_dir $chains_dir --temp_directory $temp_directory --temp_ind $temp_ind "+"\n"
						batch4 = "mpiexec -np 28 python scan_vel_disp_interface.py --run_scan $run_scan --chains_dir $chains_dir --data_file_path $data_file_path --full_file_path $full_file_path --light_profile_params_path $light_profile_params_path "
						batch5 = "--load_light_profile $load_light_profile --fix_light_profile $fix_light_profile --nu_model $nu_model --dm_model $dm_model --nbeta $nbeta --fix_breaks $fix_breaks --deltav $deltav --temp_directory $temp_directory --temp_ind $temp_ind "+"\n"
						batch6 = "python scan_vel_disp_interface.py --cleanup_dirs --chains_dir $chains_dir --temp_directory $temp_directory --temp_ind $temp_ind --make_plots $make_plots --plots_dir $plots_dir "+"\n"
						batch7 = "python scan_vel_disp_interface.py --make_plots $make_plots --save_plots $save_plots --save_post $save_post --has_truth $has_truth "
						batch8 = "--chains_dir $chains_dir --plots_dir $plots_dir --post_dir $post_dir --data_file_path $data_file_path --full_file_path $full_file_path  --light_profile_params_path $light_profile_params_path --sigmap_path $sigmap_path "
						batch9 = "--load_light_profile $load_light_profile --fix_light_profile $fix_light_profile --nu_model $nu_model --dm_model $dm_model --nbeta $nbeta --fix_breaks $fix_breaks --deltav $deltav "
						batch10 = "--true_params "
						for i in range(len(true_params)):
							batch10 = batch10+str(true_params[i])+" "
						if len(run_tag) > 0:
								batch8 = batch8+"--run_tag $run_tag "

						if run_scan or make_plots:
							if run_scan and make_plots:
								batchn = batch1+batch2+batch3+batch4+batch5+batch6+batch7+batch8+batch9+batch10
								fname = batch_dir + "/scan_plot_"+data_tag + err_tag +"_"+nu_model+"_"+dm_model+"_"+str(nbeta)+run_tag+".batch"
							elif run_scan:
								batchn = batch1+batch2+batch3+batch4+batch5+batch6
								fname = batch_dir + "/scan_"+data_tag + err_tag +"_"+nu_model+"_"+dm_model+"_"+str(nbeta)+run_tag+".batch"
							elif make_plots:
								batchn = batch1+batch2+batch7+batch8+batch9+batch10
								fname = batch_dir + "/plot_"+data_tag + err_tag +"_"+nu_model+"_"+dm_model+"_"+str(nbeta)+run_tag+".batch"

							f=open(fname, "w")
							f.write(batchn)
							f.close()
							os.system("sbatch "+fname);

							temp_ind += 1

					else:
						print("This file", chains_dir+"/post_equal_weights.dat", "should exist, I will make plots now.")
						if make_plots:
							batch2 = "chains_dir="+chains_dir+"\n"+"data_file_path="+data_file_path+"\n"+"full_file_path="+full_file_path+"\n"+"light_profile_params_path="+light_profile_params_path+"\n"+"sigmap_path="+sigmap_path+"\n"\
									+"load_light_profile="+str(load_light_profile)+"\n"+"fix_light_profile="+str(fix_light_profile)+"\n"+"nu_model="+nu_model+"\n"\
									+"dm_model="+dm_model+"\n"+"nbeta="+str(nbeta)+"\n"+"fix_breaks="+str(fix_breaks)+"\n"+"deltav="+str(deltav)+"\n"+"temp_ind="+str(temp_ind)+"\n"\
									+"run_scan="+str(run_scan)+"\n"+"make_plots="+str(make_plots)+"\n"+"plots_dir="+plots_dir+"\n"+"\n"\
									+"save_plots="+str(save_plots)+"\n"+"save_post="+str(save_post)+"\n"+"has_truth="+str(has_truth)+"\n"+"post_dir="+post_dir+"\n" \
									+"temp_directory=" + temp_directory + "\n"
							if len(run_tag) > 0:
								batch2 = batch2+"run_tag="+run_tag+"\n"

							batch7 = "python scan_vel_disp_interface.py --make_plots $make_plots --save_plots $save_plots --save_post $save_post --has_truth $has_truth "
							batch8 = "--chains_dir $chains_dir --plots_dir $plots_dir --post_dir $post_dir --data_file_path $data_file_path --full_file_path $full_file_path --light_profile_params_path $light_profile_params_path --sigmap_path $sigmap_path "
							batch9 = "--load_light_profile $load_light_profile --fix_light_profile $fix_light_profile --nu_model $nu_model --dm_model $dm_model --nbeta $nbeta --fix_breaks $fix_breaks --deltav $deltav "
							batch10 = "--true_params "
							for i in range(len(true_params)):
								batch10 = batch10+str(true_params[i])+" "
							if len(run_tag) > 0:
									batch8 = batch8+"--run_tag $run_tag "
			
							batchn = batch_plot+batch2+batch7+batch8+batch9+batch10
							fname = batch_dir + "/plot_"+data_tag + err_tag +"_"+nu_model+"_"+dm_model+"_"+str(nbeta)+run_tag+".batch"

							f=open(fname, "w")
							f.write(batchn)
							f.close()
							os.system("sbatch "+fname);

							temp_ind += 1

						if calculate_j_factor:
							batch2 = "chains_dir="+chains_dir+"\n"+"data_file_path="+data_file_path+"\n"+"full_file_path="+full_file_path+"\n"+"light_profile_params_path="+light_profile_params_path+"\n"+"sigmap_path="+sigmap_path+"\n"\
									+"load_light_profile="+str(load_light_profile)+"\n"+"fix_light_profile="+str(fix_light_profile)+"\n"+"nu_model="+nu_model+"\n"\
									+"dm_model="+dm_model+"\n"+"nbeta="+str(nbeta)+"\n"+"fix_breaks="+str(fix_breaks)+"\n"+"deltav="+str(deltav)+"\n"+"temp_ind="+str(temp_ind)+"\n"\
									+"run_scan="+str(run_scan)+"\n"+"calculate_j_factor="+str(calculate_j_factor)+"\n"+ "distance=" + str(distance) + "\n" + "angle_max=" + str(angle_max) + "\n" "full_posterior=" + str(full_posterior) + "\n" + "plots_dir="+plots_dir+"\n"+"\n"\
									+"save_plots="+str(save_plots)+"\n"+"save_post="+str(save_post)+"\n"+"has_truth="+str(has_truth)+"\n"+"post_dir="+post_dir+"\n" \
									+"temp_directory=" + temp_directory + "\n"
							if len(run_tag) > 0:
								batch2 = batch2+"run_tag="+run_tag+"\n"

							batch7 = "python scan_vel_disp_interface.py --calculate_j_factor $calculate_j_factor --distance $distance --angle_max $angle_max --full_posterior $full_posterior --save_plots $save_plots --save_post $save_post --has_truth $has_truth "
							batch8 = "--chains_dir $chains_dir --plots_dir $plots_dir --post_dir $post_dir --data_file_path $data_file_path --full_file_path $full_file_path  --light_profile_params_path $light_profile_params_path --sigmap_path $sigmap_path "
							batch9 = "--load_light_profile $load_light_profile --fix_light_profile $fix_light_profile --nu_model $nu_model --dm_model $dm_model --nbeta $nbeta --fix_breaks $fix_breaks --deltav $deltav "
							batch10 = "--true_params "
							for i in range(len(true_params)):
								batch10 = batch10+str(true_params[i])+" "
							if len(run_tag) > 0:
									batch8 = batch8+"--run_tag $run_tag "
			
							batchn = batch_jfac+batch2+batch7+batch8+batch9+batch10
							fname = batch_dir + "/jfac_"+data_tag + err_tag +"_"+nu_model+"_"+dm_model+"_"+str(nbeta)+run_tag+".batch"
							print("Submission file", fname)

							f=open(fname, "w")
							f.write(batchn)
							f.close()
							os.system("sbatch "+fname);

							temp_ind += 1


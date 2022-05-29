import os, sys

batch1='''#!/bin/bash
#SBATCH -N 1   # node count
#SBATCH --ntasks-per-node=1
#SBATCH -t 1:00:00
#SBATCH --mem=16GB
##SBATCH --mail-type=begin
##SBATCH --mail-type=end
##SBATCH --mail-user=ljchang@princeton.edu

cd /tigress/ljchang/dSph_likelihoods/RunFiles/
		'''

##################
# Gaia challenge #
##################

# extra_text = "PlumCuspIso"
# gaia_challenge_tag = "gs010_bs050_rcrs025_rarcinf_cusp_0064mpc3"
# data_folder = "/tigress/ljchang/dSph_likelihoods/RunFiles/data/gaia_challenge/"+gaia_challenge_tag+"_df/"

# extra_text = "PlumCoreIso"
# gaia_challenge_tag = "gs010_bs050_rcrs100_rarcinf_core_0400mpc3"
# data_folder = "/tigress/ljchang/dSph_likelihoods/RunFiles/data/gaia_challenge/"+gaia_challenge_tag+"_df/"

# out_dir_base = data_folder+"/processed_samps/poiss_err/"

# # for data_tag_base in ["df_100_1","df_100_2","df_100_3","df_100_4","df_100_5","df_100_6","df_100_7","df_100_8","df_100_9"]:
# for iMC in range(5):
# 	data_tag_base = "df_1000_"+str(iMC)
# 	data_tag = gaia_challenge_tag+"_"+data_tag_base
# 	out_tag = data_tag_base

# 	batch2 = "data_dir="+data_folder+"\n"+"data_tag="+data_tag+"\n"+"out_dir_base="+out_dir_base+"\n"+"out_tag="+out_tag+"\n"
# 	batch3 = "python process_data.py --data_dir $data_dir --data_tag $data_tag --out_dir_base $out_dir_base --out_tag $out_tag "
# 	batchn = batch1+batch2+batch3

# 	fname = "./batch/process_"+extra_text+"_"+data_tag_base+".batch"
# 	f=open(fname, "w")
# 	f.write(batchn)
# 	f.close()
# 	os.system("sbatch "+fname);

##############
# Lina mocks #
##############

rs_list = [0.25, 1.0]
rho0_list = [64.0, 400.0]

# extra_text = "PlumCuspIso"
# data_folder = "/tigress/ljchang/dSph_likelihoods/Generate_mocks/mocks/a_1_b_3_g_1_rdm_1_iso/"

extra_text = "PlumCoreIso"
data_folder = "/tigress/ljchang/dSph_likelihoods/Generate_mocks/mocks/a_1_b_3_g_0_rdm_1_iso/"

out_dir_base = "/tigress/ljchang/dSph_likelihoods/RunFiles/data/lina_mocks/"+extra_text+"/"

for rs in rs_list:
	for rho0 in rho0_list:
		data_tag_pref = "dist_as_2.0_bs_5.0_gs_0.1_rs_"+str(rs)+"_dm_"
		data_tag_suff = "_rho_"+str(rho0)
		out_tag_pref = "rs_"+str(int(100*rs))+"_rho0_"+str(int(rho0))

		for nstars in [20,100]:
			for iMC in range(10):
				data_tag = data_tag_pref+str(int(nstars))+"_"+str(iMC)+data_tag_suff
				out_tag = out_tag_pref+"_df_"+str(int(nstars))+"_"+str(iMC)

				batch2 = "data_dir="+data_folder+"\n"+"data_tag="+data_tag+"\n"+"out_dir_base="+out_dir_base+"\n"+"out_tag="+out_tag+"\n"
				batch3 = "python process_data.py --data_dir $data_dir --data_tag $data_tag --out_dir_base $out_dir_base --out_tag $out_tag "
				batchn = batch1+batch2+batch3

				fname = "./batch/process_lina_"+extra_text+"_"+out_tag+".batch"
				f=open(fname, "w")
				f.write(batchn)
				f.close()
				os.system("sbatch "+fname);
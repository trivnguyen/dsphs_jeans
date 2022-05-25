import os, sys
sys.path.append("../Generate_mocks/")
import add_errors


batch1='''#!/bin/bash
#SBATCH -N 1   # node count
#SBATCH --ntasks-per-node=1
#SBATCH -t 1:00:00
#SBATCH --mem=16GB
##SBATCH --mail-type=begin
##SBATCH --mail-type=end
##SBATCH --mail-user=lnecib@caltech.edu

cd /tigress/lnecib/dSph_likelihoods/RunFiles/
'''


##############
# Lina mocks #
##############



input_dir = '/tigress/lnecib/dSph_likelihoods/Generate_mocks/mocks/'
output_dir = '/tigress/ljchang/dSph_likelihoods/RunFiles/data/lina_mocks/'

#### Parameters

rs_list = [0.25, 1.0]
rho0_list = [64.0, 400.0]


#### Create Appropriate Folders



#### Submit batch jobs

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
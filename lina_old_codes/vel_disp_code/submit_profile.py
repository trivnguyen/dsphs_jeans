import os, sys
import numpy as np

batch1='''#!/bin/bash
##SBATCH -N 1
#SBATCH -n 20
#SBATCH -t 6:00:00
#SBATCH --mem=50GB
##SBATCH --mail-type=begin
##SBATCH --mail-type=end
#SBATCH --mail-user=ljchang@princeton.edu

cd /tigress/ljchang/dSph_likelihoods/RunFiles/vel_disp_code/

		'''

chains_dir_base = "/tigress/ljchang/dSph_likelihoods/RunFiles/vel_disp_code/multinest_chains/mocks/"

if not os.path.exists(chains_dir_base):
	os.makedirs(chains_dir_base)

data_dir_base = "/tigress/ljchang/dSph_likelihoods/Generate_mocks/mocks/"
plots_dir_base = "/tigress/ljchang/dSph_likelihoods/Plots/DM_plots/lina_mocks/"
post_dir_base = "/tigress/ljchang/dSph_likelihoods/RunFiles/vel_disp_code/DM_posteriors/mocks/"


batch2 = 'mpiexec -np 20 python profile_ll.py'
batchn = batch1+batch2
fname = "./batch/profile.batch"

f=open(fname, "w")
f.write(batchn)
f.close()
os.system("sbatch "+fname);

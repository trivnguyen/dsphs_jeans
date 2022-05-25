import os, sys
import numpy as np

sys.path.append("/tigress/lnecib/dSph_likelihoods/Generate_mocks")
import generation_functions

######## user input
### Specify number of sample points

nsample = int(sys.argv[1])

same_parameters = True
has_truth = True
run_scan = True
postprocess = True

dropping_bins = False
lower = False

i_initial = 0
ntotal = 10
n_distribution = -1

batch1 = '''#!/bin/bash
#SBATCH -N 1   # node count
#SBATCH --ntasks-per-node=1
#SBATCH -t 5:00:00
#SBATCH --mem=16GB
##SBATCH --mail-type=begin
##SBATCH --mail-type=end
##SBATCH --mail-user=lnecib@caltech.edu

cd /tigress/lnecib/dSph_likelihoods/RunFiles/
'''

#####################
# Light profile fit #
#####################

chains_dir = "/tigress/lnecib/dSph_likelihoods/RunFiles/multinest_chains/embedded_mocks/"
if not os.path.exists(chains_dir):
    os.makedirs(chains_dir)

plots_dir_base = "/tigress/lnecib/dSph_likelihoods/Plots/light_profile_plots/embedded_mocks/"
if not os.path.exists(plots_dir_base):
    os.makedirs(plots_dir_base)

run_dir_base = "/tigress/lnecib/dSph_likelihoods/RunFiles/batch/embedded_mocks/"
if not os.path.exists(run_dir_base):
    os.makedirs(run_dir_base)

temp_directory = "/tigress/lnecib/dSph_likelihoods/RunFiles/temp_lp/embedded_mocks/"
if not os.path.exists(temp_directory):
    os.makedirs(temp_directory)

data_folder_base = "/tigress/lnecib/dSph_likelihoods/Generate_mocks/embedded_mocks/"

if lower:
    cuts = [1, 2, 5, 10]
else:
    cuts = [0.1, 0.5, 1]  # [1, 2, 5, 10]

if lower:
    extratext = '_low_'
else:
    extratext = '_high_'

run_tag = "_1comp"
nu_model = "Plummer"

### Specify alpha/beta/gamma and scale radius (in kpc) of tracer density profile

alphalight = 2.
betalight = 5.
gammalight = 0.1

# Dark Matter

# gamma_dm_list = [0, 0, 0.5, 0.5, 1, 1, 1.5, 1.5] 
# rs_list = [1, 0.2, 1, 0.2, 1, 0.2, 1, 0.2] #, [3]
gamma_dm_list = [0, 0, 1, 1]
rs_list = [1, 0.2, 1, 0.2]

embdeddness_factor = [0.2, 0.5]  # and 0.5

# rlight_list = rs_list


if same_parameters:
    # rlight_param = 1.
    rho_param = 0.064e9
    alpha_param = 1
    beta_param = 3
    # rlight_list = [rlight_param for _ in range(len(gamma_dm_list)) ]
    rho_list = [rho_param for _ in range(len(gamma_dm_list))]
    alpha_dm_list = [alpha_param for _ in range(len(gamma_dm_list))]
    beta_dm_list = [beta_param for _ in range(len(gamma_dm_list))]

else:
    rlight_list = [1., 1., 1., 1.]  # kpc
    alpha_dm_list = [1, 1, 1, 1]
    beta_dm_list = [3, 3, 3, 3]
    rho_list = [0.064e9, 0.064e9, 0.064e9, 0.064e9]  # M_sun/kpc^3

###specify osipkov-merritt anisotropy parameter in units of tracer scale radius

ra_list = [1.e+30]  # , 0.5]#kpc
anisotropy = ['' for _ in range(len(ra_list))]
for s, ra in enumerate(ra_list):
    if ra == 1.e+30:
        anisotropy[s] = 'iso'
    else:
        anisotropy[s] = str(ra)

# ## Measurement Errors
# errors = [0.2, 2., 5., 10.]

n_extension = -4

# Start with no error runs
for embed in embdeddness_factor:
    rlight_list = embed * np.array(rs_list)

    for k in range(len(alpha_dm_list)):
        print("Starting at k", k)

        true_params = [0, np.log10(rlight_list[k]), alphalight, betalight, gammalight]

        directory = generation_functions.set_directory(rs=rs_list[k], alpha=alpha_dm_list[k], beta=beta_dm_list[k],
                                                       gamma=gamma_dm_list[k], anisotropy=anisotropy[0],
                                                       embdeddness_factor=embed)

        for benchmark in [False, True]:
            for cut in cuts:
                for iMC in range(i_initial, i_initial + ntotal):
                    filename = generation_functions.get_preprocess_filename(nsample, rs=rs_list[k],
                                                                            alpha=alpha_dm_list[k],
                                                                            beta=beta_dm_list[k],
                                                                            gamma=gamma_dm_list[k], n_distribution=iMC,
                                                                            ntotal=ntotal, rlight=rlight_list[k],
                                                                            rho=rho_list[k], data_dir=data_folder_base,
                                                                            embdeddness_factor=embed)
                    filename_no_extension = filename[:n_extension]
                    print("filename", filename)
                    print("filename_no_extension", filename_no_extension)
                    data_tag = generation_functions.get_preprocess_filename(nsample, rs=rs_list[k],
                                                                            alpha=alpha_dm_list[k],
                                                                            beta=beta_dm_list[k],
                                                                            gamma=gamma_dm_list[k], n_distribution=iMC,
                                                                            ntotal=ntotal, rlight=rlight_list[k],
                                                                            rho=rho_list[k], data_dir=data_folder_base,
                                                                            simple=True, embdeddness_factor=embed)

                    # This is the path for the light profile
                    if dropping_bins:
                        n_extension = -4
                        if benchmark:
                            data_tag = generation_functions.filename_dropped_bins_reference(data_tag, rlight_list[k],
                                                                                            cut, lower=lower,
                                                                                            n_extension=n_extension)
                            data_file_path = generation_functions.filename_dropped_bins_reference(filename,
                                                                                                  rlight_list[k], cut,
                                                                                                  lower=lower) + "_lp.npz"
                        else:
                            data_tag = generation_functions.filename_dropped_bins(data_tag, rlight_list[k], cut,
                                                                                  lower=lower, n_extension=n_extension)
                            data_file_path = generation_functions.filename_dropped_bins(filename, rlight_list[k], cut,
                                                                                        lower=lower) + "_lp.npz"
                    else:
                        data_file_path = filename_no_extension + "_lp.npz"

                    print("Data File path is", data_file_path)
                    save_dir = chains_dir + directory
                    quantiles_dir = data_folder_base + directory + "light_profile_bestfits/"
                    batch_dir = run_dir_base + directory

                    if not os.path.exists(save_dir):
                        os.makedirs(save_dir)

                    if not os.path.exists(quantiles_dir):
                        os.makedirs(quantiles_dir)

                    if not os.path.exists(batch_dir):
                        os.makedirs(batch_dir)

                    outfile_path = save_dir + data_tag + run_tag + "/"
                    quantiles_path = quantiles_dir + data_tag + run_tag
                    plots_dir = plots_dir_base + directory + data_tag + "/"

                    if not os.path.exists(plots_dir):
                        os.makedirs(plots_dir)

                    batch2 = "data_file_path=" + data_file_path + "\n" + "outfile_path=" + outfile_path + "\n" + "quantiles_path=" + quantiles_path + "\n" \
                             + "nu_model=" + nu_model + "\n" + "plots_dir=" + plots_dir + "\n" \
                             + "run_scan=" + str(run_scan) + "\n" + "postprocess=" + str(
                        postprocess) + "\n" + "has_truth=" + str(
                        has_truth) + "\n" + "temp_directory=" + temp_directory + "\n"
                    batch3 = "python scan_light_profile_interface.py --data_file_path $data_file_path --outfile_path $outfile_path --quantiles_path $quantiles_path --has_truth $has_truth --temp_directory $temp_directory "
                    batch4 = "--true_params "
                    for i in range(len(true_params)):
                        batch4 = batch4 + str(true_params[i]) + " "
                    batch5 = "--nu_model $nu_model --plots_dir $plots_dir --run_scan $run_scan --postprocess $postprocess "
                    batchn = batch1 + batch2 + batch3 + batch4 + batch5

                    fname = batch_dir + "scan_multinest_" + data_tag + run_tag + ".batch"
                    f = open(fname, "w")
                    f.write(batchn)
                    f.close()
                    os.system("sbatch " + fname)

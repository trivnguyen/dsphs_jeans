import argparse, ast
import numpy as np
import os, sys, tempfile
import scan_vel_disp_fix_params as scan_code

parser = argparse.ArgumentParser()
parser.add_argument("--chains_dir", action="store", dest="chains_dir", default="/tigress/ljchang/dSph_likelihoods/RunFiles/vel_disp_code/multinest_chains/", type=str)
parser.add_argument("--temp_ind", action="store", dest="temp_ind", type=int)
parser.add_argument("--data_file_path", action="store", dest="data_file_path", default="", type=str)
parser.add_argument("--light_profile_params_path", action="store", dest="light_profile_params_path", default="", type=str)
parser.add_argument("--sigmap_path", action="store", dest="sigmap_path", default="", type=str)
parser.add_argument("--has_truth", action="store", dest="has_truth", default=True, type=ast.literal_eval) 
parser.add_argument("--true_profile", action="store", dest="true_profile", default="", type=str)
parser.add_argument("--true_params", nargs="+", default=[], type=float) 
parser.add_argument("--plots_dir", action="store", dest="plots_dir", default="/tigress/ljchang/dSph_likelihoods/RunFiles/DM_plots/", type=str)
parser.add_argument("--post_dir", action="store", dest="post_dir", default="/tigress/ljchang/dSph_likelihoods/RunFiles/vel_disp_code/DM_posteriors/", type=str)
parser.add_argument("--load_light_profile", action="store", dest="load_light_profile", default=True, type=ast.literal_eval) 
parser.add_argument("--fix_light_profile", action="store", dest="fix_light_profile", default=False, type=ast.literal_eval) 
parser.add_argument("--nu_model", action="store", dest="nu_model", default="Plummer", type=str)
parser.add_argument("--dm_model", action="store", dest="dm_model", default="gNFW", type=str)
parser.add_argument("--fix_breaks", action="store", dest="fix_breaks", default=False, type=ast.literal_eval) 
parser.add_argument("--fix_tag", action="store", dest="fix_tag", default="", type=str)
parser.add_argument("--nbeta", action="store", dest="nbeta", default=0, type=int)
parser.add_argument("--deltav", action="store", dest="deltav", default=0.2, type=float)
parser.add_argument("--measured_errs", action="store", dest="measured_errs", default=False, type=ast.literal_eval) 
parser.add_argument("--verbose", action="store", dest="verbose", default=True, type=ast.literal_eval) 
parser.add_argument("--setup_dirs", action="store_true") 
parser.add_argument("--cleanup_dirs", action="store_true") 
parser.add_argument("--run_scan", action="store", dest="run_scan", default=False, type=ast.literal_eval) 
parser.add_argument("--make_plots", action="store", dest="make_plots", default=False, type=ast.literal_eval) 
parser.add_argument("--save_plots", action="store", dest="save_plots", default=True, type=ast.literal_eval) 
parser.add_argument("--save_post", action="store", dest="save_post", default=False, type=ast.literal_eval) 
parser.add_argument("--run_tag", action="store", dest="run_tag", default="", type=str)
parser.add_argument("--temp_directory", action="store", dest="temp_directory", default="/tigress/ljchang/dSph_likelihoods/RunFiles/temp/", type=str)

results = parser.parse_args()
chains_dir = results.chains_dir
post_dir = results.post_dir
temp_ind = results.temp_ind
data_file_path = results.data_file_path
light_profile_params_path = results.light_profile_params_path
sigmap_path = results.sigmap_path
has_truth = results.has_truth
true_profile = results.true_profile
true_params = results.true_params
plots_dir = results.plots_dir
load_light_profile = results.load_light_profile
fix_light_profile = results.fix_light_profile
nu_model = results.nu_model
dm_model = results.dm_model
fix_breaks = results.fix_breaks
fix_tag = results.fix_tag
nbeta = results.nbeta
deltav = results.deltav
measured_errs = results.measured_errs
verbose = results.verbose
setup_dirs = results.setup_dirs
cleanup_dirs = results.cleanup_dirs
run_scan = results.run_scan
make_plots = results.make_plots
save_plots = results.save_plots
save_post = results.save_post
run_tag = results.run_tag
temp_directory = results.temp_directory

# print("Interface is receiving the model ",dm_model)

args = {"data_file_path":data_file_path, "light_profile_params_path":light_profile_params_path, 
		"load_light_profile":load_light_profile, "fix_light_profile":fix_light_profile, 
		"nu_model":nu_model, "dm_model":dm_model, "fix_breaks":fix_breaks, "true_profile":true_profile,
		"nbeta":nbeta, "deltav":deltav, "measured_errs":measured_errs, "verbose":verbose, "run_tag":run_tag,
		"true_params":true_params, "fix_tag":fix_tag}

tempdirpath = temp_directory + "t"+str(temp_ind)+"/"

if setup_dirs:
	if not len(chains_dir) < 78:
		if not os.path.exists(tempdirpath):
			os.mkdir(tempdirpath)
		else: 
			raise ValueError("Temp dir already exists; aborting")

		if not os.path.exists(chains_dir):
			os.mkdir(chains_dir)

		if verbose:
			print("Made temporary directory ",tempdirpath)

if run_scan:
	scan = scan_code.run_scan(**args)
	# scan = scan_data.run_scan(**args)

	# Hacky way of getting around Multinest's 100-character limit on output base path
	if len(chains_dir) < 78: # Account for (len(base)+len(post_equal_weights.dat))<100
		scan_.perform_scan_multinest(chains_dir=chains_dir)

	else:
		scan.perform_scan_multinest(chains_dir=tempdirpath)

if cleanup_dirs:
	if not len(chains_dir) < 78:
		os.system("scp "+tempdirpath+"/* "+chains_dir)
		os.system("rm -r "+tempdirpath)

		if verbose:
			print("Copied over results to ",chains_dir)

if make_plots:
	scan = scan_code.run_scan(**args)
	scan.make_corner_plot(chains_dir=chains_dir,plots_dir=plots_dir)
	scan.make_density_plots(chains_dir=chains_dir,plots_dir=plots_dir,post_dir=post_dir,save_plots=save_plots,save_post=save_post,has_truth=has_truth)
	# scan.get_dispersion(chains_dir=chains_dir,sigmap_path=sigmap_path,plots_dir=plots_dir,post_dir=post_dir,save_plots=save_plots,save_post=save_post)

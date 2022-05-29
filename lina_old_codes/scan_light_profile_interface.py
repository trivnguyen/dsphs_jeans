import argparse, ast
import numpy as np
import os, sys, tempfile
import scan_light_profile, scan_sigmap_obs

parser = argparse.ArgumentParser()
parser.add_argument("--data_file_path", action="store", dest="data_file_path", default="/tigress/ljchang/dSph_likelihoods/RunFiles/data/segue_clean.txt", type=str)
parser.add_argument("--data_file_path_sigmap", action="store", dest="data_file_path_sigmap", default="/tigress/ljchang/dSph_likelihoods/RunFiles/data/segue_clean.txt", type=str)
parser.add_argument("--plots_dir", action="store", dest="plots_dir", default="/tigress/ljchang/dSph_likelihoods/Plots/light_profile_plots/", type=str)
parser.add_argument("--outfile_path", action="store", dest="outfile_path", default="", type=str)
parser.add_argument("--outfile_path_sigmap", action="store", dest="outfile_path_sigmap", default="", type=str)
parser.add_argument("--quantiles_path", action="store", dest="quantiles_path", default="", type=str)
parser.add_argument("--true_profile", action="store", dest="true_profile", default="Zhao", type=str)
parser.add_argument("--has_truth", action="store", dest="has_truth", default=True, type=ast.literal_eval) 
parser.add_argument("--true_params", nargs="+", default=[], type=float) 
parser.add_argument("--nu_model", action="store", dest="nu_model", default="Plummer", type=str)
parser.add_argument("--run_scan", action="store", dest="run_scan", default=True, type=ast.literal_eval) 
parser.add_argument("--postprocess", action="store", dest="postprocess", default=True, type=ast.literal_eval) 
parser.add_argument("--fit_sigmap", action="store", dest="fit_sigmap", default=False, type=ast.literal_eval) 
parser.add_argument("--nbins", action="store", dest="nbins", default=10, type=int)
parser.add_argument("--deltav", action="store", dest="deltav", default=2., type=float)
parser.add_argument("--measured_errs", action="store", dest="measured_errs", default=False, type=ast.literal_eval) 
parser.add_argument("--temp_directory", action="store", dest="temp_directory", default="/tigress/ljchang/dSph_likelihoods/RunFiles/temp_lp/", type=str)


results = parser.parse_args()
data_file_path = results.data_file_path
data_file_path_sigmap = results.data_file_path_sigmap
plots_dir = results.plots_dir
outfile_path = results.outfile_path
outfile_path_sigmap = results.outfile_path_sigmap
quantiles_path = results.quantiles_path
true_profile = results.true_profile
true_params = results.true_params
has_truth = results.has_truth
nu_model = results.nu_model
run_scan = results.run_scan
postprocess = results.postprocess
fit_sigmap = results.fit_sigmap
nbins = results.nbins
deltav = results.deltav
measured_errs = results.measured_errs
temp_directory = results.temp_directory

if run_scan or postprocess:
	scan = scan_light_profile.run_scan(data_file_path=data_file_path,nu_model=nu_model,true_profile=true_profile)

if run_scan:
	# Hacky way of getting around Multinest's 100-character limit on output base path
	if len(outfile_path) < 78: # Account for (len(base)+len(post_equal_weights.dat))<100
		scan.perform_scan_multinest(chains_dir=outfile_path)

	else:
		if not os.path.exists(outfile_path):
			os.makedirs(outfile_path)

		with tempfile.TemporaryDirectory(prefix=temp_directory) as tempdirpath:
			scan.perform_scan_multinest(chains_dir=tempdirpath+"/")
			os.system("scp "+tempdirpath+"/* "+outfile_path)

if postprocess:
	scan.save_quantiles(chains_dir=outfile_path,save_path=quantiles_path)
	scan.get_post_profiles(chains_dir=outfile_path,has_truth=has_truth,true_params=true_params)
	scan.make_plots(chains_dir=outfile_path,plots_dir=plots_dir,results_args = {"color":"cornflowerblue","alpha":0.2,"label":nu_model},has_truth=has_truth)

if fit_sigmap:
	scan_sigmap = scan_sigmap_obs.run_scan(data_file_path=data_file_path_sigmap,out_file_path=outfile_path_sigmap,nbins=nbins,deltav=deltav,measured_errs=measured_errs)
	scan_sigmap.scipy_scan()

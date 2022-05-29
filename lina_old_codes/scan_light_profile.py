import sys, os
import numpy as np
import pymultinest, corner
from tqdm import tqdm
import pandas as pd
import lp_Zhao, lp_PlumSph
import matplotlib.pyplot as plt
import matplotlib 
matplotlib.use('Agg')
from matplotlib import rcParams

rcParams['xtick.labelsize'] = 20
rcParams['ytick.labelsize'] = 20
rcParams['axes.labelsize'] = 20
rcParams['axes.titlesize'] = 20
rcParams['font.family'] = 'serif'
rcParams['font.serif'] = 'CMU Serif'
rcParams['figure.figsize'] = (10/1.2,8/1.2)
rcParams['legend.fontsize'] = 16
rcParams['text.usetex'] = True
rcParams['text.latex.preamble'] = [r"\usepackage{amsmath}"]

class run_scan():
	def __init__(self, nu_model = "Plummer", true_profile = "Zhao", 
				 data_file_path = "/tigress/ljchang/dSph_likelihoods/RunFiles/data/segue_clean.txt"):

		self.nu_model = nu_model
		self.true_profile = true_profile
		self.data_file_path = data_file_path

		# Plummer sphere priors
		if self.nu_model == "Plummer":
			self.params = [r"$\log_{10}(M_0)$",r"$\log_{10}(a_0)$"]
			self.priors = [[-2,5],[-3,3]]

		elif self.nu_model == "Plummer2":		
			self.params = [r"$\log_{10}(M_0)$",r"$\log_{10}(\frac{M_1}{M_0})$",r"$\log_{10}(a_0)$",r"$\log_{10}(\frac{a_1}{a_0})$"]
			self.priors = [[-2,5],[-10,0],[-3,3],[-10,0]]

		elif self.nu_model == "Zhao":
			self.params = [r"$\log_{10}(\nu_0)$",r"$\log_{10}(r_*)$",r"$\alpha_*$",r"$\beta_*$",r"$\gamma_*$"]
			self.priors = [[-5,5],[-2,2],[0.1,5],[2,10],[0,1]]

		# elif self.nu_model == "Zhao_bkg":
		# 	self.params = ['log_nus_star','log_rs_star','alpha','beta','gamma','log_Sigma_bkg']
		# 	self.priors = [[-5,5],[-2,2],[0,5],[0,10],[0,1],[-5,5]]

		else:
			print("The specified model",self.nu_model,"is not implemented yet!")		

		self.n_params = len(self.params)
		self.load_data()
		self.setup_prior_thetas()

	def load_data(self):
		""" Load the data (already pre-binned)
		"""
		self.data = np.load(self.data_file_path)
		self.bin_centers = self.data['bin_centers']
		self.lower_bin_edges = self.data['lower_bin_edges']
		self.upper_bin_edges = self.data['upper_bin_edges']
		self.binned_data = self.data['counts']
		self.binned_sigmas_lo = self.data['sigmas_lo']
		self.binned_sigmas_hi = self.data['sigmas_hi']

	def setup_prior_thetas(self):
		""" Set up priors and arrays for parameters to be floated
		""" 
		self.theta_min = []
		self.theta_max = []
		for iparam in range(len(self.params)):
			self.theta_min += [self.priors[iparam][0]]
			self.theta_max += [self.priors[iparam][1]]

		self.theta_interval = list(np.array(self.theta_max) - np.array(self.theta_min))

	def prior_cube(self, cube, ndim, nparams):
		""" Cube of priors in the format required by MultiNest
		"""
		for i in range(ndim):
			cube[i] = cube[i] * self.theta_interval[i] + self.theta_min[i]
		return cube

	def ll(self, theta, ndim, nparams):
		""" Log Likelihood in the form that can be used by Minuit or Multinest
		"""
		theta_ll = np.zeros((self.n_params))
		for i in range(self.n_params):
			theta_ll[i] = theta[i]

		if self.nu_model == "Plummer":
			ll_val = lp_PlumSph.lnlike_Plumsph_1comp(*theta_ll,self.lower_bin_edges,self.upper_bin_edges,self.bin_centers,self.binned_data,self.binned_sigmas_lo,self.binned_sigmas_hi)
		elif self.nu_model == "Plummer2":
			ll_val = lp_PlumSph.lnlike_Plumsph_2comp_rel2(*theta_ll,self.lower_bin_edges,self.upper_bin_edges,self.bin_centers,self.binned_data,self.binned_sigmas_lo,self.binned_sigmas_hi)			
		elif self.nu_model == "Zhao":
			ll_val = lp_Zhao.lnlike_Zhao_binned(*theta_ll,self.lower_bin_edges,self.upper_bin_edges,self.bin_centers,self.binned_data,self.binned_sigmas_lo,self.binned_sigmas_hi,use_counts=True)
		elif self.nu_model == "Zhao_bkg":
			ll_val = lp_Zhao.lnlike_Zhao_binned_bkg(*theta_ll,self.lower_bin_edges,self.upper_bin_edges,self.bin_centers,self.binned_data,self.binned_sigmas_lo,self.binned_sigmas_hi,use_counts=True)
		# print(theta_ll,ll_val)
		return ll_val

	def perform_scan_multinest(self, chains_dir, nlive=1000, pymultinest_options=None):
		""" Perform a scan with MultiNest
		"""
		self.make_dirs([chains_dir])
		if pymultinest_options is None:
			pymultinest_options_arg = {'importance_nested_sampling': False,
									'resume': False, 'verbose': True,
									'sampling_efficiency': 'model',
									'init_MPI': False, 'evidence_tolerance': 0.25,
									'const_efficiency_mode': False}
		else:
			pymultinest_options_arg = pymultinest_options

		pymultinest.run(self.ll, self.prior_cube, self.n_params, 
						outputfiles_basename=chains_dir, 
						n_live_points=nlive, **pymultinest_options_arg)

	def save_quantiles(self, chains_dir, save_path):
		post = pd.read_csv(chains_dir+"/post_equal_weights.dat",delim_whitespace=True,header=None).values[:,:-1]
		middle95 = np.array([np.sort(np.quantile(post,q=[0.025,0.975],axis=0)[:,i]) for i in range(post.shape[-1])])
		median = np.median(post,axis=0)

		np.savez(save_path,median=median,middle95=middle95)
		print("Saved quantiles from fit to ",save_path)

	def get_post_profiles(self, chains_dir, has_truth=True, true_params = [0,1,1,3,1]):
		post = pd.read_csv(chains_dir+"/post_equal_weights.dat",delim_whitespace=True,header=None).values[:,:-1]

		self.plotrvals = np.logspace(np.log10(self.lower_bin_edges[0]),np.log10(self.upper_bin_edges[-1]),100)
		self.post_dist = np.zeros((len(post),len(self.plotrvals)))

		if self.nu_model == "Plummer":
			for i in range(len(post)):
				self.post_dist[i] = np.array([lp_PlumSph.SigmaStar([10**post[i][0]],[10**post[i][1]],r,N_p=1) for r in self.plotrvals])
		elif self.nu_model == "Plummer2":
			for i in range(len(post)):
				m_ary = [10**post[i][0],10**(post[i][0]+post[i][1])]
				a_ary = [10**post[i][2],10**(post[i][2]+post[i][3])]
				self.post_dist[i] = np.array([lp_PlumSph.SigmaStar(m_ary,a_ary,r,N_p=2) for r in self.plotrvals])
		elif self.nu_model == "Zhao":
			for i in tqdm(range(len(post))):
				self.post_dist[i] = np.array([lp_Zhao.Sigma_Zhao(*post[i],np.log10(r)) for r in self.plotrvals])
		else:
			raise ValueError("The postprocessing for profile ",self.nu_model," is not implemented yet")

		if has_truth:
			if self.true_profile == "Zhao":
				norm = np.log10(np.median(self.post_dist,axis=0)[0]/lp_Zhao.Sigma_Zhao(*true_params,np.log10(self.plotrvals[0])))
				self.true_lp = [lp_Zhao.Sigma_Zhao(norm,*true_params[1:],np.log10(r)) for r in self.plotrvals]
				# print("The true params are ",true_params)
			else:
				raise ValueError("The true profile ",self.true_profile," is not implemented in postprocess code yet!")

			# if self.true_profile == "PlumCuspIso":
			# 	norm = np.log10(np.median(self.post_dist,axis=0)[0]/lp_Zhao.Sigma_Zhao(0,np.log10(0.25),2,5,0.1,np.log10(self.plotrvals[0])))
			# 	self.true_lp = [lp_Zhao.Sigma_Zhao(norm,np.log10(0.25),2,5,0.1,np.log10(r)) for r in self.plotrvals]
			# elif self.true_profile == "PlumCoreIso":
			# 	norm = np.log10(np.median(self.post_dist,axis=0)[0]/lp_Zhao.Sigma_Zhao(0,np.log10(1),2,5,0.1,np.log10(self.plotrvals[0])))
			# 	self.true_lp = [lp_Zhao.Sigma_Zhao(norm,np.log10(1),2,5,0.1,np.log10(r)) for r in self.plotrvals]
			# else:
			# 	raise ValueError("The true profile ",self.true_profile," is not implemented in postprocess code yet!")

		print("Calculated posterior light profiles")

	def make_plots(self, chains_dir, plots_dir, corner_args = {"color":"k"},
					data_points_args = {"color":"gray","capsize":4,"fmt":"o"},
					truth_line_args = {"color":"k","ls":"--","label":"Truth (Zhao)"},
					results_args = {"color":"cornflowerblue","alpha":0.2,"label":"Plummer"},
					has_truth=True):

		self.make_dirs([plots_dir])
		post = pd.read_csv(chains_dir+"/post_equal_weights.dat",delim_whitespace=True,header=None).values[:,:-1]

		# Make corner plot
		ndim = post.shape[-1]
		fig = corner.corner(post,
							labels=self.params,
							max_n_ticks=4,
							quantiles=[0.16,0.5,0.84],
							show_titles=True, title_kwargs={"fontsize": 16},
							smooth=1,color=corner_args["color"]);

		ax = np.array(fig.axes).reshape((ndim,ndim))
		for i in range(ndim):
			for j in range(ndim):
				ax[i,j].tick_params(labelsize=14)

		plt.savefig(plots_dir+"/corner_"+self.nu_model+".pdf")
		plt.close()

		plt.errorbar(self.bin_centers,self.binned_data/(np.pi*(self.upper_bin_edges**2-self.lower_bin_edges**2)),\
			yerr=[self.binned_sigmas_lo/(np.pi*(self.upper_bin_edges**2-self.lower_bin_edges**2)),self.binned_sigmas_hi/(np.pi*(self.upper_bin_edges**2-self.lower_bin_edges**2))],\
			fmt=data_points_args["fmt"],color=data_points_args["color"],capsize=data_points_args["capsize"])

		plt.plot(self.plotrvals,np.median(self.post_dist,axis=0),label=results_args["label"],color=results_args["color"])
		plt.fill_between(self.plotrvals,np.percentile(self.post_dist,q=16,axis=0),np.percentile(self.post_dist,q=84,axis=0),color=results_args["color"],alpha=results_args["alpha"])
		plt.fill_between(self.plotrvals,np.percentile(self.post_dist,q=2.5,axis=0),np.percentile(self.post_dist,q=97.5,axis=0),color=results_args["color"],alpha=results_args["alpha"])

		if has_truth:
			plt.plot(self.plotrvals,self.true_lp,color=truth_line_args["color"],ls=truth_line_args["ls"],label=truth_line_args["label"])

		plt.xscale('log')
		plt.yscale('log')
		plt.xlim(np.min(self.plotrvals),np.max(self.plotrvals))

		plt.xlabel(r"$R\,[\mathrm{kpc}]$")
		plt.ylabel(r"$\Sigma(R)\,[\mathrm{counts}\cdot\mathrm{kpc}^{-2}]$")
		plt.legend(frameon=False,loc='upper right')

		plt.tight_layout()
		plt.savefig(plots_dir+"/light_profile_"+self.nu_model+".pdf")
		plt.close()		

		print("Saved plots to ",plots_dir)

	@staticmethod
	def make_dirs(dirs):
		""" Creates directories if they do not already exist 
		"""
		for d in dirs:
			if not os.path.exists(d):
				try:
					os.mkdir(d)
				except OSError as e:
					if e.errno != 17:
						raise
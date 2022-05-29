import sys, os
import numpy as np
import pymultinest, corner
from tqdm import tqdm
import pandas as pd
import functions_multinest, DM_profiles
import matplotlib.pyplot as plt
import matplotlib 
matplotlib.use('Agg')
from matplotlib import rcParams

rcParams['xtick.labelsize'] = 20
rcParams['xtick.direction'] = 'in'
rcParams['ytick.labelsize'] = 20
rcParams['ytick.direction'] = 'in'
rcParams['axes.labelsize'] = 20
rcParams['axes.titlesize'] = 20
rcParams['font.family'] = 'serif'
rcParams['font.serif'] = 'CMU Serif'
rcParams['figure.figsize'] = (10/1.2,8/1.2)
rcParams['legend.fontsize'] = 16
rcParams['text.usetex'] = True
rcParams['text.latex.preamble'] = [r"\usepackage{amsmath}"]

log10toln = np.log10(np.exp(1))

class run_scan():
	def __init__(self, 
				 true_params, 
				 data_file_path = "", 
				 full_file_path = "",
				 light_profile_params_path = "", 
				 true_profile = "PlumCuspIso",
				 load_light_profile = True, 
				 fix_light_profile = False, 
				 nu_model = "Plummer", 
				 dm_model = "gNFW", 
				 fix_breaks = False,
				 nbeta = 0, 
				 deltav = 0.2, 
				 measured_errs = False, 
				 verbose = True,
				 run_tag = "", 
				 mock = True): 

		self.data_file_path = data_file_path
		self.full_file_path = full_file_path
		self.light_profile_params_path = light_profile_params_path
		self.true_profile = true_profile
		self.load_light_profile = load_light_profile
		self.fix_light_profile = fix_light_profile
		self.nu_model = nu_model
		self.dm_model = dm_model
		self.fix_breaks = fix_breaks
		self.nbeta = nbeta
		self.deltav = deltav
		self.measured_errs = measured_errs
		self.verbose = verbose
		self.run_tag = run_tag
		self.true_params = true_params
		self.mock = mock

		self.load_data()
		self.set_priors()

		self.n_float_params = len(self.params)
		self.n_fix_params = len(self.fixed_params['fix_inds'])
		self.n_params = self.n_float_params + self.n_fix_params

		self.setup_prior_thetas()

	def load_data(self):
		self.data = np.load(self.data_file_path)

		self.projected_distance = self.data['distances']
		self.projected_vel = self.data['velocities']

		if self.measured_errs:
			self.vel_err = self.data['v_errs']
			print("Using measured velocity errors")
		else:
			self.vel_err = np.array([self.deltav for k in range(len(self.projected_vel))])
			print("Using velocity error ",self.deltav)

		if self.mock:
			print("Reading file", self.full_file_path)
			coords = pd.read_csv(self.full_file_path,header=None,delim_whitespace=True,usecols=[0,1,2]).values
			r3d = np.sqrt(np.sum(coords**2,axis=1))

			self.logrmin = np.floor(np.log(min(r3d))*100)/100
			self.logrmax = np.ceil(np.log(max(r3d))*100)/100

			self.rvals = np.logspace(self.logrmin,self.logrmax,100,base=np.exp(1))
		else:
			#Projected radii
			self.logrmin = np.floor(np.log(min(self.projected_distance))*100)/100
			self.logrmax = np.ceil(np.log(max(self.projected_distance))*100)/100
			self.rvals = np.logspace(self.logrmin,self.logrmax,100,base=np.exp(1))


		if self.verbose:
			print("Loaded data from ", self.data_file_path)
			print("Highest break must be smaller than ", self.logrmax, "; lowest break must be larger than ", self.logrmin)

	def set_priors(self):
		self.fixed_params = {'fix_inds':[],'fix_vals':[]}

		# Start with light profile priors
		if self.load_light_profile:
			if os.path.isfile(self.light_profile_params_path):
				fit_results = np.load(self.light_profile_params_path)
				best_fit_values = fit_results['median']/log10toln
				n_light_profile_params = len(best_fit_values)
				print("Loaded light profile params from ",self.light_profile_params_path)

				if self.fix_light_profile:
					self.fixed_params['fix_inds'] = [*np.arange(n_light_profile_params)]
					self.fixed_params['fix_vals'] = best_fit_values.tolist()
					self.params = []
					self.priors = []
					self.nstarparams = 0
					self.nstarparams_theory = n_light_profile_params

					print("Fixing light profile params to best-fit values")

				else:
					
					if self.nu_model == "Plummer":
						assert(n_light_profile_params == 2), "Dimension of light profile priors file doesn't fit model!"
						self.params = [r'$\log(M_0^*)$',r'$\log(a_0^*)$']
						self.priors = (fit_results['middle95']/log10toln).tolist()
						self.nstarparams = 2
						self.nstarparams_theory = 2
					elif self.nu_model == "Plummer2":
						assert(n_light_profile_params == 4), "Dimension of light profile priors file doesn't fit model!"
						self.params = [r'$\log(M_0^*)$',r'$\log(a_0^*)$',r'$\log(M_1^*)$',r'$\log(a_1^*)$']
						self.priors = (fit_results['middle95']/log10toln).tolist()
						self.nstarparams = 4
						self.nstarparams_theory = 4
					elif self.nu_model == "Plummer3":
						assert(n_light_profile_params == 6), "Dimension of light profile priors file doesn't fit model!"
						self.params = [r'$\log(M_0^*)$',r'$\log(a_0^*)$',r'$\log(M_1^*)$',r'$\log(a_1^*)$',r'$\log(M_2^*)$',r'$\log(a_2^*)$']
						self.priors = (fit_results['middle95']/log10toln).tolist()
						self.nstarparams = 6
						self.nstarparams_theory = 6
					elif self.nu_model == "Zhao":
						assert(n_light_profile_params == 5), "Dimension of light profile priors file doesn't fit model!"
						self.params = [r'$\log(\nu_0^*)$',r'$\log(r_0^*)$',r'$\alpha^*$',r'$\beta^*$',r'$\gamma^*$']
						self.priors = (fit_results['middle95'][:2]/log10toln).tolist()+(fit_results['middle95'][2:]).tolist()
						self.nstarparams = 5
						self.nstarparams_theory = 5
					else:
						raise ValueError("The specified light profile model is not implemented")
			else:
				self.load_light_profile = False
				if self.verbose:
					raise ValueError("Light profile params file does not exist! Assuming initial guesses")

		if not self.load_light_profile:
			if self.nu_model == "Plummer":
				self.params = [r'$\log(M_0^*)$',r'$\log(a_0^*)$']
				self.priors = ([[-2,5],[-3,3]]/log10toln).tolist()
				self.nstarparams = 2
				self.nstarparams_theory = 2
			elif self.nu_model == "Plummer2":
				self.params = [r'$\log(M_0^*)$',r'$\log(a_0^*)$',r'$\log(M_1^*)$',r'$\log(a_1^*)$']
				self.priors = ([[-2,5],[-10,0],[-3,3],[-10,0]]/log10toln).tolist()
				self.nstarparams = 4
				self.nstarparams_theory = 4
			elif self.nu_model == "Zhao":
				self.params = [r'$\log(\nu_0^*)$',r'$\log(r_0^*)$',r'$\alpha^*$',r'$\beta^*$',r'$\gamma^*$']
				self.priors = ([[-5,5],[-2,2],[0.1,5],[2,10],[0,1]]/log10toln).tolist()
				self.nstarparams = 5
				self.nstarparams_theory = 5
			else:
				raise ValueError("The specified light profile model is not implemented")

		# Anisotropy priors
		if self.nbeta > 0:
			if self.nbeta == 1:
				self.params = self.params + [r'$\log(r_a)$']
				self.priors = self.priors + [[-5,5]]
				print("Using Osipkov-Merritt")
			else:
				print("Anisotropy not yet implemented")

		# Next, do DM priors
		if self.dm_model == "BPL":
			if self.fix_breaks:
				self.fixed_params['fix_inds'] = self.fixed_params['fix_inds']+[self.nstarparams_theory+1]
				self.fixed_params['fix_vals'] = self.fixed_params['fix_vals']+np.log(self.get_bins(self.projected_distance,2)).tolist()
				self.params = self.params + [r'$\log(\rho_0)$', r'$\gamma_0$',r'$\gamma_1$']
				self.priors = self.priors + [[5,25],[-1,10],[0,10]]
				self.ndmparams = 3
				self.ndmparams_theory = 4
			else:
				self.params = self.params + [r'$\log(\rho_0)$', r'$\log(r_0)$', r'$\gamma_0$', r'$\gamma_1$']
				self.priors = self.priors + [[5,25],[self.logrmin,self.logrmax],[-1,10],[0,10]]
				self.ndmparams = 4
		elif self.dm_model == "BPL2":
			if self.fix_breaks:
				self.fixed_params['fix_inds'] = self.fixed_params['fix_inds']+[self.nstarparams_theory+1,self.nstarparams_theory+4]
				self.fixed_params['fix_vals'] = self.fixed_params['fix_vals']+np.log(self.get_bins(self.projected_distance,3)).tolist()
				self.params = self.params + [r'$\log(\rho_0)$', r'$\gamma_0$', r'$\gamma_1$', r'$\gamma_2$']
				self.priors = self.priors + [[15,35],[-5,5],[-1,5],[-1,5]]
				self.ndmparams = 4
				self.ndmparams_theory = 6
			else:
				self.params = self.params + [r'$\log(\rho_0)$', r'$\log(r_0)$', r'$\gamma_0$', r'$\gamma_1$', r'$\log(r_1)$', r'$\gamma_2$']
				self.deltalogr = (self.logrmax - self.logrmin)/2
				self.priors = self.priors + [[15,35],[self.logrmin,self.logrmin+self.deltalogr],[-5,5],[-1,5],[self.logrmin+self.deltalogr,self.logrmax],[-1,5]]
				# self.bins = self.get_bins(self.projected_distance,2)
				# self.priors = self.priors + [[15,35],[self.logrmin,self.bins[0]],[-5,5],[-1,5],[self.bins[0],self.logrmax],[-1,5]]
				self.ndmparams = 6
		elif self.dm_model == "BPL3":
			if self.fix_breaks:
				self.fixed_params['fix_inds'] = self.fixed_params['fix_inds']+[self.nstarparams_theory+1,self.nstarparams_theory+4,self.nstarparams_theory+6]
				self.fixed_params['fix_vals'] = self.fixed_params['fix_vals']+np.log(self.get_bins(self.projected_distance,4)).tolist()
				self.params = self.params + [r'$\log(\rho_0)$', r'$\gamma_0$', r'$\gamma_1$', r'$\gamma_2$', r'$\gamma_3$']
				self.priors = self.priors + [[15,35],[-5,5],[-2,5],[-2,5],[-2,5]]
				self.ndmparams = 5
				self.ndmparams_theory = 8
			else:
				self.params = self.params + [r'$\log(\rho_0)$', r'$\log(r_0)$', r'$\gamma_0$', r'$\gamma_1$', r'$\log(r_1)$', r'$\gamma_2$', r'$\log(r_2)$', r'$\gamma_3$']

				self.deltalogr = (self.logrmax - self.logrmin)/3
				self.priors = self.priors + [[15,35],[self.logrmin,self.logrmin+self.deltalogr],[-10,10],[-5,10],[self.logrmin+self.deltalogr,self.logrmin+2*self.deltalogr],[-5,10],[self.logrmin+2*self.deltalogr,self.logrmax],[-5,10]]
				# bins_temp = np.log(self.get_bins(self.projected_distance,4))
				# self.bins = (bins_temp[1:] + bins_temp[:-1])/2
				# self.priors = self.priors + [[15,35],[self.logrmin,self.bins[0]],[-10,10],[-1,10],[self.bins[0],self.bins[1]],[-1,10],[self.bins[1],self.logrmax],[-1,10]]
				self.ndmparams = 8
		elif self.dm_model == "Zhao":
			self.params = self.params + [r'$\log(\rho_0)$', r'$\log(r_0)$', r'$\alpha$', r'$\beta$', r'$\gamma$']
			# self.priors = self.priors + [[5,25],[self.logrmin,self.logrmax],[0.1,3],[2,10],[0,2]]
			self.priors = self.priors + [[5,30],[-10,10],[0.1,3],[2,10],[0,2]]
			self.ndmparams = 5
		elif self.dm_model == "NFW" or self.dm_model == "Burkert" or self.dm_model == "NFWc":
			self.params = self.params + [r'$\log(\rho_0)$', r'$\log(r_0)$']
			# self.priors = self.priors + [[5,25],[self.logrmin,self.logrmax]]
			self.priors = self.priors + [[5,30],[-10,10]]
			self.ndmparams = 2
		elif self.dm_model == "gNFW":
			self.params = self.params + [r'$\log(\rho_0)$', r'$\log(r_0)$', r'$\gamma$']
			# self.priors = self.priors + [[5,25],[self.logrmin,self.logrmax],[-1,5]]
			# self.priors = self.priors + [[5,30],[-10,10],[-1,5]]
			self.priors = self.priors + [[5,30],[-1,1],[-1,5]]
			self.ndmparams = 3

			# Fix rho0, r0
			# self.fixed_params['fix_inds'] = self.fixed_params['fix_inds']+[self.nstarparams_theory,self.nstarparams_theory+1]
			# self.fixed_params['fix_vals'] = self.fixed_params['fix_vals']+[np.log(self.true_params[0]*1e6),0.0]
			# self.params = self.params + [r'$\gamma$']
			# self.priors = self.priors + [[-5,5]]
			# self.ndmparams = 1

			# Fix rho0
			# self.fixed_params['fix_inds'] = self.fixed_params['fix_inds']+[self.nstarparams_theory]
			# self.fixed_params['fix_vals'] = self.fixed_params['fix_vals']+[np.log(self.true_params[0]*1e6)]
			# self.params = self.params + [r'$\log(r_0)$', r'$\gamma$']
			# self.priors = self.priors + [[-10,10],[-5,5]]
			# self.ndmparams = 2

			# Fix r0
			# self.fixed_params['fix_inds'] = self.fixed_params['fix_inds']+[self.nstarparams_theory+1]
			# self.fixed_params['fix_vals'] = self.fixed_params['fix_vals']+[0.0]
			# self.params = self.params + [r'$\log(\rho_0)$', r'$\gamma$']
			# self.priors = self.priors + [[5,30],[-5,5]]
			# self.ndmparams = 1

			# self.ndmparams_theory = 3



		# elif self.dm_model == "gNFW_trunc":
		# 	self.params = self.params + [r'$\log(\rho_0)$', r'$\log(r_0)$', r'$\gamma$', r'$\log(r_t)$']
		# 	self.priors = self.priors + [[5,25],[self.logrmin,self.logrmax],[-1,5],[self.logrmin, self.logrmax]]
		# 	self.ndmparams = 4
		else:
			raise ValueError("Specified DM model is not implemented")

		# Mean velocity prior
		self.params = self.params + [r'$\bar v$']
		self.priors = self.priors + [[-100,100]]
		# self.priors = self.priors + [[-20,20]]

	def setup_prior_thetas(self):
		""" Set up priors and arrays for parameters to be floated
		""" 
		self.theta_min = []
		self.theta_max = []
		for iparam in range(self.n_float_params):
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
		ifloat_params = 0

		for i in range(self.n_params):
			if i in self.fixed_params['fix_inds']:
				theta_ll[i] = self.fixed_params['fix_vals'][self.fixed_params['fix_inds'].index(i)]
			else:
				theta_ll[i] = theta[ifloat_params]
				ifloat_params += 1

		# print("Trying to feed in dm model ",self.dm_model)
		ll_val = functions_multinest.lnlike(theta_ll,self.projected_vel,self.vel_err,self.projected_distance,self.nbeta,10*np.exp(self.logrmax),self.nu_model,self.dm_model)

		# print(theta_ll,ll_val)
		return ll_val

	def perform_scan_multinest(self, chains_dir, nlive=100, pymultinest_options=None):
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

		pymultinest.run(self.ll, self.prior_cube, self.n_float_params, 
						outputfiles_basename=chains_dir, 
						n_live_points=nlive, **pymultinest_options_arg)

	def make_corner_plot(self, chains_dir, plots_dir, corner_args = {"color":"k"}):
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

		out_path = plots_dir+"/corner_"+self.nu_model+"_"+self.dm_model+self.run_tag
		# if self.fix_light_profile:
		# 	out_path += "_fix_lp"

		# if not self.load_light_profile:
		# 	out_path += "_float_lp_all"

		plt.savefig(out_path+".pdf")
		plt.close()
		print("Saved corner plot to ",out_path)


	def make_dm_corner_plot(self, chains_dir, plots_dir, corner_args = {"color":"k"}):
		self.make_dirs([plots_dir])
		post = pd.read_csv(chains_dir+"/post_equal_weights.dat",delim_whitespace=True,header=None).values[:,self.nstarparams+self.nbeta:-1]
		# Make corner plot
		ndim = post.shape[-1]
		fig = corner.corner(post,
							labels=self.params[self.nstarparams+self.nbeta:],
							max_n_ticks=4,
							quantiles=[0.16,0.5,0.84],
							show_titles=True, title_kwargs={"fontsize": 16},
							smooth=1,color=corner_args["color"]);

		ax = np.array(fig.axes).reshape((ndim,ndim))
		for i in range(ndim):
			for j in range(ndim):
				ax[i,j].tick_params(labelsize=14)

		out_path = plots_dir+"/corner_dm_"+self.nu_model+"_"+self.dm_model+self.run_tag
		# if self.fix_light_profile:
		# 	out_path += "_fix_lp"

		# if not self.load_light_profile:
		# 	out_path += "_float_lp_all"

		plt.savefig(out_path+".pdf")
		plt.close()
		print("Saved corner plot to ",out_path)


	def make_density_plots(self, chains_dir, plots_dir, post_dir, manual_rval = False, rval_eval = [], 
							truth_line_args = {"color":"k","ls":"--","label":"Truth (Zhao)"},
							results_args = {"color":"cornflowerblue","alpha":0.2},
							save_plots = True, save_post = False, has_truth = True):
		self.make_dirs([plots_dir])
		self.make_dirs([post_dir])

		if manual_rval:
			assert(len(rval_eval)>0), "Manual input r array is empty"
			self.rvals = rval_eval

		post = pd.read_csv(chains_dir+"/post_equal_weights.dat",delim_whitespace=True,header=None).values[:,:-1]
		post_stars = post[:,:self.nstarparams+self.nbeta]
		post_dm = post[:,self.nstarparams+self.nbeta:self.nstarparams+self.nbeta+self.ndmparams]

		post_rho = np.zeros((len(post),len(self.rvals)))
		post_Menc = np.zeros((len(post),len(self.rvals)))


		if self.dm_model == "BPL":
			if self.fix_breaks:
				raise ValueError("Plotting for fixed breaks not implemented yet")
			else:
				post_logrho0 = post_dm[:,0]
				post_logr0 = post_dm[:,1]
				post_gamma0 = post_dm[:,2]
				post_gamma1 = post_dm[:,3]
				
				post_gamma1 = post_gamma1 + post_gamma0
				
				for i in tqdm(range(len(post))):
					post_rho[i] = [DM_profiles.rho_DM(post_logrho0[i],post_logr0[i],post_gamma0[i],post_gamma1[i],r) for r in self.rvals]
					post_Menc[i] = [DM_profiles.mass_DM(post_logrho0[i],post_logr0[i],post_gamma0[i],post_gamma1[i],r) for r in self.rvals]
		
		elif self.dm_model == "BPL2":
			if self.fix_breaks:
				raise ValueError("Plotting for fixed breaks not implemented yet")
			else:
				post_logrho0 = post_dm[:,0]
				post_logr0 = post_dm[:,1]
				post_gamma0 = post_dm[:,2]
				post_gamma1 = post_dm[:,3]
				post_logr1 = post_dm[:,4]
				post_gamma2 = post_dm[:,5]
				
				post_gamma1 = post_gamma1 + post_gamma0
				post_gamma2 = post_gamma2 + post_gamma1
				
			#     post_logr1 = post_logr1 + post_logr0
				
				for i in tqdm(range(len(post))):
					post_rho[i] = [DM_profiles.rho_DM_2(post_logrho0[i],post_logr0[i],post_logr1[i],post_gamma0[i],post_gamma1[i],post_gamma2[i],r) for r in self.rvals]
					post_Menc[i] = [DM_profiles.mass_DM_2(post_logrho0[i],post_logr0[i],post_logr1[i],post_gamma0[i],post_gamma1[i],post_gamma2[i],r) for r in self.rvals]

		elif self.dm_model == "BPL3":
			if self.fix_breaks:
				raise ValueError("Plotting for fixed breaks not implemented yet")
			else:
				post_logrho0 = post_dm[:,0]
				post_logr0 = post_dm[:,1]
				post_gamma0 = post_dm[:,2]
				post_gamma1 = post_dm[:,3]
				post_logr1 = post_dm[:,4]
				post_gamma2 = post_dm[:,5]
				post_logr2 = post_dm[:,6]
				post_gamma3 = post_dm[:,7]
				
				post_gamma1 = post_gamma1 + post_gamma0
				post_gamma2 = post_gamma2 + post_gamma1
				post_gamma3 = post_gamma3 + post_gamma2
				
			#     post_logr1 = post_logr1 + post_logr0
			#     post_logr2 = post_logr2 + post_logr1
				
				for i in tqdm(range(len(post))):
					post_rho[i] = [DM_profiles.rho_DM_3(post_logrho0[i],post_logr0[i],post_logr1[i],post_logr2[i],post_gamma0[i],post_gamma1[i],post_gamma2[i],post_gamma3[i],r) for r in self.rvals]
					post_Menc[i] = [DM_profiles.mass_DM_3(post_logrho0[i],post_logr0[i],post_logr1[i],post_logr2[i],post_gamma0[i],post_gamma1[i],post_gamma2[i],post_gamma3[i],r) for r in self.rvals]

		elif self.dm_model == "NFW":
			post_logrho0 = post_dm[:,0]
			post_logr0 = post_dm[:,1]
			
			for i in tqdm(range(len(post))):
				post_rho[i] = [DM_profiles.rho_DM_NFW(post_logrho0[i],post_logr0[i],r) for r in self.rvals]
				post_Menc[i] = [DM_profiles.mass_DM_NFW(post_logrho0[i],post_logr0[i],r) for r in self.rvals]

		elif self.dm_model == "gNFW":
			post_logrho0 = post_dm[:,0]
			post_logr0 = post_dm[:,1]
			post_gamma = post_dm[:,2]
			
			for i in tqdm(range(len(post))):
				post_rho[i] = [DM_profiles.rho_DM_GNFW(post_logrho0[i],post_logr0[i],post_gamma[i],r) for r in self.rvals]
				post_Menc[i] = [DM_profiles.mass_DM_GNFW(post_logrho0[i],post_logr0[i],post_gamma[i],r) for r in self.rvals]
		
		elif self.dm_model == "Zhao":
			post_logrho0 = post_dm[:,0]
			post_logr0 = post_dm[:,1]
			post_alpha = post_dm[:,2]
			post_beta = post_dm[:,3]
			post_gamma = post_dm[:,4]
			
			for i in tqdm(range(len(post))):
				post_rho[i] = [DM_profiles.rho_DM_Zhao(post_logrho0[i],post_logr0[i],post_alpha[i],post_beta[i],post_gamma[i],r) for r in self.rvals]
				post_Menc[i] = [DM_profiles.mass_DM_Zhao(post_logrho0[i],post_logr0[i],post_alpha[i],post_beta[i],post_gamma[i],r) for r in self.rvals]
		
		elif self.dm_model == "Burkert":
			post_logrho0 = post_dm[:,0]
			post_logr0 = post_dm[:,1]
			
			for i in tqdm(range(len(post))):
				post_rho[i] = [DM_profiles.rho_DM_Burkert(post_logrho0[i],post_logr0[i],r) for r in self.rvals]
				post_Menc[i] = [DM_profiles.mass_DM_Burkert(post_logrho0[i],post_logr0[i],r) for r in self.rvals]

		elif self.dm_model == "gNFW_trunc":
			post_logrho0 = post_dm[:,0]
			post_logr0 = post_dm[:,1]
			post_gamma = post_dm[:,2]
			post_logrt = post_dm[:,3]
			
			for i in tqdm(range(len(post))):
				post_rho[i] = [DM_profiles.rho_DM_GNFW_truncated(post_logrho0[i],post_logr0[i],post_gamma[i],post_logrt[i],r) for r in self.rvals]
				post_Menc[i] = [DM_profiles.mass_DM_GNFW_truncated(post_logrho0[i],post_logr0[i],post_gamma[i],post_logrt[i],r) for r in self.rvals]		

		elif self.dm_model == "NFWc":
			post_logrho0 = post_dm[:,0]
			post_logr0 = post_dm[:,1]
			
			for i in tqdm(range(len(post))):
				post_rho[i] = [DM_profiles.rho_DM_NFW_cored(post_logrho0[i],post_logr0[i],r) for r in self.rvals]
				post_Menc[i] = [DM_profiles.mass_DM_NFW_cored(post_logrho0[i],post_logr0[i],r) for r in self.rvals]		
	
		else:
			raise ValueError("The plotting for the specified DM profile ",self.dm_model, " is not implemented")

		if has_truth:
			if self.true_profile == "PlumCuspIso":
				Menc_true = [DM_profiles.mass_DM_Zhao(np.log(0.064e9),np.log(1),1,3,1,r) for r in self.rvals]
				rho_true = [DM_profiles.rho_DM_Zhao(np.log(0.064e9),np.log(1),1,3,1,r) for r in self.rvals]

			elif self.true_profile == "PlumCoreIso":
				Menc_true = [DM_profiles.mass_DM_Zhao(np.log(0.4e9),np.log(1),1,3,0,r) for r in self.rvals]
				rho_true = [DM_profiles.rho_DM_Zhao(np.log(0.4e9),np.log(1),1,3,0,r) for r in self.rvals]
			else:
				# raise ValueError("True DM profile is not implemented yet")
				Menc_true = [DM_profiles.mass_DM_Zhao(*self.true_params,r) for r in self.rvals]
				rho_true = [DM_profiles.rho_DM_Zhao(*self.true_params,r) for r in self.rvals]

		if save_post:
			out_path = post_dir+"/rho_Menc_"+self.nu_model+"_"+self.dm_model+self.run_tag
			if has_truth:
				np.savez(out_path,rvals=self.rvals,rho=post_rho,rho_true=rho_true,Menc=post_Menc,Menc_true=Menc_true)
			else:
				np.savez(out_path,rvals=self.rvals,rho=post_rho,Menc=post_Menc)
		else:
			return self.rvals, post_rho, post_Menc


		dm_model_plot = self.dm_model
		if self.dm_model == "gNFW_trunc":
			dm_model_plot = "Truncated gNFW"
		elif self.dm_model == "NFWc":
			dm_model_plot = "Cored NFW"

		if save_plots and has_truth:
			f, ax = plt.subplots(2,2,gridspec_kw={'height_ratios': [4, 1]},sharex=True)
			f.set_figheight(7.5)
			f.set_figwidth(14)

			ax[0,0].plot(self.rvals,np.median(post_rho,axis=0),color=results_args["color"],label=dm_model_plot+r' fit')
			ax[0,0].fill_between(self.rvals,np.percentile(post_rho,q=16,axis=0),np.percentile(post_rho,q=84,axis=0),alpha=results_args["alpha"],color=results_args["color"])
			ax[0,0].fill_between(self.rvals,np.percentile(post_rho,q=2.5,axis=0),np.percentile(post_rho,q=97.5,axis=0),alpha=results_args["alpha"],color=results_args["color"])
			ax[0,0].plot(self.rvals,rho_true,ls=truth_line_args["ls"],color=truth_line_args["color"],label=truth_line_args["label"])
			
			ax[0,0].set_yscale('log')
			ax[0,0].set_xscale('log')

			# ax[0,0].set_xlabel(r'$R$ [kpc]')
			# ax[0,0].set_ylabel(r'$\rho_\mathrm{DM}(R)\,[M_\odot/\mathrm{kpc}^3]$')
			ax[0,0].set_ylabel(r'$\rho(r)\,[M_\odot/\mathrm{kpc}^3]$')
			ax[0,0].set_xlim(np.exp(self.logrmin),np.exp(self.logrmax));
			ax[0,0].set_ylim(2e4,5e10);

			ax[0,1].plot(self.rvals,np.median(post_Menc,axis=0),color=results_args["color"],label=dm_model_plot+r' fit')
			ax[0,1].fill_between(self.rvals,np.percentile(post_Menc,q=16,axis=0),np.percentile(post_Menc,q=84,axis=0),alpha=results_args["alpha"],color=results_args["color"])
			ax[0,1].fill_between(self.rvals,np.percentile(post_Menc,q=2.5,axis=0),np.percentile(post_Menc,q=97.5,axis=0),alpha=results_args["alpha"],color=results_args["color"])
			ax[0,1].plot(self.rvals,Menc_true,ls=truth_line_args["ls"],color=truth_line_args["color"],label=truth_line_args["label"])

			ax[0,1].set_yscale('log')
			ax[0,1].set_xscale('log')

			# ax[0,1].set_xlabel(r'$R$ [kpc]')
			# ax[0,1].set_ylabel(r'M$_\mathrm{enc}(R)\,[M_\odot]$')
			ax[0,1].set_ylabel(r'M$(r)\,[M_\odot]$')
			ax[0,1].set_xlim(np.exp(self.logrmin),np.exp(self.logrmax));
			ax[0,1].set_ylim(2e4,5e10);
			ax[0,1].legend(loc='lower right')

			ax[1,0].plot(self.rvals,np.median(post_rho,axis=0)/rho_true,color=results_args["color"],label=dm_model_plot+r' fit')
			ax[1,0].fill_between(self.rvals,np.percentile(post_rho/rho_true,q=16,axis=0),np.percentile(post_rho/rho_true,q=84,axis=0),alpha=results_args["alpha"],color=results_args["color"])
			ax[1,0].fill_between(self.rvals,np.percentile(post_rho/rho_true,q=2.5,axis=0),np.percentile(post_rho/rho_true,q=97.5,axis=0),alpha=results_args["alpha"],color=results_args["color"])
			ax[1,0].axhline(1,ls=truth_line_args["ls"],color=truth_line_args["color"],label=truth_line_args["label"])
			
			ax[1,0].set_yscale('log')
			ax[1,0].set_xscale('log')

			ax[1,0].set_xlabel(r'$r$ [kpc]')
			# ax[1,0].set_ylabel(r'$\rho_\mathrm{DM}(R)/\rho_\mathrm{DM}^\mathrm{True}(R)$')
			ax[1,0].set_ylabel(r'$\rho(r)/\rho^\mathrm{True}(r)$')
			ax[1,0].set_xlim(np.exp(self.logrmin),np.exp(self.logrmax));
			ax[1,0].set_ylim(1e-2,1e2);
			ax[1,0].set_yticks([1e-2,1e0,1e2])

			ax[1,1].plot(self.rvals,np.median(post_Menc,axis=0)/Menc_true,color=results_args["color"],label=dm_model_plot+r' fit')
			ax[1,1].fill_between(self.rvals,np.percentile(post_Menc,q=16,axis=0)/Menc_true,np.percentile(post_Menc,q=84,axis=0)/Menc_true,alpha=results_args["alpha"],color=results_args["color"])
			ax[1,1].fill_between(self.rvals,np.percentile(post_Menc,q=2.5,axis=0)/Menc_true,np.percentile(post_Menc,q=97.5,axis=0)/Menc_true,alpha=results_args["alpha"],color=results_args["color"])
			ax[1,1].axhline(1,ls=truth_line_args["ls"],color=truth_line_args["color"],label=truth_line_args["label"])

			ax[1,1].set_yscale('log')
			ax[1,1].set_xscale('log')

			ax[1,1].set_xlabel(r'$r$ [kpc]')
			ax[1,1].set_ylabel(r'M$(r)/$M$^\mathrm{True}(r)$')
			ax[1,1].set_xlim(np.exp(self.logrmin),np.exp(self.logrmax));
			ax[1,1].set_ylim(1e-2,1e2);
			ax[1,1].set_yticks([1e-2,1e0,1e2])

			for thresh in [1e-1,1e1]:
				ax[1,0].axhline(thresh,ls=':',color='dimgray')
				ax[1,1].axhline(thresh,ls=':',color='dimgray')

			for i in range(2):
				for j in range(2):
					ax[i,j].minorticks_on()
					ax[i,j].xaxis.set_ticks_position('both')
					ax[i,j].yaxis.set_ticks_position('both')

					# ax[i,j].tick_params('both', length=8, width=1, which='major')
					# ax[i,j].tick_params('both', length=4, width=1, which='minor')

			sup = plt.suptitle(self.nu_model+' stars, '+dm_model_plot+' DM', y= 1.04, fontsize=28)
			f.tight_layout()
			f.subplots_adjust(hspace=0.15)

			out_path = plots_dir+"/rho_Menc_"+self.nu_model+"_"+self.dm_model+self.run_tag
			# if self.fix_light_profile:
			# 	out_path += "_fix_lp"

			# if not self.load_light_profile:
			# 	out_path += "_float_lp_all"

			plt.savefig(out_path+".pdf",bbox_extra_artists=(sup,), bbox_inches="tight")
			plt.close()
			print("Saved density plot to ",out_path)

		elif save_plots and not has_truth:
			f, ax = plt.subplots(1,2)
			f.set_figheight(6)
			f.set_figwidth(14)

			ax[0].plot(self.rvals,np.median(post_rho,axis=0),color=results_args["color"],label=dm_model_plot+r' fit')
			ax[0].fill_between(self.rvals,np.percentile(post_rho,q=16,axis=0),np.percentile(post_rho,q=84,axis=0),alpha=results_args["alpha"],color=results_args["color"])
			ax[0].fill_between(self.rvals,np.percentile(post_rho,q=2.5,axis=0),np.percentile(post_rho,q=97.5,axis=0),alpha=results_args["alpha"],color=results_args["color"])
			
			ax[0].set_yscale('log')
			ax[0].set_xscale('log')

			ax[0].set_xlabel(r'$R$ [kpc]')
			ax[0].set_ylabel(r'$\rho_\mathrm{DM}(r)\,[M_\odot/\mathrm{kpc}^3]$')
			ax[0].set_xlim(np.exp(self.logrmin),np.exp(self.logrmax));
			ax[0].set_ylim(2e4,1e10);

			ax[1].plot(self.rvals,np.median(post_Menc,axis=0),color=results_args["color"],label=dm_model_plot+r' fit')
			ax[1].fill_between(self.rvals,np.percentile(post_Menc,q=16,axis=0),np.percentile(post_Menc,q=84,axis=0),alpha=results_args["alpha"],color=results_args["color"])
			ax[1].fill_between(self.rvals,np.percentile(post_Menc,q=2.5,axis=0),np.percentile(post_Menc,q=97.5,axis=0),alpha=results_args["alpha"],color=results_args["color"])

			ax[1].set_yscale('log')
			ax[1].set_xscale('log')

			ax[1].set_xlabel(r'$R$ [kpc]')
			ax[1].set_ylabel(r'M$_\mathrm{enc}(r)\,[M_\odot]$')
			ax[1].set_xlim(np.exp(self.logrmin),np.exp(self.logrmax));
			ax[1].set_ylim(2e4,1e10);
			ax[1].legend(loc='lower right')

			sup = plt.suptitle(self.nu_model+' stars, '+dm_model_plot+' DM', y= 1.04, fontsize=28)
			f.tight_layout()

			out_path = plots_dir+"/rho_Menc_"+self.nu_model+"_"+self.dm_model+self.run_tag
			# if self.fix_light_profile:
			# 	out_path += "_fix_lp"

			# if not self.load_light_profile:
			# 	out_path += "_float_lp_all"

			plt.savefig(out_path+".pdf",bbox_extra_artists=(sup,), bbox_inches="tight")
			plt.close()


	def get_dispersion(self, chains_dir, sigmap_path, plots_dir, post_dir, save_plots = False, save_post = True,
						truth_marker_args = {"color":"k","fmt":"o","capsize":4,"label":"Data"},
						results_args = {"color":"cornflowerblue","alpha":0.2},):

		self.make_dirs([plots_dir])
		self.make_dirs([post_dir])
		post = pd.read_csv(chains_dir+"/post_equal_weights.dat",delim_whitespace=True,header=None).values[:,:-1]
		post_stars = post[:,:self.nstarparams]
		post_beta = post[:,self.nstarparams:self.nstarparams+self.nbeta]
		post_dm = post[:,self.nstarparams+self.nbeta:self.nstarparams+self.nbeta+self.ndmparams]

		post_sigmap2 = np.zeros((len(post),len(self.rvals)))

		if self.nbeta == 0:
			for i in tqdm(range(len(post))):
				post_sigmap2[i] = [functions_multinest.sigmap2(post_stars[i],np.array([]),post_dm[i],r,isotropic=1,nu_model=self.nu_model,dm_model=self.dm_model) for r in self.rvals]
		else:
			print("Dispersion plotting code for nonzero anisotropy not implemented yet")
		if save_post:
			out_path = post_dir+"/sigmap2_"+self.nu_model+"_"+self.dm_model+self.run_tag
			np.savez(out_path,rvals=self.rvals,sigmap2=post_sigmap2)
		if save_plots:
			if os.path.exists(sigmap_path+".npz"):
				sigmap_obs = np.load(sigmap_path+".npz")
				bin_centers = sigmap_obs["bin_centers"]
				vel_disp_binned = sigmap_obs["vel_disp_binned"]
				vel_disp_errs_binned = sigmap_obs["vel_disp_errs_binned"]
				plt.errorbar(bin_centers,vel_disp_binned,yerr=vel_disp_errs_binned,color=truth_marker_args["color"],fmt=truth_marker_args["fmt"],label=truth_marker_args["label"],capsize=truth_marker_args["capsize"])

			else:
				raise ValueError("Data binned dispersions not calculated; skipping plotting truths")

			plt.plot(self.rvals,np.sqrt(np.median(post_sigmap2,axis=0)),color=results_args["color"]) # Take sqrt to plot sigma_p instead of sigma_p^2
			plt.fill_between(self.rvals,np.sqrt(np.percentile(post_sigmap2,axis=0,q=16)),np.sqrt(np.percentile(post_sigmap2,axis=0,q=84)),color=results_args["color"],alpha=results_args["alpha"])

			plt.xscale('log')

			plt.xlabel(r'$R$ [kpc]')
			plt.ylabel(r'$\sigma_p(R)$ [km/s]')
			plt.xlim(np.exp(self.logrmin),np.exp(self.logrmax));
			plt.ylim(0,);

			plt.tight_layout()

			out_path = plots_dir+"/sigmap_"+self.nu_model+"_"+self.dm_model+self.run_tag
			plt.savefig(out_path+".pdf")
			plt.close()

	def calculate_j_factor(self, distance, angle_max, chains_dir, full_posterior = False, post_dir = ''):
		"""
		Function to calculate the J-factor of a dwarf at distance, with max opening angle angle_max
		:param: distance, float, in kpc
		:param: angle_max, float, in *degrees!*
		:param: chains_dir, str, location of the chains
		:param: full_posterior, boolean, if true, calculates the J-factors of the entire posterior. 
		:param: post_dir, str, location of where the files are gonna get saved
		"""
		cos_angle_max = np.cos(np.radians(angle_max))


		post = pd.read_csv(chains_dir+"/post_equal_weights.dat",delim_whitespace=True,header=None).values[:,:-1]

		post_dm = post[:,self.nstarparams+self.nbeta:self.nstarparams+self.nbeta+self.ndmparams]

		jfac_truth = DM_profiles.J_factor_Zhao(distance, cos_angle_max, *self.true_params)

		if self.dm_model == "gNFW":
			post_logrho0 = post_dm[:,0]
			post_logr0 = post_dm[:,1]
			post_gamma = post_dm[:,2]


			if full_posterior:
				jfac = DM_profiles.J_factor_gNFW_vectorized(distance, cos_angle_max, post_logrho0, post_logr0, post_gamma)
				
			else:
				log_rho0_low, log_rho0_median, log_rho0_high = np.percentile(post_logrho0, q=[16, 50, 84], axis=0)
				log_r0_low, log_r0_median, log_r0_high = np.percentile(post_logr0, q=[16, 50, 84], axis=0)
				gamma_low, gamma_median, gamma_high = np.percentile(post_gamma, q=[16, 50, 84], axis=0)

				jfac_low = DM_profiles.J_factor_gNFW(distance, cos_angle_max, log_rho0_low, log_r0_low, gamma_low)
				jfac_median = DM_profiles.J_factor_gNFW(distance, cos_angle_max, log_rho0_median, log_r0_median, gamma_median)
				jfac_high = DM_profiles.J_factor_gNFW(distance, cos_angle_max, log_rho0_high, log_r0_high, gamma_high)
				jfac = [jfac_low, jfac_median, jfac_high]

		elif self.dm_model == "NFW":
			post_logrho0 = post_dm[:,0]
			post_logr0 = post_dm[:,1]

			if full_posterior:
				jfac = DM_profiles.J_factor_NFW_vectorized(distance, cos_angle_max, post_logrho0, post_logr0)
				
			else:
				log_rho0_low, log_rho0_median, log_rho0_high = np.percentile(post_logrho0, q=[16, 50, 84], axis=0)
				log_r0_low, log_r0_median, log_r0_high = np.percentile(post_logr0, q=[16, 50, 84], axis=0)

				jfac_low = DM_profiles.J_factor_NFW(distance, cos_angle_max, log_rho0_low, log_r0_low)
				jfac_median = DM_profiles.J_factor_NFW(distance, cos_angle_max, log_rho0_median, log_r0_median)
				jfac_high = DM_profiles.J_factor_NFW(distance, cos_angle_max, log_rho0_high, log_r0_high)
				jfac = [jfac_low, jfac_median, jfac_high]

		if self.dm_model == "NFWc":
			post_logrho0 = post_dm[:,0]
			post_logr0 = post_dm[:,1]

			if full_posterior:
				jfac = DM_profiles.J_factor_NFWc_vectorized(distance, cos_angle_max, post_logrho0, post_logr0)
				
			else:
				log_rho0_low, log_rho0_median, log_rho0_high = np.percentile(post_logrho0, q=[16, 50, 84], axis=0)
				log_r0_low, log_r0_median, log_r0_high = np.percentile(post_logr0, q=[16, 50, 84], axis=0)

				jfac_low = DM_profiles.J_factor_NFWc(distance, cos_angle_max, log_rho0_low, log_r0_low)
				jfac_median = DM_profiles.J_factor_NFWc(distance, cos_angle_max, log_rho0_median, log_r0_median)
				jfac_high = DM_profiles.J_factor_NFWc(distance, cos_angle_max, log_rho0_high, log_r0_high)
				jfac = [jfac_low, jfac_median, jfac_high]

		out_path = post_dir+"jfac_"+self.nu_model+"_"+self.dm_model+self.run_tag

		if full_posterior:
			out_path += '_full'

		if self.verbose:
			print("Saving the Jfactors in", out_path)
		np.savez(out_path, jfac = jfac, jfac_truth = jfac_truth)


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

	@staticmethod
	def get_bins(rsamp,nbins):
		binned_data_0 = np.zeros((nbins))
		bin_centers_0 = np.zeros((nbins))

		num_per_bin = int(np.ceil(len(rsamp)/nbins))
		num_last_bin = len(rsamp)-num_per_bin*(nbins-1)
		binned_data = np.zeros((nbins,num_per_bin))
		bins = np.zeros(nbins-1)
		inds = np.argsort(rsamp)
		sorted_R_vals = rsamp[inds]

		if num_per_bin != num_last_bin:
			for ibin in range(nbins-1):
				binned_data[ibin] = sorted_R_vals[ibin*num_per_bin:(ibin+1)*num_per_bin]
			binned_data[-1] = np.pad(sorted_R_vals[num_per_bin*(nbins-1):],(0,num_per_bin-num_last_bin),'constant')
		else:
			for ibin in range(nbins):
				binned_data[ibin] = sorted_R_vals[ibin*num_per_bin:(ibin+1)*num_per_bin]
				
		for i in range(nbins-1):
			bins[i] = np.sqrt(binned_data[i,-1]*binned_data[i+1,0])
		return bins
#######################################
# run_MCMC.py: file for the likelihood function of the emcee for the velocity dispersion
# Lina Necib, November 18, 2018, Caltech
# Laura J. Chang, Princeton
#######################################

# from __future__ import division
import sys
# sys.path.append('/tigress/ljchang/dSph_MCMC/RunFiles/')
import anisotropy_funcs, DM_profiles, star_profiles

# from scipy.special.cython_special cimport erf, erfc
from scipy.integrate import quad, simps
from libc.math cimport log, sqrt, pi, exp, atan2
import numpy as np
cimport numpy as np

cdef double twopi = 2*pi
cdef double sqrt2 = sqrt(2)
cdef double sqrt2pi = sqrt(twopi)
cdef double G = 4.302e-6 # in units of kpc/M_sun*(km/s)**2

DTYPE = np.float64 #[double]
ctypedef np.float_t DTYPEf_t

################################################################
# Vectorized DM densities (actual functions in DM_profiles.py) #
################################################################

mass_DM_vectorized = np.vectorize(DM_profiles.mass_DM)
mass_DM_2_vectorized = np.vectorize(DM_profiles.mass_DM_2)
mass_DM_3_vectorized = np.vectorize(DM_profiles.mass_DM_3)
mass_DM_4_vectorized = np.vectorize(DM_profiles.mass_DM_4)
mass_DM_5_vectorized = np.vectorize(DM_profiles.mass_DM_5)

mass_DM_NFW_vectorized = np.vectorize(DM_profiles.mass_DM_NFW)
mass_DM_Burkert_vectorized = np.vectorize(DM_profiles.mass_DM_Burkert)
mass_DM_Zhao_vectorized = np.vectorize(DM_profiles.mass_DM_Zhao)
mass_DM_GNFW_vectorized = np.vectorize(DM_profiles.mass_DM_GNFW)
mass_DM_GNFW_truncated_vectorized = np.vectorize(DM_profiles.mass_DM_GNFW_truncated)
mass_DM_NFW_cored_vectorized = np.vectorize(DM_profiles.mass_DM_NFW_cored)

###############################################################################
# Vectorized stellar surface densities (actual functions in star_profiles.py) #
###############################################################################

nu_star_Plummer_vectorized = np.vectorize(star_profiles.nu_star_Plummer)
nu_star_Plummer_2_vectorized = np.vectorize(star_profiles.nu_star_Plummer_2)
nu_star_Plummer_3_vectorized = np.vectorize(star_profiles.nu_star_Plummer_3)
nu_star_Zhao_vectorized = np.vectorize(star_profiles.nu_star_Zhao)


#### NOTE: Changing this to relative breaks ####
def sigmap2(np.ndarray theta_star, np.ndarray theta_beta, np.ndarray theta_DM, double R, double r_max = 20, int isotropic = 0, str nu_model = "Plummer", str dm_model = "gNFW"): 
	"""
	Projected dispersion, with the Osipkov parameterization for the anisotropy beta
	:theta_star: model parameters of the stars
	:theta_beta: model parameters of the anisotropy
	:theta_DM: model parameters of the DM distribution
	:log_R: projected radial distance
	:r_max: maximum distance to integrate to, which can be np.inf
	:param: isotropic: boolean, if true, use the isotropic definition of the kernel
	:param: nu_model: the light profile model; implemented options are 
					  1-component Plummer ("Plummer"), 
					  2-component Plummer ("Plummer2"), 
					  3-component Plummer ("Plummer3"),
					  Zhao ("Zhao")
	:param: dm_model: the DM profile model; implemented options are 
					  1-break broken power law ("BPL"),
					  2-break broken power law ("BPL2"),
					  3-break broken power law ("BPL3"),
					  Zhao ("Zhao"),
					  NFW ("NFW"),
					  Burkert ("Burkert"),
					  generalized NFW ("gNFW")
	"""

	# print "theta_beta", theta_beta

	# Allocate memory for params for all the models, just to have the space allocated regardless of model
	cdef double r_a 
	cdef double log_nus_star, log_rs_star, alpha_star, beta_star, gamma_star # params for Zhao light profile
	cdef double log_M0, log_a0, log_M1, log_a1, log_M2, log_a2  # params for Plummer sphere light profile
	cdef double log_rho0, log_r0, gamma0, gamma1, log_r1, gamma2, log_r2, gamma3, log_r3, gamma4 # params for BPL DM profile
	cdef double alpha, beta, gamma # params for gNFW/Zhao DM profile
	cdef double integral

	# cdef int nplummer_parameters = 2
	# cdef int nplummer = len(theta_star) / nplummer_parameters

	cdef int n_samples = 10000
	cdef np.ndarray[DTYPEf_t, ndim=1] radial_integral = np.linspace(R, r_max, n_samples)
	cdef np.ndarray[DTYPEf_t, ndim=1] kernel = np.zeros(n_samples, dtype=DTYPE)
	cdef double norm
	cdef np.ndarray[DTYPEf_t, ndim=1] stellar_density = np.zeros(n_samples, dtype=DTYPE)
	cdef np.ndarray[DTYPEf_t, ndim=1] dark_matter_mass = np.zeros(n_samples, dtype=DTYPE)
	cdef np.ndarray[DTYPEf_t, ndim=1] y_integral = np.zeros(n_samples, dtype=DTYPE)

	if nu_model == "Plummer":
		log_M0, log_a0 = theta_star
		norm = 2*G/star_profiles.Sigma_star_Plummer(log_M0, log_a0, R)
		stellar_density = nu_star_Plummer_vectorized(log_M0, log_a0, radial_integral)

	elif nu_model == "Plummer2":
		log_M0, log_a0, log_M1, log_a1 = theta_star
		# Implement relative priors
		log_M1 = log_M1 + log_M0
		log_a1 = log_a1 + log_a0
		norm = 2*G/star_profiles.Sigma_star_Plummer_2(log_M0, log_a0, log_M1, log_a1 , R)
		stellar_density = nu_star_Plummer_2_vectorized(log_M0, log_a0, log_M1, log_a1, radial_integral)

	elif nu_model == "Plummer3":
		log_M0, log_a0, log_M1, log_a1, log_M2, log_a2 = theta_star 
		# Implement relative priors
		log_M1 = log_M1 + log_M0
		log_M2 = log_M2 + log_M1
		log_a1 = log_a1 + log_a0
		log_a2 = log_a2 + log_a1
		norm = 2*G/star_profiles.Sigma_star_Plummer_3(log_M0, log_a0, log_M1, log_a1, log_M2, log_a2, R)
		stellar_density = nu_star_Plummer_3_vectorized(log_M0, log_a0, log_M1, log_a1, log_M2, log_a2, radial_integral)

	elif nu_model == "Zhao":
		log_nus_star, log_rs_star, alpha_star, beta_star, gamma_star = theta_star
		norm = 2*G/star_profiles.Sigma_star_Zhao(log_nus_star, log_rs_star, alpha_star, beta_star, gamma_star, R)
		stellar_density = nu_star_Zhao_vectorized(log_nus_star, log_rs_star, alpha_star, beta_star, gamma_star, radial_integral)

	else:
		raise ValueError("The specified light profile model is not implemented")

	# cdef int ndm_parameters = 2
	# cdef int ndm_parameters_first = 4
	# cdef int npower_laws = int(1) + (len(theta_DM) - ndm_parameters_first )/ndm_parameters

	if isotropic:
		kernel = anisotropy_funcs.K_isotropic(radial_integral/R)
	else:
		r_a = exp(theta_beta[0])
		kernel= anisotropy_funcs.K_osipkov(radial_integral/R,r_a/R)
		# print("r_a=",r_a)

	if dm_model == "BPL":
		assert(len(theta_DM) == 4), "The dimensions of theta_DM doesn't match the number of model parameters"
		log_rho0, log_r0, gamma0, gamma1 = theta_DM
		
		# Implement relative breaks/slopes
		gamma1 = gamma0 + gamma1
		dark_matter_mass = mass_DM_vectorized(log_rho0, log_r0, gamma0, gamma1, radial_integral)

	elif dm_model == "BPL2":
		assert(len(theta_DM) == 6), "The dimensions of theta_DM doesn't match the number of model parameters"
		log_rho0, log_r0, gamma0, gamma1, log_r1, gamma2 = theta_DM
		
		# Implement relative breaks/slopes
		gamma1 = gamma0 + gamma1
		gamma2 = gamma1 + gamma2
		# log_r1 = log_r0 + log_r1
		dark_matter_mass = mass_DM_2_vectorized(log_rho0, log_r0, log_r1, gamma0, gamma1, gamma2, radial_integral)
	
	elif dm_model == "BPL3":
		assert(len(theta_DM) == 8), "The dimensions of theta_DM doesn't match the number of model parameters"
		log_rho0, log_r0, gamma0, gamma1, log_r1, gamma2, log_r2, gamma3 = theta_DM
		
		# Implement relative breaks/slopes
		gamma1 = gamma0 + gamma1
		gamma2 = gamma1 + gamma2
		gamma3 = gamma2 + gamma3
		# log_r1 = log_r0 + log_r1
		# log_r2 = log_r1 + log_r2
		dark_matter_mass = mass_DM_3_vectorized(log_rho0, log_r0, log_r1, log_r2, gamma0, gamma1, gamma2, gamma3, radial_integral)

	elif dm_model == "Zhao":
		assert(len(theta_DM) == 5), "The dimensions of theta_DM doesn't match the number of model parameters"
		log_rho0, log_r0, alpha, beta, gamma = theta_DM
		dark_matter_mass = mass_DM_Zhao_vectorized(log_rho0, log_r0, alpha, beta, gamma, radial_integral)

	elif dm_model == "NFW":
		assert(len(theta_DM) == 2), "The dimensions of theta_DM doesn't match the number of model parameters"
		log_rho0, log_r0 = theta_DM
		dark_matter_mass = mass_DM_NFW_vectorized(log_rho0, log_r0, radial_integral)

	elif dm_model == "Burkert":
		assert(len(theta_DM) == 2), "The dimensions of theta_DM doesn't match the number of model parameters"
		log_rho0, log_r0 = theta_DM
		dark_matter_mass = mass_DM_Burkert_vectorized(log_rho0, log_r0, radial_integral)

	elif dm_model == "gNFW":
		assert(len(theta_DM) == 3), "The dimensions of theta_DM doesn't match the number of model parameters"
		log_rho0, log_r0, gamma = theta_DM
		dark_matter_mass = mass_DM_GNFW_vectorized(log_rho0, log_r0, gamma, radial_integral)

	elif dm_model == "gNFW_trunc":
		assert(len(theta_DM) == 4), "The dimensions of theta_DM doesn't match the number of model parameters"
		log_rho0, log_r0, gamma, log_rt = theta_DM
		n = 1 # this is the only value of n currently implemented
		dark_matter_mass = mass_DM_GNFW_truncated_vectorized(log_rho0, log_r0, gamma, log_rt, n, radial_integral)

	elif dm_model == "NFWc":
		assert(len(theta_DM) == 2), "The dimensions of theta_DM doesn't match the number of model parameters"
		log_rho0, log_r0 = theta_DM
		dark_matter_mass = mass_DM_NFW_cored_vectorized(log_rho0, log_r0, radial_integral)

	else:
		raise ValueError("The specified DM model", dm_model, " is not implemented")

	y_integral = kernel*stellar_density*dark_matter_mass/radial_integral
	
	# print("Integrating to rmax=", r_max)

	integral = norm*simps(y_integral, radial_integral)

	return integral


######### Unbinned analysis

def lnlike(np.ndarray theta, np.ndarray vi, np.ndarray delta_vi, np.ndarray Ri, int n_beta_params, double r_max, str nu_model="Plummer", str dm_model="gNFW"):
	"""
	Log likelihood of the function evaluated, later we will add outlier model
	:param: theta: model parameters
	:param: vi: speed of the stars
	:param: delta_vi: error on the speed stars
	:param: Ri: projected distance of the stars from the distance
	:param: n_star_params: number of star parameters
	:param: n_beta_params: number of beta parameters
	:param: nu_model: the light profile model; implemented options are 
					  1-component Plummer ("Plummer"), 
					  2-component Plummer ("Plummer2"), 
					  3-component Plummer ("Plummer3"),
					  Zhao ("Zhao")
	:param: dm_model: the DM profile model; implemented options are 
					  1-break broken power law ("BPL"),
					  2-break broken power law ("BPL2"),
					  3-break broken power law ("BPL3"),
					  Zhao ("Zhao"),
					  NFW ("NFW"),
					  Burkert ("Burkert"),
					  generalized NFW ("gNFW")

	""" 

	# sigmap should be evaluated at Ri, the projected distance of that particular star
	# vmean should be also fit for as part of theta

	assert theta.dtype == DTYPE and vi.dtype == DTYPE

	cdef np.ndarray theta_beta
	cdef np.ndarray theta_DM
	cdef np.ndarray theta_star
	cdef double vmean

	# Stars
	cdef int nplummer_parameters = 2
	cdef int n_star_params
	# cdef int n_star_params = nplummer_parameters*nplummer

	# Dark Matter
	cdef int ndm_parameters_first = 4
	cdef int n_dm_params
	# cdef int n_dm_params = ndm_parameters_first + 2 * (npower_laws - 1)

	if nu_model == "Plummer": 
		n_star_params = nplummer_parameters
	elif nu_model == "Plummer2":
		n_star_params = nplummer_parameters*2
	elif nu_model == "Plummer3":
		n_star_params = nplummer_parameters*3
	elif nu_model == "Zhao":
		n_star_params = 5
	else:
		raise ValueError("The specified light profile model is not implemented")

	if dm_model == "BPL":
		n_dm_params = ndm_parameters_first
	elif dm_model == "BPL2":
		n_dm_params = ndm_parameters_first+2
	elif dm_model == "BPL3":
		n_dm_params = ndm_parameters_first+4
	elif dm_model == "Zhao":
		n_dm_params = 5
	elif dm_model == "NFW" or dm_model == "Burkert" or dm_model == "NFWc":
		n_dm_params = 2
	elif dm_model == "gNFW":
		n_dm_params = 3
	elif dm_model == "gNFW_trunc":
		n_dm_params = 4
	else: 
		raise ValueError("The specified DM density profile model is not implemented")

	theta_star = theta[:n_star_params]
	theta_beta = theta[n_star_params:n_star_params+n_beta_params]
	theta_DM = theta[n_star_params + n_beta_params : n_star_params + n_beta_params + n_dm_params]
	vmean = theta[-1]
	
	assert(len(theta_star) == n_star_params), "Length of theta_star is wrong"
	assert(len(theta_beta) == n_beta_params), "Length of theta_beta is wrong"
	assert(len(theta_DM) == n_dm_params), "Length of theta_DM is wrong"
	
	cdef np.ndarray sigmap, prefactor, exponential_term

	cdef int isotropic = 1
	if n_beta_params == 1:
		isotropic = 0

	sigmap = np.array([sigmap2(theta_star, theta_beta, theta_DM, r, r_max=r_max, isotropic=isotropic, nu_model=nu_model, dm_model=dm_model) for r in Ri])

	prefactor = -0.5*log(twopi) - 0.5*np.log(sigmap + delta_vi**2) # sigmap is already sigma squared

	exponential_term = -0.5*( (vi - vmean)*(vi - vmean) / (sigmap + delta_vi**2) )

	return np.sum(prefactor + exponential_term)

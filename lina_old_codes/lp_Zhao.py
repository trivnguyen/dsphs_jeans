###############################################################
#                                                             #
# This file includes the functions and likelihoods used in    #
# the initial light profile fit that is performed prior to    #
# performing Jeans analyses on stellar kinematic data,        #
# roughly following the analysis presented in [1504.02048].   #
# Since this fit only involves the stellar parameters, the    #
# subscript "_star" is used to avoid later confusion with DM  #
# parameters in the full Jean analysis.                       #
#                                                             #
###############################################################

import sys, os
import numpy as np
from scipy import integrate
from scipy.stats import chi2

################################
# Zhao functions & likelihoods #
################################

def nu_Zhao(log_nus_star,log_rs_star,alpha,beta,gamma,log_r):
	"""
	The tracer (3d) density of a given dSph, in the Zhao parametrization. 
	This quantity cannot be physically measured, and is related to the 
	measured projected quantity Sigma_Zhao by Abel transform.
	See, e.g. Eq(7) of [1504.02048].
	
	:param: log_nus_star: (log10 of) the normalization parameter
	:param: log_rs_star: (log10 of) the scale radius 
	:param: alpha, beta, gamma: slope parameters
	:param: log_r: (log10 of) the (3d) radius to evaluate at
	"""
	r,nus_star,rs_star = 10**log_r, 10**log_nus_star, 10**log_rs_star
	rs = r/rs_star
	return nus_star/((rs**gamma)*((1+rs**alpha)**((beta-gamma)/alpha)))

def Sigma_Zhao(log_nus_star,log_rs_star,alpha,beta,gamma,log_R):
	"""
	The tracer surface density of a given dSph, in the Zhao parametrization. 
	This is related to the 3d stellar density profile nu_Zhao by inverse Abel transform.
	See, e.g. Eq(4)&(7) of [1504.02048].
	
	:param: log_nus_star: (log10 of) the normalization parameter
	:param: log_rs_star: (log10 of) the scale radius 
	:param: alpha, beta, gamma: slope parameters
	:param: log_R: (log10 of) the projected radius to evaluate at, measured from the center
					of the dSph
	"""
	R = 10**log_R
	return integrate.quad(lambda r: 2*nu_Zhao(log_nus_star,log_rs_star,alpha,beta,gamma,np.log10(r))*r/np.sqrt(r**2-R**2),R,np.inf)[0]

def Sigma_Zhao_bkg(log_nus_star,log_rs_star,alpha,beta,gamma,log_Sigma_bkg,log_R):
	"""
	The tracer surface density of a given dSph, in the Zhao parametrization, with an 
	additional constant background term. 
	
	:param: log_nus_star: (log10 of) the normalization parameter
	:param: log_rs_star: (log10 of) the scale radius 
	:param: alpha, beta, gamma: slope parameters
	:param: log_Sigma_bkg: (log10 of) constant background
	:param: log_R: (log10 of) the projected radius to evaluate at, measured from the center
					of the dSph
	"""
	R = 10**log_R
	return integrate.quad(lambda r: 2*nu_Zhao(log_nus_star,log_rs_star,alpha,beta,gamma,np.log10(r))*r/np.sqrt(r**2-R**2),R,np.inf)[0]+10**log_Sigma_bkg

def lnlike_Zhao_binned_bkg(log_nus_star,log_rs_star,alpha,beta,gamma,log_Sigma_bkg,lower_rvals,upper_rvals,rvals,binned_counts,binned_errors_lo,binned_errors_hi,approx_scheme="VarGauss2",use_counts=False):
	"""
	The binned log likelihood in the Zhao parametrization, with an additional constant background term. 
	
	:param: log_nus_star: (log10 of) the normalization parameter
	:param: log_rs_star: (log10 of) the scale radius 
	:param: alpha, beta, gamma: slope parameters
	:param: log_Sigma_bkg: (log10 of) constant background
	:param: lower_rvals: the lower bin edges of the radially binned data, used to calculate the
						 area of each radial bin
	:param: upper_rvals: the upper bin edges of the radially binned data, used to calculate the
						 area of each radial bin
	:param: rvals: the bin centers of the radially binned data
	:param: binned_counts: the number of stars in each radial bin

 	Note: It can be numerically easier to fit for the expected number of counts in each bin, 
	rather than to fit for the surface density in each bin, which motivates the implementation here.
	"""

	log_rvals = np.log10(rvals)

	ll = 0

	if use_counts:
		for i in range(1,len(rvals)):
			n_data = binned_counts[i]
			n_model = Sigma_Zhao_bkg(log_nus_star,log_rs_star,alpha,beta,gamma,log_Sigma_bkg,log_rvals[i])*(np.pi*(upper_rvals[i]**2-lower_rvals[i]**2))

			if approx_scheme == "VarGauss1":
				sigma_plus = binned_errors_hi[i]
				sigma_minus = binned_errors_lo[i]
				
				sigma = 2*sigma_plus*sigma_minus/(sigma_plus+sigma_minus)
				sigma_prime = (sigma_plus-sigma_minus)/(sigma_plus+sigma_minus)

				ll += (n_data-n_model)**2/(sigma+sigma_prime*(n_model-n_data))**2

			elif approx_scheme == "VarGauss2":
				sigma_plus = binned_errors_hi[i]
				sigma_minus = binned_errors_lo[i]

				V = sigma_plus*sigma_minus
				Vprime = sigma_plus-sigma_minus

				ll += (n_data-n_model)**2/(V+Vprime*(n_model-n_data))

			else:
				print("Approximation scheme not implemented yet!")

	else:		
		for i in range(1,len(rvals)):
			Sigma_data = binned_counts[i]/(np.pi*(upper_rvals[i]**2-lower_rvals[i]**2))
			Sigma_model = Sigma_Zhao_bkg(log_nus_star,log_rs_star,alpha,beta,gamma,log_Sigma_bkg,log_rvals[i])

			if approx_scheme == "VarGauss1":
				sigma_plus = binned_errors_hi[i]/(np.pi*(upper_rvals[i]**2-lower_rvals[i]**2))
				sigma_minus = binned_errors_lo[i]/(np.pi*(upper_rvals[i]**2-lower_rvals[i]**2))
				
				sigma = 2*sigma_plus*sigma_minus/(sigma_plus+sigma_minus)
				sigma_prime = (sigma_plus-sigma_minus)/(sigma_plus+sigma_minus)

				ll += (Sigma_data-Sigma_model)**2/(sigma+sigma_prime*(Sigma_model-Sigma_data))**2

			elif approx_scheme == "VarGauss2":
				sigma_plus = binned_errors_hi[i]/(np.pi*(upper_rvals[i]**2-lower_rvals[i]**2))
				sigma_minus = binned_errors_lo[i]/(np.pi*(upper_rvals[i]**2-lower_rvals[i]**2))

				V = sigma_plus*sigma_minus
				Vprime = sigma_plus-sigma_minus

				ll += (Sigma_data-Sigma_model)**2/(V+Vprime*(Sigma_model-Sigma_data))

			else:
				print("Approximation scheme not implemented yet!")

	return -0.5*ll

def lnlike_Zhao_binned(log_nus_star,log_rs_star,alpha,beta,gamma,lower_rvals,upper_rvals,rvals,binned_counts,binned_errors_lo,binned_errors_hi,approx_scheme="VarGauss2",use_counts=False):
	"""
	The binned log likelihood in the Zhao parametrization, without a background term. 
	
	:param: log_nus_star: (log10 of) the normalization parameter
	:param: log_rs_star: (log10 of) the scale radius 
	:param: alpha, beta, gamma: slope parameters
	:param: lower_rvals: the lower bin edges of the radially binned data, used to calculate the
						 area of each radial bin
	:param: upper_rvals: the upper bin edges of the radially binned data, used to calculate the
						 area of each radial bin
	:param: rvals: the bin centers of the radially binned data
	:param: binned_counts: the number of stars in each radial bin
	:param: poisson: if True, use Poisson error (see function poiss_err at the end of this file);
					 if False, use sqrt(n) error to approximate the Poisson error

	Note: It can be numerically easier to fit for the expected number of counts in each bin, 
	rather than to fit for the surface density in each bin, which motivates the implementation here.
	"""
	log_rvals = np.log10(rvals)

	ll = 0

	if use_counts:
		for i in range(1,len(rvals)):
			n_data = binned_counts[i]
			n_model = Sigma_Zhao(log_nus_star,log_rs_star,alpha,beta,gamma,log_rvals[i])*(np.pi*(upper_rvals[i]**2-lower_rvals[i]**2))
			
			if approx_scheme == "VarGauss1":
				sigma_plus = binned_errors_hi[i]
				sigma_minus = binned_errors_lo[i]
				
				sigma = 2*sigma_plus*sigma_minus/(sigma_plus+sigma_minus)
				sigma_prime = (sigma_plus-sigma_minus)/(sigma_plus+sigma_minus)

				ll += (n_data-n_model)**2/(sigma+sigma_prime*(n_model-n_data))**2

			elif approx_scheme == "VarGauss2":
				sigma_plus = binned_errors_hi[i]
				sigma_minus = binned_errors_lo[i]

				V = sigma_plus*sigma_minus
				Vprime = sigma_plus-sigma_minus

				ll += (n_data-n_model)**2/(V+Vprime*(n_model-n_data))

			else:
				print("Approximation scheme not implemented yet!")

	else:
		for i in range(1,len(rvals)):
			Sigma_data = binned_counts[i]/(np.pi*(upper_rvals[i]**2-lower_rvals[i]**2))
			Sigma_model = Sigma_Zhao(log_nus_star,log_rs_star,alpha,beta,gamma,log_rvals[i])
			
			if approx_scheme == "VarGauss1":
				sigma_plus = binned_errors_hi[i]/(np.pi*(upper_rvals[i]**2-lower_rvals[i]**2))
				sigma_minus = binned_errors_lo[i]/(np.pi*(upper_rvals[i]**2-lower_rvals[i]**2))
				
				sigma = 2*sigma_plus*sigma_minus/(sigma_plus+sigma_minus)
				sigma_prime = (sigma_plus-sigma_minus)/(sigma_plus+sigma_minus)

				ll += (Sigma_data-Sigma_model)**2/(sigma+sigma_prime*(Sigma_model-Sigma_data))**2

			elif approx_scheme == "VarGauss2":
				sigma_plus = binned_errors_hi[i]/(np.pi*(upper_rvals[i]**2-lower_rvals[i]**2))
				sigma_minus = binned_errors_lo[i]/(np.pi*(upper_rvals[i]**2-lower_rvals[i]**2))

				V = sigma_plus*sigma_minus
				Vprime = sigma_plus-sigma_minus

				ll += (Sigma_data-Sigma_model)**2/(V+Vprime*(Sigma_model-Sigma_data))

			else:
				print("Approximation scheme not implemented yet!")

	return -0.5*ll

def poiss_err(n,ntrue,alpha=0.32):
    """
    Poisson error (variance) for n counts.
    An excellent review of the relevant statistics can be found in 
    the PDF statistics review: http://pdg.lbl.gov/2018/reviews/rpp2018-rev-statistics.pdf,
    specifically section 39.4.2.3

    :param: alpha corresponds to central confidence level 1-alpha, 
            i.e. alpha = 0.32 corresponds to 68% confidence
    """
    sigma_lo = chi2.ppf(alpha/2,2*n)/2+ntrue
    sigma_up = chi2.ppf(1-alpha/2,2*(n+1))/2+ntrue

    sigma = (2*sigma_lo*sigma_up)/(sigma_lo+sigma_up)
    sigma_prime = (sigma_up-sigma_lo)/(sigma_lo+sigma_up)

    return (sigma+sigma_prime*(n-ntrue))**2
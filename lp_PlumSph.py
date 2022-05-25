###############################################################
# Cleaned up version of rep_JIRead.py 12-03-2018              #
# Written by Laura Chang, Princeton University, 11-14-2018    #
#                                                             #
# This file includes the functions used in reproducing the    #
# results of dwarf galaxy stellar kinematic fits and Jeans    #
# analyses presented in J. I. Read and P. Steger [1701.04833] #
#                                                             #
# Note: for now, the likelihoods are hard-coded for 3 Plummer #
# components in the light profile and 5 DM components in the  #
# DM density profile.                                         #
#                                                             #
###############################################################

import sys, os
import numpy as np
from scipy import integrate
from scipy.stats import chi2

def nuStar(M_ary,a_ary,r,N_p=3):
	"""
	The tracer surface density (i.e. light profile) of a given dwarf, 
	where N_p is the number of Plummer components (default =3). 
	This is related to the tracer projected (2d) density profile SigmaStar by inverse Abel transform.
	See details on parameters in eq.15 of [1701.04833].
	
	:param: M_ary: array of length N_p containing the characteristic mass scale of 
				   each Plummer component
	:param: a_ary: array of length N_p containing the characteristic length scale of 
				   each Plummer component
	:param: r: the (3d) radius
	:param: N_p: the number of Plummer components
	"""
	assert len(M_ary) == N_p, "M_ary does not have the correct dimensions!"
	assert len(a_ary) == N_p, "a_ary does not have the correct dimensions!"
	
	nu = 0
	for j in range(N_p):
		nu += 3*M_ary[j]/(4*np.pi*a_ary[j]**3)*(1+(r/a_ary[j])**2)**-2.5

	return nu

def SigmaStar(M_ary,a_ary,R,N_p=3):
	"""
	The tracer surface density (i.e. light profile) of a given dwarf, 
	where N_p is the number of Plummer components (default =3). 
	This is related to the tracer (3d) density profile nuStar by Abel transform.
	See details on parameters in eq.16 of [1701.04833] **THIS EQUATION IS WRONG IN THE V1 PAPER.
	
	:param: M_ary: array of length N_p containing the characteristic mass scale of 
				   each Plummer component
	:param: a_ary: array of length N_p containing the characteristic length scale of 
				   each Plummer component
	:param: N_p: the number of Plummer components
	:param: R: the projected radius
	"""
	assert len(M_ary) == N_p, "M_ary does not have the correct dimensions!"
	assert len(a_ary) == N_p, "a_ary does not have the correct dimensions!"
	
	Sigma = 0
	for j in range(N_p):
		Sigma += M_ary[j]*a_ary[j]**2/(np.pi*(a_ary[j]**2+R**2)**2)
	
	return Sigma

def lnlike_Plumsph_1comp(logM0,loga0,lower_rvals,upper_rvals,rvals,binned_counts,binned_errors_lo,binned_errors_hi,approx_scheme="VarGauss2",use_counts=False):

	"""
	:param: approx_scheme: Approximation scheme for incorporating asymmetric Poisson errors, taken from arXiv:physics/0406120v1.
						   Current options include "VarGauss1" and "VarGauss2", which correspond to Sec. 3.5 and 3.6 from the reference,
						   respectively.
	"""
	M_ary = np.array([10**logM0])
	a_ary = np.array([10**loga0])

	ll = 0

	if use_counts:
		for i in range(1,len(rvals)):
			Sigma_model = SigmaStar(M_ary,a_ary,rvals[i],N_p=1)
			n_model = np.pi*(upper_rvals[i]**2-lower_rvals[i]**2)*Sigma_model
			if binned_errors[i] != 0:
				variance = binned_errors[i]**2
				ll += (binned_counts[i]-n_model)**2/variance
	else:
		for i in range(1,len(rvals)):
			Sigma_data = binned_counts[i]/(np.pi*(upper_rvals[i]**2-lower_rvals[i]**2))
			Sigma_model = SigmaStar(M_ary,a_ary,rvals[i],N_p=1)

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

def lnlike_Plumsph_2comp_rel(logM0,logM1,loga0,logc1,lower_rvals,upper_rvals,rvals,binned_counts,binned_errors_lo,binned_errors_hi,approx_scheme="VarGauss2",use_counts=False):

	loga1 = logc1+loga0

	M_ary = np.array([10**logM0,10**logM1])
	a_ary = np.array([10**loga0,10**loga1])

	ll = 0

	if use_counts:
		for i in range(1,len(rvals)):
			Sigma_model = SigmaStar(M_ary,a_ary,rvals[i],N_p=2)
			n_model = np.pi*(upper_rvals[i]**2-lower_rvals[i]**2)*Sigma_model
			if binned_errors[i] != 0:
				variance = binned_errors[i]**2
				ll += (binned_counts[i]-n_model)**2/variance
	else:
		for i in range(1,len(rvals)):
			Sigma_data = binned_counts[i]/(np.pi*(upper_rvals[i]**2-lower_rvals[i]**2))
			Sigma_model = SigmaStar(M_ary,a_ary,rvals[i],N_p=2)
			if approx_scheme == "VarGauss2":
				sigma_plus = binned_errors_hi[i]/(np.pi*(upper_rvals[i]**2-lower_rvals[i]**2))
				sigma_minus = binned_errors_lo[i]/(np.pi*(upper_rvals[i]**2-lower_rvals[i]**2))

				V = sigma_plus*sigma_minus
				Vprime = sigma_plus-sigma_minus

				ll += (Sigma_data-Sigma_model)**2/(V+Vprime*(Sigma_model-Sigma_data))
	return -0.5*ll

def lnlike_Plumsph_2comp_rel2(logM0,logcm1,loga0,logca1,lower_rvals,upper_rvals,rvals,binned_counts,binned_errors_lo,binned_errors_hi,approx_scheme="VarGauss2",use_counts=False):

	loga1 = logca1+loga0
	logM1 = logcm1+logM0

	M_ary = np.array([10**logM0,10**logM1])
	a_ary = np.array([10**loga0,10**loga1])

	ll = 0

	if use_counts:
		for i in range(1,len(rvals)):
			Sigma_model = SigmaStar(M_ary,a_ary,rvals[i],N_p=2)
			n_model = np.pi*(upper_rvals[i]**2-lower_rvals[i]**2)*Sigma_model
			if binned_errors[i] != 0:
				variance = binned_errors[i]**2
				ll += (binned_counts[i]-n_model)**2/variance
	else:
		for i in range(1,len(rvals)):
			Sigma_data = binned_counts[i]/(np.pi*(upper_rvals[i]**2-lower_rvals[i]**2))
			Sigma_model = SigmaStar(M_ary,a_ary,rvals[i],N_p=2)
			if approx_scheme == "VarGauss2":
				sigma_plus = binned_errors_hi[i]/(np.pi*(upper_rvals[i]**2-lower_rvals[i]**2))
				sigma_minus = binned_errors_lo[i]/(np.pi*(upper_rvals[i]**2-lower_rvals[i]**2))

				V = sigma_plus*sigma_minus
				Vprime = sigma_plus-sigma_minus

				ll += (Sigma_data-Sigma_model)**2/(V+Vprime*(Sigma_model-Sigma_data))
	return -0.5*ll

# def lnlike_Plumsph_3comp_rel(logM0,logM1,logM2,loga0,logc1,logc2,lower_rvals,upper_rvals,rvals,binned_counts,binned_errors,use_counts=False):

# 	loga1 = logc1+loga0
# 	loga2 = logc2+loga1

# 	M_ary = np.array([10**logM0,10**logM1,10**logM2])
# 	a_ary = np.array([10**loga0,10**loga1,10**loga2])

# 	ll = 0
# 	if use_counts:
# 		for i in range(1,len(rvals)):
# 			Sigma_model = SigmaStar(M_ary,a_ary,rvals[i])
# 			n_model = np.pi*(upper_rvals[i]**2-lower_rvals[i]**2)*Sigma_model
# 			if binned_errors[i] != 0:
# 				variance = binned_errors[i]**2
# 				ll += (binned_counts[i]-n_model)**2/variance
# 	else:
# 		for i in range(1,len(rvals)):
# 			Sigma_data = binned_counts[i]/(np.pi*(upper_rvals[i]**2-lower_rvals[i]**2))
# 			Sigma_model = SigmaStar(M_ary,a_ary,rvals[i])
# 			if binned_errors[i] != 0:
# 				variance = binned_errors[i]**2/(np.pi*(upper_rvals[i]**2-lower_rvals[i]**2))
# 				ll += (Sigma_data-Sigma_model)**2/variance
# 	return -0.5*ll

# def lnlike_Plumsph_3comp_rel2(logM0,logcm1,logcm2,loga0,logca1,logca2,lower_rvals,upper_rvals,rvals,binned_counts,binned_errors,use_counts=False):

# 	loga1 = logca1+loga0
# 	loga2 = logca2+loga1

# 	logM1 = logcm1+logM0
# 	logM2 = logcm2+logM1

# 	M_ary = np.array([10**logM0,10**logM1,10**logM2])
# 	a_ary = np.array([10**loga0,10**loga1,10**loga2])

# 	ll = 0
# 	if use_counts:
# 		for i in range(1,len(rvals)):
# 			Sigma_model = SigmaStar(M_ary,a_ary,rvals[i])
# 			n_model = np.pi*(upper_rvals[i]**2-lower_rvals[i]**2)*Sigma_model
# 			if binned_errors[i] != 0:
# 				variance = binned_errors[i]**2
# 				ll += (binned_counts[i]-n_model)**2/variance
# 	else:
# 		for i in range(1,len(rvals)):
# 			Sigma_data = binned_counts[i]/(np.pi*(upper_rvals[i]**2-lower_rvals[i]**2))
# 			Sigma_model = SigmaStar(M_ary,a_ary,rvals[i])
# 			if binned_errors[i] != 0:
# 				variance = binned_errors[i]**2/(np.pi*(upper_rvals[i]**2-lower_rvals[i]**2))
# 				ll += (Sigma_data-Sigma_model)**2/variance
# 	return -0.5*ll

# def rhoDM(rho0,rbins,slopes,r,nDM=5):

# 	"""
# 	The dark matter density profile, defined as a broken power law in nDM (default =5) number of 
# 	radial bins.
# 	See details on in eq.18 of [1701.04833] **THIS EQUATION IS WRONG IN THE V1 PAPER.
	
# 	:param: rho0: normalization factor
# 	:param: rbins: array of length nDM containing the radial bin edges
# 	:param: slopes: array of length nDM containing the power law slopes
# 	:param: r: the (3d) radius
# 	"""	

# 	if r < rbins[0]:
# 		return rho0*(r/rbins[0])**-slopes[0]
# 	elif r >= rbins[-1]: # Since the paper doesn't specify the profile beyond the highest radial bin edge, set a hard cutoff there for now

# 		# prefac = rho0*np.prod([(rbins[n]/rbins[n-1])**-slopes[n] for n in range(1,nDM-1)])/rbins[nDM-2]**-slopes[nDM-1]
# 		# return prefac*r**-slopes[nDM-1]
# 		return 0
# 	else: 
# 		ibin = np.argwhere(rbins<=r).max()
# 		prefac = rho0*np.prod([(rbins[n]/rbins[n-1])**-slopes[n] for n in range(1,ibin+1)])/rbins[ibin]**-slopes[ibin+1]
# 		return prefac*r**-slopes[ibin+1]

# def massDM(rho0,rbins,slopes,R,nDM=5):

# 	"""
# 	The enclosed dark matter mass at radius R, which is obtained analytically by integrating rhoDM.
	
# 	:param: rho0: normalization factor
# 	:param: rbins: array of length nDM containing the radial bin edges
# 	:param: slopes: array of length nDM containing the power law slopes
# 	:param: R: the (3d) radius
# 	"""	

# 	if R < rbins[0]:
# 		return (4*np.pi*rho0/rbins[0]**-slopes[0])*R**(3-slopes[0])/(3-slopes[0])

# 	elif R >= rbins[-1]: # Since the paper doesn't specify the profile beyond the highest radial bin edge, set a hard cutoff there for now
# 		m = (4*np.pi*rho0/rbins[0]**-slopes[0])*R**(3-slopes[0])/(3-slopes[0])
# 		for j in range(nDM-1):
# 			prefac = 4*np.pi*rho0*np.prod([(rbins[n]/rbins[n-1])**-slopes[n] for n in range(1,j+1)])/rbins[j]**-slopes[j+1]
# 			m += prefac*(rbins[j+1]**(3-slopes[j+1])-rbins[j]**(3-slopes[j+1]))/(3-slopes[j+1])
# 		return m

# 	else:
# 		m = (4*np.pi*rho0/rbins[0]**-slopes[0])*R**(3-slopes[0])/(3-slopes[0])
# 		ibin = np.argwhere(rbins<=R).max()
# 		for j in range(ibin):
# 			prefac = 4*np.pi*rho0*np.prod([(rbins[n]/rbins[n-1])**-slopes[n] for n in range(1,j+1)])/rbins[j]**-slopes[j+1]
# 			m += prefac*(rbins[j+1]**(3-slopes[j+1])-rbins[j]**(3-slopes[j+1]))/(3-slopes[j+1])
		
# 		prefac_in_bin = 4*np.pi*rho0*np.prod([(rbins[n]/rbins[n-1])**-slopes[n] for n in range(1,ibin+1)])/rbins[ibin]**-slopes[ibin+1]
# 		m += prefac_in_bin*(R**(3-slopes[ibin+1])-rbins[ibin]**(3-slopes[ibin+1]))/(3-slopes[ibin+1])

# 		return m

import sys, os
import numpy as np
from scipy import integrate

G = 4.302e-6 # in units of kpc/M_sun*(km/s)**2

def beta_baes(beta0,betainf,log_ra,eta,log_r):
	r = 10**log_r
	r_a = 10**log_ra

	r_s = r/r_a
	return (beta0+betainf*r_s**eta)/(1+r_s**eta)

def f_baes(beta0,betainf,log_ra,eta,log_r):
	r = 10**log_r
	r_a = 10**log_ra

	r_s = r/r_a
	return r**(2*beta0)*(1+r_s**eta)**(2*(betainf-beta0)/eta)

def beta_osipkov(log_ra,log_r):
	r = 10**log_r
	r_a = 10**log_ra
	return r**2/(r**2+r_a**2)

def f_osipkov(log_ra,log_r):
	r = 10**log_r
	r_a = 10**log_ra
	return (r**2+r_a**2)/r_a**2

def nuv2_baes(theta_star,theta_beta,theta_DM,log_r):
	r = 10**log_r
	return G/f_baes(*theta_beta,log_r)*integrate.quad(lambda s: f_baes(*theta_beta,np.log10(s))*nu_star(*theta_star,np.log10(s))*mass_DM(*theta_DM,np.log10(s))/s**2,r,np.inf)[0]

def sigmap2_baes(theta_star,theta_beta,theta_DM,log_R):
	R = 10**log_R
	norm = 2/Sigma_star(*theta_star,log_R)
	return norm*integrate.quad(lambda r: (1-beta_baes(*theta_beta,np.log10(r))*(R/r)**2)*nuv2_baes(theta_star,theta_beta,theta_DM,np.log10(r))*r/np.sqrt(r**2-R**2),R,np.inf)[0]

def K_osipkov(u,u_a):
	prefac = (u**2+u_a**2)*(u_a**2+1/2)/(u*(u_a**2+1)**(3./2))
	return prefac*np.arctan2(np.sqrt(u**2-1),np.sqrt(u_a**2+1))-np.sqrt(1-u**-2)/(2*(u_a**2+1))

def sigmap2_osipkov(theta_star,theta_beta,theta_DM,log_R):
	R = 10**log_R
	r_a = 10**theta_beta[0]
	norm = 2*G/Sigma_star(*theta_star,log_R)
	# return norm
	return norm*integrate.quad(lambda r: K_osipkov(r/R,r_a/R)*nu_star(*theta_star,np.log10(r))*mass_DM(*theta_DM,np.log10(r))/r,R,np.inf)[0] 

def lnlike_osipkov(theta_star,theta_beta,theta_DM,vbar,log_rvals,vvals,dvvals):
	assert len(log_rvals) == len(vvals), "r data and v data have different lengths!"
	assert len(vvals) == len(dvvals), "v data and dv data have different lengths!"

	ll = 0
	for i in range(len(log_rvals)):
		sigmap2 = sigmap2_osipkov(theta_star,theta_beta,theta_DM,log_rvals[i])
		ll += np.log(2*np.pi*(sigmap2**2+dvvals[i]**2))+(vvals[i]-vbar)**2/(sigmap2**2+dvvals[i]**2)
	return -0.5*ll

def K_isotropic(u): 
	""" 
	Kernel methods, clumpy, Skip over 6.10 integral. Can be found here http://clumpy.gitlab.io/CLUMPY/physics_jeans.html, or in appendix of https://arxiv.org/pdf/astro-ph/0405491.pdf
	:u: r/R, where r is distance in 3d, and R is projected radius
	"""
	return np.sqrt(1. - 1./(u*u))
from scipy.integrate import quad, simps
from libc.math cimport log, sqrt, pi, exp, atan2, INFINITY
import numpy as np
cimport numpy as np
from scipy.special import hyp2f1
from scipy import integrate

cdef double twopi = 2*pi
cdef double sqrt2 = sqrt(2)
cdef double sqrt2pi = sqrt(twopi)
cdef double G = 4.302e-6 # in units of kpc/M_sun*(km/s)**2

DTYPE = np.float_ #[double]
# ctypedef np.float_ DTYPE_t

def Sigma_star_Plummer(double log_M0, double log_a0, double R):
	"""
	The tracer surface density (i.e. light profile) of a given dwarf, 
	where N_p is the number of Plummer components (default =1). 
	This is related to the tracer (3d) density profile nuStar by Abel transform.
	See details on parameters in eq.16 of [1701.04833] **THIS EQUATION IS WRONG IN THE V1 PAPER.
	
	:param: M_j: the characteristic mass scale of 
				   each Plummer component
	:param: a_j: the characteristic length scale of 
				   each Plummer component
	:param: R: the projected radius
	"""
	cdef double a_0 = exp(log_a0)
	cdef double Sigma = exp(log_M0)*a_0*a_0/(pi*(a_0*a_0+R*R)*(a_0*a_0+R*R))
	return Sigma

def Sigma_star_Plummer_2(double log_M0, double log_a0, double log_M1, double log_a1, double R):
	"""
	The tracer surface density (i.e. light profile) of a given dwarf, 
	where N_p is the number of Plummer components (default =1). 
	This is related to the tracer (3d) density profile nuStar by Abel transform.
	See details on parameters in eq.16 of [1701.04833] **THIS EQUATION IS WRONG IN THE V1 PAPER.
	
	:param: M_j: the characteristic mass scale of 
				   each Plummer component
	:param: a_0: the characteristic length scale of 
				   each Plummer component
	:param: R: the projected radius
	"""
	cdef double a_0 = exp(log_a0)
	cdef double a_1 = exp(log_a1)
	cdef double Sigma = exp(log_M0)*a_0*a_0/(pi*(a_0*a_0+R*R)*(a_0*a_0+R*R))+\
						exp(log_M1)*a_1*a_1/(pi*(a_1*a_1+R*R)*(a_1*a_1+R*R))
	return Sigma

def Sigma_star_Plummer_3(double log_M0, double log_a0, double log_M1, double log_a1, double log_M2, double log_a2, double R):
	"""
	The tracer surface density (i.e. light profile) of a given dwarf, 
	where N_p is the number of Plummer components (default =1). 
	This is related to the tracer (3d) density profile nuStar by Abel transform.
	See details on parameters in eq.16 of [1701.04833] **THIS EQUATION IS WRONG IN THE V1 PAPER.
	
	:param: M_j: the characteristic mass scale of 
				   each Plummer component
	:param: a_0: the characteristic length scale of 
				   each Plummer component
	:param: R: the projected radius
	"""
	cdef double a_0 = exp(log_a0)
	cdef double a_1 = exp(log_a1)
	cdef double a_2 = exp(log_a2)

	cdef double Sigma = exp(log_M0)*a_0*a_0/(pi*(a_0*a_0+R*R)*(a_0*a_0+R*R))+\
						exp(log_M1)*a_1*a_1/(pi*(a_1*a_1+R*R)*(a_1*a_1+R*R))+\
						exp(log_M2)*a_2*a_2/(pi*(a_2*a_2+R*R)*(a_2*a_2+R*R))
	return Sigma

def Sigma_star_Zhao(double log_nus_star, double log_rs_star, double alpha, double beta, double gamma, double R):
	"""
	The tracer surface density of a given dSph, in the Zhao parametrization. 
	This is related to the 3d stellar density profile nu_Zhao by inverse Abel transform.
	See, e.g. Eq(4)&(7) of [1504.02048].
	
	:param: log_nus_star: (log of) the normalization parameter
	:param: log_rs_star: (log of) the scale radius 
	:param: alpha, beta, gamma: slope parameters
	:param: log_R: (log10 of) the projected radius to evaluate at, measured from the center
					of the dSph
	"""
	return integrate.quad(lambda r: 2*nu_star_Zhao(log_nus_star,log_rs_star,alpha,beta,gamma,log(r))*r/sqrt(r*r-R*R),R,INFINITY)[0]

###### 3D 

def nu_star_Plummer(double log_M0, double log_a0, double r):
	"""
	The tracer surface density (i.e. light profile) of a given dwarf, 
	where N_p is the number of Plummer components (default =3). 
	This is related to the tracer projected (2d) density profile SigmaStar by inverse Abel transform.
	See details on parameters in eq.15 of [1701.04833].
	
	:param: M_j: the characteristic mass scale of 
				   each Plummer component
	:param: a_0: the characteristic length scale of 
				   each Plummer component
	:param: r: the (3d) radius
	:param: N_p: the number of Plummer components
	"""
	cdef double a_0 = exp(log_a0)
	cdef double nu = 3*exp(log_M0)/(4*pi*a_0*a_0*a_0)*(1+(r/a_0)*(r/a_0))**(-2.5)
	return nu

def nu_star_Plummer_2(double log_M0, double log_a0, double log_M1, double log_a1, double r):
	"""
	The tracer surface density (i.e. light profile) of a given dwarf, 
	where N_p is the number of Plummer components (default =3). 
	This is related to the tracer projected (2d) density profile SigmaStar by inverse Abel transform.
	See details on parameters in eq.15 of [1701.04833].
	
	:param: M_j: the characteristic mass scale of 
				   each Plummer component
	:param: a_0: the characteristic length scale of 
				   each Plummer component
	:param: r: the (3d) radius
	:param: N_p: the number of Plummer components
	"""
	cdef double a_0 = exp(log_a0)
	cdef double a_1 = exp(log_a1)
	cdef double nu = 3*exp(log_M0)/(4*pi*a_0*a_0*a_0)*(1+(r/a_0)*(r/a_0))**(-2.5)+\
						3*exp(log_M1)/(4*pi*a_1*a_1*a_1)*(1+(r/a_1)*(r/a_1))**(-2.5)
	return nu

def nu_star_Plummer_3(double log_M0, double log_a0, double log_M1, double log_a1, double log_M2, double log_a2, double r):
	"""
	The tracer surface density (i.e. light profile) of a given dwarf, 
	where N_p is the number of Plummer components (default =3). 
	This is related to the tracer projected (2d) density profile SigmaStar by inverse Abel transform.
	See details on parameters in eq.15 of [1701.04833].
	
	:param: M_j: the characteristic mass scale of 
				   each Plummer component
	:param: a_0: the characteristic length scale of 
				   each Plummer component
	:param: r: the (3d) radius
	:param: N_p: the number of Plummer components
	"""
	cdef double a_0 = exp(log_a0)
	cdef double a_1 = exp(log_a1)
	cdef double a_2 = exp(log_a2)
 
	cdef double nu = 3*exp(log_M0)/(4*pi*a_0*a_0*a_0)*(1+(r/a_0)*(r/a_0))**(-2.5)+\
						3*exp(log_M1)/(4*pi*a_1*a_1*a_1)*(1+(r/a_1)*(r/a_1))**(-2.5)+\
						3*exp(log_M2)/(4*pi*a_2*a_2*a_2)*(1+(r/a_2)*(r/a_2))**(-2.5)
	return nu

def nu_star_Zhao(log_nus_star,log_rs_star,alpha,beta,gamma,log_r):
	"""
	The tracer (3d) density of a given dSph, in the Zhao parametrization. 
	This quantity cannot be physically measured, and is related to the 
	measured projected quantity Sigma_Zhao by Abel transform.
	See, e.g. Eq(7) of [1504.02048].
	
	:param: log_nus_star: (log of) the normalization parameter
	:param: log_rs_star: (log of) the scale radius 
	:param: alpha, beta, gamma: slope parameters
	:param: log_r: (log of) the (3d) radius to evaluate at
	"""
	cdef double r = exp(log_r)
	cdef double nus_star = exp(log_nus_star)
	cdef double rs_star = exp(log_rs_star)
	cdef double rs = r/rs_star

	return nus_star/((rs**gamma)*((1+rs**alpha)**((beta-gamma)/alpha)))

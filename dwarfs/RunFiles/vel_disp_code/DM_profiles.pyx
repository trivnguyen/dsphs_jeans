from libc.math cimport log, sqrt, pi, exp, atan2
import numpy as np
cimport numpy as np
from scipy.special import hyp2f1, betainc
cimport scipy.special.cython_special as sp
from scipy.integrate import dblquad, quad, simps

cdef double twopi = 2*pi
cdef double sqrt2 = sqrt(2)
cdef double sqrt2pi = sqrt(twopi)
cdef double G = 4.302e-6 # in units of kpc/M_sun*(km/s)**2

DTYPE = np.float_ #[double]
# ctypedef np.float_ DTYPE_t

###################################################################################################################################################################

def rho_DM(double log_rho0, double log_r0, double gamma0, double gamma1, double r):
    """
    Simplified enclosed mass, with one break in a power law, so simpler version of Read's.
    :rho0: normalization
    :r0: location of the break
    :gamma0: power for small r
    :gamma1: power for large r
    :r: value at which everything is evaluated
    """
    cdef double rho0 = exp(log_rho0)
    cdef double r0 = exp(log_r0)
    # cdef double r_cube = r*r*r
    # cdef double r0_cube = r0*r0*r0

    if r < r0:
        return rho0*(r/r0)**(-gamma0)
    else:
        return rho0*(r/r0)**(-gamma1)

def rho_DM_2(double log_rho0, double log_r0, double log_r1, double gamma0, double gamma1, double gamma2, double r):
    cdef double rho0 = exp(log_rho0)
    cdef double r0 = exp(log_r0)
    cdef double r1 = exp(log_r1)

    if r < r1:
        return rho_DM(log_rho0,log_r0,gamma0,gamma1,r)
    else:
        return rho0*(r1/r0)**(-gamma1)*(r/r1)**(-gamma2)

def rho_DM_3(double log_rho0, double log_r0, double log_r1, double log_r2, double gamma0, double gamma1, double gamma2, double gamma3, double r):
    cdef double rho0 = exp(log_rho0)
    cdef double r0 = exp(log_r0)
    cdef double r1 = exp(log_r1)
    cdef double r2 = exp(log_r2)

    if r < r2:
        return rho_DM_2(log_rho0,log_r0,log_r1,gamma0,gamma1,gamma2,r)
    else:
        return rho0*(r1/r0)**(-gamma1)*(r2/r1)**(-gamma2)*(r/r2)**(-gamma3)

def rho_DM_4(double log_rho0, double log_r0, double log_r1, double log_r2, double log_r3, double gamma0, double gamma1, double gamma2, double gamma3, double gamma4, double r):
    cdef double rho0 = exp(log_rho0)
    cdef double r0 = exp(log_r0)
    cdef double r1 = exp(log_r1)
    cdef double r2 = exp(log_r2)
    cdef double r3 = exp(log_r3)

    if r < r3:
        return rho_DM_3(log_rho0,log_r0,log_r1,log_r2,gamma0,gamma1,gamma2,gamma3,r)
    else:
        return rho0*(r1/r0)**(-gamma1)*(r2/r1)**(-gamma2)*(r3/r2)**(-gamma3)*(r/r3)**(-gamma4)

def rho_DM_5(double log_rho0, double log_r0, double log_r1, double log_r2, double log_r3,  double log_r4, double gamma0, double gamma1, double gamma2, double gamma3, double gamma4, double gamma5, double r):
    cdef double rho0 = exp(log_rho0)
    cdef double r0 = exp(log_r0)
    cdef double r1 = exp(log_r1)
    cdef double r2 = exp(log_r2)
    cdef double r3 = exp(log_r3)
    cdef double r4 = exp(log_r4)

    if r < r4:
        return rho_DM_4(log_rho0,log_r0,log_r1,log_r2,log_r3,gamma0,gamma1,gamma2,gamma3,gamma4,r)
    else:
        return rho0*(r1/r0)**(-gamma1)*(r2/r1)**(-gamma2)*(r3/r2)**(-gamma3)*(r4/r3)**(-gamma4)*(r/r4)**(-gamma5)

def rho_DM_NFW(double log_rho0, double log_r0, double r):
    cdef double rho0 = exp(log_rho0)
    cdef double r0 = exp(log_r0)
    cdef double rc = r/r0

    return rho0/(rc*(1+rc)*(1+rc))

def rho_DM_Burkert(double log_rho0, double log_r0, double r):
    cdef double rho0 = exp(log_rho0)
    cdef double r0 = exp(log_r0)
    cdef double rc = r/r0

    return rho0/((1+rc)*(1+rc*rc))

def rho_DM_Zhao(double log_rho0, double log_r0, double alpha, double beta, double gamma, double r):
    cdef double rho0 = exp(log_rho0)
    cdef double r0 = exp(log_r0)
    cdef double rc = r/r0

    return rho0/((rc**gamma)*(1+rc**alpha)**((beta-gamma)/alpha))

def rho_DM_GNFW(double log_rho0, double log_r0, double gamma, double r):
    cdef double rho0 = exp(log_rho0)
    cdef double r0 = exp(log_r0)
    cdef double rc = r/r0

    return rho0/(rc**gamma*(1+rc)**(3-gamma))

def rho_DM_GNFW_truncated(double log_rho0, double log_r0, double gamma, double log_rt, double r):
    """
    Generalized NFW, truncated at radius rt, with the sharpness of the truncation set by n. 
    Currently only implemented for n=1.
    """
    cdef double rho0 = exp(log_rho0)
    cdef double r0 = exp(log_r0)
    cdef double rt = exp(log_rt)
    cdef double rc = r/r0

    return rho0/(rc**gamma*(1+rc)**(3-gamma)*( 1 + (r/rt)**2))

def rho_DM_NFW_cored(double log_rho0, double log_r0, double r):
    """
    Cored NFW profile. Take gNFW and gamma = 0
    """
    cdef double rho0 = exp(log_rho0)
    cdef double r0 = exp(log_r0)


    return rho0 * (1 + (r/r0))**(-3)

###################################################################################################################################################################

def mass_DM(double log_rho0, double log_r0, double gamma0, double gamma1, double r):
    """
    Simplified enclosed mass, with one break in a power law, so simpler version of Read's.
    :rho0: normalization
    :r0: location of the break
    :gamma0: power for small r
    :gamma1: power for large r
    :r: value at which everything is evaluated
    """
    cdef double rho0 = exp(log_rho0)
    cdef double r0 = exp(log_r0)
    cdef double r_cube = r*r*r
    cdef double r0_cube = r0*r0*r0

    if r < r0:
        return (2*twopi*rho0/(3-gamma0))*r_cube*(r/r0)**(-gamma0)
    else:
        return (2*twopi*rho0)*(r0_cube/(3-gamma0)+(r_cube*(r/r0)**(-gamma1)-r0_cube)/(3-gamma1))


def mass_DM_2(double log_rho0, double log_r0, double log_r1, double gamma0, double gamma1, double gamma2, double r):
    """
    Simplified enclosed mass, with 2 breaks in a power law, so simpler version of Read's.
    :rho0: normalization
    :r0, r1: locations of the break
    :gamma0: power for small r
    :gamma1: power for medium r
    :gamma2: power for large r
    :r: value at which everything is evaluated
    """
    cdef double rho0 = exp(log_rho0)
    cdef double r0 = exp(log_r0)
    cdef double r1 = exp(log_r1)
    cdef double r_cube = r*r*r
    cdef double r1_cube = r1*r1*r1

    if r < r1:
        return mass_DM(log_rho0, log_r0, gamma0, gamma1, r)
    else:
        new_integral = 2*twopi*rho0*(r1/r0)**(-gamma1)*(r_cube*(r/r1)**(-gamma2)-r1_cube)/(3-gamma2)
        return mass_DM(log_rho0, log_r0, gamma0, gamma1, r1) + new_integral


def mass_DM_3(double log_rho0, double log_r0, double log_r1, double log_r2, double gamma0, double gamma1, double gamma2, double gamma3, double r):
    """
    Simplified enclosed mass, with 3 breaks in a power law, so simpler version of Read's.
    :rho0: normalization
    :r0, r1, r2: locations of the break
    :gamma0: power for small r
    :gamma1: power for medium r
    :gamma2: power for large r
    :gamma3: power for larger r
    :r: value at which everything is evaluated
    """
    cdef double rho0 = exp(log_rho0)
    cdef double r0 = exp(log_r0)
    cdef double r1 = exp(log_r1)
    cdef double r2 = exp(log_r2)
    cdef double r_cube = r*r*r
    cdef double r2_cube = r2*r2*r2

    if r < r2:
        return mass_DM_2(log_rho0, log_r0, log_r1, gamma0, gamma1, gamma2, r)
    else:
        new_integral = 2*twopi*rho0*(r1/r0)**(-gamma1)*(r2/r1)**(-gamma2)*(r_cube*(r/r2)**(-gamma3)-r2_cube)/(3-gamma3)
        return mass_DM_2(log_rho0, log_r0, log_r1, gamma0, gamma1, gamma2, r2) + new_integral

def mass_DM_4(double log_rho0, double log_r0, double log_r1, double log_r2, double log_r3, double gamma0, double gamma1, double gamma2, double gamma3, double gamma4, double r):
    """
    Simplified enclosed mass, with 4 breaks in a power law.
    :rho0: normalization
    :r0, r1, r2, r3: locations of the break
    :gamma0: power for small r
    :gamma1: power for medium r
    :gamma2: power for large r
    :gamma3: power for larger r
    :gamma4: power for largest r
    :r: value at which everything is evaluated
    """
    cdef double rho0 = exp(log_rho0)
    cdef double r0 = exp(log_r0)
    cdef double r1 = exp(log_r1)
    cdef double r2 = exp(log_r2)
    cdef double r3 = exp(log_r3)
    cdef double r_cube = r*r*r
    cdef double r3_cube = r3*r3*r3

    if r < r3:
        return mass_DM_3(log_rho0, log_r0, log_r1, log_r2, gamma0, gamma1, gamma2, gamma3, r)
    else:
        new_integral = 2*twopi*rho0*(r1/r0)**(-gamma1)*(r2/r1)**(-gamma2)*(r3/r2)**(-gamma3)*(r_cube*(r/r3)**(-gamma4)-r3_cube)/(3-gamma4)
        return mass_DM_3(log_rho0, log_r0, log_r1, log_r2, gamma0, gamma1, gamma2, gamma3, r3) + new_integral

def mass_DM_5(double log_rho0, double log_r0, double log_r1, double log_r2, double log_r3, double log_r4, double gamma0, double gamma1, double gamma2, double gamma3, double gamma4, double gamma5, double r):
    """
    Simplified enclosed mass, with 4 breaks in a power law.
    :rho0: normalization
    :r0, r1, r2, r3, r4: locations of the break
    :gamma0: power for small r
    :gamma1: power for medium r
    :gamma2: power for large r
    :gamma3: power for larger r
    :gamma4: power for largest r
    :gamma5: power for largestest r
    :r: value at which everything is evaluated
    """
    cdef double rho0 = exp(log_rho0)
    cdef double r0 = exp(log_r0)
    cdef double r1 = exp(log_r1)
    cdef double r2 = exp(log_r2)
    cdef double r3 = exp(log_r3)
    cdef double r4 = exp(log_r4)
    cdef double r_cube = r*r*r
    cdef double r4_cube = r4*r4*r4

    if r < r4:
        return mass_DM_4(log_rho0, log_r0, log_r1, log_r2, log_r3, gamma0, gamma1, gamma2, gamma3, gamma4, r)
    else:
        new_integral = 2*twopi*rho0*(r1/r0)**(-gamma1)*(r2/r1)**(-gamma2)*(r3/r2)**(-gamma3)*(r4/r3)**(-gamma4)*(r_cube*(r/r4)**(-gamma5)-r4_cube)/(3-gamma5)
        return mass_DM_4(log_rho0, log_r0, log_r1, log_r2, log_r3, gamma0, gamma1, gamma2, gamma3, gamma4, r4) + new_integral

def mass_DM_NFW(double log_rho0, double log_r0, double r):
    """
    Enclosed mass NFW
    :rho0: normalization
    :r0: location of the break
    :r: value at which everything is evaluated
    """
    cdef double rho0 = exp(log_rho0)
    cdef double r0 = exp(log_r0)
    cdef double r0_cube = r0*r0*r0

    return 4*pi*rho0*r0_cube* ( log( (r0 + r)/r0 ) - r/(r0 + r))

def mass_DM_Burkert(double log_rho0, double log_r0, double r):
    """
    Enclosed mass Burkert
    :rho0: normalization
    :r0: location of the break
    :r: value at which everything is evaluated
    """
    cdef double rho0 = exp(log_rho0)
    cdef double r0 = exp(log_r0)
    cdef double r0_cube = r0*r0*r0

    return 4*pi*rho0*r0_cube* ( -0.5*atan2(r, r0) + 0.5*log( (r+r0)/r0 ) + 0.25*log( (r*r + r0*r0)/(r0*r0) )    )

def mass_DM_Zhao(double log_rho0, double log_r0, double alpha, double beta, double gamma, double r):
    """
    Enclosed mass Zhao
    :rho0: normalization
    :r0: location of the break
    :r: value at which everything is evaluated
    :alpha, beta, gamma: powers
    """
    cdef double rho0 = exp(log_rho0)
    cdef double r0 = exp(log_r0)
    cdef double r0_cube = r0*r0*r0
    cdef double r_over_r0 = r/r0

    return 4*pi*rho0*r**3 * ( (1/(3 - gamma)) * (r_over_r0)**(-gamma) * sp.hyp2f1((3 - gamma)/alpha, (beta - gamma)/alpha  , 1 + (3 - gamma)/alpha , - r_over_r0**alpha    ) )

# def mass_DM_GNFW(double log_rho0, double log_r0, double gamma, double r):
#   """
#   Enclosed mass Zhao
#   :rho0: normalization
#   :r0: location of the break
#   :r: value at which everything is evaluated
#   :alpha, beta, gamma: powers
#   """
#   alpha = 1
#   beta = 3
#   return mass_DM_Zhao( log_rho0,  log_r0,  alpha, beta, gamma, r)


def mass_DM_GNFW(double log_rho0, double log_r0, double gamma, double r):
    """
    Enclosed mass Zhao
    :rho0: normalization
    :r0: location of the break
    :r: value at which everything is evaluated
    :alpha, beta, gamma: powers
    """
    cdef double rho0 = exp(log_rho0)
    cdef double r0 = exp(log_r0)
    cdef double r0_cube = r0*r0*r0
    cdef double r_over_r0 = r/r0
    # print("r0=",r0)

    return 4*pi*rho0*r**3 * ( (1/(3 - gamma)) * (r_over_r0)**(-gamma) * sp.hyp2f1((3 - gamma), (3 - gamma)  , 1 + (3 - gamma) , - r_over_r0    ) )

def mass_DM_GNFW_truncated(double log_rho0, double log_r0, double gamma, double log_rt, double r):
    """
    Enclosed mass GNFW, with truncation 
    Currently implemented only for n=1
    :rho0: normalization
    :r0: location of the break
    :r: value at which everything is evaluated
    :alpha, beta, gamma: powers
    """
    cdef double rho0 = exp(log_rho0)
    cdef double r0 = exp(log_r0)
    cdef double rt = exp(log_rt)
    cdef double overall_factor, hypergeometric_function 

    overall_factor =  (2*pi*r**3*rho0*r0**3) * (r/(r+r0))**(-gamma) /((3 - gamma )*(r + r0)**3 )
    hypergeometric_function = hyp2f1(1, 3-gamma, 4-gamma, r*complex(rt, -r0)/(rt*(r+r0)) ) + hyp2f1(1, 3-gamma, 4-gamma, r*complex(rt, r0)/(rt*(r+r0)))

    return overall_factor*hypergeometric_function

def mass_DM_NFW_cored(double log_rho0, double log_r0, double r):
    """
    Enclosed Mass of Cored NFW
    """
    cdef double rho0 = exp(log_rho0)
    cdef double r0 = exp(log_r0)

    return 4*pi*rho0*r0**3 * ( - r*(3*r + 2*r0)/(2*(r+r0)**2) + log(1 + (r/r0)) )

###################################################################################################################################################################


def J_factor_NFW(double distance, double cos_theta_max, double log_rho0, double log_r0):
    """
    Computes the J factor of the NFW profile
    :param: distance, float, distance of the center of the dwarf galaxy
    :param: cos_theta_max, float, cos max angle to integrate to
    :param: log_rho0, float, normalization of the NFW
    :param: log_r0, float, scale radius of NFW 
    """

    cdef double rho0 = exp(log_rho0)
    cdef double r0 = exp(log_r0)

    cdef double j_factor

    j_factor = dblquad(lambda s, x: 2*pi* rho_DM_NFW(log_rho0, log_r0, sqrt(distance*distance + s*s - 2*distance*s*x))**2, cos_theta_max, 1, 0, distance)[0]
    return j_factor

def J_factor_gNFW(double distance, double cos_theta_max, double log_rho0, double log_r0, double gamma):
    """
    Computes the J factor of the NFW profile
    :param: distance, float, distance of the center of the dwarf galaxy
    :param: cos_theta_max, float, cos max angle to integrate to
    :param: log_rho0, float, normalization of the NFW
    :param: log_r0, float, scale radius of NFW 
    :param: gamma, float, inner slope
    """

    cdef double rho0 = exp(log_rho0)
    cdef double r0 = exp(log_r0)

    cdef double j_factor

    j_factor = dblquad(lambda s, x: 2*pi* rho_DM_GNFW(log_rho0, log_r0, gamma, sqrt(distance*distance + s*s - 2*distance*s*x))**2, cos_theta_max, 1, 0, distance)[0]
    return j_factor

def J_factor_NFWc(double distance, double cos_theta_max, double log_rho0, double log_r0):
    """
    Computes the J factor of the NFW profile
    :param: distance, float, distance of the center of the dwarf galaxy
    :param: cos_theta_max, float, cos max angle to integrate to
    :param: log_rho0, float, normalization of the NFW
    :param: log_r0, float, scale radius of NFW 
    """

    cdef double rho0 = exp(log_rho0)
    cdef double r0 = exp(log_r0)

    cdef double j_factor

    j_factor = dblquad(lambda s, x: 2*pi* rho_DM_NFW_cored(log_rho0, log_r0, sqrt(distance*distance + s*s - 2*distance*s*x))**2, cos_theta_max, 1, 0, distance)[0]
    return j_factor


def J_factor_Zhao(double distance, double cos_theta_max, double log_rho0, double log_r0, double alpha, double beta, double gamma):
    """
    Computes the J factor of the NFW profile
    :param: distance, float, distance of the center of the dwarf galaxy
    :param: cos_theta_max, float, cos max angle to integrate to
    :param: log_rho0, float, normalization of the NFW
    :param: log_r0, float, scale radius of NFW 
    :param: alpha, beta, gamma, float, slopes of the Zhao profile
    """

    cdef double rho0 = exp(log_rho0)
    cdef double r0 = exp(log_r0)

    cdef double j_factor

    j_factor = dblquad(lambda s, x: 2*pi* rho_DM_Zhao(log_rho0, log_r0, alpha, beta, gamma, sqrt(distance*distance + s*s - 2*distance*s*x))**2, cos_theta_max, 1, 0, distance)[0]
    return j_factor


J_factor_NFW_vectorized = np.vectorize(J_factor_NFW)
J_factor_gNFW_vectorized = np.vectorize(J_factor_gNFW)
J_factor_NFWc_vectorized = np.vectorize(J_factor_NFWc)








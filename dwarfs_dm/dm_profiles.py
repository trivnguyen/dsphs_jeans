
import numpy as np
import scipy.integrate
import astropy.units as u
import astropy.constants as const

from . import light_profiles
from .model import Model


G = const.G.to_value(u.kpc**3 / u.Msun / u.s**2)
kpc_to_km = (u.kpc).to(u.km)

# DM models
def gNFW(r, logr_dm, gamma, logrho_0):
    ''' Generalized NFW profile '''
    rho_0 = 10**logrho_0
    r_dm = 10**logr_dm
    return rho_0 * (r/r_dm)**(-gamma) * (1  + r/r_dm)**(-3 + gamma)


def cumulative_mass(r, rho, axis=0):
    ''' Compute the enclosed mass cumulatively at each radius '''
    dr = r[1] - r[0]
    M = scipy.integrate.cumulative_trapezoid(
        4 * np.pi * r**2 * rho, axis=axis, dx=dr)
    M = np.append(M, M[-1])
    return M


def beta(r, r_a):
    ''' Compute the anisotropy velocity profile '''
    return r**2 / (r_a**2 + r**2)


# Integration function
def calc_g(r, beta):
    ''' Calculate the g(r) integral defined as:
    ```
        g(r) = exp( 2 \int beta(r) / r dr )
    ```
    where:
        beta(r) is the velocity anisotropy
        r is the 3d radius
    '''
    dr = r[1] - r[0]
    g = np.exp(scipy.integrate.cumulative_trapezoid(2 * beta / r, dx=dr))
    g = np.append(g, g[-1])
    return g


def calc_sigma2_nu(r, nu, dm_params, g=1):
    ''' Calculate the 3D Jeans integration:
    ```
    sigma2(r0) nu(r0) g(r_0) =  int_r0^\infty G M(r) nu(r) g(r) / r^2 dr
    ```
    where:
    - G is the gravitational constant
    - M(r) is the enclosed radius at radius r in Msun
    - nu(r) is the 3D light profile
    - g(r) is the anistropy integral

    Parameters:
    - r: (array of M float) the 3d radii
    - nu: (array of M float) the 3d light profile at each radius
    - dm_params: (array of 3 float) the gNFW parameters (r_dm, gamma, rho0)
        r_dm in unit of kpc, rho0 in unit of Msun / kpc^3

    Returns:
    - sigma2: (arrays of M floats) the 3D velocity dispersion in (km/s)^2
    '''
    rho = gNFW(r, *dm_params)
    M = cumulative_mass(r, rho)
    dr = r[1] - r[0]

    # integration
    inte = M * G * nu  * g / r**2
    sigma2_nu = scipy.integrate.cumulative_trapezoid(
        inte[::-1], dx=dr, initial=0)[::-1]
    sigma2_nu = sigma2_nu / g

    return sigma2_nu


def calc_sigma2p_Sigma(R, r, sigma2_nu, beta=None):
    ''' Calculate the projected Jeans integration:
    ```
    sigma2_p(R) Sigma(R) = 2 * int_R^\intfy (nu(r) sigma2(r) r) / sqrt(r^2 - R^2) dr
    ```
    where:
    - R is the projected radius
    - Sigma(R) is the 3D light profile

    Parameters:
    - R: (array of M float) the 2d projected radii
    - r: (array of N float) the 3d projected radii for integration
    - sigma2_nu: (array of N float) the 3d Jeans integration at each 3d projected radii
    Returns:
    - sigma2_p: (arrays of M floats) the 2D velocity dispersion

    '''
    dr = r[1] - r[0]

    if beta is not None:
        R = R[:, None]
        r = r[None, :]
        sigma2_nu = sigma2_nu[None, :]
        beta = beta[None, :]
        rminR2 = r**2 - R**2
        inte = (1 - beta * R**2 / r**2) * sigma2_nu * r
        inte = np.where(rminR2 > 0,  inte / np.sqrt(rminR2), 0)
    else:
        rminR2 = r[None, :]**2 - R[:, None]**2
        inte = (sigma2_nu * r)
        inte = np.where(rminR2 > 0, inte[None, :] / np.sqrt(rminR2), 0)

    sigma2p_Sigma = 2 * scipy.integrate.trapezoid(inte, axis=1, dx=dr)
    return sigma2p_Sigma


class JeansModel(Model):
    ''' Class for fitting DM density distribution with Jeans modeling '''
    def __init__(
        self, R, v, logL, logr_star, r_a=1e30, priors={},
        dr=0.001, v_err=0.0, r_min_factor=0.5, r_max_factor=2):
        '''
        Parameters:
        - R: (array of N float) the prosjected radii of N stars in kpc
        - v: (array of N float) the line-of-sight velocities of N stars in km/s
        - v_err: (array of N float) the velocity measurement error of N stars
        - logL: (float) Plummer luminosity in Lsun
        - logr_star: (float) Plummer scale radius in kpc
        - priors: (dict) dictionary with prior range
        - dr: (float) the radius integration resolution
        - r_min_factor: (float) factor to convert the min projected radius R to the min 3D radius
        - r_max_factor: (float) factor to convert the max projected radius R to the max 3D radius
        '''
        super().__init__(params_list=('logr_dm', 'gamma', 'logrho_0'))

        self.R = R
        self.v = v
        self.v_err = v_err
        self.star_params = (logL, logr_star)
        self.r_a = r_a
        self.dr = dr
        self.priors = priors
        self.r_min_factor = r_min_factor
        self.r_max_factor = r_max_factor
        self.r_min = np.min(R) * r_min_factor
        self.r_max = np.max(R) * r_max_factor
        self.r = np.arange(self.r_min, self.r_max + dr, dr)

        self.priors = {
            'logr_dm': [-1, 0.7],
            'gamma': [-1, 5],
            'logrho_0': [5, 8],
        }
        self.priors.update(priors)

        # calculate the velocity anistropy parameter
        self.beta = beta(self.r, r_a)
        self.g = calc_g(self.r, self.beta)

        # calculate the 3D and 2D light profile
        self.nu = 10**light_profiles.log10_plummer3d(self.r, self.star_params)
        self.Sigma = 10**light_profiles.log10_plummer2d(self.R, self.star_params)

        # calculate the squared measurement error and velocity square error
        self.v_var = v_err**2
        self.v_square_err = (v - np.mean(v))**2

    def log_likelihood(self, x):
        ''' The log likelihood given a set of DM parameters.
        For each star the log likelihood is defined as:
        ```
        logL = -0.5 * (v - v_mean)^2 / (sigma2_p + v_err^2) - 0.5 * log(2 pi  * (sigma2_p + verr^2))
        ``
        where:
        - v is the velocity of the star
        - v_mean is the mean velocity of all stars
        - v_err is the measurement error
        - sigma2_p is the velocity dispersion

        Parameters:
        - x: (array of 3 float) the gNFW parameters (r_dm, gamma, rho0)
            r_dm in unit of kpc, rho0 in unit of Msun / kpc^3

        Returns:
        - The log likelihood

        '''
        # calculate the projected 2d velocity dispersion
        sigma2_nu = calc_sigma2_nu(self.r, self.nu, x, self.g)
        sigma2p_Sigma = calc_sigma2p_Sigma(self.R, self.r, sigma2_nu, self.beta)
        sigma2p = sigma2p_Sigma / self.Sigma * kpc_to_km**2

        # calculate the log likelihood
        var = sigma2p + self.v_var
        logL = -0.5 * self.v_square_err / var
        logL = logL - 0.5 * np.log(2 * np.pi * var)
        logL = np.sum(logL)

        return logL


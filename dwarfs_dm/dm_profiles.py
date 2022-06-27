
import numpy as np
import scipy.integrate
import astropy.units as u
import astropy.constants as const
import bilby
import logging

logger = logging.getLogger(__name__)

from . import model
from . import light_profiles

G = const.G.to_value(u.kpc**3 / u.Msun / u.s**2)
kpc_to_km = (u.kpc).to(u.km)

# DM models
def gNFW(r, r_dm, gamma, rho_0):
    ''' Generalized NFW profile '''
    return rho_0 * (r/r_dm)**(-gamma) * (1  + r/r_dm)**(-3 + gamma)


def cumulative_mass(r, rho, axis=0):
    ''' Compute the enclosed mass cumulatively at each radius '''
    dr = r[1] - r[0]
    M = scipy.integrate.cumulative_trapezoid(
        4 * np.pi * r**2 * rho, axis=axis, dx=dr)
    M = np.append(M, M[-1])
    return M


def calc_beta(r, r_a):
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


def calc_sigma2_nu(r, nu, r_dm, gamma, rho_0, g=1):
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
    rho = gNFW(r, r_dm, gamma, rho_0)
    M = cumulative_mass(r, rho)
    dr = r[1] - r[0]

    # integration
    inte = M * G * nu  * g / r**2
    sigma2_nu = scipy.integrate.cumulative_trapezoid(
        inte[::-1], dx=dr, initial=0)[::-1]
    sigma2_nu = sigma2_nu / g

    return sigma2_nu


def calc_sigma2p_Sigma(R, r, sigma2_nu, beta):
    ''' Calculate the projected Jeans integration:
    ```
    sigma2_p(R) Sigma(R) = 2 * int_R^\intfy (1 - beta * R^2 /r^2) (nu(r) sigma2(r) r) / sqrt(r^2 - R^2) dr
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
    R = R[:, None]
    r = r[None, :]
    sigma2_nu = sigma2_nu[None, :]
    rminR2 = r**2 - R**2
    beta = beta[None, :]

    with np.errstate(divide='ignore', invalid='ignore'):
        inte = (1 - beta * R**2 / r**2) * sigma2_nu * r
        inte = np.where(rminR2 > 0,  inte / np.sqrt(rminR2), 0)

    sigma2p_Sigma = 2 * scipy.integrate.trapezoid(inte, axis=1, dx=dr)
    return sigma2p_Sigma


class JeansModel(model.Model):
    ''' Class for fitting DM density distribution with Jeans modeling '''
    def __init__(
        self, R, v, priors={}, dr=0.001, v_err=0.0,
        r_min_factor=0.5, r_max_factor=2, fit_v_mean=False):
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
        super().__init__(parameters={
            "r_dm": None, "gamma": None, "rho_0": None,
            "L": None, "r_star": None, "v_mean": None})

        self.R = R
        self.v = v
        self.v_err = v_err
        self.v_var = v_err**2
        self.dr = dr
        self.priors = priors
        self.r_min_factor = r_min_factor
        self.r_max_factor = r_max_factor
        self.r_min = np.min(R) * r_min_factor
        self.r_max = np.max(R) * r_max_factor
        self.r = np.arange(self.r_min, self.r_max + dr, dr)

        self.priors = {
            "r_dm": bilby.core.prior.LogUniform(0.1, 5, "r_dm"),
            "gamma": bilby.core.prior.Uniform(-1, 2, "gamma"),
            "rho_0": bilby.core.prior.LogUniform(1e5, 1e8, "rho_0"),
            "L": bilby.core.prior.LogUniform(1e-2, 1e5, "L"),
            "r_star": bilby.core.prior.LogUniform(1e-3, 1e3, "r_star"),
            "v_mean": bilby.core.prior.Uniform(-100, 100, "v_mean"),
            "r_a_fact": bilby.core.prior.Uniform(0.5, 2, "r_a_fact")
        }
        if not fit_v_mean:
            self.priors["v_mean"] = bilby.core.prior.DeltaFunction(np.mean(v), "v_mean")
        self.priors.update(priors)
        if self.priors.get("r_a") is not None:
            self.priors.pop("r_a_fact")

        self.__preset_vars()

    def __preset_vars(self):
        ''' Preset some quantities: nu, Sigma, v_rms
        if their priors are fixed to speed up calculation
        '''
        self.nu = None
        self.Sigma = None
        self.v_rms = None
        self.beta = None
        self.g = None

        # calculate the 3D and 2D light profile
        is_r_star_delta = isinstance(
            self.priors['r_star'], (float, bilby.core.prior.DeltaFunction))
        is_L_delta = isinstance(
            self.priors['L'], (float, bilby.core.prior.DeltaFunction))
        if is_r_star_delta and is_L_delta:
            logger.info('Preset nu and Sigma')
            r_star = self.priors['r_star'].peak
            L = self.priors['L'].peak
            self.nu = 10**light_profiles.log10_plummer3d(self.r, L, r_star)
            self.Sigma = 10**light_profiles.log10_plummer2d(self.R, L, r_star)

        # calculate the RMS
        is_v_mean_delta = isinstance(
            self.priors['v_mean'], (float, bilby.core.prior.DeltaFunction))
        if is_v_mean_delta:
            logger.info('Preset RMS')
            v_mean = self.priors['v_mean'].peak
            self.v_rms = (self.v - v_mean)**2

        # calculate anisotropy parameters
        is_r_a_delta = isinstance(
            self.priors.get('r_a'), (float, bilby.core.prior.DeltaFunction))
        if is_r_a_delta:
            r_a = self.priors['r_a'].peak
            beta = calc_beta(self.r, r_a)
            g = calc_g(self.r, beta)

    def log_likelihood(self):
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
        r_dm = self.parameters['r_dm']
        gamma = self.parameters['gamma']
        rho_0 = self.parameters['rho_0']
        L = self.parameters['L']
        r_star = self.parameters['r_star']
        v_mean = self.parameters['v_mean']

        if self.parameters.get('r_a_fact') is not None:
            r_a_fact = self.parameters['r_a_fact']
            r_a = r_star * r_a_fact
        else:
            r_a = self.parameters['r_a']

        # calculate the 3D and 2D light profile
        if self.nu is None and self.Sigma is None:
            nu = 10**light_profiles.log10_plummer3d(self.r, L, r_star)
            Sigma = 10**light_profiles.log10_plummer2d(self.R, L, r_star)
        else:
            nu = self.nu
            Sigma = self.Sigma

        # calculate anisotropy parameters
        if self.beta is not None:
            beta = self.beta
            g = self.g
        else:
            beta = calc_beta(self.r, r_a)
            g = calc_g(self.r, beta)

        # calculate the projected 2d velocity dispersion
        sigma2_nu = calc_sigma2_nu(self.r, nu, r_dm, gamma, rho_0, g)
        sigma2p_Sigma = calc_sigma2p_Sigma(self.R, self.r, sigma2_nu, beta)
        sigma2p = sigma2p_Sigma / Sigma * kpc_to_km**2

        # calculate the log likelihood
        if self.v_rms is None:
            v_rms = (self.v - v_mean)**2
        else:
            v_rms = self.v_rms
        var = sigma2p + self.v_var
        logL = -0.5 * v_rms / var
        logL = logL - 0.5 * np.log(2 * np.pi * var)
        logL = np.sum(logL)

        return logL


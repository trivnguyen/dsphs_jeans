from units import *
from constants import *
import healpy as hp
import numpy as np
from scipy import integrate, interpolate

r_s_NFW = 17.356636256189983*h # ~NFW scale radius of Milky Way in kpc/h
gamma_NFW = 1 # for NFW
rho0 = 0.4 # local density at the sun in GeV/cm^3

def r_galactocentric(l, psi_deg):
    """ Distance to galactic center given distance l and angle psi_deg from us
    """
    return np.sqrt(r0**2. + l**2. - 2.*r0*l*np.cos(np.radians(psi_deg)))

def rho0_NFW(r_s=r_s_NFW,gamma=gamma_NFW):
    return rho0*(r0/r_s)**gamma*(1+r0/r_s)**(3-gamma)

def rho_NFW(r,r_s=r_s_NFW,gamma=gamma_NFW):
    return rho0_NFW(r_s,gamma)/((r/r_s)**gamma*(1+r/r_s)**(3-gamma))

def J_fac_integral_NFW(psi_deg, gamma=gamma_NFW):
    """ NFW line of sight integral of rho(r)**2
    """
    return integrate.quad(lambda l: rho_NFW(r_galactocentric(l, psi_deg))**2, 0., 100.*r_s_NFW)[0]

def make_NFW_Jfactor_map(gamma = gamma_NFW):
    """ For each pixel, get the line of sight integral of
        rho_Einasto**2 to get the J-factor map
    """

    nside = 128
    npix = hp.nside2npix(nside)
    Omegapix = hp.nside2pixarea(nside) # solid angle subtended by each pixel

    psi_deg = np.arange(0., 180.5, 0.5)
    integrand_NFW = np.vectorize(J_fac_integral_NFW)(psi_deg, gamma)
    Jfactor_NFW = interpolate.interp1d(psi_deg, integrand_NFW*Omegapix) # multiply by the solid angle subtended by each pixel to get J(delta Omega) 
    psi_deg_pixels = np.array([np.degrees(np.arccos(np.dot([1.0, 0.0, 0.0], hp.pix2vec(nside, pix)))) for pix in range(npix)]) # [1,0,0] corresponds to (theta,phi) = (90 deg, 0 deg), which corresponds to pointing from us towards the GC 
    return Jfactor_NFW(psi_deg_pixels)
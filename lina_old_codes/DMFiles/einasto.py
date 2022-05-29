import numpy as np
from scipy.special import lambertw, gammainc
from scipy.interpolate import interp1d

from constants import *
# G = 43007.1  # Gravitational constant in units [(km/s)**2*(kpc/(1e10*M_s))]
# H0 = 0.1  # Hubble in units of [(km/s)/(kpc/h)]
# h = 0.7 # Dimensionless Hubble parameter
# r0 = 8*h # Position of sun in [kpc/h]
# r_vir = 213.5*h # r200 for MW in [kpc/h]
# # Use conversion [1M_s/pc^3]=37.96[GeV/cm**3], i.e. [1e10*M_s/(kpc**3/h**2)]=379.6*h**2[GeV/cm**3]
# rho_s = 0.4/(379.6*h**2) # Local density at r0 in [1e10M_s/(kpc**3/h**2)]

#  Load Ludlow14 concentration data
# ludlow_data = np.loadtxt('Ludlow_WMAP1_cMz.dat')
# ludlow_spline = interp1d(ludlow_data[:,0], ludlow_data[:,1], bounds_error=False)

class EinastoHalo:

    def __init__(self, m=100, c=None, model = 'E-AQ', r_s=0.81, alpha_E=0.678, verbose=False):
        """ Class that defines an Einasto halo and its parameters. All radius in unit of kpc/h. 
        Effective parameters alpha_E, r_s taking into account tidal stripping are adapted from 1606.04898v2 

        :param m: Halo mass m200 in units of 1e10*M_s/h.
        :param model: Host model ('MW' = the total Milky Way DM profile, 'E-AQ' = tidally stripped subhalo density profile)
        :param r_s: Einasto scale radius (=r_-2) in units of r200
        :param alpha_E: Einasto parameter 
        :param c: Halo concentration. If concentration c is not given, determine c from the default 
        mass-concentration relation
        """
        if model == 'MW':
            r_s = 0.071 
            alpha_E = 0.17

        if model == 'E-AQ':
            r_s = 0.81
            alpha_E = 0.678

        self.m = m
        if c is None:
            self.c = mean_concentration(self.m)
        else:
            self.c = c

        self.r_vir = r_vir
        self.r0 = r0
        self.r_s = r_s*r_vir # in [kpc/h]
        self.alpha_E = alpha_E
        self.rho0  = rho_s/np.exp(-2/alpha_E*((r0/self.r_s)**alpha_E-1))
        if verbose:
            print "Model=", model, "Rs=", r_s, "alpha=", alpha_E, "rho0=", self.rho0

        # self.d_alpha = 3*alpha_E-1/3+0.0079/alpha_E # Approximation from 1202.5242

    # def mass(self, r):
    #     """ Cumulative mass profile, from 1202.5242
    #     """
    #     s = self.d_alpha**self.alpha_E*r/self.r_s
    #     4*np.pi*self.r_s**3*self.rho0/self.alpha_E*np.exp((3*np.log(self.alpha_E)+2-np.log(8))/np.alpha)* # Eq. 16 in 0809.0898v1
    #     return self.m*einasto_func(3*self.alpha_E,s**(1/self.alpha_E))

    def density(self, r):
        """ Einasto density profile
        """ 
        return self.rho0*np.exp(-2/self.alpha_E*((r/self.r_s)**self.alpha_E-1)) 

def mean_concentration(m200, model='Prada'):
    """ Mean concentration at z=0. 

        :param m200: Halo mass in units of 1e10*M_s/h 
        :param model: Halo concentration. 'Ludlow', 'Prada' or 'MaccioW1'.
    """
    if model == 'Prada':
        """ According to Sanchez-Conde&Prada14, with scatter of 0.14dex.
        """
        x=np.log(m200*1e10)
        pars=[37.5153, -1.5093, 1.636e-2, 3.66e-4, -2.89237e-5, 5.32e-7][::-1]
        return np.polyval(pars, x)


    if model == 'MaccioW1':
        """ Maccio08 WMAP1, C200, z=0, relaxed; sigma(log10)=0.11
        """
        return 8.26*(m200/1e2)**-0.104 

    if model == 'Ludlow':
        """ Ludlow14, z=0
        """
        return ludlow_spline(np.log10(m200)) 

    print "Unknown model!"
    raise
def einasto_func(a,x):
    """ Function that appears in the integral of rho_einasto dV. See e.g. eq 13 in 1202.5242
    """
    return gammainc(a,x)
      
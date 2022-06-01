
import numpy as np
import scipy.stats
import bilby

from . import model

def poiss_err(n, alpha=0.32):
    """
    Poisson error (variance) for n counts.
    An excellent review of the relevant statistics can be found in
    the PDF statistics review: http://pdg.lbl.gov/2018/reviews/rpp2018-rev-statistics.pdf,
    specifically section 39.4.2.3
    :param: alpha corresponds to central confidence level 1-alpha,
            i.e. alpha = 0.32 corresponds to 68% confidence
    """
    sigma_lo = scipy.stats.chi2.ppf(alpha/2,2*n)/2
    sigma_up = scipy.stats.chi2.ppf(1-alpha/2,2*(n+1))/2
    return sigma_lo, sigma_up


def log10_plummer2d(R, L, r_star):
    """ Log 10 of the Plummer 2D profile
    Args:
        R: projected radius
        params: L, a
    Returns:
        log I(R) = log {L(1 + R^2 / a^2)^{-2} / (pi * r_star^2)}
    """
    logL = np.log10(L)
    logr_star = np.log10(r_star)
    return logL - 2 * logr_star - 2 * np.log10(1 + R**2 / r_star**2) - np.log10(np.pi)


def log10_plummer3d(r, L, r_star):
    """ Log 10 of the Plummer 2D profile
    Args:
        R: projected radius
        params: L, r_star
    Returns:
        log I(R) = log {L(1 + R^2 / a^2)^{-2} / (pi * r_star^2)}
    """
    logL = np.log10(L)
    logr_star = np.log10(r_star)
    return logL - 3 * logr_star - (5/2) * np.log10(1 + r**2 / r_star**2) - np.log10(4 * np.pi / 3)


def calc_nstar(R):
    ''' Calculate the projected number of stars as a function of projected radius R'''
    # calculate the projected radius
    logR = np.log10(R)
    N_star = len(R)

    # binning
    nbins = int(np.ceil(np.sqrt(N_star)))
    logR_min = np.floor(np.min(logR)*10) / 10
    logR_max = np.ceil(np.max(logR)*10) / 10
    n_data, logR_bins = np.histogram(logR, nbins, range=(logR_min, logR_max))

    # ignore bin with zero count
    select = n_data > 0
    n_data = n_data[select]
    logR_bins_lo = logR_bins[:-1][select]
    logR_bins_hi = logR_bins[1:][select]

    # compute poisson error
    n_data_lo, n_data_hi = poiss_err(n_data, alpha=0.32)

    return n_data, n_data_lo, n_data_hi, logR_bins_lo, logR_bins_hi


def calc_Sigma(R):
    ''' Calculate the projected 2d light profile Sigma(R) where R is the projected radius '''
    n_data, n_data_lo, n_data_hi, logR_bins_lo, logR_bins_hi =  calc_nstar(R)
    R_bins_lo = 10**logR_bins_lo
    R_bins_hi = 10**logR_bins_hi
    R_bins_ce = 0.5 * (R_bins_lo + R_bins_hi)

    # light profile
    delta_R2 = (R_bins_hi**2 - R_bins_lo**2)
    Sigma_data = n_data / (np.pi * delta_R2)
    Sigma_data_lo = n_data_lo / (np.pi * delta_R2)
    Sigma_data_hi = n_data_hi / (np.pi * delta_R2)

    return Sigma_data, Sigma_data_lo, Sigma_data_hi, R_bins_ce


class PlummerModel(model.Model):
    ''' Class to fit the Plummer model to the light profile data '''
    def __init__(self, R, priors={}):
        '''
        Parameters:
        - R: (array of N floats) the projected radii of N stars
        - priors: (dict) dictionary with prior range
        '''
        super().__init__(parameters={"L": None, "r_star": None})

        self.R = R
        self.Sigma, self.Sigma_lo, self.Sigma_hi, self.Rbins_ce = calc_Sigma(R)

        # set up priors
        self.priors = {
            "L": bilby.core.prior.LogUniform(1e-2, 1e5, "L"),
            "r_star": bilby.core.prior.LogUniform(1e-3, 1e3, "r_star")
        }
        self.priors.update(priors)

        # calculate the low and high error
        self.sig_lo = self.Sigma - self.Sigma_lo
        self.sig_hi = self.Sigma_hi - self.Sigma
        self.V1 = self.sig_lo * self.sig_hi
        self.V2 = self.sig_hi - self.sig_lo

    def log_likelihood(self):
        ''' Log likelihood function defined as:
        ```
            logL = -0.5 * (Sigma - Sigma_hat)^2 / (V1 - V2 * (Sigma - Sigma_hat))
        ```
        where:
        - Sigma is the light profile as inferred from data
        - Sigma_hat is the estimated light profile
        - V1 and V2
        '''
        L = self.parameters["L"]
        r_star = self.parameters["r_star"]
        Sigma_hat = 10**log10_plummer2d(self.Rbins_ce, L, r_star)
        delta_Sigma = self.Sigma - Sigma_hat
        return - 0.5 * np.sum(delta_Sigma**2 / (self.V1 - self.V2 * delta_Sigma))



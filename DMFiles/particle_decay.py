"""
DM decay spectra and associated functions.
Everything is provided and output in natural units!
"""
import os, sys
from scipy.special import erf
from scipy.interpolate import interp1d 
import numpy as np
import pandas as pd
from units import *
from scipy import integrate

class Particle:
    def __init__(self, channel='b', m_chi=100*GeV, data_dir = '/tigress/smsharma/Fermi-SmoothGalHalo/DMFiles/'):
        """ Initialize the parameters to generate a sample of subhalos.

            :param channel: Decay channel -- 'b', 'W' and the like
            :param m_chi: Particle DM mass in natural units
            :param data_dir: Where the annihilation spectra are stored
        """

        self.data_dir = data_dir
        self.m_chi = m_chi
        self.channel = channel
        self.dNdLogx_df=pd.read_csv(self.data_dir+'AtProduction_gammas.dat', delim_whitespace=True)
       
        self.dNdE() # Interpolate spectra
            
    def dNdE(self):
        """ Make interpolated decay spectra for given mass and channel
        """
        self.m_chi_interp = self.m_chi/2. # Note: PPPC4DMID gives 2->2 spectra; the correct 1->2 spectra for mchi correspond to 2->2 spectra for mchi/2 
        dNdLogx_ann_df = self.dNdLogx_df.query('mDM == ' + (str(np.int(float(self.m_chi_interp)/GeV))))[['Log[10,x]',self.channel]]
        self.Egamma = np.array(self.m_chi_interp*(10**dNdLogx_ann_df['Log[10,x]']))
        self.dNdEgamma = np.array(dNdLogx_ann_df[self.channel]/(self.Egamma*np.log(10)))        
        self.dNdE_interp =  interp1d(self.Egamma, self.dNdEgamma)

    def Phi_decay(self, tau, Emin, Emax):
        """ Integrated flux for a given energy range from dNdE
            Everything in natural units!

            :param tau: Decay time constant
            :param Emin: Energy to integrate from
            :param Emax: Energy to integrate up to

        """
        N = 0
        if Emin < self.m_chi_interp:
            if Emax < self.m_chi_interp:
                N = integrate.quad(lambda x: self.dNdE_interp(x), Emin, Emax)[0]
            else:
                N = integrate.quad(lambda x: self.dNdE_interp(x), Emin, self.m_chi_interp)[0]
        return 1/(4*np.pi*tau*self.m_chi)*np.array(N)#/(Emax-Emin)

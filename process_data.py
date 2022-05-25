import argparse, ast
import numpy as np
import os, sys
import pandas as pd
from scipy.stats import chi2

parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", action="store", dest="data_dir", default="/tigress/ljchang/dSph_likelihoods/RunFiles/data/gaia_challenge/gs010_bs050_rcrs025_rarcinf_cusp_0064mpc3_df/", type=str)
parser.add_argument("--data_tag", action="store", dest="data_tag", default="gs010_bs050_rcrs025_rarcinf_cusp_0064mpc3_df_100_0", type=str)
parser.add_argument("--out_dir_base", action="store", dest="out_dir_base", default="/tigress/ljchang/dSph_likelihoods/RunFiles/data/gaia_challenge/gs010_bs050_rcrs025_rarcinf_cusp_0064mpc3_df/processed_samps/poiss_err/", type=str )
parser.add_argument("--out_tag", action="store", dest="out_tag", default="df_100_0", type=str)

results = parser.parse_args()
data_dir = results.data_dir
data_tag = results.data_tag
out_dir_base = results.out_dir_base
out_tag = results.out_tag

if not os.path.exists(out_dir_base):
    os.makedirs(out_dir_base)

def poiss_err(n,alpha=0.32):
    """
    Poisson error (variance) for n counts.
    An excellent review of the relevant statistics can be found in 
    the PDF statistics review: http://pdg.lbl.gov/2018/reviews/rpp2018-rev-statistics.pdf,
    specifically section 39.4.2.3

    :param: alpha corresponds to central confidence level 1-alpha, 
            i.e. alpha = 0.32 corresponds to 68% confidence
    """
    sigma_lo = chi2.ppf(alpha/2,2*n)/2
    sigma_up = chi2.ppf(1-alpha/2,2*(n+1))/2

#     sigma = (2*sigma_lo*sigma_up)/(sigma_lo+sigma_up)
#     sigma_prime = (sigma_up-sigma_lo)/(sigma_lo+sigma_up)

    return sigma_lo, sigma_up


data_file_path = data_dir+data_tag+".dat"

# By default, project along z-axis

coords_cartesian = pd.read_csv(data_file_path,header=None,delim_whitespace=True,usecols=[0,1,2]).values
vz = pd.read_csv(data_file_path,header=None,delim_whitespace=True,usecols=[5]).values.flatten()

rsamp = np.array([np.sqrt(coords_cartesian[i,0]**2+coords_cartesian[i,1]**2) for i in range(len(coords_cartesian))])
nstars = len(rsamp)

np.savez(out_dir_base+out_tag,distances=rsamp,velocities=vz)

if nstars == 100:
	nbins = 10
elif nstars == 1000:
	nbins = 32
else:
	nbins = int(round(np.sqrt(nstars)))

binned_data_0 = np.zeros((nbins))
bin_centers_0 = np.zeros((nbins))

lower_bin_edge = np.floor(np.log10(min(rsamp))*10)/10
upper_bin_edge = np.ceil(np.log10(max(rsamp))*10)/10

rbins = np.logspace(lower_bin_edge,upper_bin_edge,nbins+1)
bin_centers_0 = np.array([np.sqrt(rbins[i]*rbins[i+1]) for i in range(len(rbins)-1)])
binned_data_0, bin_edges = np.histogram(rsamp,bins=rbins)
lower_bin_edges = bin_edges[:-1]
upper_bin_edges = bin_edges[1:]

mu_lo, mu_hi = poiss_err(binned_data_0[binned_data_0>0]) # Only take bins which have nonzero stars, because Poisson error only makes sense with n=/=0
sigma_lo = binned_data_0[binned_data_0>0]-mu_lo
sigma_hi = mu_hi-binned_data_0[binned_data_0>0]

np.savez(out_dir_base+out_tag+'_lp',bin_centers=bin_centers_0[binned_data_0>0],lower_bin_edges=lower_bin_edges[binned_data_0>0],upper_bin_edges=upper_bin_edges[binned_data_0>0],counts=binned_data_0[binned_data_0>0],sigmas_lo=sigma_lo,sigmas_hi=sigma_hi)

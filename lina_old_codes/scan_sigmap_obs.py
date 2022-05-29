import sys, os
import numpy as np
import pymultinest, corner
from scipy.optimize import minimize
import sigmap_functions
# from tqdm import tqdm
# import pandas as pd
import matplotlib.pyplot as plt
import matplotlib 
matplotlib.use('Agg')
from matplotlib import rcParams

rcParams['xtick.labelsize'] = 20
rcParams['ytick.labelsize'] = 20
rcParams['axes.labelsize'] = 20
rcParams['axes.titlesize'] = 20
rcParams['font.family'] = 'serif'
rcParams['font.serif'] = 'CMU Serif'
rcParams['figure.figsize'] = (10/1.2,8/1.2)
rcParams['legend.fontsize'] = 16
rcParams['text.usetex'] = True
rcParams['text.latex.preamble'] = [r"\usepackage{amsmath}"]

class run_scan():
	def __init__(self, data_file_path, out_file_path, nbins, deltav=2., measured_errs=False):

		self.data_file_path = data_file_path
		self.out_file_path = out_file_path
		self.nbins = nbins
		self.deltav = deltav
		self.measured_errs = measured_errs
		# self.data_file_path = data_file_path

		self.params = [r"$\bar{v}$",r"$\sigma_p$"]
		self.priors = [[-30.,30.],[0.,30.]]	

		self.n_params = len(self.params)
		self.load_data()

	def load_data(self):
		""" Load the data
		"""
		self.data = np.load(self.data_file_path)
		self.projected_distance = self.data['distances']
		self.projected_vel = self.data['velocities']

		if self.measured_errs:
			self.vel_err = self.data['v_errs']
			print("Using measured velocity errors")
		else:
			self.vel_err = np.array([self.deltav for k in range(len(self.projected_vel))])

		self.bin_centers, self.binned_distances, self.binned_inds = self.get_bins(self.projected_distance,self.nbins)
		self.binned_vels = [self.projected_vel[inds] for inds in self.binned_inds]
		self.binned_errs = [self.vel_err[inds] for inds in self.binned_inds]

	def scipy_scan(self):
		self.vel_disp_binned = np.zeros(self.nbins)
		self.vel_mean_binned = np.zeros(self.nbins)
		self.vel_disp_errs_binned = np.zeros(self.nbins)

		for ibin in range(self.nbins):
			vels = self.binned_vels[ibin]
			errs = self.binned_errs[ibin]

			scpy_min_BFGS = minimize(lambda x: -sigmap_functions.lnlike(vels,errs,x), x0=[1. for i in range(2)], bounds=[[-100.,100.],[0.,100.]], options={'disp':False,'ftol':1e-12}, method='L-BFGS-B')
			best_fit_params = scpy_min_BFGS['x']
			
			vbar, sigmap = best_fit_params
			self.vel_mean_binned[ibin] = vbar
			self.vel_disp_binned[ibin] = sigmap
			
			deltasigmap = sigmap_functions.one_sig(vels,errs,best_fit_params[0],best_fit_params[1])
			self.vel_disp_errs_binned[ibin] = deltasigmap

			print("bin",str(ibin)," ",[vbar,sigmap,deltasigmap])

		np.savez(self.out_file_path,bin_centers=self.bin_centers,vel_mean_binned=self.vel_mean_binned,vel_disp_binned=self.vel_disp_binned,vel_disp_errs_binned=self.vel_disp_errs_binned)

	@staticmethod
	def get_bins(rsamp,nbins):
		num_per_bin = int(np.ceil(len(rsamp)/nbins))
		num_last_bin = len(rsamp)-num_per_bin*(nbins-1)
		binned_data = []
		binned_inds = []
		bin_edges = np.zeros(nbins+1)
		inds = np.argsort(rsamp)
		sorted_R_vals = rsamp[inds]

		if num_per_bin != num_last_bin:
			for ibin in range(nbins-1):
				binned_data.append(sorted_R_vals[ibin*num_per_bin:(ibin+1)*num_per_bin])
				binned_inds.append(inds[ibin*num_per_bin:(ibin+1)*num_per_bin])
			binned_data.append(sorted_R_vals[num_per_bin*(nbins-1):])
			binned_inds.append(inds[num_per_bin*(nbins-1):])
		else:
			for ibin in range(nbins):
				binned_data.append(sorted_R_vals[ibin*num_per_bin:(ibin+1)*num_per_bin])
				binned_inds.append(inds[ibin*num_per_bin:(ibin+1)*num_per_bin])
		
		bin_centers = [np.median(r) for r in binned_data]
		
		return np.array(bin_centers), binned_data, binned_inds

	@staticmethod
	def make_dirs(dirs):
		""" Creates directories if they do not already exist 
		"""
		for d in dirs:
			if not os.path.exists(d):
				try:
					os.mkdir(d)
				except OSError as e:
					if e.errno != 17:
						raise
import sys, os
import numpy as np
import corner
import DM_profiles
import matplotlib.pyplot as plt
import matplotlib 
matplotlib.use('Agg')
from matplotlib import rcParams

rcParams['xtick.labelsize'] = 22
rcParams['xtick.direction'] = 'in'
rcParams['ytick.labelsize'] = 22
rcParams['ytick.direction'] = 'in'
rcParams['axes.labelsize'] = 22
rcParams['axes.titlesize'] = 22
rcParams['font.family'] = 'serif'
rcParams['font.serif'] = 'CMU Serif'
rcParams['figure.figsize'] = (10/1.2,8/1.2)
rcParams['legend.fontsize'] = 18
rcParams['text.usetex'] = True
rcParams['text.latex.preamble'] = [r"\usepackage{amsmath}"]

log10toln = np.log10(np.exp(1))

class make_plots():
	def __init__(self, plots_dir,
				 post_paths_list = [],
				 has_truth = True,
				 true_params = [np.log(0.064e9),np.log(1),1,3,1]):

		self.plots_dir = plots_dir
		self.post_paths_list = post_paths_list
		self.has_truth = has_truth
		self.true_params = true_params

		self.load_posteriors()

	def load_posteriors(self):
		rvals_list = []
		post_rho_list = []
		post_Menc_list = []
		rho_true_list = []
		Menc_true_list = []
		minr_list = []
		maxr_list = []

		self.nscans = len(self.post_paths_list)

		for i in range(self.nscans):
			post_file = np.load(self.post_paths_list[i])
			rvals_list.append(post_file["rvals"])
			post_rho_list.append(post_file["rho"])
			post_Menc_list.append(post_file["Menc"])
			rho_true_list.append(post_file["rho_true"])
			Menc_true_list.append(post_file["Menc_true"])

			minr_list.append(post_file["rvals"][0])
			maxr_list.append(post_file["rvals"][-1])

		self.logrmin = np.floor(np.log(min(minr_list))*100)/100
		self.logrmax = np.ceil(np.log(max(maxr_list))*100)/100
		self.rvals = np.logspace(self.logrmin,self.logrmax,100,base=np.exp(1))

		if self.has_truth:
			self.rho_true = [DM_profiles.rho_DM_Zhao(*self.true_params,r) for r in self.rvals]
			self.Menc_true = [DM_profiles.mass_DM_Zhao(*self.true_params,r) for r in self.rvals]

		self.rvals_list = rvals_list
		self.post_rho_list = post_rho_list
		self.post_Menc_list = post_Menc_list
		self.rho_true_list = rho_true_list
		self.Menc_true_list = Menc_true_list

	def make_density_plots(self, title = '', save_tag = '',
							truth_line_args = {"color":"k","ls":"--","label":"Truth","params":[0.064e9,1,1,3,1]},
							plot_colors = ['cornflowerblue', 'firebrick' ,'forestgreen','darkgoldenrod','darkorchid'],
							plot_labels = [], post_quantiles = [[16,84]],
							results_args = {"color":"cornflowerblue","alpha":0.2}):

		assert len(plot_colors) >= self.nscans, "Need to specify more colors!"
		assert len(plot_labels) == self.nscans, "Number of labels "+str(len(plot_labels))+" doesn't match number of scans "+str(self.nscans)

		self.make_dirs([self.plots_dir])

		if self.has_truth:
			f, ax = plt.subplots(2,2,gridspec_kw={'height_ratios': [4, 1]},sharex=True)
			f.set_figheight(7.5)
			f.set_figwidth(14)

			####################
			# Plot truth lines #
			####################

			ax[0,0].plot(self.rvals,self.rho_true,ls=truth_line_args["ls"],color=truth_line_args["color"],label=truth_line_args["label"])
			ax[0,0].set_yscale('log')
			ax[0,0].set_xscale('log')
			ax[0,0].set_ylabel(r'$\rho(R)\,[M_\odot/\mathrm{kpc}^3]$')
			ax[0,0].set_xlim(np.exp(self.logrmin),np.exp(self.logrmax));
			ax[0,0].set_ylim(2e4,8e10);

			ax[0,1].plot(self.rvals,self.Menc_true,ls=truth_line_args["ls"],color=truth_line_args["color"],label=truth_line_args["label"])
			ax[0,1].set_yscale('log')
			ax[0,1].set_xscale('log')
			ax[0,1].set_ylabel(r'M$(R)\,[M_\odot]$')
			ax[0,1].set_xlim(np.exp(self.logrmin),np.exp(self.logrmax));
			ax[0,1].set_ylim(2e4,8e10);

			ax[1,0].axhline(1,ls=truth_line_args["ls"],color=truth_line_args["color"],label=truth_line_args["label"])
			ax[1,0].set_yscale('log')
			ax[1,0].set_xscale('log')
			ax[1,0].set_xlabel(r'$R$ [kpc]')
			ax[1,0].set_ylabel(r'$\rho(R)/\rho^\mathrm{True}(R)$')
			ax[1,0].set_xlim(np.exp(self.logrmin),np.exp(self.logrmax));
			ax[1,0].set_ylim(1e-2,1e2);
			ax[1,0].set_yticks([1e-2,1e0,1e2])			
			
			ax[1,1].axhline(1,ls=truth_line_args["ls"],color=truth_line_args["color"],label=truth_line_args["label"])
			ax[1,1].set_yscale('log')
			ax[1,1].set_xscale('log')
			ax[1,1].set_xlabel(r'$R$ [kpc]')
			ax[1,1].set_ylabel(r'M$(R)/$M$^\mathrm{True}(R)$')
			ax[1,1].set_xlim(np.exp(self.logrmin),np.exp(self.logrmax));
			ax[1,1].set_ylim(1e-2,1e2);
			ax[1,1].set_yticks([1e-2,1e0,1e2])

			for thresh in [1e-1,1e1]:
				ax[1,0].axhline(thresh,ls=':',color='dimgray')
				ax[1,1].axhline(thresh,ls=':',color='dimgray')

			for i in range(2):
				for j in range(2):
					ax[i,j].minorticks_on()
					ax[i,j].xaxis.set_ticks_position('both')
					ax[i,j].yaxis.set_ticks_position('both')

					# ax[i,j].tick_params('both', length=8, width=1, which='major')
					# ax[i,j].tick_params('both', length=4, width=1, which='minor')

			########################
			# Plot posterior bands #
			########################

			for i in range(self.nscans):
				ax[0,0].plot(self.rvals_list[i],np.median(self.post_rho_list[i],axis=0),color=plot_colors[i],label=plot_labels[i])
				ax[0,1].plot(self.rvals_list[i],np.median(self.post_Menc_list[i],axis=0),color=plot_colors[i],label=plot_labels[i])
				ax[1,0].plot(self.rvals_list[i],np.median(self.post_rho_list[i],axis=0)/self.rho_true_list[i],color=plot_colors[i],label=plot_labels[i])
				ax[1,1].plot(self.rvals_list[i],np.median(self.post_Menc_list[i],axis=0)/self.Menc_true_list[i],color=plot_colors[i],label=plot_labels[i])
			
				for qs in range(len(post_quantiles)):
					ax[0,0].fill_between(self.rvals_list[i],np.percentile(self.post_rho_list[i],q=post_quantiles[qs][0],axis=0),np.percentile(self.post_rho_list[i],q=post_quantiles[qs][1],axis=0),alpha=results_args["alpha"],color=plot_colors[i])
					ax[0,1].fill_between(self.rvals_list[i],np.percentile(self.post_Menc_list[i],q=post_quantiles[qs][0],axis=0),np.percentile(self.post_Menc_list[i],q=post_quantiles[qs][1],axis=0),alpha=results_args["alpha"],color=plot_colors[i])
					ax[1,0].fill_between(self.rvals_list[i],np.percentile(self.post_rho_list[i]/self.rho_true_list[i],q=post_quantiles[qs][0],axis=0),np.percentile(self.post_rho_list[i]/self.rho_true_list[i],q=post_quantiles[qs][1],axis=0),alpha=results_args["alpha"],color=plot_colors[i])
					ax[1,1].fill_between(self.rvals_list[i],np.percentile(self.post_Menc_list[i]/self.Menc_true_list[i],q=post_quantiles[qs][0],axis=0),np.percentile(self.post_Menc_list[i]/self.Menc_true_list[i],q=post_quantiles[qs][1],axis=0),alpha=results_args["alpha"],color=plot_colors[i])

			ax[0,1].legend(loc='lower right',frameon=False)

			sup = plt.suptitle(title, y= 1.04, fontsize=28)
			f.tight_layout()
			f.subplots_adjust(hspace=0.15)

			out_path = self.plots_dir+"/rho_Menc_"+save_tag

			plt.savefig(out_path+".pdf",bbox_extra_artists=(sup,), bbox_inches="tight")
			plt.close()

		else:
			f, ax = plt.subplots(1,2)
			f.set_figheight(6)
			f.set_figwidth(14)

			####################
			# Plot truth lines #
			####################

			ax[0].plot(self.rvals,np.median(post_rho,axis=0),color=results_args["color"],label=dm_model_plot+r' fit')
			ax[0].set_yscale('log')
			ax[0].set_xscale('log')
			ax[0].set_xlabel(r'$R$ [kpc]')
			ax[0].set_ylabel(r'$\rho_\mathrm{DM}(R)\,[M_\odot/\mathrm{kpc}^3]$')
			ax[0].set_xlim(np.exp(self.logrmin),np.exp(self.logrmax));
			ax[0].set_ylim(2e4,8e10);

			ax[1].plot(self.rvals,np.median(post_Menc,axis=0),color=results_args["color"],label=dm_model_plot+r' fit')
			ax[1].set_yscale('log')
			ax[1].set_xscale('log')
			ax[1].set_xlabel(r'$R$ [kpc]')
			ax[1].set_ylabel(r'M$_\mathrm{enc}(R)\,[M_\odot]$')
			ax[1].set_xlim(np.exp(self.logrmin),np.exp(self.logrmax));
			ax[1].set_ylim(2e4,8e10);
			ax[1].legend(loc='lower right',frameon=False)

			for i in range(self.nscans):
				ax[0].plot(self.rvals_list[i],np.median(self.post_rho_list[i],axis=0),color=plot_colors[i],label=plot_labels[i])
				ax[1].plot(self.rvals_list[i],np.median(self.post_Menc_list[i],axis=0),color=plot_colors[i],label=plot_labels[i])
			
				for qs in len(post_quantiles):
					ax[0].fill_between(self.rvals_list[i],np.percentile(self.post_rho_list[i],q=post_quantiles[qs][0],axis=0),np.percentile(self.post_rho_list[i],q=post_quantiles[qs][1],axis=0),alpha=results_args["alpha"],color=plot_colors[i])
					ax[1].fill_between(self.rvals_list[i],np.percentile(self.post_Menc_list[i],q=post_quantiles[qs][0],axis=0),np.percentile(self.post_Menc_list[i],q=post_quantiles[qs][1],axis=0),alpha=results_args["alpha"],color=plot_colors[i])

			sup = plt.suptitle(title, y= 1.04, fontsize=28)
			f.tight_layout()

			out_path = self.plots_dir+"/rho_Menc_"+save_tag

			plt.savefig(out_path+".pdf",bbox_extra_artists=(sup,), bbox_inches="tight")
			plt.close()

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
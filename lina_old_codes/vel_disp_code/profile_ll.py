import sys, os
import numpy as np
from tqdm import tqdm
from scipy.optimize import minimize 

import scan_vel_disp_multinest
import DM_profiles
import functions_multinest

sys.path.append("../")
import lp_PlumSph
import lp_Zhao

log10toln = np.log10(np.exp(1))

iMC = 8

rsamp = np.load('/tigress/ljchang/dSph_likelihoods/Generate_mocks/mocks/a_1_b_3_g_0_rdm_1_iso/dist_as_2.0_bs_5.0_gs_0.1_rs_1.0_dm_100_'+str(iMC)+'_rho_400.0del_v_2.0.npz')['distances']
vsamp = np.load('/tigress/ljchang/dSph_likelihoods/Generate_mocks/mocks/a_1_b_3_g_0_rdm_1_iso/dist_as_2.0_bs_5.0_gs_0.1_rs_1.0_dm_100_'+str(iMC)+'_rho_400.0del_v_2.0.npz')['velocities']

light_profile_params = np.load('/tigress/ljchang/dSph_likelihoods/Generate_mocks/mocks/a_1_b_3_g_0_rdm_1_iso/light_profile_bestfits/dist_as_2.0_bs_5.0_gs_0.1_rs_1.0_dm_100_'+str(iMC)+'_rho_400.0_1comp.npz')

nu_model = "Plummer"
dm_model = "gNFW"

nlp = 2
ndm = 3

theta = np.zeros(nlp+ndm+1)
theta[:nlp] = light_profile_params['median']/log10toln
theta[nlp:nlp+ndm] = [15,2,0]

logrmax = np.ceil(np.log(max(rsamp))*100)/100

gamma_test = np.linspace(-0.5,2,50)

ll_ary = np.zeros(len(gamma_test))
params_ary = np.zeros((len(gamma_test),5))

def ll_profile(theta,gamma,vsamp,dvsamp,rsamp,logrmax,nu_model,dm_model):
	theta_all = np.zeros(len(theta))
	theta_all[:] = theta
	theta_all[4] = gamma
	
	return functions_multinest.lnlike(theta_all,vsamp,dvsamp,rsamp,0,10*logrmax,nu_model,dm_model)
	
for i in tqdm(range(len(gamma_test))):
	scpy_min_BFGS = minimize(lambda x: -ll_profile(x,gamma_test[i],vsamp,np.array([2 for i in range(len(vsamp))]),rsamp,logrmax,nu_model,dm_model), x0=[*light_profile_params['median']/log10toln,20,0.,0.], bounds=[light_profile_params['middle95'][0]/log10toln,light_profile_params['middle95'][1]/log10toln,[5,30],[-10,10],[-400,400]], options={'disp':True,'ftol':1e-12}, method='L-BFGS-B')

	ll_ary[i] = -scpy_min_BFGS['fun']
	params_ary[i] = scpy_min_BFGS['x']

	np.save('profile_ll_MC_'+str(iMC)+'_'+str(i)+'.npy',-scpy_min_BFGS['fun'])
	np.save('profile_ll_bestfits_MC_'+str(iMC)+'_'+str(i)+'.npy',scpy_min_BFGS['x'])

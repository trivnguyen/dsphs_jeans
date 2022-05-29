import argparse, ast
import numpy as np
import os, sys
import plot

parser = argparse.ArgumentParser()
parser.add_argument("--plots_dir", action="store", dest="plots_dir", default="/tigress/ljchang/dSph_likelihoods/RunFiles/DM_plots/", type=str)
parser.add_argument("--title", action="store", dest="title", default="", type=str)
parser.add_argument("--save_tag", action="store", dest="save_tag", default="", type=str)
parser.add_argument("--post_paths_list", nargs="+", type=str,default=[])
parser.add_argument("--true_params", nargs="+", type=float,default=[])
parser.add_argument("--plot_colors", nargs="+", type=float,default=['cornflowerblue','firebrick', 'forestgreen', 'darkgoldenrod','darkorchid'])
parser.add_argument("--plot_labels", nargs="+", dest = "plot_labels", type=str,default="") # nargs="+"
parser.add_argument("--has_truth", action="store", dest="has_truth", default=True, type=ast.literal_eval) 

results = parser.parse_args()
plots_dir = results.plots_dir
title = results.title
save_tag = results.save_tag
post_paths_list = results.post_paths_list
true_params = results.true_params
plot_colors = results.plot_colors
plot_labels = results.plot_labels
has_truth = results.has_truth

# print("Interface is receiving the model ",dm_model)

plotter = plot.make_plots(plots_dir=plots_dir,post_paths_list=post_paths_list,has_truth=has_truth,true_params=true_params)
plotter.make_density_plots(title=title,save_tag=save_tag,plot_colors=plot_colors,plot_labels=plot_labels)

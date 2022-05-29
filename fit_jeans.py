#!/usr/bin/env python

import os
import h5py
import sys
import argparse
import logging
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import dynesty

from dwarfs_dm import light_profiles, dm_profiles

FLAGS = None
def parse_cmd():
    ''' Parse cmd arguments '''
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', required=True, help='Path to input coordinate table')
    parser.add_argument('-o', '--output', required=True, help='Path to output directory')
    return parser.parse_args()

# function to set logger
def set_logger():
    ''' Set up stdv out logger and file handler '''
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    # add streaming handler
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    return logger

if __name__ == '__main__':
    ''' Run Jeans analysis '''
    FLAGS = parse_cmd()
    logger = set_logger()

    # create output directory
    logger.info(f'writing output to {FLAGS.output}')
    os.makedirs(FLAGS.output, exist_ok=True)

    # read in projected coordinates
    logger.info('read coordinates from {FLAGS.input}')
    X, Y, _, _, _, v = np.genfromtxt(FLAGS.input, unpack=True)
    R = np.sqrt(X**2 + Y**2)

    # fit light profile with the Plummer model
    logger.info(f'fit light profile using Plummer 2D model')
    plummer_model = light_profiles.PlummerModel(R)
    plummer_model.sample(
        save_history=True, history_filename=os.path.join(FLAGS.output, 'plummer_fit.hdf5'),
        nlive=1000)
    logL = plummer_model.get_median('logL')
    logr_star = plummer_model.get_median('logr_star')
    logger.info(f'Best fit: logL {logL}; logr_star {logr_star}')

    # fit DM model
    logger.info('fit DM profiles using gNFW model')
    jeans_model = dm_profiles.JeansModel(R, v, logL, logr_star)
    jeans_model.sample(
        save_history=True, history_filename=os.path.join(FLAGS.output, 'jeans_model.hdf5'),
        nlive=1000)
    logger.info(f'Best fit: logL {logL}; logr_star {logr_star}')
    logr_dm = jeans_model.get_median('logr_dm')
    gamma = jeans_model.get_median('gamma')
    logrho_0 = jeans_model.get_median('logrho_0')
    logger.info(f'Best fit: logr_dm {logr_dm}; gamma {gamma}; logrho_0 {logrho_0}')


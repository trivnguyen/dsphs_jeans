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

# parse command line argument
def parse_cmd():
    ''' Parse cmd arguments '''
    parser = argparse.ArgumentParser()

    # input output args
    parser.add_argument(
        '-i', '--input', required=True, help='Path to input coordinate table')
    parser.add_argument(
        '-o', '--output',
        required=True, help='Path to output directory')

    # likelihood args
    parser.add_argument(
        '--r-step', required=False, type=float, default=0.001,
        help='Step size of the integration radius in kpc')
    parser.add_argument(
        '--v-error', required=False, type=float, default=2,
        help='Velocity measurement error')
    parser.add_argument(
        '--r-min-factor', required=False, type=float, default=2,
        help='factor to convert R_min to r_min')
    parser.add_argument(
        '--r-max-factor', required=False, type=float, default=2,
        help='factor to convert R_max to r_max')
    parser.add_argument(
        '--r-a', required=False, type=float, default=1e30,
        help='Velocity anisotropy scale radius')

    # prior args
    parser.add_argument(
        '--log-rdm', required=False, type=float, nargs=2, default=(-1, 0.7),
        help='Log scale radius of  DM')
    parser.add_argument(
        '--gamma', required=False, type=float, nargs=2, default=(-1, 5),
        help='Inner slope')
    parser.add_argument(
        '--log-rho', required=False, type=float, nargs=2, default=(5, 8),
        help='Central density')

    return parser.parse_args()


if __name__ == '__main__':
    ''' Run Jeans analysis '''
    FLAGS = parse_cmd()
    logger = set_logger()

    # create output directory
    logger.info(f'writing output to {FLAGS.output}')
    os.makedirs(FLAGS.output, exist_ok=True)

    # read in projected coordinates
    logger.info(f'read coordinates from {FLAGS.input}')
    X, Y, _, _, _, v = np.genfromtxt(FLAGS.input, unpack=True)
    R = np.sqrt(X**2 + Y**2)

    # fit light profile with the Plummer model
    logger.info(f'fit light profile using Plummer 2D model')
    plummer_model = light_profiles.PlummerModel(R)
    plummer_model.sample(
        outfile=os.path.join(FLAGS.output, 'plummer_fit.pkl'),
        nlive=1000)
    logL = plummer_model.get_median('logL')
    logr_star = plummer_model.get_median('logr_star')
    logger.info(f'Best fit: logL {logL}; logr_star {logr_star}')

    # fit DM model
    logger.info('fit DM profiles using gNFW model')
    priors = {
        'logr_dm': FLAGS.log_rdm,
        'gamma': FLAGS.gamma,
        'logrho_0': FLAGS.log_rho,
    }
    jeans_model = dm_profiles.JeansModel(
        R, v, logL, logr_star, priors=priors, dr=FLAGS.r_step, v_err=FLAGS.v_error,
        r_min_factor=FLAGS.r_min_factor, r_max_factor=FLAGS.r_max_factor,
        r_a=FLAGS.r_a
    )
    jeans_model.sample(
        outfile=os.path.join(FLAGS.output, 'jeans_fit.pkl'),
        nlive=1000)
    logr_dm = jeans_model.get_median('logr_dm')
    gamma = jeans_model.get_median('gamma')
    logrho_0 = jeans_model.get_median('logrho_0')
    logger.info(f'Best fit: logr_dm {logr_dm}; gamma {gamma}; logrho_0 {logrho_0}')

    # print out summary statement
    logger.info('Summary:')
    logger.info(f'Best fit: logL {logL}; logr_star {logr_star}')
    logger.info(f'Best fit: logr_dm {logr_dm}; gamma {gamma}; logrho_0 {logrho_0}')



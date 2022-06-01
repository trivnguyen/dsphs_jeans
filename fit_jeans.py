#!/usr/bin/env python

import os
import h5py
import sys
import argparse
import logging
import warnings

import dynesty
import numpy as np
import bilby
from dwarfs_dm import light_profiles, dm_profiles

warnings.filterwarnings("ignore")

FLAGS = None

# function to set logger
def set_logger():
    ''' Set up stdv out logger and file handler '''
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
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
        '-o', '--outdir',
        required=True, help='Path to output directory')
    parser.add_argument(
        '--resume', action='store_true', required=False, help='Enable to resume last run')

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
    logger.info(f'writing output to {FLAGS.outdir}')
    os.makedirs(FLAGS.outdir, exist_ok=True)

    # read in projected coordinates
    logger.info(f'read coordinates from {FLAGS.input}')
    X, Y, _, _, _, v = np.genfromtxt(FLAGS.input, unpack=True)
    R = np.sqrt(X**2 + Y**2)

    # fit light profile with the Plummer model
    logger.info(f'fit light profile using Plummer 2D model')
    plummer_model = light_profiles.PlummerModel(R)
    plummer_result = bilby.run_sampler(
        likelihood=plummer_model, priors=plummer_model.priors,
        sampler="dynesty", nlive=1000, sample='auto',
        label="plummer", outdir=FLAGS.outdir, resume=FLAGS.resume,
    )

    #weights = np.exp(plummer_result.logwt - plummer_result.logz[-1])
    L = np.mean(plummer_result.posterior['L'].values)
    L_sig = np.std(plummer_result.posterior['L'].values)
    r_star = np.mean(plummer_result.posterior['r_star'].values)
    r_star_sig = np.std(plummer_result.posterior['r_star'].values)
    logger.info(f'Best fit: L {L}; r_star {r_star}')

    # fit DM model
    logger.info('fit DM profiles using gNFW model')
    jeans_priors = {
        "L": bilby.core.prior.Gaussian(L, L_sig, "L"),
        "r_star": bilby.core.prior.Gaussian(r_star, r_star_sig, "r_star"),
    }
    jeans_model = dm_profiles.JeansModel(
        R, v, priors=jeans_priors, dr=FLAGS.r_step, v_err=FLAGS.v_error,
        r_min_factor=FLAGS.r_min_factor, r_max_factor=FLAGS.r_max_factor,
        r_a=FLAGS.r_a
    )

    jeans_result = bilby.run_sampler(
        likelihood=jeans_model, priors=jeans_model.priors,
        sampler="dynesty", nlive=1000, sample='auto',
        label="jeans", outdir=FLAGS.outdir, resume=FLAGS.resume
    )

    for key in jeans_model.parameters:
        quantiles = np.percentile(
            jeans_result.posterior[key].values, [16, 50, 84])
        med = quantiles[1]
        lo = med - quantiles[0]
        hi = quantiles[2] - med
        logger.info(f'{key}: {med} + {hi} - {lo}')


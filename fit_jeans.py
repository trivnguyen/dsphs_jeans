#!/usr/bin/env python

import os
import h5py
import sys
import argparse
import logging
import warnings
import shutil

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
        '--overwrite', action='store_true', required=False,
        help='Enable to overwrite previous run')

    # likelihood args
    parser.add_argument(
        '--r-step', required=False, type=float, default=0.001,
        help='Step size of the integration radius in kpc')
    parser.add_argument(
        '--v-error', required=False, type=float, default=0,
        help='Velocity measurement error')
    parser.add_argument(
        '--r-min-factor', required=False, type=float, default=2,
        help='factor to convert R_min to r_min')
    parser.add_argument(
        '--r-max-factor', required=False, type=float, default=2,
        help='factor to convert R_max to r_max')
    parser.add_argument(
        '--ra', required=False, type=float,
        help='Velocity anisotropy scale radius')

    # prior args
    parser.add_argument(
        '--lp-prior-type', required=False, default='ci', type=str.lower,
        choices=('uni', 'normal', 'ci', 'best_mean', 'best_median'),
        help='Prior type of the light profile for Jeans modeling')
    parser.add_argument(
        '--fit-v-mean', required=False, action='store_true',
        help='Enable to fit v mean')
    parser.add_argument(
        '--dm-prior-file', required=False, type=str,
        help='Path to DM prior file')

    # sampler args
    parser.add_argument(
        '--npoints', required=False, default=100, type=int,
        help="Number of live points for dynesty")

    return parser.parse_args()


if __name__ == '__main__':
    ''' Run Jeans analysis '''
    FLAGS = parse_cmd()
    logger = set_logger()

    if (FLAGS.overwrite) and os.path.exists(FLAGS.outdir):
        logger.info('Overwrite existing directory')
        shutil.rmtree(FLAGS.outdir)

    # create output directory
    logger.info(f'writing output to {FLAGS.outdir}')
    os.makedirs(FLAGS.outdir, exist_ok=True)

    # read in projected coordinates
    logger.info(f'read coordinates from {FLAGS.input}')
    try:
        X, Y, _, _, _, v = np.genfromtxt(FLAGS.input, unpack=True)
    except:
        X, Y, v = np.genfromtxt(FLAGS.input, unpack=True)
    R = np.sqrt(X**2 + Y**2)

    # fit light profile with the Plummer model
    logger.info(f'fit light profile using Plummer 2D model')
    plummer_model = light_profiles.PlummerModel(R)
    plummer_model.run_sampler(
        sampler="dynesty", npoints=FLAGS.npoints, sample='auto',
        label="plummer", outdir=FLAGS.outdir, resume=(not FLAGS.overwrite),
    )

    # Define Jeans prior
    priors = {}

    # use fit to determine Jeans modeling prior
    lp_priors = {}
    for key in plummer_model.parameters:
        if FLAGS.lp_prior_type == 'ci':
            val_lo, val_hi = plummer_model.get_credible_intervals(key, p=0.95)
            lp_priors[key] = bilby.core.prior.Uniform(val_lo, val_hi, key)
        elif FLAGS.lp_prior_type == 'normal':
            mean, std = plummer_model.get_mean_and_std(key)
            lp_priors[key] = bilby.core.prior.Gaussian(mean, std, key)
        elif FLAGS.lp_prior_type == 'best_median':
            median = plummer_model.get_median(key)
            lp_priors[key] = bilby.core.prior.DeltaFunction(median, key)
        elif FLAGS.lp_prior_type == 'best_mean':
            mean, std = plummer_model.get_mean_and_std(key)
            lp_priors[key] = bilby.core.prior.DeltaFunction(mean, key)
        elif FLAGS.lp_prior_type == 'uni':
            pass
    priors.update(lp_priors)

    # read in DM prior file if given
    if FLAGS.dm_prior_file is not None:
        dm_priors = bilby.core.prior.PriorDict(filename=FLAGS.dm_prior_file)
        priors.update(dm_priors)

    # anisotropy prior
    if FLAGS.ra is not None:
        priors['r_a'] = bilby.core.prior.DeltaFunction(FLAGS.ra, "r_a")

    # fit DM model
    logger.info('fit DM profiles using gNFW model')
    jeans_model = dm_profiles.JeansModel(
        R, v, priors=priors, dr=FLAGS.r_step, v_err=FLAGS.v_error,
        r_min_factor=FLAGS.r_min_factor, r_max_factor=FLAGS.r_max_factor,
        fit_v_mean=FLAGS.fit_v_mean
    )
    jeans_model.run_sampler(
        sampler="dynesty", npoints=FLAGS.npoints, sample='auto',
        label="jeans", outdir=FLAGS.outdir, resume=(not FLAGS.overwrite)
    )

    # print out a summary statement
    logger.info('Summary')
    for key in jeans_model.parameters:
        med = jeans_model.get_median(key)
        lo, hi = jeans_model.get_credible_intervals(key, p=0.68)
        lo = med - lo
        hi = hi - med
        logger.info(f'{key}: {med} + {hi} - {lo}')


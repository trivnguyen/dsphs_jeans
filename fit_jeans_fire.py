#!/usr/bin/env python

import os
import sys
import h5py
import argparse
import logging
import warnings
import shutil
import numpy as np
import bilby
from scipy.spatial.transform import Rotation

from dsphs_jeans import light_profiles, dm_profiles

warnings.filterwarnings("ignore")

FLAGS = None
DEFAULT_INPUT_DIR = "/ocean/projects/ast200012p/tvnguyen/FIRE/particles"
DEFAULT_OUTPUT_DIR = "/ocean/projects/ast200012p/tvnguyen/FIRE/jeans_posteriors"

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
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # input output args
    parser.add_argument(
        '-i', '--input', required=True,
        help='Path to input particle data in HDF5 format')
    parser.add_argument(
        '-o', '--outdir', required=True, help='Path to output directory')
    parser.add_argument(
        '--overwrite', action='store_true', required=False,
        help='Enable to overwrite previous run')

    # FIRE args
    parser.add_argument(
        '-N', '--num-parts', required=False, type=int,
        help='Maximum number of particles to consider')
    parser.add_argument(
        '-p', '--projection', required=False, type=str.lower, default='random',
        help='Projection to consider. Default to a random projection.')
    parser.add_argument(
        '-s', '--species', type=str, default='star', required=False,
        help='Species of particle to sample kinematics')

    # likelihood args
    parser.add_argument(
        '--r-step', required=False, type=float, default=0.01,
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
        '--npoints', required=False, default=500, type=int,
        help="Number of live points for dynesty")
    parser.add_argument(
        '--npool', required=False, default=1, type=int,
        help='Number of processes')
    parser.add_argument(
        '--dlogz', required=False, default=0.1, type=float,
        help='Stopping criteria')

    return parser.parse_args()

# Read FIRE input
def read_fire(
    path, species='star', num_parts=None, num_runs=1,
    projection="random"):
    """ Read FIRE galaxy from path"""

    # read position and velocity of FIRE galaxy from path
    if not os.path.exists(path):
        path = os.path.join(DEFAULT_INPUT_DIR, path + '.hdf5')
    with h5py.File(path, 'r') as f:
        position = f[f'{species}/position'][:]
        velocity = f[f'{species}/velocity'][:]

    features = []
    fake_labels = np.zeros((num_runs, 5))   # fake labels
    for i in range(num_runs):
        # randomly select num_parts stars
        if num_parts is None:
            num_parts = len(position)
        else:
            select = np.random.permutation(len(position))[:num_parts]
            position = position[select]
            velocity = velocity[select]

        # project coordinates and calculate the projected radius
        if projection == "random":
            rot = Rotation.random().as_matrix()
            position = position @ rot.T
            velocity = velocity @ rot.T
            X = position[:, 0]
            Y = position[:, 1]
            v = velocity[:, 2]
        elif projection in ("xy", "yx"):
            X = position[:, 0]
            Y = position[:, 1]
            v = velocity[:, 2]
        elif projection in ("yz", "zy"):
            X = position[:, 1]
            Y = position[:, 2]
            v = velocity[:, 0]
        elif projection in ("zx", "xz"):
            X = position[:, 2]
            Y = position[:, 0]
            v = velocity[:, 1]
        else:
            raise ValueError(f"invalid projection {projection}")
        features.append(np.array([X, Y, v]).T)
    features = np.stack(features)

    return features, fake_labels


def main(FLAGS):
    """ Run Jeans analysis on FIRE galaxies """
    logger = set_logger()
    if not os.path.isabs(FLAGS.outdir):
        FLAGS.outdir = os.path.join(DEFAULT_OUTPUT_DIR, FLAGS.outdir)

    if (FLAGS.overwrite) and os.path.exists(FLAGS.outdir):
        logger.info('Overwrite existing directory')
        shutil.rmtree(FLAGS.outdir)

    # create output directory
    logger.info(f'writing output to {FLAGS.outdir}')
    os.makedirs(FLAGS.outdir, exist_ok=True)

    # read in projected coordinates
    cache_input = os.path.join(FLAGS.outdir, "cache.hdf5")
    if os.path.exists(cache_input):
        logger.info(f'found cache. reading from {cache_input}')
        with h5py.File(cache_input, 'r') as f:
            features = f['data/features'][:]
            fake_labels = f['data/labels'][:]
    else:
        logger.info(f'read coordinates from {FLAGS.input}')
        features, fake_labels = read_fire(
            FLAGS.input, species=FLAGS.species, num_parts=FLAGS.num_parts, num_runs=1,
            projection=FLAGS.projection)
        # write cache
        with h5py.File(cache_input, 'w') as f:
            f.attrs.update({
                'num_runs': 1,
                'num_parts': FLAGS.num_parts,
                'projection': FLAGS.projection
            })
            gr = f.create_group('data')
            gr.create_dataset('features', data=features)
            gr.create_dataset('labels', data=fake_labels)
    X, Y, v = features[0].T
    R = np.sqrt(X**2 + Y**2)

    # fit light profile with the Plummer model
    logger.info(f'fit light profile using Plummer 2D model')
    plummer_model = light_profiles.PlummerModel(R)
    plummer_model.run_sampler(
        sampler="dynesty", npoints=FLAGS.npoints,
        sample='auto', npool=FLAGS.npool, dlogz=FLAGS.dlogz,
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
        sampler="dynesty", npoints=FLAGS.npoints,
        sample='auto', npool=FLAGS.npool, dlogz=FLAGS.dlogz,
        label="jeans", outdir=FLAGS.outdir,resume=(not FLAGS.overwrite)
    )

    # print out a summary statement
    logger.info('Summary')
    for key in jeans_model.parameters:
        med = jeans_model.get_median(key)
        lo, hi = jeans_model.get_credible_intervals(key, p=0.68)
        lo = med - lo
        hi = hi - med
        logger.info(f'{key}: {med} + {hi} - {lo}')


if __name__ == "__main__":
    FLAGS = parse_cmd()
    main(FLAGS)

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from argparse import ArgumentParser
from os import path
from time import time
from scipy.ndimage import find_objects, label

from utils import trj2blocks, tolerant

# MDAnalysis
import MDAnalysis as mda

# tiydnamics
import tidynamics


def parse():
    '''Parse command line arguments.

    Returns:
        Namespace object containing input arguments.
    '''

    parser = ArgumentParser(description='MDTools: Velocity autocorrelation')
    parser.add_argument('-i', '--input', required=True, type=str,
                        help='Input .xyz trajectory')
    parser.add_argument('-v', '--velocity', required=True, type=str,
                        help='Input .xyz velocities')
    parser.add_argument('-n', '--n_cpu', required=True, type=int,
                        help='Number of CPUs for parallel processing')
    parser.add_argument('-c', '--cell_vectors', required=True, type=float,
                        help='Lattice vectors in angstroms (a, b, c)', nargs=3)
    parser.add_argument('-t', '--thresholds', required=True, type=float,
                        help='Thresholds for solvation layers', nargs=4)
    parser.add_argument('-s', '--species', required=True, type=str,
                        help='Atomic species to consider (HH, OO, OH)')

    return parser.parse_args()


def vacf(u, u_vel, t, species, block):
    '''Computes velocity-velocity autocorrelation function for given species.

    Args:
        u: MDAnalysis Universe object containing trajectory (positions).
        u_vel: MDAnalysis Universe object containing trajectory (velocities).
        t: Thresholds specifying layer boundaries.
        species: For which species (elements) to compute autocorrelation.
        block: Range of frames composing block.

    Returns:
        Velocity-velocity autocorrelation function.
    '''

    # Which species to include in velocity autocorrelation
    if species == 'HH':
        selection = 'name H'
    elif species == 'OO':
        selection = 'name O'
    elif species == 'OH':
        selection = 'name O or name H'

    # Get position and velocity trajectories of selection
    ag_trj = u.select_atoms(selection)
    ag_vel = u_vel.select_atoms(selection)
    n_atoms = ag_vel.n_atoms

    # Initialize arrays for velocities and atom heights
    velocity_array = np.zeros((len(block), n_atoms, 3))
    height_array = np.zeros((len(block), n_atoms))
    corr = []

    for i, ts in enumerate(u_vel.trajectory[block.start:block.stop]):
        print('Processing velocities %.1f%%' % (100*i/len(block)), end='\r')

        # Update positions
        u.trajectory[ts.frame]

        # Loop over atoms
        for j, vel in enumerate(ag_vel):

            # Store heights and velocities of atom
            height_array[i, j] = ag_trj.positions[j, 2]
            velocity_array[i, j, :] = vel.position

    # Loop over atoms
    for k in range(n_atoms):

        # Get frames where atoms in specified region
        z = height_array[:, k]
        z_bool = (z > t[0])*(z < t[1])+(z > t[2])*(z < t[3])

        # Select intervals where remain remains continuously in region
        contiguous_region = find_objects(label(z_bool)[0])

        # Compute dipole autocorrelation for all intervals and starting times
        for reg in contiguous_region:
            corr.append(tidynamics.acf(
                velocity_array[reg[0].start:reg[0].stop, k, :]))

    return tolerant.mean(corr)


def main():

    args = parse()
    input = args.input
    velocity = args.velocity
    n_jobs = args.n_cpu
    a, b, c = args.cell_vectors
    thresholds = args.thresholds
    species = args.species

    CURRENT_PATH = path.dirname(path.realpath(__file__))
    DATA_PATH = path.normpath(path.join(CURRENT_PATH, path.dirname(input)))
    base = path.splitext(path.basename(input))[0]

    # Initialize universe (time step 0.5 fs, both positions and velocities)
    u = mda.Universe(input, dt=5e-4)
    u_vel = mda.Universe(velocity, dt=5e-4)
    u.dimensions = np.array([a, b, c, 90, 90, 90])

    # Split trajectory into blocks
    blocks = trj2blocks.get_blocks(u, n_jobs)

    print('Analyzing...')
    results = Parallel(n_jobs=n_jobs)(delayed(vacf)(
        u, u_vel, thresholds, species, block) for block in blocks)

    # Average and normalize autocorrelations of (possibly) different length
    results = tolerant.mean(results)
    results = results/results[0]
    t = np.arange(len(results))

    # Save results as .csv
    df = pd.DataFrame({'time': t, 'autocorr': results})
    df.to_csv('%s/%s_%s_%s.vacf' % (DATA_PATH, base, species, thresholds[1]),
              index=False)

    print('\nProgram executed in %.0f seconds' % (time()-t0))


if __name__ == '__main__':
    t0 = time()
    main()

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

# tidynamics
import tidynamics


def parse():
    '''Parse command line arguments.

    Returns:
        Namespace object containing input arguments.
    '''

    # Parse command line arguments
    parser = ArgumentParser(description='MDTools: Mean squared displacement')
    parser.add_argument('-i', '--input', required=True, type=str,
                        help='Input .xyz file')
    parser.add_argument('-n', '--n_cpu', required=True, type=int,
                        help='Number of CPUs for parallel processing')
    parser.add_argument('-c', '--cell_vectors', required=True, type=float,
                        help='Lattice vectors in angstroms (a, b, c)', nargs=3)
    parser.add_argument('-t', '--thresholds', required=True, type=float,
                        help='Thresholds for solvation layers', nargs=4)

    return parser.parse_args()


def diffusion(u, t, block):
    '''Computes mean squared displacement of water molecules within specified
    layers parallel to a surface.

    Args:
        u: MDAnalysis Universe object containing trajectory.
        t: Thresholds specifying layer boundaries.
        block: Range of frames composing block.

    Returns:
        Parallel (xy) mean squared displacement.
    '''

    # Select oxygen atoms
    oxygen = u.select_atoms('name O')
    msd = []

    # Initialize 3D position array (frame, atom, dimension)
    position_array = np.zeros((len(block), oxygen.n_atoms, 3))

    for i, ts in enumerate(u.trajectory[block.start:block.stop]):
        print('Processing blocks %.1f%%' % (100*i/len(block)), end='\r')

        # Store positions of all oxygens at current frame
        position_array[i, :, :] = oxygen.positions

    # Loop over oxygen atoms
    for k in range(oxygen.n_atoms):

        # Get atoms in region
        z = position_array[:, k, 2]
        z_bool = (z > t[0])*(z < t[1])+(z > t[2])*(z < t[3])

        # Get intervals in which atom remains continuously in region
        contiguous_region = find_objects(label(z_bool)[0])

        # Compute mean squared displacement of atom for each interval
        for reg in contiguous_region:
            msd.append(tidynamics.msd(
                position_array[reg[0].start:reg[0].stop, k, :2]))

    # Return average of all MSDs
    return tolerant.mean(msd)


def main():

    args = parse()
    input = args.input
    n_jobs = args.n_cpu
    a, b, c = args.cell_vectors
    thresholds = args.thresholds

    CURRENT_PATH = path.dirname(path.realpath(__file__))
    DATA_PATH = path.normpath(path.join(CURRENT_PATH, path.dirname(input)))
    base = path.splitext(path.basename(input))[0]

    # Initialize universe (time step 0.5 fs)
    u = mda.Universe(input, dt=5e-4)
    u.add_TopologyAttr('charges')
    u.dimensions = np.array([a, b, c, 90, 90, 90])

    # Split trajectory into blocks
    blocks = trj2blocks.get_blocks(u, n_jobs)

    print('Analyzing...')
    results = Parallel(n_jobs=n_jobs)(delayed(diffusion)(
        u, thresholds, block) for block in blocks)

    # Average MSDs of (possibly) different length over all blocks
    results = tolerant.mean(results)
    t = np.arange(len(results))

    # Save results as .csv
    df = pd.DataFrame({'time': t, 'msd': results})
    df.to_csv('%s/%s_%s.msd' % (
        DATA_PATH, base, thresholds[1]), index=False)

    print('\nProgram executed in %.0f seconds' % (time()-t0))


if __name__ == '__main__':
    t0 = time()
    main()

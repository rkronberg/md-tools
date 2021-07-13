import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from argparse import ArgumentParser
from os import path
from time import time

from utils import trj2blocks

# MDAnalysis
import MDAnalysis as mda
from MDAnalysis.analysis.distances import distance_array


def parse():
    '''Parse command line arguments.

    Returns:
        Namespace object containing input arguments.
    '''

    # Parse command line arguments
    parser = ArgumentParser(description='MDTools: OOO angle distribution')
    parser.add_argument('-i', '--input', required=True, type=str,
                        help='Input .xyz file')
    parser.add_argument('-n', '--n_cpu', required=True, type=int,
                        help='Number of CPUs for parallel processing')
    parser.add_argument('-c', '--cell_vectors', required=True, type=float,
                        help='Lattice vectors in angstroms (a, b, c)', nargs=3)
    parser.add_argument('-nb', '--n_bins', required=True, type=int,
                        help='Number of bins')
    parser.add_argument('-t', '--thresholds', required=True, type=float,
                        help='Thresholds for contact layers', nargs=4)

    return parser.parse_args()


def triplet(u, a, t, block):
    '''Computes OOO triplet angular distribution.

    Args:
        u: MDAnalysis Universe object containing trajectory.
        a: Lateral lattice vector assuming equal lateral dimensions
        t: Thresholds specifying layer boundaries.
        block: Range of frames composing block.

    Returns:
        Triplet angles.
    '''

    # Get oxygens
    oxygen = u.select_atoms('name O')

    # Initialize OO distance array
    rOO = np.zeros((len(oxygen), len(oxygen)))
    cutoff = 3.3
    triplet = []

    for i, ts in enumerate(u.trajectory[block.start:block.stop]):
        print('Processing blocks %.1f%%' % (100*i/len(block)), end='\r')

        # Wrap oxygen to box, compute OO distance array
        oxygen.wrap(box=u.dimensions)
        distance_array(oxygen.positions, oxygen.positions,
                       box=u.dimensions, result=rOO)

        # Loop over oxygen atoms
        for j, pos in enumerate(oxygen.positions):
            if (pos[2] > t[0])*(pos[2] < t[1])+(pos[2] > t[2])*(pos[2] < t[3]):

                # For each oxygen get oxygens within cutoff (exclude self)
                shell = oxygen.positions[((rOO < cutoff) & (rOO > 0))[j]]

                # Skip atom if less than 2 neighbors
                if shell.shape[0] < 2:
                    continue

                # Combine if split across boundary
                shell[shell-pos < -cutoff] += a
                shell[shell-pos > cutoff] -= a

                # Compute OO vectors and triplet angles
                vect_OO = shell-pos
                unit_OO = vect_OO/np.linalg.norm(vect_OO, axis=1)[:, None]
                cosine = np.dot(unit_OO, unit_OO.T)
                triplet += list(cosine[np.triu_indices_from(cosine, k=1)])

    return triplet


def main():

    args = parse()
    input = args.input
    n_jobs = args.n_cpu
    n_bins = args.n_bins
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
    results = Parallel(n_jobs=n_jobs)(delayed(triplet)(
        u, a, thresholds, block) for block in blocks)

    # Compute grid
    grid = np.linspace(0, 180, n_bins)

    # Concatenate results, compute histogram
    results = np.concatenate(results)
    hist, _ = np.histogram(np.rad2deg(np.arccos(results)), bins=n_bins,
                           range=(0, 180), density=True)

    # Save results as .csv
    df = pd.DataFrame({'angle': grid, 'density': hist})
    df.to_csv('%s/%s_%s.triplet' % (DATA_PATH, base, thresholds[1]),
              index=False)

    print('\nProgram executed in %.0f seconds' % (time()-t0))


if __name__ == '__main__':
    t0 = time()
    main()

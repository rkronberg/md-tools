import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from argparse import ArgumentParser
from os import path
from time import time

import trj2blocks

# MDAnalysis
import MDAnalysis as mda
from MDAnalysis.analysis.distances import distance_array


def parse():
    '''Parse command line arguments.

    Returns:
        Namespace object containing input arguments.
    '''

    parser = ArgumentParser(description='MDTools: Local structure index')
    parser.add_argument('-i', '--input', required=True, type=str,
                        help='Input .xyz file')
    parser.add_argument('-n', '--n_cpu', required=True, type=int,
                        help='Number of CPUs for parallel processing')
    parser.add_argument('-c', '--cell_vectors', required=True, type=float,
                        help='Lattice vectors in angstroms (a, b, c)', nargs=3)
    parser.add_argument('-nb', '--n_bins', required=True, type=int,
                        help='Number of bins')

    return parser.parse_args()


def lsi(u, block):
    '''Computes local structure index (LSI).

    Args:
        u: MDAnalysis Universe object containing trajectory.
        block: Range of frames composing block.

    Returns:
        Local structure index and heights of each oxygen.
    '''

    # Select oxygen atoms
    oxygen = u.select_atoms('name O')

    # Initialize OO distance array
    rOO = np.zeros((len(oxygen), len(oxygen)))
    lsindex = []
    height = []

    for i, ts in enumerate(u.trajectory[block.start:block.stop]):
        print('Processing blocks %.1f%%' % (100*i/len(block)), end='\r')

        # Compute OO distance array
        distance_array(oxygen.positions, oxygen.positions,
                       box=u.dimensions, result=rOO)

        # Loop over oxygen atoms
        for j, pos in enumerate(oxygen.positions):

            # Sort OO distance
            r = np.sort(rOO[j])

            # Consider all OO distances less than 3.7 angstrom
            delta = r[np.roll((r > 0)*(r < 3.7), 1)]-r[(r > 0)*(r < 3.7)]

            # Get mean and evaluate LSI as mean of squared differences to mean
            ave = np.mean(delta)
            lsindex.append(np.sum((delta-ave)**2)/len(delta))

            # Store height of oxygen
            height.append(pos[2])

    return np.vstack((lsindex, height)).T


def main():

    args = parse()
    input = args.input
    n_jobs = args.n_cpu
    n_bins = args.n_bins
    a, b, c = args.cell_vectors

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
    results = Parallel(n_jobs=n_jobs)(delayed(lsi)(
        u, block) for block in blocks)

    # Concatenate results
    results = np.concatenate(results)

    # Compute 2D histogram (heights vs LSI)
    hist2d, _, _ = np.histogram2d(results[:, 0], results[:, 1],
                                  bins=(n_bins, 2*n_bins), density=True,
                                  range=[[0, 0.3], [9.85, 42.65]])

    # Save results as .csv
    df = pd.DataFrame(hist2d)
    df.to_csv('%s/%s.lsi' % (DATA_PATH, base), index=False)

    print('\nProgram executed in %.0f seconds' % (time()-t0))


if __name__ == '__main__':
    t0 = time()
    main()

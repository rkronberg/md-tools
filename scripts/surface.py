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

    # Parse command line arguments
    parser = ArgumentParser(description='Molecular dynamics density tools')
    parser.add_argument('-i', '--input', required=True, type=str,
                        help='Input .xyz file')
    parser.add_argument('-n', '--n_cpu', required=True, type=int,
                        help='Number of CPUs for parallel processing')
    parser.add_argument('-c', '--cell_vectors', required=True, type=float,
                        help='Lattice vectors in angstroms (a, b, c)', nargs=3)
    parser.add_argument('-nb', '--n_bins', required=True, type=int,
                        help='Number of bins')
    parser.add_argument('-t', '--thresholds', type=float, nargs=2,
                        help='Thresholds for contact layers')

    return parser.parse_args()


def surface(u, oxygen, slab, t_down, t_up, block):

    xy = []

    for i, ts in enumerate(u.trajectory[block.start:block.stop]):
        print('Processing blocks %.1f%%' % (100*i/len(block)), end='\r')
        o_down = oxygen.positions[oxygen.positions[:, 2] < t_down]
        o_up = oxygen.positions[oxygen.positions[:, 2] > t_up]

        # Shift top molecules due to asymmetry by half a lattice constant
        o_up[:, 0] += 2.81

        # Shift all molecules to account for slab centroid drift
        shift = slab.centroid()[:-1]
        xy.append(np.vstack((o_down[:, :-1]-shift, o_up[:, :-1]-shift)))

    return np.concatenate(xy)


def main():

    args = parse()
    input = args.input
    n_jobs = args.n_cpu
    n_bins = args.n_bins
    a, b, c = args.cell_vectors
    t_down, t_up = args.thresholds

    CURRENT_PATH = path.dirname(path.realpath(__file__))
    DATA_PATH = path.normpath(path.join(CURRENT_PATH, path.dirname(input)))
    base = path.splitext(path.basename(input))[0]

    u = mda.Universe(input, dt=5e-4)
    u.add_TopologyAttr('charges')
    u.dimensions = np.array([a, b, c, 90, 90, 90])
    oxygen = u.select_atoms('name O')
    slab = u.select_atoms('name Na or name Cl')

    blocks = trj2blocks.get_blocks(u, n_jobs)

    print('Analyzing...')
    results = Parallel(n_jobs=n_jobs)(delayed(surface)(
        u, oxygen, slab, t_down, t_up, block) for block in blocks)

    results = np.concatenate(results)

    # Assuming square lateral cell dimensions
    results[results > a] -= a
    results[results < 0] += a

    hist, _, _ = np.histogram2d(results[:, 0], results[:, 1], bins=n_bins,
                                range=[[0, a]]*2, density=True)

    df = pd.DataFrame(hist)
    df.to_csv('%s/%s.sdist' % (DATA_PATH, base), index=False)

    print('\nProgram executed in %.0f seconds' % (time()-t0))


if __name__ == '__main__':
    t0 = time()
    main()

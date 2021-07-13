import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from argparse import ArgumentParser
from os import path
from time import time

from utils import trj2blocks

# MDAnalysis
import MDAnalysis as mda
from MDAnalysis.analysis.hydrogenbonds import hbond_analysis


def parse():
    '''Parse command line arguments.

    Returns:
        Namespace object containing input arguments.
    '''

    parser = ArgumentParser(description='MDTools: Hydrogen bond analysis')
    parser.add_argument('-i', '--input', required=True, type=str,
                        help='Input .xyz file')
    parser.add_argument('-n', '--n_cpu', required=True, type=int,
                        help='Number of CPUs for parallel processing')
    parser.add_argument('-c', '--cell_vectors', required=True, type=float,
                        help='Lattice vectors in angstroms (a, b, c)', nargs=3)

    return parser.parse_args()


def hbonds(u, block):
    '''Computes hydrogen bond (HB) statistics.

    Args:
        u: MDAnalysis Universe object containing trajectory.
        block: Range of frames composing block.

    Returns:
        Accepted and donated hydrogen bond counts and surface separations
    '''

    # Initialize hydrogen bond analysis
    hbonds = hbond_analysis.HydrogenBondAnalysis(
        u, d_h_a_angle_cutoff=135, d_a_cutoff=3.5)
    hbonds.donors_sel = 'name O'
    hbonds.acceptors_sel = 'name O'
    hbonds.hydrogens_sel = 'name H'

    # Run hydrogen bond analysis
    hbonds.run(start=block.start, stop=block.stop, verbose=True)
    out = hbonds.results.hbonds

    # Select oxygen atoms, initialize output arrays
    oxygen = u.select_atoms('name O')
    acc_counts = np.zeros((len(block), oxygen.n_atoms))
    don_counts = np.zeros((len(block), oxygen.n_atoms))
    heights = np.zeros((len(block), oxygen.n_atoms))

    for i, ts in enumerate(u.trajectory[block.start:block.stop]):
        print('Processing blocks %.1f%%' % (100*i/len(block)), end='\r')

        # Get all HBs of current frame
        step = out[(out[:, 0] == ts.frame)]

        # Loop over each oxygen
        for j, idx in enumerate(oxygen.indices):

            # Get number of accepted and donated HBs + position along z
            acc_counts[i, j] = len(step[(step[:, 1] == idx)])
            don_counts[i, j] = len(step[(step[:, 3] == idx)])
            heights[i, j] = oxygen[j].position[2]

    return np.stack((heights, acc_counts, don_counts))


def main():

    args = parse()
    input = args.input
    n_jobs = args.n_cpu
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
    results = Parallel(n_jobs=n_jobs)(delayed(hbonds)(
        u, block) for block in blocks)

    # Concatenate results
    results = np.concatenate(results, axis=1)

    # Save results (heights, accepted HBs, donated HBs) as .csv
    df1 = pd.DataFrame(results[0])
    df2 = pd.DataFrame(results[1])
    df3 = pd.DataFrame(results[2])
    df1.to_csv('%s/%s.heights' % (DATA_PATH, base), index=False)
    df2.to_csv('%s/%s.acc_hb' % (DATA_PATH, base), index=False)
    df3.to_csv('%s/%s.don_hb' % (DATA_PATH, base), index=False)

    print('\nProgram executed in %.0f seconds' % (time()-t0))


if __name__ == '__main__':
    t0 = time()
    main()

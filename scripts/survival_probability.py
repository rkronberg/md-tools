import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from argparse import ArgumentParser
from os import path
from time import time

from utils import trj2blocks, tolerant

# MDAnalysis
import MDAnalysis as mda


def parse():
    '''Parse command line arguments.

    Returns:
        Namespace object containing input arguments.
    '''

    parser = ArgumentParser(description='MDTools: Water survival probability')
    parser.add_argument('-i', '--input', required=True, type=str,
                        help='Input .xyz file')
    parser.add_argument('-n', '--n_cpu', required=True, type=int,
                        help='Number of CPUs for parallel processing')
    parser.add_argument('-c', '--cell_vectors', required=True, type=float,
                        help='Lattice vectors in angstroms (a, b, c)', nargs=3)
    parser.add_argument('-t', '--thresholds', required=True, type=float,
                        help='Thresholds for contact layers', nargs=4)

    return parser.parse_args()


def survival(u, t, block):
    '''Computes survival probability (SP) of water molecules within specified
    layers.

    Args:
        u: MDAnalysis Universe object containing trajectory.
        t: Thresholds specifying layer boundaries.
        block: Range of frames composing block.

    Returns:
        Survival probability (SP).
    '''

    size = len(block)

    # Select oxygens
    atoms = u.select_atoms('name O')

    # Initialize array of surviving molecules (uint8 to alleviate memory reqs.)
    atoms_in_reg = np.zeros((size, atoms.n_atoms), dtype=np.uint8)

    for i, _ in enumerate(u.trajectory[block.start:block.stop]):
        print('Processing blocks %.1f%%' % (100*i/size), end='\r')

        # Get atoms within layers at current frame and set indices to 1
        z = atoms.positions[:, 2]
        z_bool = (z > t[0])*(z < t[1])+(z > t[2])*(z < t[3])
        atoms_in_reg[i, z_bool] = 1

    # Drop all-zero columns (water not in layer at any point)
    atoms_in_reg = atoms_in_reg[:, ~np.all(atoms_in_reg == 0, axis=0)]

    # Loop over all starting times (less samples towards the end)
    for i in range(size):
        print('Processing blocks %.1f%%' % (100*i/size), end='\r')

        # Only consider atoms continuously in region (e.g. 1110011 -> 11100000)
        cont_in_reg = np.cumprod(atoms_in_reg[i:], axis=0, dtype=np.uint8)

        # Compute fraction wrt. initial number of atoms in region
        frac_in_reg = np.sum(cont_in_reg, axis=1)/np.sum(atoms_in_reg[i])
        if i == 0:
            sp = frac_in_reg
        else:
            sp[:-i] += frac_in_reg

    # Return SP appropriately averaged over the decreasing number of samples
    return sp/np.arange(1, size+1)[::-1]


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
    results = Parallel(n_jobs=n_jobs)(delayed(survival)(
        u, thresholds, block) for block in blocks)

    # Average over blocks of (possibly) different length
    results = tolerant.mean(results)
    t = np.arange(len(results))

    # Save results as .csv
    df = pd.DataFrame({'time': t, 'autocorr': results})
    df.to_csv('%s/%s_%s.sprob' % (
        DATA_PATH, base, thresholds[1]), index=False)

    print('\nProgram executed in %.0f seconds' % (time()-t0))


if __name__ == '__main__':
    t0 = time()
    main()

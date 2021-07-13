import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from argparse import ArgumentParser
from os import path
from time import time

from utils import trj2blocks, tolerant

# MDAnalysis
import MDAnalysis as mda
from MDAnalysis.analysis.hydrogenbonds import hbond_analysis


def parse():
    '''Parse command line arguments.

    Returns:
        Namespace object containing input arguments.
    '''

    parser = ArgumentParser(description='MDTools: Hydrogen bond lifetime')
    parser.add_argument('-i', '--input', required=True, type=str,
                        help='Input .xyz file')
    parser.add_argument('-n', '--n_cpu', required=True, type=int,
                        help='Number of CPUs for parallel processing')
    parser.add_argument('-c', '--cell_vectors', required=True, type=float,
                        help='Lattice vectors in angstroms (a, b, c)', nargs=3)
    parser.add_argument('-t', '--thresholds', required=True, type=float,
                        help='Thresholds for contact layers', nargs=4)

    return parser.parse_args()


def hbonds(u, t, block):
    '''Computes hydrogen bond (HB) relaxation.

    Args:
        u: MDAnalysis Universe object containing trajectory.
        t: Thresholds specifying layer boundaries.
        block: Range of frames composing block.

    Returns:
        Hydrogen bond autocorrelation function.
    '''

    size = len(block)

    # Select oxygen and hydrogen atoms
    oxygen = u.select_atoms('name O')
    hydrogen = u.select_atoms('name H')

    # Initialize hydrogen bond analysis
    hbonds = hbond_analysis.HydrogenBondAnalysis(
        u, d_h_a_angle_cutoff=135, d_a_cutoff=3.5)
    hbonds.donors_sel = 'name O'
    hbonds.acceptors_sel = 'name O'
    hbonds.hydrogens_sel = 'name H'

    # Get hydrogen and oxygen indices
    hidx = hydrogen.indices
    oidx = oxygen.indices

    # Run HB analysis
    hbonds.run(start=block.start, stop=block.stop, verbose=True)
    out = hbonds.results.hbonds

    # Initialize array of HBs (uint8 to reduce memory reqs)
    pairs_in_reg = np.zeros((size, hydrogen.n_atoms, oxygen.n_atoms),
                            dtype=np.uint8)

    for i, ts in enumerate(u.trajectory[block.start:block.stop]):
        print('Processing blocks %.1f%%' % (100*i/size), end='\r')

        # Get all HBs at current frame
        ha_pairs = out[out[:, 0] == ts.frame][:, 2:4]

        # Get indices of atoms participating in HBs
        ha_idx = np.stack((np.searchsorted(hidx, ha_pairs[:, 0]),
                           np.searchsorted(oidx, ha_pairs[:, 1])))

        # For each hydrogen bond, set appropriate pair to 1 (True)
        pairs_in_reg[i][tuple(ha_idx)] = 1

        # Reset HBs not in region to zero
        z = oxygen.positions[:, 2]
        z_bool = (z > t[0])*(z < t[1])+(z > t[2])*(z < t[3])
        pairs_in_reg[i, :, ~z_bool] = 0

    # Remove all-zero columns (no HBs or pair not in region at any point)
    pairs_in_reg = pairs_in_reg[:, :, ~np.all(pairs_in_reg == 0, axis=(0, 1))]
    pairs_in_reg = pairs_in_reg[:, ~np.all(pairs_in_reg == 0, axis=(0, 2)), :]

    # Loop over all starting times
    for i in range(size):
        print('Processing blocks %.1f%%' % (100*i/size), end='\r')

        # Only consider HBs continuously in region (e.g. 1110011 -> 11100000)
        cont_in_reg = np.cumprod(pairs_in_reg[i:], axis=0, dtype=np.uint8)

        # Compute fraction wrt. initial number of HBs in region
        frac_in_reg = np.sum(cont_in_reg, axis=(1, 2))/np.sum(pairs_in_reg[i])
        if i == 0:
            lifetime = frac_in_reg
        else:
            lifetime[:-i] += frac_in_reg

    # Return lifetime appropriately averaged over decreasing number of samples
    return lifetime/np.arange(1, size+1)[::-1]


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
    results = Parallel(n_jobs=n_jobs)(delayed(hbonds)(
        u, thresholds, block) for block in blocks)

    # Average autocorrelation functions of (possibly) different length
    results = tolerant.mean(results)
    t = np.arange(len(results))

    # Save results as .csv
    df = pd.DataFrame({'time': t, 'autocorr': results})
    df.to_csv('%s/%s_%s.hblifetime' % (
        DATA_PATH, base, thresholds[1]), index=False)

    print('\nProgram executed in %.0f seconds' % (time()-t0))


if __name__ == '__main__':
    t0 = time()
    main()

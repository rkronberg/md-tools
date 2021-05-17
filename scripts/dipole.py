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


def dipole(u, a, oxygen, hydrogen, t_down, t_up, block):

    rOH = np.zeros((len(oxygen), len(hydrogen)))
    cos_theta = []

    for i, ts in enumerate(u.trajectory[block.start:block.stop]):
        print('Processing blocks %.1f%%' % (100*i/len(block)), end='\r')
        oxygen.wrap(box=u.dimensions)
        hydrogen.wrap(box=u.dimensions)
        distance_array(oxygen.positions, hydrogen.positions,
                       box=u.dimensions, result=rOH)

        # Loop over oxygen atoms
        for j, pos in enumerate(oxygen.positions):
            if pos[2] < t_down:

                # Get bound hydrogens
                hbound = hydrogen.positions[(rOH < 1.2)[j]]

                # Exclude ions
                if len(hbound) != 2:
                    continue

                # Combine broken water molecules
                hbound[hbound-pos < -1.2] += a
                hbound[hbound-pos > 1.2] -= a

                # Compute dipole vector
                dip = np.mean(hbound, axis=0)-pos
                unit_dip = dip/np.linalg.norm(dip)
                cos_theta.append(unit_dip[2])

            elif pos[2] > t_up:

                # Get bound hydrogens
                hbound = hydrogen.positions[(rOH < 1.2)[j]]

                # Exclude ions
                if len(hbound) != 2:
                    continue

                # Combine broken water molecules
                hbound[hbound-pos < -1.2] += a
                hbound[hbound-pos > 1.2] -= a

                # Compute dipole vector
                dip = np.mean(hbound, axis=0)-pos
                unit_dip = dip/np.linalg.norm(dip)
                cos_theta.append(-unit_dip[2])

            else:
                continue

    return np.array(cos_theta)


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
    hydrogen = u.select_atoms('name H')

    blocks = trj2blocks.get_blocks(u, n_jobs)

    cosine = np.linspace(-1, 1, n_bins)

    print('Analyzing...')
    results = Parallel(n_jobs=n_jobs)(delayed(dipole)(
        u, a, oxygen, hydrogen, t_down, t_up, block) for block in blocks)

    results = np.concatenate(results).ravel()
    hist, _ = np.histogram(results, bins=n_bins, range=(-1, 1), density=True)

    df = pd.DataFrame({'cosine': cosine, 'density': hist})
    df.to_csv('%s/%s.adist' % (DATA_PATH, base), index=False)

    print('\nProgram executed in %.0f seconds' % (time()-t0))


if __name__ == '__main__':
    t0 = time()
    main()

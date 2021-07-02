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

    parser = ArgumentParser(description='MDTools: Water self-ion distribution')
    parser.add_argument('-i', '--input', required=True, type=str,
                        help='Input .xyz file')
    parser.add_argument('-n', '--n_cpu', required=True, type=int,
                        help='Number of CPUs for parallel processing')
    parser.add_argument('-c', '--cell_vectors', required=True, type=float,
                        help='Lattice vectors in angstroms (a, b, c)', nargs=3)
    parser.add_argument('-b', '--bin_size', required=True, type=float,
                        help='Bin width in angstroms')
    parser.add_argument('-s', '--species', required=True, type=str,
                        help='Type of ionic species (H3O+ or OH-)')
    parser.add_argument('-ns', '--n_species', required=True, type=int,
                        help='Number of ionic species')

    return parser.parse_args()


def sort_acid(CN, n_species):
    '''Return indices of oxygens with the largest hydrogen coordination
    numbers (hydronium ions)

    Args:
        CN: Generalized (continuous) coordination numbers
        n_species: Expected number of ions in the solution

    Returns:
        Indices of the hydronium oxygens
    '''

    return np.argpartition(CN, -n_species)[-n_species:]


def sort_base(CN, n_species):
    '''Return indices of oxygens with the smallest hydrogen coordination
    numbers (hydroxide ions)

    Args:
        CN: Generalized (continuous) coordination numbers
        n_species: Expected number of ions in the solution

    Returns:
        Indices of the hydroxide oxygens
    '''

    return np.argpartition(CN, n_species)[:n_species]


def ions(u, n_species, func, block):
    '''Computes distribution of water self-ions perpendicular to a surface.

    Args:
        u: MDAnalysis Universe object containing trajectory.
        n_species: Expected number of ions in the solution
        func: Function to sort for hydronium or hydroxide
        block: Range of frames composing block.

    Returns:
        Distribution of ions perpendicular to surface.
    '''

    # Select oxygen and hydrogen atoms
    oxygen = u.select_atoms('name O')
    hydrogen = u.select_atoms('name H')

    # Cutoff for coordination number switching function
    rc = 1.32

    # Initialize ion heights and OH distance array
    z_ion = np.zeros((len(block), n_species))
    rOH = np.zeros((len(oxygen), len(hydrogen)))

    for i, ts in enumerate(u.trajectory[block.start:block.stop]):
        print('Processing blocks %.1f%%' % (100*i/len(block)), end='\r')

        # Compute OH distance array and continuous coordination numbers
        distance_array(oxygen.positions, hydrogen.positions,
                       box=u.dimensions, result=rOH)
        CN = np.sum((1-(rOH/rc)**16)/(1-(rOH/rc)**56), axis=1)

        # Get hydronium/hydroxide indices
        ind = func(CN, n_species)

        # Store ion heights
        z_ion[i, :] = oxygen.positions[ind][:, 2]

    return z_ion


def main():

    args = parse()
    input = args.input
    n_jobs = args.n_cpu
    binsize = args.bin_size
    species = args.species
    n_species = args.n_species
    a, b, c = args.cell_vectors

    CURRENT_PATH = path.dirname(path.realpath(__file__))
    DATA_PATH = path.normpath(path.join(CURRENT_PATH, path.dirname(input)))
    base = path.splitext(path.basename(input))[0]

    # Initialize universe (time step 0.5 fs)
    u = mda.Universe(input, dt=5e-4)
    u.add_TopologyAttr('charges')
    u.dimensions = np.array([a, b, c, 90, 90, 90])

    # Select appropriate sorting function based on input
    if species == 'H3O+':
        func = sort_acid
    elif species == 'OH-':
        func = sort_base

    # Split trajectory into blocks
    blocks = trj2blocks.get_blocks(u, n_jobs)

    # Compute number of bins, grid
    nbins = int(c//binsize)
    z = np.linspace(binsize, c-binsize, nbins)

    print('Analyzing...')
    results = Parallel(n_jobs=n_jobs)(delayed(ions)(
        u, n_species, func, block) for block in blocks)

    # Concatenate and ravel results, compute histogram (probability density)
    results = np.concatenate(results).ravel()
    hist, _ = np.histogram(results, bins=nbins, range=(0.0, c), density=True)

    # Save results as .csv
    df = pd.DataFrame({'z-coord': z, 'density': hist})
    df.to_csv('%s/%s.ions' % (DATA_PATH, base), index=False)

    print('\nProgram executed in %.0f seconds' % (time()-t0))


if __name__ == '__main__':
    t0 = time()
    main()

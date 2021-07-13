import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from argparse import ArgumentParser
from os import path
from time import time

from utils import trj2blocks

# MDAnalysis
import MDAnalysis as mda
from MDAnalysis.analysis.lineardensity import LinearDensity


def parse():
    '''Parse command line arguments.

    Returns:
        Namespace object containing input arguments.
    '''

    parser = ArgumentParser(description='MDTools: Linear water density')
    parser.add_argument('-i', '--input', required=True, type=str,
                        help='Input .xyz file')
    parser.add_argument('-n', '--n_cpu', required=True, type=int,
                        help='Number of CPUs for parallel processing')
    parser.add_argument('-c', '--cell_vectors', required=True, type=float,
                        help='Lattice vectors in angstroms (a, b, c)', nargs=3)
    parser.add_argument('-b', '--bin_size', required=True, type=float,
                        help='Bin width in angstroms')

    return parser.parse_args()


def density(u, binsize, block):
    '''Computes water density profile perpendicular to a surface.

    Args:
        u: MDAnalysis Universe object containing trajectory.
        binsize: Bin size of profile (histogram).
        block: Range of frames composing block.

    Returns:
        Linear density profile.
    '''

    # Select water
    water = u.select_atoms('name O or name H')

    # Initialize and run linear density analysis
    ldens = LinearDensity(water, binsize=binsize, verbose=True)
    ldens.run(start=block.start, stop=block.stop)

    return ldens.results.z.pos


def main():

    args = parse()
    input = args.input
    n_jobs = args.n_cpu
    binsize = args.bin_size
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

    # Compute number of bins, grid
    nbins = int(c//binsize)
    grid = np.linspace(binsize, c-binsize, nbins)

    print('Analyzing...')
    results = Parallel(n_jobs=n_jobs)(delayed(density)(
        u, binsize, block) for block in blocks)

    # Save results as .csv
    df = pd.DataFrame({'z-coord': grid, 'density': np.mean(results, axis=0)})
    df.to_csv('%s/%s.ldens' % (DATA_PATH, base), index=False)

    print('\nProgram executed in %.0f seconds' % (time()-t0))


if __name__ == '__main__':
    t0 = time()
    main()

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from argparse import ArgumentParser
from os import path
from time import time

from . import trj2blocks

# MDAnalysis
import MDAnalysis as mda
from MDAnalysis.analysis.lineardensity import LinearDensity


def parse():

    # Parse command line arguments
    parser = ArgumentParser(description='Molecular dynamics density tools')
    parser.add_argument('-i', '--input', required=True, type=str,
                        help='Input .xyz file')
    parser.add_argument('-n', '--n_cpu', required=True, type=int,
                        help='Number of CPUs for parallel processing')
    parser.add_argument('-c', '--cell_vectors', required=True, type=float,
                        help='Lattice vectors in angstroms (a, b, c)', nargs=3)
    parser.add_argument('-b', '--bin_size', required=True, type=float,
                        help='Bin width in angstroms')
    parser.add_argument('-f', '--n_frames', type=int,
                        help='Total number of frames')

    return parser.parse_args()


def density(atomgroup, binsize, block):

    ldens = LinearDensity(atomgroup, binsize=binsize, verbose=True)
    ldens.run(start=block.start, stop=block.stop)

    return ldens.results.z.pos


def main():

    args = parse()
    input = args.input
    n_jobs = args.n_cpu
    binsize = args.bin_size
    n_frames = args.n_frames
    a, b, c = args.cell_vectors

    CURRENT_PATH = path.dirname(path.realpath(__file__))
    DATA_PATH = path.normpath(path.join(CURRENT_PATH, path.dirname(input)))
    base = path.splitext(path.basename(input))[0]

    u = mda.Universe(input, dt=5e-4)
    u.add_TopologyAttr('charges')
    u.dimensions = np.array([a, b, c, 90, 90, 90])
    water = u.select_atoms('name O or name H')

    blocks = trj2blocks.get_blocks(u, n_jobs)

    nbins = int(c//binsize)
    z = np.linspace(binsize, c-binsize, nbins)

    print('Analyzing...')
    results = Parallel(n_jobs=n_jobs)(delayed(density)(
        water, binsize, block) for block in blocks)

    df = pd.DataFrame({'z-coord': z, 'density': np.mean(results, axis=0)})
    df.to_csv('%s/%s.ldens' % (DATA_PATH, base), index=False)

    print('\nProgram executed in %.0f seconds' % (time()-t0))


if __name__ == '__main__':
    t0 = time()
    main()

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from argparse import ArgumentParser
from os import path
from time import time

from . import trj2blocks

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
    parser.add_argument('-b', '--bin_size', required=True, type=float,
                        help='Bin width in angstroms')
    parser.add_argument('-s', '--species', required=True, type=str,
                        help='Type of ionic species (H3O+ or OH-)')
    parser.add_argument('-ns', '--n_species', required=True, type=int,
                        help='Number of ionic species')
    parser.add_argument('-nf', '--n_frames', type=int,
                        help='Total number of frames')

    return parser.parse_args()


def sort_acid(CN, n_species):

    return np.argpartition(CN, -n_species)[-n_species:]


def sort_base(CN, n_species):

    return np.argpartition(CN, n_species)[:n_species]


def ions(u, oxygen, hydrogen, n_species, func, block):

    rc = 1.32
    z_ion = np.zeros((len(block), n_species))
    rOH = np.zeros((len(oxygen), len(hydrogen)))

    for i, ts in enumerate(u.trajectory[block.start:block.stop]):
        print('Processing blocks %.1f%%' % (100*i/len(block)), end='\r')
        distance_array(oxygen.positions, hydrogen.positions,
                       box=u.dimensions, result=rOH)
        CN = np.sum((1-(rOH/rc)**16)/(1-(rOH/rc)**56), axis=1)
        ind = func(CN, n_species)
        print(CN[ind])
        z_ion[i, :] = oxygen.positions[ind][:, 2]

    return z_ion


def main():

    args = parse()
    input = args.input
    n_jobs = args.n_cpu
    binsize = args.bin_size
    n_frames = args.n_frames
    species = args.species
    n_species = args.n_species
    a, b, c = args.cell_vectors

    CURRENT_PATH = path.dirname(path.realpath(__file__))
    DATA_PATH = path.normpath(path.join(CURRENT_PATH, path.dirname(input)))
    base = path.splitext(path.basename(input))[0]

    u = mda.Universe(input, dt=5e-4)
    u.add_TopologyAttr('charges')
    u.dimensions = np.array([a, b, c, 90, 90, 90])
    oxygen = u.select_atoms('name O')
    hydrogen = u.select_atoms('name H')

    if species == 'H3O+':
        func = sort_acid
    elif species == 'OH-':
        func = sort_base

    blocks = trj2blocks.get_blocks(u, n_jobs)

    nbins = int(c//binsize)
    z = np.linspace(binsize, c-binsize, nbins)

    print('Analyzing...')
    results = Parallel(n_jobs=n_jobs)(delayed(ions)(
        u, oxygen, hydrogen, n_species, func, block) for block in blocks)

    results = np.concatenate(results).ravel()
    hist, _ = np.histogram(results, bins=nbins, range=(0.0, c), density=True)

    df = pd.DataFrame({'z-coord': z, 'density': hist})
    df.to_csv('%s/%s.ions' % (DATA_PATH, base), index=False)

    print('\nProgram executed in %.0f seconds' % (time()-t0))


if __name__ == '__main__':
    t0 = time()
    main()

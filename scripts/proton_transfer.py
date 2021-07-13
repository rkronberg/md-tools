import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from argparse import ArgumentParser
from os import path
from time import time

from utils import trj2blocks

# MDAnalysis
import MDAnalysis as mda
from MDAnalysis.analysis.distances import distance_array


def parse():
    '''Parse command line arguments.

    Returns:
        Namespace object containing input arguments.
    '''

    parser = ArgumentParser(description='MDTools: Proton transfer analysis')
    parser.add_argument('-i', '--input', required=True, type=str,
                        help='Input .xyz file')
    parser.add_argument('-n', '--n_cpu', required=True, type=int,
                        help='Number of CPUs for parallel processing')
    parser.add_argument('-c', '--cell_vectors', required=True, type=float,
                        help='Lattice vectors in angstroms (a, b, c)', nargs=3)
    parser.add_argument('-nb', '--n_bins', required=True, type=int,
                        help='Number of bins')
    parser.add_argument('-ns', '--n_species', required=True, type=int,
                        help='Number of ionic species')

    return parser.parse_args()


def proton_transfer(u, n_ion, block):
    '''Computes proton transfer (PT) coordinate, OO distance and coordination
    number (CN) for the n_ion most active charge defects in water

    Args:
        u: MDAnalysis Universe object containing trajectory.
        n_ion: Expected number of ions in the solution.
        block: Range of frames composing block.

    Returns:
        Array of PT coordinates, OO-distances and (acceptor) CNs
    '''

    # Select oxygen and hydrogen atoms
    oxygen = u.select_atoms('name O')
    hydrogen = u.select_atoms('name H')

    # Initialize HO, OO distance arrays, output array
    rHO = np.zeros((len(hydrogen), len(oxygen)))
    rOO = np.zeros((len(oxygen), len(oxygen)))
    out = np.zeros((n_ion*len(block), 4))

    for i, ts in enumerate(u.trajectory[block.start:block.stop]):
        print('Processing blocks %.1f%%' % (100*i/len(block)), end='\r')

        # Initialize mask for the relevant HO distances
        mask = np.zeros(rHO.shape, dtype=bool)

        # Compute HO distance array
        distance_array(hydrogen.positions, oxygen.positions,
                       box=u.dimensions, result=rHO)

        # Find two oxygens closest to each hydrogen, set to True in mask
        ind = np.argpartition(rHO, 2, axis=1)[:, :2]
        mask[np.arange(ind.shape[0])[:, None], ind] = True

        # Get HO distance from each hydrogen to two nearest oxygens
        rmin = rHO[mask].reshape((len(hydrogen), n_ion))

        # Compute proton transfer coordinate for all hydrogens
        delta = rmin[:, 0]-rmin[:, 1]

        # Select two smallest delta (most active protons)
        idx = np.argpartition(abs(delta), n_ion)[:n_ion]

        # Select donor and acceptor oxygens of the most active protons
        oidx = np.argpartition(rHO[idx], n_ion, axis=1)[:, :n_ion]

        # Compute OO distance array
        distance_array(oxygen.positions, oxygen.positions,
                       box=u.dimensions, result=rOO)

        # Calculate CNs of the acceptor oxygens (-1)
        cns = [len(r[(r < 3.3) & (r > 0)]) for r in rOO[oidx[:, -1]]]

        # Store minimum deltas, respective protons heights, OO distances, CNs
        out[n_ion*i:n_ion*(i+1), 0] = delta[idx]
        out[n_ion*i:n_ion*(i+1), 1] = hydrogen.positions[idx, 2]
        out[n_ion*i:n_ion*(i+1), 2] = [rOO[tuple(j)] for j in oidx]
        out[n_ion*i:n_ion*(i+1), 3] = cns

    return out


def main():

    args = parse()
    input = args.input
    n_jobs = args.n_cpu
    n_bins = args.n_bins
    a, b, c = args.cell_vectors
    n_species = args.n_species

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
    results = Parallel(n_jobs=n_jobs)(delayed(proton_transfer)(
        u, n_species, block) for block in blocks)

    # Concatenate results, compute PT coordinate grid
    results = np.concatenate(results)
    delta = np.linspace(-1, 1, n_bins)

    # Save raw data as .csv
    df = pd.DataFrame({'delta': results[:, 0], 'z-coord': results[:, 1],
                       'OO-dist': results[:, 2], 'cnum': results[:, 3]})
    df.to_csv('%s/%s.pt' % (DATA_PATH, base), index=False)

    # Boolean indexing of protons in different layers
    cond1 = np.logical_or(results[:, 1] < 13.95, results[:, 1] > 38.55)
    cond2a = np.logical_and(results[:, 1] > 13.95, results[:, 1] < 18.05)
    cond2b = np.logical_and(results[:, 1] > 34.45, results[:, 1] < 38.55)
    cond2 = np.logical_or(cond2a, cond2b)
    cond3a = np.logical_and(results[:, 1] > 18.05, results[:, 1] < 22.15)
    cond3b = np.logical_and(results[:, 1] > 30.35, results[:, 1] < 34.45)
    cond3 = np.logical_or(cond3a, cond3b)
    cond4 = np.logical_and(results[:, 1] > 22.15, results[:, 1] < 30.35)

    slice1 = results[cond1]
    slice2 = results[cond2]
    slice3 = results[cond3]
    slice4 = results[cond4]

    # Compute histograms for each layer (slice)
    hist1, _ = np.histogram(slice1[:, 0], bins=n_bins, range=(-1, 1),
                            density=True)
    hist2, _ = np.histogram(slice2[:, 0], bins=n_bins, range=(-1, 1),
                            density=True)
    hist3, _ = np.histogram(slice3[:, 0], bins=n_bins, range=(-1, 1),
                            density=True)
    hist4, _ = np.histogram(slice4[:, 0], bins=n_bins, range=(-1, 1),
                            density=True)

    # Compute 2D histogram for whole data (height vs. PT coordinate)
    hist2d, _, _ = np.histogram2d(results[:, 0], results[:, 1],
                                  bins=(n_bins, 2*n_bins), density=True,
                                  range=[[-1, 1], [9.85, 42.65]])

    # Save results as .csv
    df = pd.DataFrame(hist2d)
    df.to_csv('%s/%s.barrier2d' % (DATA_PATH, base), index=False)

    df = pd.DataFrame({'delta': delta, 'density': hist1})
    df.to_csv('%s/%s_13.95.barrier' % (DATA_PATH, base), index=False)
    df = pd.DataFrame({'delta': delta, 'density': hist2})
    df.to_csv('%s/%s_18.05.barrier' % (DATA_PATH, base), index=False)
    df = pd.DataFrame({'delta': delta, 'density': hist3})
    df.to_csv('%s/%s_22.15.barrier' % (DATA_PATH, base), index=False)
    df = pd.DataFrame({'delta': delta, 'density': hist4})
    df.to_csv('%s/%s_26.25.barrier' % (DATA_PATH, base), index=False)

    print('\nProgram executed in %.0f seconds' % (time()-t0))


if __name__ == '__main__':
    t0 = time()
    main()

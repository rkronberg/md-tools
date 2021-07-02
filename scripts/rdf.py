import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from argparse import ArgumentParser
from os import path
from time import time

import trj2blocks

# MDAnalysis
import MDAnalysis as mda
from MDAnalysis.analysis.rdf import InterRDF


def parse():
    '''Parse command line arguments.

    Returns:
        Namespace object containing input arguments.
    '''

    parser = ArgumentParser(description='MDTools: Radial distribution func.')
    parser.add_argument('-i', '--input', required=True, type=str,
                        help='Input .xyz file')
    parser.add_argument('-n', '--n_cpu', required=True, type=int,
                        help='Number of CPUs for parallel processing')
    parser.add_argument('-c', '--cell_vectors', required=True, type=float,
                        help='Lattice vectors in angstroms (a, b, c)', nargs=3)
    parser.add_argument('-nb', '--n_bins', required=True, type=int,
                        help='Number of bins')
    parser.add_argument('-t', '--thresholds', required=True, type=float,
                        help='Thresholds for solvation layers', nargs=4)
    parser.add_argument('-p', '--pair', required=True, type=str,
                        help='Which pair of elements to consider')

    return parser.parse_args()


def rdf(u, thresholds, n_bins, pair, block):
    '''Computes water radial distribution functions (RDF).

    Args:
        u: MDAnalysis Universe object containing trajectory.
        t: Thresholds specifying layer boundaries.
        n_bins: Number of bins for RDF.
        pair: Pair of elements to consider.
        block: Range of frames composing block.

    Returns:
        Linear density profile.
    '''

    atoms = list(pair)

    # Select appropriate atom groups within specified layers
    down = 'prop z > %s and prop z < %s' % (*thresholds[:2],)
    up = 'prop z > %s and prop z < %s' % (*thresholds[2:],)
    ag1 = u.select_atoms('name %s and ((%s) or (%s))' % (atoms[0], down, up),
                         updating=True)
    ag2 = u.select_atoms('name %s' % atoms[1])

    # Compute the RDF between ag1 and ag2
    rdf = InterRDF(ag1, ag2, nbins=n_bins, range=(0, 6), verbose=True)
    rdf.run(start=block.start, stop=block.stop)

    return (rdf.results.bins, rdf.results.rdf)


def main():

    args = parse()
    input = args.input
    n_jobs = args.n_cpu
    n_bins = args.n_bins
    a, b, c = args.cell_vectors
    thresholds = args.thresholds
    pair = args.pair

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
    results = Parallel(n_jobs=n_jobs)(delayed(rdf)(
        u, thresholds, n_bins, pair, block) for block in blocks)

    # Average RDFs over all blocks
    results = np.mean(results, axis=0)

    # Save results as .csv
    df = pd.DataFrame({'z-coord': results[0], 'rdf': results[1]})
    df.to_csv('%s/%s.rdf' % (DATA_PATH, base), index=False)

    print('\nProgram executed in %.0f seconds' % (time()-t0))


if __name__ == '__main__':
    t0 = time()
    main()

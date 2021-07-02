import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from argparse import ArgumentParser
from os import path
from time import time
from scipy.ndimage import find_objects, label

import trj2blocks
import tolerant

# MDAnalysis
import MDAnalysis as mda
from MDAnalysis.analysis.distances import distance_array

# tidynamics
import tidynamics


def parse():
    '''Parse command line arguments.

    Returns:
        Namespace object containing input arguments.
    '''

    parser = ArgumentParser(description='MDTools: Water rotational relaxation')
    parser.add_argument('-i', '--input', required=True, type=str,
                        help='Input .xyz file')
    parser.add_argument('-n', '--n_cpu', required=True, type=int,
                        help='Number of CPUs for parallel processing')
    parser.add_argument('-c', '--cell_vectors', required=True, type=float,
                        help='Lattice vectors in angstroms (a, b, c)', nargs=3)
    parser.add_argument('-t', '--thresholds', required=True, type=float,
                        help='Thresholds for contact layers', nargs=4)

    return parser.parse_args()


def lg2(x):
    '''Compute second order Legendre polynomial.

    Args:
        x: Argument

    Returns:
        Second order legendre polynomial with respect to given argument
    '''

    return (3*x**2-1)/2


def rotation(u, a, t, block):
    '''Computes water orientational (dipole) relaxation.

    Args:
        u: MDAnalysis Universe object containing trajectory.
        a: Lateral lattice vector assuming equal lateral dimensions
        t: Thresholds specifying layer boundaries.
        block: Range of frames composing block.

    Returns:
        Water orientational (dipole) autocorrelation function.
    '''

    size = len(block)
    wor = []

    # Select oxygen and hydrogen atoms
    oxygen = u.select_atoms('name O')
    hydrogen = u.select_atoms('name H')

    # Initialize OH distance array, dipole array and height array
    rOH = np.zeros((oxygen.n_atoms, hydrogen.n_atoms))
    dipole_array = np.zeros((len(block), oxygen.n_atoms, 3))
    height_array = np.zeros((len(block), oxygen.n_atoms))

    for i, ts in enumerate(u.trajectory[block.start:block.stop]):
        print('Processing blocks %.1f%%' % (100*i/size), end='\r')

        # Wrap atoms to box
        oxygen.wrap(box=u.dimensions)
        hydrogen.wrap(box=u.dimensions)

        # Compute OH distance array
        distance_array(oxygen.positions, hydrogen.positions,
                       box=u.dimensions, result=rOH)

        # Store heights of all oxygens
        height_array[i, :] = oxygen.positions[:, 2]

        # Loop over all oxygen positions
        for j, pos in enumerate(oxygen.positions):

            # Get bound hydrogens and combine molecules broken over PBCs
            hbound = hydrogen.positions[(rOH < 1.2)[j]]
            hbound[hbound-pos < -1.2] += a
            hbound[hbound-pos > 1.2] -= a

            # Compute dipole vectors
            dip = np.mean(hbound, axis=0)-pos
            dipole_array[i, j, :] = dip/np.linalg.norm(dip)

    # Loop over all oxygens
    for k in range(oxygen.n_atoms):

        # Get frames where oxygen in specified region
        z = height_array[:, k]
        z_bool = (z > t[0])*(z < t[1])+(z > t[2])*(z < t[3])

        # Select intervals where oxygen remains continuously in region
        contiguous_region = find_objects(label(z_bool)[0])

        # Compute dipole autocorrelation for all intervals and starting times
        for reg in contiguous_region:
            corr = tidynamics.acf(dipole_array[reg[0].start:reg[0].stop, k, :])
            wor.append(lg2(corr))

    # Average autocorrelation functions of (possibly) different length
    return tolerant.mean(wor)


def main():

    args = parse()
    input = args.input
    n_jobs = args.n_cpu
    thresholds = args.thresholds
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
    results = Parallel(n_jobs=n_jobs)(delayed(rotation)(
        u, a, thresholds, block) for block in blocks)

    # Average autocorrelation functions of (possibly) different length
    results = tolerant.mean(results)
    tau = np.arange(0, len(results))

    # Save results as .csv
    df = pd.DataFrame({'time': tau, 'autocorr': results})
    df.to_csv('%s/%s_%s.wor' % (
        DATA_PATH, base, thresholds[1]), index=False)

    print('\nProgram executed in %.0f seconds' % (time()-t0))


if __name__ == '__main__':
    t0 = time()
    main()

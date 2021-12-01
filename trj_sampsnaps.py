#!/usr/bin/env python3

import numpy as np
import multiprocessing
from mdtools import trajectory as traj
import glob
import sys
import os


def output_snapshots(data, headers):
    """
    """
    directory = 'snaps'
    try:
        os.mkdir(directory)
    except FileExistsError:
        pass
    
    for dat, head in zip(data, headers):
        snapname = 'snapshot%07d' % head['TIMESTEP']
        assert head['NUMBER OF ATOMS'] == dat.shape[0]
#        print(dat.shape)
        with open(directory+'/'+snapname, 'w') as fh:
            dx = head['BOX BOUNDS'][0,1] - head['BOX BOUNDS'][0,0]
            dy = head['BOX BOUNDS'][1,1] - head['BOX BOUNDS'][1,0]
            dz = head['BOX BOUNDS'][2,1] - head['BOX BOUNDS'][2,0]
            fh.write('%.16f %.16f %.16f\n' % (dx, dy, dz))
            fh.write('%i F\n' % (dat.shape[0]))
            tmp = np.copy(dat)
            tmp['type'][tmp['type'] == 1] = 5
            tmp['type'][tmp['type'] == 2] = 7
            tmp['type'][tmp['type'] == 3] = 5
            tmp['xu'] /= dx
            tmp['yu'] /= dy
            tmp['zu'] /= dz
            np.savetxt(fh, tmp[['id', 'type', 'xu', 'yu', 'zu']], fmt='%6i %3i %.16f %.16f %.16f')
    
    


def sample_snapshots(trj, prop_snapsamp):
    """
    """
    data = trj.data
    header = trj.header

    nmax = data.shape[0]
    
    indices = np.arange(prop_snapsamp['skip_nsteps'],
                        nmax,
                        prop_snapsamp['every_nsteps'])

    print(indices)
    indices += np.random.randint(low=-prop_snapsamp['pm_nsteps'], high=prop_snapsamp['pm_nsteps'], size=indices.shape[0]) 
    print(indices)

    sampled_data = data[indices]
    sampled_headers = header[indices]
    print(sampled_data.shape)
    print(sampled_headers.shape)
    
    return sampled_data, sampled_headers



def read_input(argv):
    
    print(argv)
    
    if len(argv) == 1:
        fh = open('trj_sampsnaps.in', 'r')
    elif len(argv) == 2:
        inputfile = argv[1]
        fh = open(inputfile, 'r')
    else:
        sys.exit()
    
    tmp = []
    for line in fh:
        tmp.append(line.strip().split(' ')[0])
    print(tmp)
    
    assert len(tmp) == 5
    
    prop_snapsamp = {
        'trjfile':        str(tmp[0]),
        'snapfile':       str(tmp[1]),
        'skip_nsteps':    int(tmp[2]),
        'every_nsteps':   int(tmp[3]),
        'pm_nsteps':      int(tmp[4]),
    }
    
    return prop_snapsamp



def main(argv):
    """ 
    """
    
    prop_snapsamp = read_input(argv)
    
    print('Loading data from ', prop_snapsamp['trjfile'])
    
    # Load the trajectory object
    trj = traj.npz2trj(prop_snapsamp['trjfile'])
    
    print('Loaded data.')
    
    sampled_data, sampled_headers = sample_snapshots(trj, prop_snapsamp)
    
    print(sampled_headers)

    output_snapshots(sampled_data, sampled_headers)
    
    with open(prop_snapsamp['snapfile'],'wb') as fh:
        np.savez(fh, sampled_headers=sampled_headers, sampled_data=sampled_data)



if __name__=='__main__':
    main(sys.argv)


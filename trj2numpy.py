#!/usr/bin/env python3

import numpy as np
#import matplotlib.pyplot as plt
import sys
from distutils.util import strtobool
from mdtools import trajectory as traj


def read_input(trj_prop, argv):
    """
    """
    print(argv)
    
    if len(argv) == 1:
        fh = open('trj2numpy.in', 'r')
    else:
        fh = open(argv[1], 'r')
    
    tmp = []
    for line in fh:
        print(line.strip().split('#', 1)[0])
        tmp.append(line.split('#', 1)[0].strip().split(' '))
    print(tmp)
    
    
    if tmp[0][0] != '':
        trj_prop['nsteps'] = int(tmp[0][0])
    
    if tmp[1][0] != '':
        trj_prop['skipnsteps'] = int(tmp[1][0])
    
    if tmp[2][0] != '':
        trj_prop['samplensteps'] = int(tmp[2][0])
        
    if tmp[3][0] != '':
        trj_prop['dt'] = np.double(tmp[3][0])
    
    if tmp[4] == ['']:
        print('No trjfile specified in input file')
    else:
        trj_prop['trjfile'] = tmp[4][0]
        trj_prop['trjtype'] = traj.determine_filetype(trj_prop['trjfile'])
    
    # Here we get the information about which atoms to extract
    if len(tmp) >= 6 and tmp[5] != ['']:
        trj_prop['atomlist'] = np.array(tmp[5]).astype(np.uint)
    else:
        print('no atomlist specified, loading all atoms')
    
    # Save as compressed numpy array or not?
    if len(tmp) >= 7 and  tmp[6] != ['']:
        trj_prop['compressed'] = bool(strtobool(tmp[6][0]))
    
    if len(tmp) >= 8 and  tmp[7] != ['']:
        trj_prop['initdatafile'] = tmp[7][0]
    
    if len(argv) > 2:
        trj_prop['trjfile'] = argv[1]
        trj_prop['trjtype'] = traj.determine_filetype(trj_prop['trjfile'])
    
    if len(argv) > 3:
        sys.exit()
    
    
    return trj_prop



def main(argv):
    """
    """
    
    # Defaults
    trj_prop = {
        'nsteps':        None,
        'skipnsteps':    np.int_(0),
        'samplensteps':  np.int_(1),
        'dt':            np.double(0.001), # in fs
        'trjfile':       'atom_pos_vel.lammpstrj',
        'inputfile':     'trj2numpy.in',
        'atomlist':      None,
        'compressed':    False,
        'initdatafile':  None,
    }
    
    trj_prop['trjtype'] = traj.determine_filetype(trj_prop['trjfile'])
    
    trj_prop = read_input(trj_prop, argv)
    
    trj = traj.data_from_trajectory(trj_prop)
    

    print(trj_prop)
    print('trj.data.shape', trj.data.shape)
    
#    print(trj.header0)
#    print(trj.header)
#    print(trj.data0)
#    print(trj.data)
#    print(trj.masses)

    fname = 'trj_nsteps%i_skipnsteps%i_samplensteps%i.npz' % (trj.nsteps, trj.skipnsteps, 
                                                              trj.samplensteps)
    print('\nOutputting trajectory data to %s' % fname)
    
    trj.trj2npz(fname)
    
#    tmp = traj.npz2trj(fname)

#    print(tmp)

#    print(tmp.header0)
#    print(tmp.header)
#    print(tmp.data0)
#    print(tmp.data)

    
    return



if __name__=="__main__":
    main(argv=sys.argv)



#!/usr/bin/env python3

import numpy as np
import multiprocessing
from mdtools import trajectory as traj
import glob
import sys
import os
from mdtools import io as mdio

def mkdir_safe(directory):
    """
    """
    try:
        os.mkdir(directory)
    except FileExistsError:
        pass



def output_snapshots(data, headers, type2atomicn, style='drprobecel'):
    """
    """
    directory = 'snaps' + '_' + style
    
    if style == 'drprobecel':
        output_snapshots_drprobecel(data, headers, type2atomicn, directory=directory)
    elif style == 'multislice':
        output_snapshots_multislice(data, headers, type2atomicn, directory=directory)
    else:
        raise NotImplementedError('Snapshot output style %s not implemented' % style)
        



def output_snapshots_drprobecel(data, headers, type2atomicn, directory):
    """
    """
    mkdir_safe(directory)
    
    for dat, head in zip(data, headers):
        snapname = 'snapshot%07d.cel' % head['TIMESTEP']
        assert head['NUMBER OF ATOMS'] == dat.shape[0]
#        print(dat.shape)
        with open(directory+'/'+snapname, 'w') as fh:
            dx = head['BOX BOUNDS'][0,1] - head['BOX BOUNDS'][0,0]
            dy = head['BOX BOUNDS'][1,1] - head['BOX BOUNDS'][1,0]
            dz = head['BOX BOUNDS'][2,1] - head['BOX BOUNDS'][2,0]
            fh.write('%.16f %.16f %.16f\n' % (dx, dy, dz))
            fh.write('%i F\n' % (dat.shape[0]))
            tmp = np.copy(dat)
            for el in type2atomicn:
                tmp['type'][tmp['type'] == int(el[0])] = int(el[1])
            tmp['xu'] /= dx
            tmp['yu'] /= dy
            tmp['zu'] /= dz
            np.savetxt(fh, tmp[['id', 'type', 'xu', 'yu', 'zu']], fmt='%6i %3i %.16f %.16f %.16f')



def output_snapshots_multislice(data, headers, type2atomicn, directory):
    """
    """ 
    mkdir_safe(directory)
    
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
            for el in type2atomicn:
                tmp['type'][tmp['type'] == int(el[0])] = int(el[1])
            tmp['xu'] /= dx
            tmp['yu'] /= dy
            tmp['zu'] /= dz
            np.savetxt(fh, tmp[['type', 'xu', 'yu', 'zu']], fmt='%3i %.16f %.16f %.16f')



def sample_snapshots(trj, sampsnap_input):
    """
    """
    data = trj.data
    header = trj.header
    
    nmax = data.shape[0]
    print(nmax)
    
    indices = np.arange(sampsnap_input['skip_nsteps'],
                        nmax+sampsnap_input['pm_nsteps'],
                        sampsnap_input['every_nsteps'])
    
    print(indices)
    if sampsnap_input['pm_nsteps'] != 0:
        indices += np.random.randint(low=-sampsnap_input['pm_nsteps'], high=sampsnap_input['pm_nsteps'], size=indices.shape[0])
    if indices[0] < 0:
        indices[0] = 0
    if indices[-1] > nmax-1:
        indices[-1] = nmax-1
    print(indices)
    
    sampled_data = data[indices]
    sampled_headers = header[indices]
    print(sampled_data.shape)
    print(sampled_headers.shape)
    
    return sampled_data, sampled_headers



#def read_input(argv):
#    
#    print(argv)
#    
#    if len(argv) == 1:
#        fh = open('trj_sampsnaps.in', 'r')
#    elif len(argv) == 2:
#        inputfile = argv[1]
#        fh = open(inputfile, 'r')
#    else:
#        sys.exit()
#    
#    tmp = []
#    for line in fh:
#        tmp.append(line.strip().split(' ')[0])
#    print(tmp)
#    
#    assert len(tmp) == 5
#    
#    sampsnap_input = {
#        'trjfile':        str(tmp[0]),
#        'snapfile':       str(tmp[1]),
#        'skip_nsteps':    int(tmp[2]),
#        'every_nsteps':   int(tmp[3]),
#        'pm_nsteps':      int(tmp[4]),
#    }
#    
#    return sampsnap_input



def process_input(inputfile):
    """
    """
    recognized_strings = ('trjfile',
                          'outputnpz',
                          'compressed',
                          'every_nsteps',
                          'pm_nsteps',
                          'skip_nsteps',
                          'attype2atomicno',
                          'snapshot_style',
                          )
    
    # Get settings from file
    sampsnap_input = mdio.inputfile2dict(inputfile, recognized_strings)
    
    # Clean up settings
    tmp = []
    if 'attype2atomicno' in sampsnap_input.keys():
        for atomlist in sampsnap_input['attype2atomicno']:
            
            tmp.append(np.array(atomlist, dtype=np.int_))
        
        sampsnap_input['attype2atomicno'] = tmp
    
    tmp = []
    if 'attypes' in sampsnap_input.keys():
        for attype in sampsnap_input['attypes']:
            if isinstance(attype, list):
                tmp.append(attype)
            else:
                tmp.append([attype,])
        sampsnap_input['attypes'] = tmp
    
    if 'trjfile' in sampsnap_input.keys():
        sampsnap_input['trjfile'] = sampsnap_input['trjfile'][0]
    
    if 'every_nsteps' in sampsnap_input.keys():
        sampsnap_input['every_nsteps'] = int(sampsnap_input['every_nsteps'][-1])
    
    if 'pm_nsteps' in sampsnap_input.keys():
        sampsnap_input['pm_nsteps'] = int(sampsnap_input['pm_nsteps'][-1])
    
    if 'skip_nsteps' in sampsnap_input.keys():
        sampsnap_input['skip_nsteps'] = int(sampsnap_input['skip_nsteps'][-1])
    
    if 'compressed' in sampsnap_input.keys():
        sampsnap_input['compressed'] = bool(sampsnap_input['compressed'][-1])
    else:
        sampsnap_input['compressed'] = False
    
    if 'outputnpz' in sampsnap_input.keys():
        sampsnap_input['outputnpz'] = sampsnap_input['outputnpz'][-1]
    
    if 'snapshot_style' in sampsnap_input.keys():
        sampsnap_input['snapshot_style'] = sampsnap_input['snapshot_style'][-1]
    
    return sampsnap_input



def main(argv):
    """ 
    """
    
    inputfile = 'trj_sampsnaps.in'
    fname = 'trj.npz'
    fout = 'trj_sampsnaps.npz'
    
    if len(argv) > 1:
        inputfile = argv[1]
    
    sampsnap_input = process_input(inputfile)
    
    if 'trjfile' in sampsnap_input.keys():
        fname = sampsnap_input['trjfile']
    
    if 'outputnpz' in sampsnap_input.keys():
        fout = sampsnap_input['outputnpz']
    
    # overwrite fname and fout if given as arg
    if len(argv) > 2:
        fname = argv[2]
    
    if len(argv) > 3:
        fout = argv[3]
    
    sampsnap_input['dname'] = '/'.join(fname.split('/')[:-1])
    if sampsnap_input['dname'] != '':
        sampsnap_input['dname'] = './' + sampsnap_input['dname'] + '/'
    
    print(sampsnap_input)
    
    print('Loading data from ', sampsnap_input['trjfile'])
    
    # Load the trajectory object
    trj = traj.npz2trj(sampsnap_input['dname'] + fname)
    
    print('Loaded data.')
    
    sampled_data, sampled_headers = sample_snapshots(trj, sampsnap_input)
    
    print(sampled_headers)
    
    output_snapshots(sampled_data, sampled_headers, sampsnap_input['attype2atomicno'],
                     sampsnap_input['snapshot_style'])
    
    
    # Save pdos information to numpy archive
    if sampsnap_input['compressed']:
        np.savez_compressed(sampsnap_input['dname'] + fout, 
                            sampled_headers=sampled_headers,
                            sampled_data=sampled_data,
                            sampsnap_input=sampsnap_input,)
    else:
        np.savez(sampsnap_input['dname'] + fout, 
                 sampled_headers=sampled_headers,
                 sampled_data=sampled_data,
                 sampsnap_input=sampsnap_input,)
    return None



if __name__=='__main__':
    main(sys.argv)


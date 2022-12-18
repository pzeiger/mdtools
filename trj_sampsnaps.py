#!/usr/bin/env python3

import numpy as np
import multiprocessing
from mdtools import trajectory as traj
import glob
import sys
import os
from mdtools import io as mdio
import numpy.lib.recfunctions as rf
#import matplotlib.pyplot as plt
#from mpl_toolkits.mplot3d import Axes3D
from pathlib import Path


def mkdir_safe(directory):
    """
    """
    try:
        os.mkdir(directory)
    except FileExistsError:
        pass





def output_snapshots(data, headers, attype_conversion, style='drprobecel', subdirectory=None):
    """
    """
    if subdirectory is not None:
        directory = 'snaps' + '_' + style + '/' + subdirectory
    else:
        directory = 'snaps' + '_' + style
    
    if style[:10] == 'drprobecel':
        output_snapshots_drprobecel(data, headers, attype_conversion, directory=directory)
    elif style[:10] == 'multislice':
        version = style[10:].strip('_')
        if version == '':
            version = None
        output_snapshots_multislice(data, headers, attype_conversion, directory=directory, version=version)
    else:
        raise NotImplementedError('Snapshot output style %s not implemented' % style)
        



def output_snapshots_drprobecel(data, headers, attype_conversion, directory):
    """
    """
    Path(directory).mkdir(parents=True, exist_ok=True)
#    mkdir_safe(directory)
    
    fields = ('type', 'xs', 'ys', 'zs', 'occ', 'Biso', 'placeholder1', 'placeholder2', 'placeholder3')
    formats = ('U6', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8')
    format_write = '%6s  % 15.12f  % 15.12f  % 15.12f  % 15.12f  % 15.12f  % 15.12f  % 15.12f  % 15.12f'
    
    for dat, head in zip(data, headers):
        print(headers)
        print(len(headers))
        print(head)
        print(len(head))
        print(dat)
        print(len(dat))
        snapname = 'snapshot%07d.cel' % head['TIMESTEP']
        assert head['NUMBER OF ATOMS'] == dat.shape[0]
#        print(dat.shape)
        with open(directory+'/'+snapname, 'w') as fh:
            
            tmp, box_dim = prepare_data(dat, head, fields, formats)
            
            # get box dimensions in nm
            box_dim['x'] /= 10
            box_dim['y'] /= 10
            box_dim['z'] /= 10
            
            fh.write('# Cel file created by trj_sampsnaps.py\n')
            fh.write('0  % 15.12f  % 15.12f  % 15.12f  % 15.12f  % 15.12f  % 15.12f\n' \
                     % (box_dim['x'], box_dim['y'], box_dim['z'], 90., 90., 90.))
            
            for el in attype_conversion:
                tmp['type'][tmp['type'] == el[0]] = str(el[1])
            
            tmp['occ'] = 1.
            write_snapshot_data(fh, tmp, fields, fmt=format_write)
            fh.write('*\n')



def output_snapshots_multislice(data, headers, attype_conversion, directory, version=None):
    """
    """ 
    mkdir_safe(directory)
    
    print(version)
    if version is None:
        fields = ('type', 'xs', 'ys', 'zs')
        formats = ('u4', 'f8', 'f8', 'f8')
        format_write = '%3i  % 15.12f  % 15.12f  % 15.12f'
    elif version == 'pms':
        fields = ('type', 'xs', 'ys', 'zs', 'mx', 'my', 'mz', 'mabs', 'Biso')
        formats = ('u4', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8')
        format_write = '%3i  % 15.12f  % 15.12f  % 15.12f  % 15.12f  % 15.12f  % 15.12f  % 15.12f  % 15.12f'
    else:
        raise NotImplementedError('output for multislice version %s not implemented' % version)
    
    for dat, head in zip(data, headers):
        
        snapname = 'snapshot%07d' % head['TIMESTEP']
        assert head['NUMBER OF ATOMS'] == dat.shape[0]
        
        with open(directory+'/'+snapname, 'w') as fh:
            
            tmp, box_dim = prepare_data(dat, head, fields, formats)

            for el in attype_conversion:
                tmp['type'][tmp['type'] == int(el[0])] = int(el[1])
            
            fh.write('%.16f %.16f %.16f\n' % (box_dim['x'], box_dim['y'], box_dim['z']))
            fh.write('%i F\n' % (dat.shape[0]))
            
            write_snapshot_data(fh, tmp, fields, fmt=format_write)
            




def prepare_data(dat, head, fields_out, formats_out):
    """
    """
    dtype_out = np.dtype({'names':    fields_out,
                          'formats':  formats_out})

    box_dim = {
        'x':   head['BOX BOUNDS'][0,1] - head['BOX BOUNDS'][0,0],
        'y':   head['BOX BOUNDS'][1,1] - head['BOX BOUNDS'][1,0],
        'z':   head['BOX BOUNDS'][2,1] - head['BOX BOUNDS'][2,0],
    }
    
    out = np.zeros(dat.shape, dtype=dtype_out)
    copy_columns_structured_array(dat, out)
    
    fields_dat = dat.dtype.fields.keys()
    fields_missing = set(fields_out) - set(fields_dat)
    
    for field in fields_missing:
        if field in ('xs', 'ys', 'zs'):
            out[field] = dat[field.strip('s') + 'u'] / box_dim[field.strip('s')]
            out[field][out[field] < .0 ] += 1.0
        elif field in ('xu', 'yu', 'zu'):
            out[field] = dat[field.strip('u') + 's'] * box_dim[field.strip('u')]
    
    return out, box_dim



def write_snapshot_data(fh, data, fields_out, fmt):
    """
    """
    if np.__version__ >= '1.18.0':
        print(data[list(fields_out)])
        np.savetxt(fh, data[list(fields_out)], fmt=fmt)
    else:
        dtypedata = data.dtype
        dtype = np.dtype({'names':    fields_out,
                          'formats':  [dtypedata.fields[field][0] for field in fields_out],
                        })
        
        tmp = np.zeros(data.shape, dtype=dtype)
        
        for el in fields_out:
            tmp[el] = data[el]
        
        print(tmp)
        
        np.savetxt(fh, tmp, fmt=fmt)
    
    return None


def copy_columns_structured_array(sarray_src, sarray_tgt):
    """
        Copy column contents from structured (source) array sarray_src to
        structured (target) array sarray_tgt
    """
    fields_src = sarray_src.dtype.fields.keys()
    fields_tgt = sarray_tgt.dtype.fields.keys()
    
    for field in fields_src:
        if field in fields_tgt:
            sarray_tgt[field] = sarray_src[field]




def sample_snapshots(trj, sampsnap_input):
    """
    """
    
    if 'fftfreqsel' in sampsnap_input.keys():
        sampled_data, sampled_headers = sample_snapshots_fftfreqsel(trj, sampsnap_input)
    else:
        sampled_data, sampled_headers = sample_snapshots_trj(trj, sampsnap_input)
    return sampled_data, sampled_headers






def sample_snapshots_trj(trj, sampsnap_input):
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
    
#    sampled_data = np.zeros(data[indices].shape, dtype=sampsnap_input['sample_dtype'])
#    copy_columns_structured_array(data[indices], sampled_data)

    sampled_data = data[indices]
    sampled_headers = header[indices]
    print(sampled_data.shape)
    print(sampled_headers.shape)
    
    dtype_dat = sampled_data.dtype
    fields = dtype_dat.fields.keys()
    names = list(dtype_dat.fields.keys()) + sampsnap_input['extra_field_names']
    formats = [dtype_dat.fields[field][0] for field in fields] + sampsnap_input['extra_field_formats']
    
    dtype = np.dtype({'names':    names,
                      'formats':  formats,
                    })
    
    tmp = np.zeros(sampled_data.shape, dtype=dtype)
    copy_columns_structured_array(sampled_data, tmp)
    
    sampled_data = tmp
    print(sampled_data)
    
    if 'uniform_magmom_dir' in sampsnap_input.keys():
        sampled_data['mx'] = sampsnap_input['uniform_magmom_dir'][0]
        sampled_data['my'] = sampsnap_input['uniform_magmom_dir'][1]
        sampled_data['mz'] = sampsnap_input['uniform_magmom_dir'][2]

    if 'Biso' in sampsnap_input.keys():
        for ii, Biso in enumerate(sampsnap_input['Biso']):
            sampled_data['Biso'][sampled_data['type'] == (ii+1)] = Biso

    if 'magnetic_moments' in sampsnap_input.keys():
        for ii, mm in enumerate(sampsnap_input['magnetic_moments']):
            sampled_data['mabs'][sampled_data['type'] == (ii+1)] = mm

    return sampled_data, sampled_headers



def sample_snapshots_fftfreqsel(trj, sampsnap_input):
    """
    """
#    nevery = sampsnap_input['fftfreqsel']['nevery']
    df = sampsnap_input['fftfreqsel']['df']                 # width of frequency interval
    fmin = sampsnap_input['fftfreqsel']['fmin']
    fmax = sampsnap_input['fftfreqsel']['fmax']
    nsplit = sampsnap_input['fftfreqsel']['nsplit']
    chunksize = sampsnap_input['fftfreqsel']['chunksize']
    
    data = trj.data
    headers = trj.header
    
    print(data.shape)
    
    # Total number of samples in the trajectory
    nsamples_t = data.shape[0]
    nsamplesize_t = int((nsamples_t - nsamples_t % nsplit)/ nsplit)
    print(nsamplesize_t)

    # number of atoms
    natoms = data.shape[1]
    print(natoms)
    
    # get the FFT frequencies
    fft_freq = np.fft.rfftfreq(n=nsamplesize_t, d=trj.dt)
    nfft = fft_freq.shape[0]
    print(fft_freq)
    
    # Get the equilibrium positions of the data
    equpos = np.zeros(natoms, dtype=np.dtype({'names':   ['xu', 'yu', 'zu'],
                                              'formats': ['f8', 'f8', 'f8']}))
    equpos['xu'] = np.mean(data['xu'], axis=0)
    equpos['yu'] = np.mean(data['yu'], axis=0)
    equpos['zu'] = np.mean(data['zu'], axis=0)
    
    pos0 = np.zeros(natoms, dtype=np.dtype({'names':   ['xu', 'yu', 'zu'],
                                            'formats': ['f8', 'f8', 'f8']}))
    pos0['xu'] = trj.data0['xu']
    pos0['yu'] = trj.data0['yu']
    pos0['zu'] = trj.data0['zu']
    
    box_dim = {
        'x':   trj.header0['BOX BOUNDS'][0,1] - trj.header0['BOX BOUNDS'][0,0],
        'y':   trj.header0['BOX BOUNDS'][1,1] - trj.header0['BOX BOUNDS'][1,0],
        'z':   trj.header0['BOX BOUNDS'][2,1] - trj.header0['BOX BOUNDS'][2,0],
    }
    
    # Correct for atom moving to other side of the box 
    # This should not happen with unwrapped coordinates from LAMMPS, but just
    # to be sure.
    equpos['xu'][pos0['xu']-equpos['xu'] >=  box_dim['x']/2] += box_dim['x']
    equpos['xu'][pos0['xu']-equpos['xu'] <= -box_dim['x']/2] -= box_dim['x']
    equpos['yu'][pos0['yu']-equpos['yu'] >=  box_dim['y']/2] += box_dim['y']
    equpos['yu'][pos0['yu']-equpos['yu'] <= -box_dim['y']/2] -= box_dim['y']
    equpos['zu'][pos0['zu']-equpos['zu'] >=  box_dim['z']/2] += box_dim['z']
    equpos['zu'][pos0['zu']-equpos['zu'] <= -box_dim['z']/2] -= box_dim['z']
    
    print('max deviation of mean pos from perfect crystal pos in x: %f', np.abs(pos0['xu']-equpos['xu']).max())
    print('max deviation of mean pos from perfect crystal pos in y: %f', np.abs(pos0['yu']-equpos['yu']).max())
    print('max deviation of mean pos from perfect crystal pos in z: %f', np.abs(pos0['zu']-equpos['zu']).max())
    
    # This is a list of frequencies, which we want to sample
    freqs = np.linspace(fmin, fmax, int((fmax-fmin)/df+1))
    
    if natoms < chunksize:
        atchunks = np.arange(0, natoms, chunksize)
    else:
        atchunks = np.arange(0, natoms, chunksize)
    
    atchunks = [(x, x+chunksize) for x in atchunks]
    
    print('atchunks', atchunks)
    
    sampled_data = []
    sampled_headers = [] 
    
    debug_dat = {}
    
    for f0 in freqs:
        
        tmp_data = []
        sampled_headers.append((f0, []))
        
        f0str = '%04.1fTHz' % (f0/1e12)
        print(f0str)
        debug_dat[f0str] = []
        
        # Loop over the splits of the trajectory
        for ns in range(nsplit):
            
            nmax = nsamplesize_t
            
            split_data = data[ns*nsamplesize_t:(ns+1)*nsamplesize_t, :]
            split_headers = headers[ns*nsamplesize_t:(ns+1)*nsamplesize_t]
            
            indices = np.arange(sampsnap_input['skip_nsteps'],
                                nmax+sampsnap_input['pm_nsteps'],
                                sampsnap_input['every_nsteps'])
            
            print('indices1', indices)
            if sampsnap_input['pm_nsteps'] != 0:
                indices += np.random.randint(low=-sampsnap_input['pm_nsteps'], 
                                             high=sampsnap_input['pm_nsteps'],
                                             size=indices.shape[0])
            print('indices1', indices)
#            print(np.logical_and(indices >= 0, indices <= (nmax-1)))
            indices = indices[np.logical_and(indices >= 0, indices <= (nmax-1))]
            
#            metadata = header[sample_times]
            sampled_snaps = np.zeros((indices.size, natoms), dtype=data.dtype)
            
            unique_types = np.unique(split_data['type'])
            tmp_out = {}
            tmp_out['fft_select'] = np.zeros((nfft, unique_types.shape[0], 3))
            tmp_out['fft'] = np.zeros((nfft, unique_types.shape[0], 3)) 
            debug_dat[f0str].append(tmp_out) 
            
            # Candidate for parallelisation
            for ach in atchunks:
                
                split_chunked_data = split_data[:,ach[0]:ach[1]]
                print(split_chunked_data.shape)
                
                tmp = np.moveaxis(np.array([split_chunked_data['xu'],
                                            split_chunked_data['yu'],
                                            split_chunked_data['zu']]), 0, 2)
                
                dat_fft = np.fft.rfft(tmp, axis=0)
                
#                dat_fft_ifft = np.fft.irfft(dat_fft, n=dat_fft.shape[0], axis=0)
                
                selector = np.logical_and(fft_freq <  (f0+df/2),
                                          fft_freq >= (f0-df/2))
                
                dat_fft_select = dat_fft * selector[:, np.newaxis, np.newaxis]
                dat_fft_select_ifft = np.fft.irfft(dat_fft_select, n=tmp.shape[0], axis=0)
                print(dat_fft_select.shape)
                print(dat_fft_select_ifft.shape)

                sum_dat_fft_select = []
                sum_dat_fft = []
                for typ in np.unique(split_chunked_data['type']):
                    typ_selector = split_chunked_data['type'][0,:] == typ
                    print(typ_selector.shape)
                    sum_dat_fft_select.append(np.sum(np.abs(dat_fft_select[:,typ_selector,:])**2*fft_freq[:,np.newaxis,np.newaxis]**2, axis=(1)))
                    sum_dat_fft.append(np.sum(np.abs(dat_fft[:,typ_selector,:])**2*fft_freq[:,np.newaxis,np.newaxis]**2, axis=(1)))
                    print(sum_dat_fft_select[-1].shape)
                    print(sum_dat_fft[-1].shape)
                
                sum_dat_fft_select = np.moveaxis(np.array(sum_dat_fft_select), 0, 1)
                sum_dat_fft = np.moveaxis(np.array(sum_dat_fft), 0, 1)
                print(sum_dat_fft_select.shape)
                print(sum_dat_fft.shape)
                debug_dat[f0str][-1]['fft_select'] += sum_dat_fft_select
                debug_dat[f0str][-1]['fft'] += sum_dat_fft
                
                sampled_snaps[:,ach[0]:ach[1]]['id'] = split_chunked_data['id'][indices,:]
                sampled_snaps[:,ach[0]:ach[1]]['type'] = split_chunked_data['type'][indices,:]
                sampled_snaps[:,ach[0]:ach[1]]['xu'] = dat_fft_select_ifft[indices,:,0] + pos0['xu'][ach[0]:ach[1]]
                sampled_snaps[:,ach[0]:ach[1]]['yu'] = dat_fft_select_ifft[indices,:,1] + pos0['yu'][ach[0]:ach[1]]
                sampled_snaps[:,ach[0]:ach[1]]['zu'] = dat_fft_select_ifft[indices,:,2] + pos0['zu'][ach[0]:ach[1]]
            
            
            tmp_data.append(sampled_snaps)
            
            for samphead in split_headers[indices]:
                sampled_headers[-1][1].append(samphead)
        
        dtype_data = data.dtype
        fields = dtype_data.fields.keys()
        names = list(dtype_data.fields.keys()) + sampsnap_input['extra_field_names']
        formats = [dtype_data.fields[field][0] for field in fields] + sampsnap_input['extra_field_formats']
        
        dtype = np.dtype({'names':    names,
                          'formats':  formats,
                         })
        
        nsnaps = sum([x.shape[0] for x in tmp_data])
        print('nsnaps', nsnaps)
        
        tmp = np.zeros((nsnaps, natoms), dtype=dtype)
        tmp2 = np.zeros((nsnaps, natoms), dtype=data.dtype)
        
        nstart = 0
        for tdat in tmp_data: 
            tmp2[nstart:nstart+tdat.shape[0]] = tdat
            nstart += tdat.shape[0] 
        
        copy_columns_structured_array(tmp2, tmp)
        
        sampled_data.append((f0, tmp))
        
        # now we build the right data structure 
        if 'uniform_magmom_dir' in sampsnap_input.keys():
            sampled_data[-1][1]['mx'] = sampsnap_input['uniform_magmom_dir'][0]
            sampled_data[-1][1]['my'] = sampsnap_input['uniform_magmom_dir'][1]
            sampled_data[-1][1]['mz'] = sampsnap_input['uniform_magmom_dir'][2]

        if 'Biso' in sampsnap_input.keys():
            for ii, Biso in enumerate(sampsnap_input['Biso']):
                sampled_data[-1][1]['Biso'][sampled_data[-1][1]['type'] == (ii+1)] = Biso

        if 'magnetic_moments' in sampsnap_input.keys():
            for ii, mm in enumerate(sampsnap_input['magnetic_moments']):
                sampled_data[-1][1]['mabs'][sampled_data[-1][1]['type'] == (ii+1)] = mm

            
#            fig = plt.figure()
#            ax = fig.add_subplot(projection='3d')
#            ax.azim = .0
#            ax.elev = 90.
#            
#            ax.set_proj_type('ortho')
#            
#            zmin = 0.0
#            zmax = 6.6
#            selector1 = np.logical_and(sampled_snaps['zu'] >= zmin, sampled_snaps['zu'] <= zmax)
#            selector2 = np.logical_and(pos0['zu'] >= zmin, pos0['zu'] <= zmax)
#            
#            ax.scatter(pos0['xu'][selector2], pos0['yu'][selector2], pos0['zu'][selector2])
#            ax.scatter(sampled_snaps['xu'][selector1], sampled_snaps['yu'][selector1], sampled_snaps['zu'][selector1])
#            ax.scatter(pos0['xu'], pos0['yu'], pos0['zu'])
#            ax.scatter(sampled_snaps['xu'], sampled_snaps['yu'], sampled_snaps['zu'])
#            plt.show()
    
    np.savez_compressed('debug_dat.npz', debug_dat=debug_dat)
    
    return sampled_data, sampled_headers



def process_input(inputfile):
    """
    """
    recognized_strings = ('trjfile',
                          'outputnpz',
                          'compressed',
                          'every_nsteps',
                          'pm_nsteps',
                          'skip_nsteps',
                          'attype_conversion',
                          'snapshot_style',
                          'Biso',
                          'uniform_magmom_dir',
                          'magnetic_moments',
                          'fftfreqsel',
                          'chunksize',
                          )
    
    # Get settings from file
    sampsnap_input = mdio.inputfile2dict(inputfile, recognized_strings)
    
    
    # Clean up settings
    tmp = []
    if 'attype_conversion' in sampsnap_input.keys():
        for atomlist in sampsnap_input['attype_conversion']:
            tmp.append(atomlist)
        sampsnap_input['attype_conversion'] = tmp
    
#    if 'attype2el' in sampsnap_input.keys():
#        for atomlist in sampsnap_input['attype2el']:
#            tmp['%s' % atomlist[0]] = atomlist[1]
#        sampsnap_input['attype2el'] = tmp
    
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
    
    
    extra_field_names = []
    extra_field_formats = []

    if 'Biso' in sampsnap_input.keys():
        extra_field_names.append('Biso')
        extra_field_formats.append('f8')
    
    if 'magnetic_moments' in sampsnap_input.keys():
        sampsnap_input['magnetic_moments'] = np.array([ np.double(x) for x in sampsnap_input['magnetic_moments']])
        extra_field_names.append('mabs')
        extra_field_formats.append('f8')
    
    if 'uniform_magmom_dir' in sampsnap_input.keys():
        sampsnap_input['uniform_magmom_dir'] = np.array([np.double(x) for x in sampsnap_input['uniform_magmom_dir'][0]])
        assert sampsnap_input['uniform_magmom_dir'].shape == (3,)
        for el in ('mx', 'my', 'mz'):
            extra_field_names.append(el)
            extra_field_formats.append('f8')
    
    sampsnap_input['extra_field_names'] = extra_field_names
    sampsnap_input['extra_field_formats'] = extra_field_formats
    
    tmp = {}
    if 'fftfreqsel' in sampsnap_input.keys():
        for el in sampsnap_input['fftfreqsel']:
            if el[0][0].lower() == 'n':
                tmp[el[0]] = int(el[1])
            else:
                tmp[el[0]] = np.double(el[1])
        sampsnap_input['fftfreqsel'] = tmp
    

    return sampsnap_input



def main(argv):
    """ 
    """
    
    inputfile = 'trj_sampsnaps.in'
    fname = 'trj.npz'
    fout = 'trj_sampsnaps.npz'
    chunksize = 1000
    
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
    
    if 'chunksize' not in sampsnap_input['fftfreqsel'].keys():
        sampsnap_input['fftfreqsel']['chunksize'] = chunksize 
    
    print(sampsnap_input)
    
    print('Loading data from ', sampsnap_input['trjfile'])
    
    # Load the trajectory object
    trj = traj.npz2trj(sampsnap_input['dname'] + fname)
    
    print('Loaded data.')
    
    sampled_data, sampled_headers = sample_snapshots(trj, sampsnap_input)
    
    print(sampled_data)
    print(sampled_headers)
    print('\n')
    
    if 'fftfreqsel' in sampsnap_input:
        for tmpdata, tmpheaders in zip(sampled_data, sampled_headers):
            subdir = '%04.1fTHz' % (tmpdata[0]/1e12 )
            output_snapshots(tmpdata[1], tmpheaders[1], sampsnap_input['attype_conversion'],
                             sampsnap_input['snapshot_style'], subdirectory=subdir)
    else:
        output_snapshots(sampled_data, sampled_headers, sampsnap_input['attype_conversion'],
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


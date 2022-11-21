#!/usr/bin/env python3

import numpy as np
import multiprocessing
from mdtools import trajectory as traj
import glob
import sys
import os
from mdtools import io as mdio
import numpy.lib.recfunctions as rf



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
    elif style[:10] == 'multislice':
        version = style[10:].strip('_')
        if version == '':
            version = None
        output_snapshots_multislice(data, headers, type2atomicn, directory=directory, version=version)
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
#            for el in type2atomicn:
#                tmp['type'][tmp['type'] == int(el[0])] = int(el[1])
            tmp['xu'] /= dx
            tmp['yu'] /= dy
            tmp['zu'] /= dz
            
            if np.__version__ >= '1.18.0':
                np.savetxt(fh, tmp[['id', 'type', 'xu', 'yu', 'zu']], fmt='%6i %3i %.16f %.16f %.16f')
            else:
                dtypetmp = tmp.dtype
                dtype = np.dtype({'names':    ('id', 'type', 'xu', 'yu', 'zu'),
                                  'formats':  (dtypetmp.fields['id'][0],
                                               dtypetmp.fields['type'][0],
                                               dtypetmp.fields['xu'][0],
                                               dtypetmp.fields['yu'][0],
                                               dtypetmp.fields['zu'][0])})
                tmp2 = np.zeros(tmp.shape, dtype=dtype)
                
                for el in ('type', 'xu', 'yu', 'zu'):
                    tmp2[el] = tmp[el]
                
                np.savetxt(fh, tmp2, fmt='%3i %.16f %.16f %.16f')
                write_snapshot_data(fh, tmp, fields_out, fmt=format_out)



def output_snapshots_multislice(data, headers, type2atomicn, directory, version=None):
    """
    """ 
    mkdir_safe(directory)
    
    print(version)
    if version is None:
        fields_out = ('type', 'xs', 'ys', 'zs')
        formats = ('u4', 'f8', 'f8', 'f8')
        format_out = '%3i  % 15.12f  % 15.12f  % 15.12f'
    elif version == 'pms':
        fields_out = ('type', 'xs', 'ys', 'zs', 'mx', 'my', 'mz', 'mabs', 'Biso')
        formats = ('u4', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8')
        format_out = '%3i  % 15.12f  % 15.12f  % 15.12f  % 15.12f  % 15.12f  % 15.12f  % 15.12f  % 15.12f'
    else:
        raise NotImplementedError('output for multislice version %s not implemented' % version)
    
    dtype = np.dtype({'names':    fields_out,
                      'formats':  formats})
    
    for dat, head in zip(data, headers):
        
        snapname = 'snapshot%07d' % head['TIMESTEP']
        assert head['NUMBER OF ATOMS'] == dat.shape[0]
        
        with open(directory+'/'+snapname, 'w') as fh:

            box_dim = {
                'x':   head['BOX BOUNDS'][0,1] - head['BOX BOUNDS'][0,0],
                'y':   head['BOX BOUNDS'][1,1] - head['BOX BOUNDS'][1,0],
                'z':   head['BOX BOUNDS'][2,1] - head['BOX BOUNDS'][2,0],
            }
            fh.write('%.16f %.16f %.16f\n' % (box_dim['x'], box_dim['y'], box_dim['z']))
            fh.write('%i F\n' % (dat.shape[0]))
            
            tmp = np.zeros(dat.shape, dtype=dtype)
            copy_columns_structured_array(dat, tmp)
            
            fields_dat = dat.dtype.fields.keys()
            
            fields_missing = set(fields_out) - set(fields_dat)
            
            for field in fields_missing:
                if field in ('xs', 'ys', 'zs'):
                    tmp[field] = dat[field.strip('s') + 'u'] / box_dim[field.strip('s')]
                    tmp[field][tmp[field] < .0 ] += 1.0
            
            print(tmp)
            for el in type2atomicn:
                tmp['type'][tmp['type'] == int(el[0])] = int(el[1])

            write_snapshot_data(fh, tmp, fields_out, fmt=format_out)
            



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
    
    if 'uniform_spin_direction' in sampsnap_input.keys():
        sampled_data['mx'] = sampsnap_input['uniform_spin_direction'][0]
        sampled_data['my'] = sampsnap_input['uniform_spin_direction'][1]
        sampled_data['mz'] = sampsnap_input['uniform_spin_direction'][2]

    if 'Biso' in sampsnap_input.keys():
        for ii, Biso in enumerate(sampsnap_input['Biso']):
            sampled_data['Biso'][sampled_data['type'] == (ii+1)] = Biso

    if 'magnetic_moments' in sampsnap_input.keys():
        for ii, mm in enumerate(sampsnap_input['magnetic_moments']):
            sampled_data['mabs'][sampled_data['type'] == (ii+1)] = mm

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
                          'attype2atomicno',
                          'snapshot_style',
                          'Biso',
                          'uniform_spin_direction',
                          'magnetic_moments',
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
    
    
    extra_field_names = []
    extra_field_formats = []

    if 'Biso' in sampsnap_input.keys():
        extra_field_names.append('Biso')
        extra_field_formats.append('f8')
    
    if 'magnetic_moments' in sampsnap_input.keys():
        sampsnap_input['magnetic_moments'] = np.array([ np.double(x) for x in sampsnap_input['magnetic_moments']])
        extra_field_names.append('mabs')
        extra_field_formats.append('f8')
    
    if 'uniform_spin_direction' in sampsnap_input.keys():
        sampsnap_input['uniform_spin_direction'] = np.array([np.double(x) for x in sampsnap_input['uniform_spin_direction'][0]])
        assert sampsnap_input['uniform_spin_direction'].shape == (3,)
        for el in ('mx', 'my', 'mz'):
            extra_field_names.append(el)
            extra_field_formats.append('f8')
    
    sampsnap_input['extra_field_names'] = extra_field_names
    sampsnap_input['extra_field_formats'] = extra_field_formats
    
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


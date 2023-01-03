#!/usr/bin/env python3

import numpy as np
from pwtools import pydos
from pwtools import signal
from mdtools import trajectory as traj
from mdtools import io as mdio
import multiprocessing
import glob
import sys



def get_atomlists_from_plane(trj, planedir, tol=0.1):
    """
    tol      tolerance in angstroms
    """
    
    atomlists = []
    
    if planedir in ('x', 'y', 'z'):
        
        dir_index = {
            'x':   0,
            'y':   1,
            'z':   2,
        }
        
        # Find the right column in trj.data
        identifier = [x for x in trj.header0['ATOMS'] if x.startswith(planedir)]
        assert len(identifier) == 1
        identifier = identifier[0]
        
        d0 = trj.header0['BOX BOUNDS'][dir_index[planedir],0]
        d1 = trj.header0['BOX BOUNDS'][dir_index[planedir],1]
        bins = np.linspace(d0, d1, np.ceil((d1-d0)/tol).astype(np.int_))
        digitized = np.digitize(np.mean(trj.data[identifier], axis=0), bins)
        
        # make sure that no atoms outside box
#        assert digitized.max() < bins.shape[0]
#        assert digitized.min() > 0
        
        hist = [np.sum(digitized == x) for x in range(bins.shape[0])]
       
        # Find the shift di, for which the autocorrelation becomes zero
        di = 0
        while np.correlate(hist, np.roll(hist, di+1)) != 0:
            di += 1
        
        i_found = -di-1
        atomlists = []
        for i in range(bins.shape[0]):
            if hist[i] == 0:
                continue
            elif i <= i_found+di:
                atomlists[-1] = np.concatenate((atomlists[-1], trj.data0['id'][digitized == i]))
            else:
                atomlists.append(trj.data0['id'][digitized == i])
                i_found = i

#        print(atomlist)
        
        # put last and first atomlist together if last and first bin are populated
        if digitized.max() == bins.shape[0] and digitized.min() == 0:
            atomlists[0] = np.concatenate((atomlists[0], atomlists[:-1]))
            atomlists = atomlists[:-1]
        
#        atomlist = [np.array(x) for x in atomlist]
        
#        for el in atomlist:
#            print(el.shape)
        
        # try if we have only unique atoms
        test = np.array([])
        for x in atomlists:
            test = np.concatenate((test, x))
        print(np.unique(test).shape)
        
    else:
        raise NotImplementedError('PDOS calculation for specified planes not supported.')
    print('Found %i %s-planes' % (len(atomlists), planedir))
    return atomlists



def pdos2file(pdos, dname, fprefix):
    """
    """
    
    np.savetxt(dname + fprefix + '_%s_normalized.dat' % (el,),
               np.array([pdos['tot'][0],
                         pdos['tot'][1],
                         pdos['x'][1],
                         pdos['y'][1],
                         pdos['z'][1], ]).swapaxes(0,1))



def pdos(pdos_input, trj):
    """
    """
    ppdos = {}
    
    if 'planes' in pdos_input.keys():
        print('\n', 'planes')
        print(pdos_input['planes'])
        
        ppdos['planes'] = {}
        
        for plane in pdos_input['planes']:
            print('plane:', plane)
            atomlists = get_atomlists_from_plane(trj, plane)
            ppdos['planes'][str(plane)] = []
            
            for atomlist in atomlists:
                pdoss = compute_pdos(trj, atomlist, split_natoms=pdos_input['split_natoms'])
                ppdos['planes'][str(plane)].append(pdoss)
    

    if 'atomlists' in pdos_input.keys():
        print('\n', 'atomlists')
        print(pdos_input['atomlists'])
        
        ppdos['atomlists'] = []
        
        for atomlist in pdos_input['atomlists']:
            print('atomlist:', atomlist)
            pdoss = compute_pdos(trj, atomlist, split_natoms=pdos_input['split_natoms'])
            ppdos['atomlists'].append(pdoss)
   

    if 'attypes' in pdos_input.keys():
        print('\n', 'attypes')
        print(pdos_input['attypes'])
        
        ppdos['attypes'] = {}
        
        for typ in pdos_input['attypes']:
            print('attype:', typ)
            atomlist = trj.get_atids_by_attypes(typ)
            pdoss = compute_pdos(trj, atomlist, split_natoms=pdos_input['split_natoms'])
            ppdos['attypes'][str(typ)] = pdoss
    
    if 'modulus_atid' in pdos_input.keys():
        print('\n', 'modulus_atid')
        print(pdos_input['modulus_atid'])
        
        modulus = pdos_input['modulus_atid']
        ppdos['modulus_atid'] = {}
        ppdos['modulus_atid']['modulus'] = modulus
        ppdos['modulus_atid']['remainder'] = {}
        
        for remainder in range(modulus):
            print('Currently processing atoms with id %i modulus:' \
                  % modulus, remainder)
            atomlist = trj.get_atids_by_modulo(modulus, remainder)
            pdoss = compute_pdos(trj, atomlist, split_natoms=pdos_input['split_natoms'])
            ppdos['modulus_atid']['remainder'][str(remainder)] = pdoss
        

    return ppdos



def compute_pdos(trj, atomlist=[], split_natoms=None):
    """
    """
    
    if not isinstance(atomlist, np.ndarray):
        atomlist = np.ndarray(atomlist)
    
    # Check for empty array
    if atomlist.size==0:
        print('compute_pdos(): no list of atoms was specified.')
        return {}
    
    # split up the computation to avoid using too much memory
    if split_natoms and atomlist.size > split_natoms:
        atomlists = np.array_split(atomlist, np.ceil(atomlist.shape[0] / split_natoms))
    else:
        atomlists = [atomlist,]
    
    pdoss = []
    
    for al in atomlists:
        
        vel, vel_x, vel_y, vel_z = trj.get_velocities(al)
        masses = trj.get_masses(al)
        
#        print(vel.shape)
#        print(masses.shape)
        
        dt = trj.dt * trj.samplensteps
        
        pdos_tmp = {}
        pdos_tmp['tot']  = pydos.pdos(vel=vel,   dt=dt, m=masses, area=1  ,full_out=True)
        pdos_tmp['x']    = pydos.pdos(vel=vel_x, dt=dt, m=masses, area=1/3, full_out=True)
        pdos_tmp['y']    = pydos.pdos(vel=vel_y, dt=dt, m=masses, area=1/3, full_out=True)
        pdos_tmp['z']    = pydos.pdos(vel=vel_z, dt=dt, m=masses, area=1/3, full_out=True)
        pdos_tmp['atomlist'] = al
        pdos_tmp['masses'] = masses
        
        pdoss.append(pdos_tmp)
    
    if len(pdoss) == 1:
        pdos = pdoss[0]
    else:
        pdos = merge_pdoss(pdoss)
    
    return pdos



def merge_pdoss(pdoss):
    """
    """
    pdos = {}
    
    pdos['tot']      = list(pdoss[0]['tot'])
    pdos['x']        = list(pdoss[0]['x'])
    pdos['y']        = list(pdoss[0]['y'])
    pdos['z']        = list(pdoss[0]['z'])
    pdos['atomlist'] = list(pdoss[0]['atomlist'])
    pdos['masses']   = list(pdoss[0]['masses'])
    
    pdos['tot'][1] *= len(pdos['masses'])
    pdos['x'][1] *= len(pdos['masses'])
    pdos['y'][1] *= len(pdos['masses'])
    pdos['z'][1] *= len(pdos['masses'])
    
    for el in pdoss[1:]:
#        print(el['masses'])
        pdos['atomlist'] = np.concatenate((pdos['atomlist'], el['atomlist']))
        pdos['masses'] = np.concatenate((pdos['masses'], el['masses']))
        
        for component in ('x', 'y', 'z', 'tot'):
            pdos[component][1] += el[component][1] * len(el['masses'])
            pdos[component][3] += el[component][3]
    
    pdos['tot'][1] /= len(pdos['masses'])
    pdos['x'][1] /= len(pdos['masses'])
    pdos['y'][1] /= len(pdos['masses'])
    pdos['z'][1] /= len(pdos['masses'])
    
    return pdos



def process_input(inputfile):
    """
    """
    recognized_strings = ('planes', 
                          'atomlists',
                          'attypes',
                          'trjfile',
                          'split_natoms',
                          'outputfile',
                          'compressed',
                          'modulus_atid',
                          'attype2mass',)
    
    # Get settings from file
    pdos_input = mdio.inputfile2dict(inputfile, recognized_strings)
    
    # Clean up settings
    tmp = {}
    if 'attype2mass' in pdos_input.keys():
        for atomlist in pdos_input['attype2mass']:
            
            tmp[str(atomlist[0])] = np.double(atomlist[1])
        
        pdos_input['attype2mass'] = tmp
    else:
        pdos_input['attype2mass'] = None
    
    tmp = []
    if 'atomlists' in pdos_input.keys():
        for atomlist in pdos_input['atomlists']:
            
            tmp.append(np.array(atomlist, dtype=np.int_))
        
        pdos_input['atomlists'] = tmp
    
    tmp = []
    if 'attypes' in pdos_input.keys():
        for attype in pdos_input['attypes']:
            if isinstance(attype, list):
                tmp.append(attype)
            else:
                tmp.append([attype,])
        pdos_input['attypes'] = tmp
    
    if 'trjfile' in pdos_input.keys():
        pdos_input['trjfile'] = pdos_input['trjfile'][-1]
    
    if 'split_natoms' in pdos_input.keys():
        pdos_input['split_natoms'] = int(pdos_input['split_natoms'][-1])
    
    if 'modulus_atid' in pdos_input.keys():
        pdos_input['modulus_atid'] = int(pdos_input['modulus_atid'][-1])
    
    if 'compressed' in pdos_input.keys():
        pdos_input['compressed'] = bool(pdos_input['compressed'][-1])
    else:
        pdos_input['compressed'] = False
    
    if 'outputfile' in pdos_input.keys():
        pdos_input['outputfile'] = pdos_input['outputfile'][-1]
    
    return pdos_input



def main(argv):
    """ 
    """
    
    inputfile = 'pdos.in'
    fname = 'trj.npz'
    fout = 'trj_pdos.npz'
    
    if len(argv) > 1:
        inputfile = argv[1]
    
    pdos_input = process_input(inputfile)
    
    if 'trjfile' in pdos_input.keys():
        fname = pdos_input['trjfile']
    
    if 'outputfile' in pdos_input.keys():
        fout = pdos_input['outputfile']
    
    # overwrite fname and fout if given as arg
    if len(argv) > 2:
        fname = argv[2]
    
    if len(argv) > 3:
        fout = argv[3]
    
    pdos_input['dname'] = '/'.join(fname.split('/')[:-1])
    if pdos_input['dname'] != '':
        pdos_input['dname'] = './' + pdos_input['dname'] + '/'
    
    print(pdos_input)
    
    # Load the trajectory object
    trj = traj.npz2trj(fname, attype2mass=pdos_input['attype2mass'])
   
    # Compute the (projected) PDOSs
    ppdos = pdos(pdos_input, trj)
    
    # Save pdos information to numpy archive
    if pdos_input['compressed']:
        np.savez_compressed(pdos_input['dname'] + fout, pdos_input=pdos_input, ppdos=ppdos)
    else:
        np.savez(pdos_input['dname'] + fout, pdos_input=pdos_input, ppdos=ppdos)
    
    return None



if __name__=='__main__':
    main(sys.argv)


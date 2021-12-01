#!/usr/bin/env python3

import numpy as np
#import matplotlib.pyplot as plt
import sys
import scipy.constants as constants
from distutils.util import strtobool



def npz2trj(fname):
    """
    """
    
    npzfile = np.load(fname, allow_pickle=True)
    trj_prop = npzfile['trj_prop'].item()
    header0 = npzfile['trj_header0'].item()
    data0 = npzfile['trj_data0']
    header = npzfile['trj_header']
    data = npzfile['trj_data']
    attype2mass = npzfile['attype2mass'].item()

#    print(trj_prop)
#    print(header0)
#    print(header)
#    print(data0)
#    print(data)
#    print(masses)

    if trj_prop['trjtype'] == 'ipixyz':
        trj = IpixyzTrj(trj_prop, header0, header, data0, data, attype2mass)
    elif trj_prop['trjtype'] == 'lammpstrj':
        trj = LammpsTrj(trj_prop, header0, header, data0, data, attype2mass)
    
    return trj




def data_from_trajectory(trj_prop):
    """
    PARAMETERS
    ----------
    
    prop_trj   dict object containing properties and sampling properties of 
               the to be sampled trajectory
    
    
    RETURNS
    -------
   
    trj        The sampled trajectory object
    """
    if trj_prop['trjtype'] == 'ipixyz':
        trj = IpixyzTrj(trj_prop)
    elif trj_prop['trjtype'] == 'lammpstrj':
        trj = LammpsTrj(trj_prop)
    else:
        raise NotImplementedError
    return trj



class Trajectory():
    """ Template class for trajectories
    """
    def __init__(self, trj_prop, header0=None, header=[], data0=None, data=[], attype2mass={}):
        
        self.dt = trj_prop['dt']
        self.nsteps = trj_prop['nsteps']
        self.skipnsteps = trj_prop['skipnsteps']
        self.samplensteps = trj_prop['samplensteps']
        self.trjfile = trj_prop['trjfile']
        self.compressed = trj_prop['compressed']
        
        # atomlist feature
        if 'atomlist' in trj_prop.keys():
            self.atomlist = trj_prop['atomlist']
        else:
            self.atomlist = None
        
        # initial datafile
        if 'initdatafile' in trj_prop.keys():
            self.initdatafile = trj_prop['initdatafile']
        else:
            self.initdatafile = None
        
        self.header0 = header0
        self.header = header
        self.data0 = data0
        self.data = data
        self.attype2mass = attype2mass
        
        if self.header0 is None and self.header == [] and self.data0 is None and self.data == []:
            with open(self.trjfile, 'r') as self.fh:
                self.process_trjfile()
        
        if self.attype2mass == {}:
            print('self.initdatafile', self.initdatafile)
            try:
                self.attype2mass = self.find_attypes_masses()
            except:
                print('could not determine masses for atoms.')
    
    
    
    def process_timestep_header(self):
        """
        """
        pass
    

    
    def process_timestep_data(self, header):
        """
        """
        pass 
   
    

    def find_attypes_masses(self):
        """
        """
        pass
    
    
    def process_trjfile(self):
        """
        """
        self.header0 = self.process_timestep_header()
        self.data0 = self.process_timestep_data(header=self.header0)
        nsteps = self.nsteps
        if self.skipnsteps:
#            nsteps += self.skipnsteps
            try:
                for i in range((self.skipnsteps)*(self.header0['NUMBER OF ATOMS']+self.header_lines)):
                    next(self.fh)
            except StopIteration:
                print('Reached end of file')
                return None
        
        print('Timestep(s) loaded:')
        step = 1
        data = []
        header = []
        while True:
            try:
               header.append(self.process_timestep_header())
            except StopIteration:
                print('Reached end of file')
                break
            print(' %i' % header[-1]['TIMESTEP'])
            
            data.append(self.process_timestep_data(header=header[-1]))
            self.nsteps = step
            
            # Check if requested number of timesteps reached
            if nsteps:
                if step >= nsteps:
                    print('\nreached end of requested time steps')
                    break
            
            # Skip timesteps
            try:
                for i in range((self.samplensteps-1)*(self.header0['NUMBER OF ATOMS']+self.header_lines)):
                    next(self.fh)
            except StopIteration:
                print('Reached end of file')
#                self.nstep = 
                break
            
            step += 1
        self.header = header
        self.data = np.array(data)
        
        return None
    
    
    
    def get_velocities(self, atomlist=None):
        """ If atomlist None, return data for all atoms
        """
        nsteps = self.data.shape[0]
        if atomlist is not None:
            data = self.data[np.isin(self.data['id'], atomlist)].reshape((nsteps,-1))
        
        print('nsteps:', nsteps)
        vel = np.array([data['vx'],
                        data['vy'], 
                        data['vz']]).swapaxes(0,1).swapaxes(1,2)
        
        vel_x = np.array([data['vx'],
                          np.zeros(data['vy'].shape),
                          np.zeros(data['vz'].shape)]).swapaxes(0,1).swapaxes(1,2)
        
        vel_y = np.array([np.zeros(data['vx'].shape),
                          data['vy'],
                          np.zeros(data['vz'].shape)]).swapaxes(0,1).swapaxes(1,2)
        
        vel_z = np.array([np.zeros(data['vx'].shape),
                          np.zeros(data['vy'].shape),
                          data['vz']]).swapaxes(0,1).swapaxes(1,2)
        
        return vel, vel_x, vel_y, vel_z
    
    
    
    def get_masses(self, atomlist=None):
        """
        """
        if atomlist is not None:
            masses = np.array([self.attype2mass[str(x)] for x in \
                               self.data[0][np.isin(self.data[0]['id'], atomlist)]['type']])
        else:
            masses = np.array([self.attype2mass[str(x)] for x in self.data[0]['type']])
        
        return masses
    
    
    def get_atids_by_attypes(self, attypes=[]):
        """
        """
        atomlist = self.data[0][np.isin(self.data[0]['type'], attypes)]['id']
        return atomlist
    
    
    def trj2npz(self, fname):
        
        trj_prop = {
            'dt':            self.dt,
            'nsteps':        self.nsteps,
            'skipnsteps':    self.skipnsteps,
            'atomlist':      self.atomlist,
            'trjfile':       self.trjfile,
            'trjtype':       determine_filetype(self.trjfile),
            'compressed':    self.compressed,
            'samplensteps':  self.samplensteps,
        }
        
        if self.compressed:
            np.savez_compressed(fname, trj_prop=trj_prop, 
                                trj_data0=self.data0, trj_data=self.data,
                                trj_header0=self.header0, trj_header=self.header,
                                attype2mass=self.attype2mass)
        else:
            np.savez(fname, trj_prop=trj_prop, 
                     trj_data0=self.data0, trj_data=self.data,
                     trj_header0=self.header0, trj_header=self.header,
                     attype2mass=self.attype2mass)



class LammpsTrj(Trajectory):
    """
    """
    def __init__(self, trj_prop, header0=None, header=[], data0=None, data=[], attype2mass={}):
        self.header_lines = 9
        super().__init__(trj_prop, header0, header, data0, data, attype2mass)
    
    def process_timestep_header(self):
        """ Process lammps trajectory header of form:
            
            ITEM: TIMESTEP
            20000
            ITEM: NUMBER OF ATOMS
            22080
            ITEM: BOX BOUNDS pp pp pp
            -3.4910100710319129e-04 5.2078138472544119e+01
            -5.9678617969060807e-03 2.5131432190351706e+01
            -8.0965716967275447e-01 1.4946280355912924e+02
            ITEM: ATOMS id type vx vy vz
        """
        
        header = {}
        tmp = [next(self.fh).strip().split() for x in range(9)]
        header[tmp[0][1]] = int(tmp[1][0])
        header[tmp[2][1]+' '+tmp[2][2]+' '+tmp[2][3]] = int(tmp[3][0])
        header[tmp[4][1]+' '+tmp[4][2]] = np.array([tmp[5], tmp[6], tmp[7]], dtype='f8')
        header[tmp[8][1]] = tmp[8][2:]
        return header
    

    def process_timestep_data(self, header):
        """
        """
        dtype={'names': tuple(header['ATOMS']),
               'formats': ('u4', 'u1', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8')}
        tmp = [tuple(next(self.fh).split()) for x in range(header['NUMBER OF ATOMS'])]
        data = np.array(tmp, dtype=dtype)
        if not self.atomlist is None:
            data = data[np.isin(data['id'], self.atomlist)]
        data.sort(order='id')
        return data
    
    
    def find_attypes_masses(self):
        masses = {}
        
        if self.initdatafile:
            with open(self.initdatafile, 'r') as fh:
                save = False
                for line in fh:
                    tmp = line.strip().split()
                    if tmp == []:
                        continue
                    if tmp[0] == 'Masses':
                       save = True
                       continue
                    elif tmp[0] == 'Atoms':
                       save = False
                    
                    if save:
                        masses[tmp[0]] = np.double(tmp[1])
        else:
            raise NotImplementedError('Extracting masses from LAMMPS logfile not yet implemented')
        
        return masses

    
    def get_atids_by_attypes(self, attypes=[]):
        """
        """
        attypes = np.array(attypes, dtype=np.int_)
        return super().get_atids_by_attypes(attypes)



class IpixyzTrj(Trajectory):
    """
    """
    def __init__(self, trj_prop, header0=None, header=[], data0=None, data=[], attype2mass={}):
        self.header_lines = 2
        super().__init__(trj_prop, header0, header, data0, data, attype2mass)
    
    
    def process_timestep_header(self):
        """ Process ipixyz trajectory header of form:
            
            22080
            # CELL(abcABC):   98.27697    47.28353   282.90382    90.00000    90.00000    90.00000  Step:           0  Bead:       0 velocities{atomic_unit}  cell{atomic_unit}
        """
        header = {}
        tmp = [next(self.fh).strip().split() for x in range(2)]
        header['NUMBER OF ATOMS'] = int(tmp[0][0])
        header[tmp[1][1]] = tmp[1][2:8]
        header['TIMESTEP'] = int(tmp[1][9])
        header[tmp[1][10].strip(':')] = tmp[1][11]
        header[tmp[1][12].strip(':').split('{')[0]] = tmp[1][12].strip(':').split('{')[-1].strip('}')
        header[tmp[1][13].strip(':').split('{')[0]] = tmp[1][13].strip(':').split('{')[-1].strip('}')
        return header
   
    
    def process_timestep_data(self, header):
        """
        """
        names = ['sym']
        formats = ['U2',]
        
        if 'positions' in header.keys():
            names = names + ['x', 'y', 'z']
            formats = formats + ['f8', 'f8', 'f8']
        
        if 'velocities' in header.keys():
            names = names + ['vx', 'vy', 'vz']
            formats = formats + ['f8', 'f8', 'f8']
        
        dtype={'names': tuple(names),
               'formats': tuple(formats)}
        
        tmp = [tuple([x+1] + next(self.fh).split()) for x in range(header['NUMBER OF ATOMS'])]
        names = ['id'] + names
        formats = ['u4'] + formats
        dtype={'names': tuple(names),
               'formats': tuple(formats)}
        data = np.array(tmp, dtype=dtype)
        if self.atomlist is not None:
            data = data[np.isin(data['id'], self.atomlist)]
        data.sort(order='id')
        return data
    
    
    


def determine_filetype(fname):
    if fname[-3:] == 'xyz':
        out = 'ipixyz'
    elif fname[-9:] == 'lammpstrj':
        out = 'lammpstrj'
    else:
        raise NotImplementedError
    return out



def get_mass(typeid, NC=1):
    """
    """
    if typeid <= NC*4/2:
        mass = 10.810
    else:
        mass = 14.007
    return mass


def get_mass_from_elsymbol(elsymbol):
    """
    """
    symbol_to_mass = {
        'B':  10.810,
        'N':  14.007,
    }
    try:
        mass = symbol_to_mass[elsymbol]
    except:
        raise NotImplementedError
    return mass





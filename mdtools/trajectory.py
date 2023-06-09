#!/usr/bin/env python3

import numpy as np
#import matplotlib.pyplot as plt
import sys
from distutils.util import strtobool



def npz2trj(fname, attype2mass=None):
    """
    """
    
    npzfile = np.load(fname, allow_pickle=True)
    trj_prop = npzfile['trj_prop'].item()
    header0 = npzfile['trj_header0'].item()
    data0 = npzfile['trj_data0']
    header = npzfile['trj_header']
    data = npzfile['trj_data']
#    initheader = npzfile['trj_initheader']
#    initdata = npzfile['trj_initdata']
    
    if attype2mass is None:
        attype2mass = npzfile['attype2mass'].item()
    
#    print(trj_prop)
#    print(header0)
#    print(header)
#    print(data0)
#    print(data)
#    print(masses)
    
    if trj_prop['trjtype'] == 'ipixyz':
        trj = IpixyzTrj(trj_prop, header0=header0, header=header, data0=data0,
                        data=data, attype2mass=attype2mass)
    elif trj_prop['trjtype'] == 'lammpstrj':
        trj = LammpsTrj(trj_prop, header0=header0, header=header,
                        data0=data0, data=data, attype2mass=attype2mass)
    
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
        if 'data0file' in trj_prop.keys():
            self.data0file = trj_prop['data0file']
        else:
            self.data0file = None
        
        self.header0 = header0
        self.header = header
        self.data0 = data0
        self.data = data
        self.attype2mass = attype2mass
        
        print('alive', self.data0)
        
        if self.data0file and self.data0 is None:
            print('alive', self.data0file)
            self.process_data0()
        
        if self.header == [] and self.data == []:
            with open(self.trjfile, 'r') as self.fh:
                self.process_trjfile()
        
        if self.attype2mass == {}:
            print('self.data0file', self.data0file)
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
    
    

    def process_data0(self):
        """
        """
        pass
    
    
    
    def process_trjfile(self):
        """
        """
        nsteps = self.nsteps
        if self.skipnsteps:
            tmpheader = self.process_timestep_header()
            try:
                for i in range((self.skipnsteps)*(tmpheader['NUMBER OF ATOMS']+self.header_lines)-self.header_lines):
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
                self.nstep = step
                break
            
            step += 1
        self.header = header
        self.data = np.array(data)
        
        return None
    
    
    
    def get_velocities(self, atomlist=None):
        """ If atomlist None, return data for all atoms
        """
        nsteps = self.data.shape[0]
#        print(self.data)
#        print(atomlist)
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
    
    
    def get_atids_by_modulo(self, modulus=1, remainder=0):
        """
        """
        print(self.data[0]['id'] % modulus)
        print(np.sum((self.data[0]['id'] - 1) % modulus == remainder))
        atomlist = self.data[0][np.isin((self.data[0]['id'] - 1) % modulus, remainder)]['id']
        print(atomlist.shape)
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
            'data0file':     self.data0file,
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
        super().__init__(trj_prop, header0=header0, header=header, data0=data0,
                         data=data, attype2mass=attype2mass)
    
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
    
    
    
    def process_data0(self):
        """
        """
        
        header = {}
        masses = {}
        header['BOX BOUNDS'] = []
        header['BOX'] = {
                'xy': .0,
                'xz': .0,
                'yz': .0,
        }
        save = None
        
        with open(self.data0file, 'r') as fh:
            
            for line in fh:
                tmp = line.strip().split()
                if tmp == []:
                    continue
                if tmp[0] == 'Masses':
                    save = 'Masses'
                    continue
                elif tmp[0] == 'Atoms':
                    save = 'Atoms'
                    atindex = 0
                    if tmp[1] == '#':
                        atomstyle = tmp[2]
                    else:
                        atomstyle = tmp[1]
                    if atomstyle.lower() == 'atomic':
                        header['ATOMS'] = ['id', 'type', 'xu', 'yu', 'zu', 'nx', 'ny', 'nz']
                        dtype={'names': header['ATOMS'],
                               'formats': ('u4', 'u1', 'f8', 'f8', 'f8', 'i', 'i', 'i')}
                        
                        data = np.array(np.empty(header['NUMBER OF ATOMS']), dtype=dtype)
                        print(data.shape)
                    elif atomstyle.lower() == 'charge':
                        header['ATOMS'] = ['id', 'type', 'charge', 'xu', 'yu', 'zu', 'nx', 'ny', 'nz']
                        dtype={'names': header['ATOMS'],
                               'formats': ('u4', 'u1', 'f8', 'f8', 'f8', 'f8', 'i', 'i', 'i')}
                        
                        data = np.array(np.empty(header['NUMBER OF ATOMS']), dtype=dtype)
                        print(data.shape)
                    elif atomstyle.lower() == 'full':
                        header['ATOMS'] = ['id', 'molid', 'type', 'charge', 'xu', 'yu', 'zu', 'nx', 'ny', 'nz']
                        dtype={'names': header['ATOMS'],
                               'formats': ('u4', 'u1', 'u1', 'f8', 'f8', 'f8', 'f8', 'f8', 'i', 'i', 'i')}
                        
                        data = np.array(np.empty(header['NUMBER OF ATOMS']), dtype=dtype)
                        print(data.shape)
                    else:
                        raise NotImplementedError('Atom style %s not understood' % atomstyle)
                    continue
                elif tmp[0] == 'Velocities':
                    save = 'Velocities'
                    continue
                elif len(tmp) == 2:
                    if tmp[1].lower() == 'atoms':
                        header['NUMBER OF ATOMS'] = int(tmp[0])
                        continue
                elif len(tmp) == 3:
                    if tmp[1].lower() == 'atom' and tmp[2].lower() == 'types':
                        header['NUMBER OF TYPES'] = int(tmp[0])
                        continue
                elif len(tmp) == 4:
                    if tmp[2][1:] == 'lo' and tmp[3][1:] == 'hi':
                        header['BOX BOUNDS'].append(np.array(tmp[:2], dtype='f8'))
                        continue
                    elif tmp[3] == 'xy' and tmp[4] == 'xy' and tmp[5] == 'xy' :
                        header['BOX'][tmp[3]] = np.double(tmp[0])
                        header['BOX'][tmp[4]] = np.double(tmp[1])
                        header['BOX'][tmp[5]] = np.double(tmp[2])
                        continue
                    elif save == 'Velocities' or save == 'Atoms':
                        pass
                    else:
                        raise NotImplementedError('Do not understand %s' % ' '.join(tmp))
#                        save = 'Box'
                
                if save == 'Masses':
                    masses[tmp[0]] = np.double(tmp[1])
                elif save == 'Atoms':
                    if atomstyle.lower() == 'atomic':
                        data[atindex]['id'] = np.int(tmp[0])
                        data[atindex]['type'] = np.int(tmp[1])
                        data[atindex]['xu'] = np.double(tmp[2])
                        data[atindex]['yu'] = np.double(tmp[3])
                        data[atindex]['zu'] = np.double(tmp[4])
                        data[atindex]['nx'] = np.double(tmp[5])
                        data[atindex]['ny'] = np.double(tmp[6])
                        data[atindex]['nz'] = np.double(tmp[7])
                    elif atomstyle.lower() == 'charge':
                        data[atindex]['id'] = np.int(tmp[0])
                        data[atindex]['type'] = np.int(tmp[1])
                        data[atindex]['charge'] = np.double(tmp[2])
                        data[atindex]['xu'] = np.double(tmp[3])
                        data[atindex]['yu'] = np.double(tmp[4])
                        data[atindex]['zu'] = np.double(tmp[5])
                        data[atindex]['nx'] = np.double(tmp[6])
                        data[atindex]['ny'] = np.double(tmp[7])
                        data[atindex]['nz'] = np.double(tmp[8])
                    elif atomstyle.lower() == 'full':
                        data[atindex]['id'] = np.int(tmp[0])
                        data[atindex]['molid'] = np.int(tmp[1])
                        data[atindex]['type'] = np.int(tmp[2])
                        data[atindex]['charge'] = np.double(tmp[3])
                        data[atindex]['xu'] = np.double(tmp[4])
                        data[atindex]['yu'] = np.double(tmp[5])
                        data[atindex]['zu'] = np.double(tmp[6])
                        data[atindex]['nx'] = np.double(tmp[7])
                        data[atindex]['ny'] = np.double(tmp[8])
                        data[atindex]['nz'] = np.double(tmp[9])
                    else:
                        raise NotImplementedError("Don't know how to load data for atomstyle %s" % atomstyle)
                    atindex += 1
                    continue
                elif save == 'Velocities':
                    continue
                else:
                    print(save)
                    print('Line "%s" not interpreted' % ' '.join(tmp))
                
        print(header)
        
        header['BOX BOUNDS'] = np.array(header['BOX BOUNDS'], dtype='f8')
        
        xx = header['BOX BOUNDS'][0,1]-header['BOX BOUNDS'][0,0]
        yy = header['BOX BOUNDS'][1,1]-header['BOX BOUNDS'][1,0]
        zz = header['BOX BOUNDS'][2,1]-header['BOX BOUNDS'][2,0]
        header['BOX']['xx'] = xx
        header['BOX']['yy'] = yy
        header['BOX']['zz'] = zz
        xy = header['BOX']['xy']
        xz = header['BOX']['xz']
        yz = header['BOX']['yz']
        header['BOX']['matrix'] = np.array([[xx, xy, xz], [0.0, yy, yz], [0.0, 0.0, zz]])
        
        # Our code assumes an ordered in data file, but that might not be the case
        data.sort(order='id')
        self.data0 = data
        self.header0 = header
        print(self.header0)
        print(self.data0)
#        sys.exit()
        return 
    
    
    
    def find_attypes_masses(self):
        masses = {}
        
        if self.data0file:
            with open(self.data0file, 'r') as fh:
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
        super().__init__(trj_prop, header0=header0, header=header, data0=data0,
                         data=data, attype2mass=attype2mass)
    
    
    
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





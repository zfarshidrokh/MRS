

#*************#
#   imports   #
#*************#
import numpy as np
import os

from fsl_mrs.core.basis import Basis
from fsl_mrs.utils import mrs_io

from pathlib import Path
import matplotlib.pyplot as plt

from scipy.io import loadmat

# own
from loading.lcmodel import read_LCModel_raw


#***********************************#
#   loading basis as FSL-MRS sets   #
#***********************************#
def loadBasisAsFSL(path2basis, params=None, bw=None, cf=None, fmt=None):
    """
    Function wrapper of FSL-MRS read basis to loading more types of basis sets.
    The bw, cf need to be specified for certain data formats.

    @param path2basis -- The path to the basis set.
    @param params -- The parameters describing the basis (might be easier
                     than separately passing bw and cf).
    @param bw -- The bandwidth of the basis used.
    @param cf -- The central frequency of the basis used.
    @param fmt -- The file format (if not given it is inferred).
                  Attention: text .txt files are treated as JMRUI .txt file
                             if the format fmt='text' is not specified.

    @returns -- The FSL-MRS Basis object.
    """
    if params: bw, cf = params['bandwidth'], params['centralFrequency']

    if fmt:
        # test for given format
        fmts = ['jmrui', 'lcmodel', 'text', 'inspector']
        if fmt.lower() not in fmts:
            print(f'File format should be in {fmts}, but {fmt} was given.')
    
    else:
        # ... or find file format ...
        fmt = os.listdir(path2basis)[0].split(os.extsep)[-1].lower()

    #print(f"Loading basis from {path2basis} with format {fmt}, bandwidth {bw}, central frequency {cf}")  

    # ... and match
    if fmt == 'jmrui' or fmt == 'txt':
        basis = load_JMRUI_basis(path2basis)
        #print(f"Loaded JMRUI basis with names: {basis.names}")
        return basis

    elif fmt == 'lcmodel' or fmt == 'raw' or fmt == 'basis':
        basis = load_LCModel_basis(path2basis, bw, cf)
        #print(f"Loaded LCModel basis with names: {basis.names}")
        return basis

    elif fmt == 'text' or fmt == 'txt':
        basis = load_Text_basis(path2basis, bw, cf)
        #print(f"Loaded text basis with names: {basis.names}")
        return basis

    elif fmt == 'fid-a' or fmt == 'mat':
        basis = load_FID_A_basis(path2basis)
        #print(f"Loaded FID-A basis with names: {basis.names}")
        return basis

    elif fmt == 'inspector' or fmt == 'mat':
        basis = load_INSPECTOR_basis(path2basis)
        #print(f"Loaded INSPECTOR basis with names: {basis.names}")
        return basis

    elif fmt == 'json':
        basis = load_json_basis(path2basis)
        #print(f"Loaded JSON basis with names: {basis.names}")
        return basis

    else:
        print('-------------- Loading failed! --------------\n'
              ' Invalid data type or not implemented yet... \n'
              'File: ' + path2basis)
        
    


#******************************#
#   loading JMRUI basis sets   #
#******************************#
def load_JMRUI_basis(path2basis):
    """
    Loads JMRUI basis sets.

    @param path2basis -- The path to the basis set.

    @returns -- The FSL-MRS Basis object.
    """
    #print(f"Loading JMRUI basis from {path2basis}")
    return mrs_io.read_basis(path2basis)


#********************************#
#   loading LCModel basis sets   #
#********************************#
def load_LCModel_basis(path2basis, bw=None, cf=None):
    """
    Loads both .RAW and .BASIS LCModel basis sets, bw and cf need to be given for .RAW.

    @param path2basis -- The path to the basis set.
    @param bw -- The bandwidth of the basis used.
    @param cf -- The central frequency of the basis used.

    @returns -- The FSL-MRS Basis object.
    """
    fmt = os.listdir(path2basis)[0].split(os.extsep)[-1].lower()
    files = list(Path(path2basis).glob('*'))

    #print(f"Loading LCModel basis from {path2basis} with format {fmt}")

    if fmt == 'raw':
        basis = []
        names = []
        for file in files:
            data, header = read_LCModel_raw(file)

            name = os.path.splitext(os.path.split(file)[-1])[-2]

            names.append(name)
            basis.append(data)

        basis = np.asarray(basis).astype(complex).T

        names = [n + '.raw' for n in names]

        # create header
        header = {'centralFrequency': cf,
                  'bandwidth': bw,
                  'dwelltime': 1 / bw,
                  'fwhm': None}
        headers = [header for _ in names]

        #print(f"LCModel basis names: {names}")
        

        return Basis(basis, names, headers)

    elif fmt == 'basis':
        # basis format contains all sets in one file
        basis, names, headers = mrs_io.lcm_io.readLCModelBasis(files[0])

        # add fwhm element to header
        for header in headers:
            header['fwhm'] = None
        '''
        print(f"LCModel basis names: {names}")
        print(f"Header Information:\n"
              f"Bandwidth (Hz): {headers[0]['bandwidth']}\n"
              f"Central Frequency (Hz): {headers[0]['centralFrequency']}\n"
              f"Dwell Time (s): {headers[0]['dwelltime']}\n"
              f"Number of Points: {basis.shape[0]}")
        '''

        return Basis(basis, names, headers)

    else:
        print('\n------------- Loading failed! -------------\n'
              'Invalid data type or not implemented yet...\n'
              'File: ' + path2basis)


#*****************************#
#   loading text basis sets   #
#*****************************#
def load_Text_basis(path2basis, bw, cf):
    """
    Loads plain text file basis sets.

    @param path2basis -- The path to the basis set.
    @param bw -- The bandwidth of the basis used.
    @param cf -- The central frequency of the basis used.

    @returns -- The FSL-MRS Basis object.
    """
    print(f"Loading text basis from {path2basis} with bandwidth {bw} and central frequency {cf}")
    names = os.listdir(path2basis)

    basis = []
    headers = []
    for name in names:
        data  = np.loadtxt(path2basis + '/' + name)

        # create header
        header = {'centralFrequency': cf,
                  'bandwidth': bw,
                  'dwelltime': 1 / bw,
                  'fwhm': None}

        basis.append(data)
        headers.append(header)

    # transform 2 complex
    basis = np.array(basis)
    basis = basis[:, :, 0] + 1j * basis[:, :, 0]

    #print(f"Text basis names: {names}")
    return Basis(basis, names, headers)


#**********************************#
#   loading INSPECTOR basis sets   #
#**********************************#
def load_INSPECTOR_basis(path2basis):
    """
    Loads MATLAB file basis sets for INSPECTOR.

    @param path2basis -- The path to the basis set.

    @returns -- The FSL-MRS Basis object.
    """
    #print(f"Loading INSPECTOR basis from {path2basis}")
    names = os.listdir(path2basis)

    basis = []
    headers = []
    for name in names:
        try:
            data  = loadmat(path2basis + '/' + name)
            # create header
            header = {'centralFrequency': data['exptDat']['sf'][0][0],
                      'bandwidth': data['exptDat']['sw_h'][0][0],
                      'dwelltime': 1 / data['exptDat']['sw_h'][0][0],
                      'fwhm': None}

            basis.append(data['exptDat']['fid'][0][0])
            headers.append(header)
        except: print('Failed loading: ' + name + ' ...')
    #print(f"INSPECTOR basis names: {names}")
    return Basis(np.squeeze(basis)[:, :], names, headers)


#******************************#
#   loading FID-A basis sets   #
#******************************#
def load_FID_A_basis(path2basis):
    """
    Loads MATLAB file basis sets simulated with FID-A.

    @param path2basis -- The path to the basis set.

    @returns -- The FSL-MRS Basis object.
    """
    #print(f"Loading FID-A basis from {path2basis}")
    names = os.listdir(path2basis)

    basis = []
    headers = []
    for name in names:
        if name.split('.')[-1] == 'mat':
            data  = loadmat(path2basis + '/' + name)

            # create header
            header = {'centralFrequency': float(data['txfrq']),
                      'bandwidth': int(data['spectralwidth']),
                      'dwelltime': float(data['dwelltime']),
                      'fwhm': None}

            basis.append(data['fids'])
            headers.append(header)
    #print(f"FID-A basis names: {names}")
    return Basis(np.squeeze(basis)[:, :], names, headers)


#*****************************#
#   loading json basis sets   #
#*****************************#
def load_json_basis(path2basis):
    """
    Loads json file basis sets.

    @param path2basis -- The path to the basis set.

    @returns -- The FSL-MRS Basis object.
    """
    #print(f"Loading JSON basis from {path2basis}")
    basis, names, headers = mrs_io.fsl_io.readFSLBasisFiles(path2basis)
    #print(f"JSON basis names: {names}")
    return Basis(basis, names, headers)

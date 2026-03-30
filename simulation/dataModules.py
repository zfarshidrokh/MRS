

#*************#
#   imports   #
#*************#
import numpy as np
import os
import pandas as pd
import pytorch_lightning as pl
import torch

from torch.utils.data import DataLoader,TensorDataset

from spec2nii.Philips.philips import read_sdat, read_spar
from spec2nii.Philips.philips_data_list import _read_list

# own
from loading.loadData import loadDataAsFSL, loadDataSetsAsFSL, load_mat_data
from loading.loadBasis import loadBasisAsFSL
from loading.loadConc import loadConcsDir
from loading.philips import read_Philips_data
from simulation.basis import Basis
from simulation.sigModels import VoigtModel
from simulation.simulation import simulateParam, simulateRW, simulatePeaks, paramsRWP, AlexConcs
from utils.auxiliary import processBasis, processSpectra, randomWalk, randomPeak
import matplotlib.pyplot as plt

#**************************************************************************************************#
#                                       Class SynthDataModule                                      #
#**************************************************************************************************#
#                                                                                                  #
# The data module to load synthetic data.                                                          #
#                                                                                                  #
#**************************************************************************************************#
class SynthDataModule(pl.LightningDataModule):

    #*************************#
    #   initialize instance   #
    #*************************#
    def __init__(self, basis_dir, nums_test, sigModel=None, params=paramsRWP, concs=AlexConcs,
                 basisFmt='7tslaser', specType='synth', basis_dir2=None):
        super().__init__()
        self.basisObj = Basis(basis_dir, fmt=basisFmt, path2basis2=basis_dir2)
        #print(f"Shape of self.basisObj.fids after reformat: {self.basisObj.fids.shape}")
        self.basis = processBasis(self.basisObj.fids)
        self.nums = nums_test

        self.params = params
        self.concs = concs

        if sigModel:
            self.sigModel = sigModel
        else:
            self.sigModel = VoigtModel(basis=self.basis, first=0, last=self.basisObj.n,
                                       t=self.basisObj.t, f=self.basisObj.f)


    #******************************#
    #   simulate a batch of data   #
    #******************************#
    def get_batch(self, batch):
        #print(f"Shape of self.basisObj.fids before simulateParam: {self.basisObj.fids.shape}")
        theta, noise = simulateParam(self.basisObj, batch, self.params, self.concs)
        #print(f"Shape of theta: {theta.shape}")
        #print(f"Shape of noise: {noise.shape}")
        spec, bl = self.sigModel.forward(torch.from_numpy(theta), sumOut=False, baselineOut=True)
        

        cleanSpec = processSpectra(spec, self.basis)
        spec = spec.sum(-1) + torch.from_numpy(noise) + bl #add noise and baseline to it 
        
        spec = processSpectra(spec, self.basis)

        # add baseline signal as a training target
        bl = torch.from_numpy(np.stack((np.real(bl), np.imag(bl)), axis=1))
        cleanSpec = torch.cat((cleanSpec, bl[..., np.newaxis]), dim=-1)

        # add noise signal as a training target (if no more unpredictable signals are added)
        noise = torch.from_numpy(np.stack((np.real(noise), np.imag(noise)), axis=1))

        # add all unpredictable signals as one training target
        cleanSpec = torch.cat((cleanSpec, noise[..., np.newaxis]), dim=-1)

        # add a random walk to the spectra and as a training target
        if 'scale' in self.params and 'smooth' in self.params and 'limits' in self.params:
            # TODO: implement random walk as a PyTorch function and vectorize
            scale, smooth, lowLim, highLim = simulateRW(batch, self.params)
            rw = [randomWalk(waveLength=self.basis.shape[0], scale=scale[i], smooth=smooth[i],
                             ylim=[lowLim[i], highLim[i]]) for i in range(batch)]
            scale, smooth, lowLim, highLim = simulateRW(batch, self.params)
            rwCplx = [randomWalk(waveLength=self.basis.shape[0], scale=scale[i], smooth=smooth[i],
                                 ylim=[lowLim[i], highLim[i]]) for i in range(batch)]
            rw = torch.from_numpy(np.stack((np.array(rw), np.array(rwCplx)), axis=1))
            spec += rw   # add to signal
            cleanSpec[..., -1] += rw   # add to artificial channel

        # add random peaks with amplitude, linewidth, and phasing
        if 'numPeaks' in self.params and 'peakAmp' in self.params and \
                'peakWidth' in self.params and 'peakPhase' in self.params:
            nums = np.random.randint(self.params['numPeaks'][0], self.params['numPeaks'][1], batch)

            peaks = np.zeros((batch, self.basis.shape[0]), dtype=np.complex128)
            for i in range(batch):
                if nums[i] > 0:
                    pos = np.random.randint(0, self.basis.shape[0], (nums[i], 1))
                    amps, widths, phases = simulatePeaks(self.basis, nums[i], self.params)

                    peaks[i] = randomPeak(waveLength=self.basis.shape[0], batch=nums[i],
                                          amp=amps, pos=pos, width=widths, phase=phases).sum(0)

            peaks = torch.from_numpy(np.stack((np.real(peaks), np.imag(peaks)), axis=1))
            spec += peaks   # add to signal
            cleanSpec[..., -1] += peaks   # add to artificial channel

        return spec, cleanSpec, theta


    #**************************#
    #   create test data set   #
    #**************************#
    def test_dataloader(self):
        x, y, t = self.get_batch(self.nums)

        data = []
        for i in range(self.nums):
            data.append([x[i], y[i], t[i]])
        return DataLoader(data, num_workers=4, batch_size=self.nums)



#**************************************************************************************************#
#                                     Class ChallengeDataModule                                    #
#**************************************************************************************************#
#                                                                                                  #
# The data module to load synthetic data of the ISMRM 2016 Fitting Challenge.                      #
#                                                                                                  #
#**************************************************************************************************#
class ChallengeDataModule(pl.LightningDataModule):

    #*************************#
    #   initialize instance   #
    #*************************#
    def __init__(self,
                 data_dir='../Data/DataSets/ISMRM_challenge/datasets_JMRUI_WS/',
                 basis_dir='../Data/BasisSets/ISMRM_challenge/basisset_JMRUI/',
                 truth_dir='../Data/DataSets/ISMRM_challenge/ground_truth/',
                 nums_cha=None, pre_pro=True):
        super().__init__()
        self.nums_cha = nums_cha
        self.pre_pro = pre_pro

        self.basis = loadBasisAsFSL(basis_dir)
        self.data = loadDataSetsAsFSL(data_dir)
        self.concs, self.crlbs = loadConcsDir(truth_dir)


    #**************************#
    #   create test data set   #
    #**************************#
    def test_dataloader(self):
        testY = [[value for key, value in c.items() if key in self.basis.names]
                 for c in self.concs]
        dataSet = []
        for i, d in enumerate(self.data):
            if self.pre_pro:   # preprocess data according to FSL using the basis
                d.basis = self.basis
                d.processForFitting()

            # fft and normalize
            spec = np.fft.fft(d.FID)
            x = processSpectra(torch.from_numpy(spec[np.newaxis, :]))

            dataSet.append([x[0].float(), x[0, ..., np.newaxis].float(), torch.FloatTensor(testY[i])])
        return DataLoader(dataSet[:self.nums_cha], num_workers=4, batch_size=i+1)



#**************************************************************************************************#
#                                      Class InVivoDataModule                                      #
#**************************************************************************************************#
#                                                                                                  #
# The data module to load in-vivo data.                                                            #
#                                                                                                  #
#**************************************************************************************************#

class InVivoDataModule(pl.LightningDataModule):

        #*************************#
        #   initialize instance   #
        #*************************#
        def __init__(self,
                    path2data,
                    basis_dir,
                    nums_test=None, pre_pro=True, verbose=0, fmt=None):
            super().__init__()
            self.nums_test = nums_test
            self.pre_pro = pre_pro
            self.path2data = path2data
            self.basis_dir = basis_dir
            self.fmt = fmt
            self.basis = loadBasisAsFSL(basis_dir)

            # Load all data with a single call to load_MRSI_LCModel_coraw
            self.data = loadDataAsFSL(path2data, fmt=self.fmt)
            if isinstance(self.data, dict):
                #print(f"Keys in self.data: {self.data.keys()}")
                if 'coraw' in self.data:
                    print(f"Shape of 'coraw' data: {self.data['coraw'].shape}")
                else:
                    print("'coraw' key not found in self.data")
            else:
                print(f"Content of self.data: {self.data}")
            self.refs = []  # Optionally, load references separately if needed

            if verbose:
                print(f"Loaded data with shape {self.data['coraw'].shape}")
        



            '''
            self.nums_test = nums_test
            self.pre_pro = pre_pro

            self.data_dir = data_dir

            self.basis_dir = basis_dir
            self.basis = loadBasisAsFSL(basis_dir)

            self.data = []
            self.refs = []
            # go through files in folder
            for file in os.listdir(data_dir):
                # if file is a folder
                if os.path.isdir(data_dir + '/' + file):
                    # go through files in sub-folder
                    for sub_file in os.listdir(data_dir + '/' + file):
                        # load fid
                        fid = loadDataAsFSL(data_dir + '/' + file + '/' + sub_file)
                        if not isinstance(fid, type(None)):
                            if 'act' in sub_file.lower():
                                if verbose: self.visualize(fid)
                                self.data.append(fid)
                            elif 'ref' in sub_file.lower():
                                self.refs.append(fid)
                else:   # load fid
                    fid = loadDataAsFSL(data_dir + '/' + file)
                    if not isinstance(fid, type(None)):
                        if 'act' in file.lower():
                            if verbose: self.visualize(fid)
                            self.data.append(fid)
                           
                        elif 'ref' in file.lower():
                            self.refs.append(fid)
            print(self.data)
            
        '''

        #**************************#
        #   create test data set   #
        #**************************#
        def test_dataloader(self):
            dataSet = []

            # Access the 'coraw' key in the dictionary
            coraw_data = self.data['coraw']

            # Read CSV to get row and column indices
            csv_file_path = r'C:\Users\zfarshid\Downloads\major project\article\forStudents 2\forStudents\Data\alex_data\transfer_2802572_files_4511555e\v7071\top\extracted_data.csv'
            df = pd.read_csv(csv_file_path)

            if 'Row' not in df.columns or 'Col' not in df.columns:
                raise ValueError("CSV file must contain 'Row' and 'Col' columns")

            # Randomly select rows and columns based on nums_test
            if self.nums_test > len(df):
                raise ValueError("nums_test cannot be greater than the number of rows in the CSV file")

            sampled_df = df.sample(n=self.nums_test)  # Adjust random_state for reproducibility
            rows = sampled_df['Row'].values
            cols = sampled_df['Col'].values
            print(f"Sampled rows: {rows}")
            print(f"Sampled columns: {cols}")

            # Ensure rows and cols are within bounds of the coraw_data dimensions
            num_rows, num_cols, _ = coraw_data.shape
            rows = [r for r in rows if 0 <= r < num_rows]
            cols = [c for c in cols if 0 <= c < num_cols]

            # Create dataset from the selected rows and columns
            for row, col in zip(rows, cols):
                d = coraw_data[row, col, :]  # Extract the relevant slice of data

                if self.pre_pro:
                    # Apply any preprocessing steps required
                    pass

                # FFT and normalize
                spec = np.fft.fft(d)
                x = processSpectra(torch.from_numpy(spec[np.newaxis, :]))

                dataSet.append(x[0].float())

            if len(dataSet) > 0:
                data_tensor = torch.stack(dataSet)
                dataset = TensorDataset(data_tensor)
            else:
                dataset = TensorDataset(torch.empty((0, coraw_data.shape[2])))

            return DataLoader(dataset, num_workers=4, batch_size=len(dataSet))



    

#**************************************************************************************************#
#                                       Class InVivoNSAModule                                      #
#**************************************************************************************************#
#                                                                                                  #
# The data module to load in-vivo data with individual NSA.                                        #
#                                                                                                  #
#**************************************************************************************************#
class InVivoNSAModule(pl.LightningDataModule):

        #*************************#
        #   initialize instance   #
        #*************************#
        def __init__(self,
                    data_dir='../Data/DataSets/fMRSinPain/SUBJECTS/SUBJECTS/',
                    basis_dir='../Data/BasisSets/ISMRM_challenge/basisset_JMRUI/',
                    basis_fmt='', nums_test=None, fMRS=False, verbose=0):
            super().__init__()
            self.nums_test = nums_test
            self.fMRS = fMRS

            self.data_dir = data_dir
            self.basis = Basis(basis_dir, fmt=basis_fmt)

            self.data = []
            self.refs = []

            # go through files in folder
            for file in os.listdir(data_dir):
                # if file is a folder
                if os.path.isdir(data_dir + '/' + file):
                    # go through files in sub-folder
                    for sub_file in os.listdir(data_dir + '/' + file):

                        # load fid
                        if sub_file.lower().split('.')[-1] == 'list':
                            fids, h2o = self.load_DATALIST_data(data_dir + '/' + file + '/' + sub_file)

                            if self.fMRS and len(fids.shape) > 2:  # fMRS dim: (samples, NSA, f-blocks)
                                self.data.append(fids)
                                self.refs.append(h2o)
                            if not self.fMRS and len(fids.shape) <= 2:   # (samples, NSA)
                                self.data.append(fids)
                                self.refs.append(h2o)

                        elif sub_file.lower().split('.')[-1] == 'sdat' or sub_file.lower().split('.')[-1] == 'spar':
                            if 'ref' in sub_file.lower():
                                h2o = self.load_SDATSPAR_data(data_dir + '/' + file + '/' + sub_file)
                                self.refs.append(h2o)
                            else:
                                fids = self.load_SDATSPAR_data(data_dir + '/' + file + '/' + sub_file)
                                self.data.append(fids)


        #****************************#
        #   loading DATA LIST data   #
        #****************************#
        def load_DATALIST_data(self, path2data):
            df, num_dict, coord_dict, os_dict = _read_list(path2data[:-4] + 'list')
            sorted_data_dict = read_Philips_data(path2data[:-4] + 'data', df)

            fids = sorted_data_dict['STD_0'].squeeze()
            h2o = sorted_data_dict['STD_1'].squeeze()

            # combine channels
            fids = fids.sum(1)
            h2o = h2o.sum(1)
            return fids, h2o


        #****************************#
        #   loading SDAT SPAR data   #
        #****************************#
        def load_SDATSPAR_data(self, path2data):
            params = read_spar(path2data[:-4] + 'SPAR')
            data = read_sdat(path2data[:-4] + 'SDAT',
                             params['samples'],
                             params['rows'])
            return data


        #**************************#
        #   create test data set   #
        #**************************#
        def test_dataloader(self):
            dataSet = []
            for fid in self.data:
                # fft and normalize
                spec = np.fft.fft(fid, axis=-2 - int(self.fMRS))
                assert spec.shape[-2 - int(self.fMRS)] == self.basis.fids.shape[0]
                x = processSpectra(torch.from_numpy(spec[np.newaxis, :]))
                dataSet.append([x[0].float()])
            return DataLoader(dataSet, num_workers=4)

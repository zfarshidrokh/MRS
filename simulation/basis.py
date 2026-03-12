####################################################################################################
#                                             basis.py                                             #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 03/02/23                                                                                #
#                                                                                                  #
# Purpose: Defines a main structure for MRS metabolite basis sets. Encapsulates all information    #
#          and holds definitions to compute various aspects.                                       #
#                                                                                                  #
####################################################################################################



import numpy as np
import matplotlib.pyplot as plt  # For plotting
from scipy.interpolate import interp1d  # For interpolation in omit_peaks function

# Assuming loadBasisAsFSL is a custom function from a module named 'loading'
from loading.loadBasis import loadBasisAsFSL  # For loading basis sets



class Basis:
    
    # Initialize instance
    def __init__(self, path2basis, fmt='', path2basis2=None):
        """
        Main init for the Basis class.

        @param path2basis -- The path to the basis set folder.
        @param fmt -- The data format to be selected.
        """
        basis = loadBasisAsFSL(path2basis)
        print(f"Loaded basis fids shape: {basis._raw_fids.shape}")
        print(f"Loaded basis names: {basis.names}")

        if fmt.lower() == 'biggaba':
            basis = self.reformat(basis, 2000, 2048, ignore=['Cit', 'EtOH', 'Phenyl', 'Ser', 'Tyros', 'bHB', 'bHG'])
        elif fmt.lower() == 'fmrsinpain':
            basis = self.reformat(basis, 2000, 2048, ignore=['CrCH2', 'EA', 'H2O', 'Ser'])
        elif fmt.lower() == '7tslaser':
           
            basis = self.reformat(basis, 3000, 512)
            #print(basis._raw_fids.shape)

            basis = self.omit_peaks(basis)
            basis = self.correct_offset(basis)

            if path2basis2:
                # Load the other basis for rescaling using loadBasisAsFSL
                rescale_basis = loadBasisAsFSL(path2basis2)
                rescale_basis = self.omit_peaks_julian(rescale_basis)
                rescale_basis = self.correct_offset(rescale_basis)
                #self.plot_basis_signals(rescale_basis, title="Rescale Basis After Offset Correction")
                
                    
                # Rescale the current basis using the loaded rescale basis
                basis = self.rescale_using_basis(basis, rescale_basis)
            
            
            #self.plot_basis_signals(basis, title="second reformat")
          
            #basis = self.omit_peaks(basis)
            basis = self.correct_offset(basis)
            basis = self.omit_peaks(basis)
            #print(basis._raw_fids.shape)
            print(f"Loaded basis fids shape after reformat: {basis._raw_fids.shape}")
            print(f"Loaded basis names after reformat: {basis.names}")
            

        self.basisFSL = basis
        #print(f'basisFSL: {self.basisFSL}')

        if '.' in basis.names[0]:
            self.names = [n[:-4] for n in basis.names]
        else:
            self.names = basis.names
        perms = sorted(range(len(self.names)), key=lambda k: self.names[k])
        self.names = [self.names[i] for i in perms]
        #print("Reordered Names: ", self.names)
        self.n_metabs = len(self.names)
        self.fids = basis._raw_fids[:, perms]
        #print(f'Initialized fids: {self.fids[:10]}')
        self.bw = basis.original_bw
        self.dwelltime = float(basis.original_dwell)
        self.n = basis.original_points
        self.t = np.arange(self.dwelltime, self.dwelltime * (self.n + 1), self.dwelltime)
        self.f = np.arange(- self.bw / 2, self.bw / 2, self.bw / self.n)
        self.ppm = basis.original_ppm_axis
        self.cf = float(basis.cf)
        
    
    def plot_basis_signals(self, basis, title="Basis Signals", linewidth=2):
        """
        Plot the Fourier Transform of the basis signals to visualize the signals in the frequency domain.
        """
        basis_signals = []
        ppm_range = basis.original_ppm_axis + 4.65  # Shift for 1H reference

        # Iterate over each signal in the basis
        for fid in basis._raw_fids.T:  # Transpose fids for correct shape
            signal = np.fft.fftshift(np.fft.fft(fid))
            basis_signals.append(signal)

        # Plot all signals in separate figures
        for signal, label in zip(basis_signals, basis.names):
            plt.figure(figsize=(12, 6))
            plt.plot(ppm_range, signal.real, label=label, linewidth=linewidth)
            plt.title(f"{title}: {label}")
            plt.xlabel('PPM')
            plt.ylabel('Magnitude')
            plt.legend(loc='upper right')
            plt.grid(True)
            plt.gca().invert_xaxis()  # Invert PPM axis
            plt.show()
    
    
        
        

    def reformat(self, basis, bw, points, ignore=[]):
        """
        Reformat the basis set to a given bandwidth and number of points.

        @param basis -- The basis set to reformat.
        @param bw -- The bandwidth to reformat to.
        @param points -- The number of points to reformat to.
        @param ignore -- The metabolites to ignore.

        @returns -- The reformatted basis set.
        """
        basis._raw_fids = basis.get_formatted_basis(bw, points, ignore=ignore)
        basis._raw_fids /= np.mean(
            np.abs(basis._raw_fids))  # normalizes to the ISMRM 2016 challenge data
        basis._dt = 1. / bw
        basis._names = basis.get_formatted_names(ignore=ignore)
        return basis
    
    def omit_peaks(self, basis, linewidth=2, zero_range=(-0.3, 0.3), omit_range=(4.8, 9.5)):
        """
        Omit peaks near zero PPM and in a specified PPM range from the basis signals using interpolation.

        @param basis -- The basis to modify.
        @param linewidth -- Optional linewidth for plotting.
        @param zero_range -- The PPM range around zero to omit peaks (default: -0.75 to 0.75 PPM).
        @param omit_range -- The PPM range to omit peaks (default: 6 to 8 PPM).
        
        @returns -- The modified basis set.
        """
        ppm_range = basis.original_ppm_axis + 4.65  # Shift for 1H reference

        # Define the valid PPM range: exclude both zero PPM and the specified omit_range
        valid_idx = ((ppm_range < zero_range[0]) | (ppm_range > zero_range[1])) & \
                    ((ppm_range < omit_range[0]) | (ppm_range > omit_range[1]))

        # Iterate over each signal in the basis
        for idx, fid in enumerate(basis._raw_fids.T):  # Transpose fids for correct shape
            signal = np.fft.fftshift(np.fft.fft(fid))

            # Prepare for interpolation by keeping only the valid (non-zero and non-omit) points
            valid_ppm = ppm_range[valid_idx]
            valid_signal = signal[valid_idx]

            # Create an interpolation function using valid points
            interp_func = interp1d(valid_ppm, valid_signal, kind='linear', fill_value="extrapolate")

            # Apply the interpolation to fill the omitted regions
            filtered_signal = signal.copy()
            omitted_idx = ~valid_idx  # The regions to omit (near zero and within omit_range)
            filtered_signal[omitted_idx] = interp_func(ppm_range[omitted_idx])  # Interpolate the omitted regions

            # Move back to time domain and update the basis (similar to correct_offset)
            corrected_signal = np.fft.ifft(np.fft.ifftshift(filtered_signal))
            basis._raw_fids[:, idx] = corrected_signal

        return basis

    

    def omit_peaks_julian(self, basis, linewidth=2, zero_range=(-0.75, 0.75), omit_range=(4.8, 9.5)):
        """
        Omit peaks near zero PPM and in a specified PPM range from the basis signals using interpolation.

        @param basis -- The basis to modify.
        @param linewidth -- Optional linewidth for plotting.
        @param zero_range -- The PPM range around zero to omit peaks (default: -0.75 to 0.75 PPM).
        @param omit_range -- The PPM range to omit peaks (default: 6 to 8 PPM).
        
        @returns -- The modified basis set.
        """
        ppm_range = basis.original_ppm_axis + 4.65  # Shift for 1H reference

        # Define the valid PPM range: exclude both zero PPM and the specified omit_range
        valid_idx = ((ppm_range < zero_range[0]) | (ppm_range > zero_range[1])) & \
                    ((ppm_range < omit_range[0]) | (ppm_range > omit_range[1]))

        # Iterate over each signal in the basis
        for idx, fid in enumerate(basis._raw_fids.T):  # Transpose fids for correct shape
            signal = np.conjugate(fid)
            signal = np.fft.fftshift(np.fft.fft(signal))

            # Prepare for interpolation by keeping only the valid (non-zero and non-omit) points
            valid_ppm = ppm_range[valid_idx]
            valid_signal = signal[valid_idx]

            # Create an interpolation function using valid points
            interp_func = interp1d(valid_ppm, valid_signal, kind='linear', fill_value="extrapolate")

            # Apply the interpolation to fill the omitted regions
            filtered_signal = signal.copy()
            omitted_idx = ~valid_idx  # The regions to omit (near zero and within omit_range)
            filtered_signal[omitted_idx] = interp_func(ppm_range[omitted_idx])  # Interpolate the omitted regions

            # Move back to time domain and update the basis (similar to correct_offset)
            corrected_signal = np.fft.ifft(np.fft.ifftshift(filtered_signal))
            basis._raw_fids[:, idx] = corrected_signal

        return basis

    
    def correct_offset(self, basis):
        """
        Correct the DC offset for the entire basis set by subtracting the mean
        of both the real and imaginary parts of each signal.
        This ensures no part of the real signal goes below zero.
        
        @param basis -- The basis to modify.
        @returns -- The modified basis with corrected DC offset.
        """
        # Iterate over each signal in the basis
        for idx, fid in enumerate(basis._raw_fids.T):  # Transpose fids for correct shape
            signal = np.fft.fftshift(np.fft.fft(fid))  # Move to frequency domain

            # Subtract the mean from both real and imaginary parts
            real_part = np.real(signal)
            imag_part = np.imag(signal)

            real_part -= np.mean(real_part)
            imag_part -= np.mean(imag_part)

            # Ensure that the real part does not dip below zero
            real_part = np.clip(real_part, 0, None)

            # Combine the corrected real and imaginary parts
            corrected_signal = real_part + 1j * imag_part

            # Move back to time domain and update the basis
            basis._raw_fids[:, idx] = np.fft.ifft(np.fft.ifftshift(corrected_signal))

        return basis

    def rescale_using_basis(self, basis, rescale_basis):
        """
        Rescale the current basis FIDs using the min and max of another basis, with FFT applied.
        
        @param basis -- The basis to be rescaled.
        @param rescale_basis -- The reference basis for rescaling.
        
        @returns -- Rescaled basis.
        """
        common_metabolites = set(basis.names).intersection(set(rescale_basis.names))
        uncommon_metabolites = {'MM_09', 'MM17_2', 'MM27_3', 'MM12_1', 'MM30_3','MM27','MM23'}

        # Calculate NAA scaling ratio BEFORE rescaling
        if 'NAA' in basis.names and 'NAA' in rescale_basis.names:
            idx_naa_self = basis.names.index('NAA')
            idx_naa_other = rescale_basis.names.index('NAA')

            # Get NAA max before rescaling
            signal_naa_before = np.fft.fftshift(np.fft.fft(basis._raw_fids[:, idx_naa_self]))
            max_naa_before = np.max(np.abs(signal_naa_before))

            # Get NAA max after rescaling (in rescale_basis)
            signal_naa_after = np.fft.fftshift(np.fft.fft(rescale_basis._raw_fids[:, idx_naa_other]))
            max_naa_after = np.max(np.abs(signal_naa_after))

            # Calculate NAA scaling ratio
            naa_scaling_ratio = max_naa_after / max_naa_before if max_naa_before > 0 else 1.0
        else:
            naa_scaling_ratio = 1.0  # Default scaling factor if NAA is not present

        # Now loop through all metabolites in the basis
        for metabolite in basis.names:
            idx_self = basis.names.index(metabolite)
            signal_self = np.fft.fftshift(np.fft.fft(basis._raw_fids[:, idx_self]))
            max_self = np.max(np.abs(signal_self))

            # If the metabolite is Cho, handle it separately with Cr scaling
            if metabolite == 'Cho' and 'Cr' in basis.names and 'Cr' in rescale_basis.names:
                idx_cr_self = basis.names.index('Cr')
                idx_cr_other = rescale_basis.names.index('Cr')

                # Get Cr signals
                signal_cr_self = np.fft.fftshift(np.fft.fft(basis._raw_fids[:, idx_cr_self]))
                signal_cr_other = np.fft.fftshift(np.fft.fft(rescale_basis._raw_fids[:, idx_cr_other]))

                # Calculate the current Cr and Cho max values
                max_cr_self = np.max(np.abs(signal_cr_self))
                max_cho_self = np.max(np.abs(signal_self))

                # Calculate current Cho/Cr ratio
                if max_cr_self > 0:
                    current_cho_cr_ratio = max_cho_self / max_cr_self
                else:
                    current_cho_cr_ratio = 0  # Avoid division by zero

                # Target Cho/Cr ratio is 0.24
                target_cho_cr_ratio = 0.31

                # Scaling factor for Cho based on Cr
                cho_scaling_factor = target_cho_cr_ratio / current_cho_cr_ratio if current_cho_cr_ratio > 0 else 1.0

                # Rescale Cho
                scaled_signal_cho = signal_self * cho_scaling_factor
                basis._raw_fids[:, idx_self] = np.fft.ifft(np.fft.ifftshift(scaled_signal_cho))

            elif metabolite == 'mI' and 'Cr' in basis.names and 'Cr' in rescale_basis.names:
                idx_cr_self = basis.names.index('Cr')
                idx_cr_other = rescale_basis.names.index('Cr')

                # Get Cr signals
                signal_cr_self = np.fft.fftshift(np.fft.fft(basis._raw_fids[:, idx_cr_self]))
                signal_cr_other = np.fft.fftshift(np.fft.fft(rescale_basis._raw_fids[:, idx_cr_other]))

                # Calculate the current Cr and Cho max values
                max_cr_self = np.max(np.abs(signal_cr_self))
                max_mI_self = np.max(np.abs(signal_self))

                # Calculate current Cho/Cr ratio
                if max_cr_self > 0:
                    current_mI_cr_ratio = max_mI_self / max_cr_self
                else:
                    current_mI_cr_ratio = 0  # Avoid division by zero

                # Target mI/Cr ratio is 0.24
                target_mI_cr_ratio = 0.6265

                # Scaling factor for Cho based on Cr
                mI_scaling_factor = target_mI_cr_ratio / current_mI_cr_ratio if current_mI_cr_ratio > 0 else 1.0

                # Rescale Cho
                scaled_signal_mI = signal_self * mI_scaling_factor
                basis._raw_fids[:, idx_self] = np.fft.ifft(np.fft.ifftshift(scaled_signal_mI))

            # Handle PC separately with Cr scaling (PC/Cr target ratio is 0.15)
            elif metabolite == 'PC' and 'Cr' in basis.names and 'Cr' in rescale_basis.names:
                idx_cr_self = basis.names.index('Cr')
                idx_cr_other = rescale_basis.names.index('Cr')

                # Get Cr signals
                signal_cr_self = np.fft.fftshift(np.fft.fft(basis._raw_fids[:, idx_cr_self]))
                signal_cr_other = np.fft.fftshift(np.fft.fft(rescale_basis._raw_fids[:, idx_cr_other]))

                # Calculate the current Cr and PC max values
                max_cr_self = np.max(np.abs(signal_cr_self))
                max_pc_self = np.max(np.abs(signal_self))

                # Calculate current PC/Cr ratio
                if max_cr_self > 0:
                    current_pc_cr_ratio = max_pc_self / max_cr_self
                else:
                    current_pc_cr_ratio = 0  # Avoid division by zero

                # Target PC/Cr ratio is 0.15
                target_pc_cr_ratio = 0.15

                # Scaling factor for PC based on Cr
                pc_scaling_factor = target_pc_cr_ratio / current_pc_cr_ratio if current_pc_cr_ratio > 0 else 1.0

                # Rescale PC
                scaled_signal_pc = signal_self * pc_scaling_factor
                basis._raw_fids[:, idx_self] = np.fft.ifft(np.fft.ifftshift(scaled_signal_pc))


            # If it's a common metabolite, rescale based on its max values
            elif metabolite in common_metabolites:
                idx_other = rescale_basis.names.index(metabolite)
                signal_other = np.fft.fftshift(np.fft.fft(rescale_basis._raw_fids[:, idx_other]))

                # Rescale using max comparison
                magnitude_self = np.abs(signal_self)
                magnitude_other = np.abs(signal_other)

                max_self = np.max(magnitude_self)
                max_other = np.max(magnitude_other)

                scaling_factor = max_other / max_self if max_self > 0 else 1.0

                # Rescale
                scaled_signal = signal_self * scaling_factor
                basis._raw_fids[:, idx_self] = np.fft.ifft(np.fft.ifftshift(scaled_signal))

            # If it's an uncommon metabolite, rescale using NAA ratio
            elif metabolite in uncommon_metabolites:
                #print(uncommon_metabolites)
                scaled_signal = signal_self * naa_scaling_ratio
                basis._raw_fids[:, idx_self] = np.fft.ifft(np.fft.ifftshift(scaled_signal))

        return basis



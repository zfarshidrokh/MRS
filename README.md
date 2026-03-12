MRS Spectral Simulation and Modeling

This repository provides a framework for simulating Magnetic Resonance Spectroscopy (MRS) signals, generating synthetic spectra, and modeling them using physics-inspired signal models. The project integrates FSL-MRS, PyTorch, and custom simulation modules to create datasets for training and evaluating spectroscopy analysis methods.

The codebase includes tools for:

Loading and preprocessing metabolite basis sets

Simulating realistic MRS spectra

Modeling spectra using a Voigt signal model

Generating synthetic datasets for machine learning experiments

Loading and processing in-vivo or challenge datasets

The primary goal is to support algorithm development and validation for MRS quantification.


## Project Structure
```
├── loadBasis.py        # Load MRS basis sets from multiple formats

├── basis.py            # Basis class for handling metabolite basis sets

├── sigModels.py        # Signal models (e.g., Voigt model)

├── simulation.py       # Parameter simulation for synthetic spectra

├── simulationDefs.py   # Concentration ranges and simulation settings

├── dataModules.py      # Data loaders and dataset generation modules
```

##Installation
```
pip install numpy scipy torch pytorch-lightning matplotlib pandas
pip install fsl-mrs
```

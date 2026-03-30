


#*************#
#   imports   #
#*************#
import numpy as np


#**************************#
#   concentration ranges   #
#**************************#
stdConcsNorm = {
    # due to different naming conventions some metabolites
    # might be defined multiple times
    'Ace': {'name': 'Ace', 'low_limit': 0, 'up_limit': 0},        # Acetate
    'Ala': {'name': 'Ala', 'low_limit': 0.1, 'up_limit': 1.5},    # Alanine
    'Asc': {'name': 'Asc', 'low_limit': 0, 'up_limit': 0},        # Ascorbate
    'Asp': {'name': 'Asp', 'low_limit': 1.0, 'up_limit': 2.0},    # Aspartate
    'Cr': {'name': 'Cr', 'low_limit': 4.5, 'up_limit': 10.5},     # Creatine
    'GABA': {'name': 'GABA', 'low_limit': 1.0, 'up_limit': 2.0},  # y-aminobutyric acid
    'Glc': {'name': 'Glc', 'low_limit': 1.0, 'up_limit': 2.0},    # Glucose
    'Gln': {'name': 'Gln', 'low_limit': 3.0, 'up_limit': 6.0},    # Glutamine
    'Glu': {'name': 'Glu', 'low_limit': 6.0, 'up_limit': 12.5},   # Glutamate
    'Gly': {'name': 'Gly', 'low_limit': 0, 'up_limit': 0},        # Glycine
    'GPC': {'name': 'GPC', 'low_limit': 0.5, 'up_limit': 2.0},    # Glycero-phosphocholine
    'GSH': {'name': 'GSH', 'low_limit': 1.5, 'up_limit': 3.0},    # Glutathione
    'Ins': {'name': 'Ins', 'low_limit': 0, 'up_limit': 0},        #
    'Lac': {'name': 'Lac', 'low_limit': 0.2, 'up_limit': 1.0},    # Lactate
    'Mac': {'name': 'Mac', 'low_limit': 1.0, 'up_limit': 2.0},    # Macromolecules
    'mI': {'name': 'mI', 'low_limit': 4.0, 'up_limit': 9.0},      # Myo-inositol
    'NAA': {'name': 'NAA', 'low_limit':7.5, 'up_limit': 17.0},    # N-Acetylaspartic Acid
    'NAAG': {'name': 'NAAG', 'low_limit': 0.5, 'up_limit': 2.5},  # N-Acetylaspartylglutamic Acid
    'PCho': {'name': 'PCho', 'low_limit': 0, 'up_limit': 0},      #
    'PC': {'name': 'PC', 'low_limit': 0.5, 'up_limit': 2.0},      #
    'PCr': {'name': 'PCr', 'low_limit': 3.0, 'up_limit': 5.5},    # Phosphocreatine
    'PE': {'name': 'PE', 'low_limit': 1.0, 'up_limit': 2.0},      # Phosphorylethanolamine
    'Tau': {'name': 'Tau', 'low_limit': 2.0, 'up_limit': 6.0},    # Taurine
    'sIns': {'name': 'sIns', 'low_limit': 0.0, 'up_limit': 0.0},  # Scyllo-inositol

    'Ch': {'name': 'Ch', 'low_limit': 0.0, 'up_limit': 0.0},      # Choline
    'Cho': {'name': 'Cho', 'low_limit': 0.0, 'up_limit': 0.0},    # Choline
    'Eth': {'name': 'Eth', 'low_limit': 0.0, 'up_limit': 0.0},    # Ethanolamine
    'Hom': {'name': 'Hom', 'low_limit': 0.0, 'up_limit': 0.0},    # Homocarnosine

    'Glx': {'name': 'Glx', 'low_limit': 9.0, 'up_limit': 18.5},
    'tCho': {'name': 'tCho', 'low_limit': 1.0, 'up_limit': 4.0},
    'tCr': {'name': 'tCr', 'low_limit': 7.5, 'up_limit': 16.0},
    'tNAA': {'name': 'tNAA', 'low_limit': 8.0, 'up_limit': 19.5},

    'Scyllo': {'name': 'Scyllo', 'low_limit': 0., 'up_limit': 0},
    'sI': {'name': 'sIns', 'low_limit': 0., 'up_limit': 0},  # Scyllo-inositol
    'PCh': {'name': 'PCh', 'low_limit': 0, 'up_limit': 0},  #
    'MM_CMR': {'name': 'MM_CMR', 'low_limit': 1., 'up_limit': 2.0},  # Macromolecules
}


#**************************#
#   concentration ranges   #
#**************************#
unifConcs = {key: {'low_limit': 0., 'up_limit': 20} for key in stdConcsNorm.keys()}


#**************************#
#   concentration ranges   #
#**************************#
customConcs = {
    'Ace': {'name': 'Ace', 'low_limit': 0., 'up_limit': 0},        # Acetate
    'Ala': {'name': 'Ala', 'low_limit': 0., 'up_limit': 0},    # Alanine
    'Asc': {'name': 'Asc', 'low_limit': 0., 'up_limit': 0},        # Ascorbate
    'Asp': {'name': 'Asp', 'low_limit': 0., 'up_limit': 0},    # Aspartate
    'Cr': {'name': 'Cr', 'low_limit': 0, 'up_limit': 0},     # Creatine
    'GABA': {'name': 'GABA', 'low_limit': 0, 'up_limit': 0},  # y-aminobutyric acid
    'Glc': {'name': 'Glc', 'low_limit': 0., 'up_limit': 0},    # Glucose
    'Gln': {'name': 'Gln', 'low_limit': 0., 'up_limit': 0},    # Glutamine
    'Glu': {'name': 'Glu', 'low_limit': 0., 'up_limit': 0},   # Glutamate
    'Gly': {'name': 'Gly', 'low_limit': 0., 'up_limit': 0},        # Glycine
    'GPC': {'name': 'GPC', 'low_limit': 0., 'up_limit': 0},    # Glycero-phosphocholine
    'GSH': {'name': 'GSH', 'low_limit': 0., 'up_limit': 0},    # Glutathione
    'Ins': {'name': 'Ins', 'low_limit': 0, 'up_limit': 0},        #
    'Lac': {'name': 'Lac', 'low_limit': 0., 'up_limit': 0},    # Lactate
    'Mac': {'name': 'Mac', 'low_limit': 0, 'up_limit': 0},    # Macromolecules
    'NAA': {'name': 'NAA', 'low_limit':0, 'up_limit': 14},    # N-Acetylaspartic Acid
    'NAAG': {'name': 'NAAG', 'low_limit': 0., 'up_limit': 4},  # N-Acetylaspartylglutamic Acid
    'PCho': {'name': 'PCho', 'low_limit': 0., 'up_limit': 0},      #
    'PCr': {'name': 'PCr', 'low_limit': 0., 'up_limit': 0},    # Phosphocreatine
    'PE': {'name': 'PE', 'low_limit': 0., 'up_limit': 0},      # Phosphorylethanolamine
    'Tau': {'name': 'Tau', 'low_limit': 0., 'up_limit': 0},    # Taurine
    'sIns': {'name': 'sIns', 'low_limit': 0., 'up_limit': 0},  # Scyllo-inositol
}


#**************************#
#   concentration ranges   #
#**************************#
stdConcs = {
    # these are obtained by running FSL-MRS MH fitting method on the
    # 2016 ISMRM fitting challenge data set
    'Ace': {'name': 'Ace', 'low_limit': 0., 'up_limit': 3.},        # Acetate
    'Ala': {'name': 'Ala', 'low_limit': 0., 'up_limit': 8.},    # Alanine
    'Asc': {'name': 'Asc', 'low_limit': 0., 'up_limit': 7.},        # Ascorbate
    'Asp': {'name': 'Asp', 'low_limit': 1.0, 'up_limit': 8.0},    # Aspartate
    'Cr': {'name': 'Cr', 'low_limit': 0., 'up_limit': 9.},     # Creatine
    'GABA': {'name': 'GABA', 'low_limit': 0., 'up_limit': 9.0},  # y-aminobutyric acid
    'Glc': {'name': 'Glc', 'low_limit': 0., 'up_limit': 3.0},    # Glucose
    'Gln': {'name': 'Gln', 'low_limit': 1.0, 'up_limit': 14.0},    # Glutamine
    'Glu': {'name': 'Glu', 'low_limit': 4.0, 'up_limit': 15.0},   # Glutamate
    'Gly': {'name': 'Gly', 'low_limit': 0., 'up_limit': 9.0},        # Glycine
    'GPC': {'name': 'GPC', 'low_limit': 0., 'up_limit': 3.0},    # Glycero-phosphocholine
    'GSH': {'name': 'GSH', 'low_limit': 0., 'up_limit': 4.0},    # Glutathione
    'Ins': {'name': 'Ins', 'low_limit': 1.0, 'up_limit': 12.0},        #
    'Lac': {'name': 'Lac', 'low_limit': 0., 'up_limit': 38.0},    # Lactate
    'Mac': {'name': 'Mac', 'low_limit': 0., 'up_limit': 14.0},    # Macromolecules
    'NAA': {'name': 'NAA', 'low_limit':0., 'up_limit': 18.0},    # N-Acetylaspartic Acid
    'NAAG': {'name': 'NAAG', 'low_limit': 0., 'up_limit': 4.0},  # N-Acetylaspartylglutamic Acid
    'PCho': {'name': 'PCho', 'low_limit': 0., 'up_limit': 3.0},      #
    'PCr': {'name': 'PCr', 'low_limit': 0., 'up_limit': 7.0},    # Phosphocreatine
    'PE': {'name': 'PE', 'low_limit': 0., 'up_limit': 4.0},      # Phosphorylethanolamine
    'Tau': {'name': 'Tau', 'low_limit': 0., 'up_limit': 4.0},    # Taurine
    'sIns': {'name': 'sIns', 'low_limit': 0., 'up_limit': 2.0},  # Scyllo-inositol

    'Scyllo': {'name': 'Scyllo', 'low_limit': 0., 'up_limit': 2.0},  # Scyllo
    'PCh': {'name': 'PCh', 'low_limit': 0., 'up_limit': 3.0},      #
    'MM_CMR': {'name': 'MM_CMR', 'low_limit': 0., 'up_limit': 14.0},  # Macromolecules

    'mI': {'name': 'mI', 'low_limit': 1., 'up_limit': 12.0},      # myo-Inositol
    'sI': {'name': 'sI', 'low_limit': 0., 'up_limit': 2.0},      # scyllo-Inositol

    # 'Cit': {'name': 'Cit', 'low_limit': 0., 'up_limit': 0.},    # Citrate
    # 'EtOH': {'name': 'EtOH', 'low_limit': 0., 'up_limit': 0.},  # Ethanol
    # 'Phenyl': {'name': 'Phenyl', 'low_limit': 0., 'up_limit': 0.},  # Phenylalanine
    # 'Ser': {'name': 'Ser', 'low_limit': 0., 'up_limit': 0.},    # Serine
    # 'Tyros': {'name': 'Tyros', 'low_limit': 0., 'up_limit': 0.},  # Tyrosine
    # 'bHB': {'name': 'bHB', 'low_limit': 0., 'up_limit': 0.},    # b-Hydroxybutyrate
}


AlexConcs = {

    'Ace': {'name': 'Ace', 'low_limit': 0.5, 'up_limit': 1.5}, 
    'Ala': {'name': 'Ala', 'low_limit': 0.1, 'up_limit': 1.6},    # Alanine
    'Asp': {'name': 'Asp', 'low_limit': 1.0, 'up_limit': 2.0},    # Aspartate
    'Cr': {'name': 'Cr', 'low_limit': 4.5, 'up_limit': 10.5},     # Creatine
    'GABA': {'name': 'GABA', 'low_limit': 1.0, 'up_limit': 2.0},  # y-aminobutyric acid
    'Gln': {'name': 'Gln', 'low_limit': 3.0, 'up_limit': 6.0},    # Glutamine
    'Glu': {'name': 'Glu', 'low_limit': 6.0, 'up_limit': 12.5},   # Glutamate
    'Gly': {'name': 'Gly', 'low_limit': 0.2, 'up_limit': 1.0},        # Glycine
    'GPC': {'name': 'GPC', 'low_limit': 0.4, 'up_limit': 1.7},    # Glycero-phosphocholine
    'GSH': {'name': 'GSH', 'low_limit': 1.7, 'up_limit': 3.0},    # Glutathione
    'Lac': {'name': 'Lac', 'low_limit': 2.0, 'up_limit': 6.0},    # Lactate
    'mI': {'name': 'mI', 'low_limit': 4.0, 'up_limit': 9.0},      # Myo-inositol
    'NAA': {'name': 'NAA', 'low_limit':7.5, 'up_limit': 12.0},    # N-Acetylaspartic Acid
    'NAAG': {'name': 'NAAG', 'low_limit': 0.5, 'up_limit': 2.5},  # N-Acetylaspartylglutamic Acid
    'PC': {'name': 'PC', 'low_limit': 0.5, 'up_limit': 2.0},      #
    'PCr': {'name': 'PCr', 'low_limit': 3.0, 'up_limit': 5.5},    # Phosphocreatine
    'PE': {'name': 'PE', 'low_limit': 1.0, 'up_limit': 2.0},      # Phosphorylethanolamine
    'Tau': {'name': 'Tau', 'low_limit': 2.0, 'up_limit': 6.0},    # Taurine
    'Cho': {'name': 'Cho', 'low_limit': 1.0, 'up_limit': 4.0},    # Choline
    'MM12_1': {'name': 'MM12_1', 'low_limit': 50.0, 'up_limit': 100.0}, #zahra add it for alex basis
    'MM17_2': {'name': 'MM17_2', 'low_limit': 40.0, 'up_limit': 80.0}, #zahra add it for alex basis
    'MM27_3': {'name': 'MM27_3', 'low_limit': 50.0, 'up_limit': 100.0},
    'MM_09': {'name': 'MM_09', 'low_limit': 80.0, 'up_limit': 120.0},
    'MM23': {'name': 'MM23', 'low_limit': 40.0, 'up_limit': 80.0},
    'MM27': {'name': 'MM27', 'low_limit': 50.0, 'up_limit': 100.0},
    'MM30_3': {'name': 'MM30_3', 'low_limit': 50.0, 'up_limit': 80.0},
}

#********************#
#   initialization   #
#********************#
perfectParams = {
    'dist': 'unif',
    'broadening': [(0, 0), (2, 2)],   # [(low, low), (high, high)]
    'coilamps': [1, 1],  # [low, high]
    'coilphase': [0, 0],  # [low, high]
    'phi0': [0, 0],  # [low, high]
    'phi1': [0, 0],  # [low, high]
    'shifting': [0, 0],  # [low, high]
    'baseline': [[0, 0, 0, 0, 0, 0], [0 ,0 , 0, 0, 0, 0]],
    'ownBaseline': None,
    'noise': [0, 0],  # [mean, std]
}

normParams = {
    'dist': 'unif',
    'broadening': [(0, 0), (5, 5)],   # [(low, low), (high, high)]
    'coilamps': [1, 1],  # [low, high]
    'coilphase': [0, 0],  # [low, high]
    'phi0': [0, 0],  # [low, high]
    'phi1': [0, 0],  # [low, high]
    'shifting': [0, 0],  # [low, high]
    'baseline': [[0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0]],
    # 'baseline': [[-3 * 22, -4 * 22, -5 * 22, -3 * 22, -8 * 22, -2 * 22], [1 * 22, 2 * 22, 3 * 22, 5 * 22, 1 * 22, 5 * 22]],
    # 'baseline': [[-600, -800, -1000, -600, -1600, -400], [200, 300, 600, 1000, 200, 1000]],
    'ownBaseline': None,
    'noise': [0, 5],  # [mean, std]
}

normParamsRW = {
    'dist': 'unif',
    'broadening': [(0, 0), (5, 5)],   # [(low, low), (high, high)]
    'coilamps': [1, 1],  # [low, high]
    'coilphase': [0, 0],  # [low, high]
    'phi0': [0, 0],  # [low, high]
    'phi1': [0, 0],  # [low, high]
    'shifting': [0, 0],  # [low, high]
    'baseline': [[0, 0, 0, 0, 0, 0], [0 ,0 , 0, 0, 0, 0]],
    'ownBaseline': None,
    'noise': [0, 5],  # [mean, std]

    # random walk
    'scale': [0, 1000],  # [low, high]
    'smooth': [1, 1000],  # [low, high]
    'limits': [[-10000, 0], [0, 10000]],  # [[low low, low high], [high low, high high]]
}

normParamsRWP = {
    'dist': 'unif',
    'broadening': [(0, 0), (5, 5)],   # [(low, low), (high, high)]
    'coilamps': [1, 1],  # [low, high]
    'coilphase': [0, 0],  # [low, high]
    'phi0': [0, 0],  # [low, high]
    'phi1': [0, 0],  # [low, high]
    'shifting': [0, 0],  # [low, high]
    'baseline': [[0, 0, 0, 0, 0, 0], [0 ,0 , 0, 0, 0, 0]],
    'ownBaseline': None,
    'noise': [0, 5],  # [mean, std]

    # random walk
    'scale': [0, 1000],  # [low, high]
    'smooth': [1, 1000],  # [low, high]
    'limits': [[-10000, 0], [0, 10000]],  # [[low low, low high], [high low, high high]]

    # peaks
    'numPeaks': [0, 5],  # [low, high]
    'peakAmp': [0, 60000],  # [low, high]
    'peakWidth': [0, 100],  # [low, high]
    'peakPhase': [0, 2 * np.pi],  # [low, high]
}

unifParams = {
    'dist': 'unif',
    'broadening': [(0, 0), (25, 25)],   # [(low, low), (high, high)]
    'coilamps': [0.5, 1],  # [low, high]
    'coilphase': [0, 0],  # [low, high]
    'phi0': [-1.5, 1.5],  # [low, high]
    'phi1': [-1e-4, 1e-4],  # [low, high]
    'shifting': [-20, 20],  # [low, high]
    'baseline': [[-5, -5, -5, -5, -5, -5], [5, 5, 5, 5, 5, 5]],
    'ownBaseline': None,
    'noise': [0, 20],  # [mean, std]
}

customParams = {
    'dist': 'unif',
    'broadening': [(0, 0), (2, 2)],   # [(low, low), (high, high)]
    'coilamps': [1, 1],  # [low, high]
    'coilphase': [0, 0],  # [low, high]
    'phi0': [0, 0],  # [low, high]
    'phi1': [0, 0],  # [low, high]
    'shifting': [0, 0],  # [low, high]
    'baseline': [[0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0]],
    'ownBaseline': None,
    'noise': [0, 0],  # [mean, std]
}



paramsRW = {
    'dist': 'unif',
    'broadening': [(1, 0), (35, 25)],   # [(low, low), (high, high)]
    'coilamps': [1, 1],  # [low, high]
    'coilphase': [0, 0],  # [low, high]
    'phi0': [-0.5, 1.5],  # [low, high]
    'phi1': [-1e-4, 1e-4],  # [low, high]
    'shifting': [-10, 10],  # [low, high]
    # 'baseline': [[-3, -4, -5, -3, -8, -2], [1, 2, 3, 5, 1, 5]],
    'baseline': [[-600, -800, -1000, -600, -1600, -400], [200, 300, 600, 1000, 200, 1000]],
    'ownBaseline': None,
    'noise': [0, np.sqrt(2)/2 * 800],  # [mean, std] (account for complex Gaussian)

    # random walk
    'scale': [0, 1000],  # [low, high]
    'smooth': [1, 1000],  # [low, high]
    'limits': [[-10000, 0], [0, 10000]],  # [[low low, low high], [high low, high high]]
}

paramsRWP = {
    'dist': 'unif',
    'broadening': [(10, 5), (50, 30)],  # Keep broadening for smoother peaks
    'coilamps': [1, 1],
    'coilphase': [0, 0],
    'phi0': [-1.5, 1.5],  # Small phase adjustment for added variation
    'phi1': [ -3e-4, 3e-4],  # Slight asymmetry for up-down variation
    'shifting': [-10, 10],
    'baseline': [[-500, -250, -100, -250, -500,-300], [100, 200, 300, 400, 500,400]],  # More centered baseline
    'ownBaseline': None,
    'noise': [0, 200],  # Reduced noise for a smoother pattern
    'scale': [0, 100],  # Reduced scale for sharper transitions
    'smooth': [1, 50],  # Reduced smoothness for sharper peak boundaries
    'limits': [[-10000, 0], [0, 10000]],
    'numPeaks': [0, 5],
    'peakAmp': [0, 2000],  # Lowered peak amplitude to control extreme values
    'peakWidth': [0, 100],
    'peakPhase': [0, 2 * np.pi],
}

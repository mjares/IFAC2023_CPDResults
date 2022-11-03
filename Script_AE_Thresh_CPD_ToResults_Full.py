import os
import pickle

import numpy as np

# Local libs
from NewAutoencoder import AEThresholdClassifier, time_horizon_analysis, CPD
from CPDResults import CPDResults, CPDEpisodeResults

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Params
show = True
savefilename = 'CPDResults_NOM_LOE_AtNU_Ep20_2.rs'

# Parameters
time_horizons = [1, 10, 20, 30, 40]
ci_list = [95, 96, 97, 98, 99]

# Dataset
dFilename = 'CPDDataset_NOM_LOE_AtNU_EpCount_20.ds'
dsetFile = open(dFilename, 'rb')
dataset = pickle.load(dsetFile)
no_episodes = dataset.no_episodes
# Trimming
print('Trimming')
for ee in range(dataset.no_episodes):
    # print(f'Episode: {ee + 1}')
    dataset.episodes[ee].trim(200)

# Results
Results = []
for ci in ci_list:
    ae_model = f'Lat15_HL1_Leak10_E1000_Batch1K_{ci}'
    aeFile = open(ae_model + '.ae', 'rb')
    ae_classifier = pickle.load(aeFile)
    ae_classifier.load_autoencoder()
    for th in time_horizons:
        print(f'Time Horizon: {th}. Confidence Interval: {ae_classifier.conf_interval}')
        configResults = CPDResults()
        configResults.quantile = ae_classifier.Q
        configResults.conf_interval = ae_classifier.conf_interval
        configResults.time_horizon = th
        configResults.autoencoder_name = ae_model
        for ee in range(no_episodes):  # dataset.no_episodes):
            # print(f'Episode: {ee + 1}')
            feature_log = dataset.episodes[ee].features
            starttime = dataset.episodes[ee].changepoint
            stepcount = dataset.episodes[ee].no_samples
            f_type = dataset.episodes[ee].f_type
            f_mag = dataset.episodes[ee].f_mag

            # CPD
            mode_log, rec_err = ae_classifier.predict(np.array(feature_log))
            mode_log_th = time_horizon_analysis(np.array(mode_log), th)
            cpd_results, estimated_cp = CPD(mode_log_th)
            # Episode Results
            epResults = CPDEpisodeResults()
            epResults.fault_mag = f_mag
            epResults.fault_type = f_type
            epResults.starttime = starttime
            epResults.no_samples = stepcount
            epResults.reconst_error = rec_err
            epResults.mode_log = mode_log
            epResults.mode_log_th = mode_log_th
            epResults.changepoints = cpd_results
            epResults.changepoints_tstamps = estimated_cp

            configResults.episode_results.append(epResults)

        Results.append(configResults)

saveFile = open(savefilename, 'wb')
pickle.dump(Results, saveFile)
saveFile.close()

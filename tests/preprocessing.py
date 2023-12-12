# %%
import sys
import numpy as np
import pandas as pd
sys.path.append('/Volumes/Lab/Users/bhofflic/cell_classification/src/')
import cell_display_lib as cdl
from tqdm import tqdm
from matplotlib import pyplot as plt
from file_handling import wrapper

scratch_file_root = '/Volumes/Scratch/Users/bhofflic/celltable_runs' # replace my name!
corr_analysis_path = '/Volumes/Scratch/Users/bhofflic/cell_correlations/'

pieces = [
    # '2005-04-14-0', #9.8*
    # '2005-04-26-1', #20.7*
    '2005-07-07-2', #7.8*
    # '2015-09-23-7', #40.6*
    # '2016-02-17-1', #29.1*
    # '2016-02-17-6', #28.0*
    # '2016-02-17-8', #48.6*
    # '2016-04-21-1', #33.8*
    # '2017-03-15-1', #21.1*
    # '2017-03-15-8', #12.3*
    # '2017-08-14-4', #14.3*
    # '2017-11-29-0', #33.0*
    # '2018-03-01-0', #98.3-
    # '2018-08-07-1', #26.1
    # '2018-08-07-11',#8.5*
    # '2018-08-07-2', #15.7*
    # '2018-08-07-5', #22.2*
    # '2018-08-07-9', #13.9*
]

# %%
for piece_id in tqdm(pieces):
    ct = cdl.CellTable()
    ct.file_load_pieces(scratch_file_root, [piece_id])

    max_run, num_cells = None, 0
    cell_types_of_interest = ['ON parasol', 'OFF parasol', 'ON midget', 'OFF midget', 'ON smooth', 'OFF smooth', 'SBC']
    for run in ct.dataset_table['run_id'].unique():
        valid_units = ct.unit_table.query('valid==True and run_id==@run')
        ctoi = valid_units.query('label_manual_text in @cell_types_of_interest')
        if len(ctoi) > num_cells:
            max_run = run
            num_cells = len(ctoi)
    run_id = max_run

    valid_units = ct.unit_table.query('valid==True and run_id==@run_id')
    tcs_all = np.array([tc.a for tc in valid_units['tc_all']])
    tcs = np.array([tc.a for tc in valid_units['tc']])
    acfs = np.array([acf.a for acf in valid_units['acf']])
    labels = np.array([label for label in valid_units['label_manual_text']])
    distances = ct.dataset_table.query('run_id==@run_id')['rf_overlaps'][0].a
    overlaps = ct.dataset_table.query('run_id==@run_id')['rf_inner_products'][0].a
    cch_1ms = ct.dataset_table.query('run_id==@run_id')['cch_1ms'][0].a
    cch_10ms = ct.dataset_table.query('run_id==@run_id')['cch_10ms'][0].a
    uids = [uid for uid in ct.dataset_table.query('run_id==@run_id')['cch_ids'][0].a]
    eis = np.array([ei.a for ei in valid_units['ei']])
    spike_waveforms = np.array([sw.a for sw in valid_units['spike_waveform_smart']])
    spike_counts = np.array([sc for sc in valid_units['spike_count']])
    sizes = np.max(np.array([size.a for size in valid_units['rf_size_hull']]), axis=1)
    
    data = {
        'piece_id': [piece_id for _ in range(len(uids))],
        'run_id': [run_id for _ in range(len(uids))],
        'unit_id': uids,
        'tc_all': [wrapper(tc) for tc in tcs_all],
        'tc': [wrapper(tc) for tc in tcs],
        'acf': [wrapper(acf) for acf in acfs],
        'label': labels,
        'distances': [wrapper(distance) for distance in distances],
        'overlaps': [wrapper(overlap) for overlap in overlaps],
        'cch_1ms': [wrapper(cch) for cch in cch_1ms],
        'cch_10ms': [wrapper(cch) for cch in cch_10ms],
        'ei': [wrapper(ei) for ei in eis],
        'spike_waveform': [wrapper(sw) for sw in spike_waveforms],
        'spike_count': spike_counts,
        'size': sizes,
        'cch_ids': [wrapper(uids) for _ in range(len(uids))],
    }
    dataframe = pd.DataFrame(data=data, index=uids)

    dataframe.to_pickle(corr_analysis_path + 'gnn_data/' + piece_id + '.pkl')
    del ct
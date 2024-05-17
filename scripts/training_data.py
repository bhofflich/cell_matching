# %%
import numpy as np
import matplotlib.pyplot as plt
import os, sys
sys.path.append("/Volumes/Lab/Users/bhofflic/cell_classification/src")
import istarmap
from tqdm import tqdm
import features_correlations as feat_c
import features_visual as feat_v

sys.path.append("/Volumes/Lab/Users/scooler/classification/")
import cell_display_lib as cdl
import features
import deduplication as dedup
import multiprocessing as mp
from multiprocessing import Pool
import pandas as pd
from file_handling import wrapper


import elephant, warnings
import quantities as pq
from elephant.conversion import BinnedSpikeTrain
from neo import SpikeTrain

import itertools

# %%
classification_types = ['ON parasol', 'OFF parasol', 'ON midget', 'OFF midget', 
                        'SBC', 'ON smooth', 'OFF smooth', 
                        'ON A1', 'OFF A1', 'OFF BT', 'OFF blobby amacrine', 'OFF RB']

def get_merged_unit(unit_table, unit_id):
    run = unit_id[1]
    merged_units = unit_table.query("merged == True and run_id == @run")
    unit = unit_table.loc[unit_id]
    cell_id = unit['cell_id']
    if cell_id in list(merged_units['cell_id']):
        return merged_units.query("cell_id == @cell_id").index[0]
    else:
        return unit_id

# %%
cch_data = pd.read_pickle('/Volumes/Scratch/Users/bhofflic/correlations/average_cchs.pkl')
training_data = pd.read_pickle('/Volumes/Scratch/Users/bhofflic/correlations/training_data.pkl')

# %%
piece_ids = [
             "2017-11-29-0",
             "2018-08-07-11",
             "2017-11-20-1",
             "2017-11-20-8",
             "2018-03-01-6",
             "2017-08-14-1",
             "2017-10-30-6",
             '2005-04-26-1',
             '2005-07-07-2',
             '2015-09-23-7',
             '2016-02-17-6',
             '2016-02-17-8',
             '2017-03-15-1',
             '2017-03-15-8',
             '2018-08-07-1',
             '2018-08-07-2', 
             '2018-08-07-9',
            '2018-12-10-6',
            '2018-11-12-5',
            '2018-02-09-7',
            '2018-02-06-4',
            '2018-02-09-3',
            '2018-03-01-5',
            '2019-08-27-0',
            '2016-10-04-6'
             ]

for piece_id in piece_ids:
    ct = cdl.CellTable()
    print(piece_id)
    if not os.path.exists(f'/Volumes/Scratch/Users/bhofflic/celltable_runs/ctd_{piece_id}_u.pkl'):
        continue
    ct.file_load_pieces(f'/Volumes/Scratch/Users/bhofflic/celltable_runs/', [piece_id])

    # %%
    # anchor_types = ['ON parasol', 'OFF parasol', 'ON midget', 'OFF midget']
    anchor_types = classification_types
    cell_cchs = {'piece_id':[], 'run_id':[], 'unit_id':[], 'cell_type': []}
    cell_cchs.update({anchor_type: [] for anchor_type in anchor_types})
    classification_runs = cch_data.query("piece_id == @piece_id").run_id.unique()

    for run_id in classification_runs:
        overlaps = ct.dataset_table.loc[(piece_id, run_id), 'rf_overlaps'].a
        cch_1ms = ct.dataset_table.loc[(piece_id, run_id), 'cch_1ms'].a
        cch_10ms = ct.dataset_table.loc[(piece_id, run_id), 'cch_10ms'].a
        cch_ids = ct.dataset_table.loc[(piece_id, run_id), 'cch_ids'].a
        indices = {cell_type: set() for cell_type in classification_types}
        
        for cell_type in classification_types:
            units = ct.unit_table.query(f"piece_id == @piece_id and run_id == @run_id and label_manual_text == '{cell_type}' and valid == True")
            for unit in units.index:
                nd_unit = get_merged_unit(ct.unit_table, unit)
                if ct.unit_table.loc[nd_unit].valid == False:
                    # print(f"Invalid unit {nd_unit}")
                    continue
                indices[cell_type].add(np.where(cch_ids == nd_unit)[0][0])
        indices = {cell_type: list(indices[cell_type]) for cell_type in classification_types}
        
        for cch_idx, uid in enumerate(ct.dataset_table.loc[(piece_id, run_id), 'cch_ids'].a):
            valid = False
            cell_cchs['piece_id'].append(piece_id)
            cell_cchs['run_id'].append(run_id)
            cell_cchs['unit_id'].append(uid[2])
            cell_cchs['cell_type'].append(ct.unit_table.loc[uid].label_manual_text)
            for anchor_type in anchor_types:
                anchor_indices = indices[anchor_type]
                if len(anchor_indices) == 0:
                    cell_cchs[anchor_type].append(None)
                    continue
                anchor_overlap = overlaps[cch_idx, anchor_indices]
                valid_anchor_overlap = np.where(np.logical_and((anchor_overlap < 1),(anchor_overlap > 0.1)))[0]
                if len(valid_anchor_overlap) == 0:
                    cell_cchs[anchor_type].append(None)
                    continue
                valid_anchor_indices = np.array(anchor_indices)[valid_anchor_overlap]
                cchs_1ms = cch_1ms[cch_idx, valid_anchor_indices, :]
                cchs_1ms -= np.mean(np.concatenate((cchs_1ms[:,:10],cchs_1ms[:,-10:]), axis=1))
                cchs_10ms = cch_10ms[cch_idx, valid_anchor_indices, :]
                cchs_10ms -= np.mean(np.concatenate((cchs_10ms[:,:10],cchs_10ms[:,-10:]), axis=1))

                weights = 1/overlaps[cch_idx, valid_anchor_indices]
                avg_cell_anchor_cch_1ms = np.average(cchs_1ms/np.max(np.abs(cchs_1ms), axis=1)[:,None], axis=0, weights=weights)
                avg_cell_anchor_cch_10ms = np.average(cchs_10ms/np.max(np.abs(cchs_10ms), axis=1)[:,None], axis=0, weights=weights)
                avg_cell_anchor_cch = np.array([avg_cell_anchor_cch_1ms, avg_cell_anchor_cch_10ms])
                cell_cchs[anchor_type].append(wrapper(avg_cell_anchor_cch))
                valid = True
            if not valid:
                cell_cchs['piece_id'].pop()
                cell_cchs['run_id'].pop()
                cell_cchs['unit_id'].pop()
                cell_cchs['cell_type'].pop()
                for anchor_type in anchor_types:
                    cell_cchs[anchor_type].pop()

        

    # %%
    piece_training_data = pd.DataFrame(cell_cchs, index=[cell_cchs['piece_id'], cell_cchs['run_id'], cell_cchs['unit_id']])
    training_data = piece_training_data.combine_first(training_data)

    # %%
    training_data.to_pickle('/Volumes/Scratch/Users/bhofflic/correlations/training_data.pkl')



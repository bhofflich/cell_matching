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
import itertools


import elephant, warnings
import quantities as pq
from elephant.conversion import BinnedSpikeTrain
from neo import SpikeTrain

piece_ids = [
            # "2017-11-29-0",
            #  "2018-08-07-11",
            #  "2017-11-20-1",
            #  "2017-11-20-8",
            #  "2018-03-01-6",
            #  "2017-08-14-1",
            #  "2017-10-30-6",
            #  '2005-04-26-1',
            #  '2005-07-07-2',
            #  '2015-09-23-7',
            #  '2016-02-17-6',
            #  '2016-02-17-8',
            #  '2017-03-15-1',
            #  '2017-03-15-8',
            #  '2018-08-07-1',
            #  '2018-08-07-2', 
            #  '2018-08-07-9'
            '2018-12-10-6',
            '2018-11-12-5',
            '2018-02-09-7',
            '2018-02-06-4',
            '2018-02-09-3',
            '2018-03-01-5',
            '2019-08-27-0',
            '2016-10-04-6'
            ]

def get_average_cchs(ct,piece_id,run_id, classification_types, type_cchs = False, type_overlaps = False):
    avg_1ms = {}
    avg_10ms = {}
    typed_cchs = {}
    typed_overlaps = {}
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
                print(f"Invalid unit {nd_unit}")
                continue
            indices[cell_type].add(np.where(cch_ids == nd_unit)[0][0])
    indices = {cell_type: list(indices[cell_type]) for cell_type in classification_types}
    
    run_types = ct.unit_table.query(f"piece_id == @piece_id and run_id == @run_id").label_manual_text.unique()
            
    for ct1 in classification_types:
        for ct2 in classification_types:
            if ct1 not in run_types or ct2 not in run_types:
                continue
            if (ct2, ct1) in avg_1ms:
                avg_1ms[(ct1, ct2)] = np.fliplr(avg_1ms[(ct2, ct1)])
                avg_10ms[(ct1, ct2)] = np.fliplr(avg_10ms[(ct2, ct1)])
                continue
            ct1_indices = indices[ct1]
            if len(ct1_indices) == 0:
                continue
            if ct1 == ct2:
                if len(ct1_indices) == 1:
                    continue
                comb_indices = np.array(list(itertools.combinations(ct1_indices, 2)))
                valid_overlap_indices = comb_indices[np.where(np.logical_and((overlaps[comb_indices[:,0], comb_indices[:,1]] < 1.5),
                                                                             (overlaps[comb_indices[:,0], comb_indices[:,1]] > 0.5)))]
            else:
                ct2_indices = indices[ct2]
                if len(ct2_indices) == 0:
                    continue
                comb_indices = np.transpose([np.tile(ct1_indices, len(ct2_indices)), np.repeat(ct2_indices, len(ct1_indices))])
                if len(comb_indices.shape) == 1:
                    comb_indices = comb_indices.reshape((1,2))
                if len(comb_indices) == 1:
                    comb_indices = comb_indices.reshape((1,2))
                valid_overlap_indices = comb_indices[np.where(np.logical_and((overlaps[comb_indices[:,0], comb_indices[:,1]] < 1),
                                                                             (overlaps[comb_indices[:,0], comb_indices[:,1]] > 0.5)))]
                
            if len(valid_overlap_indices) == 0:
                continue
            
            if type_overlaps:
                typed_overlaps[(ct1, ct2)] = overlaps[valid_overlap_indices[:,0], valid_overlap_indices[:,1]]
            
            cchs_1ms = cch_1ms[valid_overlap_indices[:,0], valid_overlap_indices[:,1], :]
            cchs_1ms -= np.mean(np.concatenate((cchs_1ms[:,:10],cchs_1ms[:,-10:]), axis=1))
            cchs_10ms = cch_10ms[valid_overlap_indices[:,0], valid_overlap_indices[:,1], :]
            cchs_10ms -= np.mean(np.concatenate((cchs_10ms[:,:10],cchs_10ms[:,-10:]), axis=1))
            
            if type_cchs:
                typed_cchs[(ct1, ct2)] = {"1ms": cchs_1ms, "10ms": cchs_10ms}
            
            weights = 1/overlaps[valid_overlap_indices[:,0], valid_overlap_indices[:,1]]
            
            avg_1ms[(ct1, ct2)] = np.zeros((2,cchs_1ms.shape[1]))
            avg_10ms[(ct1, ct2)] = np.zeros((2,cchs_10ms.shape[1]))
            
            max_1ms = np.max(np.abs(cchs_1ms), axis=1)
            max_10ms = np.max(np.abs(cchs_10ms), axis=1)
            
            warnings.filterwarnings("error")
            try:
                avg_1ms[(ct1, ct2)][0] = np.average(cchs_1ms/max_1ms[:,None], axis=0, weights=weights)
                avg_1ms[(ct1, ct2)][1] = np.std(cchs_1ms, axis=0)
                # avg_1ms[(ct1, ct2)] /= np.max(np.abs(avg_1ms[(ct1, ct2)][0]))
                
                avg_10ms[(ct1, ct2)][0] = np.average(cchs_10ms/max_10ms[:,None], axis=0, weights=weights)
                avg_10ms[(ct1, ct2)][1] = np.std(cchs_10ms, axis=0)
                # avg_10ms[(ct1, ct2)] /= np.max(np.abs(avg_10ms[(ct1, ct2)][0]))
            except Warning as e:
                print(f"Warning: {(ct1,ct2)}, {e}")
    if type_cchs and type_overlaps:
        return avg_1ms, avg_10ms, typed_cchs, typed_overlaps
    elif type_cchs:
        return avg_1ms, avg_10ms, typed_cchs
    elif type_overlaps:
        return avg_1ms, avg_10ms, typed_overlaps         
    return avg_1ms, avg_10ms
        
def get_merged_unit(unit_table, unit_id):
        run = unit_id[1]
        merged_units = unit_table.query("merged == True and run_id == @run")
        unit = unit_table.loc[unit_id]
        cell_id = unit['cell_id']
        if cell_id in list(merged_units['cell_id']):
            return merged_units.query("cell_id == @cell_id").index[0]
        else:
            return unit_id
        
classification_types = ['ON parasol', 'OFF parasol', 'ON midget', 'OFF midget', 
                        'SBC', 'ON smooth', 'OFF smooth', 
                        'ON A1', 'OFF A1', 'OFF BT', 'OFF blobby amacrine', 'OFF RB']
# %%
for piece_id in piece_ids:
    ct = cdl.CellTable()
    print(piece_id)
    
    if os.path.exists(f'/Volumes/Analysis/{piece_id}/data999/data999.verified_rb.classification.txt'):
        label_data_path = f'/Volumes/Analysis/{piece_id}/data999/data999.verified_rb.classification.txt'
    elif os.path.exists(f'/Volumes/Analysis/{piece_id}/data999/data999.verified_bt.classification.txt'):
        label_data_path = f'/Volumes/Analysis/{piece_id}/data999/data999.verified_bt.classification.txt'
    elif os.path.exists(f'/Volumes/Analysis/{piece_id}/classifications_ak/data999.classification.txt'):
        label_data_path = f'/Volumes/Analysis/{piece_id}/classifications_ak/data999.classification.txt'
    elif os.path.exists(f'/Volumes/Analysis/{piece_id}/classifications_ak/data999_manual.classification.txt'):
        label_data_path = f'/Volumes/Analysis/{piece_id}/classifications_ak/data999_manual.classification.txt'
    else:
        print(f"Missing classification file for {piece_id}")
        continue
    
    ct.file_load_pieces(f'/Volumes/Scratch/Analysis/{piece_id}', [piece_id])
        
    # label_data_path = f'/Volumes/Analysis/{piece_id}/data999/data999.classification.txt'
    label_mode = 'list'

    ct.dataset_table.at[(piece_id, 'com'), 'label_data_path'] = label_data_path
    ct.dataset_table.at[(piece_id, 'com'), 'labels'] = label_mode

    for unit in ct.unit_table.index:
        cell_id = ct.unit_table.loc[unit].cell_id
        cell_units = ct.cell_table.loc[cell_id, 'unit_ids']
        if unit not in list(cell_units):
            updated_cell_units = [unit] + list(cell_units)
            ct.cell_table.at[cell_id, 'unit_ids'] = updated_cell_units
            
    dedup.arrange_units_by_run(ct)

    features_to_generate_by_dataset = [features.Feature_load_manual_labels]

    # indices = 'all'
    indices = ct.unit_table.query(f"run_id == 'com'").index
    ct.generate_features(indices, features_to_generate_by_dataset, [],
                        force_features=1)
    ct.copy_unit_labels_to_cells_combined()
    
    equivalent_names = {'OFF BT': ['OFF BT', 'OFF broad thorny'], 'SBC': ['SBC', 'blue SBC']}
    for cell_id in ct.cell_table.index:
        cell = ct.cell_table.loc[cell_id]
        for name in equivalent_names:
            if cell['label_manual_text'] in equivalent_names[name]:
                ct.cell_table.at[cell_id, 'label_manual_text'] = name
                break
    ct.copy_cell_labels_to_units()

    # %%
    max_runs = {}
    for classification_type in classification_types:
        for (piece_id, run_id) in ct.dataset_table.index:
            if run_id == 'com':
                continue
            unique_units = set()
            for uid, unit in ct.unit_table.query(f"run_id == '{run_id}' and label_manual_text == '{classification_type}'").iterrows():
                unique_units.add(get_merged_unit(ct.unit_table, uid))
            num_type_units = len(unique_units)
            if num_type_units > 0:
                if classification_type not in max_runs:
                    max_runs[classification_type] = (run_id, num_type_units)
                else:
                    if num_type_units > max_runs[classification_type][1]:
                        max_runs[classification_type] = (run_id, num_type_units)
    runs_to_run = [max_runs[classification_type][0] for classification_type in max_runs.keys()]
    runs_to_run = list(set(runs_to_run))

    # %%
    features_to_generate_by_dataset = [feat_v.Feature_rf_convex_hull,
                                    feat_v.Feature_rf_boundary,
                                    feat_c.Feature_rf_radii,
                                    feat_c.Feature_rf_overlaps,
                                    feat_c.Feature_cross_correlations_complete_fast,]

    indices = 'all'
    indices = ct.unit_table.query(f"run_id in @runs_to_run").index
    ct.generate_features(indices, features_to_generate_by_dataset, [],
                        force_features=1)

    # %%
    ct.file_save_pieces(f'/Volumes/Scratch/Users/bhofflic/celltable_runs/', [piece_id])

    # %%
    data = pd.read_pickle(f'/Volumes/Scratch/Users/bhofflic/correlations/average_cchs.pkl')
    for run_id in runs_to_run:
            avg_1ms, avg_10ms, typed_cchs, typed_overlaps = get_average_cchs(ct, piece_id, run_id, max_runs.keys(), type_cchs=True, type_overlaps=True)

            avgs = {"piece_id": [piece_id],
                    "run_id": [run_id],
                    "typed_cchs": [wrapper(typed_cchs)],
                    "avg_1ms": [wrapper(avg_1ms)],
                    "avg_10ms": [wrapper(avg_10ms)],
                    "delays": [wrapper(ct.dataset_table.loc[(piece_id, run_id), 'cch_delays'].a)],
                    "overlaps": [wrapper(typed_overlaps)],}
            indexers = [avgs['piece_id'], avgs['run_id']]
            index = pd.MultiIndex.from_arrays(indexers, names=('piece_id', 'run_id'))
            df = pd.DataFrame(avgs, index=index)
            data = pd.concat([data, df])
    data.to_pickle(f'/Volumes/Scratch/Users/bhofflic/correlations/average_cchs.pkl')
    del ct
    del data
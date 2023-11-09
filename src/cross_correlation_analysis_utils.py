"""
This module contains utility functions for cross-correlation analysis of neural data.

Functions:
- get_cch_matrix(ct, run, spike_bin_size='10ms'): Extracts the cross-correlogram (CCH) matrix for all valid units in a given celltable.
- get_overlap_matrix(ct, run): Extracts the receptive field overlap matrix for all valid units in a given celltable.
- get_inner_product_matrix(ct, run): Extracts the receptive field inner product matrix for all valid units in a given celltable.
- get_spike_count_matrix(ct, run): Extracts the spike count matrix for all valid units in a given celltable.
- get_normalized_cch_matrix(ct, run, spike_bin_size='10ms', cch_matrix=None, overlap_matrix=None, inner_product_matrix=None, spike_count_matrix=None): Computes the normalized cross-correlogram (CCH) matrix for a given celltable.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.io import loadmat
from IPython.display import display
import warnings, sys, importlib
from tqdm import tqdm
import os
import multiprocessing as mp

sys.path.append('/Volumes/Lab/Users/bhofflic/mini_rotation/utils/')
import istarmap

def get_cch_matrix(ct, run, spike_bin_size='10ms', cells=False):
    """
    Extracts the cross-correlogram (CCH) matrix for all valid units in a given celltable.

    Args:
        ct: The celltable containing the units to compute the CCH matrix for.
        spike_bin_size (str): The size of the time bins to use for binning the spike times. Default is '10ms'.

    Returns:
        numpy.ndarray: A 3D numpy array of shape (num_units, num_units, 101) containing the CCH matrix for all valid units in the celltable.
    """
    print("Getting CCH matrix...")
    ruleout = ['BW','unlabel','weird','duplicate','unclass', 'contaminated', 'crap', 'edge', 'weak', 'mess','Unclass','artifact']
    units = ct.unit_table.query("valid == True and run_id == @run and label_manual_text not in @ruleout")
    unit_ids = units.index
    if cells:
        unit_ids = []
        cells = ct.cell_table.query("valid == True and label_manual_text not in @ruleout")
        for cid, cell in cells.iterrows():
            if run in cell.unit_ids_by_run.a:
                unit_ids.append(cell.unit_ids_by_run.a[run][0])
        units = units.loc[unit_ids]
    num_units = len(unit_ids)
    lag_length = units.iloc[0][f"{spike_bin_size} CCHs"].a[unit_ids[1]].shape[0]
    cch_matrix = np.zeros((num_units, num_units, lag_length))
    for i, unit1 in enumerate(unit_ids):
        for j, unit2 in enumerate(unit_ids):
            cch_matrix[i,j,:] = units.iloc[i][f"{spike_bin_size} CCHs"].a[unit2]
    return cch_matrix, unit_ids

def get_overlap_matrix(ct, run, cells=False):
    """
    Extracts the receptive field overlap matrix for all valid units in a given celltable.

    Args:
        ct: The celltable containing the units to compute the overlap matrix for.

    Returns:
        numpy.ndarray: A 2D numpy array of shape (num_units, num_units) containing the overlap matrix for all valid units in the celltable.
    """
    print("Getting overlap matrix...")
    ruleout = ['BW','unlabel','weird','duplicate','unclass', 'contaminated', 'crap', 'edge', 'weak', 'mess','Unclass','artifact']
    units = ct.unit_table.query("valid == True and run_id == @run and label_manual_text not in @ruleout")
    unit_ids = units.index
    if cells:
        unit_ids = []
        cells = ct.cell_table.query("valid == True and label_manual_text not in @ruleout")
        for cid, cell in cells.iterrows():
            if run in cell.unit_ids_by_run.a:
                unit_ids.append(cell.unit_ids_by_run.a[run][0])
    num_units = len(unit_ids)
    overlap_matrix = np.zeros((num_units, num_units))
    for i, unit1 in enumerate(unit_ids):
        for j, unit2 in enumerate(unit_ids):
            overlap_matrix[i,j] = units.loc[unit1]['rf_overlaps'].a[unit2]
    return overlap_matrix, unit_ids
    
def get_inner_product_matrix(ct, run, cells=False):
    """
    Extracts the receptive field inner product matrix for all valid units in a given celltable.

    Args:
        ct: The celltable containing the units to compute the inner product matrix for.

    Returns:
        numpy.ndarray: A 2D numpy array of shape (num_units, num_units) containing the inner product matrix for all valid units in the celltable.
    """
    print("Getting inner product matrix...")
    ruleout = ['BW','unlabel','weird','duplicate','unclass', 'contaminated', 'crap', 'edge', 'weak', 'mess','Unclass','artifact']
    units = ct.unit_table.query("valid == True and run_id == @run and label_manual_text not in @ruleout")
    unit_ids = units.index
    if cells:
        unit_ids = []
        cells = ct.cell_table.query("valid == True and label_manual_text not in @ruleout")
        for cid, cell in cells.iterrows():
            if run in cell.unit_ids_by_run.a:
                unit_ids.append(cell.unit_ids_by_run.a[run][0])
    num_units = len(unit_ids)
    inner_product_matrix = np.zeros((num_units, num_units))
    for i, unit1 in enumerate(unit_ids):
        for j, unit2 in enumerate(unit_ids):
            inner_product_matrix[i,j] = units.iloc[i]['rf_inner_products'].a[unit2]
    return inner_product_matrix, unit_ids

def get_spike_count_matrix(ct, run, cells = False):
    """
    Extracts the spike count matrix for all valid units in a given celltable.

    Args:
        ct: The celltable containing the units to compute the spike count matrix for.

    Returns:
        numpy.ndarray: A 2D numpy array of shape (num_units, num_units) containing the spike count matrix for all valid units in the celltable.
    """
    print("Getting spike count matrix...")
    ruleout = ['BW','unlabel','weird','duplicate','unclass', 'contaminated', 'crap', 'edge', 'weak', 'mess','Unclass','artifact']
    units = ct.unit_table.query("valid == True and run_id == @run and label_manual_text not in @ruleout")
    unit_ids = units.index
    if cells:
        unit_ids = []
        cells = ct.cell_table.query("valid == True and label_manual_text not in @ruleout")
        for cid, cell in cells.iterrows():
            if run in cell.unit_ids_by_run.a:
                unit_ids.append(cell.unit_ids_by_run.a[run][0])
        units = units.loc[unit_ids]
    num_units = len(unit_ids)
    spike_count_matrix = np.zeros((num_units, num_units))
    spike_counts = np.array(units['spike_count'])
    spike_count_matrix = spike_counts + spike_counts[:,None]
    return spike_count_matrix, unit_ids

def get_normalized_cch_matrix(ct, run, spike_bin_size='10ms', 
                              cch_matrix=None, 
                              overlap_matrix=None, 
                            #   inner_product_matrix=None, 
                            #   spike_count_matrix=None,
                              cells = False):
    """
    Computes the normalized cross-correlogram (CCH) matrix for a given celltable.

    Parameters:
    -----------
    ct : celltable
        A celltable containing spike times and unit IDs.
    spike_bin_size : str, optional
        The size of the time bins used to bin the spike times. Default is '10ms'.

    Returns:
    --------
    normalized_cch_matrix : ndarray
        A 3D numpy array containing the normalized CCH matrix for all pairs of units in `ct`.
    unit_ids : list
        A list of unit IDs corresponding to the rows and columns of the CCH matrix.
    """
    ruleout = ['BW','unlabel','weird','duplicate','unclass', 'contaminated', 'crap', 'edge', 'weak', 'mess','Unclass','artifact']
    units = ct.unit_table.query("valid == True and run_id == @run and label_manual_text not in @ruleout")
    unit_ids = units.index
    if cells:
        unit_ids = []
        cells = ct.cell_table.query("valid == True and label_manual_text not in @ruleout")
        for cid, cell in cells.iterrows():
            if run in cell.unit_ids_by_run.a:
                unit_ids.append(cell.unit_ids_by_run.a[run][0])
        units = units.loc[unit_ids]
    num_units = len(unit_ids)
    if cch_matrix is None:
        lag_length = units.iloc[0][f"{spike_bin_size} CCHs"].a[unit_ids[1]].shape[0]
        cch_matrix = np.zeros((num_units, num_units, lag_length))
        for i, unit1 in enumerate(unit_ids):
            for j, unit2 in enumerate(unit_ids):
                cch_matrix[i,j,:] = units.iloc[i][f"{spike_bin_size} CCHs"].a[unit2]
    if overlap_matrix is None:
        overlap_matrix = np.zeros((num_units, num_units))
        for i, unit1 in enumerate(unit_ids):
            for j, unit2 in enumerate(unit_ids):
                overlap_matrix[i,j] = units.iloc[i]['rf_overlaps'].a[unit2]
    # if inner_product_matrix is None:
    #     inner_product_matrix = np.zeros((num_units, num_units))
    #     for i, unit1 in enumerate(unit_ids):
    #         for j, unit2 in enumerate(unit_ids):
    #             inner_product_matrix[i,j] = units.iloc[i]['rf_inner_products'].a[unit2]
    # if spike_count_matrix is None:
    #     spike_count_matrix = np.zeros((num_units, num_units))
    #     spike_counts = np.array(units['spike_count'])
    #     spike_count_matrix = spike_counts + spike_counts[:,None]
    # noise_weight_matrix = np.std(np.concatenate((cch_matrix[:,:,:30],cch_matrix[:,:,-30:]),axis=2),axis=2)
    print("Computing normalized CCH matrix...")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        weight = np.nan_to_num(overlap_matrix, posinf=0, neginf=0)# * inner_product_matrix * (1 / spike_count_matrix)# * noise_weight_matrix
        # normalized_cch_matrix = ((weighted_cch_matrix - np.min(weighted_cch_matrix,axis=2)[:,:,None]) / 
        #                             (np.max(weighted_cch_matrix,axis=2) - np.min(weighted_cch_matrix,axis=2))[:,:,None])
        means = np.mean(np.concatenate((cch_matrix[:,:,:10],cch_matrix[:,:,-10:]),axis=2),axis=2)
        zeroed_cch_matrix = cch_matrix - means[:,:,None]
        print("Done!")
    return zeroed_cch_matrix, unit_ids

def get_average_cells_cch(ref_ids, comp_ids, uids, normalized_cchs, overlap_matrix, overlap_thresholds = [0,1.5]):
    """
    Computes the average cross-correlogram (CCH) between a reference unit and a list of comparison units.

    Parameters:
    -----------
    ref_id : int
        The index of the reference unit.
    comp_ids : list
        A list of indices of comparison units.
    uids : list
        A MultiIndex of unit IDs corresponding to the rows and columns of the CCH matrix.
    normalized_cchs : ndarray
        A 3D numpy array containing the normalized CCH matrix for all pairs of units in `ct`.
    overlap_matrix : ndarray
        A 2D numpy array containing the receptive field overlap matrix for all pairs of units in `ct`.

    Returns:
    --------
    average_cchs : ndarray
        A 2D numpy array containing the average CCHs between the reference unit and each comparison unit.
    average_cch_stds : ndarray
        A 2D numpy array containing the standard deviations of the average CCHs between the reference unit and each comparison unit.
    """
    lag_length = normalized_cchs.shape[2]
    ref_indices = np.array([uids.get_loc(ref_id) for ref_id in ref_ids])
    comp_indices = np.array([uids.get_loc(comp_id) for comp_id in comp_ids])
    if len(ref_indices) == 0:
        return np.ma.masked_array(np.zeros((1,lag_length)), mask=True), np.ma.masked_array(np.zeros((1,lag_length)), mask=True), -1
    if len(comp_indices) == 0:
        return np.ma.masked_array(np.zeros((len(ref_indices),lag_length)), mask=True), np.ma.masked_array(np.zeros((len(ref_indices),lag_length)), mask=True), -1
    cchs = np.ma.stack([normalized_cchs[ref_index, comp_indices] for ref_index in ref_indices])
    norm_dists = np.ma.stack([overlap_matrix[ref_index, comp_indices] for ref_index in ref_indices])
    distance_mask = np.broadcast_to(np.logical_and((norm_dists < overlap_thresholds[1]), 
                                                   (norm_dists > overlap_thresholds[0]))[:,:,None], 
                                    cchs.shape)
    test_mask = np.logical_and((norm_dists < overlap_thresholds[1]),(norm_dists > overlap_thresholds[0]))
    masked_dists = np.ma.masked_array(norm_dists, mask=~test_mask)
    
    masked_cchs = np.ma.masked_array(cchs, mask=~distance_mask)
    cch_avg = np.ma.stack([np.ma.average(masked_cchs[i,:,:], axis=0,weights=1/masked_dists[i,:]) for i in range(len(ref_indices))])
    cch_std = np.ma.stack([np.ma.std(masked_cchs[i,:,:], axis=0) for i in range(len(ref_indices))])
    
    
    avg_dists = np.ma.average(masked_dists, axis=1)
    return cch_avg, cch_std, avg_dists

def get_average_cells_ctype_cch(ctype, ref_ids, uids, uid_ctypes, normalized_cchs, overlap_matrix, overlap_thresholds = [0,1.5]):
    """
    Computes the average cross-correlogram (CCH) between a reference unit and all units of a specified cell type(s).

    Parameters:
    -----------
    all_units: celltable unit_table
        unit_table from specified piece_id and run_id
    ref_unit : tuple
        The unit_index of the reference unit.
    ctype : str
        The cell type to compute the average CCHs for.
    normalized_cchs : ndarray
        A 3D numpy array containing the normalized CCH matrix for all pairs of units in `ct`.
    overlap_matrix : ndarray
        A 2D numpy array containing the receptive field overlap matrix for all pairs of units in `ct`.

    Returns:
    --------
    average_cchs : ndarray
        A 2D numpy array containing the average CCHs between the reference unit and all units of the specified cell type.
    average_cch_stds : ndarray
        A 2D numpy array containing the standard deviations of the average CCHs between the reference unit and all units of the specified cell type.
    """
    lag_length = normalized_cchs.shape[2]
    if type(ctype) == str:
        comp_ids = uids[np.array(uid_ctypes) == ctype]
        return get_average_cells_cch(ref_ids, comp_ids, uids, normalized_cchs, overlap_matrix, overlap_thresholds)
    else:
        avg_cells_cchs = np.ma.zeros((len(ref_ids), len(ctype), lag_length))
        cells_cch_std = np.ma.zeros((len(ref_ids), len(ctype), lag_length))
        avg_dists = np.ma.zeros((len(ref_ids), len(ctype)))
        for i, cell_type in enumerate(ctype):
            avg_cells_cchs[:,i], cells_cch_std[:,i], avg_dists[:,i] = get_average_cells_ctype_cch(cell_type, ref_ids, uids, uid_ctypes, normalized_cchs, overlap_matrix, overlap_thresholds)
        return avg_cells_cchs, cells_cch_std, avg_dists
    
def get_average_ctype_cch(ctype1, ctype2, uids, uid_ctypes, normalized_cchs, overlap_matrix, overlap_thresholds = [0,1.5]):
    """
    Computes the average cross-correlogram (CCH) between all units of two specified cell types.

    Parameters:
    -----------
    all_units: celltable unit_table
        unit_table from specified piece_id and run_id
    ctype1 : str
        The first cell type to compute the average CCHs for.
    ctype2 : str
        The second cell type to compute the average CCHs for.
    normalized_cchs : ndarray
        A 3D numpy array containing the normalized CCH matrix for all pairs of units in `ct`.
    overlap_matrix : ndarray
        A 2D numpy array containing the receptive field overlap matrix for all pairs of units in `ct`.

    Returns:
    --------
    average_cchs : ndarray
        A 2D numpy array containing the average CCHs between all units of the two specified cell types.
    average_cch_stds : ndarray
        A 2D numpy array containing the standard deviations of the average CCHs between all units of the two specified cell types.
    """
    ref_ids = uids[np.array(uid_ctypes) == ctype1]
    comp_ids = uids[np.array(uid_ctypes) == ctype2]
    cell_ctype1_avg_cchs, cell_ctype1_std_cchs, norm_dists = get_average_cells_cch(ref_ids, comp_ids, uids, normalized_cchs, overlap_matrix, overlap_thresholds)
    normalizers = 1/np.max(np.abs(cell_ctype1_avg_cchs),axis=1)
    ctype1_avg_cchs = cell_ctype1_avg_cchs * normalizers[:,None]
    
    ctype_average_cch = np.ma.average(ctype1_avg_cchs, axis=0, weights=np.nan_to_num(1/norm_dists, posinf=0, neginf=0))
    ctype_average_cch_std = np.ma.std(ctype1_avg_cchs, axis=0)
    return ctype_average_cch, ctype_average_cch_std

def get_ctype_template(ctype1, comp_ctypes, uids, uid_ctypes, normalized_cchs, overlap_matrix, overlap_thresholds = [0,1.5]):
    """
    Computes the average cross-correlogram (CCH) between all units of a specified cell type and all units of a list of comparison cell types.

    Parameters:
    -----------
    all_units: celltable unit_table
        unit_table from specified piece_id and run_id
    ctype1 : str
        The cell type to compute the average CCHs for.
    comp_ctypes : list
        A list of cell types to compare with the specified cell type.
    normalized_cchs : ndarray
        A 3D numpy array containing the normalized CCH matrix for all pairs of units in `ct`.
    overlap_matrix : ndarray
        A 2D numpy array containing the receptive field overlap matrix for all pairs of units in `ct`.

    Returns:
    --------
    average_cchs : ndarray
        A 2D numpy array containing the average CCHs between all units of the specified cell type and all units of the comparison cell types.
    average_cch_stds : ndarray
        A 2D numpy array containing the standard deviations of the average CCHs between all units of the specified cell type and all units of the comparison cell types.
    """
    lag_length = normalized_cchs.shape[2]
    ref_ids = uids[np.array(uid_ctypes) == ctype1]
    ctype1_avg_cchs = np.ma.zeros((len(ref_ids), len(comp_ctypes), lag_length))
    ctype1_norm_dists = np.ma.zeros((len(ref_ids), len(comp_ctypes)))
    for j, ctype2 in enumerate(comp_ctypes):
        cell_ctype1_avg_cchs, cell_ctype1_std_cchs, norm_dists = get_average_cells_ctype_cch(ctype2, ref_ids, uids, uid_ctypes, normalized_cchs, overlap_matrix, overlap_thresholds)
        normalizer = 1/np.max(np.abs(cell_ctype1_avg_cchs), axis=1)
        ctype1_avg_cchs[:,j] = cell_ctype1_avg_cchs * normalizer[:,None]
        ctype1_norm_dists[:,j] = norm_dists
    
    weights = np.tile(np.nan_to_num(1/ctype1_norm_dists, posinf=0, neginf=0), (lag_length,1,1)).transpose(1,2,0)
    ctype_average_cchs = np.ma.average(ctype1_avg_cchs, axis=0, weights=weights)
    ctype_average_cchs_std = np.ma.std(ctype1_avg_cchs, axis=0)
    return ctype_average_cchs, ctype_average_cchs_std

def get_ctype_templates(ref_types, comp_ctypes, uids, uid_ctypes, normalized_cchs, overlap_matrix, overlap_thresholds = [0,1.5]):
    """
    Computes the average cross-correlogram (CCH) between all units of a list of reference cell types and all units of a list of comparison cell types.

    Parameters:
    -----------
    all_units: celltable unit_table
        unit_table from specified piece_id and run_id
    ref_types : list
        A list of cell types to compute the average CCHs for.
    comp_ctypes : list
        A list of cell types to compare with the specified cell types.
    normalized_cchs : ndarray
        A 3D numpy array containing the normalized CCH matrix for all pairs of units in `ct`.
    overlap_matrix : ndarray
        A 2D numpy array containing the receptive field overlap matrix for all pairs of units in `ct`.

    Returns:
    --------
    average_cchs : ndarray
        A 3D numpy array containing the average CCHs between all units of the specified cell types and all units of the comparison cell types.
    average_cch_stds : ndarray
        A 3D numpy array containing the standard deviations of the average CCHs between all units of the specified cell types and all units of the comparison cell types.
    """
    lag_length = normalized_cchs.shape[2]
    ctype_average_cchs = np.ma.zeros((len(ref_types), len(comp_ctypes), lag_length))
    ctype_average_cchs_std = np.ma.zeros((len(ref_types), len(comp_ctypes), lag_length))
    for i, ctype1 in enumerate(ref_types):
        ctype_average_cchs[i], ctype_average_cchs_std[i] = get_ctype_template(ctype1, comp_ctypes, uids, uid_ctypes, normalized_cchs, overlap_matrix, overlap_thresholds)
    return ctype_average_cchs, ctype_average_cchs_std
    
##PREDICTIVE MODELING FUNCTIONS

def predict_cell_ctype(cell_cch_templates, cell_cch_stds, all_type_templates, all_type_template_stds, ref_type_inds, comp_type_inds):
    if len(np.shape(cell_cch_templates)) > 2:
        predictions = np.ma.zeros(cell_cch_templates.shape[0])
        avg_scores = np.ma.zeros((cell_cch_templates.shape[0], len(ref_type_inds)))
        scores = np.ma.zeros((cell_cch_templates.shape[0], len(ref_type_inds), len(comp_type_inds)))
        weights = np.ma.zeros((cell_cch_templates.shape[0], len(ref_type_inds), len(comp_type_inds)))
        for i, cell_templates in enumerate(cell_cch_templates):
            predictions[i], avg_scores[i], scores[i], weights[i] = predict_cell_ctype(cell_templates, cell_cch_stds[i], all_type_templates, all_type_template_stds, ref_type_inds, comp_type_inds)
        return predictions, avg_scores, scores, weights
    
    assert len(comp_type_inds) <= cell_cch_templates.shape[0]
    assert len(comp_type_inds) <= all_type_templates.shape[1]
    
    scores = np.zeros((len(ref_type_inds),len(comp_type_inds)))
    avg_template_stds = np.zeros((len(ref_type_inds),len(comp_type_inds)))
    avg_cell_stds = np.nan_to_num(np.array(np.mean(cell_cch_stds, axis=1)), posinf=0, neginf=0)
    for i, ref_type_ind in enumerate(ref_type_inds):
        comp_templates = all_type_templates[ref_type_ind,comp_type_inds]
        comp_stds = all_type_template_stds[ref_type_ind,comp_type_inds]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            scores[i] = np.corrcoef(cell_cch_templates[comp_type_inds], comp_templates)[len(comp_type_inds):,0:len(comp_type_inds)].diagonal()
            avg_template_stds[i] = np.mean(comp_stds, axis=1)
    
    avg_cell_stds[avg_cell_stds == 0] = 100
    masked_scores = np.ma.masked_array(scores, mask=np.isnan(scores))
    weights = np.nan_to_num(1/(avg_template_stds+avg_cell_stds[:,None]), posinf=0, neginf=0)
    avg_scores = np.ma.average(masked_scores, axis=1, weights=weights)
    prediction = ref_type_inds[np.ma.argmax(avg_scores)]
    return prediction, avg_scores, masked_scores, weights
    
##PLOTTING FUNCTIONS

def plot_ctype_template(average_cch, std, ctypes = [],fig = None, ax = None):
    """
    Plots the cross-correlation histogram (CCH) between two cell types.

    Args:
    average_cch (numpy.ndarray): The average CCH between two cell types.
    std (numpy.ndarray): The standard deviation of the CCH.
    ctypes (list): A list of two strings representing the cell types being compared.
    fig (matplotlib.figure.Figure): A matplotlib figure object to plot the CCH on.
    ax (matplotlib.axes.Axes): A matplotlib axes object to plot the CCH on.

    Returns:
    fig (matplotlib.figure.Figure): The matplotlib figure object containing the CCH plot.
    ax (matplotlib.axes.Axes): The matplotlib axes object containing the CCH plot.
    """
    window_size = average_cch.shape[0]
    lags = np.array(range(int(-(window_size-1)/2),int((window_size-1)/2)+1))
    if fig is None or ax is None:
        fig, ax = plt.subplots(1,1,figsize=(5,5))
    ax.plot(lags, average_cch, color = 'k')
    ax.fill_between(lags, average_cch - std, average_cch + std, alpha=0.5)
    if len(ctypes) == 2:
        ax.set_title(f"{ctypes[0]}-{ctypes[1]}")
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Normalized Correlation")
    return fig, ax

def plot_ctype_templates(average_cchs, stds, ref_type, ctypes = [],fig = None, axs = None):
    """
    Plots the cross-correlation histograms (CCHs) between a reference cell type and a list of comparison cell types.
    
    Args:
    - average_cchs (numpy.ndarray): A 2D numpy array of shape (n_cell_types, n_cell_types, n_lags) containing the average CCHs between all pairs of cell types.
    - stds (numpy.ndarray): A 2D numpy array of shape (n_cell_types, n_cell_types, n_lags) containing the standard deviations of the CCHs between all pairs of cell types.
    - ref_type (int): The index of the reference cell type.
    - ctypes (list): A list of cell types to compare with the reference cell type. If empty, all cell types will be compared.
    - fig (matplotlib.figure.Figure): The figure object to plot the templates on. If None, a new figure will be created.
    - axs (list): A list of matplotlib.axes.Axes objects to plot the templates on. If None, new axes will be created.
    
    Returns:
    - fig (matplotlib.figure.Figure): The figure object containing the plotted templates.
    - axs (list): A list of matplotlib.axes.Axes objects containing the plotted templates.
    """
    window_size = average_cchs.shape[2]
    lags = np.array(range(int(-(window_size-1)/2),int((window_size-1)/2)+1))
    if fig is None:
        fig, axs = plt.subplots(1,len(ctypes),figsize=(5*len(ctypes),5))
    axs = fig.get_axes()
    
    for i, comp_type in enumerate(ctypes):
        ax = axs[i]
        ax.plot(lags, average_cchs[ref_type,comp_type], color = 'k')
        ax.fill_between(lags, average_cchs[ref_type,comp_type] - stds[ref_type,comp_type], 
                        average_cchs[ref_type,comp_type] + stds[ref_type,comp_type], 
                        alpha=0.5)
        ax.set_title(f"{ref_type}-{comp_type}")
        ax.set_xlabel("Time (ms)")
        ax.set_ylabel("Normalized Correlation")
    fig.suptitle(f"{ref_type} Template")
    fig.tight_layout
    return fig, axs
    
def plot_ctype_template_grid(templates, stds, ctypes, figsize=()):
    """
    Plots a grid of cross-correlation histograms for each combination of cell types in the input list.

    Args:
    - templates (dict): A dictionary containing the cross-correlation histograms for each cell type combination.
    - stds (dict): A dictionary containing the standard deviations of the cross-correlation histograms for each cell type combination.
    - ctypes (list): A list of cell types to plot.
    - figsize (tuple): A tuple specifying the size of the figure. If not provided, the default size is (len(ctypes)*5, len(ctypes)*5).

    Returns:
    - fig (matplotlib.figure.Figure): The figure object containing the subplots.
    - axes (numpy.ndarray): An array of AxesSubplot objects representing the subplots.
    """
    if figsize == ():
        figsize = (len(ctypes)*5, len(ctypes)*5)
    fig, axes = plt.subplots(len(ctypes), len(ctypes), figsize=figsize)
    for i, ctype1 in enumerate(ctypes):
        for j, ctype2 in enumerate(ctypes):
            if (ctype1, ctype2) in templates.keys():
                average_ctype_cch = templates[(ctype1, ctype2)]
                average_ctype_cch_std = stds[(ctype1, ctype2)]
            else:
                continue
            if type(average_ctype_cch) == np.float64:
                print(f"Skipping {ctype1}-{ctype2}")
                continue
            fig, axes[i,j] = plot_ctype_template(average_ctype_cch, average_ctype_cch_std, [ctype1, ctype2], fig, axes[i,j])
    plt.tight_layout()
    return fig, axes

def plot_cell_ctype_template(ref_cch, ref_std, template_cchs, template_stds, template_ctype, comp_ctypes):
    """
    Plots the cross-correlation histograms (CCHs) of a reference cell against the CCH templates of a specified cell type.

    Args:
    - ref_cch (dict): A dictionary containing the cross-correlation histograms of the reference cell type.
    - ref_std (dict): A dictionary containing the standard deviations of the cross-correlation histograms of the reference cell type.
    - template_cchs (dict): A dictionary containing the template cross-correlation histograms of all cell types.
    - template_stds (dict): A dictionary containing the template standard deviations of the cross-correlation histograms of all cell types.
    - template_ctype (str): The cell type template to be plotted against the reference cell.
    - comp_ctypes (list): A list of cell types to be compared against the reference cell.

    Returns:
    - fig (matplotlib.figure.Figure): The figure object containing the subplots.
    - axs (numpy.ndarray): An array of AxesSubplot objects containing the subplots.
    """
    window_size = len(ref_cch[comp_ctypes[0]])
    lags = np.array(range(int(-(window_size-1)/2),int((window_size-1)/2)+1))
    fig, axs = plot_ctype_templates(template_cchs, template_stds, template_ctype, comp_ctypes)
    for i, comp_type in enumerate(comp_ctypes):
        axs[i].plot(lags, ref_cch[comp_type], color = 'r')
        axs[i].fill_between(lags, ref_cch[comp_type] - ref_std[comp_type], ref_cch[comp_type] + ref_std[comp_type], alpha=0.5)
    return fig, axs
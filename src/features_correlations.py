import sys, warnings

sys.path.append('../')
import warnings
import numpy as np
import pandas as pd
import cell_display_lib as cdl
from scipy import ndimage, io, spatial, stats
from scipy.signal import savgol_filter
from skimage.transform import resize
from sklearn.decomposition import PCA
from features import Feature
from skimage import measure
from shapely.geometry import LineString, Point, MultiPoint, Polygon, MultiPolygon
from scipy.ndimage.filters import gaussian_filter
import scipy
from tqdm import tqdm
from file_handling import wrapper
import istarmap
import multiprocessing as mp
from multiprocessing import Pool


import elephant, warnings
import quantities as pq
from elephant.conversion import BinnedSpikeTrain
from neo import SpikeTrain

MP_N_THREADS = 25
WINDOW_SIZE = 101
OVERLAP_THRESHOLD = 2

def calculate_radius(center_ref,center_comp,segments):
    direction = center_comp - center_ref
    connection = LineString([center_ref, center_comp + 100000*direction])
    hull = LineString(segments)
    
    int_pt = connection.intersection(hull)
    if type(int_pt) == Point:
        poi = int_pt.x, int_pt.y
    elif type(int_pt) == MultiPoint:
        poi = int_pt.geoms[-1].x, int_pt.geoms[-1].y
    else:
        minx, miny = np.min(segments,axis=0)
        maxx, maxy = np.max(segments,axis=0)
            
        hull = LineString(np.array([[minx,miny],[minx,maxy],[maxx,maxy],
                                    [maxx,miny],[minx,miny]]))
        int_pt = connection.intersection(hull)
        if type(int_pt) == Point:
            poi = int_pt.x, int_pt.y
        elif type(int_pt) == MultiPoint:
            poi = int_pt.geoms[-1].x, int_pt.geoms[-1].y
        else:
            poi = None
    return poi

def calculate_radii(params, param_cis):
    radii = {}
    for i, param in enumerate(params):
        center_ref, center_comp, segments = param
        radius = calculate_radius(center_ref,center_comp,segments)
        if radius is None:
            continue
        radii[param_cis[i]] = LineString([center_ref,radius])
    return radii

def calculate_overlap(rad1,rad2,connection):
    dist = connection.length
    rad_length1 = rad1.length
    rad_length2 = rad2.length
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        overlap = dist/np.mean([rad_length1, rad_length2])
    return overlap

def calculate_area_overlap(rfbool1,rfbool2):
    rf_or = np.logical_or(rfbool1,rfbool2)
    rf_and = np.logical_and(rfbool1,rfbool2)
    if np.sum(rf_or) == 0:
        return 0
    return np.sum(rf_and)/np.sum(rf_or)

def calculate_inner_product(sig_sta1,sig_sta2):
    inner_product = np.linalg.norm(
        (sig_sta1.T @ sig_sta2)[np.diag_indices_from(sig_sta1.T @ sig_sta2)])/(
            np.sqrt(np.linalg.norm((sig_sta1.T @ sig_sta1)
                                    [np.diag_indices_from(sig_sta1.T @ sig_sta1)])*
                    np.linalg.norm((sig_sta2.T @ sig_sta2)
                                    [np.diag_indices_from(sig_sta2.T @ sig_sta2)])))
    return inner_product
    
def calculate_overlaps(params, ci2s):
    overlaps = {}
    for i, ci2 in enumerate(ci2s):
        ci_params = params[i]
        overlaps[ci2] = calculate_overlap(ci_params[0],ci_params[1],ci_params[2])
        
    return overlaps

def calculate_area_overlaps(params, ci2s):
    area_overlaps = {}
    for i, ci2 in enumerate(ci2s):
        ci_params = params[i]
        area_overlaps[ci2] = calculate_area_overlap(ci_params[0],ci_params[1])
        
    return area_overlaps

def calculate_inner_products(params, ci2s):
    inner_products = {}
    for i, ci2 in enumerate(ci2s):
        ci_params = params[i]
        inner_products[ci2] = calculate_inner_product(ci_params[0],ci_params[1])
        
    return inner_products

def calculate_all_overlaps(overlap_params, area_params, inner_product_params, ci2s):
    overlaps = calculate_overlaps(overlap_params, ci2s)
    area_overlaps = calculate_area_overlaps(area_params, ci2s)
    inner_products = calculate_inner_products(inner_product_params, ci2s)
    
    return overlaps, area_overlaps, inner_products

def calculate_cchs(ci_index, ci2_indices, binned_spike_trains_1ms, binned_spike_trains_10ms, 
                   window_size = 101):
    cchs_10ms = []
    cchs_1ms = []
    bst1_1 = binned_spike_trains_1ms[ci_index]
    bst1_10 = binned_spike_trains_10ms[ci_index]
    for ci2_index in ci2_indices:
        
        bst2 = binned_spike_trains_1ms[ci2_index]
        
        k, lags = elephant.spike_train_correlation.cross_correlation_histogram(bst1_1, bst2, 
                                window=[int(-(window_size-1)/2), int((window_size-1)/2)], 
                                cross_correlation_coefficient=True)
        cchs_1ms.append(np.array(k.flatten()))
        
        bst2 = binned_spike_trains_10ms[ci2_index]
        
        k, lags = elephant.spike_train_correlation.cross_correlation_histogram(bst1_10, bst2, 
                                window=[int(-(window_size-1)/2), int((window_size-1)/2)], 
                                cross_correlation_coefficient=True)
        cchs_10ms.append(np.array(k.flatten()))
    return [cchs_1ms, cchs_10ms]

def generate_fast_cchs(spikes,delay,spike_bin_sizes,t_stop,num_units):
    spikes_delay = [s + delay for s in spikes]
    spikes_all = spikes + spikes_delay

    corrs = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        spike_trains = [SpikeTrain(spikes_all[cc], t_stop=t_stop) for cc in range(num_units * 2)]
        for spike_bin_size in spike_bin_sizes:
            binned_spike_trains = BinnedSpikeTrain([spike_trains[cc] for cc in range(num_units * 2)], bin_size=spike_bin_size)

            spike_corrs_big = elephant.spike_train_correlation.correlation_coefficient(binned_spike_trains)
            spike_corrs = np.zeros((num_units, num_units))
            spike_corrs[:, :] = spike_corrs_big[:num_units, num_units:]
            corrs.append(spike_corrs)
    return corrs

class Feature_rf_radii(Feature):
    name = 'rf radii'
    requires = {'unit':{'rf_convex_hull', 'hull_center_x', 'hull_center_y'}}
    provides = {'unit':{'rf_radii','rf_connections'}}
    
    def generate(self, ct, unit_indices, inpt, dtab=None):
        if dtab is None:
            dtab = ct.unit_table
        di = unit_indices[0][0:2]
        if (missing := self.check_requirements(ct, di)) is not None:
            print('Feature {}. missing requirements {}'.format(self.name, missing))
            return
        
        print("Generating Params")
        mp_params = []
        bad_units = []
        for ci in tqdm(unit_indices):
            connections = {}
            center_x = dtab.at[ci, 'hull_center_x']
            center_y = dtab.at[ci, 'hull_center_y']
            center_ref = np.array([center_x, center_y])
            segments = dtab.at[ci, 'rf_convex_hull'].a[0]
            params = []
            params_cis = []
            for ci2 in unit_indices:
                if ci == ci2:
                    continue
                center_x2 = dtab.at[ci2, 'hull_center_x']
                center_y2 = dtab.at[ci2, 'hull_center_y']
                center_comp = np.array([center_x2, center_y2])
                
                params.append((center_ref,center_comp,segments))
                params_cis.append(ci2)
                connections[ci2] = LineString([center_ref, center_comp])
            mp_params.append((params, params_cis))
            if connections is None or len(connections) == 0:
                bad_units.append(ci)
                dtab.at[ci, 'rf_connections'] = wrapper(None)
            else:
                dtab.at[ci, 'rf_connections'] = wrapper(connections)
        
        print("Calculating Radii")
            
        with mp.Pool(MP_N_THREADS) as pool:
            radii = list(tqdm(pool.istarmap(calculate_radii, mp_params), total=len(mp_params)))
            
            pool.close()
            pool.join()
        
        for i, ci in enumerate(unit_indices):
            if radii[i] is None or len(radii[i]) == 0:
                bad_units.append(ci)
                dtab.at[ci, 'rf_radii'] = wrapper(None)
            else:
                dtab.at[ci, 'rf_radii'] = wrapper(radii[i])
            
        if len(bad_units) > 0:
            print(f'Found {len(bad_units)} bad units, No radii found, Setting units as invalid')
            dtab.loc[bad_units, 'valid'] = False
        self.update_valid_columns(ct, di)
        
class Feature_rf_overlaps(Feature):
    name = 'rf overlaps'
    requires = {'unit':{'rf_radii', 'rf_connections', 'map_sig_stixels',
                        'map_sta_peak'}}
    provides = {'unit':{'rf_overlaps', 'rf_area_overlaps', 'rf_inner_products'}}
    
    def generate(self, ct, unit_indices, inpt, dtab=None):
        if dtab is None:
            dtab = ct.unit_table
        di = unit_indices[0][0:2]
        if (missing := self.check_requirements(ct, di)) is not None:
            print('Feature {}. missing requirements {}'.format(self.name, missing))
            return
        
        print("Generating Params")
        all_overlap_params = []
        primary_channels = {}
        for ctype in dtab['label_manual_text'].unique():
            
            ctype_units = dtab.query('label_manual_text == @ctype and sta_extremes == sta_extremes and valid == True')
            primary_channels[ctype] = cdl.channelize(ct.find_primary_channel(ctype_units))
        
        for i, ci in tqdm(enumerate(unit_indices), total=len(unit_indices)):
            ci_overlap_params = []
            ci_inner_product_params = []
            ci_area_overlap_params = []
            ci2s = []
            radii = dtab.at[ci, 'rf_radii'].a
            ci_connections = dtab.at[ci, 'rf_connections'].a
            sig_stixels1 = dtab.at[ci, 'map_sig_stixels'].a
            sta_peak1 = dtab.at[ci, 'map_sta_peak'].a
            
            ci_type = dtab.at[ci, 'label_manual_text']
            rf_bool1 = np.fliplr(dtab.at[ci, 'map_rf_bool'].a[:,:,primary_channels[ci_type]]).T
            
            for j, ci2 in enumerate(unit_indices):
                if i >= j:
                    continue
                if ci2 not in radii:
                    continue
                radii2 = dtab.at[ci2, 'rf_radii'].a
                if ci not in radii2:
                    continue
                connection = ci_connections[ci2]
                radius = radii[ci2]
                radius2 = radii2[ci]
                ci_overlap_params.append((radius, radius2, connection))
                
                sig_stixels2 = dtab.at[ci2, 'map_sig_stixels'].a
                sig_stixels_u = np.logical_or(sig_stixels1, sig_stixels2)
                sig_sta1 = sta_peak1[(sig_stixels_u),:]
                sig_sta2 = dtab.at[ci2, 'map_sta_peak'].a[(sig_stixels_u),:]
                ci_inner_product_params.append((sig_sta1, sig_sta2))
                
                ci2_type = dtab.at[ci2, 'label_manual_text']
                rf_bool2 = np.fliplr(dtab.at[ci2, 'map_rf_bool'].a[:,:,primary_channels[ci2_type]]).T
                ci_area_overlap_params.append((rf_bool1, rf_bool2))
                
                ci2s.append(ci2)
                
            all_overlap_params.append((ci_overlap_params, ci_area_overlap_params, 
                                       ci_inner_product_params, ci2s))
                
        print("Calculating Overlaps, Area Overlaps, and Inner Products")
        with mp.Pool(MP_N_THREADS) as pool:
            half_overlaps = list(tqdm(pool.istarmap(calculate_all_overlaps, all_overlap_params),
                                      total=len(all_overlap_params)))
            pool.close()
            pool.join()
        
        all_overlaps = np.zeros((len(unit_indices), len(unit_indices)))
        all_area_overlaps = np.zeros((len(unit_indices), len(unit_indices)))
        all_inner_products = np.zeros((len(unit_indices), len(unit_indices)))
        for i, ci in enumerate(unit_indices):
            for j, ci2 in enumerate(unit_indices):
                if ci2 not in half_overlaps[i][0]:
                    continue
                all_overlaps[i,j] = half_overlaps[i][0][ci2]
                all_overlaps[j,i] = half_overlaps[i][0][ci2]
                all_area_overlaps[i,j] = half_overlaps[i][1][ci2]
                all_area_overlaps[j,i] = half_overlaps[i][1][ci2]
                all_inner_products[i,j] = half_overlaps[i][2][ci2]
                all_inner_products[j,i] = half_overlaps[i][2][ci2]

            overlaps = {ci2: all_overlaps[i,j] for j, ci2 in enumerate(unit_indices)}
            area_overlaps = {ci2: all_area_overlaps[i,j] for j, ci2 in enumerate(unit_indices)}
            inner_products = {ci2: all_inner_products[i,j] for j, ci2 in enumerate(unit_indices)}
                
            dtab.at[ci, 'rf_overlaps'] = wrapper(overlaps)
            dtab.at[ci, 'rf_area_overlaps'] = wrapper(area_overlaps)
            dtab.at[ci, 'rf_inner_products'] = wrapper(inner_products)
        
        self.update_valid_columns(ct, di)
        
class Feature_cross_correlations_complete(Feature):
    name = 'calculate cross correlations'
    requires = {'unit':{'spike_times'}}
    provides = {'unit':{'1ms CCHs', '10ms CCHs','CCs'}}
    
    def generate(self, ct, unit_indices, inpt, dtab=None):
        if dtab is None:
            dtab = ct.unit_table
        di = unit_indices[0][0:2]
        if (missing := self.check_requirements(ct, di)) is not None:
            print('Feature {}. missing requirements {}'.format(self.name, missing))
            return
        
        num_units = len(unit_indices)
        spikes = [s.a / 20000 * pq.s for s in dtab.loc[unit_indices, 'spike_times']]
        t_stop = np.ceil(np.max([np.max(spikes[cc]) for cc in range(num_units)])) + 1
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            spike_trains = [SpikeTrain(spikes[cc], t_stop=t_stop) for cc in range(num_units)]
            binned_spike_trains_1ms = BinnedSpikeTrain([spike_trains[cc] for cc in range(num_units)],
                                                       bin_size=0.001 * pq.s)
            binned_spike_trains_10ms = BinnedSpikeTrain([spike_trains[cc] for cc in range(num_units)],
                                                        bin_size=0.01 * pq.s)
        
        all_cchs_1ms = np.zeros((num_units, num_units, WINDOW_SIZE))
        all_cchs_10ms = np.zeros((num_units, num_units, WINDOW_SIZE))
        all_ccs = np.zeros((num_units, num_units))
        
        with mp.Pool(MP_N_THREADS) as pool:
            half_cchs_1ms = list(tqdm(pool.istarmap(calculate_cchs, 
                                        [(ci_index,[*range(ci_index+1,num_units)],
                                          binned_spike_trains_1ms)
                                         for ci_index in range(num_units)]), 
                                  total=num_units))
            pool.close()
            pool.join()

        with mp.Pool(MP_N_THREADS) as pool:
            half_cchs_10ms = list(tqdm(pool.istarmap(calculate_cchs, 
                                        [(ci_index,[*range(ci_index+1,num_units)],
                                          binned_spike_trains_10ms)
                                         for ci_index in range(num_units)]), 
                                  total=num_units))
            pool.close()
            pool.join()
        
        for i, ci in enumerate(unit_indices):
            all_cchs_1ms[i, i+1:, :] = half_cchs_1ms[i]
            all_cchs_1ms[i+1:, i, :] = np.flip(half_cchs_1ms[i])
            all_cchs_10ms[i, i+1:, :] = half_cchs_10ms[i]
            all_cchs_10ms[i+1:, i, :] = np.flip(half_cchs_10ms[i])
            all_ccs[i, i+1:] = half_cchs_10ms[i][(WINDOW_SIZE-1)//2]
            all_ccs[i+1:, i] = half_cchs_10ms[i][(WINDOW_SIZE-1)//2]
            
            cchs_1ms = {ci2: all_cchs_1ms[i,j,:] for j, ci2 in enumerate(unit_indices)}
            cchs_10ms = {ci2: all_cchs_10ms[i,j,:] for j, ci2 in enumerate(unit_indices)}
            ccs = {ci2: all_ccs[i,j] for j, ci2 in enumerate(unit_indices)}
            
            dtab.at[ci, '1ms CCHs'] = wrapper(cchs_1ms)
            dtab.at[ci, '10ms CCHs'] = wrapper(cchs_10ms)
            dtab.at[ci, 'CCs'] = wrapper(ccs)
        
        self.update_valid_columns(ct, di)
        
class Feature_overlapping_cross_correlations(Feature):
    name = 'calculate overlapping cross correlations'
    requires = {'unit':{'spike_times','rf_overlaps'}}
    provides = {'unit':{'1ms CCHs', '10ms CCHs','CCs'}}
    
    def generate(self, ct, unit_indices, inpt, dtab=None):
        if dtab is None:
            dtab = ct.unit_table
        di = unit_indices[0][0:2]
        if (missing := self.check_requirements(ct, di)) is not None:
            print('Feature {}. missing requirements {}'.format(self.name, missing))
            return
        
        num_units = len(unit_indices)
        spikes = [s.a / 20000 * pq.s for s in dtab.loc[unit_indices, 'spike_times']]
        t_stop = np.ceil(np.max([np.max(spikes[cc]) for cc in range(num_units)])) + 1
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            spike_trains = [SpikeTrain(spikes[cc], t_stop=t_stop) for cc in range(num_units)]
            binned_spike_trains_1ms = BinnedSpikeTrain([spike_trains[cc] for cc in range(num_units)],
                                                       bin_size=0.001 * pq.s)
            binned_spike_trains_10ms = BinnedSpikeTrain([spike_trains[cc] for cc in range(num_units)],
                                                        bin_size=0.01 * pq.s)
        
        
        all_cchs_1ms = np.zeros((num_units, num_units, WINDOW_SIZE))
        all_cchs_10ms = np.zeros((num_units, num_units, WINDOW_SIZE))
        all_ccs = np.zeros((num_units, num_units))
        
        print("Generating Params")
        cch_params = []
        for i, ci in tqdm(enumerate(unit_indices), total=len(unit_indices)):
            overlaps = dtab.at[ci, 'rf_overlaps'].a
            paired_units = []
            for j, ci2 in enumerate(unit_indices):
                if i >= j:
                    continue
                if overlaps[ci2] < OVERLAP_THRESHOLD:
                    paired_units.append(j)
            cch_params.append((i, paired_units, binned_spike_trains_1ms, binned_spike_trains_10ms))
        
        print("Calculating CCHs")
        with mp.Pool(MP_N_THREADS) as pool:
            half_cchs = list(tqdm(pool.istarmap(calculate_cchs, cch_params),total=len(cch_params)))
            pool.close()
            pool.join() 
        
        for i, param in enumerate(cch_params):
            ci_index, ci2_indices, _, _ = param
            ci = unit_indices[ci_index]
            for j, ci2_index in enumerate(ci2_indices):
                all_cchs_1ms[ci_index, ci2_index, :] = half_cchs[i][0][j]
                all_cchs_10ms[ci_index, ci2_index, :] = half_cchs[i][1][j]
                all_cchs_1ms[ci2_index, ci_index, :] = np.flip(half_cchs[i][0][j])
                all_cchs_10ms[ci2_index, ci_index, :] = np.flip(half_cchs[i][1][j])
                all_ccs[ci_index, ci2_index] = half_cchs[i][1][j][(WINDOW_SIZE-1)//2]
                all_ccs[ci2_index, ci_index] = half_cchs[i][1][j][(WINDOW_SIZE-1)//2]
            
            cchs_1ms = {ci2: all_cchs_1ms[i,j,:] for j, ci2 in enumerate(unit_indices)}
            cchs_10ms = {ci2: all_cchs_10ms[i,j,:] for j, ci2 in enumerate(unit_indices)}
            ccs = {ci2: all_ccs[i,j] for j, ci2 in enumerate(unit_indices)}
            
            dtab.at[ci, '1ms CCHs'] = wrapper(cchs_1ms)
            dtab.at[ci, '10ms CCHs'] = wrapper(cchs_10ms)
            dtab.at[ci, 'CCs'] = wrapper(ccs)
        
        self.update_valid_columns(ct, di)

        
class Feature_cross_correlations_complete_fast(Feature):
    name = 'calculate cross correlations'
    requires = {'unit':{'spike_times'}}
    provides = {'unit':{'1ms CCHs', '10ms CCHs','CCH_delays'}}
    
    def generate(self, ct, unit_indices, inpt, dtab=None):
        if dtab is None:
            dtab = ct.unit_table
        di = unit_indices[0][0:2]
        if (missing := self.check_requirements(ct, di)) is not None:
            print('Feature {}. missing requirements {}'.format(self.name, missing))
            return
        
        # make much faster CCH

        # delays_half = np.around(np.concatenate((np.arange(0, .051, .001),np.arange(.051, .15, .002))), 3) * pq.s
        delays_half = np.around(np.arange(0, .101, .001), 3) * pq.s
        # delays = np.around(np.logspace(-3, -.5, 15), 4)
        # delays = np.concatenate([[0],delays]) * pq.s
        # delays_half = [0, .002, .003, .004] * pq.s

        num_cells = len(unit_indices)
        spikes = [s.a / 20000 * pq.s for s in dtab.loc[unit_indices, 'spike_times']]

        print(delays_half)
        cch_all_1ms = np.zeros((num_cells, num_cells, len(delays_half)))
        cch_all_10ms = np.zeros((num_cells, num_cells, len(delays_half)))

        t_stop = np.ceil(np.max([np.max(spikes[cc]) for cc in range(num_cells)])) + 1
            
        params = []
        for dd, delay in enumerate(delays_half):
            params.append((spikes, delay, [0.001 * pq.s, 0.01 * pq.s], t_stop, num_cells))
            
            # spikes_delay = [s + delay for s in spikes]
            # spikes_all = spikes + spikes_delay

            # with warnings.catch_warnings():
            #     warnings.simplefilter("ignore")
            #     spike_trains = [SpikeTrain(spikes_all[cc], t_stop=t_stop) for cc in range(num_cells * 2)]
            #     binned_spike_trains_1ms = BinnedSpikeTrain([spike_trains[cc] for cc in range(num_cells * 2)], bin_size=0.001 * pq.s)
            #     binned_spike_trains_10ms = BinnedSpikeTrain([spike_trains[cc] for cc in range(num_cells * 2)], bin_size=0.01 * pq.s)
                # params_1ms.append((binned_spike_trains_1ms, num_cells))
                # params_10ms.append((binned_spike_trains_10ms, num_cells))
                
                
        with mp.Pool(MP_N_THREADS) as pool:
            cch_delay = list(tqdm(pool.istarmap(generate_fast_cchs, params), total=len(params)))
            pool.close()
            pool.join()

        for dd, delay in enumerate(delays_half):
            cch_all_1ms[:, :, dd] = cch_delay[dd][0]
            cch_all_10ms[:, :, dd] = cch_delay[dd][1]

        cch_1ms = np.concatenate([np.flip(cch_all_1ms, 2), np.swapaxes(cch_all_1ms, 0,1)[...,1:]], axis=2)
        cch_10ms = np.concatenate([np.flip(cch_all_10ms, 2), np.swapaxes(cch_all_10ms, 0,1)[...,1:]], axis=2)
        delays = np.concatenate([-np.flip(delays_half), delays_half[1:]])
            
        for i, ci in enumerate(unit_indices):
            cchs_1ms = {ci2: cch_1ms[i,j,:] for j, ci2 in enumerate(unit_indices)}
            cchs_10ms = {ci2: cch_10ms[i,j,:] for j, ci2 in enumerate(unit_indices)}
            
            dtab.at[ci, '1ms CCHs'] = wrapper(cchs_1ms)
            dtab.at[ci, '10ms CCHs'] = wrapper(cchs_10ms)
            dtab.at[ci, 'CCH_delays'] = wrapper(delays)
        
        self.update_valid_columns(ct, di)
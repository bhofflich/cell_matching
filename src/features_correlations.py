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

def calculate_radii(params):
    radii_values = []
    for i, param in enumerate(params):
        center_ref, center_comp, segments = param
        if np.array_equal(center_ref, center_comp):
            radii_values.append(0)
            continue
        radius = calculate_radius(center_ref, center_comp, segments)
        if radius is not None:
            radii_values.append(LineString([center_ref, radius]).length)
        else:
            radii_values.append(0)
    
    return np.array(radii_values)

def calculate_overlap(rad1,rad2,dist):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if rad1 == 0 or rad2 == 0:
            return 0
        overlap = dist/np.mean([rad1, rad2])
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
    
def calculate_overlaps(params, num_ci2s):
    overlap_values = []
    for i in range(num_ci2s):
        ci_params = params[i]
        overlap_values.append(calculate_overlap(ci_params[0], ci_params[1], ci_params[2]))
        
    return np.array(overlap_values)

def calculate_area_overlaps(params, num_ci2s):
    area_overlap_values = []
    for i in range(num_ci2s):
        ci_params = params[i]
        area_overlap_values.append(calculate_area_overlap(ci_params[0], ci_params[1]))
        
    return np.array(area_overlap_values)

def calculate_inner_products(params, num_ci2s):
    inner_product_values = []
    for i in range(num_ci2s):
        ci_params = params[i]
        inner_product_values.append(calculate_inner_product(ci_params[0], ci_params[1]))
        
    return np.array(inner_product_values)

def calculate_all_overlaps(overlap_params, area_params, inner_product_params, num_ci2s):
    overlaps = calculate_overlaps(overlap_params, num_ci2s)
    area_overlaps = calculate_area_overlaps(area_params, num_ci2s)
    inner_products = calculate_inner_products(inner_product_params, num_ci2s)
    
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
    requires = {'unit': {'rf_convex_hull', 'hull_center_x', 'hull_center_y'}}
    provides = {'unit': {'rf_radii'}}

    def generate(self, ct, unit_indices, inpt, dtab=None):
        if dtab is None:
            dtab = ct.unit_table
        di = unit_indices[0][0:2]
        if (missing := self.check_requirements(ct, di)) is not None:
            print(f'Feature {self.name}. missing requirements {missing}')
            return

        print("Generating Params")
        mp_params = []
        bad_units = []
        for ci in tqdm(unit_indices):
            center_x = dtab.at[ci, 'hull_center_x']
            center_y = dtab.at[ci, 'hull_center_y']
            center_ref = np.array([center_x, center_y])
            segments = dtab.at[ci, 'rf_convex_hull'].a[0]

            # Collect parameters for each unit
            params = [(center_ref, np.array([dtab.at[ci2, 'hull_center_x'], dtab.at[ci2, 'hull_center_y']]), segments)
                      for ci2 in unit_indices]
            
            if not params:
                bad_units.append(ci)

            mp_params.append(params)
        
        print("Calculating Radii")

        with mp.Pool(MP_N_THREADS) as pool:
            radii_results = list(tqdm(pool.imap(calculate_radii, mp_params), total=len(mp_params)))

        # Process results
        radii_results = np.around(np.array(radii_results), decimals=4)
        ct.dataset_table.at[di, 'rf_radii'] = wrapper(radii_results)
        ct.dataset_table.at[di, 'radii_ids'] = wrapper(unit_indices)

        if bad_units:
            print(f'Found {len(bad_units)} bad units, No radii found, Setting units as invalid')
            dtab.loc[bad_units, 'valid'] = False

        self.update_valid_columns(ct, di)

        
class Feature_rf_overlaps(Feature):
    name = 'rf overlaps'
    requires = {'unit':{ 'map_sig_stixels', 'map_sta_peak'},
                'dataset':{'rf_radii', 'radii_ids'}}
    provides = {'dataset':{'rf_overlaps', 'rf_area_overlaps', 'rf_inner_products', 'overlap_ids'}}
    
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
            
            ctype_units = dtab.query('label_manual_text == @ctype and valid == True')
            if len(ctype_units) == 0:
                continue
            primary_channels[ctype] = cdl.channelize(ct.find_primary_channel(ctype_units))
        
        all_radii = ct.dataset_table.at[di, 'rf_radii'].a
        for i, ci in tqdm(enumerate(unit_indices), total=len(unit_indices)):
            ci_overlap_params = []
            ci_inner_product_params = []
            ci_area_overlap_params = []
            radii = all_radii[i,:]
            ci_center = np.array([dtab.at[ci, 'hull_center_x'], dtab.at[ci, 'hull_center_y']])
            sig_stixels1 = dtab.at[ci, 'map_sig_stixels'].a
            sta_peak1 = dtab.at[ci, 'map_sta_peak'].a
            
            ci_type = dtab.at[ci, 'label_manual_text']
            rf_bool1 = np.fliplr(dtab.at[ci, 'map_rf_bool'].a[:,:,primary_channels[ci_type]]).T
            
            for j, ci2 in enumerate(unit_indices):
                radii2 = all_radii[j,:]
                ci2_center = np.array([dtab.at[ci2, 'hull_center_x'], dtab.at[ci2, 'hull_center_y']])
                dist = np.linalg.norm(ci_center - ci2_center)
                radius = radii[j]
                radius2 = radii2[i]
                ci_overlap_params.append((radius, radius2, dist))
                
                sig_stixels2 = dtab.at[ci2, 'map_sig_stixels'].a
                sig_stixels_u = np.logical_or(sig_stixels1, sig_stixels2)
                sig_sta1 = sta_peak1[(sig_stixels_u),:]
                sig_sta2 = dtab.at[ci2, 'map_sta_peak'].a[(sig_stixels_u),:]
                ci_inner_product_params.append((sig_sta1, sig_sta2))
                
                ci2_type = dtab.at[ci2, 'label_manual_text']
                rf_bool2 = np.fliplr(dtab.at[ci2, 'map_rf_bool'].a[:,:,primary_channels[ci2_type]]).T
                ci_area_overlap_params.append((rf_bool1, rf_bool2))
                
            all_overlap_params.append((ci_overlap_params, ci_area_overlap_params, 
                                       ci_inner_product_params, len(unit_indices)))
                
        print("Calculating Overlaps, Area Overlaps, and Inner Products")
        with mp.Pool(MP_N_THREADS) as pool:
            overlaps = list(tqdm(pool.istarmap(calculate_all_overlaps, all_overlap_params),
                                      total=len(all_overlap_params)))
            pool.close()
            pool.join()
        
        all_overlaps = np.array([overlaps[i][0] for i in range(len(overlaps))])
        all_area_overlaps = np.array([overlaps[i][1] for i in range(len(overlaps))])
        all_inner_products = np.array([overlaps[i][2] for i in range(len(overlaps))])
        all_overlaps = np.around(all_overlaps, decimals=4)
        all_area_overlaps = np.around(all_area_overlaps, decimals=4)
        all_inner_products = np.around(all_inner_products, decimals=4)
            
        ct.dataset_table.at[di, 'rf_overlaps'] = wrapper(all_overlaps)
        ct.dataset_table.at[di, 'rf_area_overlaps'] = wrapper(all_area_overlaps)
        ct.dataset_table.at[di, 'rf_inner_products'] = wrapper(all_inner_products)
        ct.dataset_table.at[di, 'overlap_ids'] = wrapper(unit_indices)
        
        self.update_valid_columns(ct, di)
        
class Feature_cross_correlations_complete_fast(Feature):
    name = 'calculate cross correlations'
    requires = {'unit':{'spike_times'}}
    provides = {'dataset':{'cch_1ms', 'cch_10ms','cch_delays', 'cch_ids'}}
    
    def generate(self, ct, unit_indices, inpt, dtab=None):
        if dtab is None:
            dtab = ct.unit_table
        di = unit_indices[0][0:2]
        if (missing := self.check_requirements(ct, di)) is not None:
            print('Feature {}. missing requirements {}'.format(self.name, missing))
            return
        
        # make much faster CCH

        # delays_half = np.around(np.arange(0, .101, .001), 3) * pq.s
        delays_half = np.hstack((np.around(np.arange(0, .01, .0005), 4), np.around(np.arange(.01, .101, .001), 3))) * pq.s
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
            
        ct.dataset_table.at[di, 'cch_1ms'] = wrapper(np.around(cch_1ms, decimals=6))
        ct.dataset_table.at[di, 'cch_10ms'] = wrapper(np.around(cch_10ms, decimals=6))
        ct.dataset_table.at[di, 'cch_delays'] = wrapper(delays)
        ct.dataset_table.at[di, 'cch_ids'] = wrapper(unit_indices)
        
        self.update_valid_columns(ct, di)
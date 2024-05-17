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
from shapely.geometry import Polygon, MultiPolygon
from scipy.ndimage.filters import gaussian_filter
import scipy
from tqdm import tqdm
import file_handling
import skimage
from skimage.morphology import convex_hull_image

# from skfda.preprocessing.dim_reduction.projection import FPCA
# from skfda import FDataGrid
# # from skfda.misc.regularization import TikhonovRegularization
# from skfda.misc.regularization import L2Regularization
# # from skfda.misc.operators import LinearDifferentialOperator
# # regularization = TikhonovRegularization(LinearDifferentialOperator(2))
# regularization = L2Regularization()


def calculate_sig_stixels_simple(S, num_frames=20, alpha=1e-100, color_channels=None, num_sig_stixels_threshold=3):
    '''
    S is the STA with dimensions X, Y, Time, Color (zero mean aka contrast mode)
    num_frames is the number of frames from the beginning and end of the STA that will be used for calculating
       the noise and the signal values, respectively
    alpha is the chance of errors
    '''
    if color_channels is None:
        color_channels = range(S.shape[3])

    # alpha_orig = alpha
    sig_stixels = [False]

    early_frames = np.mean(S[..., 0:num_frames, color_channels], 2)  # first num_frames of the sta should be empty noise
    std_this_cell = stats.median_abs_deviation(early_frames.flatten(), scale='normal')
    input_frame = np.sum(np.power(np.mean(S[..., -(num_frames):-2, color_channels], 3) / std_this_cell, 2), 2)
                # mean over color, normalize by std, square for power, sum over time

    pvalues = stats.chi2(df=num_frames).sf(input_frame)# + np.finfo(float).eps  # chi squares survival = 1 - CDF
    pvalues *= (S.shape[0] * S.shape[1])  # bonferroni correction

    # loop over alpha threshold level
    # loops = 0
    # while np.sum(sig_stixels) < 3:
    while np.count_nonzero(sig_stixels) < num_sig_stixels_threshold:
        if alpha > 100:
            sig_stixels = np.zeros([S.shape[0], S.shape[1]], dtype=bool)
            sig_stixels[0,0] = True # something isn't working, just return the top left stixel for now
            return sig_stixels, None, None

        sig_stixels = pvalues < alpha
        alpha *= 1.01
        # loops += 1
    # print(f'alpha {alpha}, count {np.count_nonzero(sig_stixels)}, loops {loops}, pvalmin {np.min(pvalues)}, std_this_cell {std_this_cell}')

    # print(f'std {std_this_cell}, alpha {alpha}, count {np.count_nonzero(sig_stixels)}')
    return sig_stixels, alpha / 2, pvalues



def calculate_sig_stixels(S, num_frames=9, alpha=1e-8, color_channels=None, valid_stixels=None):
    # S is the STA with dimensions X, Y, Time, Color (zero mean aka contrast mode)
    # num_frames is the number of frames from the beginning and end of the STA that will be used for calculating
    #    the noise and the signal values, respectively
    # alpha is the chance of errors
    if color_channels is None:
        color_channels = range(S.shape[3])

    # alpha_orig = alpha
    sig_stixels = [False]
    loop_count = 0
    # loop over alpha threshold level
    while not np.any(sig_stixels):
        loop_count += 1
        # if loop_count > 10:
        #     print('sig stix loop count {}, alpha {}'.format(loop_count, alpha))
        if alpha > 100:
            sig_stixels = np.zeros([S.shape[0], S.shape[1]], dtype=bool)
            sig_stixels[0,0] = True # something isn't working, just return the top left stixel for now
            return sig_stixels, None, None
            # return None, None, None

        done = False
        using_sig_valid = False
        # loop over selection of noise region
        while not done:
            # some combined STA have regions of zero data, which throws off noise calculations
            if valid_stixels is None:
                valid_stixels = np.sum(np.sum(np.abs(S[..., 0:9, :]), axis=2), axis=2)
                valid_stixel_thresh = np.min(valid_stixels) * 2 # seems to work fine, but should be any stix that sums to zero. Noise must add up?
                valid_stixels = valid_stixels > valid_stixel_thresh

            S_valid = S[valid_stixels]
            early_frames = np.mean(S_valid[...,0:num_frames,color_channels], 2) # first num_frames of the sta should be empty noise

            std_this_cell = stats.median_abs_deviation(early_frames.flatten(), scale='normal')

            input_frame = np.sum(np.power(np.mean(S[...,-(num_frames+1):-1,color_channels], 3) / std_this_cell, 2), 2) # mean over color, normalize by std, square for power, sum over time
            pvalues = stats.chi2(df=num_frames).sf(input_frame) + np.finfo(float).eps # chi squares survival = 1 - CDF
            pvalues *= (S.shape[0] * S.shape[1]) # bonferroni correction
            # if iterate:
            #     sig_stixels = 0
            #     alpha_sigstix_temp = alpha
            #     while not np.any(sig_stixels):
            #         if not np.any(sig_stixels):
            #             # print('no sig stix at alpha = {}, increasing x 10'.format(self.alpha_sigstix))
            #             alpha_sigstix_temp *= 10
            # else:
            sig_stixels = pvalues < alpha

            if np.count_nonzero(sig_stixels) < 10 or using_sig_valid: # found a small RF okay
                done = True
            else:
                valid_stixels = sig_stixels # use the sig stixels as the valid sources of noise and go again
                using_sig_valid = True # only do that loop once

            # print(done, using_sig_valid, np.count_nonzero(sig_stixels))

        # BY correction alternative:
        # threshold_by = baseline_by * a_ratio
        # h, pvalues_by = fdrcorrection(pvalues.flatten(), alpha=1, method='negcorr')
        # pvalues_by = pvalues_by.reshape(input_frame.shape)
        # hypothesis_by = pvalues_by < threshold_by

        alpha *= 2

    return sig_stixels, alpha, pvalues, valid_stixels




def calculate_sta_fpca(S, sig_stixels_pca, coli, time_l):
    num_components = 1
    # choose which stixels
    S_to_fit = S[sig_stixels_pca, ..., coli]
    # S_to_fit = S_to_fit.reshape(S_to_fit.shape[0], S_to_fit.shape[1] * S_to_fit.shape[2])

    # create representation of timecourses
    S_fdgrid = FDataGrid(S_to_fit, grid_points=time_l)
    # basis = skfda.representation.basis.BSpline(domain_range=[0,1], n_basis=10)
    # S_fdgrid = S_fdgrid.to_basis(basis)

    # Do PCA on those TCs
    weights = list(time_l - np.min(time_l) + .1) # up sloping linear ramp
    fpca = FPCA(n_components=num_components, weights=weights, regularization=regularization)
    fpca.fit(S_fdgrid)
    # fpca = PCA(n_components=num_components)
    # fpca.fit(S_to_fit)
    eigenvect = fpca.components_ #.to_grid()
    # eigenvect = np.zeros(S_to_fit.shape[1])
    # ic(eigenvect)
    eigenvect = eigenvect.data_matrix[0].T[0]

    # Smoothing
    # eigenvect = savgol_filter(eigenvect, 7, 3, mode='constant', cval=0.0)

    polarity_component = 1 if np.max(eigenvect) >= np.max(np.abs(eigenvect)) else -1
    eigenvect *= polarity_component
    # eigenvect_orig *= polarity_component

    # eigenvect /= np.max(eigenvect)
    # eigenvect_orig /= np.max(eigenvect_orig)

    return eigenvect


def calculate_sta_fpca_single(S, sig_stixels_pca, time_l):
    num_components = 1
    # choose which stixels
    S_to_fit = S[sig_stixels_pca, ...]
    print(S_to_fit.shape)
    S_to_fit = S_to_fit.reshape(S_to_fit.shape[0], S_to_fit.shape[1] * S_to_fit.shape[2])
    print(S_to_fit.shape)

    # create representation of timecourses
    grid = np.concatenate([time_l, time_l, time_l])
    print(grid.shape)
    S_fdgrid = FDataGrid(S_to_fit, grid_points=grid)
    # basis = skfda.representation.basis.BSpline(domain_range=[0,1], n_basis=10)
    # S_fdgrid = S_fdgrid.to_basis(basis)

    print(np.any(np.isinf(S_to_fit)))

    # Do PCA on those TCs
    weights = list(time_l - np.min(time_l) + .1) # up sloping linear ramp
    weights = list(np.concatenate([weights, weights, weights]))
    fpca = FPCA(n_components=num_components, weights=weights, regularization=regularization)
    fpca.fit(S_fdgrid)
    # fpca = PCA(n_components=num_components)
    # fpca.fit(S_to_fit)
    eigenvect = fpca.components_ #.to_grid()
    # eigenvect = np.zeros(S_to_fit.shape[1])
    # ic(eigenvect)
    eigenvect = eigenvect.data_matrix[0].T[0]

    # Smoothing
    # eigenvect = savgol_filter(eigenvect, 7, 3, mode='constant', cval=0.0)

    polarity_component = 1 if np.argmax(eigenvect) > np.argmin(eigenvect) else -1
    # polarity_component = 1 if np.max(eigenvect) >= np.max(np.abs(eigenvect)) else -1
    eigenvect *= polarity_component
    # eigenvect_orig *= polarity_component

    # eigenvect /= np.max(eigenvect)
    # eigenvect_orig /= np.max(eigenvect_orig)

    return eigenvect


def calculate_sta_pca_single(S, sig_stixels_pca):
    num_components = 1
    S_to_fit = S[sig_stixels_pca, ...]
    S_to_fit = S_to_fit.reshape(S_to_fit.shape[0], S_to_fit.shape[1] * S_to_fit.shape[2])
    pca = PCA(n_components=num_components)
    pca.fit(S_to_fit)

    eigenvect = pca.components_.reshape(S.shape[2], S.shape[3])
    polarity_component = 1 if np.argmax(eigenvect) > np.argmin(eigenvect) else -1
    # polarity_component = 1 if np.max(eigenvect) >= np.max(np.abs(eigenvect)) else -1
    eigenvect *= polarity_component

    return eigenvect


def tc_snr(tc, mode='energy', no_noise=False, frame_noise=10, frame_signal=25):
    if mode == 'energy':
        noise = np.mean(np.power(tc[:frame_noise], 2))
        signal = np.mean(np.power(tc[frame_signal:], 2))
    elif mode == 'sum':
        noise = np.mean(np.abs(tc[:frame_noise]))
        signal = np.mean(np.abs(tc[frame_signal:]))
    elif mode == 'peak':
        noise = np.std(tc[:frame_noise])
        signal = np.max(tc[frame_signal:])
    if no_noise:
        return signal
    if noise == 0:
        return 0
    else:
        return signal / noise

def rotate_sta(S, primary_colors, copy=True):
    if copy:
        S_ = S.copy()
    else:
        S_ = S
    S_rot = np.zeros_like(S_)
    for i in range(S.shape[3]):
        S_rot[...,i] = np.tensordot(S, primary_colors[i,:], (3,0))
    return S_rot

def calculate_sta_tc(S, sig_stixels):
    N_select = 3
    tcs_3 = np.zeros([S.shape[2], 3])
    tcs_6 = np.zeros([S.shape[2], 6])

    sig_stixels_pca = ndimage.binary_dilation(sig_stixels, iterations=15)
    signal_frame = int(S.shape[2] / 2)

    for coli in [0,1,2]:
        S_ = S.copy()
        S_ = S_[sig_stixels_pca, signal_frame:, coli]
        eigenvect = PCA(n_components=1).fit(S_).components_[0]
        eigenvect *= find_tc_polarity(eigenvect)

        # project STA on evector
        rf_map = np.tensordot(S[..., signal_frame:, (coli,)], eigenvect[:, np.newaxis], ((2, 3), (0, 1)))

        rf_map_flat = rf_map.flatten()
        sorted = np.argsort(rf_map_flat)

        proj_top = sorted[-N_select:]
        proj_top = np.unravel_index(proj_top, rf_map.shape)
        a = [S[proj_top[0][i], proj_top[1][i], :, coli] for i in range(N_select)]
        tc_ON = np.mean(a,0)

        proj_bottom = sorted[:N_select]
        proj_bottom = np.unravel_index(proj_bottom, rf_map.shape)
        a = [S[proj_bottom[0][i], proj_bottom[1][i], :, coli] for i in range(N_select)]
        tc_OFF = np.mean(a,0)

        tcs_6[:, coli * 2] = tc_ON
        tcs_6[:, coli * 2 + 1] = tc_OFF

        if np.max(np.abs(tc_ON)) > np.max(np.abs(tc_OFF)):
            tcs_3[:, coli] = tc_ON
        else:
            tcs_3[:, coli] = tc_OFF * -1

    return tcs_3, tcs_6

def find_tc_polarity(tc):
    extrema = np.abs([np.min(tc), np.max(tc)])
    positions = [np.argmin(tc), np.argmax(tc)]
    if positions[0] > positions[1]:
        pol = -1
    else:
        pol = 1
    if extrema[1] > 2 * extrema[0]:
        pol = 1
    if extrema[0] > 2 * extrema[1]:
        pol = -1
    return pol

def calculate_sta_tc_dev(S, sig_stixels):
    tcs_3 = np.zeros([S.shape[2], 3])
    tcs_6_a = np.zeros([S.shape[2], 6])
    tcs_6_b = np.zeros([S.shape[2], 6])
    tcs_6_c = np.zeros([S.shape[2], 6])

    signal_frame = int(S.shape[2] / 2)
    sig_stixels_pca = ndimage.binary_dilation(sig_stixels, iterations=15)

    for coli in [1,2]:
        S_ = S.copy()
        S_ = S_[sig_stixels_pca, signal_frame:, coli]
        eigenvect = PCA(n_components=1).fit(S_).components_[0]
        eigenvect *= find_tc_polarity(eigenvect)

        # project STA on evector
        rf_map = np.tensordot(S[..., signal_frame:, (coli,)], eigenvect[:, np.newaxis], ((2, 3), (0, 1)))

        for poli in [0,1]:
            pol = [1,-1][poli]
            snr_map = rf_map * pol

            snr_threshold = 0.5
            N_select = 3
            best_stix = np.unravel_index(np.flip(np.argsort(snr_map.flatten())), snr_map.shape)
            tcs = []
            snr_cum = []
            tcs_above = []
            snr_start = 0
            scale_base = 0
            for i in range(1500):
                tc = S[best_stix[0][i], best_stix[1][i], :, coli]
                scale = np.max(tc * pol)
                if i == 0:
                    scale_base = scale
                if scale == 0:
                    continue
                tc *= scale_base
                if i >= 4: #cumulative SNR
                    if i % 3 == 0:
                        try:
                            snr = tc_snr(np.mean(tcs, axis=0), 'peak')
                        except:
                            snr = np.nan
                            print('****snr calc failed\n*****\n*****\n*****')
                        snr_cum.append(snr)
                    else:
                        snr_cum.append(np.nan)
                else:
                    snr_cum.append(np.nan)

                # local SNR threshold
                snr_this = snr_map[best_stix[0][i], best_stix[1][i]]
                if i == 0:
                    snr_start = snr_this
                tcs.append(tc)
                if snr_this > snr_threshold * snr_start:
                    tcs_above.append(tc)
                # else:
                #     break

            stop_index = np.nanargmax(snr_cum[20:]) + 20
            tc_combined_c = np.mean(tcs[:stop_index], axis=0)
            # print(len(tcs_above))
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                tc_combined_b = np.mean(tcs_above, axis=0)
                tc_combined_a = np.mean(tcs[:N_select], axis=0)

            ind = coli * 2 + poli
            tcs_6_a[:, ind] = tc_combined_a
            tcs_6_b[:, ind] = tc_combined_b
            tcs_6_c[:, ind] = tc_combined_c

        if np.max(np.abs(tcs_6_a[:, coli * 2])) > np.max(np.abs(tcs_6_a[:, coli * 2 + 1])):
            tcs_3[:, coli] = tcs_6_a[:, coli * 2]
        else:
            tcs_3[:, coli] = tcs_6_a[:, coli * 2 + 1]

    return tcs_3, tcs_6_a, tcs_6_b, tcs_6_c


def calculate_sta_tc_dev2(S, sig_stixels, snr_threshold=0.5):
    tcs_3 = np.zeros([S.shape[2], 3])
    tcs_6 = np.zeros([S.shape[2], 6])

    signal_frame = int(S.shape[2] * .3)
    noise_frame = int(S.shape[2] * .2)
    dim = np.mean(S.shape[:1])
    sig_stixels_pca = ndimage.binary_dilation(sig_stixels, iterations=int(0.2 * dim))
    sig_stixels_boundary = ndimage.binary_dilation(sig_stixels, iterations=int(0.1 * dim))

    for coli in [0,1,2]:
        S_pca = S[sig_stixels_pca, signal_frame:, coli]
        eigenvect = PCA(n_components=1).fit(S_pca).components_[0]
        eigenvect *= find_tc_polarity(eigenvect)

        # project STA on evector
        rf_map = np.tensordot(S[..., signal_frame:, (coli,)], eigenvect[:, np.newaxis], ((2, 3), (0, 1)))

        for poli in [0,1]:
            pol = [1,-1][poli]
            # print('this is ', coli, pol)
            snr_map = rf_map * pol * sig_stixels_boundary
            best_stixel = np.unravel_index(np.argmax(snr_map.flatten()), snr_map.shape)
            best_tc = S[best_stixel[0], best_stixel[1], :, coli]
            best_peak_abs = np.max(pol * best_tc) # working with un-inverted TCs
            sel_stix = snr_map > snr_threshold * np.max(snr_map)
            good_tc = S[sel_stix, :, coli]
            scales = np.max(np.abs(good_tc), axis=1)
            good_tc = good_tc[scales != 0, :]

            if good_tc.shape[0] > 0:
                scales = np.max(abs(good_tc), axis=1)
                # with warnings.catch_warnings():
                #     warnings.simplefilter("ignore", category=RuntimeWarning)
                good_tc = good_tc / scales[:, np.newaxis] # each input TC has max 1 now
                tc_combined = np.mean(good_tc, axis=0) # now rescale it to match the first input TC
                tc_combined /= np.max(np.abs(tc_combined))
                tc_combined *= best_peak_abs
            else:
                tc_combined = np.zeros(good_tc.shape[1])

            # look for flat line DC offset TCs
            early_level = np.mean(tc_combined[:noise_frame])
            if np.abs(early_level) > 0:
                peak = np.max(np.abs(tc_combined))
                ratio = np.abs(peak / early_level)
                if ratio < 5:
                    tc_combined -= early_level

            ind = coli * 2 + poli
            tcs_6[:, ind] = tc_combined

        if np.max(np.abs(tcs_6[:, coli * 2])) > np.max(np.abs(tcs_6[:, coli * 2 + 1])):
            # ON dominant
            tcs_3[:, coli] = tcs_6[:, coli * 2]
        else:
            # OFF dominant (invert tc)
            tcs_3[:, coli] = -1 * tcs_6[:, coli * 2 + 1]

    return tcs_3, tcs_6

# def calculate_sta_pca(S, stixel_mask=None, num_components=1):
#     if stixel_mask is None:
#         stixel_mask = calculate_sig_stixels(S)[0]
#
#     components = np.zeros([num_components, S.shape[2], S.shape[3]])
#
#     if np.sum(stixel_mask) == 0:
#         print('no stixels in mask')
#         return components
#
#     S_to_fit = S[stixel_mask]
#     S_to_fit = S_to_fit.reshape(S_to_fit.shape[0], S_to_fit.shape[1] * S_to_fit.shape[2])
#     masked_time_course = np.mean(S[stixel_mask,:,:], 0)
#
#     # do PCA if more than 1 sigstix
#     if S_to_fit.shape[0] > 1:
#         pca = PCA(n_components=num_components)
#         pca.fit(S_to_fit)
#         # pca_result = pca.transform(S_to_fit)
#
#         # reshape eigenvectors into time courses
#         for compi in range(num_components):
#             eigenvect = pca.components_[compi,:].reshape(S.shape[2], S.shape[3])
#             components[compi] = eigenvect
#
#         # PCA first component can be negative of expected
#         # PCA doesn't care cause eigenvalues can be negative, but we care about cell polarity
#         # so check if it looks flipped and flip them all to match
#         if np.tensordot(masked_time_course, components[0]) < 0:
#             components *= -1
#
#     elif S_to_fit.shape[0] == 1:
#         components[0] = masked_time_course
#     else:
#         print('shouldnt be here')
#
#     return components

def find_contours_and_scale(rf_map, thresh, stixel_size):
    """
    Fairly fast generation of contours for RF display use
    Args:
        rf_map: X x Y map of RF (floats)
        thresh: level at which to draw the contours
        stixel_size: size of stixels for contour rescaling, in µm

    Returns:
        segs_out, the scaled contours for the RF

    """
    segs = measure.find_contours(rf_map, level=thresh)
    segs_out = []
    for seg in segs:  # shift segs to correct space
        # seg[:, 0] -= 1
        # seg[:, 1] += 0
        seg_scaled = seg * (stixel_size)
        # seg_scaled[:, 0] *= -1
        # seg_scaled[:, 0] += np.max(stixel_size * rf_map.shape[0])
        segs_out.append(seg_scaled)
    return segs_out


# make a gaussian blob map located at the sig stixels to reject distant noise.
#  - Ranges from 0 at the sigstix + surround to -4 * noise_level at baseline
#  - a basic implementation of the neural dendritic continuity assumption.
#  - Will cause us to miss cells with very widely dispersed weak RF spots
def make_boost_map(sig_stixels, noise_level, stixel_size):
    sig_stixels_blurred = gaussian_filter(sig_stixels * 1.0, 500 / stixel_size) # TODO: normalize dimension
    sig_stixels_blurred -= np.min(sig_stixels_blurred.flatten())
    sig_stixels_blurred /= np.max(sig_stixels_blurred.flatten())
    sig_stixels_blurred += -.6
    sig_stixels_blurred = np.clip(sig_stixels_blurred, -np.inf, 0)
    sig_stixels_blurred *= noise_level * .5
    return sig_stixels_blurred

def large_region_rf_boost(rf_map, good_threshold, stixel_size):
    # make a boolean map of just the large dtabs of RF
    rf_map_bool = rf_map >= good_threshold
    rf_map_central = rf_map_bool.copy()
    # find large regions
    rf_labels = measure.label(rf_map_bool)
    # np.set_printoptions(threshold=np.inf)
    # print(rf_labels)
    regionprops = pd.DataFrame(measure.regionprops_table(rf_labels, properties=['label', 'area']))
    # zero out small regions
    if np.max(regionprops.area) > 1:
        for r in regionprops.itertuples():
            if r.area == 1:
                rf_map_central[rf_labels == r.label] = 0

    # use big dtabs map to make our central region, and lower everything peripheral
    sig_stixels_blurred = make_boost_map(rf_map_central, 1, stixel_size)
    rf_map_boost = rf_map + sig_stixels_blurred * np.max(rf_map)
    # rf_map_boost = np.clip(rf_map_boost, 0, 1)
    return rf_map_boost

def calculate_rf_threshold(rf_map, thresholds, stixel_size):

    # boolean morphology, no-contours, analysis over threshold
    areas = []
    island_counts = []
    segments = [] # not actually used?
    max_island_areas = []
    position_variances = []

    x = np.arange(rf_map.shape[0])
    y = np.arange(rf_map.shape[1])

    for ti, thresh in enumerate(thresholds):
        map_bool = rf_map >= thresh
        # map_bool_dilated = ndimage.morphology.binary_dilation(map_bool)
        area = np.count_nonzero(map_bool)

        if area > 0:
            segs = []

            map_labeled, island_count = ndimage.label(map_bool)
            # regions = [map_labeled == i for i in range(island_count)]
            # rf_props_table = measure.regionprops_table(map_labeled[0], intensity_image=rf_map, properties=['label','area','centroid'])
            # max_island_area = np.max([np.count_nonzero(regions[i]) for i in range(island_count)])

            # if island_count > 4: # only need this for noise reduction
            #     centers = np.zeros([island_count, 2])
            #     for i in range(island_count):
            #         xrange_sig = x[np.any(regions[i], 1)]
            #         yrange_sig = y[np.any(regions[i], 0)]
            #         centers[i,:] = (np.median(xrange_sig), np.median(yrange_sig))
            #     position_var = np.mean([np.var(centers[:,0]), np.var(centers[:,1])])
            # else:
            #     position_var = 0
            # ic(max_island_area)
        else:
            segs = []
            island_count = 0
            max_island_area = 0
            position_var = 0
        areas.append(area)
        island_counts.append(island_count)
        # segments.append(segs)
        # position_variances.append(position_var)
        # max_island_areas.append(max_island_area)
    areas = np.array(areas)
    island_counts = np.array(island_counts)
    position_variances = np.array(position_variances)
    max_island_areas = np.array(max_island_areas)
    mean_island_area = np.zeros_like(thresholds)
    mean_island_area[island_counts > 0] = areas[island_counts > 0] / island_counts[island_counts > 0]
    # mean_island_area[np.isnan(mean_island_area)] = 0

    # filter threshold options to avoid low thresh noisey RFs at the low mean RF size:
    decreasing = np.diff(mean_island_area) <= 0
    decreasing = np.concatenate([[True], decreasing])
    start_index = np.argmin(decreasing)
    # ic(decreasing, start_index)
    # start_index = 0

    #                 conv_hull = morphology.convex_hull_image(map_bool)
    #         fill = np.count_nonzero(np.logical_and(map_bool, conv_hull)) / np.count_nonzero(conv_hull)


    # make the heuristic
    h = mean_island_area
    h = savgol_filter(h, 5, 3, mode='nearest')

    # show a subset of thresholds as contours
    # print(mean_island_area)
    # mean_island_area -= thresholds * 1
    best_index = np.argmax(h[start_index:]) + start_index
    # ic(best_index, mean_island_area)
    if best_index > 0:
        good_rf = True

        # check whether the RF is too noisy
        # calculate position variance
        if island_counts[best_index] > 2:
            map_bool = rf_map >= thresholds[best_index]
            map_labeled, island_count = ndimage.label(map_bool)
            regions = [map_labeled == i for i in range(island_count)]
            centers = np.zeros([island_count, 2])
            for i in range(island_count):
                xrange_sig = x[np.any(regions[i], 1)]
                yrange_sig = y[np.any(regions[i], 0)]
                centers[i,:] = (np.median(xrange_sig), np.median(yrange_sig))
            position_var = np.mean([np.var(centers[:,0]), np.var(centers[:,1])])
        else:
            position_var = 0
        position_var

        # print(mean_island_area[best_index], island_counts[best_index], position_var)

        h_noise = np.count_nonzero([island_counts[best_index] > 6, mean_island_area[best_index] < 2, position_var > 100])
        # print(h_noise)
        if h_noise >= 2:
        #     # best_index += 3
            good_rf = False
            # print('rejected for noisy')
    else:
        good_rf = False

    return h, areas, segments, island_counts, mean_island_area, max_island_areas, position_variances, start_index, best_index, good_rf



class Feature_sta_vision_fit(Feature):
    name = 'sta vision fit'
    requires = {'unit_id'}
    provides = {'unit':{'sta_vision_fit'}}
    input = {'analysis_data'}

    def generate(self, ct, unit_indices, inpt, dtab=None):
        if dtab is None:
            dtab = ct.unit_table
        di = unit_indices[0][0:2]
        analysis_data = inpt['analysis_data']
        fits = []
        for ci in unit_indices:
            fit = analysis_data.get_stafit_for_cell(dtab.at[ci,'unit_id'])
            # params = ['center_x', 'center_y', 'std_x', 'std_y', 'rot']
            fits.append(file_handling.wrapper(np.array(list(fit))))
        dtab.loc[unit_indices, 'sta_vision_fit'] = fits

        self.update_valid_columns(ct, di)


def params_from_metadata(dataset):
    noise = dataset['params_wn'].a

    if dataset['display'] == 'OLED':
        display_width_pixels = 800
        display_rate = 60.0

        base_mpp = 1.8
        base_lens = 6.5

    else: # CRT, but could be NaN or "missing" or "CRT"
        display_width_pixels = 640
        display_rate = 120.0

        # if dataset['optics'] in ['below','bottom']:
        base_mpp = 5.5
        base_lens = 6.5

        # elif dataset['optics'] == 'top':
        #     base_mpp = None
        #     base_lens = None

    try:
        lens = float(dataset['lens'])
        assert(not np.isnan(lens))
    except:
        print(f'... Lens "{dataset["lens"]}" failed to parse, assuming objective 6.5')
        lens = 6.5

    lens_multiplier = 1
    if np.abs(lens - 6.5) < 0.1:
        lens_multiplier = 1
    elif np.abs(lens - 4) < 0.1:
        lens_multiplier = 1.45
    elif np.abs(lens - 2) < 0.1:
        lens_multiplier = 4.59

    print(f'Lens {lens} mult {lens_multiplier}')

    try:
        microns_per_pixel = lens_multiplier * base_mpp
        pixels_per_stixel = float(noise['stixel width'])
    except:
        print(f'Failed stixel width {noise["stixel width"]}')
        print(dataset)


    stixel_size = microns_per_pixel * pixels_per_stixel
    # print(microns_per_pixel, pixels_per_stixel, dataset)
    assert(not np.isnan(stixel_size))
    print(noise)
    interval = float(noise['interval'])
    frame_time = interval / display_rate

    assert(not np.isnan(frame_time))
    params = {'stixel_size': stixel_size, 'frame_time': frame_time, 'stimulus': 'whitenoise'}
    return params


def load_vision_sta(analysis_data, cell_id):

    sta = analysis_data.get_sta_for_cell(cell_id)

    # reshape sta array: X, Y, time, color
    S = np.array([sta.red, sta.green, sta.blue])
    S = S.swapaxes(0, 2)
    S = S.swapaxes(2, 3)
    S = np.fliplr(S) # verified to match vision view

    S_var = np.array([sta.red_error, sta.green_error, sta.blue_error])
    S_var = S_var.swapaxes(0, 2)
    S_var = S_var.swapaxes(2, 3)

    num_frames = S.shape[2]
    frame_range = np.array((0, S.shape[2] - 1))
    frames = np.arange(frame_range[0], frame_range[1] + 1)
    time = -1 * (num_frames - frames - 1) * sta.refresh_time / 1000

    stixel_size = sta.stixel_size

    x = np.arange(S.shape[0]) * stixel_size
    y = np.arange(S.shape[1]) * stixel_size

    return x, y, time, S, S_var

    # I think this has some issue with x and y lengths
    # if not interp_factor == 1:
    #     # interp = interpolate.RectBivariateSpline(range(0, S.shape[0]))
    #     S_upsampled = np.empty(((S.shape[0]-1) * interp_factor, (S.shape[1]-1) * interp_factor, S.shape[2], S.shape[3]))
    #     for ci in range(S.shape[3]):
    #         for ti in range(S.shape[2]):
    #             image = S[:,:,ti,ci]
    #
    #             x = np.arange(S.shape[0])
    #             y = np.arange(S.shape[1])
    #             # print(image.shape)
    #             # pretty(x)
    #
    #             # maybe replace with RectBivariateSpline
    #             image_interp = interpolate.interp2d(x, y, image.T, 'linear', fill_value=np.nan)
    #             # image_interp = interpolate.RectBivariateSpline(x, y, image)
    #             image_upsampled = image_interp(new_x, new_y)
    #
    #             # pretty(image_upsampled)
    #
    #             S_upsampled[:,:,ti,ci] = image_upsampled.T
    #
    #     S = S_upsampled
    # x = new_x
    # y = new_y


class Feature_wn_params(Feature):
    name = 'white noise stimulus parameters'
    requires = {'unit':{'unit_id'}}
    provides = {'dataset':{'stimulus_params'} # a dict with stimulus ('white noise'), stixel_size (µm), and frame_time (s)
                }

    def generate(self, ct, unit_indices, inpt, dtab=None):
        if dtab is None:
            dtab = ct.unit_table
        di = unit_indices[0][0:2]
        if (missing := self.check_requirements(ct, di)) is not None:
            print('Feature {}: missing requirements {}'.format(self.name, missing))
            return

        dataset = ct.dataset_table.loc[unit_indices[0][0:2]]

        if di[1] == 'com':
            print('... loading combined STA')
            stixel_size = 44
            params = {'frame_time': 1 / 120.0, 'stixel_size': stixel_size, 'stimulus': 'whitenoise'}
        else:
            if 'params_wn' in dataset['valid_columns_dataset'].a:
                params = params_from_metadata(dataset)
                print('... making WN params from metadata (from database table)')
            else:
                print('... making WN params from out of thin air (must not be in database table)')
                params = {'frame_time': 1 / 120.0, 'stixel_size': 44, 'stimulus': 'whitenoise'}

            if 'wu_sta' in dataset['sta_path']:
                params['frame_time'] = 1 / 120.0  # simple override
        print('... stimulus params are: \n', params)

        ct.dataset_table.at[di, 'stimulus_params'] = file_handling.wrapper(params)

        self.update_valid_columns(ct, di)

class Feature_load_sta(Feature):
    """
    Load STA from vision .sta files
    """
    name = 'load STA'
    requires = {'unit':{'unit_id'}}
    provides = {'unit':{'sta'}} # array of X x Y x T x C}}
    input = {'analysis_data','sta'}

    num_early_frames = 9  # number of frames for noise calculation

    def generate(self, ct, unit_indices, inpt, dtab=None):
        if dtab is None:
            dtab = ct.unit_table
        di = unit_indices[0][0:2]
        if (missing := self.check_requirements(ct, di)) is not None:
            print('Feature {}: missing requirements {}'.format(self.name, missing))
            return

        stas = []
        pars = []

        dataset = ct.dataset_table.loc[unit_indices[0][0:2]]
        # print(f'Loading dataset from {dataset["sta_path"]}')
        if len(dataset['sta_path']) > 0:
            print(f'... Loading single-frame STA from {dataset["sta_path"]}')
            import h5py

            mode = 'single_frame'
            sta_file = h5py.File(dataset["sta_path"],'r')
        elif dataset['stimulus_type'] == 'whitenoise':
            mode = 'vision'
        else:
            print('...not white noise stimulus, skipping STA load')
            return

        print(f'Using STA mode {mode}')

        for ci in unit_indices:
            if mode == 'vision':
                try:
                    x_l, y_l, time, S, S_var = load_vision_sta(inpt['analysis_data'], dtab.at[ci, 'unit_id'])
                    if ci == unit_indices[0]:
                        print('... STA Vision dimensions {} x {} stixels, {} time, {} color'.format(S.shape[0], S.shape[1], S.shape[2],
                                                                                             S.shape[3]))
                except:
                    print(f'error missing vision STA for unit_id/alex_id {ci}, set invalid')
                    dtab.loc[ci, 'valid'] = False
                    stas.append(np.nan)
                    pars.append(np.nan)
                    continue

            if mode == 'single_frame':
                try:
                    S_in = sta_file[str(dtab.at[ci, 'unit_id'])]
                except:
                    print(f'error missing STA for unit_id/alex_id {ci} {dtab.at[ci, "unit_id"]}, label {dtab.at[ci, "label_manual_text_input"]}')
                    dtab.loc[ci, 'valid'] = False
                    stas.append(np.nan)
                    pars.append(np.nan)
                    continue

                S = np.zeros(S_in.shape, dtype='float32')
                S_in.read_direct(S)
                if S.shape[3] == 3: # hacky code
                    S = S.swapaxes(0, 2)
                else:
                    S = np.moveaxis(S, [0,1,2,3], [2,3,0,1])
                assert(S.shape[3] <= 3) # check that we got the colors at the end

            if ci == unit_indices[0]:
                if mode == 'single_frame':
                    print('... STA single-frame dimensions {} x {} stixels, {} time, {} color'.format(S.shape[0], S.shape[1], S.shape[2], S.shape[3]))

            if np.all(np.isnan(S.flat)):
                print(f'... unit {ci} has all nan STA, marking invalid')
                dtab.at[ci, 'valid'] = False

            stas.append(file_handling.wrapper(S))
            # dtab.at[ci, 'mv_sta_var'] = file_handling.wrapper(S_var)
            # params = {'stixel_size': y_l[1] - y_l[0], 'frame_time': np.abs(time[1] - time[0]), 'stimulus':'whitenoise'}

        dtab.loc[unit_indices, 'sta'] = stas

        if mode == 'single_frame':
            sta_file.close()

        self.update_valid_columns(ct, di)


class Feature_sta_basic(Feature):
    """
    generate basic features from STA
    """
    name = 'STA basic properties'
    requires = {'unit':{'sta'}}
    provides = {'unit':{'map_sta_peak',
                        'sta_noise_level',
                        'sta_signal_level',
                        'sta_snr',
                        'sta_extremes'}
                }
    input = set()

    num_early_frames = 9  # number of frames for noise calculation

    def generate(self, ct, unit_indices, inpt, dtab=None):
        if dtab is None:
            dtab = ct.unit_table
        di = unit_indices[0][0:2]
        if (missing := self.check_requirements(ct, di)) is not None:
            print('Feature {}: missing requirements {}'.format(self.name, missing))
            return

        noises = []
        signals = []
        snrs = []
        extremes = []
        peaks = []
        ind_invalid = []

        for ci in unit_indices:
            S = dtab.at[ci, 'sta'].a

            early_frames = np.mean(S[..., 0:self.num_early_frames, :], 3)
            noises.append(stats.median_abs_deviation(early_frames.flatten(), scale='normal'))
            signals.append(np.max(np.abs(S).flatten()))
            if noises[-1] == 0:
                snrs.append(0)
                ind_invalid.append(ci)
            else:
                snrs.append(signals[-1] / noises[-1])

            ext = np.array([np.max(S[..., 0]), np.min(S[..., 0]), np.max(S[..., 1]), np.min(S[..., 1]), np.max(S[..., 2]),
                   np.min(S[..., 2])])
            extremes.append(file_handling.wrapper(ext))

            peak_frame_index = np.argmax([np.max(np.abs(S[..., fi, :])) for fi in range(S.shape[2] - 1)])
            peak_frames = np.clip(peak_frame_index + np.array([-1, 0, 1]), 0, S.shape[2] - 1)
            peak_frame_composite = np.mean(S[:, :, peak_frames, :], 2)
            peaks.append(file_handling.wrapper(peak_frame_composite))

        dtab.loc[unit_indices, 'sta_signal_level'] = signals
        dtab.loc[unit_indices, 'sta_noise_level'] = noises
        dtab.loc[unit_indices, 'sta_snr'] = snrs
        dtab.loc[unit_indices, 'sta_extremes'] = extremes
        dtab.loc[unit_indices, 'map_sta_peak'] = peaks
        if len(ind_invalid) > 0:
            dtab.loc[ind_invalid, 'valid'] = False
            print(f'Marking {len(ind_invalid)} units invalid due to blank STA')

        self.update_valid_columns(ct, di)


class Feature_sta_fit(Feature):
    name = 'STA fit'
    requires = {'unit':{'sta'}}
    provides = {'unit':{'par_fit_sta','sc_fit_sta_metric'}}

    def generate(self, ct, unit_indices, inpt, dtab=None):
        if dtab is None:
            dtab = ct.unit_table
        di = unit_indices[0][0:2]
        if (missing := self.check_requirements(ct, di)) is not None:
            print('Feature {}. missing requirements {}'.format(self.name, missing))
            return

        for ci in unit_indices:
            print('Fitting STA for cell {}'.format(ci))
            try:
                stimulus_params = ct.dataset_table.at[ci[0:2], 'stimulus_params'].a
                x_l, y_l, X, Y, time_l = cdl.make_sta_dimensions(dtab.at[ci, 'sta'].a, stimulus_params)
                fit_params_stc, fit_stc, fit_metric = cdl.sta_fit(x_l, y_l, time_l, dtab.at[ci, 'sta'].a, False)
                params = cdl.name_fit_params(fit_params_stc)
                dtab.at[ci, 'par_fit_sta'] = file_handling.wrapper(fit_params_stc)
                dtab.at[ci, 'sc_fit_sta_metric'] = fit_metric
            except:
                print('error processing cell fit')
                pass
        self.update_valid_columns(ct, di)

class Feature_sigstix(Feature):
    name = 'sigstix'
    requires = {'unit':{'sta'}}
    provides = {'unit':{'map_sig_stixels'}}

    def generate(self, ct, unit_indices, inpt, dtab=None):
        if dtab is None:
            dtab = ct.unit_table
        di = unit_indices[0][0:2]
        if (missing := self.check_requirements(ct, di)) is not None:
            print('Feature {}. missing requirements {}'.format(self.name, missing))
            return

        sigs = []
        for ci in tqdm(unit_indices, ncols=60):
            S = dtab.at[ci, 'sta'].a

            sig_stixels = calculate_sig_stixels_simple(S, color_channels=(1,2))[0]
            sigs.append(file_handling.wrapper(sig_stixels.astype(bool)))
        dtab.loc[unit_indices, 'map_sig_stixels'] = sigs
        self.update_valid_columns(ct, di)



class Feature_center_of_mass(Feature):
    name = 'RF center of mass'
    requires = {'unit':{'map_sig_stixels'}}
    provides = {'unit':{'rf_center_x', 'rf_center_y'}}

    def generate(self, ct, unit_indices, inpt, dtab=None):
        if dtab is None:
            dtab = ct.unit_table
        di = unit_indices[0][0:2]
        if (missing := self.check_requirements(ct, di)) is not None:
            print('Feature {}. missing requirements {}'.format(self.name, missing))
            return

        stixel_size = ct.dataset_table.at[di, 'stimulus_params'].a['stixel_size']

        center_x = []
        center_y = []
        for ci in unit_indices:
            sig_stix = dtab.at[ci, 'map_sig_stixels'].a

            x = np.arange(sig_stix.shape[0]) * stixel_size
            y = np.arange(sig_stix.shape[1]) * stixel_size

            xrange_sig = x[np.any(sig_stix, 1)]
            yrange_sig = y[np.any(sig_stix, 0)]

            xcenter = np.median(xrange_sig)
            ycenter = np.median(yrange_sig)

            center_x.append(xcenter)
            center_y.append(ycenter)
        dtab.loc[unit_indices, 'rf_center_x'] = center_x
        dtab.loc[unit_indices, 'rf_center_y'] = center_y
        self.update_valid_columns(ct, di)


class Feature_primary_colors(Feature):
    """
    find the wavelength sensitivity space (color) vectors of max explained variation via PCA
    can be used to convert S to S_rot, which has rotated dimensions to probably match R: +L-M, G: +M, B: +S
    """
    name = 'primary colors'
    requires = {'unit':{'map_sig_stixels'}}
    provides = {'dataset':{'primary_colors'}}

    def generate(self, ct, unit_indices, inpt, dtab=None):
        di = unit_indices[0][0:2]
        if (missing := self.check_requirements(ct, di)) is not None:
            print('Feature {}. missing requirements {}'.format(self.name, missing))
            return

        if dtab is None:
            dtab = ct.unit_table
        # bw_mode = np.all(dtab.loc[unit_indices, 'sta_bw'])
        # if bw_mode:
        #     print('All cells are BW, using BW mode')
        #     colors = np.array([[0,0,0],[1,1,1],[0,0,0]])
        # else:
        tc_all = []
        all_bw_mode = False
        for ci in unit_indices:
            if dtab.at[ci, 'sta_bw']:
                continue
            S = dtab.at[ci, 'sta'].a
            sig_stixels = dtab.at[ci, 'map_sig_stixels'].a
            tcs = S[sig_stixels, :, :]
            if np.any(np.isnan(tcs)):
                continue
            # for si in range(tcs.shape[0]):
            tc_all.extend(tcs)
        if len(tc_all) > 0:
            tc_all = np.array(tc_all)
            tc_all = tc_all.reshape([tc_all.shape[0] * tc_all.shape[1], tc_all.shape[2]])
        # tc_all[np.any(np.isnan(tc_all), axis=1),:] = [0,0,0]

            num_components = 3
            pca = PCA(n_components=num_components)
            pca.fit(tc_all)
            # pca_result = pca.transform(tc_all)

            colors = pca.components_
            for i in range(num_components):
                color = colors[i,:]
                # make colors primarily positive
                if np.max(np.abs(color)) > np.max(color):
                    color *= -1
                colors[i,:] = color

            # print(colors)
            # order = np.argmax(colors, axis=0)
            # print(order)
            order = (2,0,1)
            # assert(order[0] == 2 and order[1] == 0)
            colors = colors[order, :]
            for cc in range(3):
                if colors[cc,cc] < 0:
                    colors[cc, :] *= -1
            # colors = colors[(2,0,1),:] # needs to be automated, probably some datasets will swap blue and red
            # colors = colors[(1, 0, 2), :]  # for export2 aka human1
        else: # all BW run
            print('All BW run, using simple colors')
            colors = np.array([[1,0,0],[0,1,0],[0,0,1]])

        print('Primary colors:')
        print(colors)

        ct.dataset_table.at[unit_indices[0][0:2], 'primary_colors'] = file_handling.wrapper(colors.copy())
        # for ci in unit_indices:
        #     dtab.at[ci, 'primary_colors'] = file_handling.wrapper(colors)
        self.update_valid_columns(ct, di)


class Feature_timecourses(Feature):
    name = 'timecourses'
    requires = {'unit':{'sta','map_sig_stixels'}, 'dataset':{'primary_colors'}}
    provides = {'unit':{'tc','tc_all'}}#,'tc_all_b','tc_all_c'}}
    snr_threshold = 0.5

    def generate(self, ct, unit_indices, inpt, dtab=None):
        if dtab is None:
            dtab = ct.unit_table
        di = unit_indices[0][0:2]
        if (missing := self.check_requirements(ct, di)) is not None:
            print('Feature {}. missing requirements {}'.format(self.name, missing))
            return

        print('... timecourse mode: direct from STA (via PCA map), rotated')
        print(f'... Using SNR_threshold={self.snr_threshold:.2f}')

        primary_colors = ct.dataset_table.at[di, 'primary_colors'].a

        tc = []
        tc_all_a = []
        # tc_all_b = []
        # tc_all_c = []
        for ci in tqdm(unit_indices, total=len(unit_indices), ncols=60):
            # print(ci)
            S = dtab.at[ci, 'sta'].a
            S_rot = rotate_sta(S, primary_colors)

            sig_stixels = dtab.at[ci, 'map_sig_stixels'].a

            # tcs_3, tcs_6 = calculate_sta_tc(S_rot, sig_stixels)
            # tcs_3, tcs_6_a, tcs_6_b, tcs_6_c = calculate_sta_tc_dev(S_rot, sig_stixels)
            tcs_3, tcs_6_a = calculate_sta_tc_dev2(S_rot, sig_stixels, self.snr_threshold)

            tc.append(file_handling.wrapper(tcs_3))
            tc_all_a.append(file_handling.wrapper(tcs_6_a))
            # tc_all_b.append(file_handling.wrapper(tcs_6_b))
            # tc_all_c.append(file_handling.wrapper(tcs_6_c))
        dtab.loc[unit_indices, 'tc'] = tc
        dtab.loc[unit_indices, 'tc_all'] = tc_all_a
        # dtab.loc[unit_indices, 'tc_all_b'] = tc_all_b
        # dtab.loc[unit_indices, 'tc_all_c'] = tc_all_c
        self.update_valid_columns(ct, di)


# process STA to make RF maps for each color, ON and OFF
# also extracts timecourses (mean of each RF tc) and single sigstix map of bools
# color_names = ('+L-M', '+M', '+S')
color_names = ('red ON (pc2)','red OFF (pc2)','green ON (pc0)', 'green OFF (pc0)', 'blue ON (pc1)', 'blue OFF (pc1)')



class Feature_rf_histogram(Feature):
    name = 'rf histogram'
    requires = {'unit':{'sta', 'tc'},'dataset': {'primary_colors'}}
    provides = {'unit':{'rf_projection_histograms'},'dataset':{'rf_projection_bins'}}

    # rotate_colors = True
    # use_rf_boost = True
    # noise_multipliers = np.logspace(np.log10(1.5), np.log10(10), 10)
    projection_bins = np.linspace(-1, 1, 12)
    noise_floor = 3 # ignore STA less than this times noise level

    def generate(self, ct, unit_indices, inpt, dtab=None):
        if dtab is None:
            dtab = ct.unit_table
        di = unit_indices[0][0:2]
        if (missing := self.check_requirements(ct, di)) is not None:
            print('Feature {}. missing requirements {}'.format(self.name, missing))
            return

        ct.dataset_table.at[di, 'rf_projection_bins'] = file_handling.wrapper(self.projection_bins)

        stixel_size = ct.dataset_table.at[di, 'stimulus_params'].a['stixel_size']
        hist_list = []
        for cc, ci in enumerate(unit_indices):
            # cdl.show(dtab.at[ci, 'map_sta_peak'].a)
            # plt.show()
            S = dtab.at[ci, 'sta'].a
            # sig_stixels = dtab.at[ci, 'map_sig_stixels'].a
            tc = dtab.at[ci, 'tc'].a
            # scale = np.max(np.abs(S))
            noise_level = dtab.at[ci, 'sta_noise_level']
            if noise_level < 0.000000001:
                noise_level = 1

            hists = np.zeros([3, len(self.projection_bins)-1])
            scale = 0

            for coli in range(3):
                tc_coli = tc[:, coli, np.newaxis]
                rf_map = np.tensordot(S[..., (coli,)], tc_coli, ((2, 3), (0, 1)))
                scale = np.max([scale, np.max(np.abs(rf_map))])
            # print(scale)
            for coli in range(3):
                # rf_map = np.tensordot(S[...,(coli,)], tc[:, coli], ((2,3), (0,1)))
                tc_coli = tc[:, coli, np.newaxis]
                # tc_coli /= np.max(np.abs(tc_coli))
                # ic(tc_coli.shape)
                rf_map = np.tensordot(S[..., (coli,)], tc_coli, ((2, 3), (0, 1))) / scale

                noise_level = scipy.stats.median_abs_deviation(rf_map.flat, scale='normal')


                # rf_map = S[..., coli] / scale
                # print(rf_map)
                # print(noise_level, scale)
                # cdl.show(rf_map)
                # rf_map /= noise_level

                sel_rf = np.abs(rf_map) > self.noise_floor * noise_level
                rf_map_ = rf_map.copy()
                rf_map_[~sel_rf] = 0
                # cdl.show(rf_map_)
                # plt.show()
                # print(scipy.stats.describe(rf_map[sel_rf]))

                hist = np.histogram(rf_map[sel_rf], self.projection_bins)[0]
                hist[int(len(self.projection_bins) / 2)-1] = 0
                hists[coli,:] = hist * stixel_size ** 2
            # print(hists)
            hist_list.append(file_handling.wrapper(hists))
            # for coli in range(3):
            #     plt.plot(self.projection_bins[:-1]+1/12, hists[coli,:], color=['r','g','b'][coli])
            # plt.show()

        dtab.loc[unit_indices, 'rf_projection_histograms'] = hist_list
        self.update_valid_columns(ct, di)



class Feature_rf_map(Feature):
    name = 'rf map'
    requires = {'unit':{'sta', 'tc'},'dataset': {'primary_colors'}}
    provides = {'unit':{'map_rf',
                        'map_rf_bool',
                        'rf_noise_level',
                        'rf_threshold', 'rf_solidity','rf_size'}}

    rotate_colors = True
    use_rf_boost = True
    noise_multipliers = np.logspace(np.log10(1.5), np.log10(10), 15)

    def generate(self, ct, unit_indices, inpt, dtab=None):
        if dtab is None:
            dtab = ct.unit_table
        di = unit_indices[0][0:2]
        if (missing := self.check_requirements(ct, di)) is not None:
            print('Feature {}. missing requirements {}'.format(self.name, missing))
            return

        tim = cdl.Timer(start=True, count=len(unit_indices))
        color_channels_significant = (1,2)
        # struct = ndimage.generate_binary_structure(2, 1) # make a circular binary shape for expansion
        # struct = ndimage.iterate_structure(struct, 2).astype(bool)

        proc_colors = (0, 1, 2)
        proc_polarities = (1, -1)
        # proc_polarity_names = ('ON','OFF')

        # print('... sta rf map sig stixels: power of color mean, bonf corrected. Starting alpha = {}, increasing 2x until min 1 sig stixel, using channels {} for significance'.format(self.alpha_sigstix, color_channels_significant))
        if self.rotate_colors:
            print('... rotating colors to PCA vectors')
        if self.use_rf_boost:
            print('... using RF boost')

        stimulus_params = ct.dataset_table.at[di, 'stimulus_params'].a
        stixel_size = stimulus_params['stixel_size']
        primary_colors = ct.dataset_table.at[di, 'primary_colors'].a

        for cc, ci in tqdm(enumerate(unit_indices), ncols=60, total=len(unit_indices)):
            S = dtab.at[ci, 'sta'].a
            map_count = 6 # use this to avoid working with all nan maps
            if np.all(np.isnan(S.flat)):
                map_count = 0
                print(f'Unit {ci} has all nan STA')
            tc_all = dtab.at[ci, 'tc_all'].a

            map_rf_bool = np.zeros((S.shape[0], S.shape[1], len(proc_colors) * len(proc_polarities)), dtype=bool)
            map_rf = np.zeros((S.shape[0], S.shape[1], len(proc_colors) * len(proc_polarities)))
            sc_rf_noise_level = np.zeros(len(proc_colors) * len(proc_polarities))
            sc_rf_threshold = np.zeros(len(proc_colors) * len(proc_polarities))
            rf_size = np.zeros(len(proc_colors) * len(proc_polarities))
            rf_size_hull = np.zeros(len(proc_colors) * len(proc_polarities))
            rf_num_islands = np.zeros(len(proc_colors) * len(proc_polarities))
            rf_solidity = np.zeros(len(proc_colors) * len(proc_polarities))

            # rotate S into new primary colors
            if self.rotate_colors:
                S_rot = rotate_sta(S, primary_colors, True)
    
            # make six RF maps here
    
            for map_index in range(map_count):
                coli = int(np.floor(map_index / 2))
                pol = [1, -1][int(map_index % 2)]
    
                if pol == 1:
                    thresh_by_color = -1
    
                # eigenvect = tc_sig_stix[:, coli]
                eigenvect = tc_all[:, map_index]
    
                # make RF maps of positive and negative projections of the time course
                # print('{} {} {}, {}, {}'.format(cci, pi, coli, color_names[coli], proc_polarity_names[pi]))
                # rf_map = pol * np.tensordot(S[..., (coli,)], eigenvect[:, np.newaxis], ((2, 3), (0, 1)))
                rf_map = np.tensordot(S_rot[..., (coli,)], eigenvect[:, np.newaxis], ((2, 3), (0, 1)))
    
                # rf_map = np.max(np.abs(projected_sigstix_polcorrected), 2)
                # rf_map = gaussian_filter(rf_map, 1)
    
                # rf_map /= np.max(np.abs(rf_map.flat)) # measure center and surround (or ON and OFF) on the same scale
                # find contour level
                # noise_level = 2 * np.median(rf_map.flatten()) # one-direction abs maps
                noise_level = stats.median_abs_deviation(rf_map.flat, scale='normal') # zero mean maps (use before zeroing low noise)
    
                rf_map[rf_map < 0] = 0
                if noise_level < 0.000000001:
                    noise_level = 1
                rf_map /= noise_level
    
                # &*
                #### ANALYSIS ACROSS THRESHOLD LEVEL
                # 8&
    
                # threshold = noise_multiplier * threshold_sta_mult_by_cell.get(index, 1)
                thresholds = self.noise_multipliers
                # ic(noise_level, np.max(rf_map), threshold, np.count_nonzero(rf_map > threshold))
    
                h, areas, segments, island_counts, mean_island_area, max_island_areas, \
                position_variances, start_index, best_index, good_rf = calculate_rf_threshold(
                    rf_map, thresholds, stixel_size)
    
                if good_rf:
                    good_threshold = thresholds[best_index]
    
                if good_rf and self.use_rf_boost:
                    rf_map = large_region_rf_boost(rf_map, good_threshold, stixel_size)
    
                if good_rf:
                    if good_threshold > np.max(rf_map): # fix an issue where the threshold is very near the max value
                        good_threshold = np.max(rf_map) - 0.01

                    rf_map_bool = rf_map >= good_threshold
                    thresh_by_color = np.max([thresh_by_color, good_threshold])

                    solidity = measure.regionprops(rf_map_bool * 1)[0].solidity
                else:
                    rf_map_bool = np.zeros_like(rf_map)
                    good_threshold = 0
                    solidity = 0
    
                # if good_rf:
                #     print('good thresh ci {} coli {} pol {}, {}'.format(ci, coli, pol, good_threshold))
    
                map_rf[...,map_index] = rf_map
                sc_rf_noise_level[map_index] = noise_level
                sc_rf_threshold[map_index] = good_threshold
                map_rf_bool[...,map_index] = rf_map_bool
                # rf_map_bool_color[rf_map_bool] = True
                rf_size[map_index] = np.sum(rf_map_bool) * (stixel_size ** 2)
                rf_size_hull[map_index] = rf_size[map_index] / solidity if solidity > 0 else 0
                rf_num_islands[map_index] = ndimage.label(rf_map_bool)[1]
                rf_solidity[map_index] = solidity

    
            dtab.at[ci, 'map_rf'] = file_handling.wrapper(map_rf) # these are boosted, norm to 1
            dtab.at[ci, 'map_rf_bool'] = file_handling.wrapper(map_rf_bool)
            dtab.at[ci, 'rf_noise_level'] = file_handling.wrapper(sc_rf_noise_level)
            dtab.at[ci, 'rf_threshold'] = file_handling.wrapper(sc_rf_threshold)
            dtab.at[ci, 'rf_size'] = file_handling.wrapper(rf_size)
            dtab.at[ci, 'rf_size_hull'] = file_handling.wrapper(rf_size_hull)
            dtab.at[ci, 'rf_num_islands'] = file_handling.wrapper(rf_num_islands)
            dtab.at[ci, 'rf_solidity'] = file_handling.wrapper(rf_solidity)

        self.update_valid_columns(ct, di)
    

# class Feature_rf_contours(Feature):
#     '''
#     takes the RF maps generated by Feature_rf_map and uses contour analysis to get advanced properties
#     can generate the rf_centers like the center of mass, but better
#
#     '''
#     name = 'RF contours'
#     requires = {'unit':{'map_rf'}}
#     provides = {'unit':{'rf_contours'}}
#
#     def generate(self, ct, unit_indices, inpt, dtab=None):
#         if dtab is None:
#             dtab = ct.unit_table
#         di = unit_indices[0][0:2]
#         if (missing := self.check_requirements(ct, di)) is not None:
#             print('Feature {}. missing requirements {}'.format(self.name, missing))
#             return
#
#         stixel_size = ct.dataset_table.at[di, 'stimulus_params'].a['stixel_size']
#
#         for ci in unit_indices:
#             # print('\n', ci, dtab.at[ci, 'label_manual_text'])
#             # get RF map and thresholds
#             rf_maps = dtab.at[ci, 'map_rf'].a
#             thresholds = dtab.at[ci, 'rf_threshold'].a
#
#


class Feature_rf_advanced_properties(Feature):
    '''
    takes the RF maps generated by Feature_rf_map and uses contour analysis to get advanced properties
    can generate the rf_centers like the center of mass, but better

    '''
    name = 'RF advanced properties'
    requires = {'unit':{'map_rf', 'rf_threshold'}}
    provides = {'unit':{'rf_center_x', 'rf_center_y', 'rf_solidity', 'rf_size', 'rf_size_hull','rf_contours'}}

    def generate(self, ct, unit_indices, inpt, dtab=None):
        if dtab is None:
            dtab = ct.unit_table
        di = unit_indices[0][0:2]
        if (missing := self.check_requirements(ct, di)) is not None:
            print('Feature {}. missing requirements {}'.format(self.name, missing))
            return

        print('This feature is disabled because it struggles with RF maps with holes.')
        return

        stixel_size = ct.dataset_table.at[di, 'stimulus_params'].a['stixel_size']

        center_x = []
        center_y = []
        area_list = []
        area_hull_list = []
        solidity_list = []
        contours_list = []

        for ci in unit_indices:
            # print('\n', ci, dtab.at[ci, 'label_manual_text'])
            # get RF map and thresholds
            rf_maps = dtab.at[ci, 'map_rf'].a
            thresholds = dtab.at[ci, 'rf_threshold'].a

            # generate contours and hull contour
            area_coli = np.zeros(7)
            area_hull_coli = np.zeros(7)
            solidity_coli = np.zeros(7)
            polygons_all = []
            contours_coli = []
            valid = False
            for coli in range(6):
                map = rf_maps[...,coli]
                if np.isnan(thresholds[coli]):
                    contours_coli.append([])
                    continue
                valid = True
                # cont = feat_v.find_contours_and_scale(map, thresholds[coli], stixel_size)
                cont = measure.find_contours(map, level=thresholds[coli])
                cont = [stixel_size * seg for seg in cont if len(seg) > 2]
                contours_coli.append(cont)

                print(f'Color {color_names[coli]} got {len(cont)} contours')
                print(cont)
                try:
                    polygons = [Polygon(c) for c in cont]
                    # polygons = [p for p in polygons if p.area > stixel_size ** 2 / 2]
                except:
                    print(f'Failed RF contours for cell {ci}, color {color_names[coli]}')
                    continue
                polygons_all.extend(polygons)
                poly = MultiPolygon(polygons)
                hull = poly.convex_hull
                area_coli[coli] = poly.area
                area_hull_coli[coli] = hull.area
                solidity_coli[coli] = area_coli[coli] / area_hull_coli[coli]

            # metrics of shapes combined over colors
            if valid:
                poly_all = MultiPolygon(polygons_all)
                area_coli[6] = poly_all.area
                area_hull_coli[6] = poly_all.convex_hull.area
                solidity_coli[6] = area_coli[6] / area_hull_coli[6]

                center = poly_all.centroid
                try:
                    center_x.append(center.x)
                    center_y.append(center.y)
                except:
                    center_x.append(0)
                    center_y.append(0)

            else:
                center_x.append(0)
                center_y.append(0)

            # add this cell to all cell list
            area_list.append(file_handling.wrapper(area_coli))
            area_hull_list.append(file_handling.wrapper(area_hull_coli))
            solidity_list.append(file_handling.wrapper(solidity_coli))
            contours_list.append(file_handling.wrapper(contours_coli))

            # print('hull area', hull.area)
            # print(f'solidity {solidity_coli}')
        # print(len(center_x))
            # print(f'center {center.x, center.y}')
            # print(dtab.loc[ci, ('rf_center_x', 'rf_center_y')])
                # hull =
        # ct.show_cell_grid(dtab=ct.unit_table, cell_ids=unit_indices[:10], color_channels=range(6))


        dtab.loc[unit_indices, 'rf_center_x'] = center_x
        dtab.loc[unit_indices, 'rf_center_y'] = center_y
        dtab.loc[unit_indices, 'rf_size'] = area_list
        dtab.loc[unit_indices, 'rf_size_hull'] = area_hull_list
        dtab.loc[unit_indices, 'rf_solidity'] = solidity_list
        dtab.loc[unit_indices, 'rf_contours'] = contours_list

        self.update_valid_columns(ct, di)


class Feature_edge_detection(Feature):
    name = 'STA edge detection'
    requires = {'unit':{'map_sig_stixels'}}
    provides = {'unit':{'sta_edge'}}

    stix_hist_thresh = 0.5  # fraction of peak histogram to draw edge line at
    width_modifier = 1.2  # amount to inset edges, lower is smaller valid region
    edge_threshold = 0.9  # fraction of stixels outside boundary to call it an edge cell

    def generate(self, ct, unit_indices, inpt, dtab=None):
        if dtab is None:
            dtab = ct.unit_table
        di = unit_indices[0][0:2]
        if (missing := self.check_requirements(ct, di)) is not None:
            print('Feature {}. missing requirements {}'.format(self.name, missing))
            return
        print(f'Finding aperture edge cells using sig stixels maps. Using stix_hist_thresh={self.stix_hist_thresh}, width_modifier={self.width_modifier}, edge_threshold={self.edge_threshold}')

        # get sigstix RF maps & combine
        map_all = np.zeros_like(dtab.loc[unit_indices[0],'map_sig_stixels'].a, dtype='int32')
        for index in unit_indices:
            cmap = dtab.at[index, 'map_sig_stixels'].a
            if np.count_nonzero(cmap.flat) > 1:
                map_all += 1 * cmap

        bounds = np.zeros([2, 2])
        other = [1, 0]
        for ax in [0, 1]:
            hist = np.sum(map_all, axis=other[ax])
            threshold = self.stix_hist_thresh * np.max(hist)
            crossings = np.where(np.diff(np.signbit(hist - threshold)))[0] + .5
            if len(crossings) >= 2:
                bounds[ax, :] = [np.min(crossings), np.max(crossings)]
            else:
                bounds[ax, :] = [0, len(hist)]
                print('Do not have two histogram crossings, using the whole thing')

        # inset edges to make new boundaries
        bounds_inset = np.zeros_like(bounds)
        for ax in [0, 1]:
            width = np.diff(bounds[ax, :])
            center = np.mean(bounds[ax, :])
            bounds_inset[ax, :] = center + width * self.width_modifier * np.array([-0.5, 0.5])

        # check if cells are within boundaries and mark in new column
        edge = np.zeros_like(unit_indices, dtype='bool')
        for i, index in enumerate(unit_indices):
            map = dtab.at[index,'map_sig_stixels'].a
            coords = np.transpose(map.nonzero())
            outside = 0
            # ic(coords)
            for pi in range(coords.shape[0]):
                if (coords[pi, 0] < bounds_inset[0, 0] or coords[pi, 0] > bounds_inset[0, 1]) \
                        or (coords[pi, 1] < bounds_inset[1, 0] or coords[pi, 1] > bounds_inset[1, 1]):
                    outside += 1
            outside /= coords.shape[0]
            edge[i] = outside

        # exclude cells by that column
        c = np.count_nonzero(edge > self.edge_threshold)

        print('Found {} edge cells of {} total {:.0f}%'.format(c, unit_indices.shape[0],
                                                                          c / unit_indices.shape[0] * 100))
        dtab.loc[unit_indices, 'sta_edge'] = edge > self.edge_threshold
        # dtab = dtab.astype({'edge': 'bool'}, copy=False)
        self.update_valid_columns(ct, di)


class Feature_detect_grayscale_STA(Feature):
    name = 'grayscale STA detection'
    requires = {'unit':{'sta'}}
    provides = {'unit':{'sta_bw'}}

    def generate(self, ct, unit_indices, inpt, dtab=None):
        if dtab is None:
            dtab = ct.unit_table
        di = unit_indices[0][0:2]
        if (missing := self.check_requirements(ct, di)) is not None:
            print('Feature {}. missing requirements {}'.format(self.name, missing))
            return

        vals = [np.abs(np.sum(np.diff(dtab.at[ci, 'sta'].a, axis=3))) < 1e-6 for ci in unit_indices]
        dtab.loc[unit_indices, 'sta_bw'] = vals

        print(f'Found {np.count_nonzero(vals)} grayscale cells of {len(unit_indices)} total')
        self.update_valid_columns(ct, di)
        
class Feature_rf_convex_hull(Feature):
    name = 'rf convex hull'
    requires = {'unit':{'map_rf'}, 'dataset':{'stimulus_params'}}
    provides = {'unit':{'rf_convex_hull','hull_center_x','hull_center_y'}}
    
    def generate(self, ct, unit_indices, inpt, dtab=None):
        if dtab is None:
            dtab = ct.unit_table
        di = unit_indices[0][0:2]
        if (missing := self.check_requirements(ct, di)) is not None:
            print('Feature {}. missing requirements {}'.format(self.name, missing))
            return
        
        stimulus_params = ct.dataset_table.at[di, 'stimulus_params'].a
        
        bad_units = []
        for ci in tqdm(unit_indices):
            input_map = dtab.at[ci, 'map_rf'].a
            map_index = cdl.channelize(ct.find_primary_channel(dtab.loc[[ci]]))
            rf_map = input_map[...,map_index].copy()
            threshold = dtab.at[ci, 'rf_threshold'].a[map_index]*1.1
            try:
                rf_bool = convex_hull_image(np.squeeze(rf_map >= threshold))
            except:
                bad_units.append(ci)
                dtab.at[ci, 'rf_convex_hull'] = file_handling.wrapper(None)
                dtab.at[ci, 'hull_center_x'] = None
                dtab.at[ci, 'hull_center_y'] = None
                continue
            rf_map[np.logical_not(rf_bool)] = 0
            fill_fraction = np.count_nonzero(rf_bool) / (rf_map.shape[0] * rf_map.shape[1])
            if np.any(rf_bool) and (fill_fraction < 0.2):
                segments = find_contours_and_scale(np.squeeze(rf_map), threshold, stimulus_params['stixel_size'])
                segments_filtered = []
                # loop over segments found
                if len(segments):
                    for si, seg in enumerate(segments):
                        if seg.shape[0] < 3:
                            continue
                        segments_filtered.append(Polygon(seg))
                    segments = segments_filtered
                    segments_filtered = []
                if len(segments):
                    poly = MultiPolygon(segments).convex_hull
                    center = poly.centroid
                    area = poly.area
                    # centers.append([center.x, center.y])
                    # areas.append(area)
                    if area > 80:  # use 80 for bubbly
                        segments_filtered.append(np.array(poly.boundary.xy).T)
                    # centers = np.array(centers);  areas = np.array(areas)
                # ic(areas)
                segments = segments_filtered
                if segments is None or len(segments) == 0:
                    bad_units.append(ci)
                    dtab.at[ci, 'rf_convex_hull'] = file_handling.wrapper(None)
                    dtab.at[ci, 'hull_center_x'] = None
                    dtab.at[ci, 'hull_center_y'] = None
                else:
                    dtab.at[ci, 'rf_convex_hull'] = file_handling.wrapper(segments)
                    dtab.at[ci, 'hull_center_x'] = center.x
                    dtab.at[ci, 'hull_center_y'] = center.y
            else:
                bad_units.append(ci)
                dtab.at[ci, 'rf_convex_hull'] = file_handling.wrapper(None)
                dtab.at[ci, 'hull_center_x'] = None
                dtab.at[ci, 'hull_center_y'] = None
        if len(bad_units) > 0:
            print(f'Found {len(bad_units)} bad units, Fill Fraction too low or no rf, Setting units as invalid')
            dtab.loc[bad_units, 'valid'] = False
            
        self.update_valid_columns(ct, di)
        
class Feature_rf_boundary(Feature):
    name = 'rf boundary'
    requires = {'unit':{'map_rf', 'rf_center_x', 'rf_center_y'}, 'dataset':{'stimulus_params'}}
    provides = {'unit':{'rf_boundary'}}
    
    def generate(self, ct, unit_indices, inpt, dtab=None):
        if dtab is None:
            dtab = ct.unit_table
        di = unit_indices[0][0:2]
        if (missing := self.check_requirements(ct, di)) is not None:
            print('Feature {}. missing requirements {}'.format(self.name, missing))
            return
        
        stimulus_params = ct.dataset_table.at[di, 'stimulus_params'].a
        
        bad_units = []
        for ci in tqdm(unit_indices):
            input_map = dtab.at[ci, 'map_rf'].a
            map_index = cdl.channelize(ct.find_primary_channel(dtab.loc[[ci]]))
            rf_map = input_map[...,map_index].copy()
            threshold = dtab.at[ci, 'rf_threshold'].a[map_index]*1.1
            rf_bool = (rf_map >= threshold)
            rf_map[np.logical_not(rf_bool)] = 0
            fill_fraction = np.count_nonzero(rf_bool) / (rf_map.shape[0] * rf_map.shape[1])
            if np.any(rf_bool) and (fill_fraction < 0.2):
                segments = find_contours_and_scale(np.squeeze(rf_map), threshold, stimulus_params['stixel_size'])
                segments_filtered = []
                # loop over segments found
                if len(segments):
                    for si, seg in enumerate(segments):
                        if seg.shape[0] < 3:
                            continue
                        poly = Polygon(seg)
                        # center = poly.centroid
                        area = poly.area
                        # centers.append([center.x, center.y])
                        # areas.append(area)
                        if area > 80:  # use 80 for bubbly
                            segments_filtered.append(seg)
                    # centers = np.array(centers);  areas = np.array(areas)
                # ic(areas)
                segments = segments_filtered
                
                if segments is None or len(segments) == 0:
                    bad_units.append(ci)
                    dtab.at[ci, 'rf_boundary'] = file_handling.wrapper(None)
                    continue
                else:
                    dtab.at[ci, 'rf_boundary'] = file_handling.wrapper(segments)
                minx, miny = np.min(segments[0],axis=0)
                maxx, maxy = np.max(segments[0],axis=0)
                minx,miny = np.Inf, np.Inf
                maxx,maxy = -np.Inf, -np.Inf
                for segs in segments:
                    tminx, tminy = np.min(segs,axis=0)
                    tmaxx, tmaxy = np.max(segs,axis=0)

                    minx = np.min([minx, tminx])
                    miny = np.min([miny, tminy])
                    maxx = np.max([maxx, tmaxx])
                    maxy = np.max([maxy, tmaxy])

                center_x = dtab.at[ci, 'rf_center_x']
                center_y = dtab.at[ci, 'rf_center_y']
                if center_x < minx or center_x > maxx or center_y < miny or center_y > maxy:
                    bad_units.append(ci)
            else:
                bad_units.append(ci)
                dtab.at[ci, 'rf_boundary'] = file_handling.wrapper(None)
        if len(bad_units) > 0:
            print(f'Found {len(bad_units)} bad units, Fill Fraction too low or no rf, Setting units as invalid')
            dtab.loc[bad_units, 'valid'] = False
            
        self.update_valid_columns(ct, di)
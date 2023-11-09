import sys

sys.path.append('../')

from tqdm import tqdm
import scipy
from scipy.optimize import curve_fit
import numpy as np
from scipy.interpolate import griddata
from features import Feature
import file_handling


class Feature_load_spike_times(Feature):
    name = 'spike times'
    requires = {'unit':{'unit_id'}}
    provides = {'unit':{'spike_times'}}
    input = {'analysis_data'}

    def generate(self, ct, unit_indices, inpt, dtab=None):
        if dtab is None:
            dtab = ct.unit_table
        di = unit_indices[0][0:2]
        if (missing := self.check_requirements(ct, di)) is not None:
            print('Feature {}: missing requirements {}'.format(self.name, missing))
            return

        dtab.loc[unit_indices, 'spike_times'] = [file_handling.wrapper(np.array(inpt['analysis_data'].get_spike_times_for_cell(dtab.at[ci, 'unit_id']), dtype='uint32')) for ci in unit_indices]

        self.update_valid_columns(ct, di)


class Feature_spikes_basic(Feature):
    name = 'spikes basic'
    requires = {'unit': {'spike_times'}}
    provides = {'unit': {'spike_count', 'spike_duration', 'spike_rate_mean'}}
    input = {}

    def generate(self, ct, unit_indices, inpt, dtab=None):
        if dtab is None:
            dtab = ct.unit_table
        di = unit_indices[0][0:2]
        if (missing := self.check_requirements(ct, di)) is not None:
            print('Feature {}: missing requirements {}'.format(self.name, missing))
            return

        invalids = []
        for ci in unit_indices:
            spike_times = dtab.at[ci, 'spike_times'].a
            dtab.at[ci, 'spike_count'] = len(spike_times)

            if len(spike_times) > 20:
                dtab.at[ci, 'spike_duration'] = np.max(spike_times) / 20000
                dtab.at[ci, 'spike_rate_mean'] = dtab.at[ci, 'spike_count'] / dtab.at[ci, 'spike_duration']

                # check for ISI violations indicative of unit contamination
                # isi = np.diff(spike_times)
                # min = np.min(isi)
                # if min <= 2:
                #     print(np.min(isi))
                #     print(dtab.at[ci, 'label_manual_text'])

            else:
                dtab.at[ci, 'spike_duration'] = 0
                dtab.at[ci, 'spike_rate_mean'] = 0
                dtab.at[ci, 'valid'] = False
                invalids.append(ci)
        if len(invalids) > 0:
            print(f'... Found {len(invalids)} units with no spikes, marked invalid')

        print('... processed spikes, mean {:.0f} count by cell'.format(np.mean(dtab['spike_count'])))

        self.update_valid_columns(ct, di)



class Feature_generate_acf_from_spikes(Feature):
    name = 'acf from spike times'
    requires = {'unit':{'spike_times'}}
    provides = {'unit':{'acf'}, 'dataset':{'acf_bins'}}
    input = {'analysis_data'}

    def generate(self, ct, unit_indices, inpt, dtab=None):
        if dtab is None:
            dtab = ct.unit_table
        di = unit_indices[0][0:2]
        if (missing := self.check_requirements(ct, di)) is not None:
            print('Feature {}: missing requirements {}'.format(self.name, missing))
            return

        interval = 500 # ms
        step = interval * 20
        count = 101
        end = 1000 * 20000
        # bins = np.arange(0,count*step, step=step)
        # bins = np.logspace(0, 101*step, 20)
        bins = np.logspace(1.3, np.log10(end), count)
        bins = np.concatenate([bins, [np.inf]])
        # print(bins.astype(float) / 20)
        diffs = [bins[1] - bins[0], bins[-2] - bins[-3]]
        print(f'... Calculating ISI using {len(bins) - 1} logarithmic bins, from {diffs[0]:.1f} to {diffs[1]:.1f} samples wide, from {bins[0]:.1f} to {bins[-2]:.1f}, density mode')

        acfs = [file_handling.wrapper(np.histogram(np.diff(dtab.at[ci, 'spike_times'].a), bins=bins, density=True)[0][:-1]) for ci in unit_indices]
        dtab.loc[unit_indices, 'acf'] = acfs
        ct.dataset_table.at[di, 'acf_bins'] = file_handling.wrapper(bins[:-2] / 20000)

        self.update_valid_columns(ct, di)

class Feature_load_acf(Feature):
    name = 'acf'
    requires = {'unit':{'unit_id'}}
    provides = {'unit':{'acf'}}
    input = {'analysis_data'}

    def generate(self, ct, unit_indices, inpt, dtab=None):
        if dtab is None:
            dtab = ct.unit_table
        di = unit_indices[0][0:2]
        if (missing := self.check_requirements(ct, di)) is not None:
            print('Feature {}: missing requirements {}'.format(self.name, missing))
            return

        if inpt['analysis_data'] is not None:
            dtab.loc[unit_indices, 'acf'] = [file_handling.wrapper(inpt['analysis_data'].get_acf_numpairs_for_cell(dtab.at[ci, 'unit_id'])) for ci in unit_indices]
            # acfs = []
            # fails = 0
            # for ci in unit_indices:
            #     try:
            #         acfs.append(file_handling.wrapper(inpt['analysis_data'].get_acf_numpairs_for_cell(dtab.at[ci, 'unit_id'])))
            #     except:
            #         acfs.append(np.nan)
            #         # fails += 1
            #         # print(ci)
            # # assert(fails == 0)
            # dtab.loc[unit_indices, 'acf'] = acfs
        else:
            path = ct.dataset_table.at[di, 'sta_path'].replace('_sta','_acf')
            print(f'Loading ACF from combined dataset, path: {path}')
            import h5py

            acf_file = h5py.File(path,'r')
            for ci in unit_indices:
                try:
                    acf_in = acf_file[str(dtab.at[ci, 'unit_id'] + 1)]
                    acf = np.zeros(acf_in.shape, dtype='float32')
                    acf_in.read_direct(acf)
                    acf = acf.transpose()[:,0]
                except:
                    dtab.at[ci, 'valid'] = False
                    acf = 0
                dtab.at[ci, 'acf'] = file_handling.wrapper(acf)

            acf_file.close()
        self.update_valid_columns(ct, di)

class Feature_load_ei(Feature):
    name = 'load ei'
    requires = {'unit':{'unit_id'}}
    provides = {'unit':{'ei'},'dataset':{'ei_electrode_locations'}}
    input = {'analysis_data','ei'}

    def generate(self, ct, unit_indices, inpt, dtab=None):
        if dtab is None:
            dtab = ct.unit_table
        di = unit_indices[0][0:2]
        if (missing := self.check_requirements(ct, di)) is not None:
            print('Feature {}: missing requirements {}'.format(self.name, missing))
            return

        eis = []

        if inpt['analysis_data'] is not None:
            ct.dataset_table.at[di, 'ei_electrode_locations'] = file_handling.wrapper(inpt['analysis_data'].get_electrode_map())
            for ci in unit_indices:
                try:
                    ei = inpt['analysis_data'].get_ei_for_cell(dtab.at[ci, 'unit_id']).ei
                except:
                    eis.append(np.nan)
                    dtab.at[ci, 'valid'] = False
                    print(f'!!! unit_id {ci} EI not found in analysis data for dataset, marked invalid')
                    continue
                if ci == unit_indices[0]:
                    print(f'... EI shape is {ei.shape}')
                if np.all(np.isnan(ei)):
                    ei = np.zeros_like(ei)
                    print('...got all nan EI at unit index {}, converting to zeros'.format(ci))
                eis.append(file_handling.wrapper(ei))

        else:
            path = ct.dataset_table.at[di, 'ei_path']
            print(f'Loading ei from combined dataset, path: {path}')
            import h5py

            ei_file = h5py.File(path,'r')
            for ci in unit_indices:
                try:
                    ei_in = ei_file[str(dtab.at[ci, 'unit_id'] + 1)]
                    ei = np.zeros(ei_in.shape, dtype='float32')
                    ei_in.read_direct(ei)
                    ei = ei.transpose()
                    if np.all(np.isnan(ei)):
                        ei = np.zeros_like(ei)
                        print('...got all nan EI at unit index {}, converting to zeros'.format(ci))
                except:
                    dtab.at[ci, 'valid'] = False
                    ei = 0
                eis.append(file_handling.wrapper(ei))

            ei_sample = dtab.at[unit_indices[0], 'ei'].a
            if ei_sample.shape[0] == 512:
                print('... using saved electrode map for 512 electrodes (60 µm array)')
                electrode_map = np.loadtxt('/Volumes/Lab/Users/scooler/classification/electrode_arrays/array_512.csv', delimiter=',')
                ct.dataset_table.at[di, 'ei_electrode_locations'] = file_handling.wrapper(
                    electrode_map)

        dtab.loc[unit_indices, 'ei'] = eis
        self.update_valid_columns(ct, di)

class Feature_stimulus_TTL(Feature):
    name = 'stimulus TTL'
    requires = {}
    provides = {'dataset':{'stimulus_ttl'}}
    input = {'analysis_data'}

    def generate(self, ct, unit_indices, inpt, dtab=None):
        if dtab is None:
            dtab = ct.unit_table
        di = unit_indices[0][0:2]
        if (missing := self.check_requirements(ct, di)) is not None:
            print('Feature {}: missing requirements {}'.format(self.name, missing))
            return

        di = unit_indices[0][0:2]
        ct.dataset_table.at[di, 'stimulus_ttl'] = file_handling.wrapper(inpt['analysis_data'].get_ttl_times())

        self.update_valid_columns(ct, di)

class Feature_spike_waveform(Feature):
    name = 'spike waveform'
    requires = {'unit':{'ei'}, 'dataset':{'ei_electrode_locations'}}
    provides = {'unit':{'spike_waveform_maxenergy', 'spike_waveform_maxamplitude','spike_waveform_smart',
                        'ei_edge', 'ei_peak','ei_axon_only'}}
    input = {}

    def generate(self, ct, unit_indices, inpt, dtab=None):
        if dtab is None:
            dtab = ct.unit_table
        di = unit_indices[0][0:2]
        if (missing := self.check_requirements(ct, di)) is not None:
            print('Feature {}: missing requirements {}'.format(self.name, missing))
            return

        electrode_locations = ct.dataset_table.loc[di,'ei_electrode_locations'].a
        x_range = [np.min(electrode_locations[:, 0]), np.max(electrode_locations[:, 0])]
        y_range = [np.min(electrode_locations[:, 1]), np.max(electrode_locations[:, 1])]

        waves_maxenergy = []
        waves_maxamplitude = []
        waves_smart = []
        ei_edge = []
        peaks = []
        axons = []
        for ci in unit_indices:
            ei = dtab.at[ci, 'ei'].a
            electrode = np.argmax(np.mean(np.power(ei, 2), axis=1))
            waves_maxenergy.append(file_handling.wrapper(ei[electrode, :]))

            electrode = np.argmax(np.max(np.abs(ei), axis=1))
            waves_maxamplitude.append(file_handling.wrapper(ei[electrode, :]))

            n = 5
            electrodes = np.argsort(np.max(-1 * ei, axis=1))[::-1]
            waves = ei[electrodes[:n], :]
            scales = -1 / np.min(waves, axis=1)[:, np.newaxis]
            waves *= scales
            ratios = []
            axon_detect = []
            for wi in range(waves.shape[0]):
                # waves[wi, :] = cdl.center_wave(waves[wi, :], 201, 100)
                center = np.argmin(waves[wi,:])
                if center == 0:
                    center = 1
                early_max = np.max(waves[wi, :center])
                late_max = np.max(waves[wi, center:])
                ratios.append(late_max - np.clip(early_max, 0, np.inf))
                axon_detect.append(np.abs(late_max / early_max))
            # ord = np.argsort(ratios)[::-1]
            wave = waves[np.argsort(ratios)[-1], :]
            waves_smart.append(file_handling.wrapper(wave))

            if np.max(axon_detect) < 1:
                axons.append(True)
            else:
                axons.append(False)

            max_loc = electrode_locations[electrode]
            if max_loc[0] in x_range or max_loc[1] in y_range:
                ei_edge.append(True)
            else:
                ei_edge.append(False)

            peaks.append(np.max(np.abs(ei)))

        dtab.loc[unit_indices, 'spike_waveform_maxenergy'] = waves_maxenergy
        dtab.loc[unit_indices, 'spike_waveform_maxamplitude'] = waves_maxamplitude
        dtab.loc[unit_indices, 'spike_waveform_smart'] = waves_smart
        dtab.loc[unit_indices, 'ei_edge'] = ei_edge
        dtab.loc[unit_indices, 'ei_peak'] = peaks
        dtab.loc[unit_indices, 'ei_axon_only'] = axons

        self.update_valid_columns(ct, di)


class Feature_ei_select_electrodes(Feature):
    # %% EI electrode selection

    name = 'ei select electrodes'
    requires = {'unit':{'ei'}}
    provides = {'dataset':{'ei_electrode_selection'}}
    input = {}

    thresh_var_var_log = 0.5

    def generate(self, ct, unit_indices, inpt, dtab=None):
        if dtab is None:
            dtab = ct.unit_table
        di = unit_indices[0][0:2]
        if (missing := self.check_requirements(ct, di)) is not None:
            print('Feature {}: missing requirements {}'.format(self.name, missing))
            return

        sample_ei = dtab.at[unit_indices[0], 'ei'].a
        num_electrodes = sample_ei.shape[0]
        num_frames = sample_ei.shape[1]

        print('...filtering EI variance variance log using {} electrodes, {} frames, {} threshold'.format(num_electrodes, num_frames, self.thresh_var_var_log))
        ei_var_bycell = np.zeros((len(unit_indices), num_electrodes))
        # ei_value_bycell = np.zeros((len(indices), num_electrodes))

        for cc, ci in enumerate(unit_indices):
            # col = -1
            ei = dtab.at[ci, 'ei'].a
            # ei_value_bycell[ci,:] = ei[:, 20]

            # ei_energy = np.mean(np.power(ei, 2), axis=1)
            ei_var_bycell[cc, :] = np.var(ei, axis=1)

        var_var_log = np.log10(np.var(ei_var_bycell, axis=0))
        sel_electrodes = var_var_log >= self.thresh_var_var_log

        di = unit_indices[0][0:2]
        ct.dataset_table.at[di, 'ei_electrode_selection'] = file_handling.wrapper(sel_electrodes.astype(bool))

        self.update_valid_columns(ct, di)
        # plt.figure()
        # plt.plot(var_var_log)
        # plt.gca().axhline(thresh_var_var_log, color='k')
        # plt.title('variance variance')

        print('...found {} good, {} bad electrodes'.format(np.count_nonzero(sel_electrodes),
                                                    len(sel_electrodes) - np.count_nonzero(sel_electrodes)))

class Feature_ei_map(Feature):
    name = 'ei map'
    requires = {'unit':{'ei'},'dataset':{'ei_electrode_selection'}}
    provides = {'unit':{'map_ei_energy', 'map_ei_energy_early', 'map_ei_energy_late'}}
    input = {'analysis_data'}

    y_dim = 30 # general EI map scaling

    def generate(self, ct, unit_indices, inpt, dtab=None):
        if dtab is None:
            dtab = ct.unit_table
        di = unit_indices[0][0:2]
        if (missing := self.check_requirements(ct, di)) is not None:
            print('Feature {}: missing requirements {}'.format(self.name, missing))
            return

        di = unit_indices[0][0:2]
        sel_electrodes = ct.dataset_table.at[di, 'ei_electrode_selection'].a

        electrode_map = ct.dataset_table.at[di, 'ei_electrode_locations'].a
        # if not 'analysis_data' in inpt or inpt['analysis_data'] is not None:
        #
            # print(electrode_map)
            # np.savetxt('/Volumes/Lab/Users/scooler/classification/electrode_arrays/array_512.csv', electrode_map, delimiter=',')
        # else:
        #     ei_sample = dtab.at[unit_indices[0], 'ei'].a
        #     if ei_sample.shape[0] == 512:
        #         print('... using saved electrode map for 512 electrodes (60 µm array)')
        #         electrode_map = np.loadtxt('/Volumes/Lab/Users/scooler/classification/electrode_arrays/array_512.csv', delimiter=',')
        electrode_map = electrode_map[sel_electrodes]

        xrange = (np.min(electrode_map[:, 0]), np.max(electrode_map[:, 0]))
        yrange = (np.min(electrode_map[:, 1]), np.max(electrode_map[:, 1]))

        x_dim = int((xrange[1] - xrange[0])/(yrange[1] - yrange[0]) * self.y_dim) # x dim is proportional to y dim

        x_e = np.linspace(xrange[0], xrange[1], x_dim)
        y_e = np.linspace(yrange[0], yrange[1], self.y_dim)

        grid_x, grid_y = np.meshgrid(x_e, y_e)
        grid_x = grid_x.T
        grid_y = grid_y.T

        sample_ei = dtab.at[unit_indices[0], 'ei'].a
        num_electrodes = sample_ei.shape[0]
        num_frames = sample_ei.shape[1]

        print('... formatting EI energy maps: {}/{} electrodes, {} time frames, to rectangular {} by {}'.format(np.count_nonzero(sel_electrodes), num_electrodes, num_frames, len(x_e), len(y_e)))

        maps = []
        maps_early = []
        maps_late = []

        for counter, ci in enumerate(unit_indices):
            ei = dtab.at[ci, 'ei'].a  # add a .copy() if I do anything with it
            # ei = ei[sel_electrodes, :]

            # ei_grid = np.zeros([len(x_e), len(y_e), num_frames])
            # for ti in range(ei.shape[1]):
            #     grid = griddata(electrode_map, ei[sel_electrodes, ti], (grid_x, grid_y), method='linear', fill_value=0)
            #     ei_grid[:, :, ti] = grid

            # dtab.at[ci, 'ei_grid'] = file_handling.wrapper(ei_grid)

            # ei_energy_grid = np.log10(np.mean(np.power(ei_grid, 2), axis=2) + .000000001)
            peak_electrode = np.argmax(np.max(np.abs(ei), axis=1))
            peak_frame = np.argmin(ei[peak_electrode, :])
            split_frame = peak_frame + 30
            # split_frame = int(num_frames / 2)

            # skip generating the whole map and do energy directly from EI
            ei_energy = np.log10(np.mean(np.power(ei, 2), axis=1) + .000000001)
            ei_energy_grid = griddata(electrode_map, ei_energy[sel_electrodes], (grid_x, grid_y), method='linear', fill_value=np.median(ei_energy[sel_electrodes]))
            maps.append(file_handling.wrapper(ei_energy_grid))

            ei_energy = np.log10(np.mean(np.power(ei[:, :split_frame], 2), axis=1) + .000000001)
            ei_energy_grid = griddata(electrode_map, ei_energy[sel_electrodes], (grid_x, grid_y), method='linear', fill_value=np.median(ei_energy[sel_electrodes]))
            maps_early.append(file_handling.wrapper(ei_energy_grid))

            ei_energy = np.log10(np.mean(np.power(ei[:, split_frame:], 2), axis=1) + .000000001)
            ei_energy_grid = griddata(electrode_map, ei_energy[sel_electrodes], (grid_x, grid_y), method='linear', fill_value=np.median(ei_energy[sel_electrodes]))
            maps_late.append(file_handling.wrapper(ei_energy_grid))


            if counter % 200 == 0:
                print('done with {} of {}'.format(counter, len(unit_indices)))

        dtab.loc[unit_indices, 'map_ei_energy'] = maps
        dtab.loc[unit_indices, 'map_ei_energy_early'] = maps_early
        dtab.loc[unit_indices, 'map_ei_energy_late'] = maps_late
        self.update_valid_columns(ct, di)

        print('... done formatting EI energy')


def exponential(x, b):
    return np.exp(-b * x)


class Feature_ei_profile(Feature):
    name = 'EI profile'
    requires = {'unit':{'ei'}, 'dataset':{'ei_electrode_locations'}}
    provides = {'unit':{'ei_energy_profile', 'ei_energy_profile_std', 'ei_energy_profile_decay'}}
    input = {}

    angles = np.linspace(0, 2 * np.pi, 20)[:-1]
    dist_max = 1000
    amp = np.linspace(0, dist_max, 20)

    def generate(self, ct, unit_indices, inpt, dtab=None):
        if dtab is None:
            dtab = ct.unit_table
        di = unit_indices[0][0:2]
        if (missing := self.check_requirements(ct, di)) is not None:
            print('Feature {}: missing requirements {}'.format(self.name, missing))
            return
        electrode_map = ct.dataset_table.at[di, 'ei_electrode_locations'].a

        values = []
        values_std = []
        decays = []
        for ind in tqdm(unit_indices, total=len(unit_indices), ncols=60):
            cell = dtab.loc[ind]
            # try:
            ei = cell['ei'].a

            ei_energy = np.log10(np.mean(np.power(ei, 2), axis=1) + .000000001)
            center = electrode_map[np.argmax(ei_energy), :]
            ray_energy = np.zeros([len(self.angles), len(self.amp)])
            for ai, angle in enumerate(self.angles):
                ray_x = np.cos(angle) * self.amp + center[0]
                ray_y = np.sin(angle) * self.amp + center[1]
                ray_energy[ai, :] = griddata(electrode_map, ei_energy, (ray_x, ray_y), method='linear')
            valid = np.count_nonzero(~np.isnan(ray_energy), axis=0) >= 4
            ray_energy_m = np.nanmedian(ray_energy, axis=0)
            ray_energy_m_std = np.nanvar(ray_energy, axis=0)
            ray_energy_m[~valid] = np.nan
            ray_energy_m_std[~valid] = np.nan

            tail_values = ray_energy_m[~np.isnan(ray_energy_m)][-4:]
            profile = ray_energy_m - np.mean(tail_values)
            profile /= np.nanmax(profile)
            popt, pcov = curve_fit(exponential, self.amp[valid], profile[valid], p0=[.005])  # , sigma=profile_std[~np.isnan(profile)])

            energy_floor = np.median(np.sort(ei_energy)[:50]) # the lowest 50 electrodes
            ray_energy_m -= energy_floor
            scale = np.nanmax(ray_energy_m)
            # print(scale)
            ray_energy_m /= scale


            values.append(file_handling.wrapper(ray_energy_m))
            values_std.append(file_handling.wrapper(ray_energy_m_std))
            decays.append(popt[0])

        # noise = scipy.stats.median_abs_deviation(ei_energy[ei_energy < np.max(ei_energy) * .05])


        dtab.loc[unit_indices, 'ei_energy_profile'] = values
        dtab.loc[unit_indices, 'ei_energy_profile_std'] = values_std
        dtab.loc[unit_indices, 'ei_energy_profile_decay'] = decays

        self.update_valid_columns(ct, di)

class Feature_ei_correlation_data(Feature):
    name = 'correlation data'
    requires = {'unit':{'ei'}}
    provides = {'unit':{'ei_energy_raw'}}
    input = {}

    def generate(self, ct, unit_indices, inpt, dtab=None):
        if dtab is None:
            dtab = ct.unit_table
        di = unit_indices[0][0:2]
        if (missing := self.check_requirements(ct, di)) is not None:
            print('Feature {}: missing requirements {}'.format(self.name, missing))
            return

        energy = []
        for ci in unit_indices:
            ei = dtab.at[ci, 'ei'].a
            energy.append(file_handling.wrapper(np.mean(np.power(ei, 2), axis=1)))
        dtab.loc[unit_indices, 'ei_energy_raw'] = energy
        self.update_valid_columns(ct, di)


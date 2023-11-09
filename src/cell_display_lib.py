# a library of functions for analysis of the primate retina
# by Sam Cooler, Chichilnisky Lab 2022
import importlib
import sys
# sys.path.append('../')
sys.path.append('/Volumes/Lab/Development/artificial-retina-software-pipeline/utilities/')


from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings, time, colorsys, datetime, textwrap, pickle, os, random
from scipy.interpolate import griddata

from scipy import stats, spatial, io, ndimage
from shapely import geometry
from skimage import measure
from matplotlib import  cm
from sklearn.preprocessing import LabelEncoder
from datetime import datetime
from neo import SpikeTrain
from elephant.conversion import BinnedSpikeTrain
from elephant.spike_train_correlation import correlation_coefficient, cross_correlation_histogram
import quantities as pq
from tqdm import tqdm
import re

import visionloader as vl
import file_handling
import features as feat

def pretty(gg):
    print('shape: {}'.format(gg.shape))
    print(gg)

class Timer:
    run_times = []
    total_count = None
    name = ''

    def __init__(self, start=True, count=None, name=''):
        if start:
            self.tick()
        self.name = name
        self.total_count = count

    def tick(self):
        t = time.time()
        self.run_times = [t]
        print(f'*** timer {self.name} started')
        return t

    def tock(self, count=None):
        t = time.time()
        # todo: add duration estimation
        total_elapsed = t - self.run_times[0]

        time_per_count = 1
        if count is not None and self.total_count is not None:
            time_per_count = total_elapsed / (count + 1)
            total_estimate = time_per_count * self.total_count
        else:
            total_estimate = -1

        if total_estimate > 0:  #datetime.timedelta(seconds=total_estimate)
            print('*** elapsed {:.0f}s of {:.0f}s = {:.1f}m elapsed, of {:.1f}m estimated ({}/{}) ({:.1f} / sec)'
                  .format(t - self.run_times[-1], total_elapsed, (t - self.run_times[0]) / 60, total_estimate / 60, count + 1, self.total_count, 1 / time_per_count))
        else:
            print('*** elapsed {:.0f}s of {:.0f}s = {:.1f}m elapsed'.format(t - self.run_times[-1], total_elapsed, (t - self.run_times[0]) / 60))
        self.run_times.append(t)
        return t

def make_display_colors(primary_colors):
    proc_colors = range(3)
    proc_polarities = (1, -1)
    primary_colors_contrast = np.zeros([6,3])
    for cci, coli in enumerate(proc_colors):
        for pi, pol in enumerate(proc_polarities):
            map_index = coli * 2 + pi
            ccc = primary_colors[coli,:] * pol
            ccc = ccc / np.max(np.abs(ccc)) * 0.5 + 0.5 # convert contrast to color in [0,1]
            primary_colors_contrast[map_index, :] = ccc

    return primary_colors_contrast

def colorsets(colorset=0):
    if colorset == 0:
        colors = ['xkcd:dark red', 'xkcd:blue', 'xkcd:cyan', 'xkcd:dark orange', 'xkcd:orange', 'xkcd:purple',
                  'xkcd:dark green', 'xkcd:lime', 'xkcd:light purple', 'xkcd:pink', 'xkcd:yellow',
                  'xkcd:brown', 'xkcd:bright red', 'xkcd:gold', 'xkcd:green yellow', 'xkcd:navy',
                  'xkcd:dark periwinkle',
                  'xkcd:mauve', 'xkcd:avocado']
    elif colorset == 1:
        colors = ['#e6194B', '#3cb44b', '#ffe119', '#4363d8', '#f58231', '#911eb4', '#42d4f4', '#f032e6', '#bfef45',
                  '#fabed4', '#469990', '#dcbeff', '#9A6324', '#800000', '#aaffc3', '#808000', '#ffd8b1', '#000075']
    elif colorset == 2:
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22',
                  '#17becf', 'xkcd:bright teal', 'xkcd:golden yellow', 'xkcd:hot pink', 'xkcd:gray']

    return colors


def make_plot_format(types, colorset=1):
    colors = colorsets(colorset)

    type_format = dict()
    color_indices = {'other': 0, 'blue': 0, 'ON': 0, 'OFF': 0}
    color_order_by_group = {'other': np.random.permutation(colors), 'blue': np.random.permutation(colors),
                            'ON': np.random.permutation(colors), 'OFF': np.random.permutation(colors)}
    for typ in types:
        if 'ON OFF' in typ:
            group = 'other'

        elif 'blue' in typ or 'SBC' in typ or typ == 'S' or typ == 'christmas':
            group = 'blue'
        elif 'ON' in typ or 'On' in typ:
            group = 'ON'
        elif 'OFF' in typ or 'Off' in typ:
            group = 'OFF'
        else:
            group = 'other'
        if color_indices[group] >= len(colors):
            color = 'gray'
        else:
            color = color_order_by_group[group][color_indices[group]]
        color_indices[group] += 1
        type_format[typ] = {'group': group, 'color': color}
    type_format = pd.DataFrame(type_format).T
    return type_format

def channelize(words):
    """
    Convert "green ON" to 2 for rf_map lookup and plotters
    """
    colis = []
    if isinstance(words, str):
        words = [words]
    if isinstance(words, int):
        print('got a single number')
        return None
    for word in words:
        coli = 0
        if 'green' in word or 'pc0' in word:
            coli += 2
        if 'blue' in word or 'pc1' in word:
            coli += 4
        if 'off' in word or 'OFF' in word:
            coli += 1
        colis.append(coli)
    return colis

def cid_to_str(cid,uid=None):
    o = f'{cid[0]}:c{cid[1]}'
    if uid is not None:
        o += f'u{uid}'
    return o


def new_dataset_dict():
    dataset_dict = {'run_id':[],'run_file_name': [], 'sorter':[], 'labels':[], 'piece_id':[], 'path':[],
                    'note':[], 'label_data_path':[], 'sta_path':[], 'ei_path':[],
                    'stimulus_type':[], 'species':[]}
    return dataset_dict


def make_paths(ct, piece_id, run_id, analysis_path=None, run_file_name=None, verbose=False, export_path_base=None, sta_path_base=None, sorter='kilosort'):
    if export_path_base is None:
        export_path_base = '/Volumes/Scratch/Users/alexth/all_data'

    if sta_path_base is None:
        # sta_path_base = ct.paths['scratch_analysis']
        sta_path_base = '/Volumes/Scratch/Users/alexth/supersample-stas'

    if run_file_name is None:
        run_file_name = f'data{run_id}'

    if sorter == 'vision':
        ks = ''
    elif sorter == 'kilosort':
        ks = f'/kilosort_data{run_id}'
    elif sorter == 'yass':
        ks = f'/yass_data{run_id}'

    if analysis_path is None:
        analysis_path = ct.paths['analysis_root'] + f'/{piece_id}{ks}/data{run_id}'
    if verbose:
        print('Analysis path: ', analysis_path)

    # if sorter == 'kilosort':
    #     label_mode = 'alexandra'
    #     label_data_path = f'{export_path_base}/{piece_id}/all_ids_{piece_id}_grant.mat'
    #     if not os.path.isfile(label_data_path):
    #         label_data_path = label_data_path.replace('_grant', '_last')
    # else:
    label_data_path = ''
    label_mode = ''
    if not os.path.isfile(label_data_path):
        label_data_path = f'{analysis_path}/{run_file_name}.classification_scooler.txt'
        label_mode = 'list'
    if not os.path.isfile(label_data_path):
        label_data_path = f'{analysis_path}/{run_file_name}.classification_bhofflic.txt'
        label_mode = 'list'
    if not os.path.isfile(label_data_path):
        label_data_path = f'{analysis_path}/{run_file_name}.classification_agogliet.txt'
        label_mode = 'list'
    # print(label_data_path, run_id)
    if (not os.path.isfile(label_data_path)) and run_id == 'com':
        label_data_path = f'{sta_path_base}/{piece_id}/data999/data999.classification.txt'
        label_mode = 'list'
    # print(label_data_path, run_id)
    if not os.path.isfile(label_data_path):
        # check for params file:
        if os.path.isfile(f'{analysis_path}/{run_file_name}.params'):
            label_data_path = ''
            label_mode = 'vision'
        else:
            label_data_path = ''
            label_mode = ''
            print('Vision labels load disabled I suppose!')
    # print(label_data_path, run_id)

    if label_mode == 'alexandra':
        run_ids, vision_ids = feat.load_alex_mat(label_data_path, verbose)
        if run_ids is None:
            if verbose:
                print(f'no alexandra types file found at {label_data_path}. Reverting to vision.')
            label_data_path = ''
            label_mode = 'vision'
        elif not run_id in run_ids:
            if verbose:
                print(f'checked alex types file, but {piece_id} {run_id} not present. Reverting to vision.')
            label_data_path = ''
            label_mode = 'vision'

    sta_path = f'{sta_path_base}/{piece_id}/{sorter}_data{run_id}/data{run_id}/data{run_id}.wu_sta'
    if not os.path.isfile(sta_path):
        if verbose:
            print(f'Missing 50-frame STA {sta_path}')
        sta_path = ''  # don't actually use the path if it's just the standard .sta file

    ei_path = f'{sta_path_base}/{piece_id}/{sorter}_data{run_id}/data{run_id}/data{run_id}.ei'
    if not os.path.isfile(ei_path):
        if verbose:
            print(f'Missing long EI {ei_path}')
        # if make_long_ei:
        #     cdl.make_long_ei(piece_id, run_id)
        ei_path = ''

    return label_mode, label_data_path, sta_path, ei_path


def find_wn_runs(piece_id, filter_bw=False, filter_large_stixel=False):
    """
    Finds white noise runs using the database spreadsheet. Searches for simple standard WN runs by default.
    :param piece_id:
    :param filter_bw:
    :param filter_large_stixel:
    :return:
    """
    fn_runs = '/Volumes/Lab/Users/scooler/database_spreadsheet/database_spreadsheet_runs.csv'

    table_runs = pd.read_csv(fn_runs)
    s = f"Piece == @piece_id and `exp type` == 'Visual' and `jitter` != 1.0 and type == 'binary' and loop != '1'"
    table_runs = table_runs.query(s)
    table_runs = table_runs.astype({'stixel width': 'float', 'interval': 'float'})
    if filter_bw:
        table_runs = table_runs.query("`BW/RGB` == 'RGB'")
    if filter_large_stixel:
        table_runs = table_runs.query("`stixel width` <= 20 and interval <= 8")

    # display(table_runs)

    run_ids = [d[-3:] for d in table_runs['datarun']]
    return run_ids


def make_piece_run(pieces):
    """
    Loops through pieces to call find_wn_runs
    :param pieces:
    :return:
    """
    piece_run = dict()
    for piece_id in pieces:
        piece_run[piece_id] = find_wn_runs(piece_id)
    return piece_run

def sort_labels(labels_in):
    labels = np.array([l for l in labels_in if len(l) > 1])

    # sort labels
    scores = np.zeros(len(labels))
    for i, label in enumerate(labels):
        if 'OFF' in label:
            scores[i] += 0.3
        if 'blue' in label:
            scores[i] += 0.6
        if 'large' in label:
            scores[i] += 0.01

        if 'parasol' in label:
            scores[i] += 0
        elif 'midget' in label:
            scores[i] += 1
        elif 'RGC' in label:
            scores[i] += 2
        elif 'PAC' in label:
            scores[i] += 3
        elif 'unlabeled' in label or len(label) == 0 or label == 'BW':
            scores[i] += 10
    order = np.argsort(scores)
    return labels[order]

def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

def get_piece_directory(ct, piece_id, make=True):
    """
    generates the file locations for a given piece, requires that ct.paths be set correctly
    :param ct:
    :param piece_id:
    :param make:
    :return:
    """
    piece_directory = f"{ct.paths['h5_labels']}/{piece_id}/"
    if os.path.exists(piece_directory):
        return piece_directory
    else:
        if make:
            os.mkdir(piece_directory)
            return piece_directory
        else:
            return None


class CellTable:
    """
    CellTable is wonderful
    """
    unit_table = pd.DataFrame()
    cell_table = pd.DataFrame()
    dataset_table = pd.DataFrame()
    corr_table = pd.DataFrame()
    pdict = {}
    features_table = pd.DataFrame()
    file_info = []
    log_level = -1
    paths = dict()

    def __init__(self,
                 ct=None,
                 dir_analysis_root= '/Volumes/Analysis',
                 dir_scratch_analysis='/Volumes/Scratch/Analysis',
                 dir_h5_labels='/Volumes/Scratch/Users/scooler/analysis',
                 dir_datasets='/Volumes/Scratch/Users/scooler/celltable_datasets'):

        self.log('Welcome to the CellTable ~experience~')
        self.log('')

        if ct is None:
            self.log('Starting a fresh new CellTable')
            self.paths['analysis_root'] = dir_analysis_root
            self.paths['scratch_analysis'] = dir_scratch_analysis
            self.paths['h5_labels'] = dir_h5_labels
            self.paths['datasets'] = dir_datasets
        else:
            self.log('Loading all content from another CellTable')
            self.unit_table = ct.unit_table
            self.cell_table = ct.cell_table
            self.dataset_table = ct.dataset_table
            self.corr_table = ct.corr_table
            self.pdict = ct.pdict
            self.features_table = ct.features_table
            self.file_info = ct.file_info
            self.paths = ct.paths

    def log(self, message, importance=0):
        if importance > self.log_level:
            symbols = ('~','+','!!!')
            print(f'{symbols[importance]} {message}')

    def __str__(self):
        return f'**CellTable of {len(self.unit_table)}'

    def info(self):
        if self.cell_table.shape[0] > 0:
            self.log('cells table has {} active of {} entries'.format(np.count_nonzero(self.cell_table.valid), self.cell_table.shape[0]))
        else:
            self.log('cells table is empty')


        if self.cell_table.shape[0] > 0:
            self.log('cells table has {} active of {} entries'.format(np.count_nonzero(self.cell_table.valid), self.cell_table.shape[0]))
        else:
            self.log('cells table is empty')

    # LOADING & SAVING
    def load_tables(self, unit_table, cell_table, dataset_table, pdict):
        self.unit_table = unit_table; self.cell_table = cell_table; self.dataset_table = dataset_table
        self.pdict = pdict

    def make_dataset_table(self,
                           piece_run_pairs:dict,
                           note='', labels='vision', sorter='vision', unique_add_on=''):
        """Helper function to create a dataset_table from your piece_run_pairs dictionary"""
        dataset_dict = {'piece_id': [], 'run_id': [], 'note': [], 'sorter': [], 'labels': [],  'path': [],
                        'label_data_path': []}
        index = []

        for piece_name, runs in piece_run_pairs.items():
            for run_name in runs:
                path = self.paths['analysis_root'] + f'/{piece_name}/{run_name}/'
                run_id = f'{run_name[-3:]}{unique_add_on}'
                dataset_dict['run_id'].append(run_id)
                dataset_dict['piece_id'].append(piece_name)
                dataset_dict['note'].append(note)
                dataset_dict['path'].append(path)
                dataset_dict['labels'].append(labels)
                dataset_dict['sorter'].append(sorter)
                dataset_dict['label_data_path'].append('')
                index.append((piece_name, run_id))
        dataset_table = pd.DataFrame(dataset_dict, index=pd.MultiIndex.from_tuples(index, names=['piece_id','run_id']))
        return dataset_table

    def add_datasets(self, dataset_table_add):
        """Add datasets within an input table to the internal dataset_table"""
        self.log(f'Adding {len(dataset_table_add)} datasets')
        added = []
        # for di in dataset_table_add.index:
        try:
            self.dataset_table = pd.concat([self.dataset_table, dataset_table_add], copy=False)
            # added.extend(dataset_table_add.index)
            # print(f'Added {di}')
        except ValueError:
            print(f'Duplicate datasets already in dataset_table')
        # ct.dataset_table.set_index(keys=['piece_id', 'run_id'], inplace=True, drop=False)
        # return added
        return dataset_table_add.index

    def drop_piece(self, a):
        print(f'piece_id {a}')
        piece_id = a
        indices = self.unit_table.query(f"piece_id == '{piece_id}'").index
        self.log(f'Dropping {len(indices)} units')
        self.unit_table.drop(index=indices, inplace=True)

        indices = self.cell_table.query(f"piece_id == '{piece_id}'").index
        self.log(f'Dropping {len(indices)} cells')
        self.cell_table.drop(index=indices, inplace=True)

        indices = self.dataset_table.query(f"piece_id == '{piece_id}'").index
        self.log(f'Dropping {len(indices)} datasets')
        self.dataset_table.drop(index=indices, inplace=True)

    def drop_dataset(self, a):
        print(f'di {a}')
        piece_id, run_id = a
        indices = self.unit_table.query(f"piece_id == '{piece_id}' and run_id == '{run_id}'").index
        self.log(f'Dropping {len(indices)} units')
        self.unit_table.drop(index=indices, inplace=True)

        # indices = self.cell_table.query(f"piece_id == '{piece_id}'").index
        # self.log(f'Dropping {len(indices)} cells')
        # self.cell_table.drop(index=indices, inplace=True)

        indices = self.dataset_table.query(f"piece_id == '{piece_id}' and run_id == '{run_id}'").index
        self.log(f'Dropping {len(indices)} datasets')
        self.dataset_table.drop(index=indices, inplace=True)

        self.log('Still need to handle the cells containing these units!')

    def drop_noncom(self):
        print('Dropping all runs except combined')
        for piece_id, run_id in self.dataset_table.index:
            if run_id != 'com':
                self.drop_dataset((piece_id, run_id))


    def filter_datasets_lowest_only(self):
        print('Filtering datasets to keep only lowest run for each piece')
        for piece_id in self.dataset_table.piece_id.unique():
            datasets = self.dataset_table.query(f"piece_id == '{piece_id}'")
            if len(datasets) > 1:
                runs = datasets.run_id.copy()
                best = np.min([int(r) for r in runs])
                for r in runs:
                    if not int(r) == best:
                        # print(f'drop {piece_id}, {r}')
                        self.drop_dataset((piece_id, r))

    def initialize_units_for_datasets(self,
                                      dataset_indices_add=None
                                      ):
        if dataset_indices_add is None:
            dataset_indices_add = self.dataset_table.index

        # highest_current_dataset_id = np.max(self.dataset_table.index) if self.dataset_table.shape[0] > 0 else -1
        self.log(f'Loading new units from {dataset_indices_add}, currently have {self.dataset_table.shape[0]} datasets')
        # dataset_indices_add = np.array(dataset_input.index)

        # load units for each cell and setup basic units table
        # unit_table holds all detections of single neurons. Very likely to be one cell, has a single stimulus From one dataset, will be combined
        #   into the cell_table later
        self.log(f'Starting with unit_table having {self.unit_table.shape[0]} units')

        for di in dataset_indices_add:
            # add dataset to table
            # print(di)
            dataset = self.dataset_table.loc[di]

            if len(dataset['path']) > 0:
                try:
                    run_file_name = dataset['run_file_name']
                except:
                    # load basic unit info
                    run_id = dataset['run_id']
                    if re.match(r"^\d{3}$", run_id):
                        run_file_name = f"data{run_id}"
                    else:
                        run_file_name = run_id
                analysis_data = vl.load_vision_data(dataset['path'], run_file_name,
                                                    include_params=False,include_ei=False,include_sta=False,include_neurons=True)
                vision_ids_this_dataset = analysis_data.get_cell_ids()
            elif len(dataset['label_data_path']) > 0:
                self.log('Loading Alexandra\'s combined dataset IDs')
                run_ids, vision_ids = feat.load_alex_mat(dataset['label_data_path'])
                vision_ids_this_dataset = list(range(vision_ids.shape[0]))
                # vision_ids_this_dataset = io.loadmat(dataset['label_data_path'])['ids'][0]
                # vision_ids_this_dataset = np.arange(len(vision_ids_this_dataset)).astype(int)
            else:
                self.log('Making a fresh cElL-tAbLe deduplication comBINED dataset')
                vision_ids_this_dataset = list(self.dataset_table.at[di, 'unit_ids'].a)

            # add units to the units table
            index = pd.MultiIndex.from_product(([di[0]],[di[1]], vision_ids_this_dataset), names=('piece_id','run_id','unit_id'))
            unit_table_add = pd.DataFrame({'unit_id':vision_ids_this_dataset, 'dataset_id':[di for a in range(len(vision_ids_this_dataset))], 'run_id':di[1], 'piece_id':di[0], 'valid':True}, index=index)

            if index[0] in self.unit_table.index:
                self.log(f'Error: unit {index[0]} is already present in the unit_table, may already have initialized these units. Halting.', 2)
                return
            # self.unit_table = self.unit_table.append(unit_table_add)
            self.unit_table = pd.concat([self.unit_table, unit_table_add], copy=False)
            self.dataset_table.loc[di, 'unit_ids'] = file_handling.wrapper(index.to_numpy())
            self.dataset_table.loc[di, 'valid_columns_unit'] = file_handling.wrapper({'unit_id'})
            self.dataset_table.loc[di, 'valid_columns_dataset'] = file_handling.wrapper({'piece_id','run_id'})

            self.log('Dataset {}: Loaded {} units, sorter/alex indices {} through {}'.format(
                di, len(vision_ids_this_dataset), np.min(vision_ids_this_dataset),
                np.max(vision_ids_this_dataset)))

        self.unit_table = self.unit_table.astype({'unit_id': 'uint32', 'valid': 'bool'}, copy=False)
        self.unit_table.sort_index(inplace=True)
        self.log('done with dataset setup, now have {} datasets, {} units'.format(self.dataset_table.shape[0], self.unit_table.shape[0]))
        return dataset_indices_add


    def generate_save_file_names(self, ds_fname):
        filename_unit_table = 'ct_{}_unit_table.pkl'.format(ds_fname)
        filename_dataset_table = 'ct_{}_dataset_table.pkl'.format(ds_fname)
        filename_cell_table = 'ct_{}_cell_table.pkl'.format(ds_fname)
        filename_corr_table = 'ct_{}_corr_table.pkl'.format(ds_fname)
        filename_etc = 'ct_{}_etc.pkl'.format(ds_fname)

        return {'filename_cell_table': filename_cell_table,
                'filename_unit_table': filename_unit_table,
                'filename_dataset_table': filename_dataset_table,
                'filename_corr_table': filename_corr_table,
                'filename_etc': filename_etc}

    def file_load(self, file_root='/Volumes/Scratch/Users/scooler/celltable/', file_name='default'):

        fnames = self.generate_save_file_names(file_name)
        if file_root[-1] != '/':
            file_root += '/'
        self.file_info = (file_root, file_name)

        self.log(f'starting file load from {file_name} in {file_root}')
        try:
            unit_table_size = os.stat(file_root + fnames['filename_unit_table']).st_size
            self.log('Loading units table, weighing in at a hefty {:.2f} gigabytes'.format(unit_table_size / 1e9))

            self.unit_table = pd.read_pickle(file_root + fnames['filename_unit_table'])
        except:
            self.log('no units table found???')
            return
        self.log('loaded units table with {} entries, {} cols'.format(self.unit_table.shape[0], self.unit_table.shape[1]))
        try:
            self.cell_table = pd.read_pickle(file_root + fnames['filename_cell_table'])
            self.log('loaded cells table with {} entries, {} cols'.format(self.cell_table.shape[0], self.cell_table.shape[1]))
        except:
            self.log('missing cell table, not great')
            return
        try:
            self.dataset_table = pd.read_pickle(file_root + fnames['filename_dataset_table'])
            self.log('loaded dataset table with {} entries, {} cols'.format(self.dataset_table.shape[0], self.dataset_table.shape[1]))
        except:
            self.log('no dataset table? OPE')
            return
        try:
            self.corr_table = pd.read_pickle(file_root + fnames['filename_corr_table'])
            self.log('loaded corr table with {} entries, {} cols'.format(self.corr_table.shape[0], self.corr_table.shape[1]))
        except:
            self.log('no corr table? WOMP WOMP (but fine)')
        try:
            with open(file_root + fnames['filename_etc'], 'rb') as f:
                etc = pickle.load(f)
                self.pdict = etc['pdict']
            self.log('loaded other files also')
        except:
            self.log('missing etc files')
            return

        self.log('done loading files')

        if self.cell_table.shape[0] == 0:
            self.log('Cells table is empty, might want to initialize_cell_table() or deduplicate.')

    def file_save(self, file_root='/Volumes/Scratch/Users/scooler/celltable/', file_name=None):
        if file_name is None:
            if len(self.file_info) == 2:
                file_root, file_name = self.file_info
                self.log(f'Using stored file location {file_root} --- {file_name}')
            else:
                self.log('missing file save location info')
                return

        fnames = self.generate_save_file_names(file_name)
        tim = Timer()
        self.log('saving has begun to {}'.format(fnames.values()))
        tim.tick()
        self.unit_table.to_pickle(file_root + fnames['filename_unit_table'])
        self.log('managed to saved unit_table to {}'.format(fnames['filename_unit_table']))
        tim.tock()

        self.cell_table.to_pickle(file_root + fnames['filename_cell_table'])
        self.log('scribbled down cell_table to {}'.format(fnames['filename_cell_table']))
        tim.tock()

        self.dataset_table.to_pickle(file_root + fnames['filename_dataset_table'])
        self.log('glady saved dataset_table to {}'.format(fnames['filename_dataset_table']))
        tim.tock()

        self.corr_table.to_pickle(file_root + fnames['filename_corr_table'])
        self.log('jotted corr_table to {}'.format(fnames['filename_corr_table']))
        tim.tock()

        etc = {'pdict': self.pdict}
        f = open(file_root + fnames['filename_etc'], 'wb')
        pickle.dump(etc, f)
        self.log('felt okay about saving other vars to {}'.format(fnames['filename_etc']))

        unit_table_size = os.stat(file_root + fnames['filename_unit_table']).st_size
        self.log('FYI units table is {:.2f} GB'.format(unit_table_size / 1e9))

        self.log("all done! you're safe for now.")

        now = datetime.now()
        current_time = now.strftime("%H:%M:%S")
        self.log(f"Time now is {current_time}")


    def file_save_pieces(self,
                           file_root=None,
                           piece_ids=None,
                           save_name=None
                           ):
        """ Save internal tables to per-dataset pickle files.
        Use save_name to give your project a name so you can reload a set of datasets easily"""

        if file_root is None:
            file_root = self.paths['datasets']
        if piece_ids is None:
            piece_ids = self.dataset_table.piece_id.unique()
        if isinstance(piece_ids, str):
            print('pieces is a string, converting to list')
            piece_ids = [piece_ids]

        self.log(f'Saving {len(piece_ids)} pieces, to {file_root}')

        if save_name is not None:
            self.log(f'Saving entire datasets table for later per-piece file loading')
            fname = f'{file_root}/ctds_{save_name}.pkl'
            self.dataset_table.to_pickle(fname)
            self.log('glady saved the whole dataset_table to {}'.format(fname))
            # tim.tock()

        tim = Timer(start=True, count=len(piece_ids))
        for pp, piece_id in enumerate(piece_ids):

           # save the units for this piece, from multiple datasets
            cells = self.cell_table.query(f"piece_id == '{piece_id}'").copy()
            units = self.unit_table.query(f"piece_id == '{piece_id}'").copy()
            datasets = self.dataset_table.query(f"piece_id == '{piece_id}'").copy()
            # correlations = self.corr_table.query(f"piece_id == '{piece_id}'").copy()

            self.log(f'piece_id {piece_id} with {len(datasets)} runs, {len(cells)} cells, {len(units)} units')
            file_name = file_root + '/' + f'ctd_{piece_id}'

            datasets.to_pickle(file_name + '_d.pkl')
            cells.to_pickle(file_name + '_c.pkl')
            units.to_pickle(file_name + '_u.pkl')

            self.log(f'Saved datasets, cells, units to pandas DataFrames at\n{file_name}_[d,c,u].pkl')
            tim.tock(pp)
        self.log('Done saving, go in peace.')

    def file_load_pieces(self,
                           file_root=None,
                           pieces=None,
                           save_name=None,
                           count=None,
                           process_labels=True,
                           ):
        """
        Load internal tables from per-dataset pickle files
        :param file_root:
        :param pieces:
        :param save_name:
        :param count:
        :param process_labels:
        :return:
        """

        if file_root is None:
            file_root = self.paths['datasets']

        if save_name is not None:
            self.log(f'Loading and overwriting entire datasets table for per-dataset file loading')
            fname = f'{file_root}/ctds_{save_name}.pkl'
            self.dataset_table = pd.read_pickle(fname)

        if pieces is None:
            pieces = np.array(self.dataset_table.piece_id.unique())
        if isinstance(pieces, str):
            print('pieces is a string, converting to list')
            pieces = [pieces]

        if count is not None:
            assert False
            pieces = pieces[:count]
            self.dataset_table = self.dataset_table.loc[pieces]

        self.log(f'Loading {len(pieces)} pieces from {file_root}: {pieces}')
        tim = Timer(start=True, count=len(pieces))
        for pp, piece_id in enumerate(pieces):
            try:
                file_name = f'ctd_{piece_id}'

                print(f'Loading piece {piece_id}')
                datasets = pd.read_pickle(file_root + '/' + file_name + '_d.pkl')
                cells = pd.read_pickle(file_root + '/' + file_name + '_c.pkl')
                units = pd.read_pickle(file_root + '/' + file_name + '_u.pkl')

                # self.dataset_table.loc[piece_id, dataset.columns] = dataset.loc[piece_id, dataset.columns]
                self.dataset_table = pd.concat([self.dataset_table, datasets], verify_integrity=True)
                self.cell_table = pd.concat([self.cell_table, cells], verify_integrity=True)
                self.unit_table = pd.concat([self.unit_table, units], verify_integrity=True)

                self.log(f'Successfully loaded piece {piece_id}')
                tim.tock(pp)
            except:
                print(f'Failure loading piece {piece_id}')
                raise

        if process_labels:
            self.log('Processing labels (replace nan, update label encoder and unique names)')
            self.cell_table.label_manual_text.fillna('unlabeled', inplace=True)
            le = LabelEncoder()
            label_manual = le.fit_transform(self.cell_table.label_manual_text)
            self.cell_table['label_manual'] = label_manual
            self.pdict['label_manual_uniquenames'] = le.classes_
            # num_types = len(label_manual_uniquenames)

            # self.generate_features('all', [], [features.Feature_process_manual_labels])
            # self.cell_table.label_manual = self.get_label_encoder().transform(self.cell_table.label_manual_text.fillna(''))

            print('Copying cell labels to units')
            cids = self.cell_table.query('valid == True').index
            if np.any(self.unit_table['run_id'].apply(lambda x: x == 'com')):
            # if 'unit_id_wn_combined' in self.cell_table.columns:
                print('combined mode')
                for cid in cids:
                    uid = (cid[0], 'com', cid[1])
                    self.unit_table.loc[uid, ['label_manual', 'label_manual_text']] = \
                        self.cell_table.loc[cid, ['label_manual', 'label_manual_text']]
            else:
                for cid in cids:
                    for uid in self.cell_table.at[cid, 'unit_ids']:
                        self.unit_table.loc[uid, ['label_manual', 'label_manual_text']] = \
                            self.cell_table.loc[cid, ['label_manual', 'label_manual_text']]

        self.log('Done loading, time to analyze.')

    def drop_cells(self, cell_ids):
        pass

    def initialize_cell_table(self, pieces=None):
        """
        Starts a new cell_table for this CellTable, with one cell for each unit.
        :param pieces: list of piece_id strings
        :return:
        """
        self.cell_table = pd.DataFrame()

        # assign one unit for each cell, will deduplicate these next
        self.log('Initialize cell table')
        if pieces is None:
            pieces = self.dataset_table.piece_id.unique()
        for piece_id in pieces:
            datasets = self.dataset_table.query(f'piece_id == "{piece_id}"')
            unit_ids = pd.MultiIndex.from_tuples(np.concatenate([u.a for u in datasets.unit_ids]))
            print(f'Piece {piece_id} has {len(unit_ids)} units from {datasets.shape[0]} datasets')

            cell_table_dict = dict()
            # unit_ids = pd.MultiIndex.from_tuples(self.dataset_table.loc[di].unit_ids.a)
            cell_table_dict['unit_ids'] = [(ind,) for ind in unit_ids]
            cell_table_dict['unit_ids_by_run'] = [file_handling.wrapper({ind[1]:(ind,)}) for ind in unit_ids]
            cell_table_dict['unit_ids_wn'] = [tuple() for ind in unit_ids]
            cell_table_dict['unit_id_wn_combined'] = [np.nan for ind in unit_ids]
            cell_table_dict['unit_id_nsem'] = [np.nan for ind in unit_ids]

            cell_table_dict['corr_included'] = [[] for ind in unit_ids]
            cell_table_dict['merge_scores'] = [[] for ind in unit_ids]
            try:
                cell_table_dict['label_manual'] = np.array(self.unit_table.loc[unit_ids, 'label_manual_input'])
                cell_table_dict['label_manual_text'] = np.array(self.unit_table.loc[unit_ids, 'label_manual_text_input'])
            except:
                self.log('failed to move label_manual')

            # cell_table_dict['dataset_id'] = [di for a in unit_ids]
            cell_table_dict['piece_id'] = piece_id
            cell_table_dict['merge_strategy'] = ''
            cell_table_dict['valid'] = np.array(self.unit_table.loc[unit_ids, 'valid'])

            # for c in cell_table_dict.keys():
            #     print(c)
            #     v = cell_table_dict[c]
            #     try:
            #         print(len(v))
            #     except:
            #         print(v)

            cell_index_numbers = np.arange(len(unit_ids))
            index = pd.MultiIndex.from_product([[piece_id], cell_index_numbers], names=['piece_id', 'cell_id'])
            cell_table_add = pd.DataFrame(cell_table_dict, index=index)
            self.cell_table = pd.concat([self.cell_table, cell_table_add], copy=False)

            # store cell_id in each unit
            # index = pd.MultiIndex.from_product([[di], unit_ids])
            self.unit_table.loc[unit_ids, 'cell_id'] = index.to_numpy()

        # move label input to label for each unit, to be deduplicated for some cells
        try:
            self.unit_table['label_manual_text'] = self.unit_table['label_manual_text_input']
            self.unit_table['label_manual'] = self.unit_table['label_manual_input']
        except:
            self.log('failed to move label_manual')

        self.cell_table['unit_ids'] = self.cell_table['unit_ids'].astype(object)
        self.cell_table['unit_id_nsem'] = self.cell_table['unit_id_nsem'].astype(object)

        self.log('Created fresh new cell_table and reset unit_table links. Now, you may deduplicate.')

    # def deduplicate(self, use_sta=True, do_combine_unit_data=True, verbose=False, pieces=None):
    #     import deduplication, importlib
    #     importlib.reload(deduplication)
    #     deduplication.deduplicate(self, use_sta,do_combine_unit_data, verbose, pieces)
    #     print('Deduplication complete.')

    def copy_cell_labels_to_units(self, source_table=None, label_source='manual'):
        if source_table is None:
            source_table = self.cell_table
        print(f'Copying cell {label_source} labels to units')
        cids = source_table.index
        for cid in tqdm(cids, ncols=40, total=len(cids)):
            for uid in self.cell_table.at[cid, 'unit_ids']:
                self.unit_table.loc[uid, [f'label_{label_source}', f'label_{label_source}_text']] = \
                    source_table.loc[cid, [f'label_{label_source}', f'label_{label_source}_text']]
        print('done')

    def copy_unit_labels_to_cells(self):
        # such as after Alexandra updates the manual labels
        for col in ['label_manual', 'label_manual_text']:
            self.unit_table[col] = self.unit_table[col + '_input']

        cell_ids = self.cell_table.index
        for ci in tqdm(cell_ids, ncols=40):
            uids = pd.MultiIndex.from_tuples(self.cell_table.loc[ci, 'unit_ids'])
            label_text = self.unit_table.loc[uids, 'label_manual_text_input'].to_numpy()

            # evaluate labels and see which is better
            scores = []
            for i, t in enumerate(label_text):
                try:
                    if 'dupli' in t:
                        s = -2
                    elif t in ['crap', 'bad', 'other', 'weird', 'contaminated', 'edge', 'weak', 'unlabeled', 'wasnan','Unclassified','unclassified']:
                        s = -1
                    elif t in ['ON parasol', 'ON midget', 'OFF parasol', 'OFF midget']:
                        s = 2
                    elif 'ON' in t or 'OFF' in t or 'blue' in t or 'SBC' in t or 'A1' in t or 'PAC' in t or 'RGC' in t:
                        s = 1
                    else:
                        s = 0
                    scores.append(s)
                except:
                    scores.append(0)
            if len(scores) > 0:
                best_label_index = np.argmax(scores)
                # ic(labels_text[best_label_index])
                uid_selected = uids[best_label_index]
                # if label_text[best_label_index] not in ['ON parasol', 'ON midget', 'OFF parasol', 'OFF midget']:
                #     print(label_text, label_text[best_label_index])
            else:
                uid_selected = uids[0]

            # alternatively, use label_out = labels[0] # just pick the highest one, should be fine, but will lose some possible info
            # move labels from best unit (_input) into cell
            for col in ['label_manual', 'label_manual_text']:
                self.cell_table.at[ci, col] = self.unit_table.at[uid_selected, col + '_input']
        print('Done moving labels')

    def copy_unit_labels_to_cells_combined(self):
        print('copying unit labels to cells (combined)')
        self.unit_table['label_manual_text'] = self.unit_table['label_manual_text_input']
        self.unit_table['label_manual'] = self.unit_table['label_manual_input']
        units = self.unit_table.query(f"run_id == 'com'")
        for uid, unit in units.iterrows():
            # print(uid, unit['label_manual_text_input'])
            self.cell_table.loc[(uid[0], uid[2]), 'label_manual'] = unit['label_manual_input']
            self.cell_table.loc[(uid[0], uid[2]), 'label_manual_text'] = unit['label_manual_text_input']
        print(f'Done for {units.shape[0]} cells')

    def get_label_encoder(self): # deprecated
        le = LabelEncoder()
        le.classes_ = self.pdict['label_manual_uniquenames']
        return le

    def make_label_table(self, dtab):
        label_table = pd.DataFrame(index=dtab.index)
        label_table['piece_id'] = [a[0] for a in label_table.index]
        # label_table['label_manual'] = dtab['label_manual']
        label_table['label_manual_text'] = dtab['label_manual_text']
        label_table['label_auto_text'] = ''
        # label_table['label_auto'] = -1
        return label_table

    # DATASET ACCESS
    def cid_to_uid(self, cell_ids, stimulus_type='whitenoise', use_combined=True, use_combined_only=False):
        '''

        :param cell_ids:
        :param stimulus_type: for now, 'whitenoise' is all but we should be able to select other units
        :param use_combined: return the combined dataset units (from deduplication)
        :param use_combined_only: if a cell does not exist in the combined run, do not return it
        :return: list of unit ids
        '''
        # look up units for each cell using basic merge strategies
        unit_ids = []
        for ci, cell_id in enumerate(cell_ids):
            cell = self.cell_table.loc[cell_id]
            if stimulus_type == 'nsem':
                try:
                    unit_id = cell.unit_id_nsem.a
                except:
                    unit_id = cell.unit_ids[0]
            else:
                if use_combined and stimulus_type == 'whitenoise' and len(cell.unit_ids_wn) > 0:
                    unit_id = (cell_id[0], 'com', cell_id[1])
                else:
                    if use_combined_only:
                        unit_id = np.nan
                    else:

                        unit_id = cell.unit_ids[0] # choose the first one

            # if cell.merge_strategy == 'h':
            unit_ids.append(unit_id)
        return unit_ids

    def get_cells(self, cell_ids=None, reset_index=False, columns=None, generate=False, types=None, datasets=None, use_combined=True, use_combined_only=False):
        # compose units into a table as cells

        if cell_ids is None: # get all cells
            cell_ids = self.cell_selection()
        if len(cell_ids) == 0:
            print('no ids passed to me')
            return
        if types is not None or datasets is not None:
            cell_ids = self.cell_selection(types=types, datasets=datasets)

        unit_ids = self.cid_to_uid(cell_ids, use_combined=use_combined, use_combined_only=use_combined_only)
        cell_ids_out = []
        unit_ids_out = []
        for uid, cid in zip(unit_ids, cell_ids):
            if isinstance(uid, tuple):
                unit_ids_out.append(uid)
                cell_ids_out.append(cid)
            else:
                print(f'removed {cid} due to use_combined_only')
        cell_ids = cell_ids_out
        unit_ids = unit_ids_out

        dtab = self.unit_table.loc[unit_ids]

        #todo: also return cell data or move cell data info into returned table
        # ctab = self.cell_table.loc[cell_ids]
        # dtab

        # if reset_index:
        index = pd.MultiIndex.from_tuples(cell_ids)
        dtab.set_index(index, inplace=True, drop=False)
        return dtab

    def get_cell(self, cell_id, columns=None, generate=False):
        return self.unit_table.loc[self.cid_to_uid([cell_id])[0]]

    def cell_selection(self, dtab=None,
                        types=None,
                        datasets=None,
                        include_invalid=False,
                        label_source='label_manual_text',
                        mode='indices',
                        pieces=None
                        ):
        """
        select a subset of a cells or records table, returned as a boolean 1D nparray for dtab.loc[sel] use
        and a sel_indices list of indices
        specify a list of type names and/or datasets (with all included by default)
        """

        if dtab is None:
            dtab = self.cell_table
        if types == 'bigfive':
            types = ['ON parasol', 'OFF parasol', 'ON midget', 'OFF midget', 'SBC']

        if isinstance(types, str):
            raise RuntimeError('Oops, types should be a list; you gave a single string, add brackets around it')
        if types is not None:
            # le = self.get_label_encoder()
            sel_types = np.zeros(dtab.shape[0], dtype=bool)
            for typ in types:
                # try:
                #     # sel_types = np.logical_or(sel_types, dtab[label_source] == le.transform([t])[0])
                sel_types = np.logical_or(sel_types, np.array(dtab[label_source].apply(lambda t: t == typ)))
                # print(f'types count {np.count_nonzero(sel_types)}')
        else:
            sel_types = np.ones(dtab.shape[0], dtype=bool)

        if pieces is not None:
            datasets = pieces
        if datasets is not None:
            sel_datasets = np.zeros(dtab.shape[0], dtype=bool)
            if isinstance(datasets[0], int):
                datasets = self.dataset_table.iloc[datasets, :].index
            for d in datasets:
                if len(d) == 2:
                    d = d[0]
                sel_datasets = np.logical_or(sel_datasets, np.array(dtab.piece_id == d))
        else:
            sel_datasets = np.ones(dtab.shape[0], dtype=bool)

        if include_invalid:
            sel_valid = np.ones(dtab.shape[0], dtype=bool)
        else:
            if 'valid' in dtab.columns:
                sel_valid = np.array(dtab.valid, dtype=bool)
            elif 'active' in dtab.columns:
                sel_valid = np.array(dtab.active, dtype=bool)
            else:
                # print('no validity col found, check input table format')
                sel_valid = np.ones(dtab.shape[0], dtype=bool)

        sel_bool = np.array(np.logical_and.reduce((sel_types, sel_datasets, sel_valid)))
        sel_indices = np.array(dtab.iloc[sel_bool].index)  # cell_table mode

        return sel_indices if mode == 'indices' else sel_bool

    def cell_selection_examples(self,
                        dtab=None,
                        types=None,
                        datasets=None,
                        pieces=None,
                        include_invalid=False,
                        label_source='label_manual_text',
                        count_each_type=4,
                        count_random=0,
                        random_exclude_common_types=True,
                        manual_indices=tuple()
                        ):
        """
        Make a random selection of cells for data exploration
        Args:
            types:
            datasets:
            include_invalid:
            label_source:
            count_each_type:
            count_random:
            random_exclude_common_types:
            manual_indices:

        Returns:

        """
        if dtab is None:
            dtab = self.cell_table
        if types == 'bigfive':
            types = ['ON parasol', 'OFF parasol', 'ON midget', 'OFF midget', 'SBC']
        if datasets is None:
            datasets = pieces
        cell_indices = []
        if types is not None:
            for t in types:
                indices = self.cell_selection(dtab=dtab, types=[t], datasets=datasets, include_invalid=include_invalid, label_source=label_source)
                indices = random.sample(list(indices), np.min((count_each_type, len(indices))))
                cell_indices.extend(indices)

        # get some random cells for even sampling
        rng = np.random.default_rng()
        random_indices = []

        indices = self.cell_selection(dtab=dtab, datasets=datasets, include_invalid=include_invalid, label_source=label_source)
        for ri in range(count_random):
            c = 0
            while c < 100:
                r = rng.integers(0, indices.shape[0], 1)[0]
                if not random_exclude_common_types:
                    break
                cell = dtab.loc[indices[r]]
                # if np.isnan(cell[label_source]):
                #     continue
                cell_type = cell[label_source]
                if cell_type not in ['ON midget', 'OFF midget', 'ON parasol', 'OFF parasol'] and \
                        indices[r] not in random_indices:
                    break
                c += 1
            random_indices.append(indices[r])
        cell_indices.extend(random_indices)

        cell_indices.extend(manual_indices)
        return cell_indices

    # FEATURES
    def register_features(self, features):
        features_dict = {'feature':[], 'provides':[], 'requires':[], 'version':[]}
        index = []
        for feat in features:
            index.append(feat.name)
            features_dict['feature'].append(feat)
            provides = [f'{thing}:{col}' for thing in ['unit','dataset'] for col in feat.provides.get(thing,[])]
            requires = [f'{thing}:{col}' for thing in ['unit','dataset'] for col in feat.requires.get(thing,[])]
            features_dict['provides'].append(provides)
            features_dict['requires'].append(requires)
            features_dict['version'].append(feat.version)

        self.features_table = pd.DataFrame(features_dict, index=index)

    def find_feature_requirements(self, columns, depth_remaining=5): # RECURSION, IT'S NOT JUST FOR YOUR CS 101 HOMEWORK
        if self.features_table.shape[0] == 0:
            self.log('No features registered, do that first', 2)
            return
        features_to_run = set()
        for col_combo in columns:
            # col could be 'sta'
            thing, col = col_combo.split(':')
            print(f'checking for {thing} : {col}')
            if thing == 'unit' and col in self.unit_table.columns:
                continue
            if thing == 'dataset' and col in self.dataset_table.columns:
                continue
            for thing in ['dataset','unit']:
                useful_feature_indices = self.features_table.index[self.features_table.provides.map(lambda provides: col_combo in provides)]
                print(f'got useful features {useful_feature_indices}')
                if len(useful_feature_indices) == 0:
                    self.log(f'No feature found to generate {col}', 2)
                    return
                features_to_run.add(self.features_table.loc[useful_feature_indices[0], 'feature'])

                if depth_remaining > 0:
                    cols_required_by_that_feature = self.features_table.loc[useful_feature_indices[0], 'requires']
                    features_for_those_cols = self.find_feature_requirements(cols_required_by_that_feature, depth_remaining-1)
                    features_to_run.update(features_for_those_cols)
                else:
                    self.log('oy, hit recursion limits. Check your features for circular dependencies or something weird like that.')

        return features_to_run


    def get_vision_data(self, dataset, load_sta=None, load_ei=None):
        """
        gets an analysis_data object from vision loader for data loading from vision files
        :param dataset: row in the ct.dataset_table
        :param load_sta: enables vision-format STA loading (disable for time saving)
        :param load_ei: enables vision-format EI loading (disable for time saving)
        :return:
        """

        load_analysis_data = len(dataset['path']) > 0
        if load_analysis_data:
            # load_labels = len(dataset['labels']) > 0
            load_labels = dataset['labels'] == 'vision'
            ei_path = dataset['ei_path']
            load_long_ei = os.path.isfile(ei_path)
            if load_ei is None:
                load_ei = not load_long_ei
            if load_sta is None:
                load_sta = dataset['stimulus_type'] == 'whitenoise' and dataset['sta_path'] == ''
            if load_sta:
                if len(dataset['sta_path']) > 0:
                    load_sta = False

            # load_labels = False
            # print(f'trying EI path {ei_path}')

            print(f'Loading vision data (thanks Eric), using load_sta {load_sta}, load_labels {load_labels}, load_ei {load_ei}, load_long_ei {load_long_ei}')

            try:
                run_file_name = dataset['run_file_name']
            except:
                # load basic unit info
                run_id = dataset['run_id']
                if re.match(r"^\d{3}$", run_id):
                    run_file_name = f"data{run_id}"
                else:
                    run_file_name = run_id

            a = vl.load_vision_data(dataset['path'],
                                    run_file_name,
                                    include_params=load_labels,
                                    # needs to be True to load cell types from vision
                                    include_ei=load_ei,
                                    include_sta=load_sta,
                                    include_neurons=True)
            # except AssertionError as E:
            #     print(f"Loading files failed for {dataset['path']} data{dataset['run_id']}")
            #     print(E)
            #     a = None

            # read longer EI from alexandra's scratch
            if load_ei and load_long_ei:
                print(f'... Loading long-EI file from {ei_path}')
                p = Path(ei_path)
                ei_path = str(p.parents[0])
                # print(f'ei_path redone: {ei_path}')
                # print('copying globals file')

                import shutil
                globals_path = dataset['path'] + f'/{run_file_name}.globals'
                shutil.copyfile(globals_path, ei_path + f'/{run_file_name}.globals')

                # print(a.main_datatable)
                with vl.EIReader(ei_path, run_file_name) as eir:
                    eis_by_cell_id = eir.get_all_eis_by_cell_id()
                    a.add_ei_from_loaded_ei_dict(eis_by_cell_id, restrict_to_existing_cells=True)
                    a.set_electrode_map(eir.get_electrode_map())
                    a.set_disconnected_electrodes(eir.get_disconnected_electrodes())
                # print('Done loading long EI')
        else:
            print('No analysis data loading')
            a = None

        return a


    def reset_features(self, dataset_index=None):
        """
        Resets the "valid_columns" entries for units and datasets so you can re-generate features without
        having to force them
        :param dataset_index: index for row in ct.dataset_table
        :return:
        """
        if dataset_index is None:
            l = self.dataset_table.index
        else:
            l = [dataset_index]
        for di in l:
            self.dataset_table.loc[di, 'valid_columns_unit'] = file_handling.wrapper({'unit_id'})
            self.dataset_table.loc[di, 'valid_columns_dataset'] = file_handling.wrapper({'piece_id','run_id'})
        self.log('Reset all features')

    def generate_features(self,unit_indices='all',
                          features_to_activate_per_dataset=(),
                          features_to_activate_overall=(),
                          drop_invalid_units=False,
                          drop_big_columns=False,
                          big_columns_per_dataset=None,
                          big_columns_overall=None,
                          force_features=False,
                          load_analysis_data=True,
                          ignore_errors=False,
                          autosave_interval=None):
        """
        generates features as requested. Handles errors, looping, etc
        :param unit_indices: indices of units to run on, use 'all' for all
        :param features_to_activate_per_dataset: List of Features that get generated for each dataset
        :param features_to_activate_overall: List of features that are generated once for all units
        :param drop_invalid_units:
        :param drop_big_columns:
        :param big_columns_per_dataset:
        :param big_columns_overall:
        :param force_features:
        :param load_analysis_data:
        :param ignore_errors:
        :param autosave_interval:
        :return:
        """

        if isinstance(unit_indices, str) and unit_indices == 'all':
            unit_indices = np.array(self.unit_table.index)
        if big_columns_per_dataset is None:
            big_columns_per_dataset = ['ei', 'ei_grid', 'sta', 'sta_var']
        if big_columns_overall is None:
            big_columns_overall = []

        self.log('Activating per-dataset features {}'.format([str(f) for f in features_to_activate_per_dataset]))

        features_active = set()
        # process per-dataset features
        datasets_to_process = np.unique(self.unit_table.loc[unit_indices, 'dataset_id'])
        # datasets_to_process = [1]

        tim = Timer(start=True, count=len(datasets_to_process))

        failures = {'di':[],'feature':[]}

        if len(features_to_activate_per_dataset) > 0:
            load_sta = False
            load_ei = False
            for feat_class in features_to_activate_per_dataset:
                if 'ei' in feat_class.input:
                    load_ei = True
                if 'sta' in feat_class.input:
                    load_sta = True

            for dd, di in enumerate(datasets_to_process):
                dataset = self.dataset_table.loc[di]
                # display(dataset)

                self.log(f'\n\nGenerating features for dataset {di}, {dd+1} of {len(datasets_to_process)}')

                if load_analysis_data:
                    analysis_data = self.get_vision_data(dataset, load_sta=load_sta, load_ei=load_ei)
                    inpt = {'analysis_data': analysis_data}
                else:
                    inpt = dict()

                for fi, feat_class in enumerate(features_to_activate_per_dataset):
                    feature = feat_class()
                    self.log('Feature: {}'.format(feature))

                    # get all valid indices in this dataset
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        indices = self.unit_table.loc[di].index

                    indices = pd.MultiIndex.from_product([[di[0]],[di[1]],indices], names=['dataset_id','run_id','unit_id'])
                    indices = indices[self.unit_table.loc[indices,'valid']]
                    if len(indices) == 0:
                        print('No valid units remaining in dataset {}'.format(di))
                        continue

                    # check if results are present already
                    if not force_features:
                        provides_unit = feature.provides.get('unit',set())
                        provides_dataset = feature.provides.get('dataset', set())
                        valid_columns_unit = dataset['valid_columns_unit'].a
                        valid_columns_dataset = dataset['valid_columns_dataset'].a
                        # print(provides_unit)
                        # print(provides_dataset)
                        # print(valid_columns)

                        if provides_unit.issubset(valid_columns_unit) and provides_dataset.issubset(valid_columns_dataset):
                            self.log(f'... Feature results already present; skipping generation')
                            continue

                    # generate feature now
                    try:
                        # print(f'First index: {indices[0]}')
                        feature.generate(self, indices, inpt)
                        features_active.add(feat_class)
                    except Exception as e:
                        self.log('''
                        
                        BAD NEWS
                        FEATURE FAILED
                        SORRY SCIENTISTS
                        
                        %%%%%(((((*-*-*)))))%%%%
                        
                        ''')
                        self.log(f'Failed Feature {str(feat_class)}', 2)
                        failures['di'].append(di)
                        failures['feature'].append(feat_class)
                        if not ignore_errors:
                            raise e

                    valid_count = np.count_nonzero(self.unit_table['valid'])
                    self.log('... feature complete. unit_table has {} ({} valid) entries, with {} columns'.format(
                        self.unit_table.shape[0], valid_count, self.unit_table.shape[1]))

                if drop_big_columns:
                    dropped_cols = []
                    for col in big_columns_per_dataset:
                        if col in self.unit_table.columns:
                            self.unit_table.drop(columns=col, inplace=True)
                            if col in dataset['valid_columns_unit'].a:
                                dataset['valid_columns_unit'].a.remove(col)
                            dropped_cols.append(col)
                    self.log('Dropped columns {}'.format(dropped_cols))

                if autosave_interval is not None:
                    if dd % autosave_interval == 0 and dd > 0:
                        print('\n\n\nAutomatically saving at interval, hopefully this helps')
                        self.file_save()

                tim.tock(dd)
                print(line_break(1))

        else:
            self.log('no per-dataset features are enabled')

        # display(unit_table)
        tim.tock()
        # %% process overall features

        if len(features_to_activate_overall):
            for fi, feat_class in enumerate(features_to_activate_overall):
                feature = feat_class()
                self.log('Enabling: {}'.format(feature))

                # indices = np.array(self.unit_table['valid'], dtype=bool).nonzero()[0]
                indices = self.unit_table.index[self.unit_table['valid']]

                # if not force_features:
                #     results_present = feat.check_results_present(unit_table)
                #
                #     if results_present:
                #         self.log('results already present for this feature')
                #
                #         features_active.add(feat_class)
                #         continue

                # missing = feat.check_requirements(unit_table)
                inpt=dict()
                feature.generate(self, indices, inpt)
                # features_active.add(feat_class)
        else:
            self.log('no overall features are enabled')

        if drop_big_columns:
            dropped_cols = []
            for col in big_columns_overall:
                if col in self.unit_table.columns:
                    self.unit_table.drop(columns=col, inplace=True)
                    dropped_cols.append(col)
            self.log('Dropped columns {} gently'.format(dropped_cols))

        # %%
        if drop_invalid_units:
            drop = self.unit_table.index[np.logical_not(self.unit_table['valid'])]
            self.log('!!! dropping {} units because, well, they were invalid. Probably had no spikes.\n'
                  'This does not update cells or datasets!'.format(len(drop)))
            self.unit_table.drop(index=drop, inplace=True)
            # self.unit_table.reset_index(inplace=True, drop=True)

        # save memory by downcasting the column types to smaller ones
        # column_types = {'spike_duration':'uint32', 'spike_count':'uint32', 'sta_noise_level':'float32', 'sta_signal_level':'float32'}
        # for col in column_types.keys():
        #     if col in self.unit_table.columns:
        #         self.unit_table = self.unit_table.astype({col: column_types[col]}, copy=False)

        tim.tock()
        self.log('\nAll done generating features! Congrats & be well')
        if len(failures['di']) > 0:
            self.log('Errors were:',2)
            display(pd.DataFrame(failures))

    # DISPLAY
    def show_cell_grid(self,
                       cell_ids,
                       dtab=None,
                       plots=None,
                       color_channels=(2, 3, 4),
                       enable_zoom=True,
                       zoom_span=2000,
                       generate_rf_maps=False,# vs loading them, for development use
                       svg_mode=False,
                       extra_columns=(), # supply a list of column titles here and the returned axs will have them at the end
                       dpi_scale=1, size_scale=1,
                       scale_bar_length=200,
                       row_labels=None,
                       transpose_plots=False,
                       split_timecourse=True,
                       spike_waveform_colname=None):
        import features_visual

        if len(cell_ids) == 0:
            self.log('Empty cell index list provided', 2)
            return
        if dtab is None:
            if len(cell_ids[0]) > 2:
                print('Assuming input IDs are units, not cells')
                dtab = self.unit_table.loc[cell_ids]
            else:
                dtab = self.get_cells(cell_ids)
        if plots is None:
            plots = ['rf_maps', 'time_courses', 'acf', 'ei_map']
        if len(cell_ids) > 50:
            self.log('More than 50 cells is probably too many for this plotter', 2)
            return

        # which plots to display
        plot_sta_peak_frame = True
        plot_sig_stixels = 'sig_stixels' in plots
        plot_time_courses = 'time_courses' in plots
        plot_projection_histograms = 'projection_histograms' in plots
        plot_noise_region = 'noise_region' in plots
        plot_acf = 'acf' in plots
        plot_gaussian_fit = 'gaussian_fit' in plots
        plot_rf_maps = 'rf_maps' in plots
        # generate_rf_maps = False # vs loading them
        force_show_rf_maps = 'force_show_rf_maps' in plots  # even if they are not good_rf
        plot_rf_threshold_variables = 'rf_threshold' in plots
        plot_ei_map = 'ei_map' in plots
        plot_ei_profile = 'ei_profile' in plots
        plot_ei_contours = 'ei_contours' in plots
        plot_ei_contours_on_sta = 'ei_contours_on_sta' in plots
        plot_rf_contours_on_sta = 'rf_contours' in plots
        plot_spike_waveform = 'spike_waveform' in plots

        # which colors to display for RF map
        color_map_names = ('pc2 ON', 'pc2 OFF', 'pc0 ON', 'pc0 OFF', 'pc1 ON', 'pc1 OFF')
        # color_channels = range(6) # all colors
        # color_channels = [1,2,3,4] # RED OFF, GREEN ON+OFF, BLUE ON
        # color_channels = [2, 3, 4, 5] # GREEN ON, OFF

        # span_rf_noise = 16 # human 1
        # span_rf_noise = 60 # human 2
        span_rf_noise = 1000
        contrast_multiplier = 1.2  # for display only
        # noise_multiplier = 10 # set RF contour threshold from noise level
        noise_multipliers = np.logspace(np.log10(1.5), np.log10(10), 10)
        # threshold_sta_mult_by_cell = {416: 0.5, 48: 0.5, 648: 0.3, 364: 0.75, 211: 0.75} # human 1 xmas
        # threshold_sta_offset_by_cell = {289: -4}
        # threshold_ei_mult_by_cell  = {412: .8, 504:.8, 2:.5, 18:.3, 185: 1.4, 330: 1.5, 73:.6}
        threshold_ei_mult_by_cell = {}
        # contour_size_threshold_ei = 100
        # contour_size_threshold_sta = 0

        rotate_colors = True

        # if rotate_colors:
        #     color_names = ('+L-M', '+M', '+S')
        # else:
        #     color_names = ('Red', 'Green', 'Blue')

        # indices = np.array(dtab.index)
        indices = cell_ids
        self.log('Analyzing & plotting {} cells'.format(len(indices)))
        # self.log('Display colors {}'.format(', '.join([color_map_names[a] for a in color_channels])))

        # ^^
        # SET UP FIGURES
        # __
        plot_count = plot_sta_peak_frame + plot_sig_stixels + plot_time_courses + plot_gaussian_fit + \
                     plot_projection_histograms + plot_acf + plot_ei_map + plot_spike_waveform + \
                     plot_noise_region + len(color_channels) * (plot_rf_maps + plot_rf_threshold_variables) + \
                     plot_ei_profile + len(extra_columns)

        fig, axs = plt.subplots(len(indices), plot_count, figsize=[2.3 * plot_count * size_scale, 3 * len(indices) * size_scale], dpi=100*dpi_scale)
        if len(indices) == 1:
            axs = axs[np.newaxis, :]
        if plot_count == 1:
            axs = axs[:, np.newaxis]


        if svg_mode:
            for ax in axs.flat:
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                ax.spines['bottom'].set_visible(False)
                ax.spines['left'].set_visible(False)

        # fig.set_facecolor([.7,.7,.7])
        fig.set_facecolor('w')
        # if plot_ei_contours_on_sta or plot_rf_contours_on_sta:
        #     plt.figure(figsize=[1,1])
        #     contour_axis = plt.axes() # temp axis for storing contour output

        # LOOP THROUGH CELLS (figure rows)
        for ci, index in enumerate(indices):
            col = -1
            cell = dtab.loc[index]

            # print('cell', cell)
            # if cell.shape[0] > 1:
            #     print('Selecting first result, this means duplicate rows')
            # print(type(cell))
            if len(cell.shape) > 1:
                cell = cell.iloc[0]
                print('Selecting first result, this means duplicate rows or a weird table thing')
            di = (cell.piece_id, cell.run_id)
            # print(cell.shape)
            uid = cell.unit_id
            dataset = self.dataset_table.loc[cell.dataset_id]
            cell_type = f"m:{cell['label_manual_text']}"
            try:
                cell_type += f", a:{cell['label_auto_text']}"
            except:
                pass
            self.log('Cell {}, type {}'.format(cid_to_str(index, uid), cell_type))
            if row_labels is not None:
                self.log(row_labels[ci])
            if not dataset['stimulus_type'] == 'whitenoise':
                self.log('stimulus not whitenoise')
                continue
            # try:
            #     cell_type = ct.pdict['label_manual_uniquenames'][int(cell['label_manual'])]
            # except:
            #     cell_type = 'unlabeled'
            # fit_named = name_fit_params(cell['par_fit_sta'].a)
            # print('peak',cell['map_sta_peak'])
            try:
                sta_peak = cell['map_sta_peak'].a
            except:
                print('No STA peak map found')

            try:
                timecourses = cell['tc'].a
            except:
                print('No time courses found')
            try:
                S = cell['sta'].a
                S_dims = S.shape
            except:
                S = None
                S_dims = [sta_peak.shape[0], sta_peak.shape[1], timecourses.shape[0], timecourses.shape[1]]
            # S_var = cell['mv_sta_var'].a


            # Rotate color space to maximize separation of channels (should be a multi-component model of spiking responses)
            if rotate_colors and S is not None:
                primary_colors = dataset['primary_colors'].a
                primary_colors = primary_colors[np.argmax(primary_colors, axis=0), :]  # rearrange
                # self.log(primary_colors)
                # primary_colors = primary_colors[(0,1,2),:]
                # rotate S into new primary colors
                S_rot = S.copy()
                for i in range(S.shape[3]):
                    S_rot[..., i] = np.tensordot(S, primary_colors[i, :], (3, 0))
                S = S_rot
            else:
                primary_colors = np.diag([1, 1, 1])
            # if ci == 0:
            #     self.log('using primary colors {}'.format(primary_colors))

            display_colors = make_display_colors(primary_colors)

            # spike_count = cell['spike_count']
            # self.log('\n** cell {}, type {}, spikes {}'.format(indices[ci], cell_type, spike_count))
            # if spike_count < 1000:
            #     self.log('cell {} low spike count {}'.format(indices[ci], spike_count))
            #     continue

            stimulus_params = dataset['stimulus_params'].a
            stixel_size = stimulus_params['stixel_size']
            # ic(stixel_size, stimulus_params['frame_time'])
            x_l = np.arange(S_dims[0]) * stixel_size
            y_l = np.arange(S_dims[1]) * stixel_size
            time_l = -1 * np.flip(np.arange(S_dims[2])) * stimulus_params['frame_time']
            # x_l, y_l, X_l, Y_l, time_l = make_sta_dimensions(S, stimulus_params, ct.pdict)
            # x_l, y_l, time_l = ct.pdict['x_l'], ct.pdict['y_l'], ct.pdict['time_l']
            # signal_time_range = time_l > -.150
            # frame_count = np.sum(signal_time_range)

            ### LOAD OR CALC SIG STIXELS
            # alpha_sigstix = 1e-8 # from feature definition
            try:
                sig_stixels = cell['map_sig_stixels'].a
            except:
                print('No sig stixels map found')
            # sig_stixels = calculate_sig_stixels(S, 9, alpha_sigstix, color_channels_significant)[0]

            ### FIND STA SPATIAL ZOOM REGION
            xrange_zoom, yrange_zoom = spatial_zoom_region(sig_stixels, zoom_span, stixel_size)

            ### SPATIAL RF at peak frame w/ contours added later
            if plot_sta_peak_frame:
                # peak_frame_index = np.argmax([np.max(np.abs(S[...,fi,:])) for fi in range(S.shape[2])])
                plt.sca(axs[ci, (col := col + 1)])
                axis_sta = axs[ci, col]
                # frames = np.clip(peak_frame_index + np.array([-1,0,1]), 0, S.shape[2]-1)
                # peak_frame_composite = np.mean(S[:,:,frames,:], 2)
                peak_frame_composite = cell['map_sta_peak'].a

            # calibration bar
            if enable_zoom:
                x = [xrange_zoom[0] + 50, xrange_zoom[0] + scale_bar_length + 50]
                # y_base = np.max([yrange_zoom[0], 0])
                y_base = yrange_zoom[0]
                y = [y_base + 100, y_base + 100]
            else:
                x, y = [100, scale_bar_length + 100], [100, 100]
            plt.plot(x, y, linestyle='-', color='white', linewidth=1)

            # SET UP ROW LABEL
            if not svg_mode:
                row_label = '{}\n{}'.format(cell_type.replace(', ','\n'), cid_to_str(index, uid))
                try:
                    row_label += '\nfrate: {:.0f}, SNR {:.0f}'.format(0, cell['sta_snr'])
                    if row_labels is not None:
                        row_label += '\n' + row_labels[ci]
                except:
                    pass
                # row_label = '\n'.join(textwrap.wrap(row_label, 15))
                axs[ci, 0].annotate(row_label, xy=(0.02, 0.5), xytext=(-plt.gca().yaxis.labelpad - 20, 0),
                                    xycoords=plt.gca().yaxis.label, textcoords='offset points',
                                    size='large', ha='right', va='center')

            ### Display SIG STIXELS MAP
            if plot_sig_stixels:
                plt.sca(axs[ci, (col := col + 1)])
                show(sig_stixels, 'auto', None, stimulus_params, interpolation='nearest')
                plt.title('count {}'.format(np.sum(sig_stixels)))
                # plt.title(str([xcenter, ycenter]))
                col_title(axs[ci, col], ci, 'Significant Stixels\nfrom stixel energy')
                # plt.xlim(xrange_zoom)
                # plt.ylim(yrange_zoom)
                plt.xticks([])
                plt.yticks([])

            ### DISPLAY TIME COURSE
            axis_tc = None
            if plot_time_courses:
                time_courses = cell['tc_all'].a
                plt.sca(axs[ci, (col := col + 1)])
                scale = np.max(np.abs(time_courses))

                # axs[ci,col].axvline(0, color='k', linestyle=':')
                # axs[ci,col].axhline(0, color='k')
                if split_timecourse:
                    for coli in range(3):
                        axs[ci, col].axhline(coli, color='k', linestyle='-')
                        plt.plot(time_l, time_courses[:,coli*2] / scale * 0.8 + coli, color=['red','green','blue'][coli])
                        plt.plot(time_l, time_courses[:, coli * 2 + 1] / scale * 0.8 + coli, ':',
                                 color=['red', 'green', 'blue'][coli])
                else:
                    axs[ci, col].axhline(0, color='k', linestyle='-')
                    plt.plot(time_l, np.mean(time_courses[:,2:4], axis=1) / scale, color='k')
                plt.yticks([])
                plt.xlim([-50/120, 0])
                # plt.yticks([0,1,2],['R','G','B'])
                col_title(axs[ci, col], ci, 'time course')

                # plt.sca(axs[ci,(col := col + 1)])
                axis_tc = axs[ci, col]

            ## DISPLAY ACF
            if plot_acf:
                plot_func = plt.plot
                try:
                    acf = cell['acf'].a.astype(float)
                    if np.max(acf) > 0:
                        acf /= np.max(acf)
                    x_acf = dataset['acf_bins'].a
                    if x_acf[-1] > 10:
                        plot_func = plt.semilogx
                except:
                    acf = [0]
                    x_acf = [0]
                plt.sca(axs[ci, (col := col + 1)])
                plot_func(x_acf, acf, color='k')
                plt.yticks([])
                # plt.xticks([.02, 1000])
                # plt.title(f'peak {x_acf[np.argmax(acf)]:.0f} ms')
                col_title(axs[ci, col], ci, 'ISI\nhistogram')

            if plot_spike_waveform:
                if spike_waveform_colname is None:
                    for colname in ['spike_waveform_smart','spike_waveform_maxamplitude', 'spike_waveform_maxenergy', 'spike_waveform']:
                        if colname in cell:
                            break
                else:
                    colname = spike_waveform_colname
                try:
                    # print(colname)
                    wave = cell[colname].a
                    wave = center_wave(wave, wave_len=len(wave), wave_center=np.int(len(wave) / 2))
                    t_wave = np.arange(len(wave)) / 20 - 5
                except:
                    wave = [0]
                    t_wave = [0]
                plt.sca(axs[ci, (col := col + 1)])
                plt.plot(t_wave, wave, color='k')
                # plt.xticks([])
                plt.yticks([])
                plt.xlim([-1,5])
                col_title(axs[ci, col], ci, 'Spike Waveform')


            # RF MAPS w/ CONTOURS
            # contour_sets = cell['list_rf_contours'].a
            # label = np.nonzero(ct.pdict['label_manual_uniquenames'] == cell_type)[0][0]
            # indices_type = np.nonzero(np.array(dtab['label_manual'] == label))[0]

            contour_sets = [[] for a in range(6)]
            # rf_maps = cell['map_rf'].a.copy()

            # find region for RF noise calculations
            use_tuned_noise_region = False
            if use_tuned_noise_region:
                ranges = []
                span_rf_noise_stix = int(span_rf_noise / stixel_size)
                for dim in range(2):
                    range_sig_noise = np.arange(S.shape[dim])[np.any(sig_stixels, [1, 0][dim])]
                    center = np.median(range_sig_noise)
                    range_noise = (center - span_rf_noise_stix / 2, center + span_rf_noise_stix / 2)
                    range_noise = np.clip(range_noise, 0, S.shape[0])
                    range_noise = [int(a) for a in range_noise]
                    ranges.append(range_noise)

                noise_region = np.zeros((S.shape[0], S.shape[1]), dtype=bool)
                noise_region[ranges[0][0]:ranges[0][1], ranges[1][0]:ranges[1][1]] = True
                noise_region[sig_stixels] = False
            else:
                noise_region = np.ones((S_dims[0], S_dims[1]), dtype=bool)
                noise_region[sig_stixels] = False

            ### DISPLAY NOISE REGION MAP
            if plot_noise_region:
                plt.sca(axs[ci, (col := col + 1)])
                show(noise_region, 'auto', True, stimulus_params, interpolation='nearest')
                plt.title('count {}'.format(np.count_nonzero(noise_region)))
                # plt.title(str([xcenter, ycenter]))
                col_title(axs[ci, col], ci, 'Noise region for RF map')

            ## GAUSSIAN FITS
            if plot_gaussian_fit:
                plt.sca(axs[ci, (col := col + 1)])
                fit_params_stc, fit_stc, fit_metric = sta_fit(x_l, y_l, time_l, S, time_courses, display=True)
                fit_params_stc = name_fit_params(fit_params_stc)
                self.log(fit_params_stc)

            # NOW IT'S TIME FOR
            # RF MAPS BY COLOR/POLARITY
            # A CLASSIC LOOP
            if plot_rf_maps or plot_rf_contours_on_sta or plot_projection_histograms or plot_time_courses:
                if plot_projection_histograms:
                    if generate_rf_maps:
                        sig_stixels_pca = ndimage.morphology.binary_dilation(sig_stixels, iterations=3)
                        rf_projection_histograms = np.zeros([-1 + self.pdict['projection_bins'].shape[0], 3])
                    else:
                        try:
                            rf_projection_histograms = cell['rf_projection_histograms'].a
                        except:
                            bins = dataset['rf_projection_bins'].a
                            rf_projection_histograms = np.zeros([3, -1 + bins.shape[0]])

                for map_index in color_channels:
                    coli = int(np.floor(map_index / 2))
                    pol = [1, -1][int(map_index % 2)]
                    if pol == 1:
                        thresh_by_color = 0

                    # load data:
                    if not generate_rf_maps:
                        if plot_rf_maps:
                            rf_map = cell['map_rf'].a[..., map_index]
                            good_threshold = cell['rf_threshold'].a[map_index]
                            thresholds = [good_threshold]
                            best_index = 0
                            good_rf = not np.isnan(cell['rf_threshold'].a[map_index])
                            areas = [cell['rf_size'].a[map_index]]
                        # noise_level = cell['rf_noise_level'].a[map_index]
                        if plot_time_courses:
                            time_courses = cell['tc'].a
                            eigenvect = time_courses[:, coli]

                    else:
                        # ********** REGENERATE RF
                        # Functional PCA of each RF channel (to be a real model soon)
                        time_courses = cell['tc'].a
                        eigenvect = time_courses[:, coli]
                        # eigenvect = features_visual.calculate_sta_fpca(S, sig_stixels_pca, coli, time_l)

                        # generate RF map, by projecting STA onto the vector PCA gave us as maximally informative
                        rf_map = pol * np.tensordot(S[..., (coli,)], eigenvect[:, np.newaxis], ((2, 3), (0, 1)))
                        noise_level = stats.median_abs_deviation(rf_map[noise_region], scale='normal')

                        rf_map[rf_map < 0] = 0
                        if noise_level < 0.000000001:
                            noise_level = 1
                        rf_map /= noise_level

                        # &*
                        #### ANALYSIS ACROSS THRESHOLD LEVEL
                        # 8&

                        # threshold = noise_multiplier * threshold_sta_mult_by_cell.get(index, 1)
                        thresholds = noise_multipliers
                        # ic(noise_level, np.max(rf_map), threshold, np.count_nonzero(rf_map > threshold))

                        h, areas, segments, nums_islands, mean_island_area, max_island_areas, \
                        position_variances, start_index, best_index, good_rf = features_visual.calculate_rf_threshold(
                            rf_map, thresholds, stixel_size)
                        # print(h, areas, segments, nums_islands, mean_island_area, max_island_areas)

                        # select only nearby rf spots to keep things compact, reject distant noise

                        if good_rf:
                            good_threshold = thresholds[best_index]
                            # ic(nums_islands[best_index], position_variances[best_index], mean_island_area[best_index])

                        use_rf_boost = True
                        if good_rf and use_rf_boost:
                            rf_map = features_visual.large_region_rf_boost(rf_map, good_threshold, stixel_size)

                        # projection histogram
                        if plot_projection_histograms and pol == -1 and thresh_by_color > 0:
                            hist = features_visual.calculate_projection_histogram(S, coli, eigenvect, thresh_by_color,
                                                                                  noise_region,
                                                                                  self.pdict['projection_bins'])
                            rf_projection_histograms[:, coli] = hist
                    if plot_rf_maps:
                        if good_rf:
                            if generate_rf_maps:
                                thresh_by_color = np.max([thresh_by_color, thresholds[best_index]])
                            rf_map_bool = rf_map >= good_threshold

                            # store contour segments for this map index (color - pol)
                            # contour_sets[map_index] = segments[best_index]
                            contour_sets[map_index] = features_visual.find_contours_and_scale(rf_map, good_threshold,
                                                                                              stixel_size)
                            # self.log('good thresh ci {} coli {} pol {}, {}'.format(index, coli, pol, good_threshold))
                        else:
                            rf_map_bool = np.zeros_like(rf_map)

                    ###### DISPLAY TIME BELOW

                    # draw STA peak frame around green peak time
                    # if plot_sta_peak_frame and coli == 1:
                    #     peak_frame_index = np.argmax(eigenvect)


                    # PLOT resulting params over threshold values
                    if plot_rf_maps or plot_rf_threshold_variables:
                        if good_rf:
                            # ic(thresholds[best_index])
                            thresh_disp_indices = [best_index]
                            thresh_disp_colors = ['r']
                        else:
                            thresh_disp_indices = [2, int(len(noise_multipliers) / 3), len(noise_multipliers) - 6]
                            thresh_disp_colors = ['g', 'yellow', 'magenta']
                        if plot_rf_threshold_variables:
                            plt.sca(axs[ci, (col := col + 1)])

                            plt.loglog(thresholds, h)
                            plt.loglog(thresholds, areas)
                            plt.loglog(thresholds, nums_islands)
                            plt.loglog(thresholds, mean_island_area * 10)
                            # plt.loglog(thresholds, max_island_areas)
                            # plt.loglog(thresholds, position_variances)
                            plt.loglog(thresholds[0:start_index], mean_island_area[0:start_index] * 10, color='k')

                            for ti, thi in enumerate(thresh_disp_indices):
                                axs[ci, col].axvline(thresholds[thi], color=thresh_disp_colors[ti], linestyle='--')

                            col_title(axs[ci, col], ci,
                                      'params over \nthreshold\n{}'.format(color_map_names[map_index]),
                                      block_color=display_colors[map_index, :])

                            if not good_rf:
                                plt.legend(['h', 'area', 'num islands', 'mean area'])

                    # $$$$$$$$$$
                    # Plot RF map
                    #######
                    if plot_rf_maps:
                        plt.sca(axs[ci, (col := col + 1)])
                        col_title(axs[ci, col], ci, 'RF {}'.format(color_map_names[map_index]),
                                  block_color=display_colors[map_index, :])
                        rf_display_mode = 'map'

                        if good_rf or force_show_rf_maps:
                            range_max = np.max([10, np.max(rf_map)])  # keep noise looking low
                            # range_max = np.max(rf_map)
                            if rf_display_mode == 'map':
                                show(rf_map, [0, range_max], False, stimulus_params, interpolation='spline16')
                            elif rf_display_mode == 'bool':
                                show(rf_map >= thresholds[best_index], [0, 1], True, stimulus_params,
                                     interpolation='spline16')
                            elif rf_display_mode == 'expansion bool':
                                show(rf_map_bool, 'auto', True, stimulus_params, interpolation='nearest')

                            plt.set_cmap('viridis')
                            # plt.title('noise:{:.0f} area:{:.0f}'.format(1000 * noise_level, areas[best_index]))
                            # color = lighten_color(display_colors[map_index,:], 0.3)
                        # else:
                        #     axs[ci, col].set_facecolor([.7,.7,.7])

                        # plot contour segments
                        if good_rf or force_show_rf_maps:
                            for ti, thi in enumerate(thresh_disp_indices):
                                segs_regen = features_visual.find_contours_and_scale(rf_map, thresholds[thi],
                                                                                     stixel_size)
                                # segs_regen = cell['rf_contours'].a[map_index]
                                for si, seg in enumerate(segs_regen):
                                    plt.plot(seg[:, 0], seg[:, 1], color=thresh_disp_colors[ti], linewidth=1)

                        # for si, seg in enumerate(segments):
                        # plt.plot(seg[:,0], seg[:,1], color='red')   #display_colors[map_index,:]
                        # plt.plot(seg[:,0], seg[:,1], linestyle=':',color='white')   #display_colors[map_index,:]
                        # plt.plot(seg[:,0], seg[:,1], color=color)

                        # for other_ci in indices_type:
                        #     if other_ci == indices[ci]:
                        #         continue
                        #     segments = dtab.loc[other_ci,'list_rf_contours'].a[map_index]
                        #     for si, seg in enumerate(segments):
                        #         plt.plot(seg[:,0], seg[:,1],color=colors[(other_ci % 20) + 1,:])   #display_colors[map_index,:]
                        if enable_zoom:
                            plt.xlim(xrange_zoom)
                            plt.ylim(yrange_zoom)
                        plt.xticks([])
                        plt.yticks([])

                        if enable_zoom:
                            x, y = [xrange_zoom[0] + 50, xrange_zoom[0] + scale_bar_length + 50], [yrange_zoom[0] + 50,
                                                                                                   yrange_zoom[0] + 50]
                        else:
                            x, y = [100, scale_bar_length + 100], [100, 100]
                        plt.plot(x, y, linestyle='-', color='white', linewidth=1)

                    # display RF contours on EI plot
                    # colors_custom = [[],[],'green','orange']
                    # plt.sca(axis_ei)
                    # for si, seg_ in enumerate(segments):
                    #     seg = seg_.copy()
                    #     seg -= offset
                    #     seg /= scaling_factor
                    #     # plt.plot(seg[:,0], seg[:,1], color='black')
                    #     plt.plot(seg[:,0], seg[:,1], color=colors_custom[map_index])

            ## DISPLAY EVECT PROJECTION HISTOGRAMS
            if plot_projection_histograms:
                # load:
                # rf_projection_histograms = cell['rf_projection_histograms'].a
                # ic(rf_projection_histograms)
                bins = dataset['rf_projection_bins'].a
                bin_centers = np.round((bins[:-1] + bins[1:]) / 2, 4)

                plt.sca(axs[ci, (col := col + 1)])
                for coli in range(3):
                    plt.plot(bin_centers, rf_projection_histograms[coli,:], color=['red', 'green', 'blue'][coli])
                col_title(axs[ci, col], ci, 'projection hist')

            # show all colors contours for this one cell
            ### SPATIAL RF at peak frame w/ contours added later
            if plot_sta_peak_frame:
                # peak_frame_index = np.argmax([np.max(np.abs(S[...,fi,:])) for fi in range(S.shape[2])])

                # peak_frame_index = np.argmax()
                plt.sca(axis_sta)
                # frames = np.clip(peak_frame_index + np.array([-1,0,1]), 0, S.shape[2]-1)
                # peak_frame_composite = np.mean(S[:,:,frames,:], 2)
                show(peak_frame_composite, 'auto', None, stimulus_params, contrast_multiplier=contrast_multiplier,
                     interpolation='nearest')
                # display_fit_ellipse(fit, False)
                # display_fit_ellipse_vision(fit_v)
                # plt.title('{:.0f} spikes'.format(cell['spike_count']))
                col_title(axis_sta, ci, 'STA peak frames\nmean of 3')
                axis_sta.set_aspect('equal')
                # axis_sta.get_xaxis().set_visible(False)
                # axis_sta.get_yaxis().set_visible(False)
                plt.xticks([])
                plt.yticks([])

                if enable_zoom:
                    plt.xlim(xrange_zoom)
                    plt.ylim(yrange_zoom)

                if plot_rf_contours_on_sta:
                    plt.sca(axis_sta)
                    for map_index, segments in enumerate(contour_sets):
                        # if map_index is not 3:
                        #     continue
                        for si, seg in enumerate(segments):
                            # plt.plot(seg[:,0], seg[:,1], color='white', linewidth=0.3)
                            plt.plot(seg[:, 0], seg[:, 1], color=lighten_color(display_colors[map_index, :], 0.3),
                                     linewidth=.9)

            # EI ENERGY
            if plot_ei_map or plot_ei_contours_on_sta:
                for temporal in ['']:#,'_early','_late']:
                    try:
                        ei_energy = cell['map_ei_energy'+temporal].a.copy().astype(np.float32) + 0.01
                    except:
                        continue
                    # threshold_e = 0.5 * threshold_ei_mult_by_cell.get(index, 1)
                    # if index == 710:
                    # threshold_e = 1
                    # thresholds = (threshold_e / 5, threshold_e, threshold_e * 2)
                    thresholds = (np.percentile(ei_energy, 90) * threshold_ei_mult_by_cell.get(index, 1), np.percentile(ei_energy, 99.8))
                    color_by_threshold = ['white', 'red']

                    x_e = np.arange(ei_energy.shape[0])
                    y_e = np.arange(ei_energy.shape[1])

                    # display(ei_energy)
                    pdict_e = self.pdict.copy()
                    pdict_e['x_l'] = x_e
                    pdict_e['y_l'] = y_e
                    # X_e, Y_e = np.meshgrid(x_e, y_e)
                    # X_e = X_e.T
                    # Y_e = Y_e.T
                    # segments_all_ei = calculate_contour(ei_energy, X_e, Y_e, thresholds, contour_axis)[0]
                    segments_all_ei = [measure.find_contours(ei_energy, level=thresholds[i]) for i in range(len(thresholds))]

                    if plot_ei_map:
                        plt.sca(axs[ci, (col := col + 1)])
                        axis_ei = axs[ci, col]
                        show(ei_energy, [-1, np.max(ei_energy)], None, interpolation='bilinear')
                        # plt.set_cmap('gray')
                        plt.xticks([])
                        plt.yticks([])

                        # print(ei_energy.shape)
                        # calibration bar
                        calib_length = 100 * (ei_energy.shape[1] / 900)
                        plt.plot([2,calib_length+2],[2,2],linestyle='-',color='white',linewidth=1)

                        ### plot EI contours
                        if plot_ei_contours:
                            for ssi, segments in enumerate(segments_all_ei):
                                for si, seg in enumerate(segments):
                                    # if geometry.Polygon(seg).area < contour_size_threshold_ei and ssi == 0:
                                    #     continue
                                    plt.plot(seg[:,0], seg[:,1], color=color_by_threshold[ssi], linewidth=1)

                        col_title(axs[ci, col], ci, 'EI energy ' + temporal)
                        #
                        # if index == 18:
                        #     plt.xlim([0,50])
                        #     plt.ylim([-0,125])
                        # if index == 330:
                        #     plt.xlim([-3,40])
                        #     plt.ylim([-25,100])
                        # if index == 73:
                        #     plt.xlim([3,50])
                        #     plt.ylim([-15,100])
                        axis_ei.set_facecolor('black')
                    #
                    # # draw EI contours on STA plot
                    if plot_ei_contours_on_sta and temporal == '_early':
                        plt.sca(axis_sta)
                        for ssi, segments in enumerate(segments_all_ei):
                            for si, seg_ in enumerate(segments):
                                # if geometry.Polygon(seg_).area < contour_size_threshold_ei and ssi == 0:
                                #     continue
                                seg = seg_.copy()
                                # ic(np.mean(seg,axis=0))
                                seg *= scaling_factor
                                seg += offset
                                plt.plot(seg[:, 0], seg[:, 1], color=color_by_threshold[ssi],
                                         linewidth=.6)  # color_by_threshold[ssi]

            if plot_ei_profile:
                plt.sca(axs[ci, (col := col + 1)])
                try:
                    profile = cell['ei_energy_profile'].a
                    profile_std = cell['ei_energy_profile_std'].a * .3
                    decay = cell['ei_energy_profile_decay']
                except:
                    print('no EI profile info found')

                amp = np.linspace(0, 1000, len(profile))
                plt.fill_between(amp, profile - profile_std, profile + profile_std, color='lightgray')
                plt.plot(amp, profile, 'k')
                plt.ylim([0, 1])
                plt.title(f'decay {decay:.4f}')

            for extra_title in extra_columns:
                plt.sca(axs[ci, (col := col + 1)])
                col_title(axs[ci, col], ci, extra_title)

        # fig.tight_layout(pad=0)

        # if svg_mode:
            # svg_name = 'renders/cell table {} - {}.svg'.format(indices, plot_count)
            # fig.savefig(svg_name, format='svg', dpi=100*dpi_scale)
            # self.log('Saved SVG to {}'.format(svg_name))

        return fig, axs


    def show_mosaic(self,
                    cell_ids=tuple(),
                    dtab=None,
                    map_source='rf_map',
                    color_channels=(3,),
                    map_mode='rf strength', # map mode options: 'rf strength', 'color by cell', 'coverage count', 'rf max', 'rf strength colored'
                    enable_display=True,
                    display_contours=True,
                    plot_hull_boundary = False,
                    zoom_to_hull=True,
                    display_scale_bar=True,
                    scale_bar_length = 200, # µm
                    annotations=False, # annotate cell IDs over cells
                    output_svg=False,
                    cell_index_highlight=None,
                    rotation_angle=0,
                    threshold_multiplier=1.0,
                    segment_area_threshold = 80,
                    display_legend_index = False,
                    invert_color_map = False,
                    interpolation_mode='spline16'):

        '''

        :param cell_ids:
        :param dtab:
        :param map_source:
        :param color_channels:
        :param map_mode: 
        :param enable_display:
        :param display_contours:
        :param plot_hull_boundary:
        :param zoom_to_hull:
        :param display_scale_bar:
        :param scale_bar_length:
        :param annotations:
        :param output_svg:
        :param cell_index_highlight:
        :param rotation_angle:
        :param threshold_multiplier:
        :param segment_area_threshold:
        :param display_legend_index:
        :param invert_color_map:
        :param interpolation_mode:
        :return:
        '''
        import features_visual

        # if dtab.shape[0] == 0:
        #     self.log('No cells in table'); return
        # if len(color_channels) == 0:
        #     self.log('Missing color channels'); return
        if len(cell_ids) == 0:
            self.log('Gotta give me a list of cell indices to make a mosaic', 2)
            return
        if dtab is None:
            if len(cell_ids[0]) > 2:
                print('Assuming input IDs are units, not cells')
                dtab = self.unit_table.loc[cell_ids]
            else:
                dtab = self.get_cells(cell_ids)
        if len(cell_ids) > 500:
            self.log('You gave me way too many cell ids', 2)
            return

        rainbow = cm.get_cmap('gist_rainbow')
        color_map_names = ('red ON (pc2)','red OFF (pc2)','green ON (pc0)', 'green OFF (pc0)', 'blue ON (pc1)', 'blue OFF (pc1)')

        # cell_ids = cell_ids
        # self.log('Mosaic processing {} cells, channels {}'.format(len(cell_ids), [color_map_names[a] for a in color_channels]))

        # set up basic variables

        example_cell = dtab.loc[cell_ids[0]]
        di = example_cell.dataset_id
        try:
            stimulus_params = self.dataset_table.loc[di, 'stimulus_params'].a
        except:
            print(f'!!! no stimulus params found for dataset {di}')
            return
        # primary_colors = self.dataset_table.loc[di, 'primary_colors'].a
        # display_colors = make_display_colors(primary_colors)

        if map_source == 'sta_peak':
            input_map = example_cell['map_sta_peak'].a
        elif map_source == 'rf_map':
            input_map = example_cell['map_rf'].a

        if map_source in ['sta_peak', 'rf_map']:
            map_shape = input_map.shape[0:2]
            x_l, y_l, X, Y, time_l = make_sta_dimensions(input_map, stimulus_params)

        elif map_source == 'sta':
            S = example_cell['sta'].a
            x_l, y_l, X, Y, time_l = make_sta_dimensions(S, stimulus_params)
            map_shape = S.shape[0:2]
        else:
            self.log('EI stuff not working atm.')
            return
            ei_map = example_cell['map_ei_energy'].a.copy()
            map_shape = ei_map.shape[0:2]
            x_e = range(ei_map[0])
            y_e = range(ei_map[1])
            pdict_e = self.pdict.copy()
            pdict_e['x_l'] = x_e
            pdict_e['y_l'] = y_e
            X_e, Y_e = np.meshgrid(np.flip(x_e), y_e)
            X_e = X_e.T; Y_e = Y_e.T

        map_coverage = np.zeros([*map_shape[0:2], len(cell_ids)])

        # loop over all cells
        segments_by_cell = []
        for ci, index in enumerate(cell_ids):
            segments_by_cell.append([])
            cell = dtab.loc[index]
            # self.log(f'index {index}')

            # load data for this cell
            if map_source == 'sta_peak':
                input_map = cell['map_sta_peak'].a
            elif map_source == 'rf_map':
                input_map = cell['map_rf'].a
            else:
                input_map = cell[index, 'map_ei_energy'].a.copy()

            if not(input_map.shape[0] == map_shape[0] and input_map.shape[1] == map_shape[1]):
                self.log(f'Error: RF map wrong shape for cell {index}, expected {map_shape} got {input_map.shape[0:2]}. Halting.')
                return

            # for plot_index, color_channels in enumerate(plot_color_channels):
            plot_index = 0
            if map_source in ['sta_peak','rf_map']:
                # if map_source == 'sta_peak': # this code combines multiple color channels, but is turned off
                #     rf_map = np.zero
                # for cc, map_index in enumerate(color_channels):
                #     coli = int(np.floor(map_index / 2))
                #     pol = [1,-1][int(map_index % 2)]
                #
                #     if map_source == 'sta_peak':
                #         threshold = fixed_threshold
                #         rf_map = sta_peak.copy()[...,coli] * pol
                #     else:
                map_index = color_channels[0]
                rf_map_composed = input_map[...,map_index].copy()
                threshold = cell['rf_threshold'].a[map_index] * threshold_multiplier
            else:
                rf_map_composed = input_map
            rf_map = rf_map_composed # / len(color_channels)

            if map_source == 'ei':
                # threshold = np.percentile(rf_map, 100 - 100 * 1/len(cell_ids + 20))
                threshold = np.percentile(rf_map, 99.5)

            rf_bool = (rf_map >= threshold)
            rf_map[np.logical_not(rf_bool)] = 0
            # rf_map[np.logical_not(rf_bool)] *= 0.5  # reduce appearance of the noise outside the RF
            # check the fill fraction to avoid display-spanning RFs
            fill_fraction = np.count_nonzero(rf_bool) / (rf_map.shape[0] * rf_map.shape[1])
            # ic(fill_fraction)

            # find coutour segments
            if np.any(rf_bool) and (fill_fraction < 0.2):
                if rotation_angle != 0:
                    angle = rotation_angles[ci]
                    center = ndimage.center_of_mass(rf_map)
                    center = [int(a)-1 for a in center]
                    rf_map = rotate_image(rf_map, angle, center)
                    rf_map[np.abs(rf_map) < threshold] = 0 # remove floating point noise from rotation

                # ic(map_coverage.shape, rf_map.shape)
                map_coverage[...,ci] += rf_map / np.max(rf_map)
                if map_source in ['sta_peak','rf_map']:
                    segments = features_visual.find_contours_and_scale(rf_map, threshold, stimulus_params['stixel_size'])
                else:
                    segments = calculate_contour(input_map, X_e, Y_e, (threshold,), contour_axis)[0][0]
                # segments = [convex_hull(segments)]
                # centers = []; areas = [];
                segments_filtered = []
                # loop over segments found
                if len(segments):
                    for si, seg in enumerate(segments):
                        if seg.shape[0] < 3:
                            continue
                        poly = geometry.Polygon(seg)
                        # center = poly.centroid
                        area = poly.area
                        # centers.append([center.x, center.y])
                        # areas.append(area)
                        if area > segment_area_threshold:  # use 80 for bubbly
                            segments_filtered.append(seg)
                    # centers = np.array(centers);  areas = np.array(areas)
                # ic(areas)
                segments = segments_filtered
            else:
                segments = []
            segments_by_cell[ci].append(segments)

        # plotting time
        # self.log('...display mosaic')
        # for plot_index, color_channels in enumerate(plot_color_channels):
        plot_index = 0

        # plt.sca(axs[ti,(col := col + 1)])
        # plt.sca(axs[di, ti])
        coverage_count = np.count_nonzero(map_coverage[...,:] > 0, axis=2)
        cell_ids_random = np.random.permutation(len(cell_ids))
        # color_map = rf_map.copy()
        color_map = 1 + np.argmax(map_coverage[..., cell_ids_random], axis=2) # highest strength cell index map
        color_map[coverage_count == 0] = 0

        if map_source == 'ei':
            pdict = pdict_e

        if map_mode == 'color by cell':
            rf_map = color_map
            show(rf_map, 'auto', None, stimulus_params)
            plt.set_cmap('gist_rainbow')
        elif map_mode == 'coverage count':
            rf_map = coverage_count
            show(rf_map, 'auto', True, stimulus_params)
            # plt.clim([0,3])
        elif map_mode == 'rf strength':
            rf_map = np.sum(map_coverage[...,:], axis=2) # sum map
            show(rf_map, 'auto', False, stimulus_params, contrast_multiplier=1.2, interpolation=interpolation_mode)
            plt.set_cmap('gray')

        elif map_mode == 'rf max':
            # rf_map_sorted = np.sort(map_coverage_type[...,plot_index,:], axis=2)
            # rf_map = np.mean(rf_map_sorted[:,:,-2:-1], axis=2)
            rf_map = np.max(map_coverage[...,:], axis=2) # max map
            if invert_color_map:
                rf_map[rf_map > 0] += 0.5
                # print('max',np.max(rf_map))
            show(rf_map, 'auto', False, stimulus_params, contrast_multiplier=1.0, interpolation='spline16')
            if invert_color_map:
                plt.set_cmap('gray_r')
            else:
                plt.set_cmap('gray')

        elif map_mode == 'rf strength colored':
            rf_map = np.mean(map_coverage[...,:], axis=2) # sum map
            color_map_rgb = np.zeros((color_map.shape[0], color_map.shape[1], 3))
            color_map = color_map.astype(np.float)
            color_map *= 1.0 / np.max(color_map)
            for xi in range(color_map.shape[0]):
                for yi in range(color_map.shape[1]):
                    # color_map_rgb[xi,yi] =
                    id = color_map[xi,yi]
                    color = np.array(rainbow(id))
                    c = rf_map[xi,yi] * color[0:3]
                    # if id > 0:
                    #     self.log(id, color, c)
                    color_map_rgb[xi,yi,:] = c
            show(color_map_rgb, 'auto', None, stimulus_params, contrast_multiplier=4, contrast_basis=0, interpolation='nearest')
            # plt.set_cmap('gray')

        ## display cell contours
        segments_type_map_index = []
        if display_contours:
            for ci in np.random.permutation(len(cell_ids)):
                # color = cell_colors[(ci % 20) + 1,:]
                color = rainbow(cell_ids_random[ci] / len(cell_ids))[:3]
                color = list(colorsys.rgb_to_hls(*color))
                color[1] = np.clip(color[1] + 0.2, 0, 1)
                color = colorsys.hls_to_rgb(*color)

                segments = segments_by_cell[ci][plot_index]
                for si, seg in enumerate(segments):
                    label = cell_ids[ci] if si == 0 else None
                    plt.plot(seg[:,0], seg[:,1], color='r' if cell_index_highlight == cell_ids[ci] else color, linewidth=0.75, label=label)
                    if si == 0 and annotations:
                        poly = geometry.Polygon(seg)
                        center = poly.centroid
                        # plt.plot(center.x,center.y,'.',markersize=1, color=colors[(ci % 20) + 1,:])

                        s = ''
                        if 'cell' in annotations:
                            cell_id = cell_ids[ci]
                            s += str(cell_id[1])
                        if 'unit' in annotations:
                            unit_id = self.cid_to_uid([cell_ids[ci]])[0]
                            s += str(unit_id[2])
                        plt.annotate(s,(center.x-60, center.y + np.random.random(1) * 50),color='r' if cell_index_highlight == cell_ids[ci] else 'cyan')
                segments_type_map_index.extend(segments)

            if display_legend_index:
                ax = plt.gca()
                box = ax.get_position()
                ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
                # Put a legend to the right of the current axis
                ax.legend(loc='center left', bbox_to_anchor=(1, 0.5),fontsize=6)

        map_name = ', '.join([color_map_names[i] for i in color_channels])
        # if len(color_channels) == 1:
        #     col_title(axs[ti,col],ti, '{} rotation {}'.format(map_name, rotation_angle), block_color=display_colors[color_channels[0],:])
        # else:
        # if not output_svg:
        #     col_title(axs[col], ti, '{}'.format(map_name))

        # UNIFORMITY
        # calculate hull
        if len(segments_type_map_index):
            hull_polygon = geometry.Polygon(convex_hull(segments_type_map_index))
            map_hull = np.zeros(rf_map.shape[0:2], dtype=bool)
            for xi in range(len(x_l)):
                for yi in range(len(y_l)):
                    point = geometry.Point([x_l[xi], y_l[yi]])
                    if hull_polygon.intersects(point):
                        map_hull[xi,yi] = True

            # display hull boundary
            if plot_hull_boundary:
                coords = hull_polygon.exterior.coords
                x,y = list(zip(*coords))
                plt.plot(x,y,color='white')

        # valid_area = np.flip(map_hull, axis=0)
        # valid_area = np.zeros(S.shape[0:2], dtype=bool)
        # valid_area =
        # x_sel = np.logical_and(x_l > 500, x_l < 800)
        # y_sel = np.logical_and(y_l > 650, y_l < 1000)
        # for xi in range(len(x_l)):
        #     for yi in range(len(y_l)):
        #         if x_sel[xi] and y_sel[yi]:
        #             valid_area[xi,yi] = True
        # if np.any(valid_area):
        #     uniformity = np.count_nonzero(coverage_count[valid_area] == 1) / np.count_nonzero(valid_area)
        #     variance = np.var(rf_map[valid_area])
        #     # print()
        #     print(f'uniformity: {uniformity:.2f} var: {variance:.2f}')
        # else:
        #     print(f'no valid area in hull')
        # plt.xlim((700,2000))
        # plt.ylim((500,1800))

        if len(segments_type_map_index) and zoom_to_hull:
            xy = np.array(hull_polygon.exterior.coords.xy)
            mins = np.min(xy, axis=1)
            maxs = np.max(xy, axis=1)
            add = 50
            plt.xlim((mins[0]-add, maxs[0]+add))
            plt.ylim((mins[1]-add, maxs[1]+add))
            # print(mins, maxs)
        else:
            mins = [0,0]

        ### calibration bar
        if display_scale_bar:
            # lower left horizontal:
            add = 0
            plt.plot([mins[0]-add+50,mins[0]-add+scale_bar_length+50],[mins[1]-add+50,mins[1]-add+50],linestyle='-',color='white',linewidth=1)
            # upper left vertical:
            # plt.plot([mins[0],mins[0]],[maxs[1],maxs[1]-calib_length],linestyle='-',color='white',linewidth=1)

        plt.xticks([])
        plt.yticks([])

        # if plot_index == 0 and not output_svg:
        #     # dataset_name = dataset_table.loc[dataset_indices[di],'sorter']
        #     row_label = '{}\n({} count)'.format(type_name, len(indices))
        #     # row_label = 'Sorter: {}'.format(dataset_name)
        #     row_label = '\n'.join(textwrap.wrap(row_label, 15))
        #     plt.gca().annotate(row_label, xy=(0.02, 0.5), xytext=(-plt.gca().yaxis.labelpad - 20, 0),
        #                 xycoords=plt.gca().yaxis.label, textcoords='offset points',
        #                 size='large', ha='right', va='center')

        # fig.tight_layout(pad=1)
        # if output_svg:
        #     fig.savefig('renders/mosaic {}, {}{}.svg'.format(types[0], map_name, ' rotated' if rotation_angle != 0 else ''), format='svg', dpi=300)
        #     self.log('Saved SVG')

    def find_primary_channel(self, cells):
        rf_ratio_green = []
        rf_ratio_blue = []
        for e in cells['sta_extremes']:
            extremes = e.a
            green_on = np.abs(extremes[2])
            green_off = np.abs(extremes[3])
            blue = np.max(np.abs(extremes[4:6]))
            rf_ratio_green.append((green_on - green_off) / (green_on + green_off + blue))
            rf_ratio_blue.append(blue / np.max([green_on, green_off]))
        rf_ratio_green = np.median(rf_ratio_green)
        rf_ratio_blue = np.median(rf_ratio_blue)
        if rf_ratio_blue > 1:
            channel = 'blue ON'
        else:
            channel = 'green ON' if rf_ratio_green > 0 else 'green OFF'
        return channel

    def make_groups(self, names, indices, channels=None):
        if channels is None:
            channels = [self.find_primary_channel(self.get_cells(ind)) for ind in indices]
        return [[ind, {'name': nam, 'channel': chan}] for ind, nam, chan in zip(indices, names, channels)]

    def show_group_grid(self, cid_lists=None,
                        dtab=None,
                        names=None,
                        indices=None,
                        plots=None,
                        dpi_scale=1,
                        size_scale=1,
                        mosaic_threshold_multiplier=1,
                        enable_mosaic_zoom=False,
                        mosaic_interpolation_mode='spline16',
                        example_interpolation_mode='spline16',
                        mosaic_map_mode='rf strength',
                        mosaic_annotation=None,
                        num_example_cells=1,
                        show_example_ei=False,
                        show_example_cellid=False,
                        zoom_span=1200,
                        show_traces=True,
                        normalize_waveforms=True,
                        svg_mode=False,
                        highlight_cids=None,
                        extra_columns=None,
                        show_pre_rec=False,
                        tc_col='tc_all',acf_col='acf',
                        normalization=None):
        """

        :param cid_lists:
        :param names:
        :param indices:
        :param plots:
        :param dpi_scale:
        :param size_scale:
        :param mosaic_threshold_multiplier:
        :param enable_mosaic_zoom:
        :param mosaic_interpolation_mode:
        :param example_interpolation_mode:
        :param mosaic_map_mode:
        :param mosaic_annotation:
        :param num_example_cells:
        :param show_example_ei:
        :param show_example_cellid:
        :param zoom_span:
        :param show_traces:
        :param normalize_waveforms:
        :param svg_mode:
        :param highlight_cids:
        :param extra_columns: list of titles for the extra columns, access using axs returned
        :param show_pre_rec:
        :param tc_col:
        :return: fig, axs for adding the extra_columns plots
        """

        if cid_lists is None:
            if indices is None:
                print('Need to specify cid_lists or indices')
                return
            if names is None:
                names = [f'group {k}' for k in range(len(indices))]
            cid_lists = self.make_groups(names, indices)
        if highlight_cids is None:
            highlight_cids = []
        if extra_columns is None:
            extra_columns = []
        if plots is None:
            plots = ['example_cell','mosaic','time_course','acf'] #,'projections','nnd','area','correlation_over_distance']

        plot_example_cell = 'example_cell' in plots
        plot_mosaic = 'mosaic' in plots
        plot_time_course = 'time_course' in plots
        plot_acf = 'acf' in plots
        plot_projections = 'projections' in plots
        plot_areas = 'area' in plots
        plot_nearest_neighbor = 'nnd' in plots
        plot_best_correlation = 'best_correlation' in plots
        plot_correlation_over_distance = 'correlation_over_distance' in plots
        plot_waves = 'spike_waveforms' in plots
        plot_ei_profile = 'ei_profile' in plots
        plot_ei_mosaic = 'ei_mosaic' in plots
        show_outlier_grid = False
        color = '#EF553B'
        var_multiplier = 1

        num_analyses = plot_example_cell * num_example_cells * (2 if show_example_ei else 1) + plot_mosaic + plot_time_course + plot_acf + plot_projections + \
                       plot_areas + plot_nearest_neighbor + plot_waves + plot_correlation_over_distance + plot_best_correlation + plot_ei_profile + plot_ei_mosaic + \
                       len(extra_columns)
        num_rows = len(cid_lists)
        fig, axs = plt.subplots(nrows=num_rows, ncols=num_analyses, figsize=[num_analyses * 2.5 * size_scale + 3, num_rows * 2.5 * size_scale], dpi=60 * dpi_scale)
        if num_rows == 1 and num_analyses == 1:
            axs = np.array([axs])
        if num_rows == 1:
            axs = axs[np.newaxis,:]
        outliers = []
        inliers = []
        # timecourses = pd.Series(dtab.tc.apply(lambda entry: entry.a[:,1:3].flatten(order='F')))
        # display(timecourses)

        for tt, dat in enumerate(cid_lists):
            col = -1
            cell_ids, typ = dat
            num_cells = len(cell_ids)
            scores = np.zeros([num_cells, 2])
            inliers = []
            outliers.append([])
            # print(f"Examining type {typ['name']}, count {num_cells}")

            if num_cells == 0:
                # print(f'No cells included. On to the next group.')
                # inliers.append()
                continue
            if dtab is None:
                cells = self.get_cells(cell_ids)
            else:
                cells = dtab.loc[cell_ids]
            cell_example = cells.iloc[0]
            # display(cells)
            # find mosaic outliers
            # for each cell, check if mosaic conformity is improved with removing it

            # timecourses, find outliers
            di = cell_example['dataset_id']
            dataset = self.dataset_table.loc[di]
            # print(di)
            # if dtab is None:
            #     unit_ids = self.cid_to_uid(cell_ids)
            # else:
            #     unit_ids = cell_ids
            # tc_col = 'tc_norm'
            # print(f'using {tc_col} for tc')
            timecourses = pd.Series(cells[tc_col].apply(lambda entry: entry.a.flatten(order='F')))
            tc = np.stack(timecourses)
            T = self.dataset_table.loc[di]['stimulus_params'].a['frame_time']
            time_l = np.flip(np.arange(timecourses[0].shape[0]/6) * T * -1)

            tc_av = np.median(tc, axis=0)
            tc_var = stats.median_abs_deviation(tc, axis=0) * var_multiplier
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                sc = np.mean(np.abs((tc - tc_av) / tc_var), axis=1)
            scores[:, 0] = sc
            # sc_mn = np.mean(scores[:,0])

            # choose outliers
            # find example inlier (lowest deviation from mean)

            # ACF, find outliers
            if plot_acf:
                acf_log_mode = False
                try:
                    acf = cell_example[acf_col].a.astype(float)
                    if np.max(acf) > 0:
                        acf /= np.max(acf)
                except:
                    acf = [0]
                    x_acf = [0]

                try:
                    x_acf = dataset['acf_bins'].a
                    if x_acf[-1] > 10:
                        acf_log_mode = True
                except:
                    try:
                        x_acf = np.linspace(1, 100, acf.shape[0])
                    except:
                        x_acf = np.linspace(1, 100, len(acf))
                    acf_log_mode = False
                acfs = np.stack([k.a / np.max(k.a) for k in cells['acf']])
                acf_av = np.median(acfs, axis=0)
                acf_var = stats.median_abs_deviation(acfs, axis=0) * var_multiplier
                sel = acf_var > 0
                sc = np.mean(np.abs((acfs[:,sel] - acf_av[sel]) / acf_var[sel]), axis=1)
                # print(acf_av, acf_var, sc)
                scores[:, 1] = sc

            # projection histograms
            if plot_projections:
                projhist = np.stack([k.a[1:3,:].flatten(order='C') for k in cells['rf_projection_histograms']])
                # print(projhist.shape)
                projhist_av = np.median(projhist, axis=0)
                projhist_var = stats.median_abs_deviation(projhist, axis=0) * var_multiplier

            if plot_ei_profile:
                ei_profile = np.stack([k.a / np.nanmax(k.a) for k in cells['ei_energy_profile']])
                ei_profile_av = np.nanmedian(ei_profile, axis=0)
                ei_profile_var = stats.median_abs_deviation(ei_profile, axis=0) * var_multiplier

            # RF sizes and polarities
            if plot_areas:
                area_on = cluster_data_orig_nonorm.loc[cell_ids, 'rf_size_green_on']
                area_off = cluster_data_orig_nonorm.loc[cell_ids, 'rf_size_green_off']

            # choose outliers and one inliers
            inliers.extend(cell_ids[np.argsort(np.mean(scores, axis=1))][:num_example_cells])
            sel_out = np.max(scores, axis=1) > 1.0
            ind_out = cell_ids[sel_out]
            outliers[tt].extend(ind_out)

            # spike correlation analysis
            centers = np.array([[cell['rf_center_x'], cell['rf_center_y']] for ci, cell in cells.iterrows()])
            distances = np.tril(spatial.distance.cdist(centers, centers, 'euclidean'), k=-1)

            if plot_best_correlation or plot_correlation_over_distance:
                # print(cells.iloc[0]['run_id'])
                if cell_example['run_id'] == 'com':
                    # combined mode, need to find the run with the most cells having spikes
                    count_by_run = dict()
                    for ci, cell in cells.iterrows():
                        st = cell['spike_times_by_run'].a
                        for key in st.keys():
                            if key in count_by_run.keys():
                                count_by_run[key] += 1
                            else:
                                count_by_run[key] = 1
                    best_run = ''
                    best_count = 0
                    for key, count in count_by_run.items():
                        if count > best_count:
                            best_run = key
                            best_count = count
                    spikes = [s.a.get(best_run, np.array([0])) / 20000 for s in cells.spike_times_by_run]
                else:
                    spikes = [s.a/20000 for s in cells.spike_times]
                t_stop = np.ceil(np.max([np.max(spikes[cc]) for cc in range(num_cells)]))

                spike_bin_size = .05 * pq.s
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    sts = [SpikeTrain(spikes[cc] * pq.s, t_stop=t_stop) for cc in range(num_cells)]
                    bst = BinnedSpikeTrain([sts[cc] for cc in range(num_cells)], bin_size=spike_bin_size)

                corrs = np.tril(correlation_coefficient(bst), k=-1)
                corrs_filt = corrs.copy()
                corrs_filt[distances < 10] = 0
                best = np.unravel_index(np.argmax(corrs_filt), corrs.shape)
                cch, lags = cross_correlation_histogram(bst[best[0]], bst[best[1]], window=[-75, 75], cross_correlation_coefficient=True)

            # Nearest neighbor distribution
            nnd_bins = np.linspace(0, 800, 20) # if num_cells > 40 else 30)
            dist_filt = distances.copy()
            dist_filt[dist_filt < 1] = np.nan
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                nnd = np.histogram(np.nanmin(dist_filt, axis=1), bins=nnd_bins)[0]



            ################################
            ################################
            ################################
            # Display
            scale_bar_length = 200
            scale_bar_offset = 50
            if plot_mosaic:
                # if 'parasol' in typ['name'] or 'midget' in typ['name']:
                #     tm = mosaic_threshold_multiplier * 4
                # else:
                tm = mosaic_threshold_multiplier
                plt.sca(axs[tt, (col := col + 1)])
                self.show_mosaic(cell_ids=cell_ids, color_channels=channelize([typ['channel']]),
                               zoom_to_hull=enable_mosaic_zoom,interpolation_mode=mosaic_interpolation_mode,
                                 threshold_multiplier=tm, scale_bar_length=scale_bar_length,
                                 annotations=mosaic_annotation, map_mode=mosaic_map_mode)
                plt.gca().axis('off')
                if not svg_mode:
                    plt.title(f'count {len(cell_ids)}')
                if not svg_mode:
                    col_title(plt.gca(), tt, f'mosaic')

            if plot_ei_mosaic:
                plt.sca(axs[tt, (col := col + 1)])

                contour_size_threshold_ei = 5
                segments_by_cell = []

                for ci, cell in cells.iterrows():
                    # cell = ct.get_cell(cell_id)
                    ei_energy = cell['map_ei_energy'].a.copy().astype(np.float32) + 0.01
                    threshold = np.max(ei_energy) * 0.7
                    segments = measure.find_contours(ei_energy, level=threshold)

                    for si, seg in enumerate(segments):
                        # print(geometry.Polygon(seg).area)
                        if geometry.Polygon(seg).area < contour_size_threshold_ei:
                            continue
                        plt.plot(seg[:, 0], -1 * seg[:, 1] + ei_energy.shape[1], color='k', linewidth=1)
                plt.xlim([0, ei_energy.shape[0]])
                plt.ylim([0, ei_energy.shape[1]])

                plt.yticks([])
                plt.xticks([])
                plt.gca().set_aspect('equal')
                col_title(plt.gca(), tt, f'EI mosaic')

            # if show_manual_mosaic:
            #     plt.sca(axs[tt, (col := col + 1)])
            #     self.show_mosaic(cell_ids=cells_manual_ind, color_channels=channelize([typ['channel']]),
            #                    zoom_to_hull=True, threshold_multiplier=tm, scale_bar_length=200)
            #     plt.title(f'count {len(cells_manual_ind)}')
            #     col_title(plt.gca(), tt, 'mosaic manual labels')

            variance_fill_color = [.7,.7,.7]

            # timecourses
            if plot_time_course:
                plt.sca(axs[tt, (col := col + 1)])
                for coli in [2,3,4]:
                    offset = (coli - 2) * -1
                    sel = np.arange(tc.shape[1])
                    sel = np.logical_and(sel >= coli * tc.shape[1]/6, sel < (coli + 1) * tc.shape[1]/6)
                    # if coli == 1:
                    #     sel = sel < tc.shape[1]/2
                    # else:
                    #     sel = sel >= tc.shape[1]/2

                    scale = np.percentile(np.abs(tc), 99) * 2
                    plt.gca().axhline(offset, color='lightgray')
                    if show_traces:
                        for cc, ci in enumerate(cell_ids):
                            if ci in highlight_cids:
                                plt.plot(time_l, tc[cc, sel]/scale+offset, linewidth=2, color='lightgreen')
                            else:
                                plt.plot(time_l, tc[cc,sel]/scale+offset, linewidth=0.3)
                        plt.plot(time_l, (tc_av[sel]+tc_var[sel])/scale+offset, linewidth=1, color='k')
                        plt.plot(time_l, (tc_av[sel]-tc_var[sel])/scale+offset, linewidth=1, color='k')
                    plt.plot(time_l, tc_av[sel]/scale+offset, linewidth=2, color=['r','r','g','g','b','b'][coli])
                    if not show_traces:
                        plt.fill_between(time_l, (tc_av[sel]-tc_var[sel])/scale+offset, (tc_av[sel]+tc_var[sel])/scale+offset, facecolor=variance_fill_color)
                plt.xlim([-.35, 0])
                plt.yticks([])
                # if tt < num_rows - 1:
                #     plt.xticks([])
                ax = plt.gca()
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                ax.spines['left'].set_visible(False)
                if not svg_mode:
                    col_title(plt.gca(), tt, 'Temporal RF' + (' (norm)' if tc_col == 'tc_norm' else ''))

            # ACFs
            if plot_acf:
                plt.sca(axs[tt, (col := col + 1)])
                x = x_acf
                if acf_log_mode:
                    plotfun = plt.semilogx
                else:
                    plotfun = plt.plot

                if show_traces:
                    for cc, ci in enumerate(cell_ids):
                        if ci in highlight_cids:
                            plotfun(x, acfs[cc], linewidth=2, color='lightgreen')
                        else:
                            plotfun(x, acfs[cc], linewidth=0.3)

                    plotfun(x, acf_av+acf_var, linewidth=1, color='k')
                    plotfun(x, acf_av-acf_var, linewidth=1, color='k')
                else:
                    plt.fill_between(x, acf_av-acf_var, acf_av+acf_var,facecolor=variance_fill_color)
                plotfun(x, acf_av, linewidth=2, color='red' if show_traces else 'k')
                # plt.xlim([0, 100])
                plt.yticks([])
                # if tt < num_rows - 1:
                #     plt.xticks([])
                # plt.gca().axvline(100, color='gray')
                # plt.gca().axvline(300, color='gray')
                ax = plt.gca()
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                ax.spines['left'].set_visible(False)
                if not svg_mode:
                    col_title(plt.gca(), tt, 'ISI hist' + (' (norm)' if acf_col == 'acf_norm' else ''))

            # waveforms
            if plot_waves:
                # spike waveforms
                if normalize_waveforms:
                    waves = np.stack([center_wave(k.a)/np.max(np.abs(k.a)) for k in cells['spike_waveform_smart']])
                else:
                    waves = np.stack([center_wave(k.a) for k in cells['spike_waveform_smart']])
                # waves = np.stack([center_wave(k.a) for k in dtab.loc[cell_ids, 'spike_waveform']])
                waves_av = np.median(waves, axis=0)
                waves_var = stats.median_abs_deviation(waves, axis=0) * var_multiplier

                plt.sca(axs[tt, (col := col + 1)])
                t_wave = np.arange(waves.shape[1]) / 20 - 5

                if show_traces:
                    for cc, ci in enumerate(cell_ids):
                        if ci in highlight_cids:
                            plt.plot(t_wave, waves[cc], linewidth=2, color='lightgreen')
                        else:
                            plt.plot(t_wave, waves[cc], linewidth=0.3)
                    plt.plot(t_wave, waves_av+waves_var, linewidth=1, color='k')
                    plt.plot(t_wave, waves_av-waves_var, linewidth=1, color='k')
                else:
                    plt.fill_between(t_wave, waves_av-waves_var, waves_av+waves_var, facecolor=variance_fill_color)
                plt.plot(t_wave, waves_av, linewidth=2, color='red' if show_traces else 'k')

                plt.xlim([-.5,3])
                plt.yticks([])
                if tt < num_rows - 1:
                    plt.xticks([])
                # plt.ylim([-1.1, 1.1])
                if not svg_mode:
                    col_title(plt.gca(), tt, 'spike waveform')

            if plot_ei_profile:
                plt.sca(axs[tt, (col := col + 1)])
                x = np.linspace(0, 1000, len(ei_profile_av))
                if show_traces:
                    for cc, ci in enumerate(cell_ids):
                        if ci in highlight_cids:
                            plt.plot(x, ei_profile[cc], linewidth=2, color='lightgreen')
                        else:
                            plt.plot(x, ei_profile[cc], linewidth=0.3)
                    plt.plot(x, ei_profile_av+ei_profile_var, linewidth=1, color='k')
                    plt.plot(x, ei_profile_av-ei_profile_var, linewidth=1, color='k')
                else:
                    plt.fill_between(x, waves_av-waves_var, waves_av+waves_var, facecolor=variance_fill_color)
                plt.plot(x, ei_profile_av, linewidth=2, color='red' if show_traces else 'k')

            # RF projection histograms
            if plot_projections:
                plt.sca(axs[tt, (col := col + 1)])
                x = np.linspace(-1,1,len(projhist[0]))
                for cc, ci in enumerate(cell_ids):
                    if ci in highlight_cids:
                        plt.plot(x, projhist[cc], linewidth=2, color='lightgreen')
                    else:
                        plt.plot(x, projhist[cc], linewidth=0.3)

                plt.plot(x, projhist_av, linewidth=2, color='red')
                plt.plot(x, projhist_av+projhist_var, linewidth=1, color='k')
                plt.plot(x, projhist_av-projhist_var, linewidth=1, color='k')
                # plt.xlim([80,150])
                if not svg_mode:
                    col_title(plt.gca(), tt, 'RF proj hist')


            # zscore distribution (errors scatter)
            # plt.sca(axs[tt, (col := col + 1)])
            # # plt.hist(scores[:,0])
            # plt.scatter(scores[:,0], scores[:,1]) # , c=np.sum(scores, axis=1), cmap='viridis'
            # plt.xlabel('time course')
            # plt.ylabel('ACF')
            # col_title(plt.gca(), tt, 'error (sum norm diff from median)')


            # spatial distribution of correlation
            if plot_correlation_over_distance:
                plt.sca(axs[tt, (col := col + 1)])
                plt.gca().axhline(0, color='black', linewidth=.5)
                d = distances.flatten()
                ord = np.argsort(d)
                d = d[ord]
                corrs = corrs.flatten()[ord]
                sel = d > 0

                plt.plot(d[sel], corrs[sel], '.', markersize=2, color=color)
                # sel_fit = np.logical_and(d > 1, d < 1000)
                # if np.count_nonzero(sel_fit):
                #     ccc = np.polyfit(d[sel_fit], corrs[sel_fit], 2)
                #     corr_fit = np.polyval(ccc, d[sel_fit])
                #     # print(d, corrs, corr_fit)
                #     plt.plot(d[sel_fit], corr_fit)
                # plt.ylim([np.min(corrs), np.percentile(corrs,95)])
                plt.xlim([0, 1500])
                # plt.title(typ)
                if not svg_mode:
                    col_title(plt.gca(), tt, 'Spatial correlation')


            if plot_best_correlation:
                # cross correlation histogram, best
                plt.sca(axs[tt, (col := col + 1)])
                plt.plot(lags*spike_bin_size*1000, cch.flatten(), color=color)
                # plot_cross_correlation_histogram(cch, axs[tt, 1])
                plt.title(f'dist {distances[best]:.0f} um')
                if not svg_mode:
                    col_title(plt.gca(), tt, 'best cross corr hist')

            # plt.ylabel('')

            # plt.sca(axs[tt, (col := col + 1)])
            # plot_corrcoef(corrs, axs[tt, col])
            # # plt.sca(axs[tt,col])
            # col_title(plt.gca(), tt, 'corr coef matrix')

            # nearest neighbor distribution
            if plot_nearest_neighbor:
                plt.sca(axs[tt, (col := col + 1)])
                plt.plot(nnd_bins[:-1], nnd, color=color)
                # plt.title('NND')
                if not svg_mode:
                    col_title(plt.gca(), tt, 'NN distr.')

            # areas scatter plot
            if plot_areas:
                plt.sca(axs[tt, (col := col + 1)])
                plt.scatter(area_on, area_off, color=color)
                plt.xlabel('area ON')
                plt.ylabel('area OFF')
                # plt.xlim([-2, 3])
                # plt.ylim([-2, 3])
                if not svg_mode:
                    col_title(plt.gca(), tt, 'RF area')



            if plot_example_cell:
                for cc in range(num_example_cells):
                    plt.sca(axs[tt, (col := col + 1)])
                    try:
                        cid = inliers[-(cc + 1)]
                    except:
                        continue
                    example_cell = self.get_cell(cid)
                    # print(f'example {cc} {cid}')
                    rf_map = example_cell['map_sta_peak'].a
                    stimulus_params = self.dataset_table.loc[di, 'stimulus_params'].a
                    show(rf_map, 'auto', None, stimulus_params, contrast_multiplier=1,
                         interpolation=example_interpolation_mode)

                    if normalization is not None:
                        zoom_span_scaled = zoom_span / np.sqrt(60000 / normalization.loc[cid[0], 'rf_size_green_hull'][0])
                    else:
                        zoom_span_scaled = zoom_span
                    xrange_zoom, yrange_zoom = spatial_zoom_region(example_cell['map_sig_stixels'].a, zoom_span_scaled, stimulus_params['stixel_size'])
                    plt.xlim(xrange_zoom)
                    plt.ylim(yrange_zoom)
                    # print(xrange_zoom, yrange_zoom)
                    plt.xticks([])
                    plt.yticks([])
                    plt.gca().axis('off')
                    if show_example_cellid:
                        plt.annotate(f'{cid[1]+1}', xy=[0.01, 0.88], xycoords='axes fraction', color='white', fontsize='xx-large')
                    if not svg_mode:
                        col_title(plt.gca(), tt, f'cell {cc} RF')
                    # calibration bar
                    x, y = [xrange_zoom[0] + scale_bar_offset, xrange_zoom[0] + scale_bar_length + scale_bar_offset], [yrange_zoom[0] + scale_bar_offset,
                                                                                         yrange_zoom[0] + scale_bar_offset]
                    plt.plot(x, y, linestyle='-', color='white', linewidth=1)

                    if show_example_ei:
                        plt.sca(axs[tt, (col := col + 1)])
                        ei_map = example_cell['map_ei_energy'].a.transpose()
                        stimulus_params = self.dataset_table.loc[di, 'stimulus_params'].a
                        show(ei_map, 'auto', None, stimulus_params, contrast_multiplier=1,
                             interpolation=example_interpolation_mode)
                        plt.xticks([])
                        plt.yticks([])
                        plt.gca().axis('off')
                        if not svg_mode:
                            col_title(plt.gca(), tt, f'cell {cc} EI energy')

            # row label
            if not svg_mode:
                if show_pre_rec:
                    try:
                        pre, rec, f1 = report.loc[typ['name'], ['precision','recall','f1-score']]
                    except:
                        pre, rec, f1 = 0,0,0

                    row_label = f"{typ['name']}\nacc {f1:.2f}\npre {pre:.2f}  rec {rec:.2f}"
                else:
                    row_label = f"{typ['name']}"
            else:
                row_label = f'{tt}'
            axs[tt, 0].annotate(row_label, xy=(0.02, 0.5), xytext=(-axs[tt, 0].yaxis.labelpad - 20, 0),
                                xycoords=axs[tt, 0].yaxis.label, textcoords='offset points',
                                size='x-large', ha='right', va='center',)

            for extra_title in extra_columns:
                plt.sca(axs[tt, (col := col + 1)])
                col_title(axs[tt, col], tt, extra_title)

        fig.tight_layout()


        # Cell grids of outliers
        if show_outlier_grid:
            plt.show()
            for tt, typ in enumerate(key_types):
                print(f"Examining type {typ['name']}")
                plots = ['rf_maps','time_courses', 'acf','spike_waveform']

                if len(outliers[tt]) > 0:
                    print(f'{len(outliers[tt])} outliers')
                    color_channels = channelize([typ['channel']])
                    # [archetype[typ['name']]] +
                    cell_ids = [inliers[tt]] + outliers[tt][:5]
                    row_labels = ['central example'] + ['outlier'] * (len(cell_ids) - 1)
                    self.show_cell_grid(cell_ids, plots=plots, zoom_span=800, color_channels=color_channels, dpi_scale=0.7, row_labels=row_labels)
                    plt.show()
                else:
                    print('No outliers')

            # if pp == 0:
            #     break

        return fig, axs

    # display(self.unit_table)


    # cells x label_manual_text, label_auto_text, display params, experiment params,


    # %%

    # print(self.pdict['label_manual_uniquenames'])

def show_rf(ct, cell, dataset, normalization=None, zoom_span=1200, show_cellid=False, scale_bar_offset=50, scale_bar_length=200):
    # print(f'example {cc} {cid}')
    rf_map = cell['map_sta_peak'].a
    stimulus_params = dataset['stimulus_params'].a
    show(rf_map, 'auto', None, stimulus_params, contrast_multiplier=1,
         interpolation='nearest')

    if normalization is not None:
        zoom_span_scaled = zoom_span / np.sqrt(60000 / normalization.loc[cell['piece_id'], 'rf_size_green_hull'][0])
    else:
        zoom_span_scaled = zoom_span
    xrange_zoom, yrange_zoom = spatial_zoom_region(cell['map_sig_stixels'].a, zoom_span_scaled,
                                                   stimulus_params['stixel_size'])
    plt.xlim(xrange_zoom)
    plt.ylim(yrange_zoom)
    # print(xrange_zoom, yrange_zoom)
    plt.xticks([])
    plt.yticks([])
    if show_cellid:
        plt.annotate(f'{cell.index[-1] + 1}', xy=[0.01, 0.88], xycoords='axes fraction', color='white', fontsize='xx-large')

    # calibration bar
    x, y = [xrange_zoom[0] + scale_bar_offset, xrange_zoom[0] + scale_bar_length + scale_bar_offset], [
        yrange_zoom[0] + scale_bar_offset,
        yrange_zoom[0] + scale_bar_offset]
    plt.plot(x, y, linestyle='-', color='white', linewidth=1)


def lighten_color(color, amount=0.2):
    color = list(colorsys.rgb_to_hls(*color))
    color[1] = np.clip(color[1] + amount, 0, 1)
    color = colorsys.hls_to_rgb(*color)
    return color

def rotate_image(img, angle, pivot):
    pad_x = [img.shape[0] - pivot[0], pivot[0]]
    pad_y = [img.shape[1] - pivot[1], pivot[1]]
    # ic(pivot, padX, padY)
    img_padded = np.pad(img, [pad_x, pad_y], 'constant')
    img_rotated = ndimage.rotate(img_padded, angle, reshape=False, prefilter=False)
    # img_rotated = img_padded.T
    return img_rotated[pad_x[0] : -pad_x[1], pad_y[0] : -pad_y[1]]


def col_title(ax, ci, title, block_color=None):
    if ci == 0:
        ax.annotate(title, xy=(0.5, 1), xytext=(0,50), xycoords='axes fraction', textcoords='offset points',
                    size='x-large', ha='center', va='baseline')
        if block_color is not None:
            ax.annotate('███', xy=(0.5, .8), xytext=(0,50), xycoords='axes fraction', textcoords='offset points',
                        size='x-large', ha='center', va='baseline', color=block_color)

def make_sta_dimensions(S, params, pdict=None):
    x_l = np.arange(S.shape[0]) * params['stixel_size']
    y_l = np.arange(S.shape[1]) * params['stixel_size']
    X, Y = np.meshgrid(np.flip(x_l), y_l)
    X = X.T
    Y = Y.T
    time_l = -1 * np.flip(np.arange(S.shape[2])) * params['frame_time']

    if pdict:
        pdict['x_l'] = x_l
        pdict['y_l'] = y_l
        pdict['time_l'] = time_l
        pdict['stixel_size'] = params['stixel_size']

    return x_l, y_l, X, Y, time_l


# utility for aligning spike waveforms (or anything else)
# puts the minimum at the wave_center of a total wave_len length
def center_wave(wave, wave_len=201, wave_center=100, offset=None):
    if offset is None:
        offset = np.argmin(wave)
    pad_front = wave_center - offset
    pad_back = wave_len - len(wave) - pad_front

    if pad_front >= 0 and pad_back >= 0:
        wave = np.pad(wave, (pad_front, pad_back), mode='constant')
    elif pad_front >=0:
        wave = np.pad(wave, (pad_front, 0), mode='constant')
        wave = wave[:pad_back]
    elif pad_back >=0:
        wave = np.pad(wave, (0, pad_back), mode='constant')
        wave = wave[-1*pad_front:]
    else:
        wave = wave[-1*pad_front:pad_back]
    return wave



def sta_time_course(S, mask_spatial=None, display=False):

    colors = ['r','g','b']
    st = np.zeros([S.shape[2], S.shape[3]])
    if display:
        plt.figure()
    S_ = S.copy()
    for ci in range(3):
        S_color = S_[:,:,:,ci]
        if np.any(mask_spatial[:]):
            S_color[mask_spatial == False, :] = np.nan
        s = np.nanmean(S_color, 1)
        s = np.nanmean(s, 0)

        if display:
            plt.gca().axhline(color='k')
            plt.plot(s, color=colors[ci])
            plt.title('contrast over time by color, w/ spatial threshold')
        st[:,ci] = s
    return st


    # plt.imshow(m)
    # size over time

    # derivatives? leading edges


    # frame_range = (15, num_frames)
    #%%



def display_sta_frames(S, mask_spatial, frame_range, enable_zoom):
    # display setup

    num_frames_to_show = frame_range[1] - frame_range[0]
    plot_cols = 5
    plot_rows = np.int(np.ceil(num_frames_to_show / plot_cols))
    fig, axs = plt.subplots(plot_rows, plot_cols)
    fig.set_size_inches(16, 8)

    # display contrast from a mean (find numbers)
    scale = 2
    offset = .5

    # spatial zoom to RF
    # enable_zoom = False
    zoom_factor = 2

    x = np.arange(S.shape[0])
    y = np.arange(S.shape[1])

    xrange = [np.min(x), np.max(x)]
    yrange = [np.min(y), np.max(y)]

    if enable_zoom:
        xrange = x[np.any(mask_spatial, 1)]
        xrange = [np.min(xrange), np.max(xrange)]

        yrange = y[np.any(mask_spatial, 0)]
        yrange = [np.min(yrange), np.max(yrange)]

        xcenter = np.mean(xrange)
        xspan = xrange[1] - xrange[0]
        xspan = xspan * zoom_factor
        xrange = (xcenter - xspan / 2, xcenter + xspan / 2)

        ycenter = np.mean(yrange)
        yspan = yrange[1] - yrange[0]
        yspan = yspan * zoom_factor
        yrange = (ycenter - yspan / 2, ycenter + yspan / 2)

    s_disp = S * scale + offset

    # loop through frames of the spatial STA
    for fi in range(frame_range[0], frame_range[1]):
        pi = fi - frame_range[0] # plot index
        # S # X, Y, T, C

        # where to plot this one
        c = np.mod(pi, plot_cols)
        r = np.int(np.floor(pi / plot_cols))

        ax = axs[r, c]
        plt.sca(ax)

        display_sta(x, y, s_disp[:,:,fi,:])

        plt.xlim(xrange)
        plt.ylim(yrange)

        # NUM_SIGMAS = 1.7 # MAGIC ??
        # tilt = np.array(sta_fit.rot) * (180 / np.pi) * -1
        # fit = patches.Ellipse(xy = (sta_fit.center_x, sta_fit.center_y), width= (NUM_SIGMAS * sta_fit.std_x),
        #       height= (NUM_SIGMAS * sta_fit.std_y),angle=tilt,lw=0.5)
        # ax.add_artist(fit)
        # fit.set_facecolor('None')
        # fit.set_edgecolor('red')

        plt.title('frame: {}, {} ms'.format(fi, '??'))

    plt.show()


# SHOW A 2D IMAGE PLOT in STANDARD COORDINATES
# lots of nice features
def show(a, clims=None, cbar=True, stimulus_params=None, contrast_multiplier=1, contrast_basis=0.5, interpolation='spline16', extent=None):
    # a: input matrix, X by Y, or X by Y by Color. Should be zero-mean
    # clims: set color map limits, like [0, 1], or set it to 'auto', default is [-1, 1]
    # cbar: boolean, enables color map legend bar
    # pdict: the usual parameters dictionary. Here we use x_l and y_l for plot scaling
    # contrast_multiplier & contrast_basis: change how the zero-mean STA is converted to colors/intensities
    # interpolation: fed to plt.imshow so see the doc options there
    #args are pretty self-explanatory
    #todo: add auto zoom w/ sigstix

    # if clims is None:
    #     clims = [-1, 1]
    if clims == 'auto' or clims is None:
        clims = [np.nanmin(a[:]),np.nanmax(a[:])]

    # k = np.flip(np.swapaxes(a, 0, 1), 1)
    k = np.swapaxes(a, 0, 1)

    if stimulus_params is not None:
        x = np.arange(a.shape[0]) * stimulus_params['stixel_size']
        y = np.arange(a.shape[1]) * stimulus_params['stixel_size']
        # rotate_colors = pdict.get('rotate_colors', False)
    else:
        x = range(a.shape[0])
        y = range(a.shape[1])
        # rotate_colors = False

    if extent is None:
        if a.shape[0] == a.shape[1] and a.shape[0] == 1:
            extent = [0,1,0,1]
        else:
            extent = [x[0], x[-1], y[0], y[-1]]
    # print(extent)

    # if rotate_colors:
    #     k = k[...,[2,0,1]]

    if k.ndim == 3:
        # scale the image to put the peak at 1 or -1 (this doesn't invert it)
        if np.max(k) > np.max(-1 * k): # positive bias
            plt.imshow(np.clip(k / np.max(k) * 0.5 * contrast_multiplier + contrast_basis, 0, 1), origin='lower', extent=extent, interpolation=interpolation)
        else:
            plt.imshow(np.clip(k / -np.min(k) * 0.5 * contrast_multiplier + contrast_basis, 0, 1), origin='lower', extent=extent, interpolation=interpolation)
        background_color = [contrast_basis,contrast_basis,contrast_basis]

    elif k.ndim == 2:
        plt.imshow(k * contrast_multiplier, origin='lower', extent=extent, interpolation=interpolation)
        if cbar:
            plt.colorbar()
        plt.set_cmap('viridis')
        plt.clim(clims)
        cmap = plt.cm.get_cmap('viridis')
        background_color = cmap(clims[0])

    # plt.gca().invert_yaxis()
    plt.gca().set_aspect('equal')
    plt.gca().set_facecolor(background_color)
    plt.xticks([])
    plt.yticks([])

# cell_colors = np.clip(gb.generate_palette(size=21+1) + 0,0,1)
cell_colors = \
np.array([[0.00000000e+00, 6.27553565e-16, 5.60784314e-01],
[7.13725490e-01, 0.00000000e+00, 2.46538900e-16],
[7.17204074e-15, 5.49019608e-01, 3.92220978e-16],
[7.64705882e-01, 3.09803922e-01, 1.00000000e+00],
[3.92156863e-03, 6.47058824e-01, 7.92156863e-01],
[9.25490196e-01, 6.15686275e-01, 3.13776782e-16],
[4.62745098e-01, 1.00000000e+00, 0.00000000e+00],
[3.49019608e-01, 3.25490196e-01, 3.29411765e-01],
[1.00000000e+00, 4.58823529e-01, 5.96078431e-01],
[5.80392157e-01, 0.00000000e+00, 4.50980392e-01],
[4.30322444e-14, 9.52941176e-01, 8.00000000e-01],
[2.82352941e-01, 3.25490196e-01, 1.00000000e+00],
[6.50980392e-01, 6.31372549e-01, 6.03921569e-01],
[1.43440815e-15, 2.62745098e-01, 3.92156863e-03],
[9.29411765e-01, 7.17647059e-01, 1.00000000e+00],
[5.41176471e-01, 4.07843137e-01, 0.00000000e+00],
[3.80392157e-01, 0.00000000e+00, 6.39215686e-01],
[3.60784314e-01, 8.96505092e-17, 6.66666667e-02],
[1.00000000e+00, 9.60784314e-01, 5.21568627e-01],
[1.43440815e-15, 4.82352941e-01, 4.11764706e-01],
[5.72549020e-01, 7.21568627e-01, 3.25490196e-01]])


def spatial_zoom_region(map_bool, span_distance, stixel_size):
    x = np.arange(map_bool.shape[0]) * stixel_size
    y = np.arange(map_bool.shape[1]) * stixel_size
    # x = pdict['x_l']
    # y = pdict['y_l']
    # xrange_zoom = [np.min(x), np.max(x)]
    # yrange_zoom = [np.min(y), np.max(y)]

    xrange_sig = x[np.any(map_bool, 1)]
    yrange_sig = y[np.any(map_bool, 0)]

    xcenter = np.median(xrange_sig)
    xrange_zoom = (xcenter - span_distance / 2, xcenter + span_distance / 2)
    ycenter = np.median(yrange_sig)
    yrange_zoom = (ycenter - span_distance / 2, ycenter + span_distance / 2)
    return xrange_zoom, yrange_zoom

def convex_hull(segments):
    if len(segments) == 0:
        return []
    points = []
    for seg in segments:
        points.extend(seg)

    points = np.array(points)
    convex_hull = spatial.ConvexHull(points)
    convex = points[convex_hull.vertices,:]
    convex = np.vstack([convex, convex[0,:]])
    return convex



def calculate_contour(rf_map, X, Y, contour_levels, contour_axis):
    plt.sca(contour_axis)
    contour_axis.cla()

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        contours = plt.contour(X, Y, rf_map, levels=contour_levels)

    area = np.array([np.count_nonzero(rf_map.flatten() >= thresh) for thresh in contour_levels])
    num_segments = np.array([len(contours.collections[i].get_segments()) for i in range(len(contours.collections))])
    # optim = area * .1 + num_segments * 20 + contour_level_multipliers * contour_threshold_penalty # region count
    # optim = area + contour_level_multipliers * contour_threshold_penalty # area in contours
    # best_index = np.argmin(optim)
    #
    # if np.all(num_segments == 0):
    #     print('no seg')
    # if len(contours.collections) == 1:
    #     return None, None, None, None
    segments_all = [contours.collections[i].get_segments() for i in range(len(contour_levels))]

    median_distance = np.zeros_like(contour_levels) * np.nan
    # loop over thresholds
    for threshi in range(len(contour_levels)):
        centers = []
        # loop over segments found
        segments = segments_all[threshi]
        if len(segments) < 2:
            continue
        for seg in segments:
            if seg.shape[0] == 1: # segments can have 2 or 1 point somehow
                continue
            try:
                center = geometry.Polygon(seg).centroid
                # print(center)
                centers.append([center.x, center.y])
            except:
                pass
                # print('err!')
        centers = np.array(centers)
        distances = spatial.distance.pdist(centers)
        median_distance[threshi] = np.median(distances)

    return segments_all, num_segments, area, median_distance


def render_ei(ct, uid, y_dim=50, ei=None, use_electrode_selection=True):
    di = uid[0:2]
    electrode_map = ct.dataset_table.at[di, 'ei_electrode_locations'].a
    if use_electrode_selection:
        sel_electrodes = ct.dataset_table.at[di, 'ei_electrode_selection'].a
    else:
        sel_electrodes = np.ones(electrode_map.shape[0]) * True
    # electrode_map = electrode_map[sel_electrodes]
    xrange = (np.min(electrode_map[:, 0]), np.max(electrode_map[:, 0]))
    yrange = (np.min(electrode_map[:, 1]), np.max(electrode_map[:, 1]))
    x_dim = int((xrange[1] - xrange[0])/(yrange[1] - yrange[0]) * y_dim) # x dim is proportional to y dim
    x_e = np.linspace(xrange[0], xrange[1], x_dim)
    y_e = np.linspace(yrange[0], yrange[1], y_dim)
    grid_x, grid_y = np.meshgrid(x_e, y_e)
    grid_x = grid_x.T
    grid_y = grid_y.T

    if ei is None:
        if len(uid) == 3:
            ei = ct.unit_table.at[uid, 'ei'].a
        else:
            ei = ct.get_cell(uid)['ei'].a
    if len(ei.shape) == 1:
        # print('got a single-frame EI, adding a new dimension')
        ei = ei[:, np.newaxis]
    num_electrodes = ei.shape[0]
    num_frames = ei.shape[1]
    # print('... formatting EI energy maps: {}/{} electrodes, {} time frames, to rectangular {} by {}'.format(np.count_nonzero(sel_electrodes), num_electrodes, num_frames, len(x_e), len(y_e)))
    frames = []
    for fi in range(num_frames):
        ei_frame =  griddata(electrode_map, ei[:, fi], (grid_x, grid_y), method='linear', fill_value=0)
    # polarities = np.sign(ei_frame)
    # ei_frame = np.log10(np.abs(ei_frame)) * polarities

        frames.append(ei_frame)
    if num_frames == 1:
        movie = frames[0]
    else:
        movie = np.stack(frames, axis=0)
    return movie

def show_ei_frames(movie, num_frames, crop_to_peak=False, crop_width=50, frame_limits=None, scale=3):
    plot_cols = 3
    plot_rows = int(np.ceil(num_frames / plot_cols))
    fig, axs = plt.subplots(plot_rows, plot_cols, figsize=[6,3], dpi=150)

    if frame_limits is None:
        frame_limits = [0, movie.shape[0]-1]
        if crop_to_peak:
            peak_frame = np.argmax(np.max(np.abs(movie), axis=(1,2)))
            # print(peak_frame)
            frame_limits = [int(np.clip(peak_frame + a, 0, np.inf)) for a in [-crop_width/2, crop_width/2]]
            # print(frame_limits)

    frames = np.linspace(frame_limits[0], frame_limits[1], num_frames, dtype=int)
    limits = [np.nanmin(movie[frame_limits[0]:frame_limits[1]]) / scale, np.nanmax(movie[frame_limits[0]:frame_limits[1]]) / scale]

    for fi, frame in enumerate(frames):
        plt.sca(axs.flat[fi])
        show(movie[frame], cbar=False)
        plt.xticks([]); plt.yticks([])
        plt.title(frame, fontsize='xx-small')
        plt.clim(limits[0], limits[1])
        plt.set_cmap('Greys_r')
    plt.show()


def analyze_fourier_ei(x_ei, y_ei, EI, mask_ei_variance, enable_display=True):

    peak_bias = np.abs(np.max(EI, 2)) - np.abs(np.min(EI, 2))

    freq_over_space = np.abs(np.fft.rfft(EI, 100))
    freq_pref = np.argmax(freq_over_space, 2)
    max_freq = np.max(freq_pref[:])

    maxabs_over_space = np.max(np.abs(EI), 2)

    if enable_display:
        plt.figure()
        plt.plot(EI[30,30,:])
        plt.xlabel('time')

        plt.figure()
        plt.plot(np.abs(freq_over_space[60,30,:]))
        plt.xlabel('frequency')


        plt.figure()
        display_ei(x_ei, y_ei, freq_pref)
        cbar = plt.colorbar()
        cbar.ax.set_ylabel('fund freq', rotation=-90, va="bottom")
        plt.title('fundamental frequency')

        plt.figure()
        display_ei(x_ei, y_ei, peak_bias)
        plt.colorbar()
        plt.title('positive peak bias')

        plt.figure()
        temp = freq_pref.copy()
        temp[np.logical_not(mask_ei_variance)] = 0
        display_ei(x_ei, y_ei, temp)
        plt.title('fundamental frequency')

    #%% separate low freq areas from high frequency areas


    freq_threshold = 4 # could be found using a bimodal distribution splitter

    f_p = freq_pref.copy()
    f_p[mask_ei_variance == False] = -1

    if enable_display:
        display_ei(x_ei, y_ei, f_p)

    freq_pref_soma = f_p.copy()
    freq_pref_axon = f_p.copy()
    freq_pref_dendrites = f_p.copy()

    # combine masks w/ freq map
    # peak bias: max positive less max abs negative

    freq_pref_soma[np.logical_not(np.logical_and(peak_bias < 0, freq_pref < freq_threshold))] = -1
    freq_pref_dendrites[np.logical_not(np.logical_and(peak_bias > 0, freq_pref < freq_threshold))] = -1
    freq_pref_axon[np.logical_not(freq_pref > freq_threshold)] = -1

    mask_soma = freq_pref_soma > 0
    mask_dendrites = freq_pref_dendrites > 0
    mask_axon = freq_pref_axon > 0

    maxabs_soma = maxabs_over_space.copy()
    maxabs_soma[np.logical_not(mask_soma)] = -1
    maxabs_dendrites = maxabs_over_space.copy()
    maxabs_dendrites[np.logical_not(mask_dendrites)] = -1
    maxabs_axon = maxabs_over_space.copy()
    maxabs_axon[np.logical_not(mask_axon)] = -1

    if enable_display:

        fig, axs = plt.subplots(3, 2, )
        fig.set_size_inches(30, 30)
        plt.sca(axs[0, 0])
        display_ei(x_ei, y_ei,freq_pref_soma)
        plt.colorbar()
        plt.title('soma (low freq, neg wave)')
        plt.clim(0, max_freq)

        plt.sca(axs[1, 0])
        display_ei(x_ei, y_ei,freq_pref_dendrites)
        plt.colorbar()
        plt.title('dendrites (low freq, positive wave)')
        plt.clim(0, max_freq)

        plt.sca(axs[2, 0])
        display_ei(x_ei, y_ei,freq_pref_axon)
        plt.colorbar()
        plt.title('axon (high freq)')
        plt.clim(0, max_freq)

        plt.sca(axs[0, 1])
        display_ei(x_ei, y_ei, maxabs_soma)
        plt.title('soma (low freq, neg wave)')
        plt.colorbar()

        plt.sca(axs[1, 1])
        display_ei(x_ei, y_ei, maxabs_dendrites)
        plt.title('dendrites (low freq, positive wave)')
        plt.colorbar()

        plt.sca(axs[2, 1])
        display_ei(x_ei, y_ei, maxabs_axon)
        plt.title('axon (high freq)')
        plt.colorbar()

        plt.tight_layout()

    return freq_over_space, maxabs_soma, maxabs_dendrites, maxabs_axon
    #%%

    # soma_regions_by_cell[:,:,cell_index] = map_maxabs_soma
    # print(soma_regions_by_cell[1,1,30])
    # # pretty(np.nanmax(soma_regions_by_cell, 2))
    # plt.figure()
    # display_ei(x_ei, y_ei, np.nanmax(soma_regions_by_cell, 2))

def line_break(N=None):
    if N == 1:
        return '_/"-._/"-._/"-._/"-._/"-._/"-._/"-._/"-._/"-._/"-._/"-.'
    elif N == 2:
        return '_/~\_/~\_/~\_/~\_/~\_/~\_/~\_/~\_/~\_/~\_/~\_/~\_/~\_/~\_'
    elif N == 3:
        return '^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^'

    return '.:*~*:._.:*~*:._.:*~*:._.:*~*:._.:*~*:._.:*~*:._.:*~*:._.:*~*:.'

def plot_cell_cross_correlations(self,
                            reference_cell_id,
                            cell_ids=tuple(),
                            dtab=None,
                            cell_type_grouping = True,
                            overlap_bounds = [0,1.5],
                            individual_overlay = True,
                            average_overlay = True,
                            plot_individual_cchs = False,
                            spike_bin_size = '10ms',
                            show_cell_type = True,
                            cols = 4):
    plot_cols = cols
    if len(cell_ids) == 0:
        self.log('no cells to plot',2)
        return
    if dtab is None:
        if len(cell_ids[0]) > 2:
            print('Assuming input IDs are units, not cells')
            dtab = self.unit_table.loc[cell_ids]
        else:
            dtab = self.get_cells(cell_ids)
            
    if len(reference_cell_id) > 2:
        print('Assuming reference cell ID is a unit, not a cell')
        ref_cell = self.unit_table.loc[reference_cell_id]
    else:
        ref_cell = self.get_cell(reference_cell_id)
    ref_cchs = ref_cell[f"{spike_bin_size} CCHs"].a
    overlaps = ref_cell['rf_overlaps'].a
    window_size = len(list(ref_cchs.values())[0])
    lags = np.array(range(int(-(window_size-1)/2),int((window_size-1)/2)+1))
    
    if cell_type_grouping:
        types = dtab['label_manual_text'].unique()
        
        valid_types = []
        all_type_cchs = {}
        for ctype in types:
            ctype_cells = dtab.query('label_manual_text == @ctype')
            type_cchs = []
            com_units = [self.cell_table.at[ref_cell.cell_id,'unit_id_wn_combined']]
            for ci, (uid, unit2) in enumerate(ctype_cells.iterrows()):
                if uid not in ref_cchs:
                    continue
                if np.sum(ref_cchs[uid]) == 0:
                    continue
                if overlaps[uid] <= overlap_bounds[0] or overlaps[uid] > overlap_bounds[1]:
                    continue
                if self.cell_table.at[unit2.cell_id,'unit_id_wn_combined'] not in com_units:
                    com_units.append(self.cell_table.at[unit2.cell_id,'unit_id_wn_combined'])
                else:
                    continue
                type_cchs.append(ref_cchs[uid])
            if len(type_cchs) > 0:
                valid_types.append(ctype)
                all_type_cchs[ctype] = type_cchs
            
        if plot_individual_cchs: 
            for ctype in valid_types:
                cols = plot_cols
                type_cchs = all_type_cchs[ctype]
                if len(type_cchs) < cols:
                    cols = len(type_cchs)
                rows = int(np.ceil(len(type_cchs) / cols))
                fig,axs = plt.subplots(rows,cols,figsize=(cols*5,rows*5))
                for ti, cch in enumerate(type_cchs):
                    if rows > 1:
                        axs[ti//cols,ti%cols].plot(lags,cch)
                        axs[ti//cols,ti%cols].set_title(f"{dtab.iloc[ti].name}")
                    elif cols == 1:
                        axs.plot(lags,cch)
                        axs.set_title(f"{dtab.iloc[ti].name}")
                    else:
                        axs[ti%cols].plot(lags,cch)
                        axs[ti%cols].set_title(f"{dtab.iloc[ti].name}")
                if show_cell_type:
                    fig.suptitle(f"Reference cell: {ref_cell.name}, {ref_cell['label_manual_text']}\n"
                             + f"{ctype}, n = {len(type_cchs)}")     
                else:
                    fig.suptitle(f"Reference cell: {ref_cell.name}\n"
                             + f"{ctype}, n = {len(type_cchs)}") 
                fig.tight_layout(rect=[0, 0.03, 1, 0.95])
        else:
            cols = plot_cols
            if len(valid_types) < cols:
                cols = len(valid_types)
            rows = int(np.ceil(len(valid_types) / cols))
            fig, axs = plt.subplots(rows, cols, figsize=[cols*5, rows*5])
            
            for ti, t in enumerate(valid_types):
                type_cchs = all_type_cchs[t]
                for typ_cch in type_cchs:
                    if individual_overlay:
                        if rows > 1:
                            axs[ti//cols, ti%cols].plot(lags, typ_cch, alpha = 0.2)
                        elif cols == 1:
                            axs.plot(lags, typ_cch, alpha = 0.2)
                        else:
                            axs[ti%cols].plot(lags, typ_cch, alpha = 0.2)
                if average_overlay:
                    if rows > 1:
                        axs[ti//cols, ti%cols].plot(lags, np.nanmean(type_cchs, axis=0), 
                                                    color='k', linewidth=1)
                    elif cols == 1:
                        axs.plot(lags, np.nanmean(type_cchs, axis=0), 
                                                    color='k', linewidth=1)
                    else:
                        axs[ti%cols].plot(lags, np.nanmean(type_cchs, axis=0), 
                                                    color='k', linewidth=1)
                if rows > 1:
                    axs[ti//cols, ti%cols].set_title(f"{t}, n = {len(type_cchs)}")
                    axs[ti//cols, ti%cols].set_xlabel('Time Lag(ms)')
                    axs[ti//cols, ti%cols].set_ylabel('Correlation')
                elif cols == 1:
                    axs.set_title(f"{t}, n = {len(type_cchs)}")
                    axs.set_xlabel('Time Lag (ms)')
                    axs.set_ylabel('Correlation')
                else:
                    axs[ti%cols].set_title(f"{t}, n = {len(type_cchs)}")
                    axs[ti%cols].set_xlabel('Time Lag(ms)')
                    axs[ti%cols].set_ylabel('Correlation')
            if show_cell_type:
                plt.suptitle(f"Reference cell: {ref_cell.name}, {ref_cell['label_manual_text']}")
            else:
                plt.suptitle(f"Reference cell: {ref_cell.name}")
            fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    else:
        all_cchs = []
        com_units = [self.cell_table.at[ref_cell.cell_id,'unit_id_wn_combined']]
        for ci, (uid, unit2) in enumerate(dtab.iterrows()):
            if uid not in ref_cchs:
                continue
            if overlaps[uid] < overlap_bounds[0] or overlaps[uid] > overlap_bounds[1]:
                continue
            if self.cell_table.at[unit2.cell_id,'unit_id_wn_combined'] not in com_units:
                com_units.append(self.cell_table.at[unit2.cell_id,'unit_id_wn_combined'])
            else:
                continue
            all_cchs.append(ref_cchs[uid])
        if plot_individual_cchs:
            rows = int(np.ceil(len(all_cchs) / cols))
            fig, axs = plt.subplots(rows, cols, figsize=[cols*5, rows*5])
            for ti, cch in enumerate(all_cchs):
                if rows > 1:
                    axs[ti//cols, ti%cols].plot(lags, cch)
                    axs[ti//cols, ti%cols].set_title(f"{dtab.iloc[ti].name}")
                elif cols == 1:
                    axs.plot(lags, cch)
                    axs.set_title(f"{dtab.iloc[ti].name}")
                else:
                    axs[ti%cols].plot(lags, cch)
                    axs[ti%cols].set_title(f"{dtab.iloc[ti].name}")
        else:
            cols = 1
            rows = 1
            fig, axs = plt.subplots(rows, cols, figsize=[cols*5, rows*5])
            for cch in all_cchs:
                if individual_overlay:
                    axs.plot(lags, cch, alpha = 0.2)
            if average_overlay:
                axs.plot(lags, np.nanmean(all_cchs, axis=0), color='k', linewidth=1)
            axs.set_title(f"n = {len(all_cchs)}")
            axs.set_xlabel('Time Lag(ms)')
            axs.set_ylabel('Correlation')
            if show_cell_type:
                plt.suptitle(f"Reference cell: {ref_cell.name}, {ref_cell['label_manual_text']}")
            else:
                plt.suptitle(f"Reference cell: {ref_cell.name}")
            fig.tight_layout(rect=[0, 0.03, 1, 0.95])
        
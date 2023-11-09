
import sys

import pandas as pd

import file_handling

sys.path.append('../')

import numpy as np
import cell_display_lib as cdl
from scipy import io
from skimage.transform import resize
from sklearn.preprocessing import LabelEncoder
import re
# The Feature is a class machine that does work on sets of cells/records/datasets (in tables) to extract measured features
#   from the neural data for later display and clustering

class Feature:
    name = ''
    requires = dict()
    provides = dict()
    input = set()
    maintainer = 'Sam'
    version = 1

    def __str__(self):
        return f'[{self.name}] v{self.version} by {self.maintainer} provides {self.provides}, requires {self.requires}'

    def check_requirements(self, ct, di):
        # print('checking requirements')
        missing = []

        # check columns
        for thing, table in [['unit', ct.unit_table],['dataset', ct.dataset_table]]:
            for i, req in enumerate(self.requires.get(thing, [])):
                if req not in table.columns:
                    missing.append(req)


        # check dataset valid columns
        valid_cols = ct.dataset_table.at[di, 'valid_columns_unit'].a
        # print(f'valid cols {valid_cols}')
        for col in self.requires.get('unit', []):
            if col not in valid_cols:
                missing.append(col)

        if missing:
            return missing
        else:
            return None

    def check_results_present(self, dtab):
        for pro in self.provides:
            if pro not in dtab.columns:
                return False
        return True

    def update_valid_columns(self, ct, di):
        ct.dataset_table.at[di, 'valid_columns_unit'].a.update(self.provides.get('unit',set()))
        ct.dataset_table.at[di, 'valid_columns_dataset'].a.update(self.provides.get('dataset', set()))

    def clear(self, dtab):
        dtab.drop(columns=self.provides)
        return dtab


def spelling_correct_type(type):
    """Corrects some common spelling errors in the type column of the unit table"""
    fixes = {'amacinre':'amacrine','Midget':'midget','Parasol':'parasol','midgets':'midget','parasols':'parasol','Blue':'blue',
             'on ':'ON ','off ':'OFF ','On ':'ON ','Off ':'OFF ','sbc':'SBC','SBCs':'SBC','OFF OFF':'OFF','ON ON':'ON','a1':'A1',
             'backpropagatin':'backprop', 'backpropagating':'backprop', 'backpropagation':'backprop', 'backpropogation':'backprop'}
    for f in fixes.keys():
        type = type.replace(f, fixes[f])
    type = re.sub("duplicate of [0-9]+", 'duplicate', type)
    return type



def load_alex_mat(path, verbose=True):
    try:
        matfile = io.loadmat(path)
    except:
        if verbose:
            print(f'ERROR: cannot find alex export mat file at {path}')
        return None, None

    run_ids = np.array([i[0] for i in matfile['dataset_list'][0]])
    labels = [i[0][0] if len(i[0]) > 0 else 'unlabeled' for i in matfile['labels']]
    labels = [i if isinstance(i, str) else 'unlabeled' for i in labels]

    vision_ids = pd.DataFrame(matfile['vision_ids'], columns=run_ids)
    for r in vision_ids.index:
        for c in vision_ids.columns:
            try:
                vision_ids.at[r,c] = vision_ids.at[r,c][0]
            except:
                pass
    vision_ids['label_manual_text'] = labels
    vision_ids.index.rename('alex_id', inplace=True)

    return run_ids, vision_ids


class Feature_load_manual_labels(Feature):
    """
    Load manual labels from various sources
    in dataset:
    'label_data_path' - path to file with labels
    'labels' - mode of labels
    options:
    'alexandra' - load labels from alexandra's matlab file
    'list' - load labels from a list of labels (.txt like from vision)
    'combined' - load labels from a combined dataset
    'vision' - load labels from vision (.params file via analysis_data)
    """
    name = 'load manual labels'
    requires = {'unit':{'unit_id'}}
    provides = {'unit':{'label_manual_text_input'}}
    input = {'analysis_data'}

    def generate(self, ct, unit_indices, inpt, dtab=None):
        if dtab is None:
            dtab = ct.unit_table
        di = unit_indices[0][0:2]
        if (missing := self.check_requirements(ct, di)) is not None:
            print('Feature {}. missing requirements {}'.format(self.name, missing))
            return

        dataset = ct.dataset_table.loc[unit_indices[0][0:2]]
        mode = dataset['labels']

        print('... loading manual labels via mode "{}"'.format(mode))

        if len(mode) == 0:
            print('... Label mode is empty in dataset, filling all "unlabeled"')
            dtab.loc[unit_indices, 'label_manual_text_input'] = 'unlabeled'

        elif mode == 'alexandra':
            path = dataset['label_data_path']
            run_id = dataset['run_id']
            print(f'... label path: {path}')
            run_ids, vision_ids = load_alex_mat(path)

            if run_ids is None:
                print('ERROR: missing manual labels for this dataset, filling empty strings')
                dtab.loc[unit_indices, 'label_manual_text_input'] = ''

            # for cc, ci in enumerate(unit_indices):
            #     label = 'unlabeled'
            #     alex_id = np.nan
            #     for index, row in vision_ids.iterrows():
            #         if ci[2] in row[run_id]:
            #             label = row['label_manual_text']
            #             alex_id = index
            #             # print(ci, label, alex_id)
            #             break

            # process the table row-wise
            for index, row in vision_ids.iterrows():
                for uid in row[run_id]:
                    ci = (dataset['piece_id'], dataset['run_id'], uid)
                    dtab.at[ci, 'label_manual_text_input'] = spelling_correct_type(row['label_manual_text'])
                    dtab.at[ci, 'alex_id'] = index

        elif mode == 'list':
            print('... reading vision-style types in a list document from {}'.format(dataset['label_data_path']))
            vision_ids_this_dataset = np.array(dtab.loc[unit_indices, 'unit_id'])
            file = open(dataset['label_data_path'],'r')

            for line in file:
                # print(line)
                unit_id,label = line.split('  ')
                # print(unit_id, label)
                try:
                    index = np.nonzero(vision_ids_this_dataset == int(unit_id))[0][0]
                    if di[1] == 'com':
                        index -= 1
                except:
                    continue
                label = label.strip('\n')
                label = label.replace('All/', '')
                label = label[:-1]
                label = label.replace('/',' ')
                label = spelling_correct_type(label)
                # print(label)
                # if label == 'All/ON/midget/' or label == 'All/':
                #     label = 'ON midget'
                # if label == 'All/ON/parasol/' or label == 'All/OFF parasol':
                #     label = 'ON parasol'
                # if label == 'All/OFF/midget/' or label == 'All/OFF Midget/':
                #     label = 'OFF midget'
                # if label == 'All/OFF/parasol/':
                #     label = 'OFF parasol'
                # if label == 'All/blue/nc32/':
                #     label = 'christmas'
                dtab.at[unit_indices[index], 'label_manual_text_input'] = label

        elif mode == 'combined':
            print('reading combined dataset export')
            # data = io.loadmat(dataset['label_data_path'])
            # dtab.loc[unit_indices, 'label_manual_text_input'] = [spelling_correct_type(t[0][0]) for t in data['labels']]

            path = dataset['label_data_path']
            print(f'... label path: {path}')
            run_ids, vision_ids = load_alex_mat(path)
            for cc, ci in enumerate(unit_indices):
                label = vision_ids.loc[ci[2], 'label_manual_text']
                vision_id_data000 = vision_ids.loc[ci[2], '000']
                if len(vision_id_data000) > 0:
                    vision_id_data000 = int(vision_id_data000[0])
                else:
                    vision_id_data000 = np.nan
                dtab.at[ci, 'unit_id_data000'] = vision_id_data000

                vision_id_data001 = vision_ids.loc[ci[2], '001']
                if len(vision_id_data001) > 0:
                    vision_id_data001 = int(vision_id_data001[0])
                else:
                    vision_id_data001 = np.nan
                dtab.at[ci, 'unit_id_data001'] = vision_id_data001

                dtab.at[ci, 'label_manual_text_input'] = spelling_correct_type(label)


        elif mode == 'vision':
            print('using vision types')
            labels = []
            for ci in unit_indices:
                typ = inpt['analysis_data'].get_cell_type_for_cell(dtab.at[ci, 'unit_id'])
                labels.append(spelling_correct_type(typ))

                # if not isinstance(typ, str):
                # print('typ {}'.format(typ))
                    # typ = ''
            dtab.loc[unit_indices, 'label_manual_text_input'] = labels
        else:
            print('!!! unknown label mode')

        self.update_valid_columns(ct, di)


def simplify_alexandra_labels(labels):
    labels = np.array(labels)
    out = []
    for i in range(len(labels)):
        label = labels[i]
        if 'OFF a ' in label or 'OFF spider amacrine' in label:
            label = 'OFF amacrine'
        if 'ON a ' in label or 'ON branchy amacrine' in label:
            label = 'ON amacrine'
        if 'OFF s ' in label or 'OFF large' in label:
            label = 'OFF RGC'
        if 'ON s ' in label or 'ON large' in label:
            label = 'ON RGC'
        if 'blue ' in label:
            if 'blue a' in label:
                label = 'blue amacrine'
            else:
                label = 'blue RGC'
        if 'ON OFF' in label:
            label = 'ON OFF'
        out.append(label)
    return out

class Feature_simplify_label_text(Feature):
    name = 'simplify label text'
    requires = {'unit':{'label_manual_text_input'}}
    provides = {'unit':set()}
    input = {'analysis_data'}

    def generate(self, ct, unit_indices, inpt, dtab=None):
        if dtab is None:
            dtab = ct.unit_table
        di = unit_indices[0][0:2]
        if (missing := self.check_requirements(ct, di)) is not None:
            print('Feature {}. missing requirements {}'.format(self.name, missing))
            return

        print('... simplifying text labels')
        dtab.loc[unit_indices, 'label_manual_text_input'] = simplify_alexandra_labels(
            dtab.loc[unit_indices, 'label_manual_text_input'])


class Feature_process_manual_labels(Feature):
    name = 'process manual labels'
    requires = {'unit':set()}
    provides = {'unit':set()}
    input = {''}

    def generate(self, ct, unit_indices, inpt, dtab=None):
        if dtab is None:
            dtab = ct.unit_table
        di = unit_indices[0][0:2]
        # if (missing := self.check_requirements(ct, di)) is not None:
        #     print('Feature {}. missing requirements {}'.format(self.name, missing))
        #     return

        if 'label_manual_text_input' in ct.dataset_table.at[di, 'valid_columns_unit'].a:
            source = 'label_manual_text_input'
            output = 'label_manual_input'
        else:
            source = 'label_manual_text'
            output = 'label_manual'
        print(f'...Using label source {source}')

        label_manual_text = dtab.loc[unit_indices,source]
        types_nan = [not isinstance(l, str) for l in label_manual_text]
        label_manual_text[types_nan] = 'wasnan'

        le = LabelEncoder()
        label_manual = le.fit_transform(label_manual_text)
        label_manual_uniquenames = le.classes_
        ct.pdict['label_manual_uniquenames'] = label_manual_uniquenames
        num_types = len(label_manual_uniquenames)
        dtab.loc[unit_indices, output] = label_manual
        print(f'... Found {num_types} cell types, storing label_manual_uniquenames in ct.pdict')

        self.update_valid_columns(ct, di)


class Feature_load_dataset_metadata(Feature):
    name = 'load dataset metadata'
    requires = {'dataset':set(), 'unit':set()}
    provides = {'dataset':{'location_eccentricity','location_angle','temperature','display','optics','lens','params_wn'}}
    input = {''}

    fn_runs = '/Volumes/Lab/Users/scooler/database_spreadsheet/database_spreadsheet_runs.csv'
    fn_pieces = '/Volumes/Lab/Users/scooler/database_spreadsheet/database_spreadsheet_pieces.csv'

    def generate(self, ct, unit_indices, inpt, dtab=None):
        if dtab is None:
            dtab = ct.unit_table
        di = unit_indices[0][0:2]
        if (missing := self.check_requirements(ct, di)) is not None:
            print('Feature {}. missing requirements {}'.format(self.name, missing))
            return

        fail = False
        di = unit_indices[0][0:2]
        piece = ct.dataset_table.loc[di,'piece_id']
        run = ct.dataset_table.loc[di,'run_id']
        if len(run) > 3:
            run = run[:3]
        error = False

        table_runs = pd.read_csv(self.fn_runs)
        s = f'datarun == "data{run}" and Piece == "{piece}"'
        try:
            table_runs_row = table_runs.query(s).iloc[0]

            # print(s, table_runs_row)
            temp, display, optics, lens = table_runs_row[
                ['temperature', 'display (CRT, OLED)', 'optics (below, above)', 'objective lens']]
            type, color, interval, rgb, seed, stixel_width, stixel_height = table_runs_row[
                ['type', 'BW/RGB', 'interval', 'rgb', 'seed', 'stixel width', 'stixel height']]

            ct.dataset_table.loc[di, ['temperature','display','optics','lens']] = [temp, display, optics, lens]
            ct.dataset_table.loc[di, 'params_wn'] = file_handling.wrapper({'type':type, 'color':color, 'interval':interval, 'rgb':rgb, 'seed':seed, 'stixel width':stixel_width, 'stixel_height':stixel_height})

        except:
            print(f'Run table query {s} failed')
            fail = True

        table_pieces = pd.read_csv(self.fn_pieces)
        s = f'Date == "{piece}"'
        try:
            table_pieces = table_pieces.query(s).iloc[0]
            eccentricity, angle = table_pieces[['Eccentricity (mm)', 'Angle (clock angle)']]
            ct.dataset_table.loc[di, ['location_eccentricity', 'location_angle']] = [eccentricity, angle]
        except:
            print(f'Piece table query {s} failed')
            fail = True

        if not fail:
            self.update_valid_columns(ct, di)

# class Feature_load_export_labels(Feature):
#     name = 'load export labels'
#     requires = {'unit':{'id_e'}}
#     provides = {'unit':{'label_manual_text'}}
#
#     def generate(self, ct, unit_indices, inpt, dtab=None):
#         if dtab is None:
#             dtab = ct.unit_table
#         dataset = ct.dataset_table.loc[unit_indices[0][0:2]]
#         data = io.loadmat(dataset['label_data_path'])
#         label_data_manual = data['labels']
#         for ci in unit_indices:
#             index_export = dtab.at[ci, 'id_e']
#             lab = label_data_manual[index_export][0][0]
#             dtab.at[ci, 'label_manual_text_input'] = lab
#
#
# class Feature_load_export_data(Feature):
#     name = 'load export data'
#     requires = {'unit':{'id_e'}}
#     provides = {'unit':{'sta','acf'}}
#     input = {'analysis_data'}
#
#     def generate(self, ct, unit_indices, inpt, dtab=None):
#         if dtab is None:
#             dtab = ct.unit_table
#         dataset = ct.dataset_table.loc[unit_indices[0][0:2]]
#
#         spatial_rescale_factor = 0.5
#         temporal_rescale_factor = 1
#         params = {'stixel_size': 2.925 / spatial_rescale_factor, 'frame_time': 0.0083 / temporal_rescale_factor}
#
#         tim = cdl.Timer()
#         tim.tick()
#         for ci in unit_indices:
#             index_export = dtab.at[ci, 'id_e']
#             try:
#                 dtab.at[ci, 'acf'] = file_handling.wrapper(np.array(inpt['analysis_data']['new_data']['acf'][index_export]))
#             except:
#                 pass
#             dtab.at[ci, 'stimulus_params'] = file_handling.wrapper(params)
#             dtab.at[ci, 'stimulus'] = 'white noise'
#
#             S = np.array(io.loadmat(dataset['path'] + 'indiv_stas/cell_{}.mat'.format(index_export+1))['sta']).astype(np.float32)
#             dim = list(S.shape)
#             if len(dim) < 4:
#                 print('STA for cell {} (export ID {}) is too small, marking invalid'.format(ci, index_export))
#                 dtab.at[ci, 'sta'] = None
#                 dtab.at[ci, 'valid'] = False
#                 continue
#             dim[0] *= spatial_rescale_factor
#             dim[1] *= spatial_rescale_factor
#             dim[3] *= temporal_rescale_factor
#             S_resampled = resize(S, dim, anti_aliasing=True, preserve_range=True).swapaxes(2,3)
#             dtab.at[ci, 'sta'] = file_handling.wrapper(S_resampled)
#
#             if ci % 10 == 0:
#                 print('done with {} of {}'.format(index_export, len(unit_indices)))
#                 tim.tock()
#
#
# class Feature_filter_labels(Feature):
#     name = 'filter labels'
#     requires = {'unit':{'label_manual_text'}}
#
#     def generate(self, ct, unit_indices, inpt, dtab=None):
#         if dtab is None:
#             dtab = ct.unit_table
#         keywords = ['crap','contaminated','duplicate']
#         removed_count = 0
#
#         for ci in unit_indices:
#             good = True
#             for word in keywords:
#                 if word in dtab.at[ci, 'label_manual_text']:
#                     good = False
#             # good = np.any([word in cell_type for word in keywords])
#
#             if not good:
#                 dtab.at[ci, 'valid'] = False
#                 removed_count += 1
#
#         print('... found {} cells having bad types, which were marked invalid'.format(removed_count))
#
#
#
# class Feature_assign_combined_datasets(Feature):
#     name = 'assign combined datasets'
#     requires = {'unit':{'id_e'}}
#     provides = {'unit':{'unit_id','dataset_name'}}
#
#     def generate(self, ct, unit_indices, inpt, dtab=None):
#         if dtab is None:
#             dtab = ct.unit_table
#         dataset = ct.dataset_table.loc[unit_indices[0][0:2]]
#         label_data_path = dataset['label_data_path']
#
#         data = io.loadmat(label_data_path)
#         datasets = data['dataset_list'][0]
#
#         for ci in unit_indices:
#             id_e = dtab.at[ci, 'id_e']
#
#             ids_v = data['vision_ids'][id_e]
#             dataset_index = -1
#             unit_id = -1
#             for di, ds in enumerate(ids_v):
#                 if len(ds[0]):
#                     dataset_index = di
#                     unit_id = ids_v[di][0][0]
#                     break
#
#             dtab.at[ci, 'dataset_name'] = datasets[dataset_index][0]
#             dtab.at[ci, 'unit_id'] = unit_id
#
#         print('... assigned datasets')
#
# # class Feature_sta_mosaic(Feature):
# #     name = 'rf mosaic'
# #     requires = {'par_fit_sta'}
# #     provides = {'figure'}
# #
# #     def generate(self, dtab, pdict, inpt=None):
# #         if (missing := self.check_requirements(ct)) is not None:
# #             print('Feature {}. missing requirements {}'.format(self.name, missing))
# #             return
# #
# #

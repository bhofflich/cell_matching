import h5py, sys, os, tqdm
import numpy as np
import pandas as pd

sys.path.append('/Volumes/Lab/Users/bhofflic/mini_rotation/utils/')
import feature_lists
import features as feat
import features_visual as feat_v
import features_electrical as feat_e

import cell_display_lib as cdl

sys.path.append('/Volumes/Lab/Users/scooler/classification/')
sys.path.append('/Volumes/Lab/Development/artificial-retina-software-pipeline/utilities/')

import file_handling
import visionloader as vl
import electrode_map as elmap


def file_save_h5(ct, piece_id, components=('ids', 'labels'),
                 label_version='B', label_types=('auto', 'manual'),
                 fileroot_in=None, fname=None,
                 save_backup_labels=True):
    '''
    starts a new h5 and saves the ids and labels
    unit_table contains: cell_id, columns: ordered unit_id, rows: ordered run_id
    vision_id_table contains: unit_id, columns: ordered unit_id, rows: ordered run_id
    '''
    ct.log(f'Exporting {piece_id}')
    if fname is None:
        if fileroot_in is None:
            fileroot = cdl.get_piece_directory(ct, piece_id)
        else:
            fileroot = fileroot_in

        fname = f'{fileroot}{piece_id}.h5'

    print(f'Overwriting file at {fname}')
    if 'alexth' in fname:
        raise Exception('Error: saving to alexth path')

    with h5py.File(fname, 'w') as h5:
        h5_ids = h5.create_group('IDs')

        # setup IDs and write
        if 'ids' in components:
            datas = []
            datas_ids = []
            unit_id_lists = []
            runs = ct.dataset_table.query(f"piece_id == '{piece_id}' and run_id != 'com'")['run_id']
            if len(runs) > 0:
                runs = np.sort(runs)
                ct.log(f'have {len(runs)} runs: {runs}')
                for ri, run_id in enumerate(runs):
                    ct.log(f'run {run_id}')
                    units = ct.unit_table.query(f"piece_id == '{piece_id}' and run_id == '{run_id}'")
                    unit_ids = units.unit_id
                    unit_id_lists.append(unit_ids)
                    unit_ids_to_ordered = dict(zip(unit_ids, range(len(unit_ids))))

                    data = np.zeros(len(unit_ids))
                    data_id = np.zeros_like(data)
                    ct.log(f'Found {len(data)} units')
                    for uid, unit in units.iterrows():
                        uid_ordered = unit_ids_to_ordered[uid[2]]

                        # add 1 to make it MATLAB friendly, use 0 as missing
                        cid_out = unit['cell_id'][1] + 1 if not np.isnan(unit['cell_id'][1]) else 0
                        data[uid_ordered] = cid_out
                        data_id[uid_ordered] = uid[2]
                    datas.append(data)
                    datas_ids.append(data_id)

                # combine the IDs into 2D matrices
                maxlen = np.max([len(a) for a in datas])
                table = np.zeros([len(runs), maxlen])
                table_ids = np.zeros_like(table)
                for ri in range(len(runs)):
                    table[ri, 0:len(datas[ri])] = datas[ri]
                    table_ids[ri, 0:len(datas_ids[ri])] = datas_ids[ri]

                # write out data to file
                h5_ids['vision_id_table'] = table_ids
                h5_ids['unit_table'] = table
                h5_ids_runs = h5_ids.create_group('runs')
                for ri, run_id in enumerate(runs):
                    h5_ids_run_ = h5_ids_runs.create_group(run_id)
                    h5_ids_run_['unit_ids'] = unit_id_lists[ri]

                h5_meta = h5.create_group('meta')
                h5_meta['data_list'] = np.array([('data'+str(a)).encode('utf-8') for a in runs])
            else:
                # just 'com' present
                ct.log(f'combined run only')
                units = ct.unit_table.query(f"piece_id == '{piece_id}' and run_id == 'com'")
                num_cells = len(units)

                h5_ids['vision_id_table'] = np.arange(num_cells)[:,np.newaxis].transpose() + 1
                # print(h5_ids['vision_id_table'][()])
                h5_ids['unit_table'] = np.arange(num_cells)[:,np.newaxis].transpose() + 1

        # h5_meta['creation_date']
        if 'labels' in components:
            # select all the cells we are exporting
            cell_ids = ct.cell_table.query(f"piece_id == '{piece_id}'").index
            cell_ids_nums = [cid[1] for cid in cell_ids]

            # setup labels and write
            h5_labels = h5_ids.create_group('labels')
            ct.log('Exporting labels:')
            for lt in label_types:
                if not f'label_{lt}_text' in ct.cell_table.columns:
                    ct.log(f'No labels of type "{lt}" found, skipping')
                    continue
                colname = lt
                if 'auto' in colname:
                    colname += f'_{label_version}'

                labels = list(ct.cell_table.loc[cell_ids, f'label_{lt}_text'].fillna('unlabeled'))
                # print(labels)
                h5_labels[colname] = labels
                ct.log(f'exported "{colname}" labels')

                if save_backup_labels:
                    fname_csv = f'{fileroot}{piece_id}_labels_{colname}.csv'
                    labels_pd = pd.Series(labels)
                    labels_pd.to_csv(fname_csv, header=False, index=False)
                    # np.savetxt(fname_csv, np.array(labels), delimiter=',')
                    ct.log(f'saved backup labels to {fname_csv}')

        h5.close()
    ct.log(f'Saved to {fname}')


def file_save_h5_params(ct, piece_id, fname=None,
                        filter_columns=None, filter_runs=None):
    '''
    Save the parameters to an HDF5 file
    :param ct: CellTable instance
    :param piece_id:
    :param fname: h5 file name
    :param filter_columns: select which columns to include
    :param filter_runs: select which runs to include
    :return:
    '''
    if fname is None:
        fname = f'{ct.paths["scratch_analysis"]}/{piece_id}/{piece_id}.unified.h5'
    print(f'Saving to {fname}')
    if 'alexth' in fname:
        raise Exception('Error: saving to alexth path')

    directory = os.path.split(fname)[0]
    if not os.path.exists(directory):
        print(f'Creating directory {directory}')
        os.makedirs(directory)

    # Set up column selection and data types
    columns = ct.unit_table.columns
    columns_filtered = []
    sample = ct.unit_table.iloc[0]
    # print(columns)
    for col in columns:
        if '_id' in col or 'label_' in col:
            continue
        if filter_columns is not None:
            if col not in filter_columns:
                continue
        columns_filtered.append(col)
    print(columns_filtered)
    col_data = []
    for col in columns_filtered:
        data = sample[col]
        col_dict = {'col': col, 'mode': None, 'exported': False, 'to_export': True}

        col_dict['mode'] = None
        if isinstance(data, float):
            # print('float')
            col_dict['mode'] = 'float'
        elif isinstance(data, file_handling.wrapper):
            if isinstance(data.a, list):
                col_dict['mode'] = 'list'
            elif isinstance(data.a, np.ndarray):
                col_dict['mode'] = 'array'
                if 'map_' in col:
                    col_dict['mode_extended'] = 'map'
            elif isinstance(data.a, dict):
                col_dict['mode'] = 'dict'

        elif isinstance(data, bool) or isinstance(data, np.bool_):
            col_dict['mode'] = 'bool'
        else:
            print(type(data))
            assert (False)
        col_data.append(col_dict)

    ctab = pd.DataFrame(col_data)
    ctab.set_index('col', inplace=True)
    # display(ctab)
    print(f'column spec table shape is {ctab.shape}')

    cells = ct.cell_table.query("piece_id == @piece_id")
    utab = ct.unit_table.query("piece_id == @piece_id")
    run_ids = np.sort(utab['run_id'].unique())
    run_id_indices = dict(zip(run_ids, [a + 1 for a in range(len(run_ids))]))
    print(f'Run IDs: {run_id_indices}')
    print(f'cell count: {len(cells)}, unit count: {len(utab)}')

    print(f'Appending to file at {fname}')
    with h5py.File(fname, 'a') as h5:
        h5.require_dataset('version', shape=(1,), data=2, dtype=np.int8)
        h5_params = h5.require_group('params')

        print('Exporting data')
        ctab_floats = ctab.query('mode in ["float", "bool", "array"] and not exported and to_export')
        for col, col_info in ctab_floats.iterrows():

            if col_info['mode'] == 'float':
                dtype = np.float16
            elif col_info['mode'] == 'bool':
                dtype = np.int8
            else:
                dtype = np.float16
            try:
                print(f'exporting column {col}, dtype: {dtype}')
                h5_params_col = h5_params.require_group(col)

                for ci, cell in cells.iterrows():
                    h5_params_col_cell = h5_params_col.require_group(str(ci[1] + 1))
                    for ri, run_id in enumerate(run_ids):
                        if filter_runs is not None:
                            if run_id not in filter_runs:
                                continue

                        uid = cell['unit_ids_by_run'].get(run_id, [])
                        if len(uid) > 0:
                            uid = uid[0]
                            # print(uid)
                            data = utab.loc[uid, col]
                            if isinstance(data, file_handling.wrapper):
                                data = data.a
                            data = np.array(data)
                            # print(data)
                            # if not np.isnan(data):
                            try:
                                data = data.astype(dtype)
                            except:
                                pass
                            run_label = 'combo' if run_id == 'com' else str(run_id_indices[run_id])
                            if run_label in h5_params_col_cell.keys():
                                if h5_params_col_cell[run_label].shape != data.shape:
                                    del h5_params_col_cell[run_label]

                            dataset = h5_params_col_cell.require_dataset(run_label,
                                                                         shape=data.shape,
                                                                         data=data, dtype=dtype)
                ctab.loc[col, 'exported'] = True
            except Exception as e:
                print(f'Error exporting {col}: {e}')

        print('done exporting data to h5')

        h5.flush()
        h5.close()

def file_load_h5_indices(ct, piece_id, fname=None,
                         process_labels=True,
                         use_individual_runs=True,
                         force_data999_labels=True,
                         data999_sorter='kilosort'):
    '''
    Load indexing data from h5 file, can also process data into params
    :param ct:
    :param piece_id:
    :param fname:
    :param process_labels:
    :param use_individual_runs:
    :param force_data999_labels:
    :return:
    '''
    # fname = f'/Volumes/Scratch/Users/scooler/h5/{piece_id}.h5'
    if fname is None:
        fname = f'{ct.paths["scratch_analysis"]}/{piece_id}/new_structure_data.h5'
    print(f"Loading H5 from {fname}")
    with h5py.File(fname, 'r') as h5:
        if 'version' in h5:
            version = h5['version'][0]
        else:
            version = 1
        print(f'h5 format version {version}')
        assert(version in [1,2])

        if use_individual_runs:
            runs = [str(int(a)).zfill(3) for a in list(h5['meta']['data_list'])]
            run_id_reference = runs[0]
        else:
            runs = []
            run_id_reference = None
        runs.append('com')
        print(f'Found {len(runs)} runs: {runs}')

        # load analysis data from runs
        index = []
        dataset_dict = cdl.new_dataset_dict()
        sorter = 'kilosort'
        for run_id in runs:
            path = f'{ct.paths["analysis_root"]}/{piece_id}/{sorter}_data{run_id}/data{run_id}/'
            if not (os.path.isdir(path) or run_id == 'com'):
                print(f'Missing main vision analysis data {path}')
                continue

            if run_id == 'com':
                label_mode, label_data_path, sta_path, ei_path = ['', '', '', '']
                path = ''
            else:
                label_mode, label_data_path, sta_path, ei_path = cdl.make_paths(ct, piece_id, run_id)
            dataset_dict['run_id'].append(run_id)
            if run_id == 'com':
                dataset_dict['run_file_name'].append('')
            else:
                dataset_dict['run_file_name'].append(f'data{run_id}')
            dataset_dict['piece_id'].append(piece_id)
            dataset_dict['note'].append('')
            dataset_dict['path'].append(path)
            dataset_dict['labels'].append(label_mode)
            dataset_dict['sorter'].append(sorter)
            dataset_dict['label_data_path'].append(label_data_path)
            dataset_dict['sta_path'].append(sta_path)
            dataset_dict['ei_path'].append(ei_path)
            dataset_dict['species'].append('')
            dataset_dict['stimulus_type'].append('whitenoise')
            index.append((piece_id, run_id))
        dataset_table = pd.DataFrame(dataset_dict,
                                     index=pd.MultiIndex.from_tuples(index, names=['piece_id', 'run_id']))
        # display(dataset_table)
        # display(cto.dataset_table)
        for ind in dataset_table.index:
            if ind in ct.dataset_table.index:
                raise Exception(f'Run {ind} already in dataset table')
        ct.dataset_table = pd.concat([ct.dataset_table, dataset_table])

        # %% set up cells from h5
        unit_table = h5['IDs']['unit_table']
        sorter_id_table = h5['IDs']['vision_id_table']

        num_cells = int(np.max(unit_table))
        num_units = int(unit_table.shape[1])
        print(f'Piece {piece_id}, found {num_cells} cells, {num_units} max units per run (highest unit_id)')

        # process H5 table
        # setup lists (all by cell index)
        all_unit_ids_by_run = {run_id: [] for run_id in runs}
        all_unit_ids_by_run['com'] = list(range(num_cells))
        unit_id_by_run = []
        unit_ids = []
        for ci in range(num_cells):
            r = {r: [] for r in runs}
            r['com'] = ((piece_id, 'com', ci),)
            unit_id_by_run.append(r)
            unit_ids.append([(piece_id, 'com', ci)])

        # parse H5 units into lists
        for ri, run_id in enumerate(runs[:-1]):  # rows of table
            for unit_index in range(num_units):  # columns of table
                # if run_id == 'com':
                #     cid = unit_index
                #     unit_id = unit_index
                # else:
                cid = int(unit_table[ri, unit_index]) - 1  # cell ID is stored in table
                unit_id = int(sorter_id_table[ri, unit_index])
                # print(unit_index, cid, unit_id)
                if cid == -1:
                    continue

                all_unit_ids_by_run[run_id].append(unit_id)

                unit_id_tuple = (piece_id, run_id, unit_id)
                unit_id_by_run[cid][run_id].append(unit_id_tuple)
                unit_ids[cid].append(unit_id_tuple)

        # add combined
        unit_ids_com = [(piece_id, 'com', uid) for uid in range(num_cells)]

        # convert lists to tuples
        unit_ids = [tuple(ids) for ids in unit_ids]
        unit_id_by_run = [{key: tuple(value) for key, value in d.items()} for d in unit_id_by_run]

        # print(all_unit_ids_by_run)
        # print(unit_id_by_run)
        # print(unit_ids)

        # make cell_table
        cell_table_dict = dict()
        cell_table_dict['unit_ids'] = unit_ids
        cell_table_dict['unit_ids_by_run'] = unit_id_by_run
        cell_table_dict['unit_ids_wn'] = unit_ids
        cell_table_dict['unit_id_wn_combined'] = list(unit_ids_com)
        cell_table_dict['unit_id_nsem'] = [np.nan for ci in range(num_cells)]

        cell_table_dict['corr_included'] = [[] for ci in range(num_cells)]
        cell_table_dict['label_manual'] = [0 for ci in range(num_cells)]
        cell_table_dict['label_manual_text'] = ['unlabeled' for ci in range(num_cells)]

        # cell_table_dict['dataset_id'] = [di for a in unit_ids]
        cell_table_dict['piece_id'] = piece_id
        cell_table_dict['merge_strategy'] = 'h'
        cell_table_dict['valid'] = [True for ci in range(num_cells)]

        # for c in cell_table_dict.keys():
        #     print(c)
        #     v = cell_table_dict[c]
        #     try:
        #         print(len(v))
        #     except:
        #         print(v)

        cell_index_numbers = np.arange(num_cells)
        index = pd.MultiIndex.from_product([[piece_id], cell_index_numbers], names=['piece_id', 'cell_id'])
        cell_table = pd.DataFrame(cell_table_dict, index=index)
        ct.cell_table = pd.concat([ct.cell_table, cell_table])
        # display(cell_table)
        # display(cto.cell_table)

        # %% make per-run units
        unit_tables = []
        for run_id in runs:
            # if run_id == 'com':
            #     sorter_ids_this_dataset = list(range(num_cells))
            # else:
            sorter_ids_this_dataset = all_unit_ids_by_run[run_id]
            index = pd.MultiIndex.from_product(([piece_id], [run_id], sorter_ids_this_dataset),
                                               names=('piece_id', 'run_id', 'unit_id'))
            unit_table = pd.DataFrame({'unit_id': sorter_ids_this_dataset,
                                       'dataset_id': [(piece_id, run_id) for a in
                                                      range(len(sorter_ids_this_dataset))],
                                       'run_id': run_id, 'piece_id': piece_id, 'valid': True,
                                       'cell_id': [np.nan for a in range(len(sorter_ids_this_dataset))],
                                       'label_manual_text': 'unlabeled', 'label_manual': 0}, index=index)
            #
            # display(unit_table)
            unit_tables.append(unit_table)
            ct.dataset_table.at[(piece_id, run_id), 'valid_columns_unit'] = file_handling.wrapper({'unit_id'})

            # add units IDs to the dataset/run
            ct.dataset_table.at[(piece_id, run_id), 'unit_ids'] = file_handling.wrapper(index.to_numpy())
        unit_table = pd.concat(unit_tables)
        ct.unit_table = pd.concat([ct.unit_table, unit_table])
        ct.unit_table['cell_id'] = ct.unit_table['cell_id'].astype('object') # required to store tuples in there

        # record the cell id for each unit, of the cells we just added
        print('adding cell ids to units')
        for ci, cell in cell_table.iterrows():
            unit_ids = cell['unit_ids']
            for uid in unit_ids:
                ct.unit_table.at[uid, 'cell_id'] = ci

    # initialize the valid columns
    for run_id in runs:
        ct.dataset_table.at[(piece_id, run_id), 'valid_columns_unit'] = file_handling.wrapper({'unit_id'})
        ct.dataset_table.at[(piece_id, run_id), 'valid_columns_dataset'] = file_handling.wrapper(set())

    if process_labels:
        # %% load cell types for all units
        if use_individual_runs:
            features_to_generate_by_dataset = [feat.Feature_load_manual_labels]
            features_to_generate_overall = [feat.Feature_process_manual_labels]
            # features_to_generate_overall = []

            # indices = 'all'
            indices = ct.unit_table.query(f"piece_id == '{piece_id}' and run_id != 'com'").index
            # print(indices)

            # drop_big_columns = ct.unit_table.shape[0] > 20000
            ct.generate_features(indices, features_to_generate_by_dataset, features_to_generate_overall,
                                 drop_big_columns=0,
                                 big_columns_per_dataset=[],
                                 force_features=0,
                                 load_analysis_data=1,
                                 ignore_errors=0,
                                 autosave_interval=None)

        # %% generate the rest of the parameters for the combined data (run 'com')
        features_to_generate_by_dataset = [] #feature_lists.features_standard_postcombined

        label_mode, label_data_path, sta_path, ei_path = cdl.make_paths(ct, piece_id, 'com')
        # this might have selected the wrong labels. Override with data999
        sta_path_base = '/Volumes/Scratch/Users/alexth/supersample-stas'
        if data999_sorter == 'kilosort':
            label_data_path_999 = f'{sta_path_base}/{piece_id}/data999/data999.classification.txt'
        elif data999_sorter == 'yass':
            label_data_path_999 = f'{sta_path_base}/{piece_id}/data999_yass_no_duplicates/data999/data999.classification.txt'
        else:
            label_data_path_999 = None

        if force_data999_labels and os.path.exists(label_data_path_999):
            print('forcing data999 labels')
            label_data_path = label_data_path_999
            label_mode = 'list'
        elif force_data999_labels:
            print('tried forcing data999 labels, but file not found, using default labels')
            print(f'path: {label_data_path_999}')

        # if label_mode == 'list':
        print('Loading labels into combined data')
        print(f'label mode "{label_mode}", path {label_data_path}')
        ct.dataset_table.at[(piece_id, 'com'), 'label_data_path'] = label_data_path
        ct.dataset_table.at[(piece_id, 'com'), 'labels'] = label_mode
        features_to_generate_by_dataset.append(feat.Feature_load_manual_labels)

        features_to_generate_overall = [feat.Feature_process_manual_labels]
        # features_to_generate_overall = []

        # indices = 'all'
        indices = ct.unit_table.query(f"piece_id == '{piece_id}' and run_id == 'com'").index
        # print(indices)
        # drop_big_columns = ct.unit_table.shape[0] > 20000
        ct.generate_features(indices, features_to_generate_by_dataset, features_to_generate_overall,
                             drop_big_columns=0,
                             big_columns_per_dataset=[],
                             force_features=0,
                             load_analysis_data=0,
                             ignore_errors=0,
                             autosave_interval=None)

        # %% combine labels into cells
        import deduplication
        deduplication.arrange_units_within_cells(ct)

        # for piece_id in pieces:
        print('sorting out the labels now')
        if label_mode == 'list':
            # copy units to cells, if we loaded Alexandra's stuff
            ct.unit_table['label_manual_text'] = ct.unit_table['label_manual_text_input']
            ct.unit_table['label_manual'] = ct.unit_table['label_manual_text']
            for uid, unit in ct.unit_table.query(f"piece_id == '{piece_id}' and run_id == 'com'").iterrows():
                # print(uid, unit['label_manual_text_input'])
                ct.cell_table.loc[(uid[0], uid[2]), 'label_manual'] = unit['label_manual_input']
                ct.cell_table.loc[(uid[0], uid[2]), 'label_manual_text'] = unit['label_manual_text_input']
        else:
            # copy cells to units
            for cid, cell in ct.cell_table.iterrows():
                ct.unit_table.loc[(cid[0], 'com', cid[1]), 'label_manual'] = cell['label_manual']
                ct.unit_table.loc[(cid[0], 'com', cid[1]), 'label_manual_text'] = cell['label_manual_text']

        ct.dataset_table.at[(piece_id, 'com'), 'valid_columns_unit'] = file_handling.wrapper(
            {'unit_id','label_manual', 'label_manual_text'})
        # setup other runs



    print(f'Done loading h5 for {piece_id}')
    return fname


def file_load_h5_params(ct, piece_id, fname=None, filter_columns=None, filter_runs=None):
    """
    Load the parameters from the h5 file combo data
    :param ct:
    :param piece_id:
    :param fname:
    :return:
    """

    if fname is None:
        fname = f'{ct.paths["scratch_analysis"]}/{piece_id}/{piece_id}.unified.h5'
    print(f"Loading H5 from {fname}")
    with h5py.File(fname, 'r') as h5:
        if 'version' in h5:
            version = h5['version'][0]
        else:
            version = 1
        print(f'h5 format version {version}')
        assert(version in [1,2])

        # runs = [str(int(a)).zfill(3) for a in list(h5['meta']['data_list'])]
        # run_id_reference = runs[0]

        h5_params = h5['params']
        columns = h5_params.keys()
        new_valid_columns = set()
        print(f'found columns: {columns}')

        # col_data = dict()
        unit_ids = ct.unit_table.index

        for col in columns:
            # col_data[col] = {'mode': '', 'imported': False, 'to_import': True}
            # print(col)
            data = h5_params[col]
            col_out = col
            if version == 1:
                if col == 'full_sta':
                    col_out = 'sta'
                elif col == 'full_ei':
                    col_out = 'ei'
                elif col == 'n_spikes':
                    col_out = 'spike_count'
                elif col == 'ei_power':
                    # print('skip ei_power from version 1')
                    col_out = None
                    # col_out = 'map_ei_power'
                elif col == 'mean_spike':
                    col_out = 'spike_rate_mean'
                elif col == 'time_course':
                    col_out = 'tc'
                elif col == 'sta':
                    # print('skip flat STA from version 1')
                    col_out = None
                elif col == 'contours':
                    # print('skip contours from version 1')
                    col_out = None
            if col_out is None:
                continue
            if filter_columns is not None and col_out not in filter_columns:
                # print(f'Skipping {col_out} because it is not in filter_columns list provided')
                continue

            print(f'\ndata: {col} put in column: {col_out}')

            if isinstance(data, h5py.Dataset):
                # print('per-cell number or bool')
                dtype = data.dtype
                if dtype == np.int8:
                    dtype = bool
                val = np.zeros(data.shape, dtype=dtype)
                data.read_direct(val)
                ct.unit_table.loc[unit_ids, col_out] = val
                new_valid_columns.add(col_out)

            elif isinstance(data, h5py.Group):
                # print('per-cell array')
                vals = []
                ids = []
                missing_count = 0
                for group_unit in data:
                    if missing_count > 10:
                        print('Looks like combo is missing, skipping this param')
                        break
                    uid = (piece_id, 'com', int(group_unit) - 1)
                    try:
                        data_unit = data[group_unit]['combo']
                    except KeyError:
                        print(f'no combo run entry for {group_unit} {uid}')
                        missing_count += 1
                    # print(data_unit)
                    dtype = data_unit.dtype
                    val = np.zeros(data_unit.shape, dtype=dtype)
                    data_unit.read_direct(val)

                    # handle special cases
                    if version == 1:
                        if col_out == 'acf':
                            val = val.transpose()[:, 0]
                        elif col_out == 'sta':
                            val = np.moveaxis(val, [0, 1, 2, 3], [2, 3, 0, 1])
                        elif col_out == 'ei':
                            val = val.transpose()
                        elif col_out == 'tc':
                            val = val.transpose()[:, [0, 3, 1, 4, 2, 5]]
                    if col_out == 'sta':
                        sta_shape = val.shape
                    elif col_out == 'acf':
                        acf_shape = val.shape
                    elif col_out == 'ei':
                        ei_shape = val.shape
                    vals.append(file_handling.wrapper(val))
                    ids.append(uid)
                ct.unit_table.loc[ids, col_out] = vals
                # print(vals, ids)
                new_valid_columns.add(col_out)

        valid_columns_unit = ct.dataset_table.at[(piece_id, 'com'), 'valid_columns_unit'].a
        valid_columns_unit.update(new_valid_columns)
        ct.dataset_table.at[(piece_id, 'com'), 'valid_columns_unit'] = file_handling.wrapper(valid_columns_unit)

        # %% Set up unit parameters
        if 'acf' in new_valid_columns:
            print(f'acf shape: {acf_shape}')
            bins = np.linspace(1, 100, acf_shape[0])
            ct.dataset_table.at[(piece_id, 'com'), 'acf_bins'] = file_handling.wrapper(bins)

        # electrode_map = np.random.random([ei_shape[0], 2])
        if 'ei' in new_valid_columns:
            if ei_shape[0] == 512:
                electrode_map = elmap.LITKE_512_ARRAY_MAP
                print('Using 512 electrode map')
            else:
                electrode_map = elmap.LITKE_519_ARRAY_MAP
                print('Using 519 electrode map')
            ct.dataset_table.at[(piece_id, 'com'), 'ei_electrode_locations'] = file_handling.wrapper(electrode_map)

        if 'sta' in new_valid_columns:
            stixel_size = 5 * (640 / sta_shape[0])
            print(f'using hard-coded approximate stixel size {stixel_size} for combined data')
            ct.dataset_table.at[(piece_id, 'com'), 'stimulus_params'] = file_handling.wrapper(
                {'stixel_size': stixel_size, 'frame_time': 1.0 / 120})

        valid_columns_dataset = ct.dataset_table.at[(piece_id, 'com'), 'valid_columns_dataset'].a
        valid_columns_dataset.update({'ei_electrode_locations', 'acf_bins', 'stimulus_params'})
        ct.dataset_table.at[(piece_id, 'com'), 'valid_columns_dataset'] = file_handling.wrapper(valid_columns_dataset)

        print('done with data load')


def file_save_notes(ct, piece_id, fname=None):
    if fname is None:
        fname = f'{ct.paths["scratch_analysis"]}/{piece_id}/unified.h5'

    print(f"Saving H5 notes to {fname}")
    with h5py.File(fname, 'a') as h5:
        if 'version' in h5:
            version = h5['version'][0]
        else:
            version = 1
        print(f'h5 format version {version}')
        assert(version in [1,2])

        ntab = ct.notes_table
        if 'notes' in h5:
            del h5['notes']
        h5_notes = h5.create_group('notes')
        for col in ['text', 'date', 'type']:
            data = [a.encode('utf-8') for a in ntab[col]]
            h5_notes.create_dataset(col, data=data)
        h5.close()



def file_load_notes(ct, piece_id, fname=None):
    if fname is None:
        fname = f'{ct.paths["scratch_analysis"]}/{piece_id}/unified.h5'

    print(f"Loading H5 notes from {fname}")
    with h5py.File(fname, 'r') as h5:
        if 'version' in h5:
            version = h5['version'][0]
        else:
            version = 1
        print(f'h5 format version {version}')
        assert(version in [1,2])

        if not 'notes' in h5:
            print('No notes table found')
            return pd.DataFrame()

        h5_notes = h5['notes']
        dates = [a.decode('utf-8') for a in h5_notes['date']]
        texts = [a.decode('utf-8') for a in h5_notes['text']]
        types = [a.decode('utf-8') for a in h5_notes['type']]

        ntab = pd.DataFrame(zip(dates, texts, types), columns=['date','text','type'], index=dates)

        h5.close()

    try:
        a = ct.notes_table
    except:
        print('making new empty notes table')
        ct.notes_table = pd.DataFrame()
    for ni, note in ntab.iterrows():
        if ni in ct.notes_table.index:
            continue
        ct.notes_table = pd.concat([ct.notes_table, pd.DataFrame(note).transpose()])

    return ntab
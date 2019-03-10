from ftps_functions import *
from configuration_mapping_file import *
from nfl_utils import chunks
from multiprocessing import Pool, cpu_count


def identify_nl_plays(nl_df, detail_loc_mapping):

    nl_index = nl_df[nl_df['loc'] == 'NL'].index

    text_loc_list = detail_loc_mapping.keys()
    for i in nl_index:
        temp_nl = nl_df.loc[i]['detail']

        for loc in text_loc_list:
            if loc in temp_nl:
                nl_df.loc[i, 'loc'] = detail_loc_mapping[loc]
                break

    return nl_df


def identify_nl_plays_pool(pbp_df):

    use_cpu = cpu_count()
    pool = Pool(use_cpu)

    nl_index = pbp_df[pbp_df['loc'] == 'NL'].index

    loc_nl_list = []
    for nl_chunk in chunks(nl_index, use_cpu):
        loc_nl_list.append([pbp_df[pbp_df.index.isin(nl_chunk)],
                            detail_loc_mapping])

    df_pool = pool.map(identify_nl_plays, loc_nl_list)
    df_concat = pd.concat(df_pool).fillna(0)

    pbp_df = pd.concat((pbp_df[~pbp_df.index.isin(nl_index)],
                        df_concat), axis=1)

    return pbp_df



def def_stats(df):

    df = df[['gid', 'off', 'def', 'ints', 'sk1']]
    df['sk1'] = df['sk1'].fillna(0)
    df['ints'] = df['ints'].fillna(0)
    df.loc[df['sk1'] != 0, 'sk1'] = 1
    df.loc[df['ints'] != 0, 'ints'] = 1
    df = df.groupby(['gid', 'off', 'def'])[['ints', 'sk1']].sum().reset_index()

    return df


def prepare_pbp_data(armchair_data_path, pbp_file,
                     schedule_file, pbp_keep_cols_):

    pbp_df = pd.read_csv(armchair_data_path + pbp_file, usecols=pbp_keep_cols_)
    pbp_df2 = pd.read_csv(armchair_data_path + 'current_year/' + pbp_file,
                          usecols=pbp_keep_cols_)
    pbp_df = pd.concat((pbp_df, pbp_df2))

    pbp_df = pbp_df[(pbp_df['type'] == 'RUSH') |
                    (pbp_df['type'] == 'PASS')]

    spikes_df = pbp_df[pbp_df['spk'] == 'Y']
    spikes_df = spikes_df.groupby(['gid', 'psr'])['spk'].count().reset_index()
    spikes_df.columns = ['gid', 'player', 'spk_count']

    pbp_df = pbp_df[pbp_df['kne'] != 'Y']
    pbp_df = pbp_df[pbp_df['spk'] != 'Y']

    pbp_df = pbp_df[pbp_df.dwn > 0]  # Eliminate kickoffs, only interested in where play starts
    # combine 3rd and 4th downs
    pbp_df.dwn[pbp_df.dwn == 4] = 3

    pbp_df.loc[:, 'dir'].replace(rush_dir_mapping, inplace=True)
    pbp_df.loc[:, 'loc'].replace(pass_dir_mapping, inplace=True)
    pbp_df.loc[:, 'loc'] = pbp_df['loc'].fillna(pbp_df['dir'])
    pbp_df.loc[:, 'loc'] = pbp_df['loc'].fillna('NL')

    pbp_df = identify_nl_plays(pbp_df, detail_loc_mapping)

    pbp_df.loc[pbp_df['loc'] == 'sacked', 'sk1'] = 1

    pbp_def = def_stats(pbp_df)
    pbp_df = pbp_df[pbp_df['sk1'].isnull()]

    pbp_df = pbp_df[(pbp_df['loc'] != 'sacked') &
                    (pbp_df['loc'] != 'NL')]

    pbp_df = pbp_df.drop(['kne', 'sk1', 'dir', 'spk',
                          'detail', 'off', 'def', 'type'], axis=1)

    pbp_df_melt = pd.melt(pbp_df,
                          id_vars=['gid', 'dwn', 'ytg',
                                   'yfog', 'yds',
                                   'pts', 'loc',
                                   'comp', 'ints'])

    pbp_df_melt['pts'] = pbp_df_melt['pts'].fillna(0)
    pbp_df_melt['ints'] = pbp_df_melt['ints'].fillna(0)
    pbp_df_melt['comp'] = pbp_df_melt['comp'].fillna(0)
    pbp_df_melt['yds'] = pbp_df_melt['yds'].fillna(0)
    pbp_df_melt.rename(columns={'comp': 'rec'}, inplace=True)
    pbp_df_melt.loc[pbp_df_melt['variable'] == 'psr', 'rec'] = 0.
    pbp_df_melt.loc[pbp_df_melt['rec'] == 'Y', 'rec'] = 1.
    pbp_df_melt.loc[pbp_df_melt['ints'] != 0, 'ints'] = 1.
    pbp_df_melt = pbp_df_melt.dropna()

    pbp_df_melt.loc[pbp_df_melt['variable'] == 'psr', 'loc_psr'] = '_psr'
    pbp_df_melt.loc[:, 'loc_psr'] = pbp_df_melt['loc_psr'].fillna('')
    pbp_df_melt.loc[:, 'loc'] = pbp_df_melt['loc'] + pbp_df_melt['loc_psr']

    pbp_df_melt.loc[pbp_df_melt['variable'] == 'psr', 'py'] = pbp_df_melt.loc[:, 'yds']
    pbp_df_melt.loc[pbp_df_melt['variable'] == 'bc', 'ry'] = pbp_df_melt.loc[:, 'yds']
    pbp_df_melt.loc[pbp_df_melt['variable'] == 'trg', 'recy'] = pbp_df_melt.loc[:, 'yds']

    pbp_df_melt.loc[((pbp_df_melt['variable'] == 'trg') &
                     (pbp_df_melt['pts'] > 5)), 'tdrec'] = 1
    pbp_df_melt.loc[((pbp_df_melt['variable'] == 'psr') &
                     (pbp_df_melt['pts'] > 5)), 'tdp'] = 1
    pbp_df_melt.loc[((pbp_df_melt['variable'] == 'bc') &
                     (pbp_df_melt['pts'] > 5)), 'tdr'] = 1

    pbp_df_melt = pbp_df_melt.fillna(0)
    dk_ftps = pd.DataFrame(calculate_ftps(pbp_df_melt, dk_scoring_by_col), columns=['dk_ftps_field_bin'])
    fd_ftps = pd.DataFrame(calculate_ftps(pbp_df_melt, fd_scoring_pbp), columns=['fd_ftps_field_bin'])

    pbp_df_melt = pd.concat((pbp_df_melt, dk_ftps, fd_ftps), axis=1)

    schedule_df = pd.read_csv(armchair_data_path_main + schedule_file)
    pbp_df_melt = pd.merge(pbp_df_melt, schedule_df[['gid',
                                                     'seas',
                                                     'wk']],
                           on='gid',
                           how='left')

    pbp_df_melt.loc[:, 'seas_wk'] = (pbp_df_melt['seas'].astype(str) + '.' +
                                     pbp_df_melt['wk'].astype(str).str.zfill(2)).astype(float)

    pbp_df_melt = pbp_df_melt.drop(['yds', 'pts', 'variable', 'seas', 'wk'] +
                                   dk_scoring_by_col.keys(), axis=1)

    return pbp_df_melt, pbp_def, spikes_df


def agg_groubpy_stats(pbp_data_n):

    pbp_data_n_sum = pbp_data_n.groupby(['seas_wk', 'loc',
                                         'ytg_state', 'yfog_state',
                                         'dwn_state'])[['dk_ftps_field_bin',
                                                        'fd_ftps_field_bin']].sum().reset_index()
    pbp_data_n_sum.rename(columns={'dk_ftps_field_bin': 'dk_ftps_field_bin_sum',
                                   'fd_ftps_field_bin': 'fd_ftps_field_bin_sum'}, inplace=True)
    pbp_data_n_count = pbp_data_n.groupby(['seas_wk', 'loc',
                                           'ytg_state', 'yfog_state',
                                           'dwn_state'])[
        ['dk_ftps_field_bin', 'fd_ftps_field_bin']].count().reset_index()
    pbp_data_n_count.rename(columns={'dk_ftps_field_bin': 'state_bin_count'}, inplace=True)
    pbp_data_n_count = pbp_data_n_count.drop('fd_ftps_field_bin', axis=1)
    pbp_data_n_merged = pd.merge(pbp_data_n_count, pbp_data_n_sum,
                                 on=['seas_wk', 'loc', 'ytg_state',
                                     'yfog_state', 'dwn_state'])

    return pbp_data_n_merged


def calc_rolling_cumsum(pbp_data_n_loc, n_periods_, unique_locs):

    for loc in unique_locs:
        idx = pbp_data_n_loc[pbp_data_n_loc['loc'] == loc].index
        pbp_data_n_loc.loc[idx, ['state_bin_count',
                                 'dk_ftps_field_bin_sum',
                                 'fd_ftps_field_bin_sum']] = pbp_data_n_loc.loc[
            idx, ['state_bin_count',
                  'dk_ftps_field_bin_sum',
                  'fd_ftps_field_bin_sum']].rolling(n_periods_, min_periods=1).sum()

    return pbp_data_n_loc


def calc_exp_state_pts_by_year(args):

    [pbp_data, state_chunks, n_periods_] = args

    unique_states = state_chunks.reset_index().drop('index', axis=1)
    unique_locs = list(pbp_data['loc'].unique())
    state_expected_stats = pd.DataFrame()

    for state in range(0, unique_states.shape[0]):
        state_n = unique_states.loc[state]
        state_dict = dict(state_n)
        pbp_data_n = pbp_data.copy()
        for keys in state_dict:
            if keys == 'yfog':
                below_threshold = state_dict[keys] - 6
                above_threshold = state_dict[keys] + 6
                pbp_data_n = pbp_data_n[(pbp_data_n[keys] > below_threshold) &
                                        (pbp_data_n[keys] < above_threshold)]
            elif keys == 'ytg':
                if state_dict[keys] >= 15:
                    below_threshold = 15
                    above_threshold = state_dict[keys] + 100
                else:
                    below_threshold = state_dict[keys] - 3
                    above_threshold = state_dict[keys] + 3
                pbp_data_n = pbp_data_n[(pbp_data_n[keys] > below_threshold) &
                                        (pbp_data_n[keys] < above_threshold)]
            else:
                pbp_data_n = pbp_data_n[(pbp_data_n[keys] == state_dict[keys])]

            pbp_data_n.loc[:, keys + '_state'] = state_dict[keys]

        # pbp_data_n = smooth_outliers(std_threshold_, pbp_data_n)
        pbp_data_n = agg_groubpy_stats(pbp_data_n)
        pbp_data_n = pbp_data_n.sort_values('seas_wk', ascending=True)
        pbp_data_n = calc_rolling_cumsum(pbp_data_n,
                                         n_periods_,
                                         unique_locs)

        pbp_data_n.rename(columns={'state_bin_count': 'state_bin_count_cumsum',
                                   'dk_ftps_field_bin_sum': 'dk_ftps_field_bin_cumsum',
                                   'fd_ftps_field_bin_sum': 'fd_ftps_field_bin_cumsum'},
                          inplace=True)

        pbp_data_n = pbp_data_n[pbp_data_n['seas_wk'] >
                                (2000 + n_periods_ / 21) - 2]

        state_expected_stats = pd.concat((state_expected_stats, pbp_data_n))

    state_expected_stats.loc[:, 'expected_dk_ftps_field_bin'] = (state_expected_stats['dk_ftps_field_bin_cumsum'] /
                                                                 state_expected_stats['state_bin_count_cumsum'])

    state_expected_stats.loc[:, 'expected_fd_ftps_field_bin'] = (state_expected_stats['fd_ftps_field_bin_cumsum'] /
                                                                 state_expected_stats['state_bin_count_cumsum'])

    state_expected_stats = state_expected_stats.reset_index().drop(['index',
                                                                    'fd_ftps_field_bin_cumsum',
                                                                    'state_bin_count_cumsum',
                                                                    'dk_ftps_field_bin_cumsum'],
                                                                   axis=1)

    return state_expected_stats


def smooth_exp_states_values(args):

    [pbp_data_n, unique_states_agg, seas_chunk_list, seas_list] = args

    smooth_state_expected_stats = pd.DataFrame()

    for seas in seas_chunk_list:

        seas_idx = seas_list.index(seas)
        pbp_data_n_seas = pbp_data_n[(pbp_data_n['seas_wk'] > seas_list[seas_idx - 21]) &
                                     (pbp_data_n['seas_wk'] <= seas)]

        for state in range(0, unique_states_agg.shape[0]):
            state_n = unique_states_agg.loc[state]

            if state_n['ytg'] >= 15:
                below_threshold = 15
                above_threshold = state_n['ytg'] + 100
            else:
                below_threshold = state_n['ytg'] - 3
                above_threshold = state_n['ytg'] + 3

            pbp_data_n_seas_state = pbp_data_n_seas[(pbp_data_n_seas[
                                                         'ytg_state'] > below_threshold) &
                                                    (pbp_data_n_seas[
                                                         'ytg_state'] < above_threshold) & (
                pbp_data_n_seas['dwn_state'] == state_n['dwn'])]

            if pbp_data_n_seas_state.shape[0] > 0:
                pbp_data_n2_mean_groupby = pbp_data_n_seas_state.groupby(['loc',
                                                                          'yfog_state',
                                                                          'ytg_state',
                                                                          'dwn_state'])[
                    ['dk_expected',
                     'fd_expected']].mean().reset_index()

                pbp_data_n2_mean_groupby.rename(columns={'dk_expected': 'dk_expected_mean',
                                                         'fd_expected': 'fd_expected_mean'}, inplace=True)

                pbp_data_n2_mean_groupby = smooth_loc_groupby(pbp_data_n2_mean_groupby)
                pbp_data_n2_mean_groupby.loc[:, 'seas_wk'] = seas

                pbp_data_n_seas_state = pbp_data_n_seas_state[pbp_data_n_seas_state['seas_wk'] == seas]

                if state_n['ytg'] >= 15.:

                    pbp_data_n_seas_state = pbp_data_n_seas_state[(pbp_data_n_seas_state['seas_wk'] == seas) &
                                                                  (pbp_data_n_seas_state['ytg'] >= state_n['ytg']) &
                                                                  (pbp_data_n_seas_state['dwn'] == state_n['dwn'])]
                else:
                    pbp_data_n_seas_state = pbp_data_n_seas_state[(pbp_data_n_seas_state['seas_wk'] == seas) &
                                                                  (pbp_data_n_seas_state['ytg'] == state_n['ytg']) &
                                                                  (pbp_data_n_seas_state['dwn'] == state_n['dwn'])]

                pbp_data_n_seas_state = pd.merge(pbp_data_n_seas_state,
                                                 pbp_data_n2_mean_groupby,
                                                 on=['dwn_state', 'ytg_state',
                                                     'yfog_state', 'loc', 'seas_wk'],
                                                 how='left')

                smooth_state_expected_stats = pd.concat((smooth_state_expected_stats, pbp_data_n_seas_state))

    return smooth_state_expected_stats


def smooth_loc_groupby(pbp_data_n2_mean):

    smooth_exp_window = pbp_data_n2_mean.groupby('loc')[
        ['dk_expected_mean',
         'fd_expected_mean']].rolling(window=6,
                                      min_periods=0,
                                      center=True).mean().reset_index(drop=True)
    smooth_exp_window.columns = ['dk_expected_smoothed', 'fd_expected_smoothed']
    pbp_data_n2_mean = pd.concat((pbp_data_n2_mean, smooth_exp_window), axis=1)

    return pbp_data_n2_mean


def calculate_exp_points(pbp_data, n_periods_):

    use_cpu = cpu_count()
    pool = Pool(use_cpu)

    unique_states = pbp_data[['dwn',
                              'ytg',
                              'yfog']].drop_duplicates().reset_index().drop('index', axis=1)

    unique_states = unique_states.sort_values('dwn', ascending=False)
    unique_states_list = []

    for state_chunk in chunks(unique_states, use_cpu):
        dwn_list = list(state_chunk['dwn'].unique())
        unique_states_list.append([pbp_data[(pbp_data['dwn'].isin(dwn_list)) &
                                            (pbp_data['dk_ftps_field_bin'] >= -.10)],
                                   state_chunk, n_periods_])

    df_pool = pool.map(calc_exp_state_pts_by_year, unique_states_list)
    df_concat = pd.concat(df_pool).fillna(0)

    pool.close()
    pool.join()

    df_concat = pd.merge(pbp_data,
                         df_concat,
                         left_on=['seas_wk', 'dwn', 'ytg', 'yfog', 'loc'],
                         right_on=['seas_wk', 'dwn_state',
                                   'ytg_state', 'yfog_state', 'loc'],
                         how='left')

    df_concat.rename(columns={'expected_dk_ftps_field_bin': 'dk_expected',
                              'expected_fd_ftps_field_bin': 'fd_expected',
                              'dk_ftps_field_bin': 'dk_actual',
                              'fd_ftps_field_bin': 'fd_actual',
                              'value': 'player',
                              }, inplace=True)

    df_concat = df_concat[df_concat['seas_wk'] >
                          (2000 + (n_periods_ / 21)) - 2]

    df_concat.loc[df_concat['dk_expected'] < 0, 'dk_expected'] = 0

    seas_list = list(df_concat['seas_wk'].unique())
    unique_states_agg = unique_states.drop('yfog', axis=1)
    unique_states_agg.loc[unique_states_agg['ytg'] > 15., 'ytg'] = 15.
    unique_states_agg = unique_states_agg.drop_duplicates().reset_index().drop('index', axis=1)

    smooth_exp_state_pts = []
    for seas_chunk in chunks(seas_list[21:], use_cpu):
        min_seas_idx = seas_list.index(np.min(seas_chunk))
        smooth_exp_state_pts.append([df_concat[(df_concat['seas_wk'] > min_seas_idx)],
                                     unique_states_agg, seas_chunk, seas_list])

    pool = Pool(use_cpu)

    df_pool = pool.map(smooth_exp_states_values, smooth_exp_state_pts)
    df_concat = pd.concat(df_pool)

    pool.close()
    pool.join()

    df_concat.loc[:, 'dk_diff'] = (df_concat['dk_actual'] -
                                   df_concat['dk_expected_smoothed'])
    df_concat.loc[:, 'fd_diff'] = (df_concat['fd_actual'] -
                                   df_concat['fd_expected_smoothed'])

    df_concat = df_concat.drop(['yfog_state', 'ytg_state', 'dwn_state',
                                'dk_expected', 'fd_expected',
                                'dk_expected_mean', 'fd_expected_mean'], axis=1)

    df_concat.rename(columns={'dk_expected_smoothed': 'dk_expected',
                              'fd_expected_smoothed': 'fd_expected'}, inplace=True)

    df_concat = df_concat.groupby(['player', 'gid',
                                   'seas_wk', 'loc'])[['dk_expected', 'fd_expected',
                                                       'dk_actual', 'fd_actual',
                                                       'dk_diff', 'fd_diff']].sum().reset_index()

    return df_concat
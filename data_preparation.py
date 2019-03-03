import copy
from multiprocessing import Pool, cpu_count
from ftps_functions import map_historical_ftps
from nfl_utils import *


# convert series to supervised learning
def series_to_supervised(data, input_cols, n_games_, n_in=1, n_out=1):
    data = data[input_cols]
    n_vars = 1 if type(data) is list else data.shape[1]
    df = pd.DataFrame(data)

    cols, names = list(), list()

    # input sequence (t-n, ... t-1)
    for i in range(n_games_, 0, -1):
        cols.append(df.shift(i))
        names += [(input_cols[j] + '(t-%d)' % i) for j in range(n_vars)]

    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [(input_cols[j] + '(t)') for j in range(n_vars)]
        else:
            names += [(input_cols[j] + '(t+%d)' % i) for j in range(n_vars)]

    # put it all together
    agg = pd.concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    agg = agg.loc[n_in:].fillna(0)

    return agg


def create_stat_lag_variables(args):

    [player_df, player_chunk,
     cols_scaled, n_games_,
     data_source, index_cols] = args

    player_n_games_df = pd.DataFrame()
    for player in player_chunk:
        temp_player = player_df[player_df[data_source] == player]
        temp_player = temp_player.reset_index().drop('index', axis=1)
        if temp_player.shape[0] > 1:
            player_n_games_temp = series_to_supervised(temp_player,
                                                       cols_scaled,
                                                       n_games_)

            player_n_games_temp = pd.concat((temp_player[index_cols].loc[1:],
                                             player_n_games_temp), axis=1)
            player_n_games_df = pd.concat((player_n_games_df, player_n_games_temp))
        else:
            pass

    return player_n_games_df


def create_stat_lag_variables_pool(df, cols_scaled, n_games_,
                                   data_source, sort_list, index_cols):
    use_cpu = cpu_count()
    pool = Pool(use_cpu)

    source_counts = df[data_source].value_counts().sort_values()
    source_counts_single_game = list(source_counts[source_counts == 1].index)
    df = df[~df[data_source].isin(source_counts_single_game)]

    unique_obs = list(df[data_source].unique())
    df = df.sort_values(sort_list, ascending=True)
    df = df.reset_index().drop('index', axis=1)

    player_input_list = []
    for data_chunk in chunks(unique_obs, use_cpu):
        player_input_list.append([df[df[data_source].isin(data_chunk)],
                                  data_chunk, cols_scaled,
                                  n_games_, data_source, index_cols])

    df_pool = pool.map(create_stat_lag_variables, player_input_list)
    df_concat = pd.concat(df_pool)

    pool.close()
    pool.join()

    return df_concat


def create_player_scaled_lag_variables(player_df, n_games_,
                                       player_idx_cols_,
                                       n_periods,
                                       player_specs_cols,
                                       binary_threshold=0):

    if binary_threshold > 0:
        player_df.loc[:, 'data_FD_pts_binary'] = player_df.loc[:, 'data_FD_pts']
        player_df.loc[player_df['data_FD_pts_binary'] <= binary_threshold, 'data_FD_pts_binary'] = 0.
        player_df.loc[player_df['data_FD_pts_binary'] > binary_threshold, 'data_FD_pts_binary'] = 1.

        player_df.loc[:, 'data_DK_pts_binary'] = player_df.loc[:, 'data_DK_pts']
        player_df.loc[player_df['data_DK_pts_binary'] <= binary_threshold, 'data_DK_pts_binary'] = 0.
        player_df.loc[player_df['data_DK_pts_binary'] > binary_threshold, 'data_DK_pts_binary'] = 1.

    player_df = player_df.fillna(0)
    # player_df = player_df[player_df['data_DK_pts'] > 5]
    player_df.loc[player_df['data_DK_pts'] > 30., 'data_DK_pts'] = 30.
    player_df.loc[player_df['data_FD_pts'] > 30., 'data_FD_pts'] = 30.

    #player_df.loc[:, 'experience'] = player_df['year'] - player_df['start']
    player_df.loc[player_df['year'] >= 2015, 'posd_flag'] = 1
    player_df.loc[:, 'posd_flag'] = player_df['posd_flag'].fillna(0)

    player_cols = list(player_df.columns)
    sos_cols = [col for col in player_cols if '(t' in col]
    player_cols = [col for col in player_cols if '(t' not in col]
    player_cols = list(set(player_cols) - set(player_idx_cols_))
    player_cols = list(set(player_cols) - set(player_idx_cols_))

    expected_cols = [col for col in player_df.columns if '_expected_' in col]
    actual_cols = [col for col in player_df.columns if '_actual_' in col]
    player_cols = list(set(player_cols) - set(expected_cols + actual_cols))

    values = player_df[player_cols].values
    # ensure all data is float
    values = values.astype('float32')

    [player_scaled_df,
     cols_scaled] = scale_cols(values, player_cols)

    #loc_dict = {'dk_expected': [col for col in expected_cols if 'dk_' in col],
    #            'fd_expected': [col for col in expected_cols if 'fd_' in col],
    #            'dk_actual': [col for col in actual_cols if 'dk_' in col],
    #            'fd_actual': [col for col in actual_cols if 'fd_' in col]
    #            }

    #concat_loc_df = scale_locs(player_df, loc_dict)

    #player_scaled_df = pd.concat((player_df, player_scaled_df, concat_loc_df), axis=1)
    player_scaled_df = pd.concat((player_df, player_scaled_df), axis=1)
    #cols_scaled = [col for col in player_scaled_df.columns if '_scaled' in col]
    loc_cols = expected_cols + actual_cols #+ ['data_DK_pts', 'data_FD_pts']
    cols_scaled = cols_scaled + loc_cols

    # player cols do not need lag!!!
    # cols_scaled = list(set(cols_scaled) - set(player_specs_cols))

    # remove_cols = ['fuml_scaled', 'wspd_scaled', 'wk_scaled', 'seas_scaled',
    #               'seas_wk_scaled', 'gstat_Playing_scaled', 'gstat_Questionable_scaled',
    #               'gstat_Probable_scaled', 'temp_scaled', 'humd_scaled', 'wspd_scaled',
    #               'start_scaled']

    # player_scaled_df = player_scaled_df.drop(remove_cols, axis=1)
    # cols_scaled = list(set(cols_scaled) - set(remove_cols))
    cols_scaled = list(set(cols_scaled) - set(player_specs_cols))
    cols_scaled = list(set(cols_scaled) - set(sos_cols))

    cols_scaled_updated = []
    for col in cols_scaled:
        player_scaled_df.rename(columns={col: 'player_' + col}, inplace=True)
        cols_scaled_updated.append('player_' + col)

    player_n_games_df = create_stat_lag_variables_pool(player_scaled_df,
                                                       cols_scaled_updated,
                                                       n_games_,
                                                       data_source='player',
                                                       sort_list=['gid'],
                                                       index_cols=['year', 'wk', 'gid', 'team',
                                                                   'opp', 'player', 'fname',
                                                                   'lname', 'pname', 'nflid',
                                                                   'data_FD_pts', 'data_FD_pts_binary',
                                                                   'data_DK_pts', 'data_DK_pts_binary'] +
                                                                  # player_specs_cols +
                                                                  sos_cols)


    player_n_games_df = player_n_games_df[player_n_games_df['year'] >=
                                          (2000 + n_periods / 21)]

    player_n_games_df = player_n_games_df[player_n_games_df['data_DK_pts'] >= 5]

    return player_n_games_df


def create_team_scaled_lag_varaibles(player_results_df,
                                     team_ftps_allowed, team_ftp_totals_df,
                                     team_index_cols, n_games_):

    [team_ftp_team_cols,
     team_ftps_allowed_pivot] = create_pos_indicators(team_ftps_allowed, team_index_cols)

    team_ftps_allowed_pivot = team_ftps_allowed_pivot.drop_duplicates().reset_index(drop=True)
    team_ftps_allowed_pivot = team_ftps_allowed_pivot.sort_values(['year', 'wk'],
                                                                  ascending=True)

    team_ftps_allowed = team_ftps_allowed_pivot[team_index_cols]

    team_ftp_totals_df_opp = team_ftp_totals_df.copy()
    for col in team_ftp_totals_df_opp.columns[5:]:
        team_ftp_totals_df_opp.rename(columns={col: col + '_opp_total'}, inplace=True)

    team_ftps_allowed_merged = pd.merge(team_ftps_allowed_pivot,
                                        team_ftp_totals_df_opp.drop(['team', 'gid'], axis=1),
                                        on=['opp', 'year', 'wk'],
                                        how='left')

    for col in team_ftp_team_cols:
        team_ftps_allowed_merged.rename(columns={col: 'posd_' + col}, inplace=True)

    # only use the posd_ cols
    opp_posd_pts = [col for col in team_ftps_allowed_merged.columns if 'posd_' in col]

    values = team_ftps_allowed_merged[opp_posd_pts].values
    # ensure all data is float
    values = values.astype('float32')

    [team_scaled_df,
     cols_scaled] = scale_cols(values, opp_posd_pts)

    team_scaled_df = pd.concat((team_ftps_allowed,
                                team_scaled_df),
                               axis=1)

    opp_n_games_df = create_stat_lag_variables_pool(team_scaled_df,
                                                    cols_scaled,
                                                    n_games_ + 1,
                                                    data_source='opp',
                                                    sort_list=['year', 'wk'],
                                                    index_cols=team_index_cols)

    #team_ftp_totals_df_team = team_ftp_totals_df.copy()
    #for col in team_ftp_totals_df_team.columns[5:]:
    #    team_ftp_totals_df_team.rename(columns={col: col + '_team_total'}, inplace=True)

    #team_ftps_allowed_team = team_ftp_totals_df_team[['team', 'year', 'wk']]

    #values = team_ftp_totals_df_team[team_ftp_totals_df_team.columns[5:]].values
    # ensure all data is float
    #values = values.astype('float32')

    #[team_scaled_df,
    # cols_scaled] = scale_cols(values, team_ftp_totals_df_team.columns[5:])

    #team_scaled_df = pd.concat((team_ftps_allowed_team,
    #                            team_scaled_df),
    #                           axis=1)

    #team_n_games_df = create_stat_lag_variables_pool(team_scaled_df,
    #                                                 cols_scaled,
    #                                                 n_games_ + 1,
    #                                                 data_source='team',
    #                                                 sort_list=['year', 'wk'],
    #                                                 index_cols=['year', 'wk', 'team'])

    #team_n_games_df.to_csv('data/def_team_n_games_df.csv', index=False)

    player_results_df = pd.merge(player_results_df, opp_n_games_df,
                                 on=['year', 'wk', 'opp'],
                                 how='left')

    #player_results_df = pd.merge(player_results_df, team_n_games_df,
    #                             on=['year', 'wk', 'team'],
    #                             how='left')

    return player_results_df


def create_player_data(cols_keep,
                       armchair_data_path,
                       #dk_historical_pts_file,
                       #fd_historical_pts_file,
                       injury_file,
                       game_file,
                       player_file,
                       offense_file,
                       schedule_file,
                       drop_player_vars_
                       ):

    injury_file_path = armchair_data_path + injury_file
    game_df = pd.read_csv(armchair_data_path + game_file, low_memory=False)
    game_df2 = pd.read_csv(armchair_data_path + 'current_year/' + game_file,
                           low_memory=False)
    game_df = pd.concat((game_df, game_df2)).drop_duplicates()

    player_df = pd.read_csv(armchair_data_path + player_file, low_memory=False)
    player_df2 = pd.read_csv(armchair_data_path + 'current_year/' + player_file,
                           low_memory=False)
    player_df = pd.concat((player_df, player_df2)).drop_duplicates()

    offense_df = pd.read_csv(armchair_data_path + offense_file, low_memory=False)
    offense_df2 = pd.read_csv(armchair_data_path + 'current_year/' + offense_file,
                             low_memory=False)
    offense_df = pd.concat((offense_df, offense_df2)).drop_duplicates()

    schedule_df = pd.read_csv(armchair_data_path + schedule_file, low_memory=False)

    player_df_drop_cols = ['player', 'posd', 'dcp']
    offense_df_merged = pd.merge(offense_df[cols_keep],
                                 player_df.drop(player_df_drop_cols, axis=1),
                                 on='nflid', how='left')
    offense_df_merged_w_game = pd.merge(offense_df_merged,
                                        game_df,
                                        on='gid', how='left')

    offense_df_merged_w_game = pd.merge(offense_df_merged_w_game,
                                        schedule_df[['date', 'gid']],
                                        on=['gid'], how='left')

    #offense_df_merged_w_ftps = map_historical_ftps(dk_historical_pts_file,
    #                                               offense_df_merged_w_game)

    #offense_df_merged_w_ftps = map_historical_ftps(fd_historical_pts_file,
    #                                               offense_df_merged_w_ftps)

    offense_df_merged_w_ftps = add_injury_info(injury_file_path, offense_df_merged_w_game)

    offense_df_merged_w_ftps = backfill_posd(offense_df_merged_w_ftps, 'QB', 'pa')
    offense_df_merged_w_ftps = backfill_posd(offense_df_merged_w_ftps, 'RB', 'ra')
    offense_df_merged_w_ftps = backfill_posd(offense_df_merged_w_ftps, 'WR', 'trg')
    offense_df_merged_w_ftps = backfill_posd(offense_df_merged_w_ftps, 'TE', 'trg')

    # update the team abv for teams that moved locations
    offense_df_merged_w_ftps['team'].replace(teams_moved, inplace=True)
    offense_df_merged_w_ftps['v'].replace(teams_moved, inplace=True)
    offense_df_merged_w_ftps['h'].replace(teams_moved, inplace=True)
    # Add opp to dataframe
    offense_df_merged_w_ftps.loc[:, 'opp'] = assign_opp(offense_df_merged_w_ftps)
    # Add an indicator that identifies home games
    # offense_df_merged_w_ftps.loc[:, 'home_indicator'] = home_indicator(offense_df_merged_w_ftps)

    offense_df_merged_w_ftps.rename(columns=ftp_cols_rename, inplace=True)

    team_snaps = offense_df_merged_w_ftps.groupby(['team', 'year',
                                                   'wk', 'gid'])['snp'].max().reset_index()

    offense_df_merged_w_ftps = offense_df_merged_w_ftps.drop(drop_player_vars_, axis=1)

    return offense_df_merged_w_ftps, team_snaps


def player_data_scoring_only(player_results_df,
                             team_snaps,
                             redzone_touches,
                             spikes_df,
                             pbp_def,
                             pbp_data,
                             offensive_skills_positions_,
                             indicator_cols_,
                             ftp_cols_,
                             n_periods_,
                             counting_stats):

    index_pts = player_results_df[['gid', 'player', 'year', 'wk',
                                   'data_FD_pts', 'data_DK_pts']]
    index_pts.to_csv('data/player_historical_ftp.csv', index=False)

    player_results_df = player_results_df[player_results_df['year'] >
                                          (2000 + n_periods_ / 21) - 2]

    player_results_df = pd.merge(player_results_df,
                                 redzone_touches.fillna(0),
                                 on=['gid', 'player'],
                                 how='left')

    player_results_df.loc[player_results_df['posd'] == 'QB',
                          'pass_redzone_trg'] = 0

    player_results_df = pd.merge(player_results_df,
                                 spikes_df.fillna(0),
                                 on=['gid', 'player'],
                                 how='left')

    player_results_df.loc[:,
    'redzone_touches'] = (player_results_df['pass_redzone_trg'] +
                          player_results_df['rush_redzone_ra'])
    player_results_df.loc[:,
    'pa'] = (player_results_df['pa'] -
             player_results_df['spk_count'])

    # add ftp player bonus indicator flag
    player_results_df.loc[player_results_df['py'] >= 300., 'py_bonus'] = 1
    player_results_df.loc[player_results_df['ry'] >= 100., 'ry_bonus'] = 1
    player_results_df.loc[player_results_df['recy'] >= 100, 'recy_bonus'] = 1

    player_results_df = player_results_df.drop('spk_count', axis=1)

    [player_results_df,
     loc_cols] = assign_exp_actual_loc(pbp_data, player_results_df)

    ftp_cols_ = ftp_cols_  #  + loc_cols

    team_ftp_totals_df = team_ftp_totals(player_results_df,
                                         team_snaps,
                                         ftp_cols_)

    pbp_def = pd.merge(pbp_def,
                       team_ftp_totals_df[['gid', 'team', 'opp', 'fuml']],
                       left_on=['gid', 'off', 'def'],
                       right_on=['gid', 'team', 'opp'],
                       how='left').drop(['team', 'opp'], axis=1)
    team_ftp_totals_df = team_ftp_totals_df.drop('fuml', axis=1)

    # filter to dfs scoring offensive players only
    player_results_df = player_results_df[player_results_df['posd'].isin(
        offensive_skills_positions_)]

    team_ftps_allowed_df = team_ftps_allow(player_results_df)

    player_results_df = add_ftp_totals(player_results_df,
                                       team_ftp_totals_df, # team_ftp_totals_df
                                       ftp_cols_)

    player_results_df = clean_player_variables(player_results_df,
                                               indicator_cols_
                                               ).fillna(0)

    player_results_df.loc[pd.DataFrame.sum(
        pd.DataFrame.abs(
            player_results_df[counting_stats]), axis=1) == 0, 'dnp'] = 1

    player_results_df.loc[:, 'dnp'] = player_results_df['dnp'].fillna(0)
    player_results_df = player_results_df[player_results_df['dnp'] != 1]

    counting_keep = ['ra', 'trg']
    for keep in counting_stats:
        if keep in counting_keep:
            counting_stats.remove(keep)

    player_results_df = player_results_df.reset_index().drop(['index', 'dnp'] +
                                                             counting_stats,
                                                             axis=1).drop_duplicates()

    return pbp_def, team_ftps_allowed_df, player_results_df, team_ftp_totals_df


def team_ftps_allow(player_results_subset_df):

    team_ftps_allowed = player_results_subset_df.groupby(['year',
                                                          'wk',
                                                          'opp',
                                                          'posd'])[['data_FD_pts',
                                                                    'data_DK_pts']].sum().reset_index()

    # aggregate posd to only include QB, WR, RB, TE
    team_ftps_allowed_parent_pos = copy.deepcopy(team_ftps_allowed)
    team_ftps_allowed_parent_pos['posd'].replace(posd_parent_pos_mapping,
                                                 inplace=True)
    team_ftps_allowed_parent_pos = team_ftps_allowed_parent_pos.groupby(['year',
                                                                         'wk',
                                                                         'opp',
                                                                         'posd'])[['data_FD_pts',
                                                                                   'data_DK_pts']].sum().reset_index()

    team_ftps_allowed.rename(columns={'posd': 'posd_level2',
                                      'data_FD_pts': 'data_FD_pts_level2',
                                      'data_DK_pts': 'data_DK_pts_level2'}, inplace=True)
    team_ftps_allowed.loc[:, 'posd'] = team_ftps_allowed['posd_level2']
    team_ftps_allowed['posd'].replace(posd_parent_pos_mapping,
                                      inplace=True)
    team_ftps_allowed = pd.merge(team_ftps_allowed_parent_pos,
                                 team_ftps_allowed,
                                 on=['year', 'wk', 'opp', 'posd'])
    team_ftps_allowed.rename(columns={'posd': 'posd_level1'}, inplace=True)

    return team_ftps_allowed


def team_ftp_totals(df, snap_df, ftp_cols_):

    team_ftp_totals_df = df.groupby(['team', 'opp', 'year',
                                     'wk', 'gid'])[
        ftp_cols_].agg(lambda x: x[x>0].sum()).reset_index().drop('snp', axis=1)

    team_ftp_totals_df = pd.merge(team_ftp_totals_df,
                                  snap_df,
                                  on=['team', 'year', 'wk', 'gid'],
                                  how='left').fillna(0)

    return team_ftp_totals_df


def add_ftp_totals(df, team_ftp_totals_df, ftp_cols_):

    ftp_cols_.remove('fuml')
    for suffix in ['_team_total']:  # _opp_total
        team_ftp_totals_df_temp = team_ftp_totals_df.copy()
        if 'opp' in suffix:
            left_cols_join = ['team', 'opp', 'year', 'wk', 'gid']
            right_cols_join = ['opp_opp_total', 'team_opp_total',
                               'year_opp_total', 'wk_opp_total',
                               'gid_opp_total']
        else:
            left_cols_join = ['team', 'opp', 'year', 'wk', 'gid']
            right_cols_join = ['team_team_total', 'opp_team_total',
                               'year_team_total', 'wk_team_total',
                               'gid_team_total']

        for col in team_ftp_totals_df_temp.columns:
            team_ftp_totals_df_temp.rename(columns={col: col + suffix}, inplace=True)

        df = pd.merge(df,
                      team_ftp_totals_df_temp,
                      left_on=left_cols_join,
                      right_on=right_cols_join,
                      how='left').drop(right_cols_join, axis=1)

    # we want to keep the expected raw,
    # to adjust bin expected value for current game missing players
    # using the percent and raw value, we can unnormalize each players
    # expected and then normalize the values again
    # keep_expected_raw = ['snp', 'trg', 'ra', 'pass_redzone_trg', 'rush_redzone_ra', 'redzone_touches']
    # keep_expected_raw = [col for col in ftp_cols_ if '_expected_' in col]

    for col in ftp_cols_:
        df.loc[:, col] = (df[col] / df[col + '_team_total']).replace(-np.inf, np.nan)
        df = df.drop(col + '_team_total', axis=1)

    df[ftp_cols_] = df[ftp_cols_].fillna(0)

    return df

# may be able to delete
def find_previous_games_played(temp_player, counting_stats_):

    if temp_player['year'].min() > 2011:
        temp_player = temp_player[temp_player['snp'] > 0]
    elif temp_player['year'].max() < 2012:
        temp_player = temp_player[pd.DataFrame.sum(temp_player[counting_stats_ +
                                                               ['pa']], axis=1) != 0]
    else:
        pre_2012_games = temp_player[temp_player['year'] < 2012]
        pre_2012_games = pre_2012_games[pd.DataFrame.sum(pre_2012_games[counting_stats_ +
                                                                        ['pa']], axis=1) != 0]

        post_2011_games = temp_player[temp_player['year'] > 2011]
        post_2011_games = post_2011_games[post_2011_games['snp'] > 0]

        temp_player = pd.concat((pre_2012_games, post_2011_games))

    return temp_player

# may be able to delete
def prev_games_targeted_stats(temp_player, input_cols, counting_stats_,
                              n_games_, n_out=1, n_in=1):

    # filter out 'dnp' == 1
    temp_player_subset = find_previous_games_played(temp_player, counting_stats_)
    played_idx_list = list(temp_player_subset.index)
    temp_player_subset = temp_player_subset.reset_index().drop('index', axis=1)[input_cols]

    n_vars = len(input_cols)
    agg = pd.DataFrame()

    if temp_player.shape[0] > 1:

        cols, names = list(), list()
        # input sequence (t-n, ... t-1)
        for i in range(n_games_, 0, -1):
            cols.append(temp_player_subset.shift(i))
            names += [(input_cols[j] + '(t-%d)' % i) for j in range(n_vars)]

        # forecast sequence (t, t+1, ... t+n)
        for i in range(0, n_out):
            cols.append(temp_player_subset.shift(-i))
            if i == 0:
                names += [(input_cols[j] + '(t)') for j in range(n_vars)]
            else:
                names += [(input_cols[j] + '(t+%d)' % i) for j in range(n_vars)]

        # put it all together
        agg = pd.concat(cols, axis=1)
        agg.columns = names
        # drop rows with NaN values
        agg = agg.loc[n_in:].fillna(0)

    return agg, played_idx_list

# may be able to delete
def assign_prev_game_targeted_stats(player_results_df,
                                    player, input_cols,
                                    n_games_, counting_stats_,
                                    player_n_games_trg_stats_df):

    temp_player = player_results_df[player_results_df['player'] == player]
    temp_player = temp_player.reset_index().drop('index', axis=1)

    if temp_player.shape[0] > 1:
        agg, played_idx = prev_games_targeted_stats(temp_player,
                                                    input_cols,
                                                    counting_stats_,
                                                    n_games_,
                                                    n_out=1,
                                                    n_in=1)

        for col in agg.columns:
            temp_player.loc[played_idx, col] = agg[col]

        dnp_idx = set(temp_player.index) - set(played_idx)
        temp_player.loc[dnp_idx, 'dnp'] = 1

        player_n_games_trg_stats_df = pd.concat((player_n_games_trg_stats_df,
                                                 temp_player))
    else:
        pass

    return player_n_games_trg_stats_df


def calc_n_game_mean(player_n_games_trg_stats_df, input_cols):

    for stat in input_cols:
        stat_cols = [col for col in player_n_games_trg_stats_df.columns if stat + '(t' in col]
        player_stat_mean = pd.DataFrame.mean(player_n_games_trg_stats_df[stat_cols[:-1]], axis=1)
        player_n_games_trg_stats_df.loc[:, stat + '_n4_mean'] = player_stat_mean
        player_n_games_trg_stats_df = player_n_games_trg_stats_df.drop(stat_cols, axis=1)

    return player_n_games_trg_stats_df


# may be able to delete
def create_targeted_stats_lag(args):

    [player_df, player_chunk,
     input_cols, n_games_, counting_stats_] = args

    player_n_games_trg_stats_df = pd.DataFrame()
    for player in player_chunk:
        player_n_games_trg_stats_df = assign_prev_game_targeted_stats(player_df,
                                                                      player, input_cols, n_games_,
                                                                      counting_stats_,
                                                                      player_n_games_trg_stats_df)

    return player_n_games_trg_stats_df


# may be able to delete
def create_targeted_stats_lag_pool(df, input_cols, n_games_, counting_stats_):

    use_cpu = cpu_count()
    pool = Pool(use_cpu)

    df = df.sort_values('gid', ascending=True)
    df = df.reset_index().drop('index', axis=1)

    player_list = list(df['player'])

    player_input_list = []
    for player_chunk in chunks(player_list, use_cpu):
        player_input_list.append([df[df['player'].isin(player_chunk)],
                                  player_chunk, input_cols,
                                  n_games_, counting_stats_])

    df_pool = pool.map(create_targeted_stats_lag, player_input_list)
    df_concat = pd.concat(df_pool).fillna(0)

    pool.close()
    pool.join()

    return df_concat


def assign_exp_actual_loc(pbp_data, player_results_df):

    df_pivot = pbp_data.pivot_table(values=['dk_expected', 'fd_expected', 'dk_actual', 'fd_actual'],
                                    index=['player', 'gid', 'seas_wk'],
                                    columns='loc',
                                    aggfunc='sum',
                                    fill_value=0)

    loc_cols = ["_".join((j, k)) for j, k in
                df_pivot.columns]
    df_pivot.columns = loc_cols
    df_pivot = df_pivot.reset_index().fillna(0)

    player_results_df = pd.merge(player_results_df,
                    df_pivot,
                    on=['player', 'gid'],
                    how='left')

    return player_results_df, loc_cols


def normalize_locs(df):
    min_ = df.min().min()
    max_ = df.max().max()

    return (df - min_) / (max_ - min_)


def scale_locs(player_scaled_df, loc_dict):
    locs_normalized = dict()
    for site in loc_dict:
        temp = player_scaled_df[loc_dict[site]]
        temp_scaled = normalize_locs(temp)
        col_updated = []
        for col in temp_scaled.columns:
            col_updated.append(col + '_scaled')

        temp_scaled.columns = col_updated
        locs_normalized[site] = temp_scaled

    concat_loc_df = pd.DataFrame()
    for i in locs_normalized:
        concat_loc_df = pd.concat((concat_loc_df, locs_normalized[i]), axis=1)

    return concat_loc_df
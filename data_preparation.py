import copy
from sklearn.preprocessing import MinMaxScaler
from multiprocessing import Pool, cpu_count
from ftps_functions import map_historical_ftps
from nfl_utils import *


# convert series to supervised learning
def series_to_supervised(data, input_cols, n_games, n_in=1, n_out=1):
    data = data[input_cols]
    n_vars = 1 if type(data) is list else data.shape[1]
    df = pd.DataFrame(data)

    cols, names = list(), list()

    # input sequence (t-n, ... t-1)
    for i in range(n_games, 0, -1):
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


def scale_cols(values, ftp_player_cols, scale_suffix):
    # normalize features
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(values)

    cols_scaled = []
    for i in ftp_player_cols:
        cols_scaled.append(i + scale_suffix)

    scaled_df = pd.DataFrame(scaled, columns=cols_scaled)

    return scaled_df, cols_scaled


def create_stat_lag_variables(args):

    [player_df, player_chunk,
     cols_scaled, n_games,
     data_source, index_cols] = args

    player_n_games_df = pd.DataFrame()
    for player in player_chunk:
        temp_player = player_df[player_df[data_source] == player]
        temp_player = temp_player.reset_index().drop('index', axis=1)
        if temp_player.shape[0] > 1:
            player_n_games_temp = series_to_supervised(temp_player,
                                                       cols_scaled,
                                                       n_games)

            player_n_games_temp = pd.concat((temp_player[index_cols].loc[1:],
                                             player_n_games_temp), axis=1)
            player_n_games_df = pd.concat((player_n_games_df, player_n_games_temp))
        else:
            pass

    return player_n_games_df


def create_stat_lag_variables_pool(df, cols_scaled, n_games,
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
                                  n_games, data_source, index_cols])

    df_pool = pool.map(create_stat_lag_variables, player_input_list)
    df_concat = pd.concat(df_pool)

    pool.close()
    pool.join()

    return df_concat


def create_player_scaled_lag_variables(player_df, n_games,
                                       ftp_player_cols,
                                       y_var,
                                       binary_threshold=0):

    if binary_threshold > 0:
        player_df.loc[:, y_var+'_binary'] = player_df.loc[:, y_var]
        player_df.loc[player_df[y_var+'_binary'] <= binary_threshold, y_var+'_binary'] = 0.
        player_df.loc[player_df[y_var+'_binary'] > binary_threshold, y_var+'_binary'] = 1.
        y_var = y_var + '_binary'

    player_df = player_df.fillna(0)
    ftp_player_cols = ftp_player_cols + [y_var]
    values = player_df[ftp_player_cols].values
    # ensure all data is float
    values = values.astype('float32')

    if 'DK' in y_var:
        scaled_suffix = '_scaled_dk'
    else:
        scaled_suffix = '_scaled_fd'

    [player_scaled_df,
     cols_scaled] = scale_cols(values, ftp_player_cols,
                               scale_suffix=scaled_suffix)

    player_scaled_df = pd.concat((player_df, player_scaled_df), axis=1)
    player_n_games_df = create_stat_lag_variables_pool(player_scaled_df,
                                                       cols_scaled,
                                                       n_games,
                                                       data_source='player',
                                                       sort_list=['gid'],
                                                       index_cols=['year', 'wk', 'gid', 'team', 'opp', 'player'])
    y_var_scaled = y_var + scaled_suffix + '(t)'
    n_features = len(ftp_player_cols)

    return player_n_games_df, y_var_scaled, n_features


def create_team_scaled_lag_varaibles(team_ftps_allowed, team_index_cols, y_var, n_games):

    [team_ftp_team_cols,
     team_ftps_allowed_pivot] = create_pos_indicators(team_ftps_allowed, team_index_cols)

    team_ftps_allowed_pivot = team_ftps_allowed_pivot.sort_values(['year', 'wk'],
                                                                  ascending=True)

    # will use to merege back onto original df with index
    team_ftps_allowed_unique = team_ftps_allowed[team_index_cols].drop_duplicates()

    if 'DK' in y_var:
        scaled_suffix = '_scaled_dk'
        ftp_pts = 'DK'
    else:
        scaled_suffix = '_scaled_fd'
        ftp_pts = 'FD'

    team_ftp_team_cols = [col for col in team_ftp_team_cols if ftp_pts in col]

    values = team_ftps_allowed_pivot[team_ftp_team_cols].values
    # ensure all data is float
    values = values.astype('float32')

    [team_scaled_df,
     cols_scaled] = scale_cols(values, team_ftp_team_cols,
                               scale_suffix=scaled_suffix)

    team_scaled_df = pd.concat((team_ftps_allowed
                                [team_index_cols],
                                team_scaled_df),
                               axis=1)

    team_n_games_df = create_stat_lag_variables_pool(team_scaled_df,
                                                     cols_scaled,
                                                     n_games,
                                                     data_source='opp',
                                                     sort_list=['year', 'wk'],
                                                     index_cols=team_index_cols)

    n_features = len(cols_scaled)

    return team_n_games_df, n_features


def create_player_data(cols_keep,
                       armchair_data_path,
                       dk_historical_pts_file,
                       fd_historical_pts_file,
                       injury_file,
                       game_file,
                       player_file,
                       offense_file,
                       schedule_file
                       ):

    injury_file_path = armchair_data_path + injury_file
    game_df = pd.read_csv(armchair_data_path + game_file, low_memory=False)
    player_df = pd.read_csv(armchair_data_path + player_file, low_memory=False)
    offense_df = pd.read_csv(armchair_data_path + offense_file, low_memory=False)
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

    offense_df_merged_w_ftps = map_historical_ftps(dk_historical_pts_file,
                                                   offense_df_merged_w_game)

    offense_df_merged_w_ftps = map_historical_ftps(fd_historical_pts_file,
                                                   offense_df_merged_w_ftps)

    offense_df_merged_w_ftps = add_injury_info(injury_file_path, offense_df_merged_w_ftps)

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
    offense_df_merged_w_ftps.loc[:, 'home_indicator'] = home_indicator(offense_df_merged_w_ftps)

    offense_df_merged_w_ftps.rename(columns=ftp_cols_rename, inplace=True)

    return offense_df_merged_w_ftps


def player_data_scoring_only(player_results_df,
                             offensive_skills_positions,
                             indicator_cols,
                             drop_player_vars):

    # filter to dfs scoring offensive players only
    player_results_subset_df = player_results_df[player_results_df['posd'].isin(
        offensive_skills_positions)]

    team_ftps_allowed_df = team_ftps_allow(player_results_subset_df)

    player_results_subset_dfs_xs_df = clean_player_variables(player_results_subset_df,
                                                             indicator_cols,
                                                             drop_player_vars).fillna(0)
    player_results_subset_dfs_xs_df = player_results_subset_dfs_xs_df.reset_index().drop('index', axis=1)

    return team_ftps_allowed_df, player_results_subset_dfs_xs_df


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

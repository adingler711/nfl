import pandas as pd
from data_preparation import series_to_supervised
from multiprocessing import Pool, cpu_count
from nfl_utils import chunks


def identify_first_sequence_games_missed(args):
    [player_gid, team_lag_games] = args

    player_missing_games = pd.DataFrame()
    for player in player_gid['player'].unique():
        player_gid2 = player_gid[(player_gid['player'] == player)]
        player_team_list = player_gid2['team'].unique()

        for team in player_team_list:
            team_player_df2 = player_gid2[player_gid2['team'] == team]

            temp = pd.merge(team_lag_games[(team_lag_games['year'] >=
                                            team_player_df2['year'].min()) &
                                           (team_lag_games['year'] <=
                                            team_player_df2['year'].max()) &
                                           (team_lag_games['team'] == team)],
                            team_player_df2,
                            on=['team', 'year', 'wk'], how='left')

            temp = temp.fillna(1)
            temp = temp.sort_values('gid')

            agg = series_to_supervised(temp, input_cols=['gid_player'],
                                       n_games_=1, n_in=0, n_out=0)
            temp = pd.concat((temp, agg), axis=1)

            temp = temp[(temp['gid_player(t-1)'] > 1) &
                        (temp['player'] == 1) &
                        (temp['gid(t-1)'] < temp['gid'])]
            temp.loc[:, 'player'] = player

            player_missing_games = pd.concat((player_missing_games, temp))

    return player_missing_games


def create_team_lagged_games(player_results_df):
    team_gid_wk = player_results_df[['gid', 'team', 'year', 'wk']].drop_duplicates()

    team_lag_games = pd.DataFrame()
    for team in team_gid_wk['team'].unique():
        temp_temp = team_gid_wk[team_gid_wk['team'] == team]
        agg = series_to_supervised(temp_temp, input_cols=['gid'],
                                   n_games_=1, n_in=0, n_out=0)
        temp_temp = pd.concat((temp_temp, agg), axis=1)
        team_lag_games = pd.concat((team_lag_games, temp_temp))

    return team_lag_games


def find_missing_player_observations(player_results_df):
    player_gid = player_results_df[['player', 'gid', 'team', 'year', 'wk']]
    player_gid.rename(columns={'gid': 'gid_player'}, inplace=True)

    team_lag_games = create_team_lagged_games(player_results_df)

    use_cpu = cpu_count()
    pool = Pool(use_cpu)

    player_gid = player_gid.sort_values(['team', 'year', 'player'], ascending=True)

    player_input_list = []
    for data_chunk in chunks(player_gid['player'].unique(), use_cpu):
        temp_player_df = player_gid[player_gid['player'].isin(data_chunk)]
        player_input_list.append([temp_player_df,
                                  team_lag_games[team_lag_games[
                                      'team'].isin(temp_player_df['team'].unique())]])

    df_pool = pool.map(identify_first_sequence_games_missed, player_input_list)
    player_missing_games = pd.concat(df_pool).drop(['gid(t-1)',
                                                    'gid_player',
                                                    'wk',
                                                    'year'], axis=1).fillna(0)

    pool.close()
    pool.join()

    return player_missing_games


def assign_missing_game_data(player_results_df, ftp_cols, n_games, index_cols):

    player_missing_games = find_missing_player_observations(player_results_df)
    player_missing_games.rename(columns={'gid': 'gid_missing'}, inplace=True)

    player_missing_games = pd.merge(player_missing_games,
                                    player_results_df,
                                    left_on=['team', 'gid_player(t-1)', 'player'],
                                    right_on=['team', 'gid', 'player'],
                                    how='left')

    player_missing_games_qb = player_missing_games[['team', 'gid_missing',
                                                    'posd_level2_QB_scaled(t)']]
    player_missing_games_qb.rename(columns={'player_posd_level2_QB_scaled(t)': 'qb_out'}, inplace=True)
    player_missing_games_qb_out = player_missing_games_qb[player_missing_games_qb['qb_out'] == 1]
    player_missing_games_qb = player_missing_games_qb[['team', 'gid_missing']].drop_duplicates()
    player_missing_games_qb = pd.merge(player_missing_games_qb, player_missing_games_qb_out.drop_duplicates(),
                                       on=['team', 'gid_missing'], how='left')

    actual_game_cols = [col for col in player_missing_games.columns if '_expected_' in col]
    actual_game_cols = [col for col in actual_game_cols if '(t-1)' in col]
    actual_game_cols = [col for col in actual_game_cols if '_opp_' not in col]
    actual_game_cols = [col for col in actual_game_cols if '_team_' not in col]

    for i, v in enumerate(actual_game_cols):
        actual_game_cols[i] = v.replace("(t-1)", "")

    if 'fuml' in ftp_cols:
        ftp_cols.remove('fuml')

    ftp_col_sum_dict = dict()  # pd.DataFrame()
    for col_find in ftp_cols:
        col_temp = [col for col in player_missing_games.columns if col_find in col]
        #col_temp = [col for col in col_temp if '(t)' not in col]
        col_temp = [col for col in col_temp if '_team_' in col]
        ftp_col_sum_dict[col_find + '_out'] = pd.DataFrame.mean(player_missing_games[col_temp], axis=1)

    for col_find in actual_game_cols:
        col_temp = [col for col in player_missing_games.columns if col_find in col]
        #col_temp = [col for col in col_temp if '(t)' not in col]
        col_temp = [col for col in col_temp if '_opp_' not in col]
        col_temp = [col for col in col_temp if '_team_' not in col]
        ftp_col_sum_dict[col_find + '_out'] = pd.DataFrame.mean(player_missing_games[col_temp], axis=1)

    ftp_col_sum_df = pd.DataFrame(ftp_col_sum_dict.values()).transpose()
    ftp_col_sum_df.columns = ftp_col_sum_dict.keys()

    players_obs_remove = player_missing_games[['player', 'gid_player(t-1)']]
    player_missing_games = player_missing_games[['gid_missing', 'team']].drop_duplicates()

    player_missing_games_stats = pd.concat((player_missing_games[['gid_missing',
                                                                  'team']],
                                            ftp_col_sum_df),
                                           axis=1)

    player_missing_games_stats = player_missing_games_stats.groupby(['gid_missing',
                                                                     'team']).sum().reset_index()

    player_missing_games_stats = pd.merge(player_missing_games_stats,
                                          player_missing_games_qb,
                                          on=['team', 'gid_missing'],
                                          how='left')

    player_results_df = pd.merge(player_results_df,
                                 player_missing_games_stats,
                                 left_on=['gid', 'team'],
                                 right_on=['gid_missing', 'team'],
                                 how='left').drop('gid_missing', axis=1)

    player_results_df = pd.merge(player_results_df,
                                 players_obs_remove,
                                 left_on=['gid', 'player'],
                                 right_on=['gid_player(t-1)', 'player'],
                                 how='left').fillna(0)

    player_results_df = player_results_df[player_results_df[
                                              'gid_player(t-1)'] == 0].drop('gid_player(t-1)', axis=1)

    player_results_df = assign_lagged_out_vars(player_results_df, n_games)

    return player_results_df


def create_stat_out_lag_variables(args):

    [team_df, team_chunk,
     cols_scaled, n_games_] = args

    team_n_games_df = pd.DataFrame()
    for team in team_chunk:
        temp_team = team_df[team_df['team'] == team]
        temp_team = temp_team.reset_index().drop('index', axis=1)

        team_n_games_temp = series_to_supervised(temp_team,
                                                 cols_scaled,
                                                 n_games_,
                                                 n_in=0,
                                                 n_out=0)

        team_n_games_temp = pd.concat((temp_team[['team', 'gid']],
                                       team_n_games_temp), axis=1)
        team_n_games_df = pd.concat((team_n_games_df, team_n_games_temp))

    return team_n_games_df


def assign_lagged_out_vars(player_results_df, n_games_):

    df = player_results_df.copy()
    out_game_stats = [col for col in df.columns if '_out' in col]

    use_cpu = cpu_count()
    pool = Pool(use_cpu)

    df = df[['team', 'gid'] + out_game_stats].drop_duplicates()
    unique_teams = list(df['team'].unique())
    df = df.sort_values('gid', ascending=True)
    df = df.reset_index(drop=True)

    team_input_list = []
    for data_chunk in chunks(unique_teams, use_cpu):
        team_input_list.append([df[df['team'].isin(data_chunk)],
                                data_chunk, out_game_stats,
                                n_games_])

    df_pool = pool.map(create_stat_out_lag_variables, team_input_list)
    df_concat = pd.concat(df_pool)

    pool.close()
    pool.join()

    player_results_df = pd.merge(player_results_df,
                                 df_concat,
                                 on=['team', 'gid'],
                                 how='left')

    return player_results_df
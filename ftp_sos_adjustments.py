import pandas as pd
import numpy as np
from scipy import stats
from collections import OrderedDict
from nfl_utils import chunks
from multiprocessing import Pool, cpu_count
from data_preparation import series_to_supervised
from nfl_utils import scale_cols


def remove_duplicate_game_rows(df_sos, ftps_col,
                               index_cols, rename_cols):
    gid_list = list(df_sos['gid'].unique())
    df_temp = pd.DataFrame()
    for gameid in gid_list:
        temp = df_sos[df_sos['gid'] == gameid]
        copp = temp['opp'].values[0]
        cteam = temp['team'].values[0]
        index_vars = temp[index_cols].values[0]

        if temp.shape[0] > 0:
            if temp.shape[0] == 1:
                temp2 = pd.DataFrame(np.hstack((index_vars, np.array((copp, cteam, 0., 0.))))).transpose()
                temp2.columns = temp.columns
                temp = pd.concat((temp, temp2), axis=0)

            team_def_stats = temp[ftps_col].values
            team_in_game = temp['team'].values
            game_array = np.hstack((index_vars, team_in_game, team_def_stats.flatten()))
            temp = pd.DataFrame(game_array).transpose()
            temp.columns = rename_cols
            df_temp = pd.concat((df_temp, temp))
        else:
            print 'Game', gameid, 'did not have ftp in loc'

    return df_temp


def remove_duplicate_game_rows_def(df_sos, index_cols,
                                   off_pt_cols, rename_cols):
    gid_list = list(df_sos['gid'].unique())
    df_temp = pd.DataFrame()
    for gameid in gid_list:
        temp = df_sos[df_sos['gid'] == gameid]
        index_vars = temp[index_cols].values[0]
        team_def_stats = temp[off_pt_cols].values
        game_array = np.hstack((index_vars, team_def_stats.flatten()))
        temp = pd.DataFrame(game_array).transpose()
        temp.columns = rename_cols
        df_temp = pd.concat((df_temp, temp))
        # break
    return df_temp


def identify_winner(game):
    if game['team_pts'] > game['opp_pts']:
        return game['team']
    else:
        return game['opp']


def identify_loser(game):
    if game['team_pts'] < game['opp_pts']:
        return game['team']
    else:
        return game['opp']


def normalize_massey_rankings(massey_ratings, unit):
    return stats.zscore(massey_ratings[unit])


def calculate_massey_ratings(df_sos):
    teams = list(set(list(df_sos['winner'].unique()) + list(df_sos['loser'].unique())))
    num_games = df_sos.shape[0] - 1
    team_ids = {team: i for i, team in enumerate(teams)}
    game_matrix = np.eye(len(teams)) * num_games  # M in Massey

    # Credit to Sean Taylor for an easy way to create this matrix
    for w, l in df_sos[['winner', 'loser']].itertuples(index=False):
        game_matrix[team_ids[w], team_ids[l]] -= 1
        game_matrix[team_ids[l], team_ids[w]] -= 1

    # create team dict to keep tract of nep for offense and defense ratings
    points_for = dict.fromkeys(teams, 0)
    points_against = dict.fromkeys(teams, 0)

    for team in teams:
        home_points = df_sos[df_sos['team'] == team]['team_pts'].sum()
        away_points = df_sos[df_sos['opp'] == team]['opp_pts'].sum()
        points_for[team] = home_points + away_points
        home_conceded = df_sos[df_sos['team'] == team]['opp_pts'].sum()
        away_conceded = df_sos[df_sos['opp'] == team]['team_pts'].sum()
        points_against[team] = home_conceded + away_conceded

    team_points = pd.DataFrame(index=teams)
    team_points['points_for'] = pd.Series(points_for)
    team_points['points_against'] = pd.Series(points_against)
    team_points['point_diff'] = team_points.points_for - team_points.points_against

    point_diff = team_points.point_diff.values  # p in Massey

    massey_ratings = np.linalg.lstsq(game_matrix, point_diff)[0]
    # massey_ratings = np.linalg.solve(game_matrix, point_diff)
    team_points['massey_ratings'] = massey_ratings

    t_matrix = np.diag(np.repeat(num_games, len(teams)))

    p_matrix = np.diag(np.repeat(0, len(teams)))
    for i in range(len(teams)):
        for j in range(len(teams)):
            if i == j:
                continue
            p_matrix[i][j] = abs(game_matrix[i][j])

    rhs = np.dot(t_matrix, massey_ratings) - team_points.points_for.values
    team_points['defense_ratings'] = np.linalg.lstsq((t_matrix + p_matrix), rhs)[0]
    team_points['offense_ratings'] = massey_ratings - team_points['defense_ratings']

    # team_points.loc[:, 'defense_normalized'] = normalize_massey_rankings(team_points, unit='defense_ratings')
    # team_points.loc[:, 'offense_normalized'] = normalize_massey_rankings(team_points, unit='offense_ratings')

    return team_points


def adjust_def_sos(sos_df_subset, team_def_stats):
    def_output_df = pd.DataFrame()
    for key in team_def_stats:
        def_sos = sos_df_subset[team_def_stats[key][0]]
        def_sos = remove_duplicate_game_rows_def(def_sos, team_def_stats[key][1],
                                                 team_def_stats[key][2], team_def_stats[key][3])

        def_sos.loc[:, 'winner'] = def_sos.apply(lambda x: identify_winner(x), axis=1)
        def_sos.loc[:, 'loser'] = def_sos.apply(lambda x: identify_loser(x), axis=1)
        def_sos = calculate_massey_ratings(def_sos)
        def_sos = def_sos[['defense_ratings', 'offense_ratings']]
        # we need to flip the namings since we are calculating defense sos stats
        rename_dict = {'defense_ratings': 'offense_ratings_' + key,
                       'offense_ratings': 'defense_ratings_' + key}
        def_sos.rename(columns=rename_dict, inplace=True)

        def_output_df = pd.concat((def_output_df, def_sos), axis=1)

    return def_output_df


def team_loc_sos(df_team, groupby_off_cols, off_stat, index_cols, rename_cols):
    df_team = df_team.groupby(groupby_off_cols)[off_stat].sum().reset_index()
    df_team.loc[:, 'def_' + off_stat] = df_team[off_stat]
    df_team = df_team.rename(columns={off_stat: 'off_' + off_stat})

    df_team = remove_duplicate_game_rows(df_team, 'off_' + off_stat,
                                         index_cols, rename_cols)
    df_team[['team_pts', 'opp_pts']] = df_team[['team_pts', 'opp_pts']].astype(float)
    df_team.loc[:, 'winner'] = df_team.apply(lambda x: identify_winner(x), axis=1)
    df_team.loc[:, 'loser'] = df_team.apply(lambda x: identify_loser(x), axis=1)
    df_team = calculate_massey_ratings(df_team)
    # df_team.sort_values('offense_normalized', ascending=False)

    return df_team


def find_year_wk_windows(pbp_data_team_def, n_games, start_yr):

    unique_yr_wk = pbp_data_team_def[pbp_data_team_def['year'] >= start_yr][['year', 'wk']].drop_duplicates()

    year_wk_list = zip(unique_yr_wk['year'], unique_yr_wk['wk'])

    year_wk_dict = OrderedDict()
    idx = 0
    for year, wk in year_wk_list:
        week_list = []
        year_list = []
        last_yr = (wk - n_games - 1)

        if last_yr <= 0:
            week = int(17 - last_yr * -1)
            week_list = range(week, 17)
            year_list = [year - 1] * len(week_list)

        if len(week_list) > 0 and len(week_list) < n_games:
            year_list = year_list + [year] * (n_games - len(week_list))
            week_list = week_list + range(1, wk)
        elif len(week_list) < n_games:
            year_list = [year] * (n_games)
            if wk > 16:
                wk_updated = 16
                week_list = range(wk_updated - n_games, wk_updated + 1)[-n_games:]
            else:
                week_list = range(wk - n_games - 1, wk)[-n_games:]

        year_wk_dict[str(year) + '_'
                     + str(wk)] = [year_list, week_list]
        idx += 1

    return year_wk_dict


def subset_sos_data(pbp_data_team_def, temp_year_wk_list):
    year_unique = np.unique(temp_year_wk_list[0])
    if len(year_unique) == 1:
        pbp_data_team_def_temp = pbp_data_team_def[(pbp_data_team_def['year'] == year_unique[0]) &
                                                   (pbp_data_team_def['wk'].isin(temp_year_wk_list[1]))]
    else:
        first_idx = temp_year_wk_list[0].index(year_unique[0])
        second_idx = temp_year_wk_list[0].index(year_unique[1])

        pbp_data_team_def_temp = pbp_data_team_def[((pbp_data_team_def['year'] == year_unique[0]) &
                                                    (pbp_data_team_def['wk'].isin(temp_year_wk_list[1][
                                                                                  first_idx:second_idx]))) |
                                                   ((pbp_data_team_def['year'] == year_unique[1]) &
                                                    (pbp_data_team_def['wk'].isin(temp_year_wk_list[1][second_idx:])))]

    return pbp_data_team_def_temp


def calculate_def_window_sos(args):
    [pbp_data_team_def, year_wk_dict, team_def_stats] = args

    df_sos_output = pd.DataFrame()

    for year_wk in year_wk_dict:
        temp_year_wk_list = year_wk_dict[year_wk]

        sos_df_subset = subset_sos_data(pbp_data_team_def, temp_year_wk_list)
        def_output_df = adjust_def_sos(sos_df_subset, team_def_stats)
        def_output_df.loc[:, 'year'] = int(year_wk.split('_')[0])
        def_output_df.loc[:, 'wk'] = int(year_wk.split('_')[1])
        df_sos_output = pd.concat((df_sos_output, def_output_df))

    df_sos_output.loc[:, 'team'] = df_sos_output.index
    df_sos_output = df_sos_output.reset_index(drop=True)

    return df_sos_output


def calculate_def_window_sos_pool(pbp_data_team_def,
                                  year_wk_dict,
                                  team_def_stats):
    use_cpu = cpu_count()
    pool = Pool(use_cpu)

    def_sos_input = []
    for year_wk_chunk in chunks(year_wk_dict.keys(), use_cpu):
        yr_wk_values = [year_wk_dict[a] for a in year_wk_chunk]
        year_wk_chunk_dict = dict(zip(year_wk_chunk, yr_wk_values))
        # yr_list = list(np.unique([a.split('_')[0] for a in year_wk_chunk]))

        def_sos_input.append([pbp_data_team_def,
                              year_wk_chunk_dict, team_def_stats])

    df_pool = pool.map(calculate_def_window_sos, def_sos_input)
    df_concat = pd.concat(df_pool).fillna(0)

    pool.close()
    pool.join()

    return df_concat


def assign_def_sos(pbp_def, player_results_df,
                   team_def_stats, n_games, start_yr=2010):

    pbp_data_team_def = pd.merge(pbp_def,
                                 player_results_df[['gid', 'year',
                                                    'wk', 'team',
                                                    'opp']].drop_duplicates(),
                                 left_on=['gid', 'off', 'def'],
                                 right_on=['gid', 'team', 'opp'],
                                 how='left')

    pbp_data_team_def['year'] = pbp_data_team_def['year'].fillna(0).astype(int)
    pbp_data_team_def['wk'] = pbp_data_team_def['wk'].fillna(0).astype(int)
    pbp_data_team_def = pbp_data_team_def[(pbp_data_team_def['year'] >= start_yr - 1)]
    year_wk_dict = find_year_wk_windows(pbp_data_team_def, n_games, start_yr)

    def_window_sos = calculate_def_window_sos_pool(pbp_data_team_def,
                                                   year_wk_dict,
                                                   team_def_stats)

    #player_results_df = pd.merge(player_results_df,
    #                             def_window_sos,
    #                             on=['year', 'wk', 'team'],
    #                             how='left')

    return def_window_sos, year_wk_dict


def calculate_ftp_sos(pbp_data_team, ftps_stats,
                      groupby_off_cols, index_cols,
                      rename_cols, posd_cols):
    unique_loc_list = list(pbp_data_team['loc'].unique())

    def_output_df = pd.DataFrame()
    for off_stat in ftps_stats:
        for loc in unique_loc_list:
            # for posd in posd_cols:

            temp_posd_loc = pbp_data_team[(pbp_data_team['loc'] == loc)]  # &
            # (pbp_data_team[posd] == 1)]

            if temp_posd_loc.shape[0] > 0:
                df_team_loc = team_loc_sos(temp_posd_loc,
                                           groupby_off_cols, off_stat,
                                           index_cols, rename_cols)

                def_sos = df_team_loc[['defense_ratings', 'offense_ratings']]
                # we need to flip the namings since we are calculating defense sos stats
                rename_dict = {'defense_ratings': 'defense_ratings_' + loc + '_' + off_stat,
                               # posd + '_' + off_stat,
                               'offense_ratings': 'offense_ratings_' + loc + '_' + off_stat}  # posd + '_' + off_stat}
                def_sos.rename(columns=rename_dict, inplace=True)

                def_output_df = pd.concat((def_output_df, def_sos), axis=1)

    return def_output_df


def calculate_def_ftp_window_sos(args):
    [pbp_data_team_subset, year_wk_dict,
     ftps_stats, groupby_off_cols,
     index_cols, rename_cols, posd_cols] = args

    df_sos_output = pd.DataFrame()

    for year_wk in year_wk_dict:
        temp_year_wk_list = year_wk_dict[year_wk]

        sos_df_subset = subset_sos_data(pbp_data_team_subset,
                                        temp_year_wk_list)

        sos_df_subset = calculate_ftp_sos(sos_df_subset, ftps_stats,
                                          groupby_off_cols, index_cols,
                                          rename_cols, posd_cols)

        sos_df_subset.loc[:, 'year'] = int(year_wk.split('_')[0])
        sos_df_subset.loc[:, 'wk'] = int(year_wk.split('_')[1])
        sos_df_subset.loc[:, 'team'] = sos_df_subset.index
        sos_df_subset = sos_df_subset.reset_index(drop=True)
        df_sos_output = pd.concat((df_sos_output, sos_df_subset))

    return df_sos_output


def normalize_def_stats(def_sos_merged, def_cols, year_wk_dict):

    use_cpu = cpu_count()
    pool = Pool(use_cpu)

    def_sos_normalized_input = []
    for year_wk_chunk in chunks(year_wk_dict.keys(), use_cpu):
        yr_wk_values = [year_wk_dict[a] for a in year_wk_chunk]
        year_wk_chunk_dict = dict(zip(year_wk_chunk, yr_wk_values))
        def_sos_normalized_input.append([def_sos_merged[def_cols +
                                                        ['team', 'wk', 'year']], year_wk_chunk_dict])

    df_pool = pool.map(normalize_locs, def_sos_normalized_input)
    df_concat = pd.concat(df_pool)  # .fillna(0)

    pool.close()
    pool.join()

    return df_concat


def calculate_def_ftp_window_sos_pool(pbp_data, player_results_df,
                                      year_wk_dict, ftps_stats,
                                      groupby_off_cols,
                                      index_cols, rename_cols, start_yr=2010):

    posd_cols = [col for col in player_results_df.columns if 'posd_' in col]

    pbp_data_team = pd.merge(pbp_data[['gid', 'seas_wk', 'player',
                                       'loc', 'dk_diff', 'fd_diff']],
                             player_results_df[['gid', 'player', 'team', 'opp'] +
                                               posd_cols],
                             on=['gid', 'player'],
                             how='left').groupby(['gid', 'seas_wk', 'loc',
                                                  'team', 'opp'])[['dk_diff',
                                                             'fd_diff']].sum().reset_index()
    # +  posd_cols)

    pbp_data_team = pbp_data_team[(pbp_data_team['seas_wk'] >= start_yr - 1)]
    pbp_data_team[['year', 'wk']] = (pbp_data_team['seas_wk'].astype(str).str.
                                     pad(7, side='right',
                                         fillchar='0').str.split('.',
                                                                 expand=True).astype(int))
    use_cpu = cpu_count()
    pool = Pool(use_cpu)

    def_sos_input = []
    for year_wk_chunk in chunks(year_wk_dict.keys(), use_cpu):
        yr_wk_values = [year_wk_dict[a] for a in year_wk_chunk]
        year_wk_chunk_dict = dict(zip(year_wk_chunk, yr_wk_values))
        #yr_list = list(np.unique([a.split('_')[0] for a in year_wk_chunk]))

        def_sos_input.append([pbp_data_team,
                              year_wk_chunk_dict, ftps_stats, groupby_off_cols,
                              index_cols, rename_cols, posd_cols])

    df_pool = pool.map(calculate_def_ftp_window_sos, def_sos_input)
    df_concat = pd.concat(df_pool) # .fillna(0)

    pool.close()
    pool.join()

    use_cpu = cpu_count()
    pool = Pool(use_cpu)

    def_sos_normalized_input = []
    for year_wk_chunk in chunks(year_wk_dict.keys(), use_cpu):
        yr_wk_values = [year_wk_dict[a] for a in year_wk_chunk]
        year_wk_chunk_dict = dict(zip(year_wk_chunk, yr_wk_values))
        def_sos_normalized_input.append([df_concat, year_wk_chunk_dict])

    df_pool = pool.map(normalize_locs, def_sos_normalized_input)
    df_concat = pd.concat(df_pool)  # .fillna(0)

    pool.close()
    pool.join()

    return df_concat


def normalize_locs(args):

    [df_concat, year_wk_dict] = args

    df_sos_output = pd.DataFrame()
    loc_cols = list(set(list(df_concat.columns)) -
                    set(['team', 'wk', 'year']))

    for year_wk in year_wk_dict:

        yr = year_wk.split('_')[0]
        wk = year_wk.split('_')[1]

        temp_year_wk_list = year_wk_dict[year_wk]

        sos_df_subset = subset_sos_data(df_concat,
                                        temp_year_wk_list)

        temp_loc_df = sos_df_subset[['year',
                                     'wk',
                                     'team']].reset_index(drop=True)
        for loc in loc_cols:
            temp_loc_df.loc[:,
            loc.replace('ratings',
                        'normalized')] = normalize_massey_rankings(sos_df_subset,
                                                                   unit=loc)

        if wk == 1:
            temp_loc_df = temp_loc_df[temp_loc_df['year'] == int(yr) - 1]
        else:
            temp_loc_df = temp_loc_df[temp_loc_df['year'] == int(yr)]

        curr_wk_sos = temp_loc_df.groupby('team')['wk'].max().reset_index()
        curr_wk_sos = pd.merge(curr_wk_sos, temp_loc_df, on=['team', 'wk'], how='left')
        curr_wk_sos.loc[:, 'wk'] = int(wk)

        df_sos_output = pd.concat((df_sos_output, curr_wk_sos))

    return df_sos_output


def assign_sos_stats(pbp_def, pbp_data,
                     player_results_df, ftps_stats,
                     team_def_stats, groupby_off_cols,
                     index_cols, rename_cols, n_games,
                     yr_start=2010):

    [def_window_sos,
     year_wk_dict] = assign_def_sos(pbp_def, player_results_df,
                                    team_def_stats, n_games,
                                    start_yr=yr_start)

    def_sos_loc = calculate_def_ftp_window_sos_pool(pbp_data, player_results_df,
                                                          year_wk_dict, ftps_stats,
                                                          groupby_off_cols, index_cols,
                                                          rename_cols, start_yr=yr_start)

    def_sos_merged = pd.merge(def_sos_loc,
                              def_window_sos,
                              on=['year', 'wk', 'team'],
                              how='left').fillna(0)

    def_cols = ['offense_ratings_int', 'defense_ratings_int',
                'offense_ratings_fuml', 'defense_ratings_fuml',
                'offense_ratings_sack', 'defense_ratings_sack']

    def_df = normalize_def_stats(def_sos_merged, def_cols, year_wk_dict)
    def_sos_merged = pd.merge(def_sos_merged.drop(def_cols, axis=1), def_df,
                              on=['year', 'wk', 'team'],
                              how='left')


    def_sos_opp = calculate_opp_lags(def_sos_merged, n_games)
    def_sos_opp.rename(columns={'team': 'sos_team'}, inplace=True)

    # def_sos_opp.to_csv('data/def_weekly_sos_opp.csv', index=False)
    # def_sos_merged.to_csv('data/off_weekly_sos_ratings.csv', index=False)

    index_cols = ['team', 'year', 'wk']

    ### off cols
    #off_cols = [col for col in def_sos_merged.columns if 'offense_normalized' in col]
    team_sos_opp = def_sos_opp.copy()
    for col in def_sos_opp.columns[3:]:
        team_sos_opp.rename(columns={col: col.replace('opp_var_', '')}, inplace=True)

    def_stats = [col for col in def_sos_opp.columns if 'defense_normalized' in col]
    def_sos_opp = def_sos_opp[['year', 'wk', 'sos_team'] + def_stats]

    off_stats = [col for col in team_sos_opp.columns if 'offense_normalized' in col]
    team_sos_opp = team_sos_opp[['year', 'wk', 'sos_team'] + off_stats]

    player_results_df = pd.merge(player_results_df,
                                 team_sos_opp,
                                 left_on=index_cols,
                                 right_on=['sos_team', 'year', 'wk'],
                                 how='left').drop('sos_team', axis=1)

    player_results_df = pd.merge(player_results_df,
                                 def_sos_opp,
                                 left_on=['opp', 'year', 'wk'],
                                 right_on=['sos_team', 'year', 'wk'],
                                 how='left').drop('sos_team', axis=1)

    return player_results_df


def create_stat_defopp_lag_variables(args):

    [team_df, team_chunk,
     cols_scaled, def_sos_index_cols,
     n_games_] = args

    team_n_games_df = pd.DataFrame()
    for team in team_chunk:
        temp_team = team_df[team_df['team'] == team]
        temp_team = temp_team.reset_index(drop=True)

        team_n_games_temp = series_to_supervised(temp_team,
                                                 cols_scaled,
                                                 n_games_,
                                                 n_in=0,
                                                 n_out=1)

        team_n_games_temp = pd.concat((temp_team[def_sos_index_cols],
                                       team_n_games_temp), axis=1)
        team_n_games_df = pd.concat((team_n_games_df, team_n_games_temp))

    return team_n_games_df


def calculate_opp_lags(def_sos_merged, n_games_):

    def_sos_index_cols = ['year', 'wk', 'team']
    def_sos_cols = list(def_sos_merged.columns)

    for sos_index_col in def_sos_index_cols:
        if sos_index_col in def_sos_cols:
            def_sos_cols.remove(sos_index_col)

    #values = def_sos_merged[def_sos_cols].values
    # ensure all data is float
    #values = values.astype('float32')

    #[def_scaled_df,
    # cols_scaled] = scale_cols(values, def_sos_cols)

    #def_sos_merged_scaled_df = pd.concat((def_sos_merged, def_scaled_df), axis=1).drop(def_sos_cols, axis=1)

    df = def_sos_merged.copy()

    use_cpu = cpu_count()
    pool = Pool(use_cpu)

    df = df.sort_values(['year', 'wk'], ascending=True)
    unique_teams = list(df['team'].unique())

    team_input_list = []
    for data_chunk in chunks(unique_teams, use_cpu):
        team_input_list.append([df[df['team'].isin(data_chunk)],
                                data_chunk, def_sos_cols, def_sos_index_cols,
                                n_games_])

    df_pool = map(create_stat_defopp_lag_variables, team_input_list)
    df_concat = pd.concat(df_pool)

    pool.close()
    pool.join()

    def_cols = df_concat.columns[3:]
    updated_cols = []
    for col in def_cols:
        updated_cols.append('opp_var_' + col)

    def_cols_rename = dict(zip(def_cols, updated_cols))
    df_concat.rename(columns=def_cols_rename, inplace=True)

    return df_concat


def update_weight_list(prev_weeks, sos_def, transformed_weights):

    wk_idx = -1
    if len(prev_weeks[1]) != len(list(sos_def['wk'])):
        wk_missing = list(set(prev_weeks[1]) - set(list(sos_def['wk'])))
        wk_idx = prev_weeks[1].index(wk_missing[0])

    idx = 0
    weight_list = []
    for i in range(0, len(transformed_weights)):
        if idx != wk_idx:
            weight_list.append(transformed_weights[i][0])
        else:
            pass

        idx += 1

    updated_weight_list = []
    for i in range(0, len(weight_list)):
        updated_weight_list.append(weight_list[i] / np.sum(weight_list))

    return updated_weight_list


def update_weighted_sum(player_game, loc,
                        prev_weeks, transformed_weights):

    sos_def = player_game[['wk', loc]].drop_duplicates()
    weight_list = update_weight_list(prev_weeks, sos_def,
                                     transformed_weights)

    idx = 0
    weighted_sum = []
    for wk in prev_weeks[1]:
        if wk in list(sos_def['wk']):
            sos_wk_def = sos_def[sos_def['wk'] == wk][loc].values[0]
            weight = weight_list[idx]
            weighted_sum.append(sos_wk_def * weight)
        idx += 1

    return weighted_sum


def compute_adj_locs(team_adj_df, temp_team, prev_weeks,
                     transformed_weights, loc, opp, key):

    wk_idx = -1
    if len(prev_weeks[1]) != len(list(temp_team['wk'])):
        wk_missing = list(set(prev_weeks[1]) - set(list(temp_team['wk'])))
        wk_idx = prev_weeks[1].index(wk_missing[0])

    idx = 0
    weight_list = []
    for i in range(0, len(transformed_weights)):
        if idx != wk_idx:
            weight_list.append(transformed_weights[i][0])
        else:
            pass

        idx += 1

    updated_weight_list = []
    for i in range(0, len(weight_list)):
        updated_weight_list.append(weight_list[i] / np.sum(weight_list))

    updated_weight_array = np.array(updated_weight_list)
    loc_values = np.array(temp_team[loc])
    loc_adj = np.sum(updated_weight_array * loc_values)

    adj_df = pd.DataFrame(np.array((loc_adj, loc, opp, key))).transpose()
    adj_df.columns = ['loc_adj', 'loc', 'opp', 'yr_wk']

    team_adj_df = pd.concat((team_adj_df, adj_df))

    return team_adj_df
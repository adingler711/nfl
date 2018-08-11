import copy
from nfl_utilis import *
from ftps_functions import map_historical_ftps
from configuration_mapping_file import *
from multiprocessing import Pool, cpu_count


def add_injury_info(injury_file_path, offense_df_merged):

    injury_df = pd.read_csv(injury_file_path)

    offense_df_merged_w_game = pd.merge(offense_df_merged,
                                        injury_df,
                                        on=['gid', 'player', 'team'], how='left')

    return offense_df_merged_w_game


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
    offense_df_merged_w_ftps = backfill_posd(offense_df_merged_w_ftps)

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

    return team_ftps_allowed


def add_prev_games_to_schedule(schedule_df, team, schedule_cumulative_df, vector_dict):

    cur_team = schedule_df[(schedule_df['v'] == team) |
                           (schedule_df['h'] == team)]

    # add a teams previous 4 football games to the schedule df
    cur_team['single_input_vector'] = cur_team[['gid']].apply(tuple, axis=1).apply(list)
    cur_team['single_input_vector'] = cur_team.single_input_vector.apply(
        lambda y: [list(y)])

    iter_df = find_previous_games(cur_team, vector_dict)
    iter_df.loc[:, 'team'] = team
    schedule_cumulative_df = pd.concat((schedule_cumulative_df, iter_df))

    return schedule_cumulative_df


def add_prev_team_stats(team_ftps_allowed_pivot, team, team_ftps_allowed_cumulative_df,
                        team_ftp_team_cols_dk, team_ftp_team_cols_fd,
                        vector_dict, team_index_cols):

    team_ftps_allowed_cur_team = team_ftps_allowed_pivot[team_ftps_allowed_pivot['opp'] == team]

    # create prediction tuple for DK
    team_ftps_allowed_cur_team.loc[:, 'single_input_vector_dk'] = team_ftps_allowed_cur_team[
        team_ftp_team_cols_dk].apply(tuple, axis=1).apply(list)
    team_ftps_allowed_cur_team['single_input_vector_dk'] = team_ftps_allowed_cur_team.single_input_vector_dk.apply(
        lambda y: [list(y)])

    # create prediction tuple for FD
    team_ftps_allowed_cur_team.loc[:, 'single_input_vector_fd'] = team_ftps_allowed_cur_team[
        team_ftp_team_cols_fd].apply(tuple, axis=1).apply(list)
    team_ftps_allowed_cur_team['single_input_vector_fd'] = team_ftps_allowed_cur_team.single_input_vector_fd.apply(
        lambda y: [list(y)])

    team_ftps_allowed_iter_df = find_previous_games(team_ftps_allowed_cur_team,
                                                    vector_dict,
                                                    team_index_cols)

    # removing the first game for each team, there are no previous data to predict current game
    team_ftps_allowed_cumulative_df = pd.concat((team_ftps_allowed_cumulative_df,
                                                 team_ftps_allowed_iter_df.loc[1:]))

    return team_ftps_allowed_cumulative_df


def add_prev_player_stats(args):

    [player_df, player_chunks,
     dk_ftp_player_cols,
     fd_ftp_player_cols,
     vector_dict,
     player_index_cols] = args

    player_ftps_allowed_cumulative_df = pd.DataFrame()
    for player in player_chunks:
        player_ftps_cur = player_df[player_df['player'] == player]
        player_ftps_cur.loc[:, 'single_input_vector_dk'] = player_ftps_cur[
            dk_ftp_player_cols].apply(tuple, axis=1).apply(list)
        player_ftps_cur['single_input_vector_dk'] = player_ftps_cur.single_input_vector_dk.apply(
            lambda y: [list(y)])

        player_ftps_cur.loc[:, 'single_input_vector_fd'] = player_ftps_cur[
            fd_ftp_player_cols].apply(tuple, axis=1).apply(list)
        player_ftps_cur['single_input_vector_fd'] = player_ftps_cur.single_input_vector_fd.apply(
            lambda y: [list(y)])

        player_ftps_allowed_iter_df = find_previous_games(player_ftps_cur,
                                                          vector_dict,
                                                          player_index_cols)

        if player_ftps_allowed_iter_df.shape[0] > 1:
            player_ftps_allowed_cumulative_df = pd.concat((player_ftps_allowed_cumulative_df,
                                                           # removes players first game
                                                           player_ftps_allowed_iter_df.loc[1:]))

    return player_ftps_allowed_cumulative_df


def add_rolling_window_stats_team(schedule_df, team_ftps_allowed):

    team_index_cols = ['year', 'wk', 'opp']

    [team_ftp_team_cols,
     team_ftps_allowed_pivot] = create_pos_indicators(team_ftps_allowed, team_index_cols)

    team_ftps_allowed_pivot = team_ftps_allowed_pivot.sort_values(['year', 'wk'],
                                                                  ascending=True)

    team_ftps_allowed_unique = team_ftps_allowed[['year', 'wk', 'opp']].drop_duplicates()
    team_ftp_team_cols_dk = [col for col in team_ftp_team_cols if 'DK' in col]
    team_ftp_team_cols_fd = [col for col in team_ftp_team_cols if 'FD' in col]

    [team_ftps_allowed_pivot_scaled,
     dk_cols_dk_scaled,
     fd_cols_fd_scaled] = scale_cols(team_ftps_allowed_pivot,
                                     team_ftp_team_cols_dk,
                                     team_ftp_team_cols_fd)

    team_ftps_allowed_pivot_scaled = pd.concat((team_ftps_allowed_pivot[team_index_cols],
                                                team_ftps_allowed_pivot_scaled), axis=1)

    unique_teams = set(list(schedule_df['v']) + list(schedule_df['h']))

    #schedule_df = schedule_df.sort_values('gid', ascending=True)
    #vector_schedule_dict = {'single_input_vector': 'cumulative_gid_vectors_team'}
    vector_team_dict = {'single_input_vector_dk': 'cumulative_gid_vectors_team_dk',
                        'single_input_vector_fd': 'cumulative_gid_vectors_team_fd'}

    #schedule_cumulative_df = pd.DataFrame()
    team_ftps_allowed_cumulative_df = pd.DataFrame()

    for team in unique_teams:

        #schedule_cumulative_df = add_prev_games_to_schedule(schedule_df,
        #                                                    team,
        #                                                    schedule_cumulative_df,
        #                                                    vector_schedule_dict)

        team_ftps_allowed_cumulative_df = add_prev_team_stats(team_ftps_allowed_pivot_scaled,
                                                              team,
                                                              team_ftps_allowed_cumulative_df,
                                                              dk_cols_dk_scaled,
                                                              fd_cols_fd_scaled,
                                                              vector_team_dict,
                                                              team_index_cols)

    #schedule_df = pd.merge(schedule_df, schedule_cumulative_df,
    #                       left_index=True, right_index=True)
    #schedule_df = schedule_df.reset_index().drop('index', axis=1)

    team_ftps_allowed_cum = pd.merge(team_ftps_allowed_unique,
                                 team_ftps_allowed_cumulative_df,
                                 on=['year', 'wk', 'opp'], how='right')

    team_ftps_allowed_cum = team_ftps_allowed_cum.reset_index().drop('index', axis=1)
    #schedule_df = schedule_df.reset_index().drop('index', axis=1)
    team_ftps_allowed_cum = pad_sequences_data(team_ftps_allowed_cum.copy(), 'cumulative_gid_vectors_team_dk')
    team_ftps_allowed_cum = pad_sequences_data(team_ftps_allowed_cum.copy(), 'cumulative_gid_vectors_team_fd')
    #schedule_df = pad_sequences_data(schedule_df.copy(), 'cumulative_gid_vectors_team')

    return team_ftps_allowed_cum #, schedule_df


def create_ftp_player_cols(player_df, player_non_x_cols):

    ftp_player_cols = list(set(player_df.columns) - set(player_non_x_cols))
    dk_ftp_player_cols = ftp_player_cols[:]
    fd_ftp_player_cols = ftp_player_cols[:]

    #  remove FD pts from DK cols
    dk_ftp_player_cols.remove('data_FD_pts')
    dk_ftp_player_cols.remove('actual_fd_salary')
    dk_ftp_player_cols.remove('actual_dk_salary')
    # remove DK pts from FD cols
    fd_ftp_player_cols.remove('data_DK_pts')
    fd_ftp_player_cols.remove('actual_dk_salary')
    fd_ftp_player_cols.remove('actual_fd_salary')
    # remove opp from pred cols
    fd_ftp_player_cols.remove('opp')
    dk_ftp_player_cols.remove('opp')

    return dk_ftp_player_cols, fd_ftp_player_cols


def add_rolling_window_stats_player(player_df, player_non_x_cols, save_ftp_cols):

    [dk_ftp_player_cols,
     fd_ftp_player_cols] = create_ftp_player_cols(player_df, player_non_x_cols)

    pd.DataFrame(dk_ftp_player_cols,
                 columns=['dk_ftp_player_cols_']).\
        to_csv('data/dk_ftp_player_cols_' + save_ftp_cols + '.csv', index=False)
    pd.DataFrame(fd_ftp_player_cols,
                 columns=['fd_ftp_player_cols_']). \
        to_csv('data/fd_ftp_player_cols_' + save_ftp_cols + '.csv', index=False)

    unique_player = list(player_df['player'].unique())
    player_df = player_df.sort_values('gid', ascending=True)
    player_df = player_df.reset_index().drop('index', axis=1)

    [player_df_scaled,
     dk_cols_dk_scaled,
     fd_cols_fd_scaled] = scale_cols(player_df,
                                     dk_ftp_player_cols,
                                     fd_ftp_player_cols)

    player_index_cols = ['gid', 'team', 'player']
    player_df_scaled = pd.concat((player_df[player_index_cols],
                                  player_df_scaled), axis=1)

    vector_player_dict = {'single_input_vector_dk': 'cumulative_gid_vectors_player_dk',
                          'single_input_vector_fd': 'cumulative_gid_vectors_player_fd'}

    use_cpu = cpu_count()
    pool = Pool(use_cpu)

    player_input_list = []
    for player_chunk in chunks(unique_player, use_cpu):
        player_input_list.append([player_df_scaled[player_df_scaled['player'].isin(player_chunk)],
                                  player_chunk, dk_cols_dk_scaled, fd_cols_fd_scaled,
                                  vector_player_dict, player_index_cols])

    df_pool = pool.map(add_prev_player_stats, player_input_list)
    df_concat = pd.concat(df_pool)

    pool.close()
    pool.join()

    player_ftps = pd.merge(player_df, df_concat,
                           on=player_index_cols,
                           how='right')

    player_ftps = player_ftps.reset_index().drop('index', axis=1)
    player_ftps = pad_sequences_data(player_ftps.copy(), 'cumulative_gid_vectors_player_dk')
    player_ftps = pad_sequences_data(player_ftps.copy(), 'cumulative_gid_vectors_player_fd')

    return player_ftps


def player_data_scoring_only(player_results_df,
                             offensive_skills_positions,
                             indicator_cols,
                             drop_player_vars):

    # filter to dfs scoring offensive players only
    player_results_subset_df = player_results_df[player_results_df['posd'].isin(
        offensive_skills_positions)]
    team_ftps_allowed = team_ftps_allow(player_results_subset_df)

    player_results_subset_dfs_xs_df = clean_player_variables(player_results_subset_df,
                                                             indicator_cols,
                                                             drop_player_vars).fillna(0)

    return team_ftps_allowed, player_results_subset_dfs_xs_df

if __name__ == '__main__':

    dk_historical_pts_file_main = 'DFS_NFL_Salaries/DFS_DK_Historical_Salary.csv'
    fd_historical_pts_file_main = 'DFS_NFL_Salaries/DFS_FD_Historical_Salary.csv'
    armchair_data_path_main = 'armchair_analysis_data/'
    injury_file_main = 'INJURY.csv'
    game_file_main = 'GAME.csv'
    player_file_main = 'PLAYER.csv'
    offense_file_main = 'OFFENSE.csv'
    schedule_file_main = 'SCHEDULE_2018.csv'

    cols_keep_main = ['gid', 'team', 'player', 'year', 'posd', 'dcp', 'nflid',
                      'fp2', 'fp3', 'py', 'ints', 'tdp', 'ry', 'tdr',
                      'rec', 'recy', 'tdrec', 'tdret', 'fuml', 'pa']

    player_results_df_main = create_player_data(cols_keep_main,
                                                armchair_data_path_main,
                                                dk_historical_pts_file_main,
                                                fd_historical_pts_file_main,
                                                injury_file_main,
                                                game_file_main,
                                                player_file_main,
                                                offense_file_main,
                                                schedule_file_main
                                                )

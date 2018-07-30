import copy
import pandas as pd
import numpy as np
from ftps_functions import map_historical_ftps
from configuration_mapping_file import *


def add_injury_info(injury_file_path, offense_df_merged):

    injury_df = pd.read_csv(injury_file_path)

    offense_df_merged_w_game = pd.merge(offense_df_merged,
                                        injury_df,
                                        on=['gid', 'player', 'team'], how='left')

    return offense_df_merged_w_game


def create_player_data(cols_keep,
                       armchair_data_path,
                       dk_historical_pts_file,
                       fd_historical_pts_file
                       ):

    injury_file_path = armchair_data_path + 'INJURY.csv'
    game_df = pd.read_csv(armchair_data_path + 'GAME.csv', low_memory=False)
    player_df = pd.read_csv(armchair_data_path + 'PLAYER.csv', low_memory=False)
    offense_df = pd.read_csv(armchair_data_path + 'OFFENSE.csv', low_memory=False)

    player_df_drop_cols = ['player', 'posd', 'dcp']
    offense_df_merged = pd.merge(offense_df[cols_keep],
                                 player_df.drop(player_df_drop_cols, axis=1),
                                 on='nflid', how='left')
    offense_df_merged_w_game = pd.merge(offense_df_merged,
                                        game_df,
                                        on='gid', how='left')

    offense_df_merged_w_ftps = map_historical_ftps(dk_historical_pts_file,
                                                   offense_df_merged_w_game)

    offense_df_merged_w_ftps = map_historical_ftps(fd_historical_pts_file,
                                                   offense_df_merged_w_ftps)

    offense_df_merged_w_ftps = add_injury_info(injury_file_path, offense_df_merged_w_ftps)
    offense_df_merged_w_ftps = backfill_posd(offense_df_merged_w_ftps)

    # Add opp to dataframe
    offense_df_merged_w_ftps.loc[:, 'opp'] = assign_opp(offense_df_merged_w_ftps)
    # Add an indicator that identifies home games
    offense_df_merged_w_ftps.loc[:, 'home_indicator'] = home_indicator(offense_df_merged_w_ftps)

    # update the team abv for teams that moved locations
    offense_df_merged_w_ftps['team'].replace(teams_moved, inplace=True)

    return offense_df_merged_w_ftps


def backfill_posd(player_results_df):

    # identify QBS
    qb_idx = player_results_df[((player_results_df['pos1'] == 'QB') &
                                pd.isnull(player_results_df['posd'])) |
                               ((player_results_df['pos2'] == 'QB') &
                                pd.isnull(player_results_df['posd']) &
                                (player_results_df['pa'] > 1))].index
    player_results_df.loc[qb_idx, 'posd'] = 'QB'

    # identify RB
    rb_idx = player_results_df[((player_results_df['pos1'] == 'RB') &
                                pd.isnull(player_results_df['posd'])) |
                               ((player_results_df['pos2'] == 'RB') &
                                pd.isnull(player_results_df['posd']) &
                                (player_results_df['ra'] > 1))].index
    player_results_df.loc[rb_idx, 'posd'] = 'RB'

    # identify WR
    wr_idx = player_results_df[((player_results_df['pos1'] == 'WR') &
                                pd.isnull(player_results_df['posd'])) |
                               ((player_results_df['pos2'] == 'WR') &
                                pd.isnull(player_results_df['posd']) &
                                (player_results_df['trg'] > 1))].index
    player_results_df.loc[wr_idx, 'posd'] = 'WR'

    # identify TE
    te_idx = player_results_df[((player_results_df['pos1'] == 'TE') &
                                pd.isnull(player_results_df['posd'])) |
                               ((player_results_df['pos2'] == 'TE') &
                                pd.isnull(player_results_df['posd']) &
                                (player_results_df['trg'] > 1))].index
    player_results_df.loc[te_idx, 'posd'] = 'TE'

    return player_results_df


def assign_opp(df):
    return np.where(df.loc[:, 'team'] ==
                    df.loc[:, 'v'],
                    df.loc[:, 'h'],
                    df.loc[:, 'v'])


def home_indicator(df):
    return np.where(df.loc[:, 'team'] ==
                    df.loc[:, 'h'],
                    1,  # type: int
                    0)  # type: int


def team_ftps_allow(player_results_subset_df):

    team_ftps_allowed = player_results_subset_df.groupby(['year',
                                                          'wk',
                                                          'opp',
                                                          'posd'])[['fp2',
                                                                    'fp3']].sum().reset_index()

    # aggregate posd to only include QB, WR, RB, TE
    team_ftps_allowed_parent_pos = copy.deepcopy(team_ftps_allowed)
    team_ftps_allowed_parent_pos['posd'].replace(posd_parent_pos_mapping,
                                                 inplace=True)
    team_ftps_allowed_parent_pos = team_ftps_allowed_parent_pos.groupby(['year',
                                                                         'wk',
                                                                         'opp',
                                                                         'posd'])[['fp2',
                                                                                   'fp3']].sum().reset_index()

    team_ftps_allowed.rename(columns={'posd': 'posd_level1',
                                      'fp2': 'fp2_level1',
                                      'fp3': 'fp3_level1'}, inplace=True)
    team_ftps_allowed.loc[:, 'posd'] = team_ftps_allowed['posd_level1']
    team_ftps_allowed['posd'].replace(posd_parent_pos_mapping,
                                      inplace=True)
    team_ftps_allowed = pd.merge(team_ftps_allowed_parent_pos,
                                 team_ftps_allowed,
                                 on=['year', 'wk', 'opp', 'posd'])

    return team_ftps_allowed


# OLD function, maybe delete if do not need
def find_previous_schedule(schedule_df, team, game_week, game_window=4):

    prev_games = schedule_df[(schedule_df['seas'] <= game_week['seas'].values[0]) &
                             (schedule_df['wk'] <= game_week['wk'].values[0]) &
                             ((schedule_df['v'] == team) |
                              (schedule_df['h'] == team))].tail(game_window + 1)

    return prev_games


def find_previous_games(cur_team, team, game_window=4):

    iter_df = pd.DataFrame()
    for idx in cur_team.index:
        # create an temporary empty dataframe to assign the 4 day period rolling windows per device

        temp_cur_team_df = cur_team.loc[:idx].tail(game_window + 1).cumsum().loc[idx]
        temp_df = temp_cur_team_df.to_frame()
        # The .cumsum() will include the previous day's attribute readings into the new input vector
        iter_df = pd.concat((iter_df, temp_df[temp_df.index == 'single_input_vector']), axis=1)

    iter_df = iter_df.transpose()
    iter_df.columns = ['cumulative_gid_vectors']
    iter_df.loc[:, 'team'] = team

    return iter_df


def add_prev_games_to_schedule(schedule_df, team, schedule_cumulative_df):

    cur_team = schedule_df[(schedule_df['v'] == team) |
                           (schedule_df['h'] == team)]

    # add a teams previous 4 football games to the schedule df
    cur_team['single_input_vector'] = cur_team[['gid']].apply(tuple, axis=1).apply(list)
    cur_team['single_input_vector'] = cur_team.single_input_vector.apply(
        lambda y: [list(y)])

    iter_df = find_previous_games(cur_team, team)
    schedule_cumulative_df = pd.concat((schedule_cumulative_df, iter_df))

    return schedule_cumulative_df


def add_prev_team_stats(team_ftps_allowed, team, team_ftps_allowed_cumulative_df, ftp_team_cols):

    team_ftps_allowed_cur_team = team_ftps_allowed[team_ftps_allowed['opp'] == team]
    team_ftps_allowed_cur_team[
        'single_input_vector'] = team_ftps_allowed_cur_team[ftp_team_cols].apply(tuple,
                                                                                 axis=1).apply(list)
    team_ftps_allowed_cur_team['single_input_vector'] = team_ftps_allowed_cur_team.single_input_vector.apply(
        lambda y: [list(y)])

    team_ftps_allowed_iter_df = find_previous_games(team_ftps_allowed_cur_team, team)
    team_ftps_allowed_cumulative_df = pd.concat((team_ftps_allowed_cumulative_df,
                                                 team_ftps_allowed_iter_df))

    return team_ftps_allowed_cumulative_df


def add_rolling_window_stats(schedule_df, team_ftps_allowed, ftp_team_cols):

    unique_teams = set(list(schedule_df['v']) + list(schedule_df['h']))
    schedule_df = schedule_df.sort_values('gid', ascending=True)
    schedule_cumulative_df = pd.DataFrame()
    team_ftps_allowed_cumulative_df = pd.DataFrame()
    for team in unique_teams:

        schedule_cumulative_df = add_prev_games_to_schedule(schedule_df,
                                                            team,
                                                            schedule_cumulative_df)
        team_ftps_allowed_cumulative_df = add_prev_team_stats(team_ftps_allowed,
                                                              team,
                                                              team_ftps_allowed_cumulative_df,
                                                              ftp_team_cols)

    schedule_df = pd.merge(schedule_df, schedule_cumulative_df,
                           left_index=True, right_index=True)
    schedule_df = schedule_df.reset_index().drop('index', axis=1)

    team_ftps_allowed = pd.merge(team_ftps_allowed, team_ftps_allowed_cumulative_df,
                                 left_index=True, right_index=True)
    team_ftps_allowed = team_ftps_allowed.drop('team', axis=1)

    return schedule_df, team_ftps_allowed


if __name__ == '__main__':

    dk_historical_pts_file_main = 'DFS_NFL_Salaries/DFS_DK_Historical_Salary.csv'
    fd_historical_pts_file_main = 'DFS_NFL_Salaries/DFS_FD_Historical_Salary.csv'
    armchair_data_path_main = 'armchair_analysis_data/'

    cols_keep_main = ['gid', 'team', 'player', 'year', 'posd', 'dcp', 'nflid',
                      'fp2', 'fp3', 'py', 'ints', 'tdp', 'ry', 'tdr',
                      'rec', 'recy', 'tdrec', 'tdret', 'fuml', 'pa']

    player_results_df_main = create_player_data(cols_keep_main,
                                                armchair_data_path_main,
                                                dk_historical_pts_file_main,
                                                fd_historical_pts_file_main
                                                )

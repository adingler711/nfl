import pandas as pd
import datetime
import numpy as np
from configuration_mapping_file import *


def open_schedule(armchair_data_path_main, schedule_file):

    schedule_df = pd.read_csv(armchair_data_path_main + schedule_file)
    schedule_df['v'].replace(teams_moved, inplace=True)
    schedule_df['h'].replace(teams_moved, inplace=True)

    return schedule_df


def calculate_player_age_days(player_df):

    player_df.loc[:, 'dob'] = pd.to_datetime(pd.Series(
        player_df['dob']), format="%m/%d/%Y")

    current_date = datetime.datetime.now()
    date_difference = current_date - player_df['dob']
    player_df.loc[:, 'age_days'] = date_difference / datetime.timedelta(days=1)
    player_df = player_df.drop('dob', axis=1)

    return player_df


def create_pos_indicators(df, index_cols):

    df_pivot = df.pivot_table(index=index_cols,
                              columns=['posd', 'posd_level2'])
    df_pivot.columns = ["_".join((j, k, i)) for i, j, k in
                        df_pivot.columns]
    df_pivot = df_pivot.reset_index().fillna(0)
    ftp_pos_cols = list(df_pivot.drop(index_cols, axis=1).columns)

    return ftp_pos_cols, df_pivot


def clean_player_variables(player_df, indicator_cols, drop_player_vars):

    player_df.loc[:, 'posd_level2'] = player_df['posd']
    player_df.loc[:, 'posd'].replace(posd_parent_pos_mapping, inplace=True)
    player_df.loc[:, 'gstat'].replace(gstat_mapping, inplace=True)

    just_dummies = pd.get_dummies(player_df[indicator_cols],
                                  drop_first=True)
    player_df = pd.concat([player_df, just_dummies], axis=1)
    indicator_cols.remove('opp')
    drop_ftps_cols = ftp_cols_rename.values()
    drop_ftps_cols.remove('data_FD_pts')
    drop_ftps_cols.remove('data_DK_pts')
    drop_ftps_cols.remove('actual_dk_salary')
    drop_ftps_cols.remove('actual_fd_salary')

    player_df = player_df.drop(drop_ftps_cols +
                               drop_player_vars +
                               indicator_cols,
                               axis=1)
    player_df.loc[:, 'actual_dk_salary'] = player_df['actual_dk_salary'].fillna(0)
    player_df.loc[:, 'actual_fd_salary'] = player_df['actual_fd_salary'].fillna(0)
    player_df = calculate_player_age_days(player_df)

    return player_df


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


def find_previous_games_team(cur_team, team, game_window=4):

    iter_df = pd.DataFrame()
    for idx in cur_team.index:
        # create an temporary empty dataframe to assign the 4 day period rolling windows per device

        temp_cur_team_df = cur_team.loc[:idx].tail(game_window + 1).cumsum().loc[idx]
        temp_df = temp_cur_team_df.to_frame()
        # The .cumsum() will include the previous day's attribute readings into the new input vector
        iter_df = pd.concat((iter_df, temp_df[temp_df.index == 'single_input_vector']), axis=1)

    iter_df = iter_df.transpose()
    iter_df.columns = ['cumulative_gid_vectors_team']
    iter_df.loc[:, 'team'] = team

    return iter_df


def find_previous_games_player(cur_player, game_window=4):

    iter_df = pd.DataFrame()
    for idx in cur_player.index:
        # create an temporary empty dataframe to assign the 4 day period rolling windows per device

        temp_cur_team_df = cur_player.loc[:idx].tail(game_window + 1).cumsum().loc[idx]
        temp_df = temp_cur_team_df.to_frame()
        # The .cumsum() will include the previous day's attribute readings into the new input vector
        iter_df = pd.concat((iter_df, temp_df[temp_df.index == 'single_input_vector']), axis=1)

    iter_df = iter_df.transpose()
    iter_df.columns = ['cumulative_gid_vectors_player']

    return iter_df

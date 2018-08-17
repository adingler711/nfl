import pandas as pd
import numpy as np
import datetime
from sklearn.preprocessing import MinMaxScaler
from configuration_mapping_file import *


def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]


def open_schedule(armchair_data_path_main, schedule_file):

    schedule_df = pd.read_csv(armchair_data_path_main + schedule_file)
    schedule_df['v'].replace(teams_moved, inplace=True)
    schedule_df['h'].replace(teams_moved, inplace=True)

    return schedule_df


def add_injury_info(injury_file_path, offense_df_merged):

    injury_df = pd.read_csv(injury_file_path)

    offense_df_merged_w_game = pd.merge(offense_df_merged,
                                        injury_df,
                                        on=['gid', 'player', 'team'], how='left')

    return offense_df_merged_w_game


def scale_cols(values, ftp_player_cols):
    # normalize features
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(values)

    cols_scaled = []
    for i in ftp_player_cols:
        cols_scaled.append(i + '_scaled')

    scaled_df = pd.DataFrame(scaled, columns=cols_scaled)

    return scaled_df, cols_scaled


def create_pos_indicators(df, index_cols):

    df_pivot = df.pivot_table(index=index_cols,
                              columns=['posd_level1', 'posd_level2'],
                              aggfunc='sum',
                              fill_value=0)

    df_pivot.columns = ["_".join((j, k, i)) for i, j, k in
                        df_pivot.columns]
    df_pivot = df_pivot.reset_index().fillna(0)
    ftp_pos_cols = list(df_pivot.drop(index_cols, axis=1).columns)

    return ftp_pos_cols, df_pivot


def backfill_posd(player_results_df, posd, stat):

    # identify parent posd levels
    posd_idx = player_results_df[((player_results_df['pos1'] == posd) &
                                  pd.isnull(player_results_df['posd'])) |
                                 ((player_results_df['pos2'] == posd) &
                                  pd.isnull(player_results_df['posd']) &
                                  (player_results_df[stat] > 1))].index

    player_results_df.loc[posd_idx, 'posd'] = posd

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


def fix_gstat(df):

    df.loc[:, 'gstat'].replace(gstat_mapping, inplace=True)
    gstat_fix_id = df[(df['gstat'] == 'Out') &
                      (df['data_DK_pts'] > 0)].index
    df.loc[gstat_fix_id, 'gstat'] = 'Playing'

    return df


def clean_player_variables(player_df, indicator_cols, drop_player_vars):

    player_df.loc[:, 'posd_level2'] = player_df['posd']
    player_df.loc[:, 'posd'].replace(posd_parent_pos_mapping, inplace=True)
    player_df = fix_gstat(player_df)
    player_df.rename(columns={'posd': 'posd_level1'}, inplace=True)

    # I will drop one variable when I create the embedded layer to avoid the dummy variable trap
    just_dummies = pd.get_dummies(player_df[indicator_cols],
                                  drop_first=True)
    player_df = pd.concat([player_df, just_dummies], axis=1)
    drop_ftps_cols = ftp_cols_rename.values()
    drop_ftps_cols.remove('data_FD_pts')
    drop_ftps_cols.remove('data_DK_pts')

    player_df = player_df.drop(drop_ftps_cols +
                               drop_player_vars +
                               indicator_cols,
                               axis=1)
    player_df.loc[:, 'prc_pc'] = player_df['pc'] / player_df['pa']
    player_df.loc[:, 'prc_pc'] = player_df['prc_pc'].replace(np.inf, 0)
    player_df = calculate_player_age_days(player_df)
    player_df = player_df.fillna(0)

    return player_df


def calculate_player_age_days(player_df):

    player_df.loc[:, 'dob'] = pd.to_datetime(pd.Series(
        player_df['dob']), format="%m/%d/%Y", exact=False)
    player_df.loc[:, 'date'] = pd.to_datetime(pd.Series(
        player_df['date']), format="%m/%d/%Y", exact=False)

    date_difference = player_df['date'] - player_df['dob']
    player_df.loc[:, 'age_days'] = date_difference / datetime.timedelta(days=1)
    player_df = player_df.drop(['dob', 'date'], axis=1)

    return player_df

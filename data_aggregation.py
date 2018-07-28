import pandas as pd
import numpy as np
from ftps_functions import map_historical_ftps


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

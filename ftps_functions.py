import numpy as np
import pandas as pd
from configuration_mapping_file import *


def score_player(player, scoring_pbp):

    score = 0
    for stat in scoring_pbp.keys():
        score += scoring_pbp[stat](getattr(player, stat))
    return score


def fix_pass_stat(stat_col):
    if stat_col >= 12.:
        return stat_col + 3.
    else:
        return stat_col


def fix_rush_stat(stat_col):
    if stat_col >= 10.:
        return stat_col + 3.
    else:
        return stat_col


def calculate_ftps(df, scoring_dict):
    empty_df = pd.DataFrame()
    for stat in scoring_dict.keys():
        empty_df.loc[:, stat] = scoring_dict[stat](getattr(df, stat))
        if stat == 'py':
            empty_df.loc[:, stat] = empty_df[stat].apply(lambda x_child: fix_pass_stat(x_child))
        if stat == 'ry':
            empty_df.loc[:, stat] = empty_df[stat].apply(lambda x_child: fix_rush_stat(x_child))

    return pd.DataFrame.sum(empty_df, axis=1)


def check_pts_accuracy(df):
    df.loc[:, 'diff'] = np.abs(df['DK points'] - df['dk_ftps'])
    wrong_df = df[(df['DK points'] > 0) &
                  (df['diff'] > 0.01)]
    return wrong_df


def map_historical_ftps(dk_historical_pts_file, offense_df_merged):
    historical_salaries_df = pd.read_csv(dk_historical_pts_file)
    historical_salaries_df['Team'].replace(team_mapping, inplace=True)
    historical_salaries_df['Oppt'].replace(team_mapping, inplace=True)
    historical_salaries_df['Name'] = historical_salaries_df['Name'].astype(str)
    split_names = pd.DataFrame(historical_salaries_df.Name.str.split(', ', 1).tolist(),
                               columns=['lname', 'fname'])
    historical_salaries_df = pd.concat((historical_salaries_df, split_names), axis=1)

    if '_DK_' in dk_historical_pts_file:
        offense_df_merged_w_ftps = pd.merge(offense_df_merged,
                                            historical_salaries_df[['DK salary',
                                                                    'lname', 'fname', 'Year', 'Week', 'Team']],
                                            left_on=['lname', 'fname', 'year', 'wk', 'team'],
                                            right_on=['lname', 'fname', 'Year', 'Week', 'Team'],
                                            how='left').drop(['Year', 'Week', 'Team'], axis=1)
        #offense_df_merged_w_ftps.loc[:, 'dk_ftps'] = calculate_ftps(offense_df_merged_w_ftps, dk_scoring_by_col)
    else:
        offense_df_merged_w_ftps = pd.merge(offense_df_merged,
                                            historical_salaries_df[['FD salary',
                                                                    'lname', 'fname', 'Year', 'Week', 'Team']],
                                            left_on=['lname', 'fname', 'year', 'wk', 'team'],
                                            right_on=['lname', 'fname', 'Year', 'Week', 'Team'],
                                            how='left').drop(['Year', 'Week', 'Team'], axis=1)
        #offense_df_merged_w_ftps.loc[:, 'fd_ftps'] = score_player(offense_df_merged_w_ftps, fd_scoring_pbp)

    return offense_df_merged_w_ftps

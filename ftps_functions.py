import numpy as np
import pandas as pd


dk_scoring = {
    'py': lambda x: x * 0.04 + (3. if x >= 300. else 0.),
    'tdp': lambda x: x * 4.,
    'ints': lambda x: -1. * x,
    'ry': lambda x:  x * 0.1 + (3. if x >= 100. else 0.),
    'tdr': lambda x: x * 6.,
    'tdret': lambda x: x * 6.,
    'tdrec': lambda x: x * 6.,
    'recy': lambda x: x * 0.1,
    'rec': lambda x: x,
    'fuml': lambda x: -1 * x,
    #  'passing_twoptm'  : lambda x : 2*x,
    #  'rushing_twoptm' : lambda x : 2*x,
    #  'receiving_twoptm' : lambda x : 2*x
}

dk_scoring_by_col = {
    'py': lambda x: x * 0.04,
    'tdp': lambda x: x * 4.,
    'ints': lambda x: -1. * x,
    'ry': lambda x: x * 0.1,
    'tdr': lambda x: x * 6.,
    'tdret': lambda x: x * 6.,
    'tdrec': lambda x: x * 6.,
    'recy': lambda x: x * 0.1,
    'rec': lambda x: x,
    'fuml': lambda x: -1 * x,
    #  'passing_twoptm'  : lambda x : 2*x,
    #  'rushing_twoptm' : lambda x : 2*x,
    #  'receiving_twoptm' : lambda x : 2*x
}


def score_player(player):

    score = 0
    for stat in dk_scoring.keys():
        # if stat in scoring:
        score += dk_scoring[stat](getattr(player, stat))
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
            empty_df.loc[:, stat] = empty_df[stat].apply(lambda x: fix_pass_stat(x))
        if stat == 'ry':
            empty_df.loc[:, stat] = empty_df[stat].apply(lambda x: fix_rush_stat(x))

    return pd.DataFrame.sum(empty_df, axis=1)


def check_pts_accuracy(df):
    df.loc[:, 'diff'] = np.abs(df['DK points'] - df['dk_ftps'])
    wrong_df = df[(df['DK points'] > 0) &
                  (df['diff'] > 0.01)]
    return wrong_df

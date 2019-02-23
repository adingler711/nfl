import pandas as pd
import numpy as np
from sklearn import linear_model
from scipy.stats import norm


def compute_leverage_score(player_file_historical, roto_player_ownership, dk_slate_salaries,
                           roto_projections, pos_ownership_dict, yr_filter):

    player_df = pd.read_csv(player_file_historical)
    ownership_df = pd.read_csv(roto_player_ownership)

    player_df = player_df[['year', 'wk', 'pname', 'data_DK_pts']]
    player_df = player_df[player_df['year'] >= yr_filter]
    player_df = player_df.groupby('pname')['data_DK_pts'].std().reset_index()
    player_df.columns = ['pname', 'dk_std']
    player_df.loc[:, 'dk_std'] = player_df['dk_std'].fillna(0)

    roto_df = pd.read_csv(roto_projections)
    roto_df = roto_df[roto_df['fpts'] >= 5].reset_index(drop=True)
    roto_df = roto_df[(roto_df['pos'] != 'DST') &
                      (roto_df['pos'] != 'K')].reset_index(drop=True)
    names_df = pd.DataFrame(roto_df.name.str.split(' ' ,1).tolist(),
                            columns = ['fname' ,'lname'])

    roto_df = pd.concat((roto_df, names_df), axis=1)
    roto_df.loc[:, 'pname'] = roto_df['fname'].astype(str).str[0] + '.' + roto_df["lname"]
    roto_df = pd.merge(roto_df, player_df, on='pname', how='left')

    dk_slate_salaries_df = pd.read_csv(dk_slate_salaries)

    dk_names = pd.DataFrame(dk_slate_salaries_df.Name.str.split(' ' ,1).tolist(),
                            columns = ['fname', 'lname'])
    dk_names = pd.concat((dk_names,
                          pd.DataFrame(dk_names.lname.str.split(' ' ,1)
                                       .tolist()).drop(1, axis=1)),
                         axis=1).drop('lname', axis=1)
    dk_names = dk_names['fname'] + ' ' + dk_names[0]
    dk_slate_salaries_df.loc[:, 'Name'] = dk_names

    if 'JuJu Smith-Schuster' in list(dk_slate_salaries_df['Name']):
        dk_slate_salaries_df.loc[:, 'Name'] = dk_slate_salaries_df[
            'Name'].replace('JuJu Smith-Schuster', 'Juju Smith-Schuster')

    roto_df = pd.merge(roto_df, dk_slate_salaries_df[['Name', 'Salary']],
                       left_on='name', right_on='Name', how='left')

    if roto_df[roto_df['Name'].isnull()].shape[0] > 0:
        print roto_df[roto_df['Name'].isnull()]
        roto_df = roto_df.dropna()
        roto_df = roto_df.reset_index(drop=False)

    pos_std_mean = roto_df.groupby('pos')['dk_std'].mean().reset_index()
    std_null = roto_df[roto_df['dk_std'].isnull()].index

    for player in std_null:
        player_pos = roto_df.loc[player]['pos']
        roto_df.loc[player, 'dk_std'] = pos_std_mean[pos_std_mean['pos'] == player_pos]['dk_std'].values[0]

    wr_salary_range = np.linspace(3000, 9000, 500)
    wr_pts_range = np.linspace(26, 31., len(wr_salary_range))

    qb_salary_range = np.linspace(5000, 9000, 500)
    qb_pts_range = np.linspace(31.2, 34, len(qb_salary_range))

    te_salary_range = np.linspace(3000, 7000, 500)
    te_pts_range = np.linspace(24, 29, len(te_salary_range))

    rb_salary_range = np.linspace(3000, 9000, 500)
    rb_pts_range = np.linspace(28, 32, len(rb_salary_range))

    # Create linear regression object
    regr = linear_model.LinearRegression()

    pos_dict = {'QB': regr.fit(qb_salary_range.reshape(-1, 1), qb_pts_range.reshape(-1, 1)),
                'WR': regr.fit(wr_salary_range.reshape(-1, 1), wr_pts_range.reshape(-1, 1)),
                'TE': regr.fit(te_salary_range.reshape(-1, 1), te_pts_range.reshape(-1, 1)),
                'RB': regr.fit(rb_salary_range.reshape(-1, 1), rb_pts_range.reshape(-1, 1)),
                }

    for player in range(0, roto_df.shape[0]):
        pos = roto_df.loc[player]['pos']
        player_salary = roto_df.loc[player]['Salary']
        roto_df.loc[player, 'target'] = pos_dict[pos].predict(player_salary)[0]

    for player in range(0, roto_df.shape[0]):
        roto_df.loc[player, 'target_prob'] = np.round(norm.sf(roto_df.loc[player]['target'],
                                                              roto_df.loc[player]['fpts'],
                                                              roto_df.loc[player]['dk_std']
                                                              ), 5)

    pos_target_odds = roto_df.groupby('pos')['target_prob'].sum().reset_index()

    for player in range(0, roto_df.shape[0]):
        pos = roto_df.loc[player]['pos']
        player_prob = roto_df.loc[player]['target_prob']
        target_prob = pos_target_odds[pos_target_odds['pos'] == pos]['target_prob'].values[0]
        roto_df.loc[player, 'tar_prob'] = np.round(player_prob / target_prob, 5)
        roto_df.loc[player, 'imp_own'] = np.round((player_prob / target_prob) * pos_ownership_dict[pos], 5)

    roto_df = pd.merge(roto_df, ownership_df, left_on='Name', right_on='Player', how='left').fillna(0)
    roto_df.loc[:, 'leverage'] = np.round(roto_df['imp_own'] / roto_df['pOWN'], 5)

    return roto_df


def calculate_own_weights(roto_df_subset, limit,
                          mypos_ownership, pos):

    above_limit = roto_df_subset[roto_df_subset['my_own'] >= limit]
    above_count = above_limit.shape[0]
    own_exp = mypos_ownership - (limit * above_count)

    below_exposure = roto_df_subset[roto_df_subset['my_own'] < limit]
    above_limit.loc[:, 'my_own'] = limit

    below_exposure.loc[:, 'my_own'] = (below_exposure['leverage'] /
                                       (below_exposure['leverage'].sum() /
                                        own_exp))

    exposure_df = pd.concat((above_limit, below_exposure))

    return exposure_df


def calculate_pos_own_weights(roto_df, pos_list,
                              mypos_ownership, limit, pos):

    roto_df_subset = roto_df[roto_df['Name'].isin(pos_list)].reset_index(drop=True)

    #for pos in mypos_ownership_dict:
    roto_df_subset.loc[:, 'my_own'] = (roto_df_subset['leverage'] /
                                       (roto_df_subset['leverage'].sum() /
                                        mypos_ownership))

    while roto_df_subset['my_own'].max() > limit:
        roto_df_subset = calculate_own_weights(roto_df_subset, limit,
                                               mypos_ownership, pos)

    return roto_df_subset


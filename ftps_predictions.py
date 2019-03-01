from sklearn.model_selection import train_test_split
import matplotlib.pylab as plt
import pandas as pd
import numpy as np
#from keras.models import Sequential
#from keras.layers import Dense
#from keras.layers import LSTM
#from keras.layers import Dropout
from matplotlib import pyplot
#from keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA


def add_pca_components(player_df, pca_col_dict):
    for col_type in pca_col_dict:
        cols = pca_col_dict[col_type]
        col_names = [col_type + '_1']
        for i in range(2, len(cols)):
            pca = PCA(n_components=i)
            pca.fit(player_df[cols])
            cum_sum = np.cumsum(pca.explained_variance_ratio_)[-1]
            col_names.append(col_type + '_' + str(i))
            if cum_sum >= 0.80:
                components = pca.transform(player_df[cols])
                df = pd.DataFrame(components)
                df.columns = col_names
                player_df = pd.concat((player_df, df), axis=1).drop(cols, axis=1)
                break

    return player_df


def prepare_xgb_data(player_file, index_cols,
                     n_games, cur_game_vegas_info, y_var_scaled, binary_cut=False):

    player_df = pd.read_csv(player_file)
    if binary_cut is True:
        player_df = player_df[player_df['data_DK_pts_binary'] == 1]
        player_df = player_df.reset_index(drop=True)

    prev_game_stats = [col for col in player_df.columns if '(t-1)' in col]
    prev_game_stats = [col for col in prev_game_stats if 'fd_' not in col]
    prev_game_stats = [col for col in prev_game_stats if 'FD' not in col]
    prev_game_stats = list(set(prev_game_stats) - set(index_cols))

    for i, v in enumerate(prev_game_stats):
        prev_game_stats[i] = v.replace("(t-1)", "")

    # make sure the weighted always starts in same order
    player_df = compute_weighted_average(player_df, prev_game_stats,
                                         n_games, start_t=1, end_t=n_games)

    cur_game_out = [col for col in player_df.columns if 'out' in col]
    cur_game_out = [col for col in cur_game_out if '_weighted' not in col]
    cur_game_out = [col for col in cur_game_out if '(t' not in col]
    prev_weighted_cols = [col for col in player_df.columns if '_weighted' in col]

    if 'DK' in y_var_scaled:
        cur_game_out = [col for col in cur_game_out if 'FD' not in col]
        cur_game_out = [col for col in cur_game_out if 'fd_' not in col]

    else:
        cur_game_out = [col for col in cur_game_out if 'DK' not in col]
        cur_game_out = [col for col in cur_game_out if 'dk_' not in col]

    x_pred_cols = cur_game_out + cur_game_vegas_info + prev_weighted_cols
    if y_var_scaled in x_pred_cols:
        x_pred_cols.remove(y_var_scaled)

    player_df.loc[:, 'year_wk'] = (player_df['year'].astype(str) + '.' +
                                   player_df['wk'].astype(str).str.zfill(2)).astype(float)

    return player_df[index_cols + ['year_wk', y_var_scaled] + x_pred_cols], x_pred_cols


def prepare_data_cur_wk(player_file, index_cols,
                        n_games, cur_game_vegas_info, y_var_scaled, binary_cut=False):

    player_df = pd.read_csv(player_file)

    if binary_cut is True:
        player_df = player_df[player_df['data_DK_pts_binary'] == 1]
        player_df = player_df.reset_index(drop=True)

    prev_game_stats = [col for col in player_df.columns if '(t-1)' in col]
    prev_game_stats = [col for col in prev_game_stats if 'fd_' not in col]
    prev_game_stats = [col for col in prev_game_stats if 'FD' not in col]
    prev_game_stats = [col for col in prev_game_stats if '_out' not in col]
    prev_game_stats = list(set(prev_game_stats) - set(index_cols))

    for i, v in enumerate(prev_game_stats):
        prev_game_stats[i] = v.replace("(t-1)", "")

    # make sure the weighted always starts in same order
    player_df = compute_weighted_average_cur_wk(player_df, prev_game_stats,
                                                n_games, start_t=1, end_t=n_games)

    cur_game_out = [col for col in player_df.columns if 'out' in col]
    cur_game_out = [col for col in cur_game_out if '_weighted' not in col]
    cur_game_out = [col for col in cur_game_out if '(t' not in col]
    prev_weighted_cols = [col for col in player_df.columns if '_weighted' in col]

    if 'DK' in y_var_scaled:
        cur_game_out = [col for col in cur_game_out if 'FD' not in col]
        cur_game_out = [col for col in cur_game_out if 'fd_' not in col]

    else:
        cur_game_out = [col for col in cur_game_out if 'DK' not in col]
        cur_game_out = [col for col in cur_game_out if 'dk_' not in col]

    x_pred_cols = cur_game_out + cur_game_vegas_info + prev_weighted_cols
    if y_var_scaled in x_pred_cols:
        x_pred_cols.remove(y_var_scaled)

    player_df.loc[:, 'year_wk'] = (player_df['year'].astype(str) + '.' +
                                   player_df['wk'].astype(str).str.zfill(2)).astype(float)

    return player_df[index_cols + ['year_wk', y_var_scaled] + x_pred_cols], x_pred_cols


def compute_weighted_average_cur_wk(player_df, cur_game_stats, n_games_, start_t=1, end_t=4):

    test_list = np.array(range(1, n_games_ + 1))
    scaler = MinMaxScaler(feature_range=(0.5, 2))
    scaled = scaler.fit_transform(test_list.reshape(-1, 1))
    transformed_weights = scaled / np.sum(scaled)
    n_samples = player_df.shape[0]

    x_col_dict = dict()
    drop_vars = []
    for x_cols in cur_game_stats:
        x_cols_list = [col for col in player_df.columns if x_cols in col]
        x_cols_list = [col for col in x_cols_list if '(t-' + str(n_games_) not in col]
        x_cols_list = [col for col in x_cols_list if '_out' not in col]
        drop_vars = drop_vars + x_cols_list

        x_col_weighted_sum = np.zeros(n_samples)
        for i in range(start_t - 1, end_t):
            x_col_weighted_sum += np.array(player_df[x_cols_list[i]] * transformed_weights[i])

        x_col_dict[x_cols + '_weighted'] = x_col_weighted_sum

    x_col_df = pd.DataFrame(x_col_dict.values()).transpose()
    x_col_df.columns = x_col_dict.keys()

    player_df = pd.concat((player_df, x_col_df), axis=1)

    return player_df


def create_pca_col_dict(x_pred_cols):
    # Create PCA compenents for the different group of variables!
    off_cols = [col for col in x_pred_cols if 'offense_normalized' in col]
    off_cols = [col for col in off_cols if 'opp_var' not in col]

    #def_cols = [col for col in x_pred_cols if 'defense_normalized' in col]
    #def_cols = [col for col in def_cols if 'opp_var' not in col]

    opp_var_def_cols = [col for col in x_pred_cols if 'opp_var_defense' in col]
    #opp_var_off_cols = [col for col in x_pred_cols if 'opp_var_offense' in col]

    out_cols = [col for col in x_pred_cols if '_out' in col]
    out_weighted_cols = [col for col in out_cols if '_weighted' in col]
    #cur_game_out_cols = [col for col in out_cols if '_weighted' not in col]

    player_posd1_cols = [col for col in x_pred_cols if 'posd_level1_' in col]
    player_posd2_cols = [col for col in x_pred_cols if 'posd_level2_' in col]
    player_dk_posd2_cols = [col for col in x_pred_cols if 'DK_pts_level2' in col]

    player_dk_expected_cols = [col for col in x_pred_cols if 'dk_expected_' in col]
    player_dk_expected_cols = [col for col in player_dk_expected_cols if '_out' not in col]
    #player_dk_team_expected_cols = [col for col in player_dk_expected_cols if '_team_' in col]
    #player_dk_opp_expected_cols = [col for col in player_dk_expected_cols if '_opp_' in col]
    #player_dk_expected_cols = [col for col in player_dk_expected_cols if '_opp_' not in col]
    #player_dk_expected_cols = [col for col in player_dk_expected_cols if '_team_' not in col]

    player_dk_actual_cols = [col for col in x_pred_cols if 'dk_actual_' in col]
    player_dk_actual_cols = [col for col in player_dk_actual_cols if '_out' not in col]
    #player_dk_team_actual_cols = [col for col in player_dk_actual_cols if '_team_' in col]
    #player_dk_opp_actual_cols = [col for col in player_dk_actual_cols if '_opp_' in col]
    #player_dk_actual_cols = [col for col in player_dk_actual_cols if '_opp_' not in col]
    #player_dk_actual_cols = [col for col in player_dk_actual_cols if '_team_' not in col]

    #team_totals_cols = [col for col in x_pred_cols if '_team_total_scaled_weighted' in col]
    #team_totals_cols = [col for col in team_totals_cols if 'actual' not in col]
    #team_totals_cols = [col for col in team_totals_cols if 'expected' not in col]

    #opp_totals_cols = [col for col in x_pred_cols if '_opp_total_scaled_weighted' in col]
    #opp_totals_cols = [col for col in opp_totals_cols if 'actual' not in col]
    #opp_totals_cols = [col for col in opp_totals_cols if 'expected' not in col]

    # player_posd1_pts_cols = [col for col in x_pred_cols if '_data_DK_pts_scaled_weighted' in col]

    counting_stats = ['player_ra_scaled_weighted', 'player_rush_redzone_ra_scaled_weighted',
                      'player_snp_scaled_weighted',
                      'player_redzone_touches_scaled_weighted', 'player_py_bonus_scaled_weighted',
                      'player_pass_redzone_trg_scaled_weighted', 'player_prc_pc_scaled_weighted']

    #game_info = ['ptsh_scaled_weighted', 'sprv_scaled_weighted',
    #             'ou_scaled_weighted', 'home_indicator_scaled_weighted',
    #             'ptsv_scaled_weighted']

    pca_col_dict = {'pca_off': off_cols,
                    #'pca_def': def_cols,
                    'pca_opp_var_def_cols': opp_var_def_cols,
                    #'pca_opp_var_off_cols': opp_var_off_cols,
                    # 'pca_cur_game_out_cols': cur_game_out_cols,
                    #'pca_out_weighted_cols': out_weighted_cols,
                    'pca_player_posd1_cols': player_posd1_cols,
                    'pca_player_posd2_cols': player_posd2_cols,
                    'pca_player_dk_posd2_cols': player_dk_posd2_cols,
                    'pca_player_dk_expected_cols': player_dk_expected_cols,
                    #'pca_player_dk_opp_expected_cols': player_dk_opp_expected_cols,
                    #'pca_player_dk_team_expected_cols': player_dk_team_expected_cols,
                    #'pca_player_dk_team_actual_cols': player_dk_team_actual_cols,
                    #'pca_player_dk_opp_actual_cols': player_dk_opp_actual_cols,
                    'pca_player_dk_actual_cols': player_dk_actual_cols,
                    #'pca_team_totals_cols': team_totals_cols,
                    #'pca_opp_totals_cols': opp_totals_cols,
                    # 'pca_player_posd1_pts_cols': player_posd1_pts_cols,
                    'pca_counting_stats': counting_stats
                    #'pca_game_info': game_info
                    }

    return pca_col_dict


def create_pca_components(player_df, x_pred_cols):

    # Create PCA compenents for the different group of variables!
    pca_col_dict = create_pca_col_dict(x_pred_cols)

    # x_pred_cols2 = x_pred_cols[:]
    for key in pca_col_dict:
        x_pred_cols = list(set(x_pred_cols) - set(pca_col_dict[key]))

    player_df = add_pca_components(player_df, pca_col_dict)
    pca_cols = [col for col in player_df.columns if 'pca_' in col]
    x_pred_cols = x_pred_cols + pca_cols

    return player_df, x_pred_cols


def compute_weighted_average(player_df, cur_game_stats, n_games_, start_t=1, end_t=4):

    test_list = np.array(range(1, n_games_ + 1))
    scaler = MinMaxScaler(feature_range=(0.5, 2))
    scaled = scaler.fit_transform(test_list.reshape(-1, 1))
    transformed_weights = scaled / np.sum(scaled)
    n_samples = player_df.shape[0]

    x_col_dict = dict()
    drop_vars = []
    for x_cols in cur_game_stats:
        x_cols_list = [col for col in player_df.columns if x_cols + '(' in col]
        x_cols_list = [col for col in x_cols_list if '(t-' in col]
        drop_vars = drop_vars + x_cols_list

        x_col_weighted_sum = np.zeros(n_samples)
        for i in range(start_t - 1, end_t):
            x_col_weighted_sum += np.array(player_df[x_cols_list[i]] * transformed_weights[i])

        x_col_dict[x_cols + '_weighted'] = x_col_weighted_sum

    x_col_df = pd.DataFrame(x_col_dict.values()).transpose()
    x_col_df.columns = x_col_dict.keys()

    player_df = pd.concat((player_df, x_col_df), axis=1)

    return player_df



def x_cols_to_remove(player_df, index_cols, y_var):

    cur_game_stats = [col for col in player_df.columns if '(t)' in col]

    if 'DK' in y_var:
        ftp_cols_remove = [col for col in player_df.columns if 'FD' in col]
        ftp_cols_remove =[col for col in ftp_cols_remove if 'fd_' not in col]

    else:
        ftp_cols_remove = [col for col in player_df.columns if 'DK' in col]
        ftp_cols_remove =[col for col in ftp_cols_remove if 'dk_' not in col]

    x_cols_remove_list = list(set(cur_game_stats + ftp_cols_remove))

    return x_cols_remove_list


def find_embedded_col_values(player_df, embedded_var):

    embedded_cols = [col for col in player_df.columns if embedded_var in col]
    embedded_cols_keep = [col for col in embedded_cols if '(t)' not in col]
    embedded_values = player_df[embedded_cols_keep].values  # [:, :-1]

    return embedded_values, embedded_cols


def print_results(model, history, testx, testy):

    pred_classes = model.predict_classes(testx, verbose=0)
    cm_normalized = pd.crosstab(testy, pred_classes,
                                rownames=['True'],
                                colnames=['Predicted']).apply(
        lambda r: 100.0 * r / r.sum())

    print("Normalized Confusion Matrix after optimal threshold:")
    print cm_normalized

    # summarize history for accuracy
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()


def prepare_prediction_data(player_file_main, embedded_list_prefix,
                            cur_game_vegas_info,
                            index_cols, y_var, n_games, train_percent):

    player_df = pd.read_csv(player_file_main)
    #player_df = player_df[player_df['data_DK_pts_binary'] == 1]

    cols_remove = []
    for prefix in embedded_list_prefix:
        pre_fix_list = [col for col in player_df.columns if prefix in col]
        cols_remove = cols_remove + pre_fix_list

    cols_remove = list(set(cols_remove))
    player_df = player_df.drop(cols_remove, axis=1)

    player_df = fix_cur_game_cols(player_df, n_games, cur_game_vegas_info, index_cols)


    [train_X, test_X, train_y, test_y, n_features
     ] = split_data(player_df, index_cols, y_var, n_games, train_percent)

    return [train_X, test_X, train_y, test_y, n_features
            ]


def fix_cur_game_cols(player_df, n_games, cur_game_vegas_info, index_cols):

    cur_game_stats = [col for col in player_df.columns if '(t' not in col] + cur_game_vegas_info
    cur_game_stats = set(cur_game_stats) - set(index_cols)

    for col in cur_game_stats:
        for i in range(n_games, -1, -1):

            if col in cur_game_vegas_info:
                if i > 0:
                    player_df.rename(columns={col + '(t-' + str(i) + ')': col + '(t-' + str(i + 1) + ')'}, inplace=True)
                else:
                    player_df.rename(columns={col + '(t)': col + '(t-' + str(i + 1) + ')'}, inplace=True)
            else:
                if i > 0:
                    player_df.rename(columns={col + '(t-' + str(i) + ')': col + '(t-' + str(i + 1) + ')'}, inplace=True)
                else:
                    player_df.rename(columns={col: col + '(t-' + str(i + 1) + ')'}, inplace=True)

    return player_df


def split_data(data_df, index_cols, y_var, n_games, train_percent):

    x_cols_remove_list = x_cols_to_remove(data_df,
                                          index_cols,
                                          y_var
                                          )

    y_var_scaled = y_var + '_scaled(t)'
    if y_var_scaled in x_cols_remove_list:
        x_cols_remove_list.remove(y_var_scaled)

    data_df = data_df.drop(x_cols_remove_list, axis=1)

    col_order_list = []
    for i in range(n_games, 0, -1):
        col_order_list = col_order_list + [col for col in data_df.columns
                                           if '(t-' + str(i) + ')' in col]

    [train_X, test_X,
     train_y, test_y] = train_test_split(np.array(data_df[col_order_list]),
                                         np.array(data_df[y_var_scaled]),
                                         train_size=train_percent, test_size=1 - train_percent)

    n_features = len(col_order_list) / n_games
    print(train_X.shape, len(train_X), train_y.shape)
    # reshape input to be 3D [samples, timesteps, features]
    train_X = train_X.reshape((train_X.shape[0], n_games, n_features))
    test_X = test_X.reshape((test_X.shape[0], n_games, n_features))

    print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)

    return [train_X, test_X, train_y, test_y, n_features
            ]

def reshape_embedded_array(embedded_array, n_games):
    embedded_reshaped = embedded_array.reshape((embedded_array.shape[0],
                                                n_games,
                                                len(embedded_array[0]) /
                                                n_games))
    return embedded_reshaped



def hidden_node_calc(train_X, alpha, output_dim=1):

    Ns = len(train_X)
    Ni = train_X.shape[1] #* train_X.shape[2]

    return Ns / (alpha * (Ni + output_dim))


def denormalize(normalized, y_var, historical_ftp_file,
                cap=30, floor=5):
    if 'DK' in y_var:
        ftp_col = 'data_DK_pts'
    else:
        ftp_col = 'data_FD_pts'

    historical_ftp_df = pd.read_csv(historical_ftp_file,
                                    usecols=[ftp_col])
    historical_ftp_df = historical_ftp_df[(historical_ftp_df[ftp_col] <= cap)]

    max_ = historical_ftp_df[ftp_col].max()
    min_ = historical_ftp_df[ftp_col].min()

    return (normalized * (max_ - min_) + min_)


def create_graph(train_X, train_y, test_X, test_y, y_var,
                 batch_size, learner='adam', n_epoch=100,
                 drouput_prct=0.50, n_patience=10,
                 alpha=2, output_dim=1):

    hidden_nodes = hidden_node_calc(train_X, alpha=alpha, output_dim=output_dim)

    # design network
    model = Sequential()
    model.add(LSTM(hidden_nodes,
                   # batch_input_shape=(batch_size, train_X.shape[1], 1),
                   activation='relu',
                   return_sequences=True,
                   # recurrent_dropout=0.3,
                   # stateful=True,
                   input_shape=(train_X.shape[1],
                                train_X.shape[2])))
    model.add(Dropout(drouput_prct))
    model.add(LSTM(hidden_nodes / 2,
                   # batch_input_shape=(batch_size, train_X.shape[1], 1),
                   activation='relu',
                   # recurrent_dropout=0.3,
                   # stateful=True
                   ))
    model.add(Dropout(drouput_prct))
    model.add(Dense(output_dim))

    if 'binary' in y_var:
        model.compile(loss='binary_crossentropy',
                      optimizer=learner,
                      metrics=['accuracy'])
        stopping_monitor = 'val_acc'

    else:
        model.compile(loss='mae',
                      optimizer=learner)
        stopping_monitor = 'val_loss'

    # fit network
    early_stopping = EarlyStopping(monitor=stopping_monitor,
                                   patience=n_patience)
    history = model.fit(train_X, train_y,
                        epochs=n_epoch, batch_size=batch_size,
                        validation_data=(test_X, test_y),
                        callbacks=[early_stopping],
                        verbose=1, shuffle=False)

    return model, history


def plot_history(history, y_var):
    # plot history
    if 'binary' in y_var:
        pyplot.plot(history.history['acc'], label='train')
        pyplot.plot(history.history['val_acc'], label='test')

    else:
        pyplot.plot(history.history['loss'], label='train')
        pyplot.plot(history.history['val_loss'], label='test')

    pyplot.legend()
    pyplot.show()


def cur_week_opp(player_df, cur_opp_vars):

    last_game_opp = player_df.groupby('opp')['gid'].max().reset_index()
    opp_stats_df = pd.DataFrame()

    for opp_game in range(0, last_game_opp.shape[0]):
        temp_opp = last_game_opp.loc[opp_game]
        player_df_temp = player_df[(player_df['team'] == temp_opp['opp']) &
                                   (player_df['gid'] == temp_opp['gid'])][['opp'] + cur_opp_vars].drop_duplicates()

        opp_stats_df = pd.concat((opp_stats_df, player_df_temp))

    return opp_stats_df


def expected_opp_analysis(week, player_file_historical,
                          armchair_data_path_main,
                          schedule_file_main,
                          player_file_main):

    df = pd.read_csv(player_file_historical)

    expected_cols = [col for col in df.columns if '_expected' in col]
    expected_cols = [col for col in expected_cols if '_dk_' in col]
    expected_cols = [col for col in expected_cols if '(t-1)' in col]
    expected_cols = [col for col in expected_cols if '_out' not in col]

    for i, v in enumerate(expected_cols):
        expected_cols[i] = v.replace("(t-1)", "")

    df = compute_weighted_average(df, expected_cols, 6, start_t=1, end_t=6)
    expected_weighted_cols = [col for col in df if '_weighted' in col]

    opp_expected_cols = [col for col in df.columns if 'opp_var_defense_normalized' in col]
    opp_expected_cols = [col for col in opp_expected_cols if '_dk_' in col]
    opp_expected_cols = [col for col in opp_expected_cols if '(t-1)' in col]

    for i, v in enumerate(opp_expected_cols):
        opp_expected_cols[i] = v.replace("(t-1)", "")

    df = compute_weighted_average(df, opp_expected_cols, 6, start_t=1, end_t=6)

    opp_expected_weighted_cols = [col for col in df if '_weighted' in col]
    opp_expected_weighted_cols = [col for col in opp_expected_weighted_cols if 'opp_var' in col]

    schedule_df = pd.read_csv(armchair_data_path_main + schedule_file_main, low_memory=False)
    cur_wk_df = schedule_df[(schedule_df['seas'] == 2018) &
                            (schedule_df['wk'] == week)]

    player_cur_df = pd.read_csv(armchair_data_path_main + player_file_main, low_memory=False)
    cur_wk_teams = list(cur_wk_df['v']) + list(cur_wk_df['h'])
    player_cur_df = player_cur_df[player_cur_df['cteam'].isin(cur_wk_teams)]
    player_cur_df = player_cur_df[player_cur_df['pos1'].isin(['RB', 'TE', 'WR', 'QB'])]
    cur_players = list(player_cur_df['player'])
    df = df[df['player'].isin(cur_players)]
    df = df[(df['year'] == 2018) & (df['wk'] == week - 1)]
    df.loc[:, 'exp_weighted_sum'] = pd.DataFrame.sum(df[expected_weighted_cols], axis=1)
    df.loc[:, 'opp_exp_weighted_sum'] = pd.DataFrame.sum(df[opp_expected_weighted_cols], axis=1)

    # identify opp current week sos values!!!
    opp_sos_df = df[['opp', 'opp_exp_weighted_sum'] + opp_expected_weighted_cols].drop_duplicates()

    df = df.sort_values('exp_weighted_sum', ascending=False)
    keep_cols = ['team', 'opp', 'pname', 'exp_weighted_sum'] + expected_weighted_cols
    df = df[keep_cols]

    df = pd.merge(df, opp_sos_df, on='opp', how='left')

    return df


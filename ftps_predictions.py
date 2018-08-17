from sklearn.model_selection import train_test_split
import matplotlib.pylab as plt
import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from matplotlib import pyplot
from keras.callbacks import EarlyStopping


def x_cols_to_remove(player_df, index_cols, y_var):

    cur_game_stats = [col for col in player_df.columns if '(t)' in col]

    if 'DK' in y_var:
        ftp_cols_remove = [col for col in player_df.columns if 'FD' in col]
    else:
        ftp_cols_remove = [col for col in player_df.columns if 'DK' in col]

    x_cols_remove_list = list(set(cur_game_stats + ftp_cols_remove)) + index_cols
    n_features = len(set(cur_game_stats) - set(ftp_cols_remove))

    return x_cols_remove_list, n_features


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
                            index_cols, y_var, n_games, train_percent):

    player_df = pd.read_csv(player_file_main)

    cols_remove = []
    for prefix in embedded_list_prefix:
        pre_fix_list = [col for col in player_df.columns if prefix in col]
        cols_remove = cols_remove + pre_fix_list

    cols_remove = list(set(cols_remove))
    player_df = player_df.drop(cols_remove, axis=1)

    [train_X, test_X, train_y, test_y,
     #train_posd_level1, test_posd_level1,
     #train_posd_level2, test_posd_level2
     ] = split_data(player_df, index_cols, y_var, n_games, train_percent)

    return [train_X, test_X, train_y, test_y
           # train_posd_level1, test_posd_level1,
           # train_posd_level2, test_posd_level2
            ]


def split_data(data_df, index_cols, y_var, n_games, train_percent):

    #posd_level1, posd_level1_cols = find_embedded_col_values(data_df, 'posd_level1')
    #posd_level2, posd_level2_cols = find_embedded_col_values(data_df, 'posd_level2')
    #embedded_cols = posd_level1_cols + posd_level2_cols

    [x_cols_remove_list,
     n_features] = x_cols_to_remove(data_df,#.drop(embedded_cols, axis=1),
                                    index_cols, y_var)

    y_var_scaled = y_var + '_scaled(t)'

    [train_X, test_X,
     #train_posd_level1, test_posd_level1,
     #train_posd_level2, test_posd_level2,
     train_y, test_y] = train_test_split(np.array(data_df.drop(x_cols_remove_list, # +
                                                                 #embedded_cols,
                                                                 axis=1)),
                                         #posd_level1,
                                         #posd_level2,
                                         np.array(data_df[y_var_scaled]),
                                         train_size=train_percent, test_size=1 - train_percent)

    print(train_X.shape, len(train_X), train_y.shape)
    # reshape input to be 3D [samples, timesteps, features]
    train_X = train_X.reshape((train_X.shape[0], n_games, n_features))
    test_X = test_X.reshape((test_X.shape[0], n_games, n_features))

    #train_posd_level1 = reshape_embedded_array(train_posd_level1, n_games)
    #test_posd_level1 = reshape_embedded_array(test_posd_level1, n_games)
    #train_posd_level2 = reshape_embedded_array(train_posd_level2, n_games)
    #test_posd_level2 = reshape_embedded_array(test_posd_level2, n_games)

    print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)

    return [train_X, test_X, train_y, test_y
            #train_posd_level1, test_posd_level1,
            #train_posd_level2, test_posd_level2
            ]

def reshape_embedded_array(embedded_array, n_games):
    embedded_reshaped = embedded_array.reshape((embedded_array.shape[0],
                                                n_games,
                                                len(embedded_array[0]) /
                                                n_games))
    return embedded_reshaped



def hidden_node_calc(train_X, alpha, output_dim=1):

    Ns = len(train_X)
    Ni = train_X.shape[1] * train_X.shape[2]

    return Ns / (alpha * (Ni + output_dim))


def denormalize(normalized, y_var, historical_ftp_file):
    if 'DK' in y_var:
        ftp_col = 'data_DK_pts'
    else:
        ftp_col = 'data_FD_pts'

    historical_ftp_df = pd.read_csv(historical_ftp_file,
                                    usecols=[ftp_col])

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
                        verbose=1, shuffle=True)

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
# Example is from: https://machinelearningmastery.com/multivariate-time-series-forecasting-lstms-keras/
# Tune LTSM: https://machinelearningmastery.com/tune-lstm-hyperparameters-keras-time-series-forecasting/
# Word Embeddings: https://machinelearningmastery.com/use-word-embedding-layers-deep-learning-keras/
import warnings
warnings.filterwarnings('ignore')
from data_preparation import *
import time

start = time.time()

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
                      'rec', 'recy', 'tdrec', 'tdret', 'fuml', 'pa', 'ra', 'trg']

player_results_df = create_player_data(cols_keep_main,
                                       armchair_data_path_main,
                                       dk_historical_pts_file_main,
                                       fd_historical_pts_file_main,
                                       injury_file_main,
                                       game_file_main,
                                       player_file_main,
                                       offense_file_main,
                                       schedule_file_main
                                      )

schedule_df = open_schedule(armchair_data_path_main, schedule_file_main)

offensive_skills_positions = ['WR', 'RB', 'QB', 'TE',
                              'FB', 'LWR', 'RWR', 'SWR']
drop_player_vars = ['col', 'jnum', 'dv', 'v', 'h',
                    'pos1', 'pos2', 'cteam']
indicator_cols = ['day', 'stad', 'wdir', 'cond', 'surf',
                  'details', 'pstat', 'gstat', 'opp',
                 'posd', 'posd_level2']
player_non_x_cols = ['gid', 'team', 'player',
                     'nflid', 'fname','lname',
                     'pname', 'wk']
save_ftp_cols = '20180808'

[team_ftps_allowed,
 player_df] = player_data_scoring_only(player_results_df,
                                       offensive_skills_positions,
                                       indicator_cols,
                                       drop_player_vars)

dk_ftp_player_cols = ['py', 'ints', 'tdp', 'ry', 'tdr',
                          'rec', 'recy', 'tdrec', 'tdret', 'fuml', 'pa', 'ra', 'trg']
n_games = 4
y_var = 'data_DK_pts'

#y_var_id = dk_ftp_player_cols.index(y_var)
[player_n_games_df,
 y_var_scaled,
 n_features] = create_scaled_lag_variables(player_df, n_games,
                                           dk_ftp_player_cols,
                                           y_var,
                                           binary_threshold=10)

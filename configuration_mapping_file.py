import numpy as np


n_games = 4
save_ftp_cols = '20180808'

dk_historical_pts_file_main = 'DFS_NFL_Salaries/DFS_DK_Historical_Salary.csv'
fd_historical_pts_file_main = 'DFS_NFL_Salaries/DFS_FD_Historical_Salary.csv'
armchair_data_path_main = 'armchair_analysis_data/'
injury_file_main = 'INJURY.csv'
game_file_main = 'GAME.csv'
player_file_main = 'PLAYER.csv'
offense_file_main = 'OFFENSE.csv'
schedule_file_main = 'SCHEDULE_2018.csv'
snap_file_main = 'SNAP.csv'

cols_keep_main = ['gid', 'team', 'player', 'year', 'posd', 'dcp', 'nflid',
                  'fp2', 'fp3', 'py', 'ints', 'tdp', 'ry', 'tdr', 'snp',
                  'rec', 'recy', 'tdrec', 'tdret', 'fuml', 'pa', 'ra', 'trg',
                  'ret', 'pc'] # 'conv'

offensive_skills_positions = ['WR', 'RB', 'QB', 'TE',
                              'FB', 'LWR', 'RWR', 'SWR']

drop_player_vars = ['col', 'jnum', 'dv', 'v', 'h',
                    'pos1', 'pos2', 'cteam', 'day',
                    'stad', 'wdir', 'cond', 'surf',
                    'details', 'pstat']

indicator_cols = ['gstat', 'posd_level1', 'posd_level2']

player_non_x_cols = ['gid', 'team', 'player',
                     'nflid', 'fname', 'lname',
                     'pname', 'wk']

player_idx_cols = ['gid', 'team', 'opp', 'player', 'year',
                   'nflid', 'fname', 'lname', 'pname']

ftp_cols = ['py', 'ints', 'tdp', 'ry', 'tdr', 'snp',
            'rec', 'recy', 'tdrec', 'tdret',
            'pa', 'ra', 'trg', 'ret', 'pc']

team_index_cols = ['year', 'wk', 'opp']


team_mapping = {'kan': 'KC',
                'min': 'MIN',
                'det': 'DET',
                'phi': 'PHI',
                'den': 'DEN',
                'atl': 'ATL',
                'gnb': 'GB',
                'buf': 'BUF',
                'cle': 'CLE',
                'lar': 'LA',
                'ten': 'TEN',
                'oak': 'OAK',
                'lac': 'LAC',
                'dal': 'DAL',
                'pit': 'PIT',
                'nor': 'NO',
                'car': 'CAR',
                'was': 'WAS',
                'chi': 'CHI',
                'ari': 'ARI',
                'nyj': 'NYJ',
                'nwe': 'NE',
                'jac': 'JAC',
                'sea': 'SEA',
                'bal': 'BAL',
                'nyg': 'NYG',
                'hou': 'HOU',
                'sfo': 'SF',
                'ind': 'IND',
                'cin': 'CIN',
                'mia': 'MIA',
                'tam': 'TB',
                'stl': 'STL',  # team moved
                'sdg': 'SD'  # team moved
                }

teams_moved = {'STL': 'LA',
               'SD': 'LAC'}

posd_parent_pos_mapping = {'FB': 'RB',
                           'LWR': 'WR',
                           'RWR': 'WR',
                           'SWR': 'WR'}

gstat_mapping = {'Probable\r': 'Probable',
                 'Questionable\r': 'Questionable',
                 'IR': 'Out',
                 'Suspend': 'Out',
                 np.nan: 'Playing'
                 }

ftp_cols_rename = {'fp2': 'data_FD_pts',
                   'fp3': 'data_DK_pts',
                   #  'DK points': 'actual_dk_pts',
                   'DK salary': 'actual_dk_salary',
                   #  'dk_ftps': 'calc_dk_pts',
                   #  'FD points': 'actual_FD_pts',
                   'FD salary': 'actual_fd_salary',
                   #  'fd_ftps': 'calc_pts'
                   }

dk_scoring_pbp = {
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
    'ry': lambda x:  x * 0.1,
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

fd_scoring_pbp = {
    'py': lambda x: x * 0.04,
    'tdp': lambda x: x * 4.,
    'ints': lambda x: -1. * x,
    'ry': lambda x:  x * 0.1,
    'tdr': lambda x: x * 6.,
    'tdret': lambda x: x * 6.,
    'tdrec': lambda x: x * 6.,
    'recy': lambda x: x * 0.1,
    'rec': lambda x: 0.5 * x,
    'fuml': lambda x: -2 * x,
    #  'passing_twoptm'  : lambda x : 2*x,
    #  'rushing_twoptm' : lambda x : 2*x,
    #  'receiving_twoptm' : lambda x : 2*x
}



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

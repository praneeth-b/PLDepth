from hyperopt import hp
from hyperopt.pyll import scope

lr_dict = {
    "lr": hp.loguniform('lr', -8, 0),
    # "multi": hp.uniform("mlp",0,0.5)
    "ranking_size": scope.int(hp.quniform("ranking_size", 2, 30, q=1))

}

info_dict = {

    "lr": hp.loguniform('lr', -8, 0),
    "lr_multi": hp.uniform("lr_multi", 0.05, 0.5),
    "batch_size": 1 + hp.randint('batch_size', 10),
    "ranking_size": 2 + hp.randint('ranking_size', 48),
    "rpi": 10 + hp.randint('rpi', 200) # fix to 50

}

sweep_config_i = {'method': 'bayes',
                'metric': {'goal': 'minimize', 'name': 'test_err'},
                'parameters': {
                    'batch_size': {'distribution': 'constant', 'value': 6},
                    'epochs': {'distribution': 'constant', 'value': 8},
                    'lr': {'distribution': 'log_uniform', 'max': -2, 'min': -4},
                    'ranking_size': {'distribution': 'constant', 'value':5},
                    'rpi': {'distribution': 'constant', 'value': 100},   ##  2000 // ranking_size
                    'lr_multi': {'distribution':'int_uniform', 'max':200, 'min':10}, #{'distribution': 'uniform', 'max': 0.5, 'min': 0},
                    'sampling_type': {'distribution':'constant', 'value':3},  # todo
                    'dataset_size': {'value':5000},
                    'seed':{'value':1}

                }
                }


sweep_config_t = {'method': 'bayes',
                'metric': {'goal': 'minimize', 'name': 'test_err'},
                'parameters': {
                    'batch_size': {'distribution': 'constant', 'value': 6},
                    'epochs': {'distribution': 'constant', 'value': 12},
                    'lr': {'distribution': 'log_uniform', 'max': -2, 'min': -4},
                    'ranking_size': {'distribution': 'int_uniform', 'max': 500, 'min': 4},
                    #'rpi': {'distribution': 'constant', 'value': 1},
                    'lr_multi': {'distribution':'constant', 'value':0.3}  ,#{'distribution': 'uniform', 'max': 0.5, 'min': 0},
                    'sampling_type': {'distribution':'constant', 'value':0},
                    'dataset_size': {'value':2150},
                        'seed':{'value':1}

                }
                }

sweep_config_pr = {'method': 'bayes',
                'metric': {'goal': 'minimize', 'name': 'test_err'},
                'parameters': {
                    'batch_size': {'distribution': 'constant', 'value': 6},
                    'epochs': {'distribution': 'constant', 'value': 8},
                    'lr': {'distribution': 'log_uniform', 'max': -2, 'min': -4},
                    'ranking_size': {'distribution': 'constant', 'value':5},
                    'rpi': {'distribution': 'constant', 'value': 100},
                    'lr_multi': {'distribution':'uniform', 'max':200, 'min':10}, #{'distribution': 'uniform', 'max': 0.5, 'min': 0},
                    'sampling_type': {'distribution':'constant', 'value':1},  # todo
                    'dataset_size': {'value':5000},
                    'seed':{'value':1}

                }
                }

activ_sweep = { 'method': 'bayes',
                'metric': {'goal': 'minimize', 'name': 'test_err'},
             'parameters': {
                'lr': {'distribution': 'log_uniform', 'max': -3, 'min': -7},
                'lr_multi': {'distribution': 'uniform', 'max': 0.95, 'min': 0.5},
                'ranking_size': {'distribution':'constant', 'value': 20},
                'batch_size': {'distribution': 'constant', 'value': 4},
                'epochs': {'distribution': 'constant', 'value': 6},
                 'rpi': {'distribution': 'constant', 'value':50},
                 'num_split':{'distribution': 'constant', 'value': 32}
             }
                }

activ_sweep2 = { 'method': 'bayes',
                'metric': {'goal': 'minimize', 'name': 'test_err'},
             'parameters': {
                'lr': {'distribution': 'log_uniform', 'max': -2, 'min': -5},
                'lr_multi': {'distribution': 'uniform', 'max': 1000, 'min': 10},
                'ranking_size': {"value":125},   # {'distribution': 'int_uniform', 'max': 500, 'min': 4},
                 'rpi':{"value":2},
                'batch_size': {'distribution': 'constant', 'value': 6},
                'epochs': {'distribution': 'constant', 'value': 7},
                 'num_split': {"value":16}, #{'distribution': 'constant', 'value': 32} ,  #
                 'ds_size' : {"value": 2000},  # [1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000]
                 'act_ds_size' : {"value": 5000},
                 'sampling_type': {'distribution':'constant', 'value':1},
                 'canny_sigma':{'value':1.8}
             }
                 }

rnd_base = { 'method': 'bayes',
                'metric': {'goal': 'minimize', 'name': 'test_err'},
             'parameters': {
                'lr': {'distribution': 'log_uniform', 'max': -1, 'min': -5},
                'lr_multi': {'distribution': 'uniform', 'max': 1000, 'min': 10},
                'ranking_size': {"value":125}, #{'distribution': 'int_uniform', 'max': 500, 'min': 4},
                'rpi':{"value":2},
                'batch_size': {'distribution': 'constant', 'value': 6},
                'epochs': {'distribution': 'constant', 'value': 7},
                'sampling_type': {'distribution':'constant', 'value':1},
                 'ds_size' : {"value": 2000},  # [1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000]
                'act_ds_size' : {"value": 5000},
                 'sampling_type': {'distribution':'constant', 'value':1}
	         #'dummy':{'distribution': 'uniform', 'max': 1000, 'min': 10}

             }
                 }




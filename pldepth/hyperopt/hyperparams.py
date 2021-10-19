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
                'metric': {'goal': 'minimize', 'name': 'val_loss'},
                'parameters': {
                    'batch_size': {'distribution': 'constant', 'value': 4},
                    'epochs': {'distribution': 'constant', 'value': 17},
                    'lr': {'distribution': 'log_uniform', 'max': -2, 'min': -5},
                    'ranking_size': {'distribution': 'int_uniform', 'max': 100, 'min': 10},
                    'rpi': {'distribution': 'constant', 'value': 1},
                    'lr_multi': {'distribution':'constant', 'value':0.35} , #{'distribution': 'uniform', 'max': 0.5, 'min': 0},
                    'sampling_type': {'distribution':'constant', 'value':1},
                    'dataset_size': {'value':3200}

                }
                }


sweep_config_t = {'method': 'bayes',
                'metric': {'goal': 'minimize', 'name': 'val_loss'},
                'parameters': {
                    'batch_size': {'distribution': 'constant', 'value': 4},
                    'epochs': {'distribution': 'constant', 'value': 8},
                    'lr': {'distribution': 'log_uniform', 'max': -2, 'min': -7},
                    'ranking_size': {'distribution': 'int_uniform', 'max': 30, 'min': 3},
                    'rpi': {'distribution': 'constant', 'value': 50},
                    'lr_multi': {'distribution':'constant', 'value':0.35}  ,#{'distribution': 'uniform', 'max': 0.5, 'min': 0},
                    'sampling_type': {'distribution':'constant', 'value':0},
                    'dataset_size': {'value':5355}

                }
                }

sweep_config_pr = {'method': 'bayes',
                'metric': {'goal': 'minimize', 'name': 'val_loss'},
                'parameters': {
                    'batch_size': {'distribution': 'constant', 'value': 4},
                    'epochs': {'distribution': 'constant', 'value': 17},
                    'lr': {'distribution': 'log_uniform', 'max': -2, 'min': -5},
                    'ranking_size': {'distribution': 'int_uniform', 'max': 100, 'min': 10},
                    'rpi': {'distribution': 'constant', 'value': 1},
                    'lr_multi': {'distribution':'constant', 'value':0.35}  ,#{'distribution': 'uniform', 'max': 0.5, 'min': 0},
                    'sampling_type': {'distribution':'constant', 'value':3},
                    'dataset_size': {'value':3200}

                }
                }

activ_sweep = { 'method': 'bayes',
                'metric': {'goal': 'minimize', 'name': 'val_loss'},
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
                'metric': {'goal': 'minimize', 'name': 'val_loss'},
             'parameters': {
                'lr': {'distribution': 'constant', 'value': 0.0033},
                'lr_multi': {'distribution': 'constant', 'value':0.4},
                'ranking_size': {'distribution': 'constant', 'value':20},
                'batch_size': {'distribution': 'constant', 'value': 4},
                'epochs': {'distribution': 'constant', 'value': 10},
                 'num_split': {"values": [16, 32, 56, 64]},  #{'distribution': 'constant', 'value': 32} ,  #
                 'ds_size' : {"value": 2000},  # [1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000]
                 'sampling_type': {'distribution':'constant', 'value':1},
                 'canny_sigma':{'distribution': 'uniform', 'max': 2, 'min': 0}
             }
                 }



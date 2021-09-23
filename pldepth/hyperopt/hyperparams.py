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

sweep_config = {'method': 'bayes',
                'metric': {'goal': 'minimize', 'name': 'val_loss'},
                'parameters': {
                    'batch_size': {'distribution': 'constant', 'value': 4},
                    'epochs': {'distribution': 'constant', 'value': 8},
                    'lr': {'distribution': 'log_uniform', 'max': -2, 'min': -7},
                    'ranking_size': {'distribution': 'int_uniform', 'max': 30, 'min': 3},
                    'rpi': {'distribution': 'int_uniform', 'max': 200, 'min': 10},
                    'lr_multi': {'distribution': 'uniform', 'max': 0.5, 'min': 0},
                    'sampling_type': {'value'}

                }
                }

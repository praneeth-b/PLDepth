from hyperopt import tpe, fmin, Trials
from pldepth.hyperopt.hyper_PL_depth import perform_pldepth_experiment
from hyperparams import  lr_dict
import json
from pathlib import Path
import pickle
import optuna
from optuna.distributions import LogUniformDistribution, CategoricalDistribution
import numpy as np

N_samples = 0

def main():
    #trails = Trials()
    trials = pickle.load(open("/home/praneeth/projects/thesis/git/PLDepth/pldepth/hyperopt/trial_dump/hist", "rb"))
    best_fit = fmin(fn=perform_pldepth_experiment, space=lr_dict, trials=trials, algo=tpe.suggest, max_evals=5,
                    max_queue_len=1,
                    trials_save_file="/home/praneeth/projects/thesis/git/PLDepth/pldepth/hyperopt/trial_dump/hist")
    final_result = best_fit.copy()
    print(final_result)

    #print(trails.losses())
    # pickle.dump(trails, open("data_dump","wb"))
    #
    # with open("data.json", "w") as f:
    #     json.dump(trails, f)
    # data = pickle.load(open("data_dump"), "rb" )
    # append_non_empty_trails = []
    # for trail in trails:
    #     if trail['result']['status'] == 'ok':
    #         append_non_empty_trails.append(trail)
    #
    #
    #
    # search_space={
    #     'lr':LogUniformDistribution(np.exp(-11.5), np.exp(0))
    # }
    # helper_space = {
    #
    # }
    # study = optuna.create_study()
    # for doc in append_non_empty_trails:
    #     params = dict()
    #     if doc['result']['status'] == 'ok':
    #         params= {key:value[0] if type(search_space[key]) != CategoricalDistribution}
    #         loss = doc['result']['loss']
    #         trail_optuna = optuna.create_trial(params=params, distributions=new_distributions, values=loss)
    #         study.add_trial(trail)

    print(trails.trials[0]['result']['loss'])

if __name__ == '__main__':
    main()

from hyperopt import tpe, fmin, Trials
from pldepth.hyperopt.hyper_PL_depth import perform_pldepth_experiment
from pldepth.hyperopt.hyper_active_on_rnd_pretrain import  active_pldepth_experiment
from activ_hyperparams import  lr_dict
import json
from pathlib import Path
import pickle
import optuna
from optuna.distributions import LogUniformDistribution, CategoricalDistribution
import numpy as np
import time


N_samples = 0
save_dir = "/scratch/hpc-prf-deepmde/praneeth/output/trials/"
def main():
    trials = Trials()
    #trials = pickle.load(open("/home/praneeth/projects/thesis/git/PLDepth/pldepth/hyperopt/trial_dump/hist", "rb"))
    save_dir = "/scratch/hpc-prf-deepmde/praneeth/output/trials/tr1"
    timestr = time.strftime("%d%m%y-%H%M%S")
    print("start time: ",timestr)
    save_dir = "trial_dump/1"
    best_fit = fmin(fn=active_pldepth_experiment, space=lr_dict, trials=trials, algo=tpe.suggest, max_evals=25,
                    max_queue_len=1,
                    trials_save_file=save_dir)
    final_result = best_fit.copy()
    print("final result: ",final_result)


    # study = optuna.create_study()
    # for doc in append_non_empty_trails:
    #     params = dict()
    #     if doc['result']['status'] == 'ok':
    #         params= {key:value[0] if type(search_space[key]) != CategoricalDistribution}
    #         loss = doc['result']['loss']
    #         trail_optuna = optuna.create_trial(params=params, distributions=new_distributions, values=loss)
    #         study.add_trial(trail)

    #print(trails.trials[0]['result']['loss'])

if __name__ == '__main__':
    main()

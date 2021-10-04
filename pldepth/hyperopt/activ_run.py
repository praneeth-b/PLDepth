from hyperopt import tpe, fmin, Trials
from pldepth.hyperopt.hyper_PL_depth import perform_pldepth_experiment
from pldepth.hyperopt.hyper_active_on_rnd_pretrain import  active_pldepth_experiment
from hyperparams import  lr_dict, info_dict
import json
from pathlib import Path
import pickle
#import optuna
#from optuna.distributions import LogUniformDistribution, CategoricalDistribution
import numpy as np
import time


N_samples = 0
save_dir = "/scratch/hpc-prf-deepmde/praneeth/output/trials/active_trial"
def main():
    trials = Trials()
    #trials = pickle.load(open("/home/praneeth/projects/thesis/git/PLDepth/pldepth/hyperopt/trial_dump/hist", "rb"))
    save_dir = "/scratch/hpc-prf-deepmde/praneeth/output/trials/activ_on_info_"
    timestr = time.strftime("%d%m%y-%H%M%S")
    print("start time: ",timestr)
    save_dir = save_dir + timestr
    best_fit = fmin(fn=active_pldepth_experiment, space=info_dict, trials=trials, algo=tpe.suggest, max_evals=50,
                    max_queue_len=1,
                    trials_save_file=save_dir)
    final_result = best_fit.copy()
    print("final result: ",final_result)

   
if __name__ == '__main__':
    main()

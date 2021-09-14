from hyperopt import tpe, fmin, Trials
from pldepth.hyperopt.hyper_PL_depth import perform_pldepth_experiment
from pldepth.hyperopt.hyper_active_on_rnd_pretrain import  active_pldepth_experiment
from hyperparams import  info_dict
import json
from pathlib import Path
import pickle

import numpy as np
import time


N_samples = 0

def main():

    rerun= True
    if rerun:
        trials = pickle.load(open("/scratch/hpc-prf-deepmde/praneeth/output/trials/tr_rnd120921-025933", "rb"))

    else:
        trials = Trials()

    save_dir = "/scratch/hpc-prf-deepmde/praneeth/output/trials/tr_info"
    timestr = time.strftime("%d%m%y-%H%M%S")
    print("start time: ",timestr)
    save_dir = save_dir + timestr
    best_fit = fmin(fn=perform_pldepth_experiment, space=info_dict, trials=trials, algo=tpe.suggest, max_evals=100,
                    max_queue_len=1,
                    trials_save_file=save_dir)
    final_result = best_fit.copy()
    print("final result: ",final_result)




if __name__ == '__main__':
    main()

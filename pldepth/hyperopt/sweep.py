import wandb
import os
from pldepth.hyperopt.hyperparams import sweep_config_i, sweep_config_t, sweep_config_pr, activ_sweep, activ_sweep2
from pldepth.hyperopt.hyper_PL_depth import perform_pldepth_experiment
from pldepth.hyperopt.hyper_active_on_base import active_pldepth_experiment
from pldepth.hyperopt.hyper_base_PLD import perform_base_PLD
from pldepth.hyperopt.hyper_active_PLD import perform_active_PLD
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Add long and short argument
    parser.add_argument("--sampling_type", "-s", help="set sampling type")
    # Read arguments from the command line
    args = parser.parse_args()
    print("started..",args.sampling_type)

    typ = int(args.sampling_type)
    
    if typ==0:
        sweep_id = wandb.sweep(sweep_config_t, project="PLD-rnd-sweep")
        print(sweep_id)
        wandb.agent(sweep_id, perform_pldepth_experiment, count=40)
    

    elif typ == 1:
        sweep_id = wandb.sweep(sweep_config_i, project="PLD-sweep")
        print(sweep_id)
        wandb.agent(sweep_id, perform_pldepth_experiment, count=40)

    elif typ == 3:
        sweep_id = wandb.sweep(sweep_config_pr, project="PLD-rnd-sweep")
        print(sweep_id)
        wandb.agent(sweep_id, perform_pldepth_experiment, count=40)


    elif typ==2:
        print("Active learning sweep")
        sweep_id = wandb.sweep(activ_sweep2, project="Active_sweep")
        print("&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&",sweep_id)
        wandb.agent(sweep_id, perform_base_PLD, count=30)

    else:
        print("wrong selection")

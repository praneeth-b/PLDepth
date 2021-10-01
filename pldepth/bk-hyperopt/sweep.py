import wandb
import os
from pldepth.hyperopt.hyperparams import sweep_config, activ_sweep
from pldepth.hyperopt.hyper_PL_depth import perform_pldepth_experiment
from pldepth.hyperopt.hyper_active_on_base import active_pldepth_experiment
import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Add long and short argument
    parser.add_argument("--sampling_type", "-s", help="set sampling type")
    # Read arguments from the command line
    args = parser.parse_args()
    print(args.sampling_type)



    type = args.sampling_type
    if type == 1:
        sweep_id = wandb.sweep(sweep_config, project="PLDepth-sweep-test")
        print(sweep_id)
        wandb.agent(sweep_id, perform_pldepth_experiment, count=2)

    elif type==2:
        sweep_id = wandb.sweep(activ_sweep, project="Active_sweep")
        print("&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&",sweep_id)
        wandb.agent(sweep_id, active_pldepth_experiment, count=1)

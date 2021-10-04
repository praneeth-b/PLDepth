import wandb
import os
from pldepth.hyperopt.hyper_PL_depth import perform_pldepth_experiment
from pldepth.hyperopt.hyper_active_on_base import active_pldepth_experiment
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Add long and short argument
    parser.add_argument("--sampling_type", "-s", help="set sampling type")
    parser.add_argument("--sweepID", "-sid", help="sweep ID of sweep")
    parser.add_argument("--count", "-c", help="number of runs in sweep")
    # Read arguments from the command line
    args = parser.parse_args()
    # print(args.sampling_type)
    sweep_id = args.sweepID
    typ = int(args.sampling_type)
    print(type(sweep_id))

    if typ == 0:
        #sweep_id = wandb.sweep(sweep_config_t, project="PLD-rnd-sweep")
        print(sweep_id)
        wandb.agent(sweep_id, perform_pldepth_experiment, count=3)


    elif typ == 1:
        #sweep_id = wandb.sweep(sweep_config_i, project="PLD-sweep")
        print(sweep_id)
        wandb.agent(sweep_id, perform_pldepth_experiment, count=3)

    elif typ == 2:
        #sweep_id = wandb.sweep(activ_sweep, project="Active_sweep")
        print("&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&", sweep_id)
        wandb.agent(sweep_id, active_pldepth_experiment, project="Active_sweep", count=40)
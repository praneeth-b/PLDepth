import wandb
import os
from pldepth.hyperopt.act_base_sweep import act_on_base
from pldepth.hyperopt.rnd_base_sweep import rnd_on_base


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
        wandb.agent(sweep_id, perform_pldepth_experiment,project="PLD-Thresh-rnd-sweep", count=25)


    elif typ == 1:
        #sweep_id = wandb.sweep(sweep_config_i, project="PLD-sweep")
        print(sweep_id)
        wandb.agent(sweep_id, perform_pldepth_experiment,project="PLD-sweep", count=25)
    
    elif typ==3:
        wandb.agent(sweep_id, perform_pldepth_experiment, project="PLD-rnd-sweep", count=25)
        

    elif typ == 2:
        #sweep_id = wandb.sweep(activ_sweep, project="Active_sweep")
        print("&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&", sweep_id)
        wandb.agent(sweep_id, rnd_on_base, project="Active_sweep", count=25)

    else:
        print("wrong sampling type")

import wandb
import os
from hyperparams import sweep_config
from hyper_PL_depth import perform_pldepth_experiment

if __name__ == "__main__":
    sweep_id = wandb.sweep(sweep_config, project="PLDepth-sweep-test")
    print(sweep_id)
    wandb.agent(sweep_id, perform_pldepth_experiment, count=50)

import wandb
from wandb.keras import WandbCallback
import os
from pldepth.data.dao.hr_wsi import HRWSITFDataAccessObject
from pldepth.data.io_utils import get_dataset_type_by_name
from pldepth.data.providers.hourglass_provider import HourglassLargeScaleDataProvider
from pldepth.data.sampling import ThresholdedMaskedRandomSamplingStrategy, InformationScoreBasedSampling, \
    PurelyMaskedRandomSamplingStrategy
from pldepth.losses.losses_meta import DepthLossType
from pldepth.losses.nll_loss import HourglassNegativeLogLikelihood
from pldepth.models.PLDepthNet import get_pl_depth_net
import click
from tensorflow import keras
from tensorflow.python.keras.callbacks import TerminateOnNaN, LearningRateScheduler, CSVLogger
import mlflow
import time
import tensorflow as tf
from pldepth.util.env import init_env
from pldepth.models.models_meta import ModelParameters, get_model_type_by_name
from pldepth.util.training_utils import LearningRateScheduleProvider, SGDRScheduler, LearningRateLoggingCallback
from pldepth.util.tracking_utils import construct_model_checkpoint_callback, construct_tensorboard_callback
from pldepth.active_learning.metrics import calc_err

import numpy as np


def compute_chi_sq(a, rs):
    """
    a: the rpi X ranking_size X 2 array. train_ds[1]
    """
    c2 = 0
    expected_list = np.linspace(0.001, 0.999, rs + 1)[1:]
    for ar in a:
        l = ar[:, 1]
        c2 += (np.square(l - expected_list) / expected_list).sum()
    return c2 / a.shape[0]


@click.command()
@click.option('--model_name', default='ff_effnet', help='Backbone model',
              type=click.Choice(['ff_redweb', 'ff_effnet'], case_sensitive=False))
@click.option('--epochs', default=50)
@click.option('--batch_size', default=4)  # modified
@click.option('--seed', default=0)
@click.option('--ranking_size', default=3, help='Number of elements per training ranking')
@click.option('--rankings_per_image', default=100, help='Number of rankings per image for training')
@click.option('--initial_lr', default=0.01, type=click.FLOAT)
@click.option('--equality_threshold', default=0.03, type=click.FLOAT, help='Threshold which corresponds to the tau '
                                                                           'parameter as used in Section 3.5.')
@click.option('--model_checkpoints', default=False, help='Indicator whether the currently best performing model should'
                                                         ' be saved.', type=click.BOOL)
@click.option('--load_model_path', default='', help='Specify the path to a model in order to load it')
@click.option('--augmentation', default=True, type=click.BOOL)
@click.option('--warmup', default=0, type=click.INT)
@click.option('--sampling_type', default=1, type=click.INT)
@click.option('--lr_multi', default=0.25, type=click.FLOAT)
@click.option('--ds_size', default=None, type=click.INT)
def perform_pldepth_experiment(model_name, epochs, batch_size, seed, ranking_size, rankings_per_image, initial_lr,
                               equality_threshold, model_checkpoints, load_model_path, augmentation, warmup,
                               sampling_type, lr_multi, ds_size):
    config = init_env(experiment_name=str(sampling_type), autolog_freq=1, seed=seed)
    print("dataset size is: ", ds_size)
    run_name = "Pldepth-train"
    # print("#########********** main INITIALIZING WANDB ***********  ####################")
    # os.environ['WANDB_API_KEY'] = '58ae5a04488d3faafd7b3e0e0cc0e373226104c6'
    # os.environ['WANDB_DIR'] ='/scratch/hpc-prf-deepmde/praneeth/wandb-logs/'
    # os.environ["WANDB_NAME"] = run_name
    # os.environ["WANDB_CONSOLE"] = "off"
    # os.environ['WANDB_MODE'] = 'offline'

    print("##################### train started ###########################")
    timestr = time.strftime("%d%m%y-%H%M%S")
    # config = init_env(experiment_name=timestr+str(sampling_type), autolog_freq=1, seed=seed)
    print("the env var is: ", os.environ['WANDB_DIR'])
    # run = wandb.init(project="dummy", settings=wandb.Settings(_disable_stats=True),
    #                  # dir='/scratch/hpc-prf-deepmde/praneeth/wandb-logs',
    #                  config={'model_name': model_name,
    #                          'epochs': epochs,
    #                          'batch_size': batch_size,
    #                          'seed': seed,
    #                          'ranking_size': ranking_size,
    #                          'rankings_per_image': rankings_per_image,
    #                          'initial_lr': initial_lr,
    #                          'sampling_type': sampling_type,
    #                          'lr_multi': lr_multi,
    #                          'dataset_size': ds_size
    #                          })
    # w_config = wandb.config

    timestr = time.strftime("%d%m%y-%H%M%S")

    # Determine model, dataset and loss types
    model_type = get_model_type_by_name(model_name)
    dataset = "HR-WSI"
    dataset_type = get_dataset_type_by_name(dataset)
    loss_type = DepthLossType.NLL

    # Run meta information
    model_params = ModelParameters()
    model_params.set_parameter("model_type", model_type)
    model_params.set_parameter("dataset", dataset_type)
    model_params.set_parameter("epochs", epochs)
    model_params.set_parameter("ranking_size", ranking_size)
    model_params.set_parameter("rankings_per_image", rankings_per_image)
    model_params.set_parameter('val_rankings_per_img', rankings_per_image)
    model_params.set_parameter("batch_size", batch_size)
    model_params.set_parameter("seed", seed)
    model_params.set_parameter('equality_threshold', equality_threshold)
    model_params.set_parameter('loss_type', loss_type)
    model_params.set_parameter('augmentation', augmentation)
    model_params.set_parameter('warmup', warmup)

    if sampling_type == 0:
        print("$$$$$$$$$  thresh $$$$$$$$$$$")
        sampling_strategy = ThresholdedMaskedRandomSamplingStrategy(
            model_params)  # InformationScoreBasedSampling(model_params)

    elif sampling_type == 1:
        print("$$$$$$$$$  info $$$$$$$$$$$")
        sampling_strategy = InformationScoreBasedSampling(model_params)

    elif sampling_type == 3:
        print("$$$$$$$$$  purely random $$$$$$$$$$$")
        sampling_strategy = PurelyMaskedRandomSamplingStrategy(model_params)

    else:
        print("wrong selection of sampling type")
        return 13

    model_params.set_parameter('sampling_strategy', sampling_strategy)
    model_input_shape = [448, 448, 3]

    dao = HRWSITFDataAccessObject(config["DATA"]["HR_WSI_ROOT_PATH"], model_input_shape, seed)
    ds_size = 1000  # todo
    train_imgs_ds, train_gts_ds, train_cons_masks = dao.get_training_dataset(size=ds_size)
    val_imgs_ds, val_gts_ds, val_cons_masks, = dao.get_validation_dataset()

    score_arr = []
    for i in range(5):
        print("trial :", i)

        data_provider = HourglassLargeScaleDataProvider(model_params, train_cons_masks, val_cons_masks,
                                                        augmentation=model_params.get_parameter("augmentation"),
                                                        loss_type=loss_type)

        train_ds = data_provider.provide_train_dataset(train_imgs_ds, train_gts_ds)

        calc_ds = train_ds.take(25)  # batches
        score = []
        for ar in calc_ds.as_numpy_iterator():
            # print(a[0].shape, a[1].shape)
            a = ar[1]
            a = a.reshape(-1, *a.shape[-2:])

            score.append(compute_chi_sq(a, ranking_size))

        avg_score = sum(score) / len(score)
        print('chi2_score', avg_score)
        score_arr.append(avg_score)

    print("mean= ",np.mean(score_arr), "variance = ", np.var(score_arr))


if __name__ == "__main__":
    perform_pldepth_experiment()

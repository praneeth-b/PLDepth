from pldepth.data.dao.hr_wsi import HRWSITFDataAccessObject
from pldepth.data.io_utils import get_dataset_type_by_name
from pldepth.data.providers.hourglass_provider import HourglassLargeScaleDataProvider
from pldepth.data.sampling import ThresholdedMaskedRandomSamplingStrategy, InformationScoreBasedSampling
from pldepth.losses.losses_meta import DepthLossType
from pldepth.losses.nll_loss import HourglassNegativeLogLikelihood
from pldepth.models.PLDepthNet import get_pl_depth_net
import click
from tensorflow import keras
from tensorflow.python.keras.callbacks import TerminateOnNaN, LearningRateScheduler
import mlflow
import time
import tensorflow as tf

from pldepth.util.env import init_env
from pldepth.models.models_meta import ModelParameters, get_model_type_by_name
from pldepth.util.training_utils import LearningRateScheduleProvider, SGDRScheduler, LearningRateLoggingCallback
from pldepth.util.tracking_utils import construct_model_checkpoint_callback, construct_tensorboard_callback
from pldepth.active_learning.active_learning_method import active_learning_data_provider
from pldepth.models.pl_hourglass import EffNetFullyFledged
import wandb
from wandb.keras import WandbCallback
from keras import backend as K

import os

from pldepth.active_learning.metrics import calc_err, dcg_metric, calc_depth_metrics


def eval_model():
    config = init_env(experiment_name='ev-test', autolog_freq=1, seed=0)
    run = wandb.init(project="eval-train")

    model_name = 'ff_effnet'
    epochs = 10
    batch_size = 6
    seed = 0
    ranking_size = 5
    rankings_per_image = 100
    initial_lr = 0.1
    lr_multi = 0.1
    equality_threshold = 0.03
    augmentation = False
    warmup = 0

    model_name = 'ff_effnet'
    # Determine model, dataset and loss types
    model_type = get_model_type_by_name(model_name)
    dataset = "HR-WSI"
    dataset_type = get_dataset_type_by_name(dataset)
    loss_type = DepthLossType.NLL
    #load_path = '/upb/departments/pc2/groups/hpc-prf-deepmde/praneeth/PLDepth/pldepth/weights/base/pr-base.h5'
    load_path = '/home/praneeth/projects/thesis/git/PLDepth/pldepth/weights/100921-092654base_10rpi_1k_30ep_6r_model_rnd_sampling.h5'
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

    model_input_shape = [224, 224, 3]
    model, preprocess_fn = get_pl_depth_net(model_params, model_input_shape)

    # Compile model
    lr_sched_prov = LearningRateScheduleProvider(init_lr=initial_lr, steps=[5, 10, 15, 20], warmup=warmup,
                                                 multiplier=lr_multi)
    loss_fn = HourglassNegativeLogLikelihood(ranking_size=ranking_size,
                                             batch_size=batch_size,
                                             debug=False)


    optimizer = keras.optimizers.Adam(learning_rate=initial_lr, amsgrad=True)
    # load the model here
    # model = tf.keras.models.load_model(load_path,
    #                                  custom_objects={'EffNetFullyFledged': EffNetFullyFledged}, compile=False)
    model.load_weights(load_path)
    model.compile(loss=loss_fn, optimizer=optimizer)

    data_path = config["DATA"]["HR_WSI_TEST_PATH"]
    dao_a = HRWSITFDataAccessObject(data_path, model_input_shape, seed=78)
    test_img, test_gt, test_cons_masks = dao_a.get_training_dataset()
    test_img = list(test_img.as_numpy_iterator())
    test_gt = list(test_gt.as_numpy_iterator())
    print(len(test_img), len(test_gt))

    err = calc_err(model, test_img, test_gt, img_size=tuple(model_input_shape[:2]))
    wandb.run.summary["test_error"] = err

    dcg_val = dcg_metric(model, test_img, test_gt, list_size=200)
    wandb.run.summary["ndcg_200"] = dcg_val

    d_bounary, d_comp = calc_depth_metrics(model, test_img, test_gt)
    wandb.run.summary["depth_boundary_metric"] = d_bounary
    wandb.run.summary["depth_completeness"] = d_comp

    run.finish()


if __name__ == "__main__":
    eval_model()
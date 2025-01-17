from pldepth.data.dao.hr_wsi import HRWSITFDataAccessObject
from pldepth.data.io_utils import get_dataset_type_by_name
from pldepth.data.providers.hourglass_provider import HourglassLargeScaleDataProvider
from pldepth.data.sampling import ThresholdedMaskedRandomSamplingStrategy, InformationScoreBasedSampling, PurelyMaskedRandomSamplingStrategy
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

from pldepth.active_learning.metrics import calc_err





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
    config = init_env(experiment_name='run1', autolog_freq=1, seed=seed)
    timestr = time.strftime("%d%m%y-%H%M%S")
    run = wandb.init(project="active-learning",
                     config={'model_name': model_name,
                             'epochs': epochs,
                             'batch_size': batch_size,
                             'seed': seed,
                             'ranking_size': ranking_size,
                             'rankings_per_image': rankings_per_image,
                             'initial_lr': initial_lr,
                             'sampling_type': sampling_type,
                             'lr_multi': lr_multi,
                             'ds_size': ds_size
                             })
    w_config = wandb.config

    # Determine model, dataset and loss types
    model_type = get_model_type_by_name(model_name)
    dataset = "HR-WSI"
    dataset_type = get_dataset_type_by_name(dataset)
    loss_type = DepthLossType.NLL
    # load_path = '/upb/departments/pc2/groups/hpc-prf-deepmde/praneeth/PLDepth/pldepth/weights/100921-092654base_10rpi_1k_30ep_6r_model_rnd_sampling.h5'

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
        sampling_strategy = ThresholdedMaskedRandomSamplingStrategy(
            model_params)  # InformationScoreBasedSampling(model_params)

    elif sampling_type == 1:
        sampling_strategy = InformationScoreBasedSampling(model_params)

    elif sampling_type == 3:
        sampling_strategy = PurelyMaskedRandomSamplingStrategy(model_params)

    else:
        #sampling_strategy = InformationScoreBasedSampling(model_params)
        print("wrong sampling type")
        return 13

    model_params.set_parameter('sampling_strategy', sampling_strategy)
    model_input_shape = [224, 224, 3]

    # Get model
    model, preprocess_fn = get_pl_depth_net(model_params, model_input_shape)
    # model.summary()

    # Compile model
    #lr_sched_prov = LearningRateScheduleProvider(init_lr=initial_lr, steps=[5, 8,11,12,13,14,15,16,17 ], warmup=warmup, multiplier=lr_multi)
    loss_fn = HourglassNegativeLogLikelihood(ranking_size=model_params.get_parameter("ranking_size"),
                                             batch_size=model_params.get_parameter("batch_size"),
                                             debug=False)

    steps_per_epoch = int((ds_size * 14 / 15) / batch_size)
    schedule = SGDRScheduler(min_lr=initial_lr * (1 / lr_multi),
                             max_lr=initial_lr,
                             steps_per_epoch=steps_per_epoch,
                             lr_decay=0.5,
                             cycle_length=epochs+6,  # decays over entire training- non cyclic
                             mult_factor=1)



    optimizer = keras.optimizers.Adam(learning_rate=initial_lr, amsgrad=True)

    model.compile(loss=loss_fn, optimizer=optimizer)
    #model.summary()

    dao = HRWSITFDataAccessObject(config["DATA"]["HR_WSI_3K_PATH"], model_input_shape, seed)

    train_imgs_ds, train_gts_ds, train_cons_masks, = dao.get_training_dataset(size=ds_size)
    val_imgs_ds, val_gts_ds, val_cons_masks, = dao.get_validation_dataset()

    data_provider = HourglassLargeScaleDataProvider(model_params, train_cons_masks, val_cons_masks,
                                                    augmentation=model_params.get_parameter("augmentation"),
                                                    loss_type=loss_type)

    train_ds = data_provider.provide_train_dataset(train_imgs_ds, train_gts_ds)
    val_ds = data_provider.provide_val_dataset(val_imgs_ds, val_gts_ds)
    callbacks = [ TerminateOnNaN(),schedule, LearningRateLoggingCallback(), WandbCallback() ]  # LearningRateScheduler(lr_sched_prov.get_lr_schedule)
    verbosity = 1

    # Apply preprocessing
    def preprocess_ds(loc_x, loc_y):
        return preprocess_fn(loc_x), loc_y

    train_ds = train_ds.map(preprocess_ds, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    val_ds = val_ds.map(preprocess_ds, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    steps_per_epoch = int(ds_size/ batch_size)
    model.fit(x=train_ds, epochs=model_params.get_parameter("epochs"), steps_per_epoch=steps_per_epoch,
              callbacks=callbacks, validation_data=val_ds, verbose=verbosity)

    print("Start active sampling")


    data_path = config["DATA"]["HR_WSI_POOL_PATH"]
    dao_a = HRWSITFDataAccessObject(data_path, model_input_shape, seed=seed)
    test_imgs_ds, test_gts_ds, test_cons_masks = dao_a.get_training_dataset(5000)

    img_ds_list = list(test_imgs_ds.as_numpy_iterator())
    img_gt_list = list(test_gts_ds.as_numpy_iterator())

    schedule.steps_per_epoch = 5000/batch_size
    for i in range(6):
        
        wandb.log({"active_count": i})

        active_train_ds = active_learning_data_provider(img_ds_list, img_gt_list, model, batch_size=batch_size,
                                                       ranking_size=ranking_size, split_num=16)
        active_train_ds.map(preprocess_ds, num_parallel_calls=tf.data.experimental.AUTOTUNE)

        steps_per_epoch=int(5000/batch_size)
        print("fit active sampled data")
        #n_epochs = epochs+10
        model.fit(x=active_train_ds,initial_epoch=epochs+i, epochs=epochs+i+1, steps_per_epoch=steps_per_epoch,
                  validation_data=val_ds, verbose=1, callbacks=callbacks)

        break

    # Save the weights
    timestr = time.strftime("%d%m%y-%H%M%S")
    #model.save('/scratch/hpc-prf-deepmde/praneeth/output/' + timestr + 'active_model_info_sampling.h5')

    # evaluate on test data:
    vds = list(val_imgs_ds.as_numpy_iterator())
    vgt = list(val_gts_ds.as_numpy_iterator())
    test_img = vds[:300]
    test_gt = vgt[:300]

    err = calc_err(model, test_img, test_gt, img_size=tuple(model_input_shape[:2]))
    wandb.run.summary["test_error"] = err



if __name__ == "__main__":
    perform_pldepth_experiment()

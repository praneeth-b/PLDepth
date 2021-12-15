from pldepth.data.seg_data.dao.hr_wsi import HRWSITFDataAccessObject
from pldepth.data.io_utils import get_dataset_type_by_name
from pldepth.data.seg_data.providers.hourglass_provider import HourglassLargeScaleDataProvider
from pldepth.data.sampling import ThresholdedMaskedRandomSamplingStrategy , InformationScoreBasedSampling,\
    MaskedRandomSamplingStrategy
from pldepth.data.seg_data.sampling import HeterogenousSegmentBasedSampling
from pldepth.losses.losses_meta import DepthLossType
from pldepth.losses.nll_loss import HourglassNegativeLogLikelihood
from pldepth.models.PLDepthNet import get_pl_depth_net
import click
from tensorflow import keras
from tensorflow.python.keras.callbacks import TerminateOnNaN, LearningRateScheduler
import time
import tensorflow as tf
from pldepth.util.env import init_env
from pldepth.models.models_meta import ModelParameters, get_model_type_by_name
from pldepth.util.training_utils import LearningRateScheduleProvider
from pldepth.util.tracking_utils import construct_model_checkpoint_callback
import wandb
from wandb.keras import WandbCallback
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
@click.option('--ds_size', default=5000, type=click.INT)
def perform_pldepth_experiment(model_name, epochs, batch_size, seed, ranking_size, rankings_per_image, initial_lr,
                               equality_threshold, model_checkpoints, load_model_path, augmentation, warmup,
                               sampling_type, lr_multi, ds_size):
    config = init_env(experiment_name=str(sampling_type), autolog_freq=1, seed=seed)

    run = wandb.init(project="dummy", settings=wandb.Settings(_disable_stats=True),
                     # dir='/scratch/hpc-prf-deepmde/praneeth/wandb-logs',
                     config={'model_name': model_name,
                             'epochs': epochs,
                             'batch_size': batch_size,
                             'seed': seed,
                             'ranking_size': ranking_size,
                             'rankings_per_image': rankings_per_image,
                             'initial_lr': initial_lr,
                             'sampling_type': sampling_type,
                             'lr_multi': lr_multi,
                             'dataset_size': ds_size
                             })
    w_config = wandb.config

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

    sampling_strategy = HeterogenousSegmentBasedSampling(model_params) #ThresholdedMaskedRandomSamplingStrategy(model_params)  #HeterogenousSegmentBasedSampling(model_params)  #
    model_params.set_parameter('sampling_strategy', sampling_strategy)

    model_input_shape = [448, 448, 3]

    # Get model
    model, preprocess_fn = get_pl_depth_net(model_params, model_input_shape)
    model.summary()

    # Compile model
    lr_sched_prov = LearningRateScheduleProvider(init_lr=initial_lr, steps=[4,8], warmup=warmup, multiplier=lr_multi)
    loss_fn = HourglassNegativeLogLikelihood(ranking_size=model_params.get_parameter("ranking_size"),
                                             batch_size=model_params.get_parameter("batch_size"),
                                             debug=False)

    optimizer = keras.optimizers.Adam(learning_rate=lr_sched_prov.get_lr_schedule(0), amsgrad=True)
    model.compile(loss=loss_fn, optimizer=optimizer)

    if load_model_path != "":
        model.load_weights(load_model_path)

    dao = HRWSITFDataAccessObject(config["DATA"]["HR_WSI_3K_PATH"], model_input_shape, seed)

    train_imgs_ds, train_gts_ds, train_cons_masks, train_instances_mask = dao.get_training_dataset()
    val_imgs_ds, val_gts_ds, val_cons_masks, val_instance_mask = dao.get_validation_dataset()

    data_provider = HourglassLargeScaleDataProvider(model_params, train_cons_masks, val_cons_masks,
                                                    train_instances_mask,val_instance_mask,
                                                    augmentation=model_params.get_parameter("augmentation"),
                                                    loss_type=loss_type)

    train_ds = data_provider.provide_train_dataset(train_imgs_ds, train_gts_ds)
    val_ds = data_provider.provide_val_dataset(val_imgs_ds, val_gts_ds)

    callbacks = [TerminateOnNaN(), LearningRateScheduler(lr_sched_prov.get_lr_schedule),
                WandbCallback(save_model=False, log_batch_frequency=None)]
    verbosity = 1

    # Apply preprocessing
    def preprocess_ds(loc_x, loc_y):
        return preprocess_fn(loc_x), loc_y
    train_ds = train_ds.map(preprocess_ds, num_parallel_calls=tf.data.experimental.AUTOTUNE)

    val_ds = val_ds.map(preprocess_ds, num_parallel_calls=tf.data.experimental.AUTOTUNE)

    steps_per_epoch = int(3000 / batch_size)
    model.fit(x=train_ds, epochs=model_params.get_parameter("epochs"), steps_per_epoch=steps_per_epoch,
               callbacks=callbacks, validation_data=val_ds, verbose=verbosity)

    # evaluate on test data:
    vds = list(val_imgs_ds.as_numpy_iterator())
    vgt = list(val_gts_ds.as_numpy_iterator())
    test_img = vds[:150]
    test_gt = vgt[:150]

    err = calc_err(model, test_img, test_gt)
    wandb.run.summary["test_error"] = err
    

    run.finish()



if __name__ == "__main__":
    perform_pldepth_experiment()
    print("*******&&&&&&&&&&&&&&&&&done!!")

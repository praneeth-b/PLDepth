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
from pldepth.util.training_utils import LearningRateScheduleProvider
from pldepth.util.tracking_utils import construct_model_checkpoint_callback, construct_tensorboard_callback
from pldepth.active_learning.active_learning_method import active_learning_data_provider
from pldepth.models.pl_hourglass import EffNetFullyFledged
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
@click.option('--lr_multi', default=0.1, type=click.FLOAT)
@click.option('--ds_size', default=None, type=click.INT)
def perform_pldepth_experiment(model_name, epochs, batch_size, seed, ranking_size, rankings_per_image, initial_lr,
                               equality_threshold, model_checkpoints, load_model_path, augmentation, warmup, lr_multi,
                               ds_size):
    config = init_env(experiment_name='run1', autolog_freq=1, seed=seed)
    timestr = time.strftime("%d%m%y-%H%M%S")
    run = wandb.init(project="active-learning", settings=wandb.Settings(_disable_stats=True),
                     # dir='/scratch/hpc-prf-deepmde/praneeth/wandb-logs/',
                     config={'model_name': model_name,
                             'epochs': epochs,
                             'batch_size': batch_size,
                             'seed': seed,
                             'ranking_size': ranking_size,
                             'rankings_per_image': rankings_per_image,
                             'initial_lr': initial_lr,
                             'lr_multi': lr_multi,
                             'dataset_size': ds_size,
			     'split_num': 16
                             })
    w_config = wandb.config
    # Determine model, dataset and loss types
    model_type = get_model_type_by_name(model_name)
    dataset = "HR-WSI"
    dataset_type = get_dataset_type_by_name(dataset)
    loss_type = DepthLossType.NLL
    load_path = '/upb/departments/pc2/groups/hpc-prf-deepmde/praneeth/PLDepth/pldepth/weights/base/base-inf-rs400.h5'

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
    lr_sched_prov = LearningRateScheduleProvider(init_lr=initial_lr, steps=[1,2,3,4,5,6,7,8], warmup=warmup, multiplier=lr_multi)
    loss_fn = HourglassNegativeLogLikelihood(ranking_size=ranking_size,
                                             batch_size=batch_size,
                                             debug=False)

    optimizer = keras.optimizers.Adam(learning_rate=lr_sched_prov.get_lr_schedule(0), amsgrad=True)

    model.load_weights(load_path)
    model.compile(loss=loss_fn, optimizer=optimizer)

    callbacks = [TerminateOnNaN(), LearningRateScheduler(lr_sched_prov.get_lr_schedule), WandbCallback(save_model=False)]

    # Apply preprocessing
    def preprocess_ds(loc_x, loc_y):
        return preprocess_fn(loc_x), loc_y

    data_path = config["DATA"]["HR_WSI_3K_PATH"]
    dao_a = HRWSITFDataAccessObject(data_path, model_input_shape, seed)
    test_imgs_ds, test_gts_ds, test_cons_masks = dao_a.get_training_dataset(size=ds_size)

    val_imgs_ds, val_gts_ds, val_cons_masks = dao_a.get_validation_dataset(size=200)

    data_provider = HourglassLargeScaleDataProvider(model_params, test_cons_masks, val_cons_masks,
                                                    augmentation=False,
                                                    loss_type=loss_type)
    val_ds = data_provider.provide_val_dataset(val_imgs_ds, val_gts_ds)

    vds = list(val_imgs_ds.as_numpy_iterator())
    vgt = list(val_gts_ds.as_numpy_iterator())
    test_img = vds[:150]
    test_gt = vgt[:150]

    init_err = calc_err(model, test_img, test_gt, img_size=tuple(model_input_shape[:2]))
    print("$$$$ init error $$$: ", init_err)
    wandb.log({"init_error": init_err})

    print("Start active sampling")
    img_ds_list = list(test_imgs_ds.as_numpy_iterator())
    img_gt_list = list(test_gts_ds.as_numpy_iterator())

    for i in range(5):
        wandb.log({"active_count": i})

        active_train_ds = active_learning_data_provider(img_ds_list, img_gt_list, model, batch_size=batch_size,
                                                        ranking_size=ranking_size, split_num=16,
                                                        img_size=model_input_shape)
        active_train_ds.map(preprocess_ds, num_parallel_calls=tf.data.experimental.AUTOTUNE)

        steps_per_epoch = int(ds_size / batch_size)
        print("fit active sampled data")

        model.fit(x=active_train_ds, initial_epoch=i, epochs=i + 1, steps_per_epoch=steps_per_epoch,
                  validation_data=val_ds, verbose=1, callbacks=callbacks)

        #lr_sched_prov.init_lr = initial_lr * lr_multi
        ep_error = calc_err(model, test_img, test_gt, img_size=tuple(model_input_shape[:2]))
        wandb.log({'epoch_err': ep_error})

        del active_train_ds

    # active_train_ds = active_learning_data_provider(img_ds_list, img_gt_list, model, batch_size=batch_size,
    #                                                 ranking_size=ranking_size, split_num=28, img_size=model_input_shape)
    # active_train_ds.map(preprocess_ds, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    # model.compile(loss=loss_fn, optimizer=optimizer)
    # #fit active samples over the prev trainedd model.ls
    # steps_per_epoch = int(ds_size / batch_size)
    # print("fit active sampled data")
    # model.fit(x=active_train_ds, epochs=epochs, steps_per_epoch=steps_per_epoch, validation_data=val_ds,
    #           callbacks=callbacks,verbose=1)
    # # Save the weights
    # timestr = time.strftime("%d%m%y-%H%M%S")
    # # model.save_weights('/scratch/hpc-prf-deepmde/praneeth/output/' + timestr + 'active_weight_rnd_sampling')
    # #model.save('/scratch/hpc-prf-deepmde/praneeth/output/' + timestr + 'active_model_inf_sampling.h5')
    # #evaluate on test data:

    err = calc_err(model, test_img, test_gt, img_size=tuple(model_input_shape[:2]))
    wandb.run.summary["test_error"] = err


if __name__ == "__main__":
    perform_pldepth_experiment()

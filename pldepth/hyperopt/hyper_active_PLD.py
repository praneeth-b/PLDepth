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

from pldepth.active_learning.metrics import calc_err

def perform_active_PLD(pars=None):
    with wandb.init(config=pars, settings=wandb.Settings(_disable_stats=True)):
        w_config = wandb.config

        model_name = 'ff_effnet'
        epochs = w_config['epochs']
        batch_size = w_config['batch_size']
        seed = 0
        ranking_size = w_config['ranking_size']
        rankings_per_image = 50 #w_config['rpi']
        initial_lr = w_config['lr']
        lr_multi = w_config['lr_multi']
        equality_threshold = 0.03
        model_checkpoints = False
        load_model_path = ""
        augmentation = True
        warmup = 0
        split_num = w_config['num_split']
        sampling_type = w_config['sampling_type']
        ds_size = w_config['dataset_size']

        config = init_env(autolog_freq=1, seed=seed)


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
            sampling_strategy = ThresholdedMaskedRandomSamplingStrategy(
                model_params)  # InformationScoreBasedSampling(model_params)

        elif sampling_type == 1:
            sampling_strategy = InformationScoreBasedSampling(model_params)

        else:
            # sampling_strategy = InformationScoreBasedSampling(model_params)
            print("wrong sampling type")
            return 13

        model_params.set_parameter('sampling_strategy', sampling_strategy)
        model_input_shape = [448, 448, 3]

        # Get model
        model, preprocess_fn = get_pl_depth_net(model_params, model_input_shape)
        # model.summary()

        # Compile model
        lr_sched_prov = LearningRateScheduleProvider(init_lr=initial_lr, steps=[5, 8, 15, 20, 25], warmup=warmup,
                                                     multiplier=lr_multi)
        loss_fn = HourglassNegativeLogLikelihood(ranking_size=model_params.get_parameter("ranking_size"),
                                                 batch_size=model_params.get_parameter("batch_size"),
                                                 debug=False)


        # schedule = SGDRScheduler(min_lr=initial_lr * 0.01,
        #                          max_lr=initial_lr,
        #                          steps_per_epoch=steps_per_epoch,
        #                          lr_decay=lr_multi,
        #                          cycle_length=1,
        #                          mult_factor=1)

        optimizer = keras.optimizers.Adam(learning_rate=initial_lr, amsgrad=True)

        model.compile(loss=loss_fn, optimizer=optimizer)
        # model.summary()

        dao = HRWSITFDataAccessObject(config["DATA"]["HR_WSI_1K_PATH"], model_input_shape, seed)

        train_imgs_ds, train_gts_ds, train_cons_masks, = dao.get_training_dataset()
        val_imgs_ds, val_gts_ds, val_cons_masks, = dao.get_validation_dataset()

        data_provider = HourglassLargeScaleDataProvider(model_params, train_cons_masks, val_cons_masks,
                                                        augmentation=model_params.get_parameter("augmentation"),
                                                        loss_type=loss_type)

        train_ds = data_provider.provide_train_dataset(train_imgs_ds, train_gts_ds)
        val_ds = data_provider.provide_val_dataset(val_imgs_ds, val_gts_ds)
        callbacks = [TerminateOnNaN(), LearningRateScheduler(lr_sched_prov.get_lr_schedule),
                     WandbCallback(log_batch_frequency=None)]
        verbosity = 1
        if model_checkpoints:
            callbacks.append(construct_model_checkpoint_callback(config, model_type, verbosity))

        # Apply preprocessing
        def preprocess_ds(loc_x, loc_y):
            return preprocess_fn(loc_x), loc_y

        train_ds = train_ds.map(preprocess_ds, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        val_ds = val_ds.map(preprocess_ds, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        steps_per_epoch = int(1000 / batch_size)
        model.fit(x=train_ds, epochs=model_params.get_parameter("epochs"), steps_per_epoch=steps_per_epoch,
                  callbacks=callbacks, validation_data=val_ds, verbose=verbosity)

        print("Start active sampling")
        data_path = config["DATA"]["HR_WSI_10K_PATH"]
        dao_a = HRWSITFDataAccessObject(data_path, model_input_shape, seed=seed)
        test_imgs_ds, test_gts_ds, test_cons_masks = dao_a.get_training_dataset(size=ds_size)

        active_train_ds = active_learning_data_provider(test_imgs_ds, test_gts_ds, model, batch_size=batch_size,
                                                        ranking_size=ranking_size, split_num=split_num)
        active_train_ds.map(preprocess_ds, num_parallel_calls=tf.data.experimental.AUTOTUNE)

        lr_sched_prov.init_lr = initial_lr * 3

        steps_per_epoch = int(ds_size / batch_size)
        print("fit active sampled data")
        n_epochs = epochs + 4
        model.fit(x=active_train_ds, initial_epoch=epochs, epochs=n_epochs, steps_per_epoch=steps_per_epoch,
                  validation_data=val_ds, verbose=1, callbacks=callbacks)

        # Save the weights
        timestr = time.strftime("%d%m%y-%H%M%S")
        #model.save('/scratch/hpc-prf-deepmde/praneeth/output/' + timestr + 'active_model_info_sampling.h5')

        # evaluate on test data:
        vds = list(val_imgs_ds.as_numpy_iterator())
        vgt = list(val_gts_ds.as_numpy_iterator())
        test_img = vds[:150]
        test_gt = vgt[:150]

        err = calc_err(model, test_img, test_gt)
        wandb.run.summary["test_error"] = err
        wandb.log({'val_loss': err})

if __name__ == "__main__":
    perform_active_PLD()

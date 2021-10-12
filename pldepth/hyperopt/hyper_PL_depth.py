from pldepth.data.dao.hr_wsi import HRWSITFDataAccessObject
from pldepth.data.io_utils import get_dataset_type_by_name
from pldepth.data.providers.hourglass_provider import HourglassLargeScaleDataProvider
from pldepth.data.sampling import ThresholdedMaskedRandomSamplingStrategy, InformationScoreBasedSampling, \
    MaskedRandomSamplingStrategy
from pldepth.losses.losses_meta import DepthLossType
from pldepth.losses.nll_loss import HourglassNegativeLogLikelihood
from pldepth.models.PLDepthNet import get_pl_depth_net
import click
from tensorflow import keras
from tensorflow.python.keras.callbacks import TerminateOnNaN, LearningRateScheduler
import mlflow
import time
import tensorflow as tf
from pldepth.active_learning.metrics import ordinal_error
from pldepth.util.env import init_env
from pldepth.models.models_meta import ModelParameters, get_model_type_by_name
from pldepth.util.training_utils import LearningRateScheduleProvider
from pldepth.util.tracking_utils import construct_model_checkpoint_callback, construct_tensorboard_callback

from hyperopt import STATUS_OK
import numpy as np
import wandb
from pldepth.active_learning.metrics import calc_err


def perform_pldepth_experiment(pars=None):
    with wandb.init(config=pars, settings=wandb.Settings(_disable_stats=True)):
        w_config = wandb.config

        model_name = 'ff_effnet'
        epochs = w_config['epochs']
        batch_size = w_config['batch_size']
        seed = 0
        ranking_size = w_config['ranking_size']
        rankings_per_image = w_config['rpi']
        initial_lr = w_config['lr']
        lr_multi = w_config['lr_multi']
        equality_threshold = 0.03
        model_checkpoints = False
        load_model_path = ""
        augmentation = True
        warmup = 0
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
            sampling_strategy = ThresholdedMaskedRandomSamplingStrategy(model_params)  # InformationScoreBasedSampling(model_params)

        elif sampling_type == 1:
            sampling_strategy = InformationScoreBasedSampling(model_params)

        else :
            print("wrong sampling type")
            return 13
            sampling_strategy = InformationScoreBasedSampling(model_params)

        model_params.set_parameter('sampling_strategy', sampling_strategy)

        model_input_shape = [448, 448, 3]

        # Get model
        model, preprocess_fn = get_pl_depth_net(model_params, model_input_shape)
        # model.summary()

        # Compile model
        lr_sched_prov = LearningRateScheduleProvider(init_lr=initial_lr, steps=[6, 10], warmup=warmup,
                                                     multiplier=lr_multi)
        loss_fn = HourglassNegativeLogLikelihood(ranking_size=model_params.get_parameter("ranking_size"),
                                                 batch_size=model_params.get_parameter("batch_size"),
                                                 debug=False)

        optimizer = keras.optimizers.Adam(learning_rate=lr_sched_prov.get_lr_schedule(0), amsgrad=True)
        model.compile(loss=loss_fn, optimizer=optimizer, metrics=loss_fn)

        if load_model_path != "":
            model.load_weights(load_model_path)

        dao = HRWSITFDataAccessObject(config["DATA"]["HR_WSI_ROOT_PATH"], model_input_shape, seed)

        all_imgs_ds, all_gts_ds, all_cons_masks = dao.get_training_dataset(size=ds_size)
        val_imgs_ds = all_imgs_ds.take(ds_size // 15)
        val_gts_ds = all_gts_ds.take(ds_size // 15)
        val_cons_masks = all_cons_masks.take(ds_size // 15)
        train_imgs_ds = all_imgs_ds.skip(ds_size // 15)
        train_gts_ds = all_gts_ds.skip(ds_size // 15)
        train_cons_masks = all_cons_masks.skip(ds_size // 15)

        #eval_imgs_ds, eval_gts_ds, eval_cons_masks, = dao.get_validation_dataset()

        data_provider = HourglassLargeScaleDataProvider(model_params, train_cons_masks, val_cons_masks,
                                                        augmentation=model_params.get_parameter("augmentation"),
                                                        loss_type=loss_type)

        train_ds = data_provider.provide_train_dataset(train_imgs_ds, train_gts_ds)
        #val_ds = data_provider.provide_val_dataset(val_imgs_ds, val_gts_ds)

        callbacks = [TerminateOnNaN(), LearningRateScheduler(lr_sched_prov.get_lr_schedule)]
        verbosity = 1
        if model_checkpoints:
            callbacks.append(construct_model_checkpoint_callback(config, model_type, verbosity))

        # model_params.log_parameters()

        # Apply preprocessing
        def preprocess_ds(loc_x, loc_y):
            return preprocess_fn(loc_x), loc_y

        train_ds = train_ds.map(preprocess_ds, num_parallel_calls=tf.data.experimental.AUTOTUNE)

        #val_ds = val_ds.map(preprocess_ds, num_parallel_calls=tf.data.experimental.AUTOTUNE)

        steps_per_epoch = int(5000 / batch_size)
        model.fit(x=train_ds, epochs=epochs, steps_per_epoch=steps_per_epoch,
                 callbacks=callbacks, verbose=verbosity)
        
            #evaluate on test data:
        vds = list(val_imgs_ds.as_numpy_iterator())
        vgt = list(val_gts_ds.as_numpy_iterator())
        test_img = vds[:150]
        test_gt = vgt[:150]

        loss = calc_err(model, test_img, test_gt)

        wandb.log({'val_loss': loss})


if __name__ == "__main__":
    perform_pldepth_experiment()
    print("done!!")

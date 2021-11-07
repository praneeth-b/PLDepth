from pldepth.data.dao.hr_wsi import HRWSITFDataAccessObject
from pldepth.data.io_utils import get_dataset_type_by_name
from pldepth.data.providers.hourglass_provider import HourglassLargeScaleDataProvider
from pldepth.data.sampling import ThresholdedMaskedRandomSamplingStrategy, InformationScoreBasedSampling,PurelyMaskedRandomSamplingStrategy
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
from hyperopt import STATUS_OK
import numpy as np
from pldepth.active_learning.metrics import ordinal_error
import wandb
from pldepth.active_learning.metrics import calc_err
from wandb.keras import WandbCallback


def rnd_on_base(pars=None):
    with wandb.init(settings=wandb.Settings(_disable_stats=True), config=pars):
        w_config = wandb.config

        model_name = 'ff_effnet'
        epochs = w_config['epochs']
        batch_size = w_config['batch_size']
        seed = 0;
        ranking_size = w_config['ranking_size']
        rankings_per_image = 784 // ranking_size
        wandb.log({'rpi': rankings_per_image, 'seed': seed})
        initial_lr = w_config['lr']
        lr_multi = w_config['lr_multi']

        ds_size = w_config['ds_size']
        sampling_type = w_config['sampling_type']
        equality_threshold = 0.03
        model_checkpoints = False
        load_model_path = ""
        augmentation = False
        warmup = 0
        config = init_env(autolog_freq=1, seed=seed)
        timestr = time.strftime("%d%m%y-%H%M%S")

        # Determine model, dataset and loss types
        model_type = get_model_type_by_name(model_name)
        dataset = "HR-WSI"
        dataset_type = get_dataset_type_by_name(dataset)
        loss_type = DepthLossType.NLL
        if sampling_type == 3 or sampling_type == 2:
            load_path = '/upb/departments/pc2/groups/hpc-prf-deepmde/praneeth/PLDepth/pldepth/weights/base/pr-base.h5'
        elif sampling_type == 1:
            load_path = '/upb/departments/pc2/groups/hpc-prf-deepmde/praneeth/PLDepth/pldepth/weights/base/base-inf-rs400.h5'

        else:
            print("wrong type")
        # load_path = '/home/praneeth/projects/thesis/git/PLDepth/pldepth/weights/100921-092654base_10rpi_1k_30ep_6r_model_rnd_sampling.h5'
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
        lr_sched_prov = LearningRateScheduleProvider(init_lr=initial_lr, steps=[3, 10], warmup=warmup,
                                                     multiplier=lr_multi)
        loss_fn = HourglassNegativeLogLikelihood(ranking_size=ranking_size,
                                                 batch_size=batch_size,
                                                 debug=False)

        optimizer = keras.optimizers.Adam(learning_rate=lr_sched_prov.get_lr_schedule(0), amsgrad=True)

        model.load_weights(load_path)
        model.compile(loss=loss_fn, optimizer=optimizer)

        callbacks = [TerminateOnNaN(), LearningRateScheduler(lr_sched_prov.get_lr_schedule),
                     WandbCallback(save_model=False)]

        # Apply preprocessing
        def preprocess_ds(loc_x, loc_y):
            return preprocess_fn(loc_x), loc_y

        sampling_strategy = PurelyMaskedRandomSamplingStrategy(model_params)
        model_params.set_parameter('sampling_strategy', sampling_strategy)

        data_path = config["DATA"]["HR_WSI_10K_PATH"]
        dao_a = HRWSITFDataAccessObject(data_path, model_input_shape, seed)

        test_imgs_ds, test_gts_ds, test_cons_masks = dao_a.get_training_dataset(size=ds_size)
        val_imgs_ds, val_gts_ds, val_cons_masks = dao_a.get_validation_dataset(size=200)

        data_provider = HourglassLargeScaleDataProvider(model_params, test_cons_masks, val_cons_masks,
                                                        augmentation=False,
                                                        loss_type=loss_type)


        train_ds = data_provider.provide_train_dataset(test_imgs_ds, test_gts_ds)

        val_ds = data_provider.provide_val_dataset(val_imgs_ds, val_gts_ds)

        # Apply preprocessing
        def preprocess_ds(loc_x, loc_y):
            return preprocess_fn(loc_x), loc_y

        train_ds = train_ds.map(preprocess_ds, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        val_ds = val_ds.map(preprocess_ds, num_parallel_calls=tf.data.experimental.AUTOTUNE)



        vds = list(val_imgs_ds.as_numpy_iterator())
        vgt = list(val_gts_ds.as_numpy_iterator())
        test_img = vds[:150]
        test_gt = vgt[:150]
        init_err = calc_err(model, test_img, test_gt, img_size=tuple(model_input_shape[:2]))
        print("$$$$ init error $$$: ", init_err)
        wandb.log({"init_error": init_err})
        steps_per_epoch = (ds_size//batch_size)
        model.fit(x=train_ds, epochs=model_params.get_parameter("epochs"), steps_per_epoch=steps_per_epoch,
                   callbacks=callbacks, validation_data=val_ds, verbose=1)



        min_err = calc_err(model, test_img, test_gt, img_size=tuple(model_input_shape[:2]))




        wandb.log({'test_err': min_err})



# if __name__ == "__main__":
#     perform_pldepth_experiment()


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
from hyperopt import STATUS_OK
import numpy as np
from pldepth.active_learning.metrics import ordinal_error
import wandb




def active_pldepth_experiment(pars=None):
    with wandb.init(config=pars):
        w_config = wandb.config


        model_name = 'ff_effnet'
        epochs = 30
        batch_size = w_config['batch_size']
        seed = 0;
        ranking_size = w_config['ranking_size']
        rankings_per_image = w_config['rpi']
        initial_lr = w_config['lr']
        lr_multi = w_config['lr_multi']
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
        #load_path = '/upb/departments/pc2/groups/hpc-prf-deepmde/praneeth/PLDepth/pldepth/weights/base/model-best-inf1k.h5'
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

        model_input_shape = [448, 448, 3]
        m, preprocess_fn = get_pl_depth_net(model_params, model_input_shape)

        # Compile model
        lr_sched_prov = LearningRateScheduleProvider(init_lr=initial_lr, steps=[4], warmup=warmup,
                                                     multiplier=lr_multi)
        loss_fn = HourglassNegativeLogLikelihood(ranking_size=ranking_size,
                                                 batch_size=batch_size,
                                                 debug=False)

        optimizer = keras.optimizers.Adam(learning_rate=lr_sched_prov.get_lr_schedule(0), amsgrad=True)
        # load the model here
        model = tf.keras.models.load_model(load_path,
                                           custom_objects={'EffNetFullyFledged': EffNetFullyFledged}, compile=False)
        model.compile(loss=loss_fn, optimizer=optimizer)
        model.summary()

        callbacks = [TerminateOnNaN(), LearningRateScheduler(lr_sched_prov.get_lr_schedule)
                     ]
        verbosity = 1
        if model_checkpoints:
            callbacks.append(construct_model_checkpoint_callback(config, model_type, verbosity))

        # Apply preprocessing
        def preprocess_ds(loc_x, loc_y):
            return preprocess_fn(loc_x), loc_y

        print("Start active sampling")
        data_path = config["DATA"]["HR_WSI_DEBUG_PATH"]
        dao_a = HRWSITFDataAccessObject(data_path, model_input_shape, seed)
        test_imgs_ds, test_gts_ds, test_cons_masks = dao_a.get_training_dataset()

        val_imgs_ds, val_gts_ds, val_cons_masks = dao_a.get_validation_dataset()

        data_provider = HourglassLargeScaleDataProvider(model_params, test_cons_masks, val_cons_masks,
                                                        augmentation=False,
                                                        loss_type=loss_type)
        val_ds = data_provider.provide_val_dataset(val_imgs_ds, val_gts_ds)

        active_train_ds = active_learning_data_provider(test_imgs_ds, test_gts_ds, model, batch_size=batch_size,
                                                        split_num=32)
        active_train_ds.map(preprocess_ds, num_parallel_calls=tf.data.experimental.AUTOTUNE)

        # fit active samples over the prev trainedd model.ls
        steps_per_epoch = int(5000 / batch_size)
        print("fit active sampled data")
        # model.fit(x=active_train_ds, epochs=epochs, steps_per_epoch=steps_per_epoch,
        #           callbacks=callbacks, verbose=1)
        loss = 4 #model.evaluate(val_ds)

        wandb.log({'val_loss': loss[0]})

# if __name__ == "__main__":
#     perform_pldepth_experiment()


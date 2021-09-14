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




def active_pldepth_experiment(pars):
    model_name = 'ff_effnet'
    epochs = 30
    batch_size = pars['batch_size']
    seed = 0;
    ranking_size = pars['ranking_size']
    rankings_per_image = pars['rpi']
    initial_lr = pars['lr']
    lr_multi = pars['lr_multi']
    equality_threshold = 0.03
    model_checkpoints = False
    load_model_path = ""
    augmentation = True
    warmup = 0
    config = init_env(autolog_freq=1, seed=seed)
    timestr = time.strftime("%d%m%y-%H%M%S")

    # Determine model, dataset and loss types
    model_type = get_model_type_by_name(model_name)
    dataset = "HR-WSI"
    dataset_type = get_dataset_type_by_name(dataset)
    loss_type = DepthLossType.NLL
    load_path = '/upb/departments/pc2/groups/hpc-prf-deepmde/praneeth/PLDepth/pldepth/weights/base/130921-11061710rpi_1k-0.0035lr-6r-mod_info_sampling.h5'

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

    sampling_strategy = ThresholdedMaskedRandomSamplingStrategy(
        model_params)  # InformationScoreBasedSampling(model_params)
    model_params.set_parameter('sampling_strategy', sampling_strategy)

    model_input_shape = [448, 448, 3]

    # # Get model
    m, preprocess_fn = get_pl_depth_net(model_params, model_input_shape)
    # model.summary()

    # Compile model
    lr_sched_prov = LearningRateScheduleProvider(init_lr=initial_lr, steps=[20], warmup=warmup, multiplier=lr_multi)
    loss_fn = HourglassNegativeLogLikelihood(ranking_size=ranking_size,
                                             batch_size=model_params.get_parameter("batch_size"),
                                             debug=False)

    optimizer = keras.optimizers.Adam(learning_rate=lr_sched_prov.get_lr_schedule(0), amsgrad=True)
    # load the model here
    model = tf.keras.models.load_model( load_path,
        custom_objects={'EffNetFullyFledged': EffNetFullyFledged}, compile=False)
    model.compile(loss=loss_fn, optimizer=optimizer)
    # model.summary()

    callbacks = [TerminateOnNaN(), LearningRateScheduler(lr_sched_prov.get_lr_schedule),
                 ]
    verbosity = 1
    if model_checkpoints:
        callbacks.append(construct_model_checkpoint_callback(config, model_type, verbosity))

    # Apply preprocessing
    def preprocess_ds(loc_x, loc_y):
        return preprocess_fn(loc_x), loc_y

    steps_per_epoch = int(1000 / batch_size)
    dao = HRWSITFDataAccessObject(config["DATA"]["HR_WSI_TEST_PATH"], model_input_shape, seed)

    val_imgs_ds, val_gts_ds, val_cons_masks = dao.get_validation_dataset()
    train_cons_masks=None
    data_provider = HourglassLargeScaleDataProvider(model_params, train_cons_masks, val_cons_masks,
                                                    augmentation=model_params.get_parameter("augmentation"),
                                                    loss_type=loss_type)
    val_ds = data_provider.provide_val_dataset(val_imgs_ds, val_gts_ds)

    print("Start active sampling")
    data_path = config["DATA"]["HR_WSI_TEST_PATH"]
    dao_a = HRWSITFDataAccessObject(data_path, model_input_shape, seed=42)
    test_imgs_ds, test_gts_ds, test_cons_masks = dao_a.get_training_dataset()

    active_train_ds = active_learning_data_provider(test_imgs_ds, test_gts_ds, model, batch_size=batch_size,
                                                    ranking_size=ranking_size, split_num=32)
    active_train_ds.map(preprocess_ds, num_parallel_calls=tf.data.experimental.AUTOTUNE)

    #fit active samples over the prev trainedd model.ls

    print("fit active sampled data")
    model.fit(x=active_train_ds, epochs=epochs, steps_per_epoch=steps_per_epoch, verbose=1)
    # Save the weights
    timestr = time.strftime("%d%m%y-%H%M%S")
    # model.save_weights('/scratch/hpc-prf-deepmde/praneeth/output/' + timestr + 'active_weight_rnd_sampling')
    #model.save('/scratch/hpc-prf-deepmde/praneeth/output/' + timestr + 'active_model_info_sampling.h5')

    eval_ds = list(val_imgs_ds.as_numpy_iterator())[:50]
    eval_gt = list(val_gts_ds.as_numpy_iterator())[:50]

    err_vec1 = []
    for i in range(len(eval_ds)):
        pred = model.predict(np.array([eval_ds[i]]), batch_size=None)
        err = ordinal_error(pred[0], eval_gt[i])
        err_vec1.append(err)

    los = np.mean(err_vec1)

    return {"loss":los, "status":STATUS_OK}


# if __name__ == "__main__":
#     perform_pldepth_experiment()


import numpy as np

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
from pldepth.models.pl_hourglass import EffNetFullyFledged
from pldepth.losses.nll_loss import HourglassNegativeLogLikelihood

import tensorflow as tf

from pldepth.util.env import init_env
from pldepth.models.models_meta import ModelParameters, get_model_type_by_name
from pldepth.util.training_utils import LearningRateScheduleProvider
from pldepth.util.tracking_utils import construct_model_checkpoint_callback, construct_tensorboard_callback
from pldepth.data.data_meta import TFDataAccessObject
import os
from active_learning.active_learning_method import active_learning_data_provider
# ####################
# @click.command()
# @click.option('--model_name', default='ff_effnet', help='Backbone model',
#               type=click.Choice(['ff_redweb', 'ff_effnet'], case_sensitive=False))
# @click.option('--epochs', default=50)
# @click.option('--batch_size', default=4)  # modified
# @click.option('--seed', default=0)
# @click.option('--ranking_size', default=3, help='Number of elements per training ranking')
# @click.option('--rankings_per_image', default=100, help='Number of rankings per image for training')
# @click.option('--initial_lr', default=0.01, type=click.FLOAT)
# @click.option('--equality_threshold', default=0.03, type=click.FLOAT, help='Threshold which corresponds to the tau '
#                                                                            'parameter as used in Section 3.5.')
# @click.option('--model_checkpoints', default=False, help='Indicator whether the currently best performing model should'
#                                                          ' be saved.', type=click.BOOL)
# @click.option('--load_model_path', default='', help='Specify the path to a model in order to load it')
# @click.option('--augmentation', default=True, type=click.BOOL)
# @click.option('--warmup', default=0, type=click.INT)
def perform_pldepth_experiment(model_name='ff_effnet', epochs=10, batch_size=1, seed=1, ranking_size=6, rankings_per_image=100, initial_lr=0.001,
                               equality_threshold=0.03, model_checkpoints=False, load_model_path='', augmentation=True, warmup=0):
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

    sampling_strategy = ThresholdedMaskedRandomSamplingStrategy(model_params)
    model_params.set_parameter('sampling_strategy', sampling_strategy)

    model_input_shape = [448, 448, 3]

    # Get model
    m, preprocess_fn = get_pl_depth_net(model_params, model_input_shape)
    return preprocess_fn

def read_offline_data(path, set_indicator):
    file_names_imgs = [s for s in tf.data.Dataset.list_files(os.path.join(path,
                                                                          '{}/{}/*{}'.format(set_indicator,
                                                                                             "imgs", ".jpg")),
                                                             shuffle=False).as_numpy_iterator()]
    file_name_lists = [s.replace(b'imgs', b'lists').replace(b'.jpg', b'.npy') for s in file_names_imgs]
    file_ds_list = [np.load(l)[:100] for l in file_name_lists]
    list_ds = tf.data.Dataset.from_tensor_slices(file_ds_list)


    file_ds_img = tf.data.Dataset.from_tensor_slices(file_names_imgs)
    img_ds = file_ds_img.map(TFDataAccessObject.read_file_jpg, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    img_reshape = [i.reshape(1,448,448,3) for i in img_ds.as_numpy_iterator()]
    img_ds2 = tf.data.Dataset.from_tensor_slices(img_reshape)
    return tf.data.Dataset.zip((img_ds2, list_ds))


if __name__ == "__main__":
    preprocess_fn = perform_pldepth_experiment()
    model = tf.keras.models.load_model('/home/praneeth/projects/thesis/git/bakup/PLDepth/pldepth/weights/150821-230138model_rnd_sampling.h5',
                                       custom_objects={'EffNetFullyFledged': EffNetFullyFledged}, compile=False)
    initial_lr = 0.001
    warmup=0

    # Compile model
    lr_sched_prov = LearningRateScheduleProvider(init_lr=initial_lr, steps=[25], warmup=warmup, multiplier=0.3162)
    loss_fn = HourglassNegativeLogLikelihood(ranking_size=6,
                                             batch_size=1,
                                             debug=False)

    optimizer = keras.optimizers.Adam(learning_rate=initial_lr*0.3162, amsgrad=True)
    model.compile(loss=loss_fn, optimizer=optimizer)  # pass metrics here
    model.summary()

    # f_path = '/home/praneeth/projects/thesis/git/HR-WSI/processed'
    # train_ds = read_offline_data(f_path, 'train')
    #
    #
    # Apply preprocessing
    def preprocess_ds(loc_x, loc_y):
        return preprocess_fn(loc_x), loc_y
    #
    #
    # #train_ds = train_ds.map(preprocess_ds, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    # # for a in train_ds.as_numpy_iterator():
    # #     print(a[0].shape, a[1].shape)
    # #     break
    #
    # steps_per_epoch = 2
    # print("training")
    # model.fit(x=train_ds, epochs=1, steps_per_epoch=steps_per_epoch, verbose=2)
    #
    #
    model_input_shape = [448, 448, 3]
    data_path = '/home/praneeth/projects/thesis/git/mytest/data'
    dao = HRWSITFDataAccessObject(data_path, model_input_shape, seed=42)
    test_imgs_ds, test_gts_ds, test_cons_masks = dao.get_training_dataset()

    active_train_ds = active_learning_data_provider(test_imgs_ds,test_gts_ds, model, batch_size=2,split_num=32)
    # active_train_ds.map(preprocess_ds, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    #
    # model.fit(x=active_train_ds, epochs=1, steps_per_epoch=10, verbose=2)

    print("active learning done")


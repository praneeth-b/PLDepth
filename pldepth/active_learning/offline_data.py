from pldepth.data.dao.hr_wsi import HRWSITFDataAccessObject
from pldepth.data.io_utils import get_dataset_type_by_name
from pldepth.data.providers.hourglass_provider import HourglassLargeScaleDataProvider
from pldepth.data.sampling import ThresholdedMaskedRandomSamplingStrategy , InformationScoreBasedSampling
from pldepth.losses.losses_meta import DepthLossType
from pldepth.losses.nll_loss import HourglassNegativeLogLikelihood
from tensorflow import keras
import tensorflow as tf
import numpy as np
from pldepth.util.env import init_env
from pldepth.models.models_meta import ModelParameters, get_model_type_by_name
import cv2
import logging
from pldepth.util.str_literals import DONE_STR

class Offline_data_provider(HourglassLargeScaleDataProvider):
    def __init__(self, model_params, train_cons_masks, val_cons_masks, **kwargs ):

        super().__init__(model_params, train_cons_masks, val_cons_masks, **kwargs)

    def provide_train_dataset(self, base_ds, base_ds_gts=None):

        imgs_gts_ds = tf.data.Dataset.zip((base_ds, self.train_consistency_masks, base_ds_gts))

        if self.augmentation:
            def augment_fn(loc_img, loc_mask, loc_gt):
                do_flip = tf.random.uniform([]) > 0.5

                loc_img = tf.cond(do_flip, lambda: tf.image.flip_left_right(loc_img), lambda: loc_img)

                loc_mask = tf.cond(do_flip,
                                   lambda: tf.image.flip_left_right(tf.expand_dims(tf.squeeze(loc_mask), axis=-1)),
                                   lambda: loc_mask)
                loc_gt = tf.cond(do_flip, lambda: tf.image.flip_left_right(tf.expand_dims(tf.squeeze(loc_gt), axis=-1)),
                                 lambda: loc_gt)

                loc_mask = tf.squeeze(loc_mask)
                loc_gt = tf.squeeze(loc_gt)

                return loc_img, loc_mask, loc_gt

            imgs_gts_ds = imgs_gts_ds.map(augment_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE)

        ranking_ds = imgs_gts_ds.map(lambda loc_x, loc_y, loc_z: tf.numpy_function(self.sample_rankings,
                                                                                   [loc_x, loc_y, loc_z],
                                                                                   [tf.float32, tf.float32]),
                                     num_parallel_calls=tf.data.experimental.AUTOTUNE)

        return ranking_ds

    def provide_val_dataset(self, base_ds, base_ds_gts=None, offline=False):
        imgs_gts_ds = tf.data.Dataset.zip((base_ds, self.val_consistency_masks, base_ds_gts))
        # Generates validation rankings in advance to keep them the same
        logging.debug("Generating validation rankings...")
        val_rankings = self.generate_validation_rankings(imgs_gts_ds)
        logging.debug(DONE_STR)

        val_rankings_ds = tf.data.Dataset.from_tensor_slices(val_rankings)

        return tf.data.Dataset.zip((base_ds, val_rankings_ds))

seed =42
config = init_env(autolog_freq=1, seed=seed)

# Determine model, dataset and loss types
#model_type = get_model_type_by_name(model_name)
dataset = "HR-WSI"
dataset_type = get_dataset_type_by_name(dataset)
loss_type = DepthLossType.NLL
ranking_size = 6
rankings_per_image = 1000
seed =42
augmentation = True
warmup = 0

# Run meta information
model_params = ModelParameters()
#model_params.set_parameter("model_type", model_type)
model_params.set_parameter("dataset", dataset_type)
model_params.set_parameter("ranking_size", ranking_size)
model_params.set_parameter("rankings_per_image", rankings_per_image)
model_params.set_parameter('val_rankings_per_img', rankings_per_image)
#model_params.set_parameter("batch_size", batch_size)
model_params.set_parameter("seed", seed)
#model_params.set_parameter('equality_threshold', equality_threshold)
model_params.set_parameter('loss_type', loss_type)
model_params.set_parameter('augmentation', augmentation)
model_params.set_parameter('warmup', warmup)

sampling_strategy = InformationScoreBasedSampling(model_params)
model_params.set_parameter('sampling_strategy', sampling_strategy)

model_input_shape = [448, 448, 3]


dao = HRWSITFDataAccessObject(config["DATA"]["HR_WSI_ROOT_PATH"], model_input_shape, seed)
train_imgs_ds, train_gts_ds, train_cons_masks = dao.get_training_dataset()
val_imgs_ds, val_gts_ds, val_cons_masks = dao.get_validation_dataset()
data_provider = Offline_data_provider(model_params, train_cons_masks, val_cons_masks,
                                                    augmentation=model_params.get_parameter("augmentation"),
                                                    loss_type=loss_type)


def write_tfData(dao,typ):
    if typ == 'train':
        path = "/scratch/hpc-prf-deepmde/praneeth/mytest/data/processed/train/" #"/home/praneeth/projects/thesis/git/HR-WSI/processed/train/"
    else:
        path = "/scratch/hpc-prf-deepmde/praneeth/mytest/data/processed/val/" #"/home/praneeth/projects/thesis/git/HR-WSI/processed/val/"


    d = dao.as_numpy_iterator()
    #print(len(list(d)))
    i=1
    for ele in d:
        im = cv2.normalize(ele[0], None, 0, 255, cv2.NORM_MINMAX)
        cv2.imwrite(path+"imgs/"+ str(i)+ ".jpg", im)
        np.save(path+"lists/"+str(i)+".npy", ele[1])
        i+=1
        


train_ds = data_provider.provide_train_dataset(train_imgs_ds, train_gts_ds)
#val_ds = data_provider.provide_val_dataset(val_imgs_ds, val_gts_ds)

write_tfData(train_ds, 'train')
#write_tfData(val_ds, 'val')
print("Done...!")

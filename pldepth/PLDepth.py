import wandb
from wandb.keras import WandbCallback
import os
from pldepth.data.dao.hr_wsi import HRWSITFDataAccessObject
from pldepth.data.io_utils import get_dataset_type_by_name
from pldepth.data.providers.hourglass_provider import HourglassLargeScaleDataProvider
from pldepth.data.sampling import ThresholdedMaskedRandomSamplingStrategy, InformationScoreBasedSampling, PurelyMaskedRandomSamplingStrategy
from pldepth.losses.losses_meta import DepthLossType
from pldepth.losses.nll_loss import HourglassNegativeLogLikelihood
from pldepth.models.PLDepthNet import get_pl_depth_net
import click
from tensorflow import keras
from tensorflow.python.keras.callbacks import TerminateOnNaN, LearningRateScheduler, CSVLogger
import mlflow
import time
import tensorflow as tf
from pldepth.util.env import init_env
from pldepth.models.models_meta import ModelParameters, get_model_type_by_name
from pldepth.util.training_utils import LearningRateScheduleProvider, SGDRScheduler, LearningRateLoggingCallback
from pldepth.util.tracking_utils import construct_model_checkpoint_callback, construct_tensorboard_callback
from pldepth.active_learning.metrics import calc_err, dcg_metric

import  numpy as np




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
    config = init_env(experiment_name=str(sampling_type), autolog_freq=1, seed=seed)
    print("dataset size is: ",ds_size)
    run_name= "Pldepth-train"


    
    print("##################### train started ###########################")
    timestr = time.strftime("%d%m%y-%H%M%S")
    #config = init_env(experiment_name=timestr+str(sampling_type), autolog_freq=1, seed=seed)
    print("the env var is: ", os.environ['WANDB_DIR']) 
    run = wandb.init(project="Pldepth-train", settings=wandb.Settings(_disable_stats=True), # dir='/scratch/hpc-prf-deepmde/praneeth/wandb-logs',
                     config={'model_name': model_name,
                             'epochs': epochs,
                             'batch_size': batch_size,
                             'seed': seed,
                             'ranking_size': ranking_size,
                             'rankings_per_image': rankings_per_image,
                             'initial_lr': initial_lr,
                             'sampling_type': sampling_type,
                             'lr_multi': lr_multi,
			     'dataset_size':ds_size
                             })
    w_config = wandb.config

    timestr = time.strftime("%d%m%y-%H%M%S")

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

    elif sampling_type == 3 :
        sampling_strategy = PurelyMaskedRandomSamplingStrategy(model_params)
    
    else:
        print("wrong selection of sampling type")
        return 13

    model_params.set_parameter('sampling_strategy', sampling_strategy)

    model_input_shape = [224, 224, 3]

    # Get model
    model, preprocess_fn = get_pl_depth_net(model_params, model_input_shape)
    #model.summary()

    # Compile model

    steps_per_epoch = int((ds_size*14/15)/ batch_size)
    schedule = SGDRScheduler(min_lr=initial_lr*(1/lr_multi),
                          max_lr= initial_lr,
                          steps_per_epoch=steps_per_epoch,
                          lr_decay=0.9,
                          cycle_length=epochs,    # decays over entire training- non cyclic
                          mult_factor=1)


    loss_fn = HourglassNegativeLogLikelihood(ranking_size=model_params.get_parameter("ranking_size"),
                                             batch_size=model_params.get_parameter("batch_size"),
                                             debug=False)

    optimizer = keras.optimizers.Adam(learning_rate=initial_lr, amsgrad=True)
    model.compile(loss=loss_fn, optimizer=optimizer)

    if load_model_path != "":
        model.load_weights(load_model_path)

    dao = HRWSITFDataAccessObject(config["DATA"]["HR_WSI_10K_PATH"], model_input_shape, seed)

    all_imgs_ds, all_gts_ds, all_cons_masks = dao.get_training_dataset(size=ds_size)
    val_imgs_ds = all_imgs_ds.take(ds_size//15)
    val_gts_ds = all_gts_ds.take(ds_size//15)
    val_cons_masks = all_cons_masks.take(ds_size//15)
    train_imgs_ds = all_imgs_ds.skip(ds_size//15)
    train_gts_ds = all_gts_ds.skip(ds_size//15)
    train_cons_masks = all_cons_masks.skip(ds_size//15)



    eval_imgs_ds, eval_gts_ds, eval_cons_masks, = dao.get_validation_dataset()

    data_provider = HourglassLargeScaleDataProvider(model_params, train_cons_masks, val_cons_masks,
                                                    augmentation=model_params.get_parameter("augmentation"),
                                                    loss_type=loss_type)

    train_ds = data_provider.provide_train_dataset(train_imgs_ds, train_gts_ds)
    val_ds = data_provider.provide_val_dataset(val_imgs_ds, val_gts_ds)
    


    timestr = time.strftime("%d%m%y-%H%M%S")
    callbacks = [TerminateOnNaN(), schedule, LearningRateLoggingCallback(),  #LearningRateScheduler(lr_sched_prov.get_lr_schedule),
                   WandbCallback()]
    verbosity = 1


    # Apply preprocessing
    def preprocess_ds(loc_x, loc_y):
        return preprocess_fn(loc_x), loc_y

    train_ds = train_ds.map(preprocess_ds, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    val_ds = val_ds.map(preprocess_ds, num_parallel_calls=tf.data.experimental.AUTOTUNE)


    model.fit(x=train_ds, epochs=model_params.get_parameter("epochs"), steps_per_epoch=steps_per_epoch,
               callbacks=callbacks, validation_data=val_ds, verbose=verbosity)
    # Save the weights

    model.save_weights('/scratch/hpc-prf-deepmde/praneeth/output/'+timestr+'weight_rnd_sampling')
    model.save('/scratch/hpc-prf-deepmde/praneeth/output/' + timestr + 'best-hyp_rnd_sampling.h5')

    #evaluate on test data:
    vds = list(eval_imgs_ds.as_numpy_iterator())
    vgt = list(eval_gts_ds.as_numpy_iterator())
    test_img = vds[:250]
    test_gt = vgt[:250]

    err = calc_err(model, test_img, test_gt, img_size=tuple(model_input_shape[:2]))
    wandb.run.summary["test_error"] = err

    dcg_val = dcg_metric(model, test_img, test_gt, list_size=200)
    wandb.run.summary["ndcg_200"] = dcg_val

    ## log images:
    ip_img = vds[10]
    gt_img = vgt[10]

    images = wandb.Image(np.array(ip_img), caption="input image")
    wandb.log({"ex_img": images})
    gt = wandb.Image(np.array(gt_img), caption="input ground truth")
    wandb.log({"ex_gt": gt})
    op_img = []
    for i in range(5):
        pred = model.predict(np.array([ip_img]), batch_size=None)
        op_img.append(np.squeeze(pred))
        print(pred.shape, np.squeeze(pred).shape)
        op = wandb.Image(np.squeeze(pred), caption="predicted depth")
        wandb.log({"ex_pred", op})

if __name__ == "__main__":
    perform_pldepth_experiment()

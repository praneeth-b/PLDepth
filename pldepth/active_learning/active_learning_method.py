import numpy as np
from pldepth.active_learning.preprocess_utils import splitImage, unsharp_mask, auto_canny
from pldepth.active_learning.metrics import hausdorf_dist, hausdorff_pair
import cv2
import tensorflow as tf

split_num = 32


def active_sampling(in_edges, pred_edges):
    """
    pass the edge map of input and predicted depth image respectively
    """
    split_in = splitImage(in_edges, split_num)   #32*32
    split_pred = splitImage(pred_edges, split_num)
    dist = np.zeros(split_in.shape[0])
    pts = np.zeros((split_in.shape[0], 2))

    for i in range(split_in.shape[0]):
        hd = hausdorf_dist(split_in[i], split_pred[i])
        # print(hd)
        pt_in, pt_pred = hausdorff_pair(split_in[i], split_pred[i])
        # print(pt_in)

        if not len(pt_in) == 0:  # for hausdorff dist not = inf
            st_r = (int(i / split_num) * split_in.shape[1]) + pt_in[0]
            st_c = int((i % split_num) * split_in.shape[2] + pt_in[1])
            dist[i] = hd
            pts[i] = np.array([st_r, st_c])

        else:
            st_r = int(i / split_num) * int(split_in.shape[1])
            st_c = int((i % split_num) * split_in.shape[2])
            dist[i] = 1000  # a random large value
            pts[i] = np.array([st_r, st_c])

    # sorting the hausdorf dist
    idx = np.argsort(dist)
    dist = dist[idx]
    pts = pts[idx]
    pos = pts[:, 0] * 448 + pts[:, 1]

    return pos.astype(np.uint32), pts.astype(np.uint32)    # pos, pos_XY  1024x2


def oracle(img, img_gts, pos_xy, ranking_size):
    list_size = ranking_size
    result_buffer = np.zeros([int(pos_xy.shape[0] / list_size), list_size, 2], dtype=np.float32)
    buf = np.zeros((list_size, 2))

    np.random.shuffle(pos_xy)
    j = 0
    for i in range(0, pos_xy.shape[0] - list_size, list_size):

        for k in range(list_size):
            buf[k, 0] = pos_xy[i + k, 0] * 448 + pos_xy[i+k, 1]
            buf[k, 1] = img_gts[tuple(pos_xy[i + k])]

        ix = np.argsort(buf[:,1])[::-1]
        result_buffer[j] = buf[ix]
        j += 1

    return result_buffer   # sort and return top 200   1024/6 x 6 x 2


def active_learning_data_provider(img_arr, img_gts_arr, model, batch_size, ranking_size=6, split_num=32):
    """
    inputs are a dataset of images and their respective ground truths
    """
    img_ds_in = list(img_arr.as_numpy_iterator())
    img_gt_in = list(img_gts_arr.as_numpy_iterator())

    a_ds_out = []
    sample_lists = [] #np.zeros([len(img_ds_in), split_num*split_num, 6])  # samples per img(14x14), ranking size

    i = 0
    for img_in, gts_in in zip(img_ds_in, img_gt_in):
        img_o = cv2.cvtColor(img_in, cv2.COLOR_RGB2GRAY)
        img_o = cv2.normalize(img_o, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        img_o = cv2.medianBlur(img_o, 15)  # cv2.blur(img_o, ksize=(7,7))
        # Using the Canny filter to get contours of orig image
        in_edges = auto_canny(img_o)

        # get edge map of predicted depth image/map
        pred_ele = model.predict(np.array([img_in]), batch_size=None)
        pred_ele = np.squeeze(pred_ele)
        pred_im_out = cv2.normalize(pred_ele, None, 0, 255, cv2.NORM_MINMAX)
        pred_im_sharp = unsharp_mask(pred_im_out)
        pred_edges = auto_canny(pred_im_sharp)

        pos, pos_xy = active_sampling(in_edges, pred_edges)
        oracle_samples = oracle(img_in, gts_in, pos_xy, ranking_size)
        sample_lists.append(oracle_samples)
        print("the img is",i)
        i+=1

    sample_list_tf = tf.data.Dataset.from_tensor_slices(sample_lists)

    return tf.data.Dataset.zip((img_arr, sample_list_tf)).batch(batch_size, drop_remainder=True).repeat()

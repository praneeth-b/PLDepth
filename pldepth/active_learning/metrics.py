import numpy as np
# from skimage.metrics import hausdorff_distance
import warnings
import numpy as np
from scipy.spatial import cKDTree


def hausdorff_distance(image0, image1):
    a_points = np.transpose(np.nonzero(image0))
    b_points = np.transpose(np.nonzero(image1))

    # Handle empty sets properly:
    # - if both sets are empty, return zero
    # - if only one set is empty, return infinity
    if len(a_points) == 0:
        return 0 if len(b_points) == 0 else np.inf
    elif len(b_points) == 0:
        return np.inf

    return max(max(cKDTree(a_points).query(b_points, k=1)[0]),
               max(cKDTree(b_points).query(a_points, k=1)[0]))


def hausdorf_dist(i1, i2):
    """ return hausdorf dist + the points at max distance"""

    return hausdorff_distance(i1, i2)


def hausdorff_pair(image0, image1):
    """ Returns the coordinates of the points at hausdorff distance"""

    a_points = np.transpose(np.nonzero(image0))
    b_points = np.transpose(np.nonzero(image1))

    # If either of the sets are empty, there is no corresponding pair of points
    if len(a_points) == 0 or len(b_points) == 0:
        warnings.warn("One or both of the images is empty.", stacklevel=2)
        return (), ()

    nearest_dists_from_b, nearest_a_point_indices_from_b = cKDTree(a_points).query(b_points)
    nearest_dists_from_a, nearest_b_point_indices_from_a = cKDTree(b_points) \
        .query(a_points)

    max_index_from_a = nearest_dists_from_b.argmax()
    max_index_from_b = nearest_dists_from_a.argmax()

    max_dist_from_a = nearest_dists_from_b[max_index_from_a]
    max_dist_from_b = nearest_dists_from_a[max_index_from_b]

    if max_dist_from_b > max_dist_from_a:
        return a_points[max_index_from_b], \
               b_points[nearest_b_point_indices_from_a[max_index_from_b]]
    else:
        return a_points[nearest_a_point_indices_from_b[max_index_from_a]], \
               b_points[max_index_from_a]


def ordinal_error(op, gt, imsize=(448, 448), num=100):
    np.random.seed(10)
    idx = np.random.choice(list(range(imsize[0] * imsize[1])), num * 2, replace=False)  # add seed or np random state
    idx0, idx1 = np.split(idx, 2)
    op_flat = op.flatten()
    gt_flat = gt.flatten()

    out_order = np.greater(op_flat[idx0], op_flat[idx1])
    gt_order = np.greater(gt_flat[idx0], gt_flat[idx1])
    accuracy = np.equal(out_order, gt_order).sum() / num
    return 1 - accuracy


def calc_err(model, test_im, test_gt, img_size=(448,448)):
    ev = []
    for i in range(len(test_im)):
        pred = model.predict(np.array([test_im[i]]), batch_size=None)
        err = ordinal_error(pred[0], test_gt[i], imsize=img_size)
        ev.append(err)

    return np.mean(ev)


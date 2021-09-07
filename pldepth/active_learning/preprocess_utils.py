import cv2
import numpy as np

def auto_canny(image, sigma=0.33):
    # compute the median of the single channel pixel intensities
    v = np.median(image)
    # apply automatic Canny edge detection using the computed median
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    #print("threshold is ", lower, "and", upper)
    edged = cv2.Canny(image, lower, upper)
    # return the edged image
    return edged


def unsharp_mask(image, kernel_size=(5, 5), sigma=1.0, amount=3.0, threshold=0):
    """Return a sharpened version of the image, using an unsharp mask."""
    blurred = cv2.GaussianBlur(image, kernel_size, sigma)
    sharpened = float(amount + 1) * image - float(amount) * blurred
    sharpened = np.maximum(sharpened, np.zeros(sharpened.shape))
    sharpened = np.minimum(sharpened, 255 * np.ones(sharpened.shape))
    sharpened = sharpened.round().astype(np.uint8)
    if threshold > 0:
        low_contrast_mask = np.absolute(image - blurred) < threshold
        np.copyto(sharpened, image, where=low_contrast_mask)
    return sharpened


def splitImage(img, n=32):
    """ image is broken into nxn pieces
    """
    split = [0] * (n * n)  # np.zeros(n*n)
    n = int(img.shape[0] / n)
    i = 0
    for r in range(0, img.shape[0], n):
        for c in range(0, img.shape[1], n):
            sm_img = img[r:r + n, c:c + n]
            split[i] = sm_img
            i += 1

    # plt.imshow(sm_img, cmap= 'gray')
    return np.array(split)

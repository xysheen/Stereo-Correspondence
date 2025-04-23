"""

"""
import os
import numpy as np
import cv2
import time
import maxflow
from tqdm import tqdm
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt

# input parameters
input_dir='input_images'
imnum = 4
disp_max = 300
scale = 3
lamb_rate = 0.3
doffs = [213.084, 804.175, 131.111, 125.36, 77.326, 279.184, 107.76, 176.45, 190.244, 146.53]

def load_images(img_dir, imnum, scale, ext=".png"):
    """
    load the image pair
    Args:
         img_dir: image directory
         imnum: the image number
         scale: image downgrade resizing scale
         ext: image type

    Returns:
         img_left (MxNx3 numpy array): left color image
         img_right (MxNx3 numpy array): right color image
    """
    img_left = 'img' + str(imnum) + '_left' + ext
    img_right = 'img' + str(imnum) + '_right' + ext
    img_left = cv2.imread(os.path.join(img_dir, img_left))
    img_right = cv2.imread(os.path.join(img_dir, img_right))

    h, w, _ = img_left.shape
    img_left = cv2.resize(img_left, (w // scale, h // scale),
                                interpolation=cv2.INTER_LINEAR)
    img_right = cv2.resize(img_right, (w // scale, h // scale),
                              interpolation=cv2.INTER_LINEAR)


    return img_left, img_right


def color_to_gray(image):
    """
    Convert color image to gray and return normalized image in float 64

    Args:
          image: Color image

    Return:
          image_gray: Normalized gray image
    """
    image_mono = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image_gray = image_mono.astype(np.float64)

    return image_gray

def Gaussian_blur(img, ksize = 7, sigma = 10):
    """
    Apply Gaussian blur to the images

    Args:
        img (numpy array): Gray scale

    Return:
         img_return (numpy array): Gray scale
    """
    img_pad = np.pad(img, pad_width = ksize // 2, mode = 'constant', constant_values = 0)
    img_blur = cv2.GaussianBlur(img_pad, (ksize, ksize), sigma)
    img_return = img_blur[ksize//2:-ksize//2+1, ksize//2:-ksize//2 + 1]

    return img_return


def disparity_map(img_left, img_right, disp_max, wsize = 13):
    """
    Calculate the data disparity map with given images

    Args:
        img_left: The left image
        img_right: The right image
        disp_max: Maximum value of disparity
        wsize: window size

    Returns:
        disparity: The disparity map
        cost: The minimum cost at each pixel (data cost)

    """
    height, width = img_left.shape
    disparity = np.zeros((height, width))
    cost = np.zeros((height, width, disp_max))
    half = wsize // 2

    for h in tqdm(range(half, height - half)):
        for w in range(half, width - half):
            patch_left = img_left[h - half : h + half + 1, w - half : w + half + 1]

            ssd_min = float('inf')
            disp = 0
            init = max(half, w - disp_max)
            for d in range(init, w + 1):
                patch_right = img_right[h - half : h + half + 1, d - half : d + half + 1]

                ssd = np.sum((patch_left - patch_right)**2)
                if ssd < ssd_min:
                    ssd_min = ssd
                    disp = w - d

            disparity[h, w] = disp
            cost[h, w] = ssd_min

    return disparity, cost


def SSD(img_left, img_right, disp_max, wsize = 7):
    """
    Calculate sum square difference with given two images and given
    disparity value

    Args:
        img_left (numpy array): the left image
        img_right (numpy array): the right image
        disp_max (int): the disparity value for pixel (h, w)
        wsize (int): window size for difference calculation

    Returns:
        cost (numpy array): the SSD cost value corresponds to [height, width, disp]

    """
    half = wsize // 2
    height, width = img_left.shape
    img_left = img_left.astype(np.float64)
    img_right = img_right.astype(np.float64)
    cost = np.full((height, width, disp_max), np.inf, dtype = np.float64)

    # pad the left image
    img_left_pad = np.pad(img_left, pad_width = half, mode = 'reflect')
    # pad the right image
    img_right_pad = np.pad(img_right, pad_width = half, mode='reflect')

    for d in tqdm(range(disp_max)):
        for h in range(half, height + half):
            for w in range(half, width + half):
                patch_left = img_left_pad[h - half : h + half + 1, w - half : w + half + 1]
                if w - d - half >= 0:
                    patch_right = img_right_pad[h - half : h + half + 1, w - d - half : w - d + half + 1]
                    ssd = np.sum((patch_left - patch_right)**2)
                    cost[h - half, w - half, d] = ssd

    return cost


def disparity_graph_cut(img_left, img_right, disp_max, lamb_rate):
    """
    Stereo correspondence using graph cut method

    Args:
        img_left (numpy array): the left image
        img_right (numpy array): the right image
        disp_max (int): maximum disparity value

    Return:
        disparity (numpy array): disparity map calculated from graph cut
    """
    print('Calculating SSD.')
    data_cost = SSD(img_left, img_right, disp_max)
    data_cost_max = np.max(data_cost[np.isfinite(data_cost)])
    data_cost_min = np.min(data_cost[np.isfinite(data_cost)])

    data_cost = (data_cost - data_cost_min) / (data_cost_max - data_cost_min)
    data_cost_md = np.median(data_cost[np.isfinite(data_cost)])
    data_disparity = np.argmin(data_cost, axis = 2)
    data_term = np.copy(data_disparity)

    print('Working on smoothing.')
    V = np.ones((disp_max, disp_max)) - np.eye(disp_max)
    V = V * data_cost_md * lamb_rate
    disparity_smoothed = maxflow.fastmin.aexpansion_grid(data_cost, V, labels = data_disparity)

    '''height, width = im_left.shape
    for alpha in tqdm(range(disp_max)):
        g = maxflow.Graph[float]()
        nodes = g.add_grid_nodes((height, width))

        # data cost
        for h in range(height):
            for w in range(width):
                current_disp = disparity[h, w]
                current_cost = data_cost[current_disp, h, w]
                alpha_cost = data_cost[alpha, h, w]
                current_cap = data_cost_max / (current_cost + 1.e-6)
                alpha_cap = data_cost_max / (alpha_cost + 1.e-6)
                #print(current_cap, alpha_cap)

                #g.add_tedge(nodes[h, w], current_cap, alpha_cap)
                g.add_tedge(nodes[h, w], 2, 1)


        # smoothness cost
        for h in range(height - 1):
            for w in range(width - 1):
                g.add_edge(nodes[h, w], nodes[h + 1, w], lamb, lamb)
                g.add_edge(nodes[h, w], nodes[h, w + 1], lamb, lamb)


        flow = g.maxflow()

        for h in range(height):
            for w in range(width):
                print(g.get_segment(nodes[h, w]), alpha)
                if disparity[h, w] != alpha and g.get_segment(nodes[h, w]) == 1:
                    disparity[h, w] = alpha'''

    return data_term, disparity_smoothed

def get_ground_truth(input_dir, imnum):
    """
    This function load the ground truth disparity

    Args:
        input_dir (str): input image directory
        imnum (str): input image number

    Returns:
        GT (numpy array): Ground truth
    """
    filename = os.path.join(input_dir, 'disp'+str(imnum)+'_0.pfm')
    file = open(filename, 'rb')
    header = file.readline().rstrip()
    color = header == b'PF'

    dim_line = file.readline()
    while dim_line.startswith(b'#'):
        dim_line = file.readline()
    width, height = map(int, dim_line.split())

    scale = float(file.readline().rstrip())
    if scale < 0:
        endian = '<'
    else:
        endian = '>'

    data = np.fromfile(file, endian + 'f')
    file.close()
    shape = (height, width, 3) if color else (height, width)
    data = np.reshape(data, shape)
    GT = np.flipud(data)

    return GT


def evaluate(input_dir, imnum, disp_max, doff, scale, lamb_rate):
    """
    This function make plots of the data disparity and smoothed disparity and
    compare them with the ground truth for evaluation

    Args:
        input_dir (str): input image directory
        imnum (str): input image number
        disp_max (int): maximum disparity allowed

    Return:

    """
    disp_max = disp_max // scale
    img_left, img_right = load_images(input_dir, imnum, scale)
    img_left_gray = color_to_gray(img_left)
    img_right_gray = color_to_gray(img_right)
    im_left = Gaussian_blur(img_left_gray)
    im_right = Gaussian_blur(img_right_gray)
    im_left = im_left.astype(np.float64)
    im_right = im_right.astype(np.float64)
    disp_data, disp_smoothed = disparity_graph_cut(im_left, im_right, disp_max, lamb_rate)
    disp_data = disp_data * scale
    disp_smoothed = disp_smoothed * scale

    height, width = im_left.shape

    #GT = get_ground_truth(input_dir, imnum)
    GT_data = np.load(os.path.join(input_dir, 'img'+str(imnum)+'GT.npz'))
    GT = GT_data['arr_0']
    GT = cv2.resize(GT, (width, height), interpolation=cv2.INTER_LINEAR)

    # exclude boundary
    wsize = 7
    shift = int(doff / scale)
    dd_reduce = disp_data[wsize // 2: height - wsize // 2, wsize // 2+ shift : width  - wsize // 2].astype(np.float64)
    ds_reduce = disp_smoothed[wsize // 2: height - wsize // 2, wsize // 2+ shift : width - wsize // 2].astype(np.float64)
    GT_reduce = GT[wsize // 2: height - wsize // 2, wsize // 2+ shift : width - wsize // 2].astype(np.float64)


    dd_reduce_norm = (dd_reduce - dd_reduce.min()) / (dd_reduce.max() - dd_reduce.min()) * 255
    ds_reduce_norm = (ds_reduce - ds_reduce.min()) / (ds_reduce.max() - ds_reduce.min()) * 255

    dd_color = cv2.applyColorMap(dd_reduce_norm.astype(np.uint8), cv2.COLORMAP_JET)
    cv2.imwrite('img' + str(imnum) + 'SSD.png', dd_color)

    ds_color = cv2.applyColorMap(ds_reduce_norm.astype(np.uint8), cv2.COLORMAP_JET)
    cv2.imwrite('img' + str(imnum) + 'smooth.png', ds_color)

    # remove infs and nans
    invalid_index = ~(np.isfinite(dd_reduce) & np.isfinite(ds_reduce) & np.isfinite(GT_reduce))
    dd_reduce[invalid_index] = 0
    ds_reduce[invalid_index] = 0
    GT_reduce[invalid_index] = 0

    GT_norm = (GT_reduce - GT_reduce.min()) / (GT_reduce.max() - GT_reduce.min()) * 255
    GT_color = cv2.applyColorMap(GT_norm.astype(np.uint8), cv2.COLORMAP_JET)
    cv2.imwrite('img' + str(imnum) + 'GT.png', GT_color)

    # absolute error
    mae_dd = mean_absolute_error(GT_reduce, dd_reduce)
    mae_ds = mean_absolute_error(GT_reduce, ds_reduce)

    # error map
    dd_error = np.abs(dd_reduce - GT_reduce)
    print('dd error mean: ', np.mean(dd_reduce) - np.mean(GT_reduce))
    dd_error_color = cv2.applyColorMap(dd_error.astype(np.uint8), cv2.COLORMAP_JET)
    cv2.imwrite('img' + str(imnum) + 'SSD_error.png', dd_error_color)

    ds_error = np.abs(ds_reduce - GT_reduce)
    print('ds error mean: ', np.mean( ds_reduce) - np.mean(GT_reduce))
    ds_error_color = cv2.applyColorMap(ds_error.astype(np.uint8), cv2.COLORMAP_JET)
    cv2.imwrite('img' + str(imnum) + 'Smooth_error.png', ds_error_color)

    return mae_dd, mae_ds


if __name__ == "__main__":

    mae_dd, mae_ds= evaluate(input_dir, imnum, disp_max, doffs[imnum - 1], scale, lamb_rate)
    print('SSD absolute error: ', mae_dd)
    print('Smoothed absolute error: ', mae_ds)



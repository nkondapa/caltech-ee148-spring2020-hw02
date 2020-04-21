import os
import numpy as np
import json
from PIL import Image
import matplotlib.pyplot as plt
import preprocessing
import utilities
import postprocessing
import ip_algorithms as ipa


def compute_convolution(I, T, stride=None, pixel_group=None, img_name=''):
    '''
    This function takes an image <I> and a template <T> (both numpy arrays) 
    and returns a heatmap where each grid represents the output produced by 
    convolution at each location. You can add optional parameters (e.g. stride, 
    window_size, padding) to create additional functionality. 
    '''



    '''
    BEGIN YOUR CODE
    '''

    if np.max(I) > 1:
        I = I/255

    if len(I.shape) != 3:
        I = np.expand_dims(I, axis=2)
    (n_rows, n_cols, n_channels) = np.shape(I)

    if len(T.shape) == 2 and n_channels != 1:
        (Trows, Tcols) = np.shape(T)
        T = np.expand_dims(T, axis=2)
        repT = np.tile(T, reps=3)
    elif len(T.shape) == 3 and n_channels == 1:
        T = np.mean(T, axis=2)
        (Trows, Tcols) = np.shape(T)
        repT = T
    else:
        (Trows, Tcols, Tchannels) = np.shape(T)
        repT = T

    # repT = repT/np.sum(repT)
    avg_repT = np.mean(repT, axis=(0, 1))
    center_repT = repT - avg_repT
    norm_repT = np.linalg.norm(center_repT, axis=(0, 1))
    # norm_repT = np.std(center_repT, axis=(0, 1))
    cnT = center_repT / norm_repT
    T_norm = (1 / np.linalg.norm(repT.flatten()))
    std_repT = np.std(repT)
    x, y = np.where(repT[:, :, 0] == np.max(repT[:, :, 0]))
    hot_x = round(np.mean(x).item())
    hot_y = round(np.mean(y).item())

    tr_Trows = hot_x
    br_Trows = Trows - tr_Trows
    lc_Tcols = hot_y
    rc_Tcols = Tcols - lc_Tcols
    # tr_Trows = int(Trows/2)
    # br_Trows = Trows - tr_Trows
    # lc_Tcols = int(Tcols / 2)
    # rc_Tcols = Tcols - lc_Tcols
    heatmap = np.zeros(shape=(n_rows, n_cols))

    if stride is None:
        stride = 1

    xvals = range(0, n_rows, stride)
    yvals = range(0, n_cols, stride)
    targeted = False
    yvals_master = None
    if pixel_group is not None:
        targeted = True
        xvals, yvals_master = zip(*pixel_group)

    xval_count = 0
    for i in xvals:
        if targeted:
            yvals = [yvals_master[xval_count]]
            xval_count += 1
        for j in yvals:
            padded_patch = np.zeros(shape=(T.shape[0], T.shape[1], n_channels))
            tr = max(0, i - tr_Trows)
            br = min(n_rows, i + br_Trows)
            lc = max(0, j - lc_Tcols)
            rc = min(n_cols, j + rc_Tcols)
            offsetr = abs(i - tr_Trows - tr) + abs(i + br_Trows - br)
            offsetc = abs(j - lc_Tcols - lc) + abs(j + rc_Tcols - rc)
            patch = I[tr:br, lc:rc, :]

            if patch.shape != T.shape:
                padded_patch[offsetr:, offsetc:, :] = patch
                patch = padded_patch

            # patch_norm_denom = np.linalg.norm(patch.flatten())
            # patch_norm = (1 / patch_norm_denom)
            # heatmap[i, j] = np.dot(patch.flatten(), repT.flatten()) * patch_norm * T_norm
            center_patch = patch - np.mean(patch, axis=(0, 1))
            norm_patch = np.linalg.norm(center_patch, axis=(0, 1))
            # norm_patch = np.std(center_patch, axis=(0, 1))
            cnp = center_patch/norm_patch
            for k in range(3):
                heatmap[i, j] += (np.dot(cnT[:, :, k].flatten(), cnp[:, :, k].flatten()))/3
            # heatmap[i, j] =
            #heatmap[i, j] = np.dot(patch.flatten(), repT.flatten())
            #
            # if heatmap[i, j] > 0.63:
            #     print(i, j, heatmap[i, j])
            #     plt.subplot(131)
            #     plt.imshow(patch)
            #     plt.subplot(132)
            #     plt.imshow(T)
            #     plt.subplot(133)
            #     plt.imshow(I)
            #     plt.show()
    '''
    END YOUR CODE
    '''

    return heatmap


def predict_boxes(heatmap):
    '''
    This function takes heatmap and returns the bounding boxes and associated
    confidence scores.
    '''

    output = []

    '''
    BEGIN YOUR CODE
    '''
    
    '''
    As an example, here's code that generates between 1 and 5 random boxes
    of fixed size and returns the results in the proper format.
    '''

    # box_height = 8
    # box_width = 6
    #
    # num_boxes = np.random.randint(1, 5)
    #
    # for i in range(num_boxes):
    #     (n_rows, n_cols, n_channels) = np.shape(I)
    #
    #     tl_row = np.random.randint(n_rows - box_height)
    #     tl_col = np.random.randint(n_cols - box_width)
    #     br_row = tl_row + box_height
    #     br_col = tl_col + box_width
    #
    #     score = np.random.random()
    #
    #     output.append([tl_row,tl_col,br_row,br_col, score])
    groups, group_centers, _ = postprocessing.group_pixels(img_arr=heatmap)
    bounding_boxes, scores, output = postprocessing.groups_to_bounding_boxes(groups, heatmap)


    '''
    END YOUR CODE
    '''

    return output


def detect_red_light_mf(I, img=None, name=''):
    '''
    This function takes a numpy array <I> and returns a list <output>.
    The length of <output> is the number of bounding boxes predicted for <I>. 
    Each entry of <output> is a list <[row_TL,col_TL,row_BR,col_BR,score]>. 
    The first four entries are four integers specifying a bounding box 
    (the row and column index of the top left corner and the row and column 
    index of the bottom right corner).
    <score> is a confidence score ranging from 0 to 1. 

    Note that PIL loads images in RGB order, so:
    I[:,:,0] is the red channel
    I[:,:,1] is the green channel
    I[:,:,2] is the blue channel
    '''

    '''
    BEGIN YOUR CODE
    '''
    template_height = 8
    template_width = 6
    import time
    import pickle as pkl
    # You may use multiple stages and combine the results
    num_kernels = 6
    kernel_list = []
    kernel_sizes = []
    for i in range(6):
        kernel_list.append(utilities.load_kernel(str(i), '../data/kernels/'))
        # plt.imshow(kernel_list[-1])
        # plt.show()
        kernel_sizes.append(preprocessing.get_patch_hot_spot_size(kernel_list[-1]))
    rgb_pixel_array = np.load('../data/red_light_pixels/rgb_pixel_array.npy')

    map_mih = preprocessing.color_match_red_lights(I, rgb_pixel_array, stride=2)
    thresholded_mih_map = postprocessing.threshold_convolved_image(map_mih, 0.94, mode='down')
    smoothed_thresholded_mih_map = ipa.neighbor_max_smooth_heatmap(thresholded_mih_map, np.zeros(shape=(5, 5)))
    # plt.subplot(121)
    # plt.imshow(I)
    # plt.subplot(122)
    # plt.imshow(map_mih)
    # plt.show()

    # plt.subplot(121)
    # plt.imshow(thresholded_mih_map)
    # plt.subplot(122)
    # plt.imshow(smoothed_thresholded_mih_map)
    # plt.show()
    groups, group_centers, pixels = postprocessing.group_pixels(smoothed_thresholded_mih_map)
    kernel_inds = postprocessing.match_group_to_kernel(groups, kernel_sizes)

    heatmap = np.zeros(shape=map_mih.shape)
    heatmaps = []

    for kernel in kernel_list:
        kernel_heatmaps = []
        for i, group in enumerate(groups):
            # kernel = kernel_list[kernel_inds[i]]
            group = postprocessing.group_center_to_pixel_group(group_centers[i], kernel, img_size=I.shape)
            hmap = compute_convolution(I, kernel, stride=1, pixel_group=group)
            kernel_heatmaps.append(hmap)
            # plt.imshow(heatmaps[-1])
            # plt.show()
        kernel_heatmap = np.max(kernel_heatmaps, axis=0)
        plt.imshow(kernel_heatmap)
        plt.show()
        # kernel_conf = np.max(np.max(heatmaps, axis=0) - np.min(heatmaps, axis=0))
        # print(kernel_conf)
    heatmap = np.mean(heatmaps, axis=0)
        # plt.imshow(heatmap)
        # plt.show()
    output = []
    if len(groups) > 0:
        # heatmap = np.max(heatmaps, axis=0)
        plt.imshow(heatmap)
        plt.show()
        heatmap = postprocessing.threshold_convolved_image(heatmap, 0.85, mode='down')
        groups, _, _ = postprocessing.group_pixels(heatmap)
        _, _, output = postprocessing.groups_to_bounding_boxes(groups, heatmap)

    import visualize
    visualize.visualize_image_with_bounding_boxes(img, output)
    visualize.show()
    '''
    END YOUR CODE
    '''

    for i in range(len(output)):
        assert len(output[i]) == 5
        assert (output[i][4] >= 0.0) and (output[i][4] <= 1.0)

    return output

# Note that you are not allowed to use test data for training.
# set the path to the downloaded data:
data_path = '../data/RedLights2011_Medium'

# load splits: 
split_path = '../data/hw02_splits'
file_names_train = np.load(os.path.join(split_path,'file_names_train.npy'))
file_names_test = np.load(os.path.join(split_path,'file_names_test.npy'))

# set a path for saving predictions:
preds_path = '../data/hw02_preds'
os.makedirs(preds_path, exist_ok=True) # create directory if needed

# Set this parameter to True when you're done with algorithm development:
done_tweaking = False

'''
Make predictions on the training set.
'''
preds_train = {}
# print(file_names_train)
for i in range(len(file_names_train)):
    if i not in [4]:
        continue

    print(file_names_train[i])
    # read image using PIL:
    img = Image.open(os.path.join(data_path,file_names_train[i]))

    # convert to numpy array:
    I = np.asarray(img)

    preds_train[file_names_train[i]] = detect_red_light_mf(I, img, file_names_train[i])

# save preds (overwrites any previous predictions!)
with open(os.path.join(preds_path,'preds_train.json'),'w') as f:
    json.dump(preds_train,f)

if done_tweaking:
    '''
    Make predictions on the test set. 
    '''
    preds_test = {}
    for i in range(len(file_names_test)):

        # read image using PIL:
        I = Image.open(os.path.join(data_path,file_names_test[i]))

        # convert to numpy array:
        I = np.asarray(I)

        preds_test[file_names_test[i]] = detect_red_light_mf(I)

    # save preds (overwrites any previous predictions!)
    with open(os.path.join(preds_path, 'preds_test.json'),'w') as f:
        json.dump(preds_test,f)

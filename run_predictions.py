import os
import numpy as np
import json
from PIL import Image
import matplotlib.pyplot as plt
import preprocessing
import utilities
import postprocessing
import ip_algorithms as ipa


def compute_convolution(I, T, stride=(None, None), pixel_group=None, img_name=''):
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

    I = I - np.mean(I, axis=(0, 1))

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

    if stride is (None, None):
        stride = (1, 1)

    xvals = range(0, n_rows, stride[0])
    yvals = range(0, n_cols, stride[1])
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
    kernel_list = []
    kernel_sizes = []
    exclude = [3, 5, 7, 13, 15, 17]
    n_kernels = 6 * 3
    for i in range(n_kernels):
        if i in exclude:
            continue
        kernel_list.append(utilities.load_kernel(str(i), '../data/kernels/'))
        kernel_sizes.append(preprocessing.get_patch_hot_spot_size(kernel_list[-1]))
    rgb_pixel_array = np.load('../data/red_light_pixels/rgb_pixel_array.npy')

    map_mih = preprocessing.color_match_red_lights(I, rgb_pixel_array, stride=(1, 2))
    thresholded_mih_map = postprocessing.threshold_convolved_image(map_mih, 0.94, mode='down')
    smoothed_thresholded_mih_map = ipa.neighbor_max_smooth_heatmap(thresholded_mih_map, np.zeros(shape=(5, 5)))
    groups1, group_centers1, pixels = postprocessing.group_pixels(smoothed_thresholded_mih_map)

    heatmaps = []
    group_kernel_scores = np.zeros(shape=(len(kernel_list), len(groups1)))
    plt.subplot(121)
    plt.imshow(smoothed_thresholded_mih_map)
    plt.subplot(122)
    plt.imshow(img)
    plt.show()
    exclude_group = []
    for k, kernel in enumerate(kernel_list):
        kernel_heatmaps = []
        #print('new kernel')
        #print(len(groups1))
        for i, group in enumerate(groups1):
            if i in exclude_group:
                continue
            group = postprocessing.group_center_to_pixel_group(group_centers1[i], group, kernel, img_size=I.shape)
            # print(len(group))
            hmap = compute_convolution(I, kernel, stride=(1, 1), pixel_group=group)
            kernel_heatmaps.append(hmap)
            group_kernel_scores[k, i] = np.max(hmap)
            vis = False
            if (282, 478) in group and vis:
                plt.subplot(131)
                plt.imshow(hmap)
                plt.subplot(132)
                mask = np.copy(hmap)
                mask[mask > 0] = 1
                mask[mask < np.mean(mask)] = 0.25
                plt.imshow(I/255 * mask[:, :, None])
                plt.subplot(133)
                plt.imshow(kernel)
                plt.show()

        kernel_heatmap = np.max(kernel_heatmaps, axis=0)
        heatmaps.append(kernel_heatmap)
    heatmap = np.max(heatmaps, axis=0)

    # plt.subplot(131)
    # plt.imshow(heatmap)
    # plt.subplot(132)
    # plt.imshow(avg_heatmap)
    # plt.subplot(133)
    # plt.imshow(img)
    # plt.show()

    output = []
    if len(groups1) > 0:
        thresh_heatmap = postprocessing.threshold_convolved_image(heatmap, 0.83, mode='down')
        groups, group_centers, _ = postprocessing.group_pixels(thresh_heatmap)
        matched_indices = postprocessing.match_group_centers_to_groups(group_centers, groups1)
        bb_heatmap = np.zeros(shape=heatmap.shape)
        # plt.subplot(121)
        # plt.imshow(thresh_heatmap)
        # plt.subplot(122)
        # plt.imshow(img)
        plt.show()
        cmc_list = []
        for i, ind in enumerate(matched_indices):
            if ind != -1:
                gc = group_centers[i]
                kind = int(np.argmax(group_kernel_scores[:, ind]))
                kernel = kernel_list[kind]
                cmc = postprocessing.color_match_score(gc, kernel, I)
                cmc_list.append(cmc)
                if cmc > 0.88:
                    postprocessing.add_kernel_patch(gc, kernel, heatmap, bb_heatmap)
        groups, _, _ = postprocessing.group_pixels(bb_heatmap)
        _, _, output = postprocessing.groups_to_bounding_boxes(groups, cmc_list, bb_heatmap)

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
    if i not in [7, 8, 9, 10, 11, 12]:
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

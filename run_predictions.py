import os
import numpy as np
import json
from PIL import Image
import matplotlib.pyplot as plt
import postprocessing


def compute_convolution(I, T, stride=None, img_name=''):
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

    T_norm = (1 / np.linalg.norm(repT.flatten()))
    avg_repT = np.mean(repT, axis=2)
    x, y = np.where(repT[:, :, 0] == np.max(repT[:, :, 0]))
    hot_x = round(np.mean(x).item())
    hot_y = round(np.mean(y).item())

    tr_Trows = int(Trows/2)
    br_Trows = Trows - tr_Trows
    lc_Tcols = int(Tcols / 2)
    rc_Tcols = Tcols - lc_Tcols

    # heatmap = np.random.random((n_rows, n_cols))
    heatmap = np.zeros(shape=(n_rows, n_cols))
    pixel_dist_diag_score = np.zeros(shape=(n_rows, n_cols))
    norm_matrix = np.zeros(shape=(n_rows, n_cols))
    if stride is None:
        stride = 1
    for i in range(0, n_rows, stride):
        for j in range(0, n_cols, stride):
            padded_patch = np.zeros(shape=(T.shape[0], T.shape[1], n_channels))
            tr = max(0, i - tr_Trows)
            br = min(n_rows, i + br_Trows)
            lc = max(0, j - lc_Tcols)
            rc = min(n_cols, j + rc_Tcols)
            offsetr = abs(i - tr_Trows - tr) + abs(i + br_Trows - br)
            offsetc = abs(j - lc_Tcols - lc) + abs(j + rc_Tcols - rc)
            patch = I[tr:br, lc:rc, :]
            # print(i, j)

            if patch.shape != T.shape:
                # print(i, j, offsetr, offsetc)
                # print(patch.shape)
                padded_patch[offsetr:, offsetc:, :] = patch
                patch = padded_patch

            patch_norm_denom = np.linalg.norm(patch.flatten())
            patch_norm = (1 / patch_norm_denom)
            heatmap[i, j] = np.dot(patch.flatten(), repT.flatten()) * patch_norm * T_norm
            norm_matrix[i, j] = patch_norm_denom
            # patch_diag = np.diagonal(patch)
            pixel_dist_diag_score[i, j] = 1 - np.mean(abs(repT[hot_x, hot_y] - patch[hot_x, hot_y]))
            # if pixel_dist_diag_score[i, j] > 0.7:
            #     # print(i, j)
            #     print()
            #     plt.subplot(1, 2, 1)
            #     plt.imshow(T*T_norm)
            #     plt.subplot(1, 2, 2)
            #     plt.imshow(patch*patch_norm)
            # for k in range(n_channels):
            #
            #     np.sum(patch * T)
            #     heatmap[]
            #     print(i, j, k)
    #
    # plt.subplot(1, 3, 1)
    # plt.imshow(T)
    # plt.subplot(1, 3, 2)
    # plt.imshow(np.squeeze(I))
    # plt.subplot(1, 3, 3)
    # plt.imshow(heatmap)
    # plt.colorbar()
    # plt.show()
    '''
    END YOUR CODE
    '''

    return heatmap, pixel_dist_diag_score, norm_matrix


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


def detect_red_light_mf(I, name=''):
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
    import utilities
    import ip_algorithms as ipa
    import time
    import pickle as pkl
    # You may use multiple stages and combine the results
    # T = np.random.random((template_height, template_width))
    # T = utilities.generate_gaussian_kernel(s=6, sigma=1.5)
    num_kernels = 4
    T_list = []
    for i in range(num_kernels):
        T_list.append(utilities.load_kernel(str(i), '../data/kernels/'))

    # rough search for traffic lights
    heatmaps = []
    norm_matrices = []
    pixel_dist_diag_scores = []
    st = time.time()
    load = True
    if not load:
        for i in range(len(T_list)):
            print(i)
            heatmap, pixel_dist_diag_score, norm_matrix = compute_convolution(I, T_list[i], stride=2)
            heatmaps.append(heatmap)
            norm_matrices.append(norm_matrix)
            pixel_dist_diag_scores.append(pixel_dist_diag_score)
        with open('../data/heatmaps/temp.pkl', 'wb') as f:
            pkl.dump([heatmaps, pixel_dist_diag_scores, norm_matrices], f)
    else:
        with open('../data/heatmaps/temp.pkl', 'rb') as f:
            # pkl.dump(heatmaps, f)
            heatmaps, pixel_dist_diag_scores, norm_matrices = pkl.load(f)
    print(time.time() - st)


    # smooth heatmaps
    st = time.time()
    # heatmaps2 = []
    # for heatmap in heatmaps:
    #     # heatmap = heatmap - np.mean(heatmap[heatmap > 0])
    #     # heatmap[heatmap < 0] = 0
    #     thr = np.mean(heatmap[heatmap > np.mean(heatmap[heatmap > 0])])
    #     print(thr)
    #     heatmaps2.append(postprocessing.threshold_convolved_image(heatmap, thr, 'down'))
    # heatmaps = ipa.smooth_heatmaps(heatmaps, utilities.generate_gaussian_kernel(s=4, sigma=1))
    for i in range(len(T_list)):
        # plt.subplot(1, 2, 1)
        # plt.imshow(heatmaps[i])
        # plt.subplot(1, 2, 2)
        # plt.imshow(norm_matrices[i])
        # plt.show()
        # plt.imshow(heatmaps[i] * norm_matrices[i])
        # plt.show()
        plt.imshow(heatmaps[i])
        plt.show()
        plt.imshow(pixel_dist_diag_scores[i])
        plt.show()
    avg_heatmap = np.mean(heatmaps, axis=0)
    print(time.time() - st)
    plt.imshow(avg_heatmap)
    plt.show()
    print()

    # heatmap1 = compute_convolution(I, T, stride=1)
    # np.save('../data/heatmaps/' + name.rstrip('.jpg'), heatmap1)
    # heatmap1 = np.load('../data/heatmaps/' + name.rstrip('.jpg') + '.npy')
    # # theatmap1 = postprocessing.threshold_convolved_image(heatmap1, 0.75, 'down')
    # theatmap1 = postprocessing.rank_threshold_convolved_image(heatmap1, 20, 'down')
    # groups, group_centers, _ = postprocessing.group_pixels(img_arr=theatmap1)
    # # print(group_centers)
    #
    # heatmaps = ipa.targeted_filtering(I, group_centers, [T])
    # plt.imshow(heatmaps[0])
    # plt.show()
    print()
    # for thr in range(0, 20):
    #     theatmap1 = postprocessing.threshold_convolved_image(heatmap1, thr/20, 'down')
    #     plt.subplot(1, 3, 1)
    #     plt.imshow(heatmap1)
    #     plt.subplot(1, 3, 2)
    #     plt.imshow(theatmap1)
    #     # plt.subplot(1, 4, 2)
    #     # plt.imshow(heatmap2)
    #     # plt.subplot(1, 4, 3)
    #     # plt.imshow(theatmap2)
    #     plt.subplot(1, 3, 3)
    #     plt.imshow(I)
    #     figManager = plt.get_current_fig_manager()
    #     figManager.full_screen_toggle()
    #
    #     plt.show()
    # output = predict_boxes(heatmaps[0])
    output = []
    '''
    END YOUR CODE
    '''
    #
    # for i in range(len(output)):
    #     assert len(output[i]) == 5
    #     assert (output[i][4] >= 0.0) and (output[i][4] <= 1.0)
    #
    # return output

# Note that you are not allowed to use test data for training.
# set the path to the downloaded data:
data_path = '../data/RedLights2011_tiny'

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
for i in range(len(file_names_train)):
    if i not in [3]:
        continue

    print(file_names_train[i])
    # read image using PIL:
    I = Image.open(os.path.join(data_path,file_names_train[i]))

    # convert to numpy array:
    I = np.asarray(I)

    preds_train[file_names_train[i]] = detect_red_light_mf(I, file_names_train[i])

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

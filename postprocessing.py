import numpy as np
import matplotlib.pyplot as plt
from collections import deque
import utilities


def threshold_convolved_image(img_arr_orig, threshold, mode='both'):

    img_arr = np.copy(img_arr_orig)

    if mode == 'both':
        img_arr[img_arr < threshold] = 0
        img_arr[img_arr > threshold] = 1
    elif mode == 'up':
        img_arr[img_arr > threshold] = 1
    elif mode == 'down':
        img_arr[img_arr < threshold] = 0

    return img_arr


def rank_threshold_convolved_image(img_arr_orig, rank=20, mode='both'):

    img_arr = np.copy(img_arr_orig)

    sorted_heatmap = sorted(np.unique(img_arr.flatten()), reverse=True)
    threshold = sorted_heatmap[rank+1]

    if mode == 'both':
        img_arr[img_arr < threshold] = 0
        img_arr[img_arr >= threshold] = 1
    elif mode == 'up':
        img_arr[img_arr > threshold] = 1
    elif mode == 'down':
        img_arr[img_arr < threshold] = 0

    return img_arr


def group_pixels(img_arr):

    x, y = np.where(img_arr > 0)

    points = set(zip(x, y))
    points_to_pass = set(zip(x, y))
    bfs = deque()
    groups = []
    group_centers = []
    num_items = 0
    # print('grouping...')
    while len(points) > 0:
        sub_x = []
        sub_y = []
        start_point = points.pop()
        bfs.append(start_point)
        groups.append(set())
        groups[-1].add(start_point)
        num_items += 1
        while len(bfs) > 0:
            x, y = bfs.popleft()
            sub_x.append(x)
            sub_y.append(y)
            xl = [x - 1] * 3 + [x] * 3 + [x + 1] * 3
            yl = [y - 1, y, y + 1] * 3
            candidates = set(zip(xl, yl))
            for c in candidates:
                if c in points:
                    points.remove(c)

                    groups[-1].add(c)
                    num_items += 1
                    bfs.append(c)
        group_centers.append([int(np.mean(sub_x)), int(np.mean(sub_y))])
        # if len(groups[-1]) <= 5**2:
        #     groups.pop(-1)
    return groups, group_centers, points_to_pass


def groups_to_bounding_boxes(groups, color_match_score_list, img_arr=None):

    bounding_boxes = []
    scores = []
    combined = []
    for i, group in enumerate(groups):
        most_left = None
        most_top = None
        most_right = None
        most_bottom = None
        heat_sum = 0
        for p in group:
            if most_left is None or p[1] < most_left:
                most_left = p[1]
            if most_top is None or p[0] < most_top:
                most_top = p[0]
            if most_right is None or p[1] > most_right:
                most_right = p[1]
            if most_bottom is None or p[0] > most_bottom:
                most_bottom = p[0]
            heat_sum += img_arr[p]
        score = heat_sum / len(group)
        score = (score + color_match_score_list[i]) / 2
        scores.append(score)
        print(most_top, most_left, most_bottom, most_right, score)
        bounding_box = [int(most_top), int(most_left), int(most_bottom), int(most_right)]
        bounding_boxes.append(bounding_box)
        combined.append([int(most_top), int(most_left), int(most_bottom), int(most_right), score])

    return bounding_boxes, scores, combined


def group_adjust_to_kernel_coords(group, kernel_coords):

    x, y = kernel_coords

    group_adjusted = set()
    for p in group:
        group_adjusted.add((p[0] - x, p[1]-y))

    return group_adjusted


def match_group_to_kernel(groups, kernel_sizes):

    group_sizes = [len(group) for group in groups]

    kernel_inds = []
    for gs in group_sizes:
        kernel_inds.append(np.argmin(abs(gs - np.array(kernel_sizes))))

    return kernel_inds


def group_center_to_pixel_group(group_center, group, T, img_size):

    (n_rows, n_cols, n_channels) = img_size

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

    tr = max(0, group_center[0] - tr_Trows)
    br = min(n_rows, group_center[0] + br_Trows)
    lc = max(0, group_center[1] - lc_Tcols)
    rc = min(n_cols, group_center[1] + rc_Tcols)
    offsetr = abs(group_center[0] - tr_Trows - tr) + abs(group_center[0] + br_Trows - br)
    offsetc = abs(group_center[1] - lc_Tcols - lc) + abs(group_center[1] + rc_Tcols - rc)

    pixel_group = group
    for i in range(tr, br+1):
        if i < 0 or i >= img_size[0]:
            continue
        for j in range(lc, rc+1):
            if j < 0 or j >= img_size[1]:
                continue
            pixel_group.add((i, j))

    return pixel_group


def match_group_centers_to_groups(group_centers, groups):

    group_centers_group_indices = []
    for gc in group_centers:
        group_centers_group_indices.append(-1)
        for j, group in enumerate(groups):

            if tuple(gc) in group:
                group_centers_group_indices[-1] = j

    return group_centers_group_indices


def add_kernel_patch(gc, kernel, source_heatmap, dest_heatmap):
    (Trows, Tcols, Tchannels) = np.shape(kernel)

    x, y = np.where(kernel[:, :, 0] == np.max(kernel[:, :, 0]))
    hot_x = round(np.mean(x).item())
    hot_y = round(np.mean(y).item())

    tr_Trows = hot_x
    br_Trows = Trows - tr_Trows
    lc_Tcols = hot_y
    rc_Tcols = Tcols - lc_Tcols

    n_rows, n_cols = dest_heatmap.shape
    i = gc[0]
    j = gc[1]
    tr = max(0, i - tr_Trows)
    br = min(n_rows, i + br_Trows)
    lc = max(0, j - lc_Tcols)
    rc = min(n_cols, j + rc_Tcols)
    dest_heatmap[tr:br, lc:rc] = source_heatmap[tr:br, lc:rc]


def get_group_center_from_group(group):

    x, y = zip(*group)
    cx = int(np.mean(x))
    cy = int(np.mean(y))

    return cx, cy


def color_match_score(gc, kernel, img):

    if np.max(img) > 1:
        I = img / 255
    else:
        I = img

    (Trows, Tcols, Tchannels) = np.shape(kernel)
    kernel = kernel - np.mean(kernel)
    x, y = np.where(kernel[:, :, 0] == np.max(kernel[:, :, 0]))
    hot_x = round(np.mean(x).item())
    hot_y = round(np.mean(y).item())

    tr_Trows = hot_x
    br_Trows = Trows - tr_Trows
    lc_Tcols = hot_y
    rc_Tcols = Tcols - lc_Tcols

    n_rows, n_cols, n_channels = img.shape
    i = gc[0]
    j = gc[1]
    tr = max(0, i - tr_Trows)
    br = min(n_rows, i + br_Trows)
    lc = max(0, j - lc_Tcols)
    rc = min(n_cols, j + rc_Tcols)


    patch = I[tr:br, lc:rc]
    patch = patch - np.mean(patch)

    if patch.shape[0] < kernel.shape[0]:
        d = abs(kernel.shape[0] - patch.shape[0])
        if i - tr_Trows < 0:
            kernel = kernel[d:, :, :]
        else:
            kernel = kernel[:Trows - d, :, :]

    if patch.shape[1] < kernel.shape[1]:
        d = abs(kernel.shape[1] - patch.shape[1])
        if i - lc_Tcols < 0:
            kernel = kernel[:, d:, :]
        else:
            kernel = kernel[:, :Tcols - d, :]

    n_patch = patch / np.linalg.norm(patch, axis=2)[:, :, None]
    n_kernel = kernel / np.linalg.norm(kernel, axis=2)[:, :, None]
    scores = np.sum(n_patch * n_kernel, axis=2)
    gk = utilities.generate_gaussian_kernel(s=patch.shape, sigma=1)
    score = np.sum(gk * scores)
    print(score)
    return score
    # print(scores.shape)
    # plt.subplot(131)
    # plt.imshow(scores)
    # plt.subplot(132)
    # plt.imshow(patch)
    # plt.subplot(133)
    # plt.imshow(kernel)
    # plt.show()

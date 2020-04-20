import numpy as np
import matplotlib.pyplot as plt
from collections import deque


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
        img_arr[img_arr > threshold] = 1
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
    print('grouping...')
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


def groups_to_bounding_boxes(groups, img_arr=None):

    bounding_boxes = []
    scores = []
    combined = []
    for group in groups:
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
        scores.append(score)
        print(most_top, most_left, most_bottom, most_right)
        bounding_box = [int(most_top), int(most_left), int(most_bottom), int(most_right)]
        bounding_boxes.append(bounding_box)
        combined.append([int(most_top), int(most_left), int(most_bottom), int(most_right), score])

    return bounding_boxes, scores, combined

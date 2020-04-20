import numpy as np
import postprocessing
import matplotlib.pyplot as plt


def color_match_red_lights(I, rgb_pixel, stride=1):

    if np.max(I) > 1:
        I = I/255

    if len(I.shape) != 3:
        I = np.expand_dims(I, axis=2)

    (n_rows, n_cols, n_channels) = np.shape(I)

    pixel_distance = np.zeros(shape=(n_rows, n_cols))
    for i in range(0, n_rows, stride):
        for j in range(0, n_cols, stride):
            pixel_distance[i, j] = 1 - np.mean(abs(rgb_pixel - I[i, j, :]))

    return pixel_distance


def get_hot_pixel_from_kernel(T, mode='average'):

    avg_kernel = np.mean(T, axis=2)
    if mode == 'average':
        thresh_T = postprocessing.threshold_convolved_image(avg_kernel, 0.5, 'down')
        avg_hot_pixel = np.mean(T[thresh_T > 0], axis=0)
        return avg_hot_pixel
    elif mode == 'max_intensity':
        x, y = np.where(avg_kernel == np.max(avg_kernel))
        hot_x = round(np.mean(x).item())
        hot_y = round(np.mean(y).item())
        return T[hot_x, hot_y]
    elif mode == 'max_red':
        x, y = np.where(T[:, :, 0] == np.max(T[:, :, 0]))
        hot_x = round(np.mean(x).item())
        hot_y = round(np.mean(y).item())
        return T[hot_x, hot_y]

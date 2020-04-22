import numpy as np
import os
import json
import utilities
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans


def bounding_box_to_kernel(img_arr, bb):
    tlx = int(bb[0])
    tly = int(bb[1])
    brx = int(bb[2])
    bry = int(bb[3])
    kernel = img_arr[tlx:brx, tly:bry, :]

    return kernel


def kernel_analysis(kernel):

    avg_pixel = np.mean(kernel, axis=(0, 1))
    shape = np.array(kernel.shape[:-1])

    return avg_pixel, shape


def cluster_kernels(kernel_info_list, shape_clusters):

    clustered_kernels = {}
    for entry in kernel_info_list:

        kernel, avg_pixel, kshape = entry
        shape_distance = np.linalg.norm(kshape - shape_clusters, axis=1)
        ind = int(np.argmin(shape_distance))
        target_shape = shape_clusters[ind]

        r_off = None
        c_off = None
        # print('in')
        # print(kernel.shape, target_shape)

        mod_val = np.mean(kernel, axis=(0, 1))
        if kshape[0] < target_shape[0]:
            r_off = int((target_shape[0] - kshape[0]) / 2)
            embed_kernel = np.zeros(shape=(target_shape[0], kernel.shape[1], 3)) + mod_val
            embed_kernel[r_off:r_off + kernel.shape[0], 0:kernel.shape[1], :] = kernel
            kernel = embed_kernel

        elif kshape[0] > target_shape[0]:
            r_off = int((kshape[0] - target_shape[0]) / 2)
            kernel = kernel[r_off:target_shape[0] + r_off, :, :]

        if kshape[1] < target_shape[1]:
            c_off = int((target_shape[1] - kshape[1]) / 2)
            embed_kernel = np.zeros(shape=(kernel.shape[0], target_shape[1], 3)) + mod_val
            embed_kernel[0:kernel.shape[0], c_off:c_off + kernel.shape[1], :] = kernel
            kernel = embed_kernel

        elif kshape[1] > target_shape[1]:
            c_off = int((kshape[1] - target_shape[1]) / 2)
            kernel = kernel[:, c_off:target_shape[1] + c_off, :]

        if clustered_kernels.get(str(target_shape), None) is None:
            clustered_kernels[str(target_shape)] = [kernel]
        else:
            clustered_kernels[str(target_shape)].append(kernel)

    return clustered_kernels


data_path = '../data/RedLights2011_Medium'
gts_path = '../data/hw02_annotations'

with open(os.path.join(gts_path, 'annotations_train.json'),'r') as f:
    gts_train = json.load(f)

# load splits:
split_path = '../data/hw02_splits'
file_names_train = np.load(os.path.join(split_path,'file_names_train.npy'))

avg_pixels = []
kshapes = []
kernel_info_list = []
for fname in file_names_train:
    bbs = gts_train[fname]
    img = utilities.load_image(data_path + '/' + fname)
    img_array = utilities.image_to_array(img)
    for bb in bbs:
        kernel = bounding_box_to_kernel(img_array, bb)
        avg_pixel, kshape = kernel_analysis(kernel)
        avg_pixels.append(avg_pixel)
        kshapes.append(kshape)
        kernel_info_list.append((kernel, avg_pixel, kshape))

avg_pixels = np.array(avg_pixels)
kshapes = np.array(kshapes)

km = KMeans(n_clusters=6)
km.fit(avg_pixels)

km2 = KMeans(n_clusters=6)
km2.fit(kshapes)

shape_clusters = np.array(km2.cluster_centers_, dtype=np.int64)
for i in range(shape_clusters.shape[0]):
    for j in range(shape_clusters.shape[1]):
        shape_clusters[i, j] = int(round(shape_clusters[i, j]))

clustered_kernels = cluster_kernels(kernel_info_list, shape_clusters)

avg_kernels = []
sub_cluster_size = 3
for cluster_key in clustered_kernels.keys():
    print(cluster_key)
    km3 = KMeans(n_clusters=sub_cluster_size)
    kernels = clustered_kernels[cluster_key]
    kshape = (kernels[0].shape[0], kernels[0].shape[1])
    km3.fit(np.array(kernels).reshape((len(kernels), -1)))
    sub_ck = km3.cluster_centers_.reshape(sub_cluster_size, kshape[0], kshape[1], 3)
    avg_kernels.extend(list(sub_ck))
    # avg_kernels.append(np.mean(clustered_kernels[cluster_key], axis=0))


save_path_kernels = '../data/kernels/'
save_path_pixels = '../data/red_light_pixels/'
utilities.create_nonexistent_folder(save_path_kernels)
utilities.create_nonexistent_folder(save_path_pixels)

for i, ak in enumerate(avg_kernels):

    np.save(save_path_kernels + 'kernel=' + str(i) + '.npy', ak)
    fig = plt.imshow(ak)
    plt.savefig(save_path_kernels + 'kernel=' + str(i))
np.save(save_path_pixels + 'rgb_pixel_array', km.cluster_centers_)

import preprocessing
import utilities
from PIL import Image
import os
import numpy as np
import ip_algorithms as ipa
import matplotlib.pyplot as plt

I = Image.open(os.path.join('/home/nkondapa/School/EE148/hw2/data/RedLights2011_tiny', 'RL-288.jpg'))
I = np.asarray(I)

num_kernels = 4
T_list = []
for i in range(num_kernels):
    T_list.append(utilities.load_kernel(str(i), '../data/kernels/'))

pixel_dist_maps_ah = []
pixel_dist_maps_mh = []
for i in range(num_kernels):
    #ah = preprocessing.get_hot_pixel_from_kernel(T_list[i], 'average')
    mih = preprocessing.get_hot_pixel_from_kernel(T_list[i], 'max_intensity')
    #mrh = preprocessing.get_hot_pixel_from_kernel(T_list[i], 'max_red')

    # print(ah, mih, mrh)

    #map_ah = preprocessing.color_match_red_lights(I, ah, stride=2)
    map_mih = preprocessing.color_match_red_lights(I, mih, stride=2)
    #smooth_mih = ipa.smooth_heatmap(map_mih, utilities.generate_gaussian_kernel(s=3))
    smooth_mih = ipa.neighbor_max_smooth_heatmap(map_mih, np.zeros(shape=(2, 2)))
    plt.imshow(smooth_mih)
    plt.show()
    #map_mrh = preprocessing.color_match_red_lights(I, mrh, stride=2)


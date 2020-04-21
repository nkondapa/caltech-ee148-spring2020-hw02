import numpy as np
import matplotlib.pyplot as plt
import visualize as viz
import utilities
import glob
import pickle as pkl


def search_for_red_light_pixels(img_array, pixel_path):

    events = []
    pixel_list = []
    def onclick(event):
        print('%s click: button=%d, x=%d, y=%d, xdata=%f, ydata=%f' %
              ('double' if event.dblclick else 'single', event.button,
               event.x, event.y, event.xdata, event.ydata))
        if event.button == 3:
            x = int(event.xdata)
            y = int(event.ydata)
            pixel = img_array[y, x, :]
            print(pixel)
            pixel_list.append(pixel)
        events.append(event)

    fig = plt.figure()
    plt.imshow(img_array)
    cid = fig.canvas.mpl_connect('button_press_event', onclick)
    viz.plt.show()

    im_arr = np.copy(np.asarray(img_array))
    fig.canvas.mpl_disconnect(cid)


def save_pixels(pixel_list, pixel_path):
    utilities.create_nonexistent_folder(pixel_path)
    with open(pixel_path, 'wb') as f:
        pkl.dump(pixel_list, f)

path = '../data/hw01_preds/preds.json'
pixel_path = '../data/red_pixels'
image_base_path = '../data/RedLights2011_tiny'

image_paths = sorted(glob.glob(image_base_path + '/*'))

for image_path in image_paths:
    img = utilities.load_image(image_path)
    img_arr = utilities.image_to_array(img)
    search_for_red_light_pixels(img_arr, pixel_path)

import os
import shutil
import numpy as np
from PIL import Image


def load_image(img_path):
    img = Image.open(img_path)
    return img


def image_to_array(img):
    return np.asarray(img, dtype=np.float32) / 255


def load_kernel(id, base_kernel_path):
    path = base_kernel_path + '/kernel=' + str(id) + '.npy'
    kernel = np.load(path)
    kernel = np.array(kernel, dtype=np.float32) / 255

    return kernel


def check_if_file_exists(path):
    return os.path.isfile(path)


def check_if_exists(path):
    return os.path.exists(path)


def create_nonexistent_folder(save_dir, clear_prev=False, verbose = False):
    try:
        os.makedirs(save_dir)
        return 1
    except FileExistsError:
        if clear_prev:
            import glob
            files = glob.glob(save_dir + '*')
            for f in files:
                os.remove(f)
            return 1
        else:
            if verbose:
                print('already created folder, skipping...')
            return 0


def file_modification_time(path):
    return os.path.getmtime(path)


def generate_gaussian_kernel(s=3, sigma=1.0):
    x, y = np.meshgrid(np.linspace(-1, 1, s), np.linspace(-1, 1, s))
    d = np.sqrt(x * x + y * y)
    mu = 0.0
    g = np.exp(-((d - mu) ** 2 / (2.0 * sigma ** 2)))
    return g


def delete_directory_recursively(path):

    # Delete all contents of a directory using shutil.rmtree() and  handle exceptions
    try:
        shutil.rmtree(path)
    except:
        print('Error while deleting directory')


def numpy_save(path, arr):
    np.save(path, arr)


def numpy_load(path):
    return np.load(path)


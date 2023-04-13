import imageio
import glob
import numpy as np
from PIL import Image, ImageOps
import matplotlib.pyplot as plt

def open_img_as_array(path_to_img: str) -> np.ndarray:
    return np.array(ImageOps.grayscale(Image.open(path_to_img))).astype(int)

def make_gif(path_to_imgs, path_to_save, gifname, fps=20):
    filenames = [name for name in glob.glob(f'{path_to_imgs}/*.png')]
    images = []
    for filename in filenames:
        images.append(imageio.imread(filename))
    imageio.mimsave(f'{path_to_save}/{gifname}.gif', images, format='gif', fps=fps)

def downscale_map(img, sq_size) -> np.ndarray:
    # img = (img - img.min()) / (img.max() - img.min())
    img = np.array(img)
    width, height = img.shape
    x_parts = width // sq_size
    y_parts = height // sq_size
    downsampled = np.zeros((x_parts, y_parts))
    for x in range(x_parts):
        for y in range(y_parts):
            downsampled[x, y] = (img[x*sq_size:(x+1)*sq_size, y*sq_size:(y+1)*sq_size] < img.max()).sum() == 0
    plt.imshow(downsampled, cmap='gray')
    return downsampled

def split_map(path, x_parts, scale_coef, color):
    img = open_img_as_array(path) if isinstance(path, str) else path
    width, height = img.shape
    sq_size = width // x_parts
    splitted = img.copy()
    delta = int(scale_coef * sq_size / 2)
    for x in range(sq_size // 2, width, sq_size):
        for y in range(sq_size // 2, height, sq_size): 
            splitted[x-delta:x+delta, y-delta:y+delta] = color
    return splitted

def prepare_gpr_results(img, x_parts=5, scale_coef=0.85, color=0, need_plot=False):
    # img = samples[i]
    # normalize to [-1, 1]
    img = (img - img.min()) / (img.max() - img.min()) * 2 - 1
    # print(img.min(), img.max())
    # # img = (img - img.min()) / (img.max() - img.min()) * 255
    sign = img > 0
    # values of sign are True or False
    # for every True value if at least one of its neighbours is False then it is a border
    # find all borders and make a new array with the same shape as img where borders are True and others are False
    original = np.zeros_like(img)
    for x in range(1, img.shape[0]-1):
        for y in range(1, img.shape[1]-1):
            original[x, y] = sign[x, y] and (not sign[x-1, y] or not sign[x+1, y] or not sign[x, y-1] or not sign[x, y+1])
    splitted = 1 - split_map(path=original, x_parts=x_parts, scale_coef=scale_coef, color=color)
    if need_plot:
        plt.figure(figsize=(20, 5))
        # original image
        plt.subplot(1, 3, 1)
        plt.imshow(1 - original, cmap='gray')
        plt.title('Original')
        # sign distribution
        plt.subplot(1, 3, 2)
        plt.imshow(sign, cmap='gray')
        plt.title('Sign distribution')
        # split map
        plt.subplot(1, 3, 3)
        plt.imshow(splitted, cmap='gray')
        plt.title('Splitted')

    return 1 - original, sign, splitted
import imageio
import glob
import numpy as np
from PIL import Image, ImageOps
import matplotlib.pyplot as plt

def open_img_as_array(path_to_img: str) -> np.array:
    return np.array(ImageOps.grayscale(Image.open(path_to_img))).astype(int)

def make_gif(path_to_imgs, path_to_save, gifname, fps=20):
    filenames = [name for name in glob.glob(f'{path_to_imgs}/*.png')]
    images = []
    for filename in filenames:
        images.append(imageio.imread(filename))
    imageio.mimsave(f'{path_to_save}/{gifname}.gif', images, format='gif', fps=fps)

def downscale_map(img_array, sq_size):
    width, height = img_array.shape
    tmp_arr = img_array.copy()
    if width % sq_size:
        tmp_arr = np.r_[img_array, np.zeros((sq_size - width % sq_size, height))]
        width, height = tmp_arr.shape
    if height % sq_size:
        tmp_arr = np.c_[tmp_arr, np.zeros((sq_size - height % sq_size, width)).T]
        width, height = tmp_arr.shape
    result = np.array([np.hsplit(u, height / sq_size) for u in np.vsplit(tmp_arr, width / sq_size)]).reshape((-1, sq_size, sq_size))
    final_result = np.array([x.sum() for x in result]).reshape((width // sq_size, height // sq_size)) == 0
    return final_result

def split_map(path, x_parts, scale_coef, color, need_plot=False):
    img = np.array(ImageOps.grayscale(Image.open(path))).astype(int)
    width, height = img.shape
    
    if need_plot:
        plt.figure(figsize=(20, 10))
        plt.subplot(1, 2, 1)
        plt.imshow(img, cmap='gray')

    y_parts = int(x_parts * height / width)
    dx = width // x_parts
    dy = height // y_parts
    delta = int(scale_coef * min(dx, dy) // 2)

    for x in range(dx // 2, width, dx):
        for y in range((1 + (x - dx // 2) // dx % 2) * dy // 2, height, dy):
        # for y in range(dy // 2, height, dy):
            img[x-delta:x+delta, y-delta:y+delta] = color
    
    if need_plot:
        plt.subplot(1, 2, 2)
        plt.imshow(img, cmap='gray')
        print(f'width={width}\nheight={height}\nx_parts = {x_parts}\ny_parts = {y_parts}\ndelta = {delta}\ndx={dx}\ndy={dy}')
    return img
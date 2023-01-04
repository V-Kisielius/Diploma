import imageio
import glob
import numpy as np

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
import imageio
import glob

def make_gif(dir_path, gifname, fps=20):
    filenames = [_ for _ in glob.glob(f'{dir_path}/*.png')]
    images = []
    for filename in filenames:
        images.append(imageio.imread(filename))
    imageio.mimsave(f'{dir_path}/{gifname}.gif', images, format='gif', fps=fps)
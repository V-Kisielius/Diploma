import torch
import numpy as np
import matplotlib.pyplot as plt

from scripts.synmap import SynopticMap
from scripts.helper import open_img_as_array, plot_3d_tensor

class ImageData:
    def __init__(self, path_or_img, data_mode, radius=1, reduce_factor=1):
        modes = {
            'img': lambda img: img,
            'abz': lambda path: SynopticMap(path).filaments,
            'path': lambda path: open_img_as_array(path)
        }
        if data_mode not in modes:
            raise ValueError('Data mode must be (\'img\' | \'abz\' | \'path\')')
        self.img_array = modes[data_mode](path_or_img)
        self.width, self.height = self.img_array.shape
        self.total_pixel = self.width * self.height
        self.reduce_factor = reduce_factor
        self.radius = radius
        mask = self.img_array.reshape(-1, 1) == self.img_array.max()
        self.mask = np.where(mask)[0]
        self.bound_mask = np.where(~mask)[0]
        self.bound_length = len(self.bound_mask)
        self.data_3d = self.make_data()

    def make_data(self):
        roots = 2 * torch.pi * torch.arange(self.height) / self.height
        zs = torch.linspace(1, -1, self.width)
        z, y = torch.stack(torch.meshgrid(zs, roots, indexing='ij'), dim=2).view(-1, 2).T
        z = z * torch.pi * self.width / self.height
        x = torch.cos(y)
        y = torch.sin(y)
        return self.radius * torch.stack((x, y, z), dim=1)

    def get_info(self):
        print(f'width: {self.width}\n'
            f'height: {self.height}\n'
            f'total_pixel: {self.total_pixel}\n'
            f'bound length: {self.bound_length}\n'
            f'percent of bound pixels: {100 * self.bound_length / self.total_pixel:.1f}%')

    def show_image(self, figsize=(10, 5), cmap='gray'):
        plt.figure(figsize=figsize)
        plt.title(f'Data visualization')
        plt.imshow(self.img_array, cmap=cmap)

    def show_3d(self, marker_size=2, colorscale='oxy'):
        plot_3d_tensor(tensor=self.data_3d,
                       color=self.img_array.flatten(),
                       colorscale=colorscale,
                       marker_size=marker_size)
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly_express as px

from synmap import SynopticMap
from helper import open_img_as_array

class MyData(torch.utils.data.Dataset):
    def __init__(self, path_or_img, data_mode, mode_3d, radius, reduce_fctor=1, need_info=False):
        match data_mode:
            case 'img':
                self.map_number = '1'
                self.img_array = path_or_img
            case 'abz':
                self.smap = SynopticMap(path_or_img)
                self.img_array = self.smap.filaments
                self.map_number = self.smap.map_number
            case 'path':
                self.map_number = '1'
                self.img_array = open_img_as_array(path_or_img)
            case _:
                raise ValueError('data_mode must be (\'img\' | \'abz\' | \'path\')')

        self.width, self.height = self.img_array.shape
        self.total_pixel = self.width * self.height
        self.reduce_factor = reduce_fctor

        self.mode_3d = mode_3d
        self.radius = radius

        self.mask = np.where(self.img_array.reshape(-1, 1) == self.img_array.max())[0]
        self.bound_mask = np.where(self.img_array.reshape(-1, 1) < self.img_array.max())[0]
        self.bound_length = len(np.where(self.img_array.reshape(-1, 1) < self.img_array.max())[0])
        self.row_numbers = 5 * self.height
        # self.sign_mask = np.where((self._add_grad_label() * self.img_array).reshape(-1, 1) != 0)[0]

        self.data_2d, self.data_3d = self.make_data(mode_3d=self.mode_3d)
        self.data_3d.x, self.data_3d.y, self.data_3d.z = self.data_3d.T
        
        if need_info:
            print(f'width: {self.width}\n' +
                  f'height: {self.height}\n' +
                  f'total_pixel: {self.total_pixel}\n' +
                  f'bound length: {self.bound_length}\n' +
                  f'percent of bound pixels: {(100 * self.bound_length / self.total_pixel):.1f}%')

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data_2d[idx], self.data_3d[idx]
        
    def _add_grad_label(self):
        tmp = self.img_array / self.img_array.max()

        left_shift, right_shift, up_shift, down_shift = np.ones((4, self.width, self.height))

        result = np.zeros(self.img_array.shape)

        left_shift[:, :-1] = tmp[:, 1:]
        right_shift[:, 1:] = tmp[:, :-1]
        up_shift[1:, :] = tmp[:-1, :]
        down_shift[:-1, :] = tmp[1:, :]

        result = left_shift + right_shift + down_shift + up_shift - 4 * tmp
        result = np.zeros(self.img_array.shape) != result
        return result.astype(int)
        
    def show_image(self):
        plt.figure(figsize=(10, 5))
        plt.title(f'Input Image №{self.map_number}')
        plt.imshow(self.img_array, cmap='gray')
    
    def show_3d_static(self):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(self.data_3d.x, self.data_3d.y, self.data_3d.z, c=self.img_array.flatten())

    def show_3d(self):
        df = pd.DataFrame({'x': self.data_3d.x[::self.reduce_factor], 
                            'y': self.data_3d.y[::self.reduce_factor], 
                            'z': -self.data_3d.z[::self.reduce_factor], 
                            'label': self.img_array.flatten()[::self.reduce_factor]})
        return px.scatter_3d(df, x='x', y='y', z='z', color='label').update_traces(marker={'size': 2})

    def make_data(self, mode_3d):
        data_2d = torch.from_numpy(np.stack(np.indices((self.width, self.height)), axis=2).reshape(-1, 2)).float()
        x, y = data_2d.T
        x -= x.mean()
        x /= x.abs().max()
        y -= y.mean()
        y /= y.abs().max()
        data_2d = torch.stack((x, y), dim=1)
        match mode_3d:
            case 'sphere':
                z = torch.pi * x
                x = torch.cos(torch.pi * y) * (1 - z.pow(2)).pow(0.5)
                y = torch.sin(torch.pi * y) * (1 - z.pow(2)).pow(0.5)
                data_3d = self.radius * torch.stack((x, y, z), dim=1)
            case 'cylinder':
                z = torch.pi * x * self.width / self.height
                x = torch.cos(torch.pi * y)
                y = torch.sin(torch.pi * y)
                data_3d = self.radius * torch.stack((x, y, z), dim=1)
            case _:
                print('Incorrect mode\n')
                raise SystemError(f'Incorrect mode: {mode_3d}\n')
        return data_2d, data_3d
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import os
#import time

from PIL import Image, ImageOps
from IPython.display import clear_output
import matplotlib.pyplot as plt

import plotly.express as px

class MyPix(torch.utils.data.Dataset):
    def __init__(self, path_to_file, mode):
        self.img_array = self.get_img_array(path_to_file, mode)

        self.width, self.height = self.img_array.shape

        self.total_pixel = self.width * self.height

        self.bound_length = len(np.where(self.img_array < 255)[0])

        print(f'width = {self.width}\n' +
              f'height = {self.height}\n' +
              f'total_pixel = {self.total_pixel}\n' +
              f'bound length: {self.bound_length}\n' +
              f'percent of bound pixels: {(100 * self.bound_length / self.total_pixel):.1f}%')

        
        self.data_2d, self.data_3d, self.labels = self.make_data()
        self.data_3d.x, self.data_3d.y, self.data_3d.z = self.data_3d[:, 0], self.data_3d[:, 1], self.data_3d[:, 2]
        
        

        self.grad_labels = torch.from_numpy(
            self._add_grad_label()).float().view(-1, 1)
        #self.grad_labels2 = self.add_grad_label()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data_2d[idx], self.data_3d[idx], self.labels[idx], self.grad_labels[idx]

    def get_img_array(self, path_to_file, mode):
        return np.array(ImageOps.grayscale(Image.open(path_to_file))).astype(int) if mode == 'img' else self.read_from_abz(path_to_file)
    
    def make_data(self):
        batch = np.stack(np.indices((self.width, self.height)), axis=2).reshape(-1, 2)
        labels = np.zeros(len(batch))
        for idx, el in enumerate(batch):
            labels[idx] = 1 if self.img_array[el[0], el[1]] == 255 else 0  # 1 - bound

        batch = torch.from_numpy(batch).float()
        batch[:, 0] -= batch[:, 0].mean()
        batch[:, 0] /= batch[:, 0].max()
        batch[:, 1] -= batch[:, 1].mean()
        batch[:, 1] /= batch[:, 1].max()

        new_batch = torch.zeros((len(batch), 3)).float()
        for i, _ in enumerate(batch):
            x, y = _
            new_batch[i, 0] = torch.cos(torch.pi * y)
            new_batch[i, 1] = torch.sin(torch.pi * y)
            new_batch[i, 2] = x
        
        

        return batch, new_batch, torch.from_numpy(labels).float()

    def _add_grad_label(self):
        tmp = self.labels.numpy().reshape(self.img_array.shape)

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
        plt.title('Input Image')
        plt.imshow(self.img_array, cmap='gray', vmin=0, vmax=255)

    def show_grad_labels(self):
        plt.figure(figsize=(10, 5))
        plt.title('Grad-points -- white')
        plt.imshow(self.grad_labels.view(self.img_array.shape),
                   cmap='gray', vmin=0, vmax=1)
    
    def read_from_abz(self, filename):
        with open(filename) as f:
            lines = [line.rstrip() for line in f]
        my_shape = [int(x) for x in lines[0].split()[-6:-4]]
        my_shape.reverse()
        img_arr = np.full(shape=my_shape, fill_value=255)
        for line in lines[3:]:
            cur_line = [int(x) for x in line.split()]
            if len(cur_line) == 3:
                y = cur_line[0]
                x1 = cur_line[1]
                x2 = -cur_line[2]
                for i in range(x1, x2 + 1):
                    img_arr[min(y, my_shape[0]-1), min(i, my_shape[1]-1)] = 0
            elif len(cur_line) == 5:
                y = cur_line[0]
                x1 = cur_line[1]
                x2 = -cur_line[2]
                x3 = cur_line[3]
                x4 = -cur_line[4]
                for i in range(x1, x2 + 1):
                    img_arr[min(y, my_shape[0]-1), min(i, my_shape[1]-1)] = 0
                for i in range(x3, x4 + 1):
                    img_arr[min(y, my_shape[0]-1), min(i, my_shape[1]-1)] = 0
        
        return img_arr
        # foo = Image.fromarray(img_arr).resize((my_shape[0] // 4, my_shape[1] // 4), Image.Resampling.BICUBIC)
        # return np.array(foo.getdata()).reshape((my_shape[0] // 4, my_shape[1] // 4))

    def show_3d(self):
        df = pd.DataFrame({'x': self.data_3d.x, 'y': self.data_3d.y, 'z': self.data_3d.z, 'label': self.labels})
        return px.scatter_3d(df, x='x', y='y', z='z', color='label').update_traces(marker={'size': 2})
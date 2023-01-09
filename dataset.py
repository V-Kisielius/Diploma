import torch
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import plotly_express as px
from PIL import Image, ImageOps
import glob

from config import PATH_TO_PM_DATA, PATH_TO_MARKUP_DATA

class MyData(torch.utils.data.Dataset):
    def __init__(self, path_to_file, mode, mode_3d, radius, reduce_fctor=1, need_help=False, need_info=False):
        self.need_help = need_help
        self.mode_3d = mode_3d
        self.radius = radius
        self.maps_markup = self.make_markup()
        self.pm_list = self.get_pm_list()
        self.reduce_factor = reduce_fctor
        if mode == 'img':
            self.map_number = ''
            self.img_array = self.get_img_array(path_to_file) 
        elif mode == 'abz':
            self.map_number = path_to_file[-11:-7]
            self.img_array, self.help_array = self.read_from_abz(path_to_file)
        else:
            self.map_number = ''
            self.img_array = path_to_file
        
        self.width, self.height = self.img_array.shape
        self.total_pixel = self.width * self.height

        self.bound_mask = np.where(self.img_array.reshape(-1, 1) < 255)[0]
        self.bound_length = len(np.where(self.img_array.reshape(-1, 1) < 255)[0])
        self.mask = np.where(self.img_array.reshape(-1, 1) == 255)[0]
        self.row_numbers = 1 * self.height
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

    def get_img_array(self, path_to_file):
        return np.array(ImageOps.grayscale(Image.open(path_to_file))).astype(int)
        
    def get_pm_list(self):
        filenames = [filename for filename in glob.glob(f'{PATH_TO_PM_DATA}/*.dat')]
        pm_list = []
        for filename in filenames:
            with open(filename) as f:
                lines = [line[4:].replace('-1', '0').replace('+1', '1').rstrip() for line in f]
            pm_list.append(np.array([list(x) for x in lines], dtype = 'int'))
        return pm_list

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

    def show_pm_list(self):
        plt.figure(figsize=(20, 30))
        maps_number = len(self.pm_list)
        for i in range(maps_number):
            plt.subplot(maps_number /  5, 5, i + 1) if maps_number % 5 == 0 else plt.subplot(maps_number // 5 + 1, 5, i + 1)
            plt.title(f'Map №{1904 + i}')
            plt.gca().axes.get_xaxis().set_visible(False)
            plt.gca().axes.get_yaxis().set_visible(False)
            plt.imshow(self.pm_list[i], cmap='gray')
        plt.show()
            
    def make_markup(self):
        res = {}
        current = 0
        with open(PATH_TO_MARKUP_DATA, 'r', encoding='windows-1251') as file:
            for line in file:
                try:
                    temp = list(map(int, line.rstrip().split()))
                except:
                    continue
                if len(temp) == 1:
                    current = temp[0]
                elif len(temp) == 0:
                    continue
                else:
                    if current not in res.keys():
                        res[current] = [temp[1:]] 
                    else:
                        res[current].append(temp[1:])
        return res
        
    def show_image(self):
        plt.figure(figsize=(10, 5))
        plt.title(f'Input Image №{self.map_number}')
        plt.imshow(self.img_array, cmap='gray', vmin=0, vmax=255)
    
    def read_from_abz(self, filename):
        with open(filename) as f:
            lines = [line.rstrip() for line in f]
        my_shape = [int(x) for x in lines[0].split()[-6:-4]]
        my_shape.reverse()
        width, height = my_shape[0], my_shape[1]
        img_array = np.full(shape=my_shape, fill_value=255)
        for line in lines[3:]:
            line = [int(x) for x in line.split()]
            if len(line) == 3:
                y = line[0]
                x1 = line[1]
                x2 = -line[2]
                for i in range(x1, x2 + 1):
                    img_array[min(y, width - 1), min(i, height - 1)] = 0
            elif len(line) == 5:
                y = line[0]
                x1 = line[1]
                x2 = -line[2]
                x3 = line[3]
                x4 = -line[4]
                for i in range(x1, x2 + 1):
                    img_array[min(y, width - 1), min(i, height - 1)] = 0
                for i in range(x3, x4 + 1):
                    img_array[min(y, width - 1), min(i, height - 1)] = 0
        if self.need_help:
            map_number = int(filename[-11:-7])
            i = 10
            help_array = np.full(shape=my_shape, fill_value=255)
            for line in self.maps_markup[map_number]:    
                for el in line:
                    x, y = int(el * (width - 1) / 180.), int(i * (height - 1) / 360.)
                    
                    lx, rx = max(0, x - 3), min(width - 1, x + 3)
                    ly, ry = max(0, y - 3), min(height - 1, y + 3)

                    help_array[lx:rx, ly:ry] = 0
                i += 10
        return img_array, help_array if self.need_help else img_array

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
                z = x
                x = torch.cos(torch.pi * y) * (1 - z.pow(2)).pow(0.5)
                y = torch.sin(torch.pi * y) * (1 - z.pow(2)).pow(0.5)
                data_3d = self.radius * torch.stack((x, y, z), dim=1)
            case 'cylinder':
                z = x
                x = torch.cos(torch.pi * y)
                y = torch.sin(torch.pi * y)
                data_3d = self.radius * torch.stack((x, y, z), dim=1)
            case _:
                print('Incorrect mode\n')
                raise SystemError(f'Incorrect mode: {mode_3d}\n')
        return data_2d, data_3d
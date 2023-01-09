import numpy as np
import matplotlib.pyplot as plt

from config import PATH_TO_MARKUP_DATA, PATH_TO_PM_DATA

class SynopticMap():
    def __init__(self, path_to_abz):
        self.filaments = self.read_abz(path_to_abz)
        self.width, self.height = self.filaments.shape
        self.map_number = int(path_to_abz[-11:-7])
        self.markup = self.get_markup()
        self.pm = self.get_pm()
    
    def read_abz(self, path_to_abz):
        with open(path_to_abz) as f:
            lines = [line.rstrip() for line in f]
        my_shape = [int(x) for x in lines[0].split()[-6:-4]]
        my_shape.reverse()
        width, height = my_shape
        img_array = np.full(shape=my_shape, fill_value=255)
        for line in lines[3:]:
            line = [int(x) for x in line.split()]
            if len(line) == 3:
                y = line[0]
                x1 = line[1]
                x2 = -line[2]
                img_array[min(y, width - 1), min(x1, height - 1):min(x2 + 1, height - 1)] = 0
            elif len(line) == 5:
                y = line[0]
                x1 = line[1]
                x2 = -line[2]
                x3 = line[3]
                x4 = -line[4]
                img_array[min(y, width - 1), min(x1, height - 1):min(x2 + 1, height - 1)] = 0
                img_array[min(y, width - 1), min(x3, height - 1):min(x4 + 1, height - 1)] = 0
        return img_array

    def show_filaments(self):
        plt.figure(figsize=(10, 5))
        plt.title(f'Filaments on map {self.map_number}')
        plt.imshow(self.filaments, cmap='gray', vmin=0, vmax=255)

    def get_markup(self):
        res = []
        with open(PATH_TO_MARKUP_DATA, 'r', encoding='windows-1251') as file:
            flag = False
            for line in file:
                try:
                    tmp = list(map(int, line.rstrip().split()))
                    if len(tmp) <= 1:
                        flag = False
                except:
                    continue
                if flag:
                    res.append(tmp[1:])
                if len(tmp) == 1 and tmp[0] == 1904:
                    flag = True
        i = 10
        help_array = np.full(shape=self.filaments.shape, fill_value=255)
        for line in res:    
            for el in line:
                x, y = int(el * (self.width - 1) / 180.), int(i * (self.height - 1) / 360.)
                delta =5
                lx, rx = max(0, x - delta), min(self.width - 1, x + delta)
                ly, ry = max(0, y - delta), min(self.height - 1, y + delta)

                help_array[lx:rx, ly:ry] = 0
            i += 10
        return help_array

    def show_markup(self):
        plt.figure(figsize=(10, 5))
        plt.title(f'Markup on map {self.map_number}')
        plt.imshow(self.markup, cmap='gray', vmin=0, vmax=255)

    def get_pm(self):
        with open(f'{PATH_TO_PM_DATA}/c{self.map_number}.dat') as f:
            lines = [line[4:].replace('-1', '0').replace('+1', '1').rstrip() for line in f]
        return np.array([list(x) for x in lines], dtype = 'int')

    def show_pm(self):
        plt.figure(figsize=(10, 5))
        plt.title(f'PM on map {self.map_number}')
        plt.imshow(self.pm, cmap='gray', vmin=0, vmax=1)
    
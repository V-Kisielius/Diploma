import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import plotly_express as px
from itertools import islice
import math
import os
from tqdm import tqdm
import matplotlib.gridspec as gridspec
from IPython.display import clear_output

from config import device, PATH_TO_EPOCH_OUTS

class Net(nn.Module):
    __slots__ = 'dataset_list'
    def __init__(self, dataset_list, lr, weight_decay=1e-3):
        super(Net, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(3, 6), nn.ELU(),
            nn.Linear(6, 12), nn.ELU(),
            nn.Linear(12, 24), nn.ELU(),
            nn.Linear(24, 1), nn.Tanh())
        # self.weight = torch.nn.Parameter(torch.FloatTensor([1]), requires_grad=True)
        self.dataset_list = dataset_list
        self.data_list = [data.data_3d for data in self.dataset_list]
        self.lr = lr
        self.weight_decay = weight_decay
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        #scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', verbose=True, patience=100)
        self.loss_dict = {'loss': [],
                          'f_abs_integral': [],
                          'bound_integral': [],
                          'orientation_integral': [],
                          'f_integral': []}
                        #   'sign_integral': [],
                        #   'sign_integral_abs': []}
        
    def forward(self, x):
        output = self.net(x)
        return output

    def pretrain(self, img_size, num_epochs):
        width, height = img_size
        img = torch.zeros(img_size, device=device)
        img[:img_size[0] // 10, :] = 1
        img[-img_size[0] // 10:, :] = -1

        data_2d = torch.from_numpy(np.stack(np.indices(img_size), axis=2).reshape(-1, 2)).float()
        x, y = data_2d.T
        x -= x.mean()
        x /= x.abs().max()
        y -= y.mean()
        y /= y.abs().max()
        data_2d = torch.stack((x, y), dim=1)

        z = torch.pi * x * width / height
        x = torch.cos(torch.pi * y)
        y = torch.sin(torch.pi * y)
        batch_list = [torch.stack((x, y, z), dim=1).to(device)]
        self.train()
        for _ in tqdm(range(1, int(num_epochs) + 1), desc='Pretraining'):
            output_list = [self(batch) for batch in batch_list]
            self.optimizer.zero_grad()
            loss = sum([torch.nn.functional.mse_loss(output, img.view(-1, 1)) for output in output_list])
            loss.backward(retain_graph=True)
            self.optimizer.step()
        mse = sum([torch.nn.functional.mse_loss(output, img.view(-1, 1)) for output in output_list])
        print(f'Pretraining MSE: {mse.item()}')
        plt.figure(figsize=(12, 6))
        plt.title('Pretrained model output')
        plt.imshow(output_list[0].view(img_size).detach().cpu(), cmap='PuOr')

    def compute_and_plot_gradient(self, input_list=None):
        batch_list = input_list if input_list is not None else self.data_list
        for batch in batch_list:
            batch.requires_grad_()
        tmp_list = [self(batch.to(device)) for batch in batch_list]
        for tmp in tmp_list:
            tmp.sum().backward()
        grad_out_list = [batch.grad.cpu() for batch in batch_list]
        grad_map_list = [((grad_out.pow(2).sum(dim=1)).pow(0.5)).view(dataset.img_array.shape) for grad_out, dataset in zip(grad_out_list, self.dataset_list)]
        plt.figure(figsize=(12, 6))
        for i, grad_map in enumerate(grad_map_list):
            plt.subplot(1, len(grad_map_list), i + 1)
            plt.title(r'Gradient map for $||\nabla f(x,y,z)||_2$' + f'\non input №{i + 1}', fontsize=15)
            plt.imshow(grad_map, cmap='plasma')
            plt.colorbar(location='bottom')
        return grad_out_list, grad_map_list

    def change_lr(self, lr):
        for g in self.optimizer.param_groups:
            g['lr'] = lr

    def show_loss_items(self):
        plt.figure(figsize=(16, 6))
        for i, (key, value) in islice(enumerate(self.loss_dict.items()), 1, len(self.loss_dict.keys())):
            plt.subplot(2, math.ceil((len(self.loss_dict.keys()) - 1) / 2), i)
            plt.xlabel('Epoch')
            plt.ylabel(key)
            plt.plot(value)  
        plt.show()

    def restart_model(self, lr, weight_decay=1e-3):
        model = Net(dataset_list=self.dataset_list, lr=lr, weight_decay=weight_decay)
        model.to(device)
        return model

    def save_state_dict(self, path):
        os.makedirs('/'.join(path.split('/')[:-1]), exist_ok=True)
        torch.save(self.state_dict(), path)

    def show_3d(self, prediction_list, map_number):
        x, y, z = self.data_list[map_number].T
        df = pd.DataFrame({'x': x.cpu().detach().numpy(),
                            'y': y.cpu().detach().numpy(),
                            'z': -z.cpu().detach().numpy(),
                            'label': prediction_list[map_number].flatten()})
        return px.scatter_3d(df, x='x', y='y', z='z', color='label').update_traces(marker={'size': 2})

    def test_model(self, input_list=None, need_plot=False):
        input_list = input_list if input_list is not None else self.data_list
        with torch.no_grad():
            output_list = [self(input.to(device)).cpu().detach() for input in input_list]
        if need_plot:
            plt.figure(figsize=(10, 20))
            for (i, output), data in zip(enumerate(output_list), self.dataset_list):
                original = torch.from_numpy(data.img_array).clone()
                prediction = output.view(data.img_array.shape).clone()

                mask = original < 255
                original[mask] = 0
                original[~mask] = 1

                plt.subplot(len(output_list) + 2, 1, i + 1)
                plt.title('Original image')
                plt.imshow(original, cmap='gray')

                plt.subplot(len(output_list) + 2, 1, i + 2)
                plt.title('Prediction')
                plt.imshow(prediction, cmap='PuOr')

        return output_list # if input_list else output_list, mse, f1, f2

    def test_model_(self):
        with torch.no_grad():
            original = torch.from_numpy(self.dataset_list[0].img_array).clone()
            x = self.data_list[0].to(device)
            prediction = self(x).cpu().detach().view(original.shape).clone()
            prediction = (prediction > 0).float()
            mse = torch.nn.functional.mse_loss(prediction, original)
        return mse
    
    def start_training(self, num_epochs, my_weight=1e-1, show_frequency=1e+2, need_plot=False, need_save=True):
        if need_save:
            os.makedirs(PATH_TO_EPOCH_OUTS, exist_ok=True)

        for value in self.loss_dict.values():
            value.clear()

        batch_list = [data.to(device) for data in self.data_list]

        for epoch in range(1, int(num_epochs) + 1):
            output_list = [self(batch) for batch in batch_list]

            self.optimizer.zero_grad()
            loss = self.compute_loss(output_list=output_list, my_weight=my_weight) 
            loss.backward(retain_graph=True)
            self.optimizer.step()
            
            if need_plot and (epoch % int(show_frequency) == 0 or epoch == num_epochs):
                output_list = [output.cpu().detach().view(dataset.img_array.shape) for output, dataset in zip(output_list, self.dataset_list)]
                clear_output(wait=True)
                gs = gridspec.GridSpec(2, len(self.dataset_list) + 1)
                fig = plt.figure(figsize=(16, 6))
                
                ax1 = fig.add_subplot(gs[:, 0])
                ax1.set_xlabel('Epoch')
                ax1.set_ylabel('Loss')
                ax1.plot(self.loss_dict['loss'])

                for (i, img), data in zip(enumerate(output_list), self.dataset_list):
                    ax2 = fig.add_subplot(gs[0, i + 1])
                    ax2.title.set_text(f'Prediction №{i + 1}')
                    ax2.get_xaxis().set_ticks([])
                    ax2.get_yaxis().set_ticks([])
                    ax2.imshow(img, cmap='PuOr', vmin=-1, vmax=1)

                    ax3 = fig.add_subplot(gs[1, i + 1])
                    ax3.get_xaxis().set_ticks([])
                    ax3.get_yaxis().set_ticks([])
                    ax3.title.set_text(f'Input image №{i + 1}')
                    ax3.imshow(data.img_array, cmap='gray')
    
                if need_save:
                    plt.savefig(PATH_TO_EPOCH_OUTS + '/epoch%06d.png' % epoch)

                plt.show()
                self.show_loss_items()

    def compute_loss(self, output_list, my_weight): 
        loss, f_abs_integral, bound_integral, orientation_integral, f_integral = torch.zeros(len(self.loss_dict.items())) #, sign_integral, sign_integral_abs = torch.zeros(7)
        loss_, f_abs_integral_, bound_integral_, orientation_integral_, f_integral_ = torch.zeros(len(self.loss_dict.items())) #, sign_integral_, sign_integral_abs_ = torch.zeros(7)

        for (output, dataset) in zip(output_list, self.dataset_list):
            output = output.flatten()
            upper_bound = (output[:dataset.row_numbers].sum() / len(output[:dataset.row_numbers]) - 1).abs()
            lower_bound = (output[-dataset.row_numbers:].sum() / len(output[-dataset.row_numbers:]) + 1).abs()

            for key, _ in self.loss_dict.items():
                locals()[key + "_"] = locals()[key].clone()

            # bound_length_integral  = bound_length_integral_ + len(torch.masked_select(output, ~output.abs().ge(0.5))) / len(output)
            f_integral = f_integral_ + my_weight * output.sum().abs() / dataset.total_pixel
            # sign_integral = sign_integral_ + output[dataset.sign_mask].sum().abs() / len(dataset.sign_mask)
            # sign_integral_abs = sign_integral_abs_ + 1 - output[dataset.sign_mask].abs().sum() / len(dataset.sign_mask)
            orientation_integral = orientation_integral_ + 0.25 * (upper_bound + lower_bound)
            f_abs_integral = f_abs_integral_ + 1 - output[dataset.mask].abs().sum() / (dataset.total_pixel - dataset.bound_length) # хочу НЕ границу +-1
            bound_integral = bound_integral_ + output[dataset.bound_mask].pow(2).sum() / dataset.bound_length # хочу на границе 0
            loss = loss_ + f_abs_integral + bound_integral + orientation_integral + f_integral# + 0.1 * (sign_integral + sign_integral_abs)

        for key, _ in self.loss_dict.items():
            self.loss_dict[key].append(locals()[key].item())
            
        return loss
        # w_bound = torch.sigmoid(self.weight)
        # w_abs = 1 - w_bound
        # return (w_abs + 1) * f_abs_integral + (w_bound + 1) * bound_integral + orientation_integral, w_bound
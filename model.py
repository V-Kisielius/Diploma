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

class Net(nn.Module):
    def __init__(self, dataset, lr, weight_decay):
        super(Net, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(3, 4), nn.ELU(),
            nn.Linear(4, 8), nn.ELU(),
            nn.Linear(8, 16), nn.ELU(),
            # nn.Linear(16, 32), nn.ELU(),
            # nn.Linear(32, 64), nn.ELU(),
            # nn.Linear(64, 128), nn.ELU(),
            # nn.Linear(128, 64), nn.ELU(),
            # nn.Linear(64, 32), nn.ELU(),
            # nn.Linear(32, 16), nn.ELU(),
            nn.Linear(16, 8), nn.ELU(),
            nn.Linear(8, 1), nn.Tanh())
        self.dataset = dataset
        self.lr = lr
        self.weight_decay = weight_decay
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        #scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', verbose=True, patience=100)
        
    def forward(self, x):
        #x = x.cuda()
        output = self.net(x)
        return output

    def compute_loss(self, output, input, epoch):#, need_grad):
        mask = np.where(self.dataset.labels == 1)[0]
        bound_mask = np.where(self.dataset.labels == 0)[0]
        # if need_grad:
        #     grad_mask = np.where(self.dataset.grad_labels == 0)[0]

        #up_integral = 1 - output[:self.dataset.height].sum() / self.dataset.height  
        exterior_integral = 1 - (output[mask].abs()).sum() / (self.dataset.total_pixel - self.dataset.bound_length) # хочу НЕ границу +-1
        #d_bound = (data.grad[d_bound_mask, :].abs() / data.grad[d_bound_mask, :].max()).sum() / len(data.grad[d_bound_mask, :]) if epoch else 0
        #d_bound = ((data.grad[d_bound_mask, :] / data.grad[d_bound_mask, :].max()) / len(data.grad[d_bound_mask, :])).sum() if epoch else 0
        #if epoch > 1000:
        # boundgrad = data.grad[np.where(img_arr.reshape(-1, 1) < 255), :]
        # boundgrad = 1. / boundgrad.view(-1, 1).abs().sum() 
        # outgrad = data.grad[np.where(img_arr.reshape(-1, 1) == 255), :]
        # outgrad = outgrad.view(-1, 1).abs().sum() / (len(outgrad.view(-1, 1)))
        # else:
        #     mygrad = 0
        second_integral = output.sum().abs() / self.dataset.total_pixel #if epoch > 5000 else 0
        bound_intgegral = (output[bound_mask].abs()).sum() / self.dataset.bound_length # хочу на границе 0
        # grad_integral = input.grad[grad_mask].abs().sum() / (len(input.grad[grad_mask]) * input.grad[grad_mask].abs().max()) if need_grad else 0
        
        return bound_intgegral + exterior_integral + 1e-1*second_integral# + up_integral #+ grad_integral + 1e-1*second_integral# + boundgrad + outgrad #second_integral #+ mygrad 

    def go_train(self, num_epochs, show_frequency, need_grad):
        loss_trace = []

        batch = self.dataset.data_3d.cuda()
        perm = torch.randperm(self.dataset.total_pixel)
        inv_perm = torch.argsort(perm)
        batch = batch[perm]
        
        # if need_grad:
        #     batch.requires_grad_()

        dir_path = 'epoch_outs'
        if not os.path.exists(dir_path):
            os.mkdir(dir_path)

        for epoch in range(1, int(num_epochs) + 1):
            output = self(batch)
            # if need_grad:
            #     output.sum().backward(retain_graph=True)

            self.optimizer.zero_grad()

            loss = self.compute_loss(output=output[inv_perm], input=batch, epoch=epoch)#, need_grad=need_grad)
            loss_trace.append(loss.item())
            loss.backward(retain_graph=True)

            self.optimizer.step()
            # scheduler.step(loss)

            if epoch % int(show_frequency) == 0:
                clear_output(wait=True)
                plt.figure(figsize=(10, 5))
                plt.subplot(1, 2, 1)
                plt.xlabel('Epoch')
                plt.ylabel('Loss')
                #plt.ylim((0, 1))
                plt.plot(loss_trace)
                plt.subplot(1, 2, 2)
                plt.imshow(output[inv_perm].cpu().detach().numpy().reshape(self.dataset.img_array.shape), cmap='PuOr', vmin=-1, vmax=1)
                plt.title(f'Epoch {epoch}')
                plt.colorbar()
                plt.savefig(dir_path + '/epoch%06d.png' % epoch)
                plt.show()

        return output[inv_perm].cpu().detach().numpy().reshape(self.dataset.img_array.shape)

    def test_model(self, prediction):

        plt.figure(figsize=(12, 6))
        plt.title('Visualization of the function $f(x,y,z)$')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.imshow(prediction, cmap='PuOr', vmin=-1, vmax=1) 
        plt.colorbar()

        dbatch = self.dataset.data_3d
        dbatch.requires_grad_()
        tmp = self(dbatch.cuda())
        tmp.sum().backward()

        test_output = dbatch.grad.cpu()

        # plt.figure(figsize=(12, 6))
        # #plt.subplot(3, 1, 1)
        # plt.title(r'Gradient map for $\frac{\partial f}{\partial x}$', fontsize=30)
        # plt.xlabel('x')
        # plt.ylabel('y')
        # my_limit = min(test_output[:, 0].max(), -test_output[:, 0].min())
        # plt.imshow(test_output[:, 0].view((self.dataset.img_array.shape)), cmap='seismic', vmin=-my_limit, vmax=my_limit)
        # plt.colorbar()

        # plt.figure(figsize=(12, 6))
        # #plt.subplot(3, 1, 2)
        # plt.title(r'Gradient map for $\frac{\partial f}{\partial y}$', fontsize=30)
        # plt.xlabel('x')
        # plt.ylabel('y')
        # my_limit = min(test_output[:, 1].max(), -test_output[:, 1].min())
        # plt.imshow(test_output[:, 1].view((self.dataset.img_array.shape)), cmap='seismic', vmin=-my_limit, vmax=my_limit)
        # plt.colorbar()

        plt.figure(figsize=(12, 6))
        #plt.subplot(3, 1, 3)
        plt.title(r'Gradient map for $||\nabla f(x,y,z)||_2$', fontsize=30) #$\sqrt{\left(\frac{\partial f}{\partial x}\right)^2 + \left(\frac{\partial f}{\partial y}\right)^2}$
        plt.xlabel('x')
        plt.ylabel('y')

        plt.imshow(((test_output[:, 0].pow(2) + test_output[:, 1].pow(2) + test_output[:, 2].pow(2)).pow(0.5)).view((self.dataset.img_array.shape)), cmap='plasma')
        plt.colorbar()

        
        return test_output

    def show_on_cylinder(self, prediction):
        df = pd.DataFrame({'x': self.dataset.data_3d.x.cpu().detach().numpy(), 'y': self.dataset.data_3d.y.cpu().detach().numpy(), 'z': self.dataset.data_3d.z.cpu().detach().numpy(), 'label': prediction.flatten()})
        return px.scatter_3d(df, x='x', y='y', z='z', color='label').update_traces(marker={'size': 2})

    def change_lr(self, lr):
        for g in self.optimizer.param_groups:
            g['lr'] = lr

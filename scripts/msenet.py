import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from IPython.display import clear_output
from tqdm import tqdm

from scripts.config import device

class MSENet(nn.Module):
    def __init__(self, img, mode, arch):
        super(MSENet, self).__init__()
        self.net = nn.Sequential()
        for i in range(len(arch)-1):
            self.net.add_module(f'linear_{i}', nn.Linear(arch[i], arch[i+1]))
            self.net.add_module(f'tanh_{i}', nn.Tanh())
        self.arch = arch
        self.img = torch.FloatTensor(img.img_array)
        self.data = img.data_3d if mode == '3d' else img.data_2d
        self.loss_dict = []

    def forward(self, x):
        output = self.net(x)
        return output
    
    def compute_loss(self, prediction):
        mse = nn.MSELoss()
        loss = mse(prediction.squeeze(), self.img.flatten().to(device))
        self.loss_dict.append(loss.item())
        return loss
    
    def train(self, num_epochs=1000, need_plot=True, show_freq=1000, lr=1e-3, weight_decay=1e-3):
        num_epochs = int(num_epochs)
        optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)
        for epoch in tqdm(range(num_epochs), desc='Epoch'):
            optimizer.zero_grad()
            prediction = self.forward(self.data.to(device))
            loss = self.compute_loss(prediction)
            loss.backward()
            optimizer.step()
            if need_plot:
                if (epoch+1) % show_freq == 0 or epoch == num_epochs-1:
                    clear_output(wait=True)
                    fig = plt.figure(figsize=(20, 10))
                    gs = fig.add_gridspec(2, 2)
                    ax1 = fig.add_subplot(gs[1, :])
                    ax1.plot(self.loss_dict)
                    ax1.set_title('Loss')
                    ax1.set_xlabel('Epoch')
                    ax2 = fig.add_subplot(gs[0, 0])
                    ax2.imshow(prediction.view(self.img.shape).cpu().detach(), cmap='PuOr')
                    ax2.set_title('prediction')
                    ax3 = fig.add_subplot(gs[0, 1])
                    ax3.imshow(self.img, cmap='PuOr')
                    ax3.set_title('target')
                    fig.suptitle(f'Epoch: {epoch+1}/{num_epochs}\nLoss: {loss.item():.4f}, lr: {lr}, weight_decay: {weight_decay}\nArchitecure: {self.arch}')
                    plt.tight_layout()
                    plt.show()
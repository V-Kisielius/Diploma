import torch
import gpytorch
from gpytorch.constraints import Interval
from gpytorch.kernels import RBFKernel, PeriodicKernel, ScaleKernel
import matplotlib.pyplot as plt
from tqdm import tqdm
from scripts.config import device

IMG_SIZE = (50, 100)

def map_to_cylinder(points): # REMOVE THIS FUNCTION. SEE IMAGEDATA
    return torch.stack((
        torch.cos(2*torch.pi*points[:, 1]),
        torch.sin(2*torch.pi*points[:, 1]),
        torch.pi*points[:, 0]), -1)

class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood,
                 RBF_lengthscale_constraint=None,
                 Periodic_lengthscale_constraint=None):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.likelihood = likelihood
        self.mean_module = gpytorch.means.ZeroMean()
        self.RBF = RBFKernel(ard_num_dims=3,
                             lengthscale_constraint=RBF_lengthscale_constraint)
        self.Periodic = PeriodicKernel(ard_num_dims=3,
                                       lengthscale_constraint=Periodic_lengthscale_constraint)
        self.covar_module = ScaleKernel(self.RBF) + ScaleKernel(self.Periodic)

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

    def start_training(self, train_x, train_y, num_iter=100, need_plot=True):
        self.train()
        self.likelihood.train()
        # Includes GaussianLikelihood parameters
        optimizer = torch.optim.Adam(self.parameters(), lr=0.1)

        # "Loss" for GPs - the marginal log likelihood
        my_loss = gpytorch.mlls.ExactMarginalLogLikelihood(
            self.likelihood, self)

        history = {"loss": [],
                   "lengthscale": [],
                   "noise": []}

        for _ in tqdm(range(num_iter), desc=f'Training on {device}'):
            optimizer.zero_grad()
            output = self(train_x)
            loss = -my_loss(output, train_y)
            loss.backward()
            history["loss"].append(loss.item())
            # lengthscale = self.covar_module.base_kernel.lengthscale.item()
            # history["lengthscale"].append(lengthscale)
            # history["noise"].append(self.likelihood.noise.item())
            optimizer.step()

        if need_plot:
            plt.figure(figsize=(10, 5))
            plt.plot(history["loss"])
            plt.title(f'Loss')
            plt.xlabel('Iteration')
            plt.ylabel('Loss')
            plt.show()

    def predict(self, data, num_samples=16, need_plot=True):
        self.eval()
        self.likelihood.eval()
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            samples = self(data.to(device)).rsample(
                sample_shape=torch.Size((num_samples,)))
            samples = (samples - samples.min(dim=1)[0].unsqueeze(1)) / (
                samples.max(dim=1)[0].unsqueeze(1) - samples.min(dim=1)[0].unsqueeze(1))
            samples = 2 * samples.cpu().detach().view((-1, *IMG_SIZE)) - 1
        if need_plot:
            n = min(int(num_samples ** 0.5), 4)
            _, axs = plt.subplots(n, n, figsize=(20, 10))
            plt.suptitle(f'Samples')
            for i in range(n):
                for j in range(n):
                    axs[i, j].imshow(samples[i*n+j], cmap='PuOr')
                    axs[i, j].contour(samples[i*n+j], levels=0, colors='k')
                    axs[i, j].axis('off')
            plt.show()
        return samples
    
def default_train(rbf_right=0.32, periodic_right=0.15, iters=100):
    # Train set and test set initialization
    dx, dy = 1 / IMG_SIZE[0], 1 / IMG_SIZE[1]
    x, y = torch.linspace(0, 1-dx, IMG_SIZE[0]), torch.linspace(0, 1-dy, IMG_SIZE[1])
    xv, yv = torch.meshgrid(x, y, indexing="ij")
    x_test = torch.cat((
        xv.contiguous().view(xv.numel(), 1),
        yv.contiguous().view(yv.numel(), 1)),
        dim=1)
    x_train = torch.cat((x_test[:5*IMG_SIZE[1], :], x_test[-5*IMG_SIZE[1]:, :]), 0)
    x_train = map_to_cylinder(x_train)
    x_test = map_to_cylinder(x_test)
    y_train = torch.cat((torch.ones(5*IMG_SIZE[1]), -torch.ones(5*IMG_SIZE[1])))
    # Model initialization and training
    rbf_lengthscale_right = Interval(0, rbf_right)
    periodic_lengthscale_right = Interval(0, periodic_right)
    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    model = ExactGPModel(x_train, y_train, likelihood,
                         RBF_lengthscale_constraint=rbf_lengthscale_right,
                         Periodic_lengthscale_constraint=periodic_lengthscale_right)
    model = model.to(device)
    x_train = x_train.to(device)
    y_train = y_train.to(device)
    model.start_training(x_train, y_train, num_iter=iters, need_plot=True)
    return model, x_test
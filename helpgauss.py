import torch
import gpytorch
import matplotlib.pyplot as plt
from tqdm import tqdm

IMG_SIZE = (50, 100)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, ker_name, dimension=3, nu=1.5, lengthscale_constraint=None, alpha_constraint=None):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.likelihood = likelihood
        self.dim = dimension
        self.mean_module = gpytorch.means.ZeroMean()
        kernels = {
                'RBF' : gpytorch.kernels.RBFKernel(ard_num_dims=self.dim, lengthscale_constraint=lengthscale_constraint),
                'Matern' : gpytorch.kernels.MaternKernel(ard_num_dims=self.dim, nu=nu, lengthscale_constraint=lengthscale_constraint),
                'RQ' : gpytorch.kernels.RQKernel(ard_num_dims=self.dim, lengthscale_constraint=lengthscale_constraint, alpha_constraint=alpha_constraint),
                   }
        self.kernel_name = ker_name
        # self.covar_module = gpytorch.kernels.ScaleKernel(kernels[self.kernel_name])
        self.covar_module = kernels[self.kernel_name]


    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

    def start_training(self, train_x, train_y, num_iter=100, need_plot=True):
        self.train()
        self.likelihood.train()
        optimizer = torch.optim.Adam(self.parameters(), lr=0.1)  # Includes GaussianLikelihood parameters

        # "Loss" for GPs - the marginal log likelihood
        my_losss = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self)

        history = {"loss": [], 
                "lengthscale": [],
                "noise": []}

        for _ in tqdm(range(num_iter), desc='Training'):
            optimizer.zero_grad()
            output = self(train_x)
            loss = -my_losss(output, train_y)
            loss.backward()
            history["loss"].append(loss.item())
            # lengthscale = self.covar_module.base_kernel.lengthscale.item()
            # history["lengthscale"].append(lengthscale)
            # history["noise"].append(self.likelihood.noise.item())
            optimizer.step()
            
        if need_plot:
            self.plot_history(history)

    def plot_history(self, history):
        plt.figure(figsize=(10, 5))
        plt.plot(history["loss"])
        plt.title(f'Loss for {self.kernel_name} kernel')
        plt.xlabel('Iteration')
        plt.ylabel('Loss')
        plt.show()

    def predict(self, data, need_plot=True):
        self.eval()
        self.likelihood.eval()

        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            sampled_preds = self(data.to(device)).rsample(sample_shape=torch.Size((16,)))

        # plot sampled_preds as a grid 4x4
        if need_plot:
            _, axs = plt.subplots(4, 4, figsize=(20, 10))
            plt.suptitle(f'Predictions for {self.kernel_name} kernel')
            for i in range(4):
                for j in range(4):
                    axs[i, j].imshow(sampled_preds[i*4+j].cpu().detach().numpy().reshape(IMG_SIZE), cmap='PuOr')
                    axs[i, j].contour(sampled_preds[i*4+j].cpu().detach().numpy().reshape(IMG_SIZE), levels=0, colors='k')
                    axs[i, j].axis('off')
            plt.show()

        return sampled_preds
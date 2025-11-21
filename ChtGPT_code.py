import torch
import torch.nn as nn
import gpytorch
from gpytorch.kernels import RBFKernel, ScaleKernel
from gpytorch.means import ConstantMean,ZeroMean
from linear_operator.operators import DiagLinearOperator
from torch.optim import Adam

# class ReplicatedNoiseExactGP(gpytorch.models.ExactGP):
#     def __init__(self, train_x, train_y, N_vec):
#         # dummy likelihood (we handle noise ourselves)
#         likelihood = gpytorch.likelihoods.GaussianLikelihood(noise_prior=None)
#         super().__init__(train_x, train_y, likelihood)
#         self.mean_module = ZeroMean()
#         base_kernel = RBFKernel(ard_num_dims=1)
#         self.covar_module = ScaleKernel(base_kernel)  # outputscale * RBF
#         # learnable scalar raw parameter for sigma^2 (use softplus to keep > 0)
#         self.raw_sigma2 = nn.Parameter(torch.tensor(0.0))  # unconstrained
#         # store 1/N as a fixed vector (float tensor)
#         self.register_buffer("invN", (1.0 / N_vec).view(-1))  # shape (n,)

#     @property
#     def sigma2(self):
#         return torch.nn.functional.softplus(self.raw_sigma2)  # positive scalar

#     def forward(self, x):
#         # mean = self.mean_module(x).squeeze(-1)
#         # covar = self.covar_module(x)  # LazyTensor
#         # # create diagonal noise = sigma2 * (1/N)
#         # noise_diag = self.sigma2 * self.invN
#         # return gpytorch.distributions.MultivariateNormal(mean, covar + DiagLinearOperator(noise_diag))
#         mean = self.mean_module(x).squeeze(-1)
#         covar = self.covar_module(x)

#         # If x are the training inputs, include replication noise
#         if self.training:
#             diag_noise = self.sigma2 * self.invN
#             covar = covar + DiagLinearOperator(diag_noise)

#         return gpytorch.distributions.MultivariateNormal(mean, covar)

# #TODO implement the chatGPT code which removes the inbuilt

import torch
import torch.nn as nn
import gpytorch
from gpytorch.kernels import RBFKernel, ScaleKernel
from gpytorch.means import ConstantMean
from linear_operator.operators import DiagLinearOperator

class ReplicatedNoiseExactGP(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, N_vec):
        # Dummy noise values for initialization (will not be used directly)
        dummy_noise = torch.full_like(train_y, 1e-8)
        likelihood = gpytorch.likelihoods.FixedNoiseGaussianLikelihood(
            noise=dummy_noise, learn_additional_noise=False
        )
        super().__init__(train_x, train_y, likelihood)

        self.mean_module = ConstantMean()
        self.covar_module = ScaleKernel(RBFKernel())
        #self.covar_module = RBFKernel()
        # Scalar learnable σ² parameter (raw)
        self.raw_sigma2 = nn.Parameter(torch.tensor(0.0))
        # Store per-point replication weights (constant buffer)
        self.register_buffer("invN", (1.0 / N_vec).view(-1))

    @property
    def sigma2(self):
        return torch.nn.functional.softplus(self.raw_sigma2)

    def forward(self, x):
        mean = self.mean_module(x).squeeze(-1)
        base_covar = self.covar_module(x)

        if self.training:
            diag_noise = self.sigma2 * self.invN
            covar = base_covar + DiagLinearOperator(diag_noise)
        else:
            # Do NOT add training-dependent noise at prediction
            covar = base_covar

        return gpytorch.distributions.MultivariateNormal(mean, covar)



def test_function(x):

    return torch.sin(5*x) + torch.cos(7*x) 


# --------------------------
# Usage / training loop
# --------------------------
# train_x: (n x d), train_y: (n,), N_vec: (n,) integers (replication counts)
k = 50
# train_x = torch.rand(k)
# train_x = train_x.sort()[0] #Sort values for ploting
train_x = torch.linspace(0,1,k)
#TODO Make an N_vec biaded towards low values of n

N_vec = torch.randint(low=3, high=50, size=(k,)).float()  # example
N_vec = torch.ones(k) * 3
true_sigma2 = 1
noise = torch.randn_like(train_x) * (true_sigma2 / N_vec)
train_y = test_function(train_x) + noise

# K_true = gpytorch.kernels.ScaleKernel(RBFKernel()).forward(train_x).evaluate()
# train_y = torch.distributions.MultivariateNormal(torch.zeros(train_x.size(0)), K_true).sample() + noise

import matplotlib.pyplot as plt

plt.plot(train_x,train_y)
plt.xlabel('$x$')
plt.ylabel('$y_{evals}$')


##Generate model

def train_model(model,training_iter):
    train_x = model.train_inputs[0]
    train_y = model.train_targets
    model.train()
    # We'll compute the marginal log-likelihood directly using the multivariate normal returned by model.forward
    optimizer = Adam([
        {"params": model.parameters()},
    ], lr=0.05)

    for it in range(training_iter):
        optimizer.zero_grad()
        mvn = model(train_x)  # returns MVN with kernel + diag(sigma2 * invN)
        # negative log marginal likelihood
        nll = -mvn.log_prob(train_y)
        nll.backward()
        optimizer.step()
        if it % 50 == 0:
            # print(f"iter {it:03d}  nll={nll.item():.4f}  sigma2={model.sigma2.item():.6f} lscale={model.covar_module.base_kernel.lengthscale.item()} mean = {model.mean_module.constant.item()}")
            print(f"iter {it:03d}  nll={nll.item():.4f}  sigma2={model.sigma2.item():.6f} lscale={model.covar_module.base_kernel.lengthscale.item()} outscale={model.covar_module.outputscale.item()}")
    return model

model = ReplicatedNoiseExactGP(train_x, train_y, N_vec)
# print(f'mean = {model.mean_module.constant}')
model = train_model(model,1000)


def model_predict(test_x,test_n,model):
    # put model and likelihood in eval mode
    model.eval()
    likelihood = model.likelihood
    likelihood.eval()
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        f_pred = model(test_x)
        y_pred = gpytorch.distributions.MultivariateNormal(
        f_pred.mean,
        f_pred.covariance_matrix + (model.sigma2/test_n) * torch.eye(test_x.size(0))
    )
    return f_pred,y_pred

# Generate test points 
N_points = 100
test_x = torch.linspace(0,1,N_points)
test_n = torch.full_like(test_x,3)
#test_n = torch.randint(low=3, high=100, size=(N_points,)).float()
true_y = test_function(test_x)
f_pred,y_pred = model_predict(test_x,test_n,model)

model_noise = model.sigma2.item()

preds = y_pred

with torch.no_grad():
    # Initialize plot
    f, ax = plt.subplots(1, 1, figsize=(8, 6))

    # Get upper and lower confidence bounds
    lower, upper = preds.confidence_region()
    # Plot training data as black stars
    ax.plot(train_x.numpy(), train_y.numpy(), 'k*')
    # Plot predictive means as blue line
    ax.plot(test_x.numpy(), preds.mean.numpy(), 'b')
    ax.plot(test_x,true_y,'r')
    # Shade between the lower and upper confidence bounds
    ax.fill_between(test_x.numpy(), lower.numpy(), upper.numpy(), alpha=0.5)
    # ax.set_ylim([-3, 3])
    ax.legend(['Observed Data', 'Mean','True', 'Confidence'])
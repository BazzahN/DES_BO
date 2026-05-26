import torch
import torch.nn as nn
import gpytorch
from gpytorch.kernels import RBFKernel, ScaleKernel
from gpytorch.means import ConstantMean,ZeroMean
from linear_operator.operators import DiagLinearOperator
from torch.optim import Adam

class NoiseExactGP(gpytorch.models.ExactGP):
    '''
    Noiseless GP for estimating Heteroscedastic Noise
    '''

    def __init__(self, train_x, train_sigma):
        # dummy likelihood (we handle noise ourselves)
        likelihood = gpytorch.likelihoods.GaussianLikelihood(noise_prior=None)
        super().__init__(train_x, train_sigma, likelihood)
        self.mean_module = ZeroMean()
        base_kernel = RBFKernel(ard_num_dims=1)
        self.covar_module = ScaleKernel(base_kernel)  
  
    def forward(self, x):
        # mean = self.mean_module(x).squeeze(-1)
        # covar = self.covar_module(x)  # LazyTensor
        # # create diagonal noise = sigma2 * (1/N)
        # noise_diag = self.sigma2 * self.invN
        # return gpytorch.distributions.MultivariateNormal(mean, covar + DiagLinearOperator(noise_diag))
        mean = self.mean_module(x).squeeze(-1)
        covar = self.covar_module(x)

        return gpytorch.distributions.MultivariateNormal(mean, covar)

class FuncGP(gpytorch.models.ExactGP):
    '''
    Noiseless GP for estimating Heteroscedastic Noise
    '''

    def __init__(self, train_x,train_y, train_s_2):
        # dummy likelihood (we handle noise ourselves)
        likelihood = gpytorch.likelihoods.FixedNoiseGaussianLikelihood(noise=train_s_2)
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = ZeroMean()
        base_kernel = RBFKernel(ard_num_dims=1)
        self.covar_module = ScaleKernel(base_kernel)  # outputscale * RBF
  
    def forward(self, x):
        mean = self.mean_module(x).squeeze(-1)
        covar = self.covar_module(x)

        return gpytorch.distributions.MultivariateNormal(mean, covar)
    
## Experiment Functions
def noise_function(x,sigma2_0):
    '''
    Inputs
    ------
    x: tensor
        The test/train locations for the GP. Here x \in [0,1]
    sigma_0: float
        Scale of the heteroscedastic noise function
    '''
    return (0.5 * torch.cos(2*torch.pi*x) + 1)* sigma2_0

def test_function(x):

    return torch.sin(5*x) + torch.cos(7*x) 



def train_model(model,training_iter=200):
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
            print(f"iter {it:03d}  nll={nll.item():.4f}")
    return model



k = 20
train_x = torch.rand(k) #Generates k values between 0 and 1
train_x = train_x.sort()[0] #Sort values for ploting

#TODO Make an N_vec biaded towards low values of n

N_vec = torch.randint(low=3, high=10, size=(k,)).float()  # example

#Calculate sigma^2(x)
sigma2= 0.5
sigma2_vec = noise_function(train_x,sigma2)
s2_vec = sigma2_vec / N_vec
noise = torch.randn_like(train_x) * torch.sqrt(s2_vec)
train_y = test_function(train_x) + noise


import matplotlib.pyplot as plt

'''
Example assuems that \hat {sigma}^2 = sigma^2. A more sophisticated model
is to be used later.
'''
noise_model = NoiseExactGP(train_x,sigma2_vec)

noise_model = train_model(noise_model)

train_s2 = s2_vec
main_model = FuncGP(train_x,train_y,sigma2_vec / N_vec)

# TODO FuncGP is useless at finding good hyperparamaters
main_model = train_model(main_model,1000)

def model_predict(test_x,model):
    # put model and likelihood in eval mode
    model.eval()
    likelihood = model.likelihood
    likelihood.eval()
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        predictors = likelihood(model(test_x))
    
    return predictors

N_points = 100
test_x = torch.linspace(0,1,N_points)
true_y = test_function(test_x)

noise_preds = model_predict(test_x,noise_model)
main_preds = model_predict(test_x,main_model)


preds = main_preds
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


# def plot_init(size)

def plot_gp_preds(test_x,train_x,train_y,preds):

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

# ub1 = 3
# test_n = torch.ones_like(test_x) * ub1
# X_1 = torch.cat([test_x.unsqueeze(-1),test_n.unsqueeze(-1)],axis=1).unsqueeze(1)

# EI_vals = AEI(X_1)

# ub2 = 10
# test_n = torch.ones_like(test_x) * ub2
# X_2 = torch.cat([test_x.unsqueeze(-1),test_n.unsqueeze(-1)],axis=1)

# print(f'penalised EI with n={ub1}')
# EI_vals_1 = EI(X_1)

# print(f'penalised EI with n={ub2}')
# EI_vals_2 = EI(X_2)


# with torch.no_grad():
#     f, ax = plt.subplots(1, 1, figsize=(8, 6))

#     ax.plot(test_x,EI_vals_1)
#     ax.plot(test_x,EI_vals_2)
#     ax.legend([f'ub {ub1}',f'ub {ub2}'])

## Mixed space optimise

# N = [3,5,10,20,30,50,60,70]

# replications = [{1:i} for i in N]

# candidates, acq_val = optimize_acqf_mixed(AEI,
#                                           bounds = bounds,
#                                           q = 1,
#                                           fixed_features_list=replications,
#                                           num_restarts=25,
#                                           raw_samples=500)

#Random x generation
# train_x = torch.rand(k)
# train_x = train_x.sort()[0] #Sort values for ploting

# def plot_AF(N_points,
#             AF,
#             n_vals = torch.tensor([3,5,10]),
#             fig_title=None,
#             f_name=None,
#             colour = 'b'):
#     '''
#     Plots the predicted intrinsic uncertainty (sigma^2_eps) and extrinsic uncertainty (sigma^2_f)
#     of the Gaussian Process at a give iteration

#     Inputs
#     ------
#         N_points: Tensor
#             Number of grid points to make prediction
#         AF: AcquisitionFunction
#             An initalised Acquisition Function
#         n_vals: int Tensor
#             Chosen n value 
#         fig_title: String Optional[Default=None] 
#             Title of the figure if needed.
#         f_name: String Optional[Default=None]
#             If a filename string is supplied the generated figure will be automatically saved under
#             that name. Format must be supplied in f_name i.e. .png or .eps
#         new_point: bool Optional[Default=False]
#             Highlights the selected candidate if set to true
#     '''

#     AF_vals = AF_output(N_points,
#                         AF,
#                         n_vals)


#     with torch.no_grad():
#             # Initialize plot
#             f, ax = plt.subplots(1, 1, figsize=(8, 6))

#             ax.plot(test_x,AF_vals,colour,label='$AF(x)$')

#             ax.set_xlabel('$x$')
#             ax.set_ylabel('$AF(x)$')
            
#             ax.legend()
#             if fig_title is not None:
#                 ax.set_title(fig_title)
        

#     if f_name is not None:
#         plt.savefig(f_name, dpi=500,bbox_inches = "tight")
    
#     plt.show()

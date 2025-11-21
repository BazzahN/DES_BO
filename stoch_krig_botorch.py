import torch
from torch import Tensor
import matplotlib.pyplot as plt
from test_utils import test_function, noise_function,InverseLinearCostModel
from DES_acqfs import DES_EI
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.acquisition import ExpectedImprovement
tkwargs = {
    "dtype": torch.double,# Datatype used by tensors
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"), # Declares the 'device' location where the Tenosrs will be stored
}

# Fit
k = 5 
# train_x = torch.rand(k)
# train_x = train_x.sort()[0] #Sort values for ploting
train_x = torch.linspace(0,1,k).to(**tkwargs)

#TODO Make an N_vec biased towards low values of n using a triangular distribution
#N_vec = torch.randint(low=3, high=10, size=(k,)).to(**tkwargs).float()  # example
N_vec = torch.ones_like(train_x) * 5
#Calculate sigma^2(x)
sigma2= 10
sigma2_vec = noise_function(train_x,sigma2).to(**tkwargs)

s2_vec = sigma2_vec / N_vec
noise = torch.randn_like(train_x) * s2_vec
train_y = test_function(train_x).to(**tkwargs) + noise

train_x = train_x.reshape(k,1)
train_y = train_y.reshape(k,1)
s2_vec = s2_vec.reshape(k,1)
sigma2_vec = sigma2_vec.reshape(k,1)


def get_stoch_kriging_model(train_x,train_n,train_y,sigma2_hat):
    '''
    Constructs and conditions on the dataset D_t = (train_x,train_n,train_y) the 
    stochastic kriging model.

    Inputs
    ------
    train_x: Tensor
        A tensor of evaluated locations
    train_y: Tensor
        A tensor of evaluations
    train_n: Tensor
        A tensor of number of replications used in the evaluations
    train_y: Tensor
        A tensor of evaluation data
    sigma2_hat: Tensor
        A tensor of estimated variance values


    Returns
    -------
    model:dict('f' = latent model, 'eps' = noise model)
        A dictionary of the stochastic kriging Gaussian process model components
    
    '''

    s2 = sigma2_hat / train_n #Sample variance transform
    #Fit main Model
    gp = SingleTaskGP(train_x,
                      train_y,
                      train_Yvar=s2,
                      outcome_transform=None)

    mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
    fit_gpytorch_mll(mll)

    #Fit Noise Model
    gp_noise = SingleTaskGP(train_x,
                            sigma2_hat,
                            train_Yvar=torch.zeros_like(sigma2_hat),
                            outcome_transform=None,
                            )
    mll = ExactMarginalLogLikelihood(gp_noise.likelihood,gp_noise)
    fit_gpytorch_mll(mll)

    #Puts model into dictionary
    model = {'f':gp,'eps':gp_noise}

    return model

sk_model = get_stoch_kriging_model(train_x,N_vec,train_y,sigma2_vec)

#fit 
#Predict with model
n_ub = 10

N_points = 100
test_x = torch.linspace(0,1,N_points).to(**tkwargs)
test_n = torch.ones_like(test_x) * n_ub

true_y = test_function(test_x)


#TODO Remove noisy prediction as posterior method interferes with covariance matrix
#TODO Implement addition of predicted noise manually 
main_preds = sk_model['f'].posterior(test_x,observation_noise=False) 

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
    ax.set_xlabel('$x$')
    ax.set_ylabel('$y$')
    # ax.set_ylim([-3, 3])
    ax.legend(['Observed Data', 'Mean','True', 'Confidence'])
plt.show()

y_best = train_y.min()

lin_cost_func = InverseLinearCostModel([1,1])

AEI = DES_EI(model_f=sk_model['f'],
             model_eps=sk_model['eps'],
             best_f=y_best,
             cost_model=lin_cost_func,
             maximize=False)
                         
from botorch.optim.optimize import optimize_acqf,optimize_acqf_mixed
bounds = torch.tensor([[0,3] * 1,
                        [1,100] * 1],
                        dtype=torch.double,
                        device=torch.device("cpu"))

def optimise_acqf_get_candidate(acq_func, bounds,num_restarts=25,raw_samples=500):
    '''
    Finds the optimal candidate point of the supplied acquisition function.

    Inputs
    ------
    acq_func: AcquisitionFunction
        The special acqf used to select both n and x
    bounds: Tensor
        Bounds for the mixed (x,n) space
    num_restarts: int
        Used during optimisation - the number of starting points to generate for optimisation
    raw_samples: int
        Used during optimisation - the number of raw samples used in the optim method's candidate generation scheme
    
    Returns
    -------
    candidates: Tensor
        d-dim selection of candidates
    acqf_val: Tensor
        Acqf val of candidate
    '''
    candidates, acqf_val = optimize_acqf(acq_func,
                                         bounds=bounds,
                                         q=1,
                                         num_restarts=num_restarts,
                                         raw_samples=raw_samples)

    return candidates.detach(), acqf_val

def candidate_acq(acq_func, bounds,num_restarts=25,raw_samples=500):
    '''
    Finds the most promising candidate pair (x,n) given the predictive posterior
    conditioned on dataset D_t. 

    Inputs
    ------
    acq_func: AcquisitionFunction
        The special acqf used to select both n and x
    bounds: Tensor
        Bounds for the mixed (x,n) space
    num_restarts: int
        Used during optimisation - the number of starting points to generate for optimisation
    raw_samples: int
        Used during optimisation - the number of raw samples used in the optim method's candidate generation scheme
    
    Returns
    -------
    candidates: Tensor
        d-dim selection of candidates
    acqf_val: Tensor
        Acqf val of candidate
    '''
    new_xn,acqf_val = optimise_acqf_get_candidate(acq_func,
                                                  bounds,
                                                  num_restarts,
                                                  raw_samples)
    
    # The selected n point is rounded to the nearest integer
    new_xn[0,1] = new_xn[0,1].round(decimals=0)

    return new_xn,acqf_val


def f_best_acq(strategy, bounds, num_restarts = 25,raw_samples=500):
    '''
    Finds the current best optimal point f^* using a stratergy based upon the 
    the supplied acquisiton function.
    
    Inputs
    ------
        strategy: PosteriorMean
            How f^* is calculated from GP posterior.
        bounds: Tensor
            Bounds of the decision space X
        num_restarts: int
            Used during optimisation - the number of starting points to generate for optimisation
        raw_samples: int
            Used during optimisation - the number of raw samples used in the optim method's candidate generation scheme
    Returns
    -------
        f_best: Tensor
            Returns the current f_best
    '''

    _,f_best = optimise_acqf_get_candidate(strategy,
                                           bounds,
                                           num_restarts,
                                           raw_samples)

    return f_best
## Mixed space optimise

# N = [3,5,10,20,30,50,60,70]

# replications = [{1:i} for i in N]

# candidates, acq_val = optimize_acqf_mixed(AEI,
#                                           bounds = bounds,
#                                           q = 1,
#                                           fixed_features_list=replications,
#                                           num_restarts=25,
#                                           raw_samples=500)

from botorch.optim import gen_batch_initial_conditions

# init_conds = gen_batch_initial_conditions(EI,bounds,1,25,500)

ub1 = 3
test_n = torch.ones_like(test_x) * ub1
X_1 = torch.cat([test_x.unsqueeze(-1),test_n.unsqueeze(-1)],axis=1).unsqueeze(1)

EI_vals = AEI(X_1)

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
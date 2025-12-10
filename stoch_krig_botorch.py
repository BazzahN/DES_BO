import torch
from torch import Tensor
import matplotlib.pyplot as plt
from test_utils import test_function, heteroscedastic_noise,flat_noise,InverseLinearCostModel, test_function_2
from DES_acqfs import DES_EI, AEI_fq
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.acquisition import ExpectedImprovement
tkwargs = {
    "dtype": torch.double,# Datatype used by tensors
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"), # Declares the 'device' location where the Tenosrs will be stored
}
seed = 12345

torch.manual_seed(seed)

#GLOBALS
SIGMA2 = 5 #Scale of noise surface
PHI = 0 #Shift of Heteroscedastic noise surface
MAXIMIZE= True #Sets problem to maximise test function or minimise test funciton

# Constants
k = 3 #number of samples points
n = 5 #flat number of replications

#Generate decision variables
#NOTE: Random train_x code moved to stoch_kriging
train_x = torch.linspace(0.1,1,k).to(**tkwargs)
train_n = torch.ones_like(train_x) * n

#Calculate sigma^2(x)
noise_function = heteroscedastic_noise
test_func = test_function_2
sigma2_vec = noise_function(train_x,SIGMA2,PHI).to(**tkwargs)

# Calculate sample variance
s2_vec = sigma2_vec / train_n
noise = torch.randn_like(train_x) * s2_vec

#Generate y values from latent function plus heteroscedastic Gaussian noise
train_y = test_func(train_x).to(**tkwargs) + noise

#Plot Test Function
N_points=500
test_x = torch.linspace(0,1,N_points).to(**tkwargs)
true_sig2 = noise_function(test_x,SIGMA2,PHI).to(**tkwargs)
true_sig = true_sig2.sqrt()
true_y = test_func(test_x)

upper = true_y + true_sig
lower = true_y - true_sig

if MAXIMIZE:
    optim_point = true_y.max()
    optim_idx = true_y.argmax()
else:
    optim_point = true_y.min()
    optim_idx = true_y.argmin()

optim_sol = test_x[optim_idx]

f, ax = plt.subplots(1, 2, figsize=(18, 6))
ax[0].plot(test_x.numpy(),true_y.numpy(),label='$f(x)$')
ax[0].plot(optim_sol.numpy(),optim_point.numpy(),'k*',label='optimal point')
ax[0].fill_between(test_x.numpy(),lower.numpy(),upper.numpy(),alpha=0.2,label='True $+/- \sigma$',color='g')
ax[0].set_xlabel('$x$')
ax[0].set_ylabel('$f(x)$')
ax[0].set_title('True Function with $\pm \sigma_{\epsilon}$')
ax[0].legend()

ax[1].plot(test_x.numpy(),true_sig2.numpy(),label='True $\sigma^2(x)$',color='g')
ax[1].set_xlabel('$x$')
ax[1].set_ylabel('$\sigma{\epsilon}^2(x)$')
ax[1].set_title(f'Heteroscedastic Noise|$\sigma^2_0 =${SIGMA2}')
ax[1].legend()

# plt.suptitle('True Function with Heteroscedastic Variance Function')
plt.savefig('BODES_test_problem.png',dpi=500,bbox_inches = 'tight')
plt.show()

def get_new_y_and_sigma(x,n,sigma2 = SIGMA2,phi = PHI):
    '''
    Calculates an evaluation of the noisy test function used to represent the output from a real heteroscedastic
    simulation. 

    Inputs
    ------
        x: Tensor/ float
            The d-dim decision arguments for the test function
        n: Tensor/int   
            Specified number of replications to evaluate test function at specified argument x
        sigma2: Optional sigma2 = SIGMA2 - Global assignment of variable in home script
            Used as argument to the toy function to scale noise. 
    '''
    sigma2_vec = noise_function(x,sigma2,phi).to(**tkwargs)

    # Calculate sample variance
    s2_vec = sigma2_vec / n
    noise = torch.randn_like(x) * s2_vec
    #Generate y values from latent function plus heteroscedastic Gaussian noise
    train_y = test_func(x).to(**tkwargs) + noise

    return train_y, sigma2_vec


train_x = train_x.reshape(k,1)
train_n = train_n.reshape(k,1)
train_y, train_sigma2 = get_new_y_and_sigma(train_x,train_n)

from gpytorch.likelihoods.gaussian_likelihood import FixedNoiseGaussianLikelihood

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

    #Include observational noise on f hyperparamter 
    ## This is done by setting

    s2 = (sigma2_hat / train_n).view(-1, 1) #Sample variance transform
    #Fit main Model
    gp = SingleTaskGP(train_x,
                      train_y,
                      train_Yvar=s2,
                      outcome_transform=None,
                      #add_noise=True,
                      )

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

sk_model = get_stoch_kriging_model(train_x,train_n,train_y,train_sigma2)


## Model Predictions
def plot_GP(test_x,
            train_x,
            train_y,
            true_y,
            posterior_distb,
            fig_title=None,
            f_name=None,
            new_point=False):

    '''
    Plots the predicted mean and confidence intervals of the GP's posterior mean
    and the training points used to condition the model.

    Inputs
    ------
        test_x: Tensor
            A d-dim tensor of points to be interpolated by the GP
        train_x: Tenosr
            The d-dim tensor of points which make up part of the dataset used to condition the GP
        train_y: Tensor
            The tensor of evaluations taken at the train_x point
        true_y: Tensor
            The output of the true latent function
        posterior_distb: Posterior
            A Posterior type object used to supply the predicted mean and confidence intervals
            after conditioning on D    
        fig_title: String Optional[Default=None] 
            Title of the figure if needed.
        f_name: String Optional[Default=None]
            If a filename string is supplied the generated figure will be automatically saved under
            that name. Format must be supplied in f_name i.e. .png or .eps
        new_point: bool Optional[Default=False]
            Highlights the selected candidate if set to true
    '''


    with torch.no_grad():
        # Initialize plot
        f, ax = plt.subplots(1, 1, figsize=(8, 6))

        if new_point:
            new_x = train_x[-1]
            new_y = train_y[-1]
            #Removes new points to avoid repetition
            #NOTE: This might not matter if being called first 
            train_x = train_x[:-1]
            train_y = train_y[:-1]
            #Plot new point as red star
            ax.plot(new_x.numpy(),new_y.numpy(),'r*',label='New Point')
        # Get upper and lower confidence bounds
        # lower, upper = posterior_distb.confidence_region()
        lower = posterior_distb.mean - posterior_distb.variance.sqrt()
        upper = posterior_distb.mean + posterior_distb.variance.sqrt()
        # Plot training data as black stars
        ax.plot(train_x.numpy(), train_y.numpy(), 'k*',label='Evaluations')
        # Plot predictive means as blue line
        ax.plot(test_x.numpy(), posterior_distb.mean.numpy(), 'b',label='Mean')
        ax.plot(test_x,true_y,'r',label='True')
        # Shade between the lower and upper confidence bounds
        ax.fill_between(test_x.numpy(), lower.flatten().numpy(), upper.flatten().numpy(), alpha=0.5,label='1 sigma')
        ax.set_xlabel('$x$')
        ax.set_ylabel('$y$')
        
        ax.legend()
        if fig_title is not None:
            ax.set_title(fig_title)
        

    if f_name is not None:
        plt.savefig(f_name, dpi=500,bbox_inches = "tight")

    plt.show()        

# def poster_plot(N_points,
#                 train_x,
#                 train_y,
#                 acqf,
#                 n_vals,
#                 model,
#                 phi = 0,
#                 theta=1,
#                 fig_title=None,
#                 f_name=None,
#                 new_point=False):
   
#     #Generate true outptu and predictions
#     test_x = torch.linspace(0,1,N_points).to(**tkwargs)
#     true_y = test_function(test_x)
#     preds = model['f'].posterior(test_x,observation_noise=False) 
    
#     x,y = AF_output(N_points,
#                     acqf,
#                     n_vals)

#     with torch.no_grad():
#         # Initialize plot
#         f, ax = plt.subplots(1, 1, figsize=(8, 6))

#         if new_point:
#             new_x = train_x[-1]
#             new_y = train_y[-1]
#             #Removes new points to avoid repetition
#             #NOTE: This might not matter if being called first 
#             train_x = train_x[:-1]
#             train_y = train_y[:-1]
#             #Plot new point as red star
#             candidate = ax.plot(new_x.numpy(),new_y.numpy(),'r*',label='New Point')
        
#         # Get upper and lower confidence bounds
#         # lower, upper = posterior_distb.confidence_region()
#         lower = preds.mean - preds.variance.sqrt()
#         upper = preds.mean + preds.variance.sqrt()
        
#         # Plot training data as black stars
#         evidence = ax.plot(train_x.numpy(), train_y.numpy(), 'k*',label='Evaluations')
#         # Plot predictive means as blue line
#         predictions =ax.plot(test_x.numpy(), preds.mean.numpy(), 'b',label='Mean')
#         truth = ax.plot(test_x,true_y,'r',label='True')
#         # Shade between the lower and upper confidence bounds
#         unertainty = ax.fill_between(test_x.numpy(), lower.flatten().numpy(), upper.flatten().numpy(), alpha=0.5,label='1 sigma')
        
#         Gp_lines = 
#         for i, n in enumerate(n_vals):
#             ax.plot(x,theta*y[i] -phi,label=f"n={n}")
#         ax.set_xlabel('$x$')
#         ax.set_ylabel('$y$')
        
#         ax.legend([''])
#         if fig_title is not None:
#             ax.set_title(fig_title)
        

#     if f_name is not None:
#         plt.savefig(f_name, dpi=500,bbox_inches = "tight")

#     plt.show()        


def poster_plot(
    N_points,
    train_x,
    train_y,
    acqf,
    n_vals,
    model,
    phi=0,
    theta=1,
    fig_title=None,
    f_name=None,
    new_point=False,
):
    # Generate test points
    test_x = torch.linspace(0, 1, N_points).to(**tkwargs)

    # Generate true output and predictions
    true_y = test_func(test_x)
    preds = model['f'].posterior(test_x, observation_noise=False)

    # Acquisition function output
    x, y = AF_output(N_points, acqf, n_vals)

    with torch.no_grad():
        # Initialize plot
        fig, ax = plt.subplots(1, 1, figsize=(12, 6))

        # Handle "new point"
        if new_point:
            new_x = train_x[-1]
            new_y = train_y[-1]
            train_x = train_x[:-1]
            train_y = train_y[:-1]
            ax.plot(new_x.cpu().numpy(), new_y.cpu().numpy(), 'r*', label='New Point')

        # Confidence bounds
        mean = preds.mean.squeeze(-1)
        std = preds.variance.sqrt().squeeze(-1)
        lower = mean - std
        upper = mean + std

        # ---- Gaussian Process plots ----
        gp_lines = []

        gp_lines += ax.plot(train_x.cpu().numpy(), train_y.cpu().numpy(), 'k*', label='Data')
        gp_lines += ax.plot(test_x.cpu().numpy(), mean.cpu().numpy(), 'b', label='Mean')
        gp_lines += ax.plot(test_x.cpu().numpy(), true_y.cpu().numpy(), 'r', label='True')
        gp_lines.append(
            ax.fill_between(
                test_x.cpu().numpy(),
                lower.cpu().numpy(),
                upper.cpu().numpy(),
                alpha=0.3,
                label='1σ'
            )
        )

        # ---- Acquisition Function plots ----
        af_lines = []
        for i, n in enumerate(n_vals):
            line, = ax.plot(x, theta * y[i] - phi, label=f"n={n}")
            af_lines.append(line)

        ax.set_xlabel('$x$')
        ax.set_ylabel('$y$')

        if fig_title is not None:
            ax.set_title(fig_title)

        # ---- Create separate legends ----
        gp_legend = ax.legend(
            handles=gp_lines,
            title="GP",
            loc='upper left',
            bbox_to_anchor=(1.02, 1.0),
            borderaxespad=0
        )

        af_legend = ax.legend(
            handles=af_lines,
            title="AF",
            loc='upper left',
            bbox_to_anchor=(1.02, 0.6),
            borderaxespad=0.
        )

        # Add the first legend back manually
        ax.add_artist(gp_legend)

        # Leave space on the right for legends
        # plt.tight_layout(rect=[0, 0, 0.75, 1])

    # Save figure if requested
    if f_name is not None:
        plt.savefig(f_name, dpi=500, bbox_inches="tight")

    plt.show()


#Constants
N_points = 100 #Number of test points

#Generate test points
test_x = torch.linspace(0,1,N_points).to(**tkwargs)
true_y = test_func(test_x)
preds = sk_model['f'].posterior(test_x,observation_noise=False) 
    
# Plot GP predictions with dataset and predictions    
plot_GP(test_x,
        train_x,
        train_y,
        true_y,
        preds,
        fig_title='Inital'
        )

#TODO Turn this into a function
# preds = main_preds
# with torch.no_grad():
#     # Initialize plot
#     f, ax = plt.subplots(1, 1, figsize=(8, 6))

#     # Get upper and lower confidence bounds
#     lower, upper = preds.confidence_region()
#     # Plot training data as black stars
#     ax.plot(train_x.numpy(), train_y.numpy(), 'k*')
#     # Plot predictive means as blue line
#     ax.plot(test_x.numpy(), preds.mean.numpy(), 'b')
#     ax.plot(test_x,true_y,'r')
#     # Shade between the lower and upper confidence bounds
#     ax.fill_between(test_x.numpy(), lower.numpy(), upper.numpy(), alpha=0.5)
#     ax.set_xlabel('$x$')
#     ax.set_ylabel('$y$')
#     # ax.set_ylim([-3, 3])
#     ax.legend(['Observed Data', 'Mean','True', 'Confidence'])
# plt.show()



def plot_iter_output(N_points,
                     train_x,
                     train_y,
                     model,
                     fig_title=None,
                     f_name=None,
                     new_point = True):
    '''
    Generates and plots the GP output for each iteration fo the Bayeisan Optimisation
    loop
    '''
    #Generate true outptu and predictions
    test_x = torch.linspace(0,1,N_points).to(**tkwargs)
    true_y = test_func(test_x)
    preds = model['f'].posterior(test_x,observation_noise=False) 
    
    
    plot_GP(test_x=test_x,
            train_x=train_x,
            train_y=train_y,
            true_y=true_y,
            posterior_distb=preds,
            fig_title=fig_title,
            f_name=f_name,
            new_point=new_point)

def plot_ex_uncertainty(N_points,
                        model,
                        fig_title=None,
                        f_name=None,
                        colour = 'b'):
    '''
    Plots the predicted intrinsic uncertainty (sigma^2_eps) and extrinsic uncertainty (sigma^2_f)
    of the Gaussian Process at a give iteration over an N_points size grid of the entire design space
    

    Inputs
    ------
        N_points: Tensor
            Number of grid points to make prediction
        model: Model
            A BODES Gaussian Process type model
        fig_title: String Optional[Default=None] 
            Title of the figure if needed.
        f_name: String Optional[Default=None]
            If a filename string is supplied the generated figure will be automatically saved under
            that name. Format must be supplied in f_name i.e. .png or .eps
        new_point: bool Optional[Default=False]
            Highlights the selected candidate if set to true
    '''

    #Generate test points
    test_x = torch.linspace(0,1,N_points).to(**tkwargs)

    post_f = model['f'].posterior(test_x)
 
    sig_f = post_f.variance #Plot predicted extrinsic uncertainty


    #TODO: Plot these as sub plots as sigma_eps >> sigma_f
    with torch.no_grad():
            # Initialize plot
            f, ax = plt.subplots(1, 1, figsize=(8, 6))

            ax.plot(test_x,sig_f , colour,label='$\sigma_f^2$')
            ax.set_xlabel('$x$')
            ax.set_ylabel('$\sigma_f^2$')

            ax.legend()
            if fig_title is not None:
                ax.set_title(fig_title)
        

    if f_name is not None:
        plt.savefig(f_name, dpi=500,bbox_inches = "tight")
    
    plt.show()

def plot_in_uncertainty(N_points,
                        model,
                        fig_title=None,
                        f_name=None,
                        colour = 'r'):
    '''
    Plots the predicted intrinsic uncertainty (sigma^2_eps) and extrinsic uncertainty (sigma^2_f)
    of the Gaussian Process at a give iteration over an N_points size grid of the entire design space
    

    Inputs
    ------
        N_points: Tensor
            Number of grid points to make prediction
        model: Model
            A BODES Gaussian Process type model
        fig_title: String Optional[Default=None] 
            Title of the figure if needed.
        f_name: String Optional[Default=None]
            If a filename string is supplied the generated figure will be automatically saved under
            that name. Format must be supplied in f_name i.e. .png or .eps
        new_point: bool Optional[Default=False]
            Highlights the selected candidate if set to true
    '''

    #Generate test points
    test_x = torch.linspace(0,1,N_points).to(**tkwargs)

    post_eps = model['eps'].posterior(test_x)

    sig_eps = post_eps.mean #Plot predicted intrinsic uncertainty

    #TODO: Plot these as sub plots as sigma_eps >> sigma_f
    with torch.no_grad():
            # Initialize plot
            f, ax = plt.subplots(1, 1, figsize=(8, 6))

            ax.plot(test_x, sig_eps, colour,label='$\sigma_{\epsilon}^2$')
            ax.set_xlabel('$x$')
            ax.set_ylabel('$\sigma_{\epsilon}^2$')
            ax.legend()
            if fig_title is not None:
                ax.set_title(fig_title)
        

    if f_name is not None:
        plt.savefig(f_name, dpi=500,bbox_inches = "tight")
    
    plt.show()


def plot_imporv(N_points,
                f_best,
                model,
                fig_title=None,
                f_name=None,
                colour = 'b'):
    
    '''
    Plots the predicted intrinsic uncertainty (sigma^2_eps) and extrinsic uncertainty (sigma^2_f)
    of the Gaussian Process at a give iteration over an N_points size grid of the entire design space
    

    Inputs
    ------
        N_points: Tensor
            Number of grid points to make prediction
        model: Model
            A BODES Gaussian Process type model
        fig_title: String Optional[Default=None] 
            Title of the figure if needed.
        f_name: String Optional[Default=None]
            If a filename string is supplied the generated figure will be automatically saved under
            that name. Format must be supplied in f_name i.e. .png or .eps
        new_point: bool Optional[Default=False]
            Highlights the selected candidate if set to true
    '''

    #Generate test points
    test_x = torch.linspace(0,1,N_points).to(**tkwargs)

    post_f = model['f'].posterior(test_x)

    improv = post_f.mean - f_best 


    with torch.no_grad():
            # Initialize plot
            f, ax = plt.subplots(1, 1, figsize=(8, 6))

            ax.plot(test_x,improv, colour,label='$\hat{f}(x) - f^*$')
            ax.set_xlabel('$x$')
            ax.set_ylabel('$\hat{f}(x) - f^*$')
            
            ax.legend()
            if fig_title is not None:
                ax.set_title(fig_title)
        

    if f_name is not None:
        plt.savefig(f_name, dpi=500,bbox_inches = "tight")
    
    plt.show()


def AF_output(N_points,
              acqf,
              n_vals,
              ):
    '''
    Caluclates the AF output and returns a len(n_vals)x N_points tensor of AF values for
    each n choice in n_vals.

    Inputs
    ------
        N_points: Tensor
            Number of grid points to make prediction
        acqf: AcquisitionFunction
            An initalised Acquisition Function
        n_val: int Tensor
            Chosen n value 
    Returns
    -------
        test_x: Tensor
            Test grid on which values were evaluated
        AF_vals: len(n_vals)xN_points sized tensor
            Evaluated AF values
    '''

    #Generate test points
    test_x = torch.linspace(0,1,N_points).to(**tkwargs)

    # Build X
    X = torch.stack([
        test_x.repeat(len(n_vals)),          # column 0
        n_vals.repeat_interleave(len(test_x))  # column 1
    ], dim=1)
    
    AF_vals = acqf(X)

    T = len(test_x)
    N = len(n_vals)
    return test_x, AF_vals.reshape(N,T)



def plot_AF(N_points,
            AF,
            n_vals = torch.tensor([3,5,10]),
            fig_title=None,
            f_name=None,
            colour = 'b'):
    '''
    Plots the predicted intrinsic uncertainty (sigma^2_eps) and extrinsic uncertainty (sigma^2_f)
    of the Gaussian Process at a give iteration

    Inputs
    ------
        N_points: Tensor
            Number of grid points to make prediction
        AF: AcquisitionFunction
            An initalised Acquisition Function
        n_vals: int Tensor
            Chosen n value 
        fig_title: String Optional[Default=None] 
            Title of the figure if needed.
        f_name: String Optional[Default=None]
            If a filename string is supplied the generated figure will be automatically saved under
            that name. Format must be supplied in f_name i.e. .png or .eps
        new_point: bool Optional[Default=False]
            Highlights the selected candidate if set to true
    '''

    x,y = AF_output(N_points,
                    AF,
                    n_vals)


    with torch.no_grad():
        # Initialize plot
        plt.figure(figsize=(6,4))

        for i, n in enumerate(n_vals):
            plt.plot(x,y[i],label=f"n={n}")

        plt.xlabel('$x$')
        plt.ylabel('$AF(x)$')
        
        plt.legend()
        if fig_title is not None:
            plt.title(fig_title)
        

    if f_name is not None:
        plt.savefig(f_name, dpi=500,bbox_inches = "tight")
    
    plt.show()


from botorch.optim.optimize import optimize_acqf,optimize_acqf_mixed


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
            Returns the current - f_best as this is a minimisation problem
    '''

    _,f_best = optimise_acqf_get_candidate(strategy,
                                           bounds,
                                           num_restarts,
                                           raw_samples)

    if not MAXIMIZE:
        f_best = -f_best

    return f_best



from DES_acqfs import AEI_fq
from botorch.acquisition import PosteriorMean
# Define Acquisition Strategies
acq_strat_AEI = AEI_fq(sk_model['f'],maximize=MAXIMIZE)
acq_strat_SEI = PosteriorMean(sk_model['f'],maximize=MAXIMIZE) 
## Define Linear Cost Function
'''
Linear cost function is:

c(n) = 1/(ax+b)
where a and b are the linear coeffs
'''
a = 0.5
b = 2
lin_cost_func = InverseLinearCostModel([a,b])

## Define f^*

# f_str = train_y.min() #f* as current best
# f_str_AEI = f_best_acq(acq_strat_AEI,bounds=X_BOUNDS) #f* as convervative posteriorMin

# ## Define Acquisition Function

# f_best = f_str_SEI
# AEI = DES_EI(model_f=sk_model['f'],
#              model_eps=sk_model['eps'],
#              best_f= f_best,
#              cost_model=lin_cost_func,
#              maximize=False) #Define Cost aware and penalised EI

# State Bounds

bounds = torch.tensor([[0,3] * 1,
                        [1,100] * 1],
                        dtype=torch.double,
                        device=torch.device("cpu")) # Bounds of combined X and N space

X_BOUNDS = torch.tensor([[0] * 1,
                        [1] * 1],
                        dtype=torch.double,
                        device=torch.device("cpu")) # Bounds of combined X and N space

#Mixed Space Optim and EI plotter moved to stoch_kriging


# BO Optimisation Looop
n_dir = 'images/'
T = 15
AF_vals = []
f_bests =[]

class run_DES_exp_itr:

    def __init__(self,
                 AF,
                 f_best_strat,
                 model_call_func,
                 run_sim):

        r"""Single iteration of BODES
            Args:
                AF: AnalyticalAcquisition Function Type
                    candidate sets X will be)
                posterior_transform: A PosteriorTransform. If using a multi-output model,
                    a PosteriorTransform that transforms the multi-output posterior into a
                    single-output posterior is required.
                maximize: If True, consider the problem a maximization problem. Note
                    that if `maximize=False`, the posterior mean is negated. As a
                    consequence `optimize_acqf(PosteriorMean(gp, maximize=False))`
                    actually returns -1 * minimum of the posterior mean.
            """
        self.AF = AF
        self.f_best_strat = f_best_strat
        self.model_call_func = model_call_func
        self.run_sim = run_sim #y,sigma2 =func(x,n)
    
    def run_iter(self,model,train_x,train_n,train_y,train_sigma2):
        
        f_best = self.f_best_strat(model)

        #Initialise AF for candidate selection
        AF = self.AF(model_f=model['f'],
                     model_eps=model['eps'],
                     best_f=f_best, #TODO: curry this acqf so that cost_model and maximise are implemented beforehand
                     cost_model=lin_cost_func,
                     maximize=MAXIMIZE) #Define Cost aware and penalised EI

        ## Optimise AF and get candidates
        xn_new, _ = candidate_acq(AF,bounds)

        ## Update Dataset
        new_x = xn_new[0,0].reshape(1,1)
        new_n = xn_new[0,1].reshape(1,1)
        new_y, new_sigma2 = self.run_sim(new_x,new_n)

        train_x = torch.cat([train_x,new_x])
        train_n = torch.cat([train_n,new_n])
        train_y = torch.cat([train_y,new_y])
        train_sigma2 = torch.cat([train_sigma2,new_sigma2])

        ## Re-condtion model
        model = self.model_call_func(train_x,train_n,train_y,train_sigma2)

        return model, train_x, train_n, train_y, train_sigma2


def get_best_fs_AEI(model,maximise=MAXIMIZE,bounds=X_BOUNDS):

    acq_strat_AEI = AEI_fq(model['f'],maximize=maximise)
    f_best = f_best_acq(acq_strat_AEI,bounds=bounds)

    return f_best

BODES = run_DES_exp_itr(DES_EI,
                        get_best_fs_AEI,
                        get_stoch_kriging_model,
                        get_new_y_and_sigma)

def get_best_f_SEI(model,maximise=MAXIMIZE,bounds=X_BOUNDS):

    #Posterior MEaximise
    acq_strat_SEI = PosteriorMean(model['f'],maximize=maximise) 
    f_best = f_best_acq(acq_strat_SEI,bounds=bounds) #f* as posteriormin

    return f_best





#Iteration Funciton
sk_model,train_x,train_n,train_y,train_sigma2 = BODES.run_iter(sk_model,
                                                               train_x,
                                                               train_n,
                                                               train_y,
                                                               train_sigma2)

#Best f_acqf
f_best_SEI = get_best_f_SEI(sk_model)


for t in range(0,T): 
    print(f'Starting iter {t} of {T}....\n')

    # Acquire best point
    ## Current Method: Optimsie Conservative PostMean
    acq_strat_SEI = PosteriorMean(sk_model['f'],maximize=MAXIMIZE) 
    f_str_SEI = f_best_acq(acq_strat_SEI,bounds=X_BOUNDS) #f* as posteriormin
    f_best = f_str_SEI
    f_bests.append(f_best.unsqueeze(-1))
    print(f'Current $f^*$={f_best.item()}')
    #Conservative Best
    acq_strat_AEI = AEI_fq(sk_model['f'],maximize=MAXIMIZE)
    f_str_AEI = f_best_acq(acq_strat_AEI,bounds=X_BOUNDS)
    f_best = f_str_AEI
    

    #Initialise AF for candidate selection
    AEI = DES_EI(model_f=sk_model['f'],
                model_eps=sk_model['eps'],
                best_f=f_best,
                cost_model=lin_cost_func,
                maximize=MAXIMIZE) #Define Cost aware and penalised EI
    
    ## Optimise AF and get candidates
    xn_new,AF_val = candidate_acq(AEI,bounds)
    AF_vals.append(AF_val.unsqueeze(-1))

    if not t > 0:
        np_toggle = False
        title = f'Iteration {t}'
    else:
        np_toggle =True
        title = f'Iteration {t}|x={new_x.item()}|n={new_n.item()}'
    ##Plot current iteration and state output
    plot_iter_output(N_points,train_x,train_y,sk_model,title,n_dir+'GP_plot_' + str(t) + '.png',np_toggle)
    plot_in_uncertainty(N_points,sk_model,f'Noise|iter {t}',n_dir+'Noise_plot_' + str(t) + '.png','b')
    plot_ex_uncertainty(N_points,sk_model,f'Uncertainty|iter {t}',n_dir+'Uncer_plot_' + str(t) + '.png','r')
    plot_AF(N_points,AEI,torch.tensor([3,5,10,50]),f'AF| iter {t}',n_dir+'AF_plot_' + str(t) + '.png','g')
    plot_imporv(N_points,f_best,sk_model,f'improv|iter {t}',n_dir+'improv_plot_' + str(t) + '.png','y') 
    
    ## Update Dataset
    new_x = xn_new[0,0].reshape(1,1)
    new_n = xn_new[0,1].reshape(1,1)
    new_y, new_sigma2 = get_new_y_and_sigma(new_x,new_n)

    train_x = torch.cat([train_x,new_x])
    train_n = torch.cat([train_n,new_n])
    train_y = torch.cat([train_y,new_y])
    train_sigma2 = torch.cat([train_sigma2,new_sigma2])

    ## Re-condtion model
    sk_model = get_stoch_kriging_model(train_x,train_n,train_y,train_sigma2)
  

    ##Print Additional Informative plots
    
    print(f'Acquired Points:\n',
          f'x={new_x.item()}|n={new_n.item()}|y={new_y.item()}\n')
    
t= t+1
plot_iter_output(N_points,train_x,train_y,sk_model,f'Iteration {t}|x={new_x.item()}|n={new_n.item()}',n_dir+'GP_plot_' + str(t) + '.png',np_toggle)
plot_in_uncertainty(N_points,sk_model,f'Noise|iter {t}',n_dir+'Noise_plot_' + str(t) + '.png','b')
plot_ex_uncertainty(N_points,sk_model,f'Uncertainty|iter {t}',n_dir+'Uncer_plot_' + str(t) + '.png','r')
plot_AF(N_points,AEI,torch.tensor([3,5,10,50]),f'AF| iter {t}',n_dir+'AF_plot_' + str(t) + '.png','g')
plot_imporv(N_points,f_best,sk_model,f'improv|iter {t}',n_dir+'improv_plot_' + str(t) + '.png','y')

AF_vals = torch.cat(AF_vals)
f_bests = torch.cat(f_bests)
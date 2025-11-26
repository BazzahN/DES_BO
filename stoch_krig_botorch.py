import torch
from torch import Tensor
import matplotlib.pyplot as plt
from test_utils import test_function, noise_function,InverseLinearCostModel
from DES_acqfs import DES_EI, AEI_fq
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.acquisition import ExpectedImprovement
tkwargs = {
    "dtype": torch.double,# Datatype used by tensors
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"), # Declares the 'device' location where the Tenosrs will be stored
}

#GLOBALS
SIGMA2 = 5 #Scale of noise surface
MAXIMIZE= True #Sets problem to maximise test function or minimise test funciton

# Constants
k = 10 #number of samples points
n = 3 #flat number of replications

#Generate decision variables
#NOTE: Random train_x code moved to stoch_kriging
train_x = torch.linspace(0,1,k).to(**tkwargs)
train_n = torch.ones_like(train_x) * n

#Calculate sigma^2(x)
sigma2_vec = noise_function(train_x,SIGMA2).to(**tkwargs)

# Calculate sample variance
s2_vec = sigma2_vec / train_n
noise = torch.randn_like(train_x) * s2_vec

#Generate y values from latent function plus heteroscedastic Gaussian noise
train_y = test_function(train_x).to(**tkwargs) + noise

#Plot Test Function
N_points=500
test_x = torch.linspace(0,1,N_points).to(**tkwargs)
true_sig2 = noise_function(test_x,SIGMA2).to(**tkwargs)
true_sig = true_sig2.sqrt()
true_y = test_function(test_x)

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

#SNR - Signal to noise ratio
'''
Defined as the ratio SNR = \mu/\sigma or SNR = \mu^2/\sigma^2
'''
SNR = true_y**2/true_sig2 # Calculate change in SNR for this problem
f,ax = plt.subplots(1,1,figsize=(8,6))

ax.plot(test_x,SNR,label='SNR')
ax.plot(optim_sol,SNR[optim_idx],'k*',label='optmal point')
ax.set_xlabel('$x$')
ax.set_ylabel('SNR(x)')
ax.set_title('Signal to noise ratio of test function')
ax.legend()
plt.savefig('BODES_tp_SNR.png',dpi=500,bbox_inches = 'tight')
plt.show()


def get_new_y_and_sigma(x,n,sigma2 = SIGMA2):
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
    sigma2_vec = noise_function(x,sigma2).to(**tkwargs)

    # Calculate sample variance
    s2_vec = sigma2_vec / n
    noise = torch.randn_like(x) * s2_vec
    #Generate y values from latent function plus heteroscedastic Gaussian noise
    train_y = test_function(x).to(**tkwargs) + noise

    return train_y, sigma2_vec


train_x = train_x.reshape(k,1)
train_n = train_n.reshape(k,1)
train_y, train_sigma2 = get_new_y_and_sigma(train_x,train_n)


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
        lower, upper = posterior_distb.confidence_region()
        # Plot training data as black stars
        ax.plot(train_x.numpy(), train_y.numpy(), 'k*',label='Evaluations')
        # Plot predictive means as blue line
        ax.plot(test_x.numpy(), posterior_distb.mean.numpy(), 'b',label='Mean')
        ax.plot(test_x,true_y,'r',label='True')
        # Shade between the lower and upper confidence bounds
        ax.fill_between(test_x.numpy(), lower.numpy(), upper.numpy(), alpha=0.5,label='CI')
        ax.set_xlabel('$x$')
        ax.set_ylabel('$y$')
        
        ax.legend()
        if fig_title is not None:
            ax.set_title(fig_title)
        

    if f_name is not None:
        plt.savefig(f_name, dpi=500,bbox_inches = "tight")

    plt.show()        


#Constants
N_points = 100 #Number of test points

#Generate test points
test_x = torch.linspace(0,1,N_points).to(**tkwargs)
true_y = test_function(test_x)
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
    true_y = test_function(test_x)
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

def plot_AF(N_points,
            AF,
            n_val = 3,
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
        n_val: int Tensor
            Chosen n value 
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
    test_n = torch.ones_like(test_x) * n_val
    X_1 = torch.cat([test_x.unsqueeze(-1),test_n.unsqueeze(-1)],axis=1).unsqueeze(1)
    
    AF_vals = AF(X_1)


    with torch.no_grad():
            # Initialize plot
            f, ax = plt.subplots(1, 1, figsize=(8, 6))

            ax.plot(test_x,AF_vals,colour,label='$AF(x)$')

            ax.set_xlabel('$x$')
            ax.set_ylabel('$AF(x)$')
            
            ax.legend()
            if fig_title is not None:
                ax.set_title(fig_title)
        

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
a = 0.1
b = 1.5
lin_cost_func = InverseLinearCostModel([a,b])

## Define f^*

# f_str = train_y.min() #f* as current best
# f_str_AEI = f_best_acq(acq_strat_AEI,bounds=x_bounds) #f* as convervative posteriorMin

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

x_bounds = torch.tensor([[0] * 1,
                        [1] * 1],
                        dtype=torch.double,
                        device=torch.device("cpu")) # Bounds of combined X and N space

#Mixed Space Optim and EI plotter moved to stoch_kriging


# BO Optimisation Looop
n_dir = 'images/'
T = 10
AF_vals = []
f_bests =[]


for t in range(0,T): 
    print(f'Starting iter {t} of {T}....\n')

    # Acquire best point
    ## Current Method: Optimsie Conservative PostMean
    # acq_strat_SEI = PosteriorMean(sk_model['f'],maximize=MAXIMIZE) 
    # f_str_SEI = f_best_acq(acq_strat_SEI,bounds=x_bounds) #f* as posteriormin
    acq_strat_AEI = AEI_fq(sk_model['f'],maximize=MAXIMIZE)
    f_str_AEI = f_best_acq(acq_strat_AEI,bounds=x_bounds)
    f_best = f_str_AEI
    f_bests.append(f_best.unsqueeze(-1))
    print(f'Current $f^*$={f_best.item()}')

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
    plot_AF(N_points,AEI,3,f'AF| iter {t}',n_dir+'AF_plot_' + str(t) + '.png','g')
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
plot_AF(N_points,AEI,3,f'AF| iter {t}',n_dir+'AF_plot_' + str(t) + '.png','g')
plot_imporv(N_points,f_best,sk_model,f'improv|iter {t}',n_dir+'improv_plot_' + str(t) + '.png','y')

AF_vals = torch.cat(AF_vals)
f_bests = torch.cat(f_bests)
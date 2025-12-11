import torch
from torch import Tensor
import matplotlib.pyplot as plt
from test_utils import test_function,test_function_2, heteroscedastic_noise,flat_noise,InverseLinearCostModel
from DES_acqfs import DES_EI, AEI_fq
from DES_IG_acqf import BODES_IG
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
SIGMA2 = 1 #Scale of noise surface
PHI = 1.5 #Shift of Heteroscedastic noise surface
MAXIMIZE= True #Sets problem to maximise test function or minimise test funciton


class Target_function:

    def __init__(self,
                 test_function,
                 noise_function,
                 phi=PHI,
                 theta=SIGMA2):
        
        self.test_function = test_function
        self.noise_function = noise_function
        self.phi = phi
        self.theta = theta

    def eval_target_noisy(self,test_x,test_n):
        
        sigma_2 = self.noise_function(test_x,self.theta,self.phi).to(**tkwargs)
        # Calculate sample variance
        s2_vec = sigma_2 / test_n
        noise = torch.randn_like(test_x) * s2_vec        

        y_evals = self.test_function(test_x).to(**tkwargs) + noise

        return y_evals,sigma_2
    
    def eval_target_true(self,test_x):

        return self.test_function(test_x).to(**tkwargs) 

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


def get_k_inital_evals(k,n,target_function):
    '''
    Gets k inital observations for a flat n replications each  
    '''
    train_x = torch.linspace(0.1,1,k).reshape(k,1).to(**tkwargs)
    train_n = torch.ones_like(train_x) * n


    #Generate y values from latent function plus heteroscedastic Gaussian noise
    train_y, train_sigma2 = target_function.eval_target_noisy(train_x,train_n)

    return train_x,train_n,train_y,train_sigma2



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
    true_y = test_function(test_x)
    preds = model['f'].posterior(test_x, observation_noise=False)

    # Acquisition function output
    x, y = AF_output(N_points, acqf, n_vals)

    with torch.no_grad():
        # Initialize plot
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))

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
            bbox_to_anchor=(1.02, 0.7),
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

    x_best,f_best = optimise_acqf_get_candidate(strategy,
                                                bounds,
                                                num_restarts,
                                                raw_samples)

    if not MAXIMIZE:
        f_best = -f_best

    return x_best,f_best


from DES_acqfs import AEI_fq
from botorch.acquisition import PosteriorMean,ExpectedImprovement

'''
Linear cost function is:

c(n) = 1/(ax+b)
where a and b are the linear coeffs
'''
a = 0.5
b = 1
lin_cost_func = InverseLinearCostModel([a,b])

# State Bounds

BOUNDS = torch.tensor([[0,3] * 1,
                        [1,100] * 1],
                        dtype=torch.double,
                        device=torch.device("cpu")) # Bounds of combined X and N space

X_BOUNDS = torch.tensor([[0] * 1,
                        [1] * 1],
                        dtype=torch.double,
                        device=torch.device("cpu")) # Bounds of combined X and N space

class BODES_loop_initialiser:

    def __init__(self,
                 k,
                 n,
                 target_function_class):
        
        self.k = k
        self.n = n
        self.target_function_class = target_function_class

    def initialise(self):

        train_x,train_n,train_y,train_sigma2 = get_k_inital_evals(self.k,
                                                                  self.n,
                                                                  self.target_function_class)

        model = get_stoch_kriging_model(train_x,train_n,train_y,train_sigma2)

        return model,train_x,train_n,train_y,train_sigma2


class run_vanilla_exp_itr:

    def __init__(self,
                 n, #Constant selection of n
                 AF, #Expected Improvement
                 f_best_strat, #SEI
                 model_call_func, #same with modifications
                 target_function,
                 bounds=X_BOUNDS):

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
        self.n = torch.tensor([n]).reshape(1,1)
        self.f_best_strat = f_best_strat
        self.model_call_func = model_call_func
        self.run_sim = target_function.eval_target_noisy #y,sigma2 =func(x,n)
        self.bounds = bounds

    def run_iter(self,model,train_x,train_n,train_y,train_sigma2):
        
        f_best = self.f_best_strat(model)

        #Initialise AF for candidate selection
        AF = self.AF(model=model['f'],
                     best_f=f_best[0], #TODO: curry this acqf so that cost_model and maximise are implemented beforehand
                     maximize=MAXIMIZE) #Define Cost aware and penalised EI

        ## Optimise AF and get candidates
        new_x, _ = candidate_acq(AF,self.bounds)

        ## Update Dataset (constant n)
        new_y, new_sigma2 = self.run_sim(new_x,self.n)

        train_x = torch.cat([train_x,new_x])
        train_n = torch.cat([train_n,self.n])
        train_y = torch.cat([train_y,new_y])
        train_sigma2 = torch.cat([train_sigma2,new_sigma2])

        ## Re-condtion model
        model = self.model_call_func(train_x,train_n,train_y,train_sigma2)

        return model, train_x, train_n, train_y, train_sigma2

class run_DES_exp_itr:

    def __init__(self,
                 AF,
                 f_best_strat,
                 model_call_func,
                 target_function,
                 bounds=BOUNDS):

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
        self.run_sim = target_function.eval_target_noisy #y,sigma2 =func(x,n)
        self.bounds= bounds

    def run_iter(self,model,train_x,train_n,train_y,train_sigma2):
        
        f_best = self.f_best_strat(model)

        #Initialise AF for candidate selection
        AF = self.AF(model = model,
                     best_f=f_best, #TODO: curry this acqf so that cost_model and maximise are implemented beforehand
                     cost_model=lin_cost_func,
                     maximize=MAXIMIZE) #Define Cost aware and penalised EI

        ## Optimise AF and get candidates
        xn_new, _ = candidate_acq(AF,self.bounds)

        # The selected n point is rounded to the nearest integer
        xn_new[0,1] = xn_new[0,1].round(decimals=0)
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


class run_IG_exp_itr:

    def __init__(self,
                 AF,
                 model_call_func,
                 target_function,
                 bounds=BOUNDS):

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
        self.model_call_func = model_call_func
        self.run_sim = target_function.eval_target_noisy #y,sigma2 =func(x,n)
        self.bounds= bounds
        self.discrete_space = torch.linspace(bounds[0,0],bounds[1,0],10).unsqueeze(1)
    def run_iter(self,model,train_x,train_n,train_y,train_sigma2):
        
        #Initialise AF for candidate selection
        AF = self.AF(model = model,
                     cost_model=lin_cost_func,
                     candidate_set = self.discrete_space,
                     maximize=MAXIMIZE) #Define Cost aware and penalised EI

        ## Optimise AF and get candidates
        xn_new, _ = candidate_acq(AF,self.bounds)

        # The selected n point is rounded to the nearest integer
        xn_new[0,1] = xn_new[0,1].round(decimals=0)
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
    _,f_best = f_best_acq(acq_strat_AEI,bounds=bounds)

    return f_best

def get_best_f_SEI(model,maximise=MAXIMIZE,bounds=X_BOUNDS):

    #Posterior MEaximise
    acq_strat_SEI = PosteriorMean(model['f'],maximize=maximise) 
    x_best,f_best = f_best_acq(acq_strat_SEI,bounds=bounds) #f* as posteriormin

    return x_best,f_best


# def trace(text):
#     '''
#     Helper Function
#     '''

class experiment_handler:

    def __init__(self,
                 loop_initialiser_func,
                 BO_handler):
        
        self.BO_handler = BO_handler.run_iter
        self.initialise = loop_initialiser_func.initialise
    
    
    def run_T_BO_iters(self,T,seed):
        #Set Experiment seed
        torch.manual_seed(seed)

        #Obtain model and initial evaluations
        sk_model,train_x,train_n,train_y,train_sigma2 = self.initialise()

        #Best f_acqf
        x_strs, f_strs = get_best_f_SEI(sk_model)

        f_strs = f_strs.reshape(1,1)


        for t in range(0,T):
            print(f'Starting iter {t} of {T}....\n')
            #Iteration Funciton
            sk_model,train_x,train_n,train_y,train_sigma2 = self.BO_handler(sk_model,
                                                                            train_x,
                                                                            train_n,
                                                                            train_y,
                                                                            train_sigma2)   
            #Best f_acqf
            x_best, f_best_SEI = get_best_f_SEI(sk_model)
            #Append best evals to the list
            x_strs = torch.cat([x_strs,x_best])
            f_strs = torch.cat([f_strs,f_best_SEI.reshape(1,1)])
            
        return train_x,train_n,train_y,train_sigma2,x_strs,f_strs
    
    def run_MT_BO_macros(self,M,T,master_seed = 12345):
        torch.manual_seed(master_seed)
        xs = []
        ns =[]
        ys =[]
        sigma2s = []
        x_strs = []
        f_strs =[]
        
        seeds = torch.randint(1000,100000,(M,))

        #TODO seed splitter for m replications
        for m in range(0,M):
            seed = seeds[m].item()
            print(f'Starting macroreplication {m} of {M}....\n')
            print(f'\nFor seed {seed}\n')
            train_x,train_n,train_y,train_sigma2,x_str,f_str =self.run_T_BO_iters(T,seed)

            #Append all data to lists

            xs.append(train_x)
            ns.append(train_n)
            ys.append(train_y)
            sigma2s.append(train_sigma2)
            x_strs.append(x_str)
            f_strs.append(f_str)

        xs = torch.stack(xs)
        ns = torch.stack(ns)
        ys = torch.stack(ys)
        sigma2s = torch.stack(sigma2s)
        x_strs = torch.stack(x_strs)
        f_strs = torch.stack(f_strs)

        return xs,ns,ys,sigma2s,x_strs,f_strs




#Experimental Parameters
T = 10 #Number of iterations
M = 25 #Number of MacroReplications

# Constants
k = 5 #number of samples points
n = 5 #flat number of replications

#Initalise Target Function For Experiments
target = Target_function(test_function_2,
                         heteroscedastic_noise)

noise_function = heteroscedastic_noise



N_points=500
test_x = torch.linspace(0,1,N_points).to(**tkwargs)
true_sig2 = noise_function(test_x,SIGMA2,PHI).to(**tkwargs)
true_sig = true_sig2.sqrt()
true_y = target.eval_target_true(test_x)


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

plt.suptitle('True Function with Heteroscedastic Variance Function')
plt.savefig('BODES_test_problem_S1.png',dpi=500,bbox_inches = 'tight')
plt.show()
BODES = run_DES_exp_itr(DES_EI,
                        get_best_fs_AEI,
                        get_stoch_kriging_model,
                        target)

BODES_LI = BODES_loop_initialiser(k,
                                  n,
                                  target)



# experiments = experiment_handler(BODES_LI,
#                                  BODES)



# out = experiments.run_MT_BO_macros(M,T)

# dir = 'Data/'
# names = ['train_x','train_n','train_y','train_sigma2','x_strs','f_strs']

# exp = 'AEI_'

# #Save as tensors
# for name,d in zip(names,out):

#     fname = dir + exp + name + '.pt'
#     torch.save(d,dir + exp + name + '.pt')




# VAN_BO = run_vanilla_exp_itr(5,
#                              ExpectedImprovement,
#                              get_best_f_SEI,
#                              get_stoch_kriging_model,
#                              target)

# van_experiments = experiment_handler(BODES_LI,VAN_BO)

# out = van_experiments.run_MT_BO_macros(M,T)


# exp = 'VANIL_'
# #Save as tensors
# for name,d in zip(names,out):

#     fname = dir + exp + name + '.pt'
#     torch.save(d,dir + exp + name + '.pt')


# BIG = run_IG_exp_itr(BODES_IG,get_stoch_kriging_model,target)

# big_experiments = experiment_handler(BODES_LI,BIG)

# out = big_experiments.run_MT_BO_macros(M,T)


# exp = 'BIG_'
# #Save as tensors
# for name,d in zip(names,out):

#     fname = dir + exp + name + '.pt'
#     torch.save(d,dir + exp + name + '.pt')
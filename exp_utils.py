import torch
from torch import Tensor
import matplotlib.pyplot as plt
from test_utils import TEST_FUNCTION_DIAL,NOISE_FUNCTION_DIAL,InverseLinearCostModel
from DES_acqfs import DES_EI, AEI_fq,BODES_IG
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
MAXIMIZE= True #Sets problem to maximise test function or minimise test funciton

class output_handler:

    def __init__(self):
        
        #Sets paramaters so that standardisation does nothing
        self.sig_std = 1
        self.mu_std = 0

    def standardise_and_update(self,y):
        '''
        Standardises the inputed data to N(0,1) distribution
        and updates the transformation paramaters sigma_std and mu_std
        
        Inputs
        ------
        train_y: kx1
            The training data to be standardised.
        '''
        #Calculate mean and std of data
        self.mu_std = y.mean()
        self.sig_std = y.std()

        return (y-self.mu_std)/self.sig_std
    
    def standardise(self,y):
        '''
            Standardises the inputed data to N(0,1) distribution

            Inputs
            ------
            train_y: kx1
                The training data to be standardised.
        '''

        return (y-self.mu_std)/self.sig_std

    def unstandardise(self,y_std):
        '''
            Reverts standardised input back to its previous state
        
        '''
        return y_std*self.sig_std + self.mu_std

    def log_transform(self,sigma2):

        return sigma2.log()
    
    def inv_log_transform(self,log_sigma2,log_sigma2_var):
        """
        Inverse transformation on GP output.
        
        :param log_sigma2: Description
        :param log_sigma2_var: Description
        """
        return torch.exp(log_sigma2 + 0.5*log_sigma2_var)
    
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
    # Initialise output_handler objects
    y_handler = output_handler()
    sigma2_handler = output_handler()

    print(f"train_x:{train_x}")
    print(f"train_y: {train_y}")
    print(f"train_sig2:{sigma2_hat}")
    #Standardise y input
    train_y_std = y_handler.standardise_and_update(train_y)
    sig_scale = 1 / y_handler.sig_std
    s2 = (sigma2_hat / train_n).view(-1, 1) * sig_scale #Sample variance transform
    
    #Fit main Model
    gp = SingleTaskGP(train_x,
                      train_y_std,
                      train_Yvar=s2,
                      outcome_transform=None,
                      #add_noise=True,
                      )

    mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
    fit_gpytorch_mll(mll)

    #Fit Noise Model
    ##Standardise sigma2 hat

    #Log sigma2 to ensure nonnegative values
    print(f"sigma2: {sigma2_hat}")
    sigma2_hat_log = sigma2_hat.log()

    sigma2_hat_std = sigma2_handler.standardise_and_update(sigma2_hat_log)

    
    gp_noise = SingleTaskGP(train_x,
                            sigma2_hat_std,
                            train_Yvar=torch.full_like(sigma2_hat,1e-16),
                            outcome_transform=None,
                            )
    mll = ExactMarginalLogLikelihood(gp_noise.likelihood,gp_noise)
    fit_gpytorch_mll(mll)

    #Puts model into dictionary
    model = {'f':gp,'eps':gp_noise}

    #Puts transformers into dictionary
    out_transform = {'f':y_handler,'eps':sigma2_handler}
    return model, out_transform

from HGP_utils import HeteroscedasticBOModel,HeteroscedasticELBO,fit_vihgp_elbo

class VI_HGP():

    def __init__(self,n_u, #Int: number of inducing points
                 iters, #Int: number of iterations
                 standardise=False,#bool: If true then standardise output
                 verbose=False
                ):
        """
        
        Inputs
        
        """
        self.n_u = n_u
        self.iters = iters
        self.standardise = standardise
        self.verbose = verbose

    def get_VI_HGP_model(self,train_x,train_n,train_y,sigma2_hat):
        '''
        Constructs and conditions on the dataset D_t = (train_x,train_y) the 
        Variational Heteroscedastic GP. Train_n and sigma2_hat are not needed 

        Inputs
        ------
        train_x: Tensor
            A tensor of evaluated locations
        train_y: Tensor
            A tensor of evaluations
        train_n: Tensor (DUMMY)
            Input data goes nowhere as not requried for the HGP
        train_y: Tensor
            A tensor of evaluation data
        sigma2_hat: Tensor (DUMMY)
            Input data goes nowhere as it is not requried for the HGP
            

        Returns
        -------
        model:HeteroscedasticBOModel
            A dictionary of the stochastic kriging Gaussian process model components
        
        '''
        # Initialise output_handler objects
        out_transform = output_handler()

        # TODO Standardise Step on y input
        
        if self.standardise:
            train_y = out_transform.standardise_and_update(train_y)

        inducing_init = train_x[torch.linspace(0, train_x.size(0) - 1, steps=self.n_u).long()]
        #Fit main Model
        hgp_model = HeteroscedasticBOModel(train_x,
                                            train_y,
                                            inducing_init.clone())

        elbo = HeteroscedasticELBO(hgp_model.likelihood, hgp_model.model)
        hgp_model,_ = fit_vihgp_elbo(model=hgp_model,
                                     elbo=elbo,
                                     iters = self.iters,
                                     verbose=self.verbose)


        #NOTE: In contrast, to get stoch_kriging this just outputs a single class for the model and transformer
        return hgp_model, out_transform


def get_k_inital_evals(k,n,target_function):
    '''
    Gets k inital observations for a flat n replications each  
    '''
    train_x = torch.linspace(0.1,1,k).reshape(k,1).to(**tkwargs)
    train_n = torch.ones_like(train_x) * n


    #Generate y values from latent function plus heteroscedastic Gaussian noise
    train_y, train_sigma2 = target_function.eval_target_noisy(train_x,train_n)

    return train_x,train_n,train_y,train_sigma2


from botorch.optim.optimize import optimize_acqf


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

        model,output_handle = get_stoch_kriging_model(train_x,train_n,train_y,train_sigma2)

        return model,train_x,train_n,train_y,train_sigma2,output_handle

#TODO: Note too self. In future use an abstract baseclass when using the same model three times
class run_vanilla_exp_itr:

    def __init__(self,
                 n, #Constant selection of n
                 AF, #Expected Improvement
                 f_best_strat, #SEI
                 model_call_func, #same with modifications
                 cost_function, #Does nothing
                 bounds):

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
        # self.run_sim = target_function.eval_target_noisy #y,sigma2 =func(x,n)
        self.bounds = bounds

    def run_iter(self,model,train_x,train_n,train_y,train_sigma2,target_function,output_transform): #Here OT does nothing
        
        f_best = self.f_best_strat(model,self.bounds)

        #Initialise AF for candidate selection
        AF = self.AF(model=model['f'],
                     best_f=f_best[0], #TODO: curry this acqf so that cost_model and maximise are implemented beforehand
                     maximize=MAXIMIZE) #Define Cost aware and penalised EI

        ## Optimise AF and get candidates
        new_x, _ = candidate_acq(AF,self.bounds[:,0].view(-1,1))
        #Removes number of replication bounds from vanilla

        ## Update Dataset (constant n)
        _,new_y, new_sigma2 = target_function.eval_target_noisy(new_x,self.n)

        train_x = torch.cat([train_x,new_x])
        train_n = torch.cat([train_n,self.n])
        train_y = torch.cat([train_y,new_y])
        train_sigma2 = torch.cat([train_sigma2,new_sigma2])

        ## Re-condtion model
        model,output_handle = self.model_call_func(train_x,train_n,train_y,train_sigma2)

        return model, train_x, train_n, train_y, train_sigma2,output_handle

class run_DES_exp_itr:

    def __init__(self,
                 n,
                 AF,
                 f_best_strat,
                 model_call_func,
                #  target_function,
                 cost_function,
                 bounds):

        r"""Single iteration of BODES
            Args:
                n: int
                does nothing
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
        # self.run_sim = target_function.eval_target_noisy #y,sigma2 =func(x,n)
        self.cost_function = cost_function
        self.bounds= bounds


    def run_iter(self,model,train_x,train_n,train_y,train_sigma2,target_function,output_transform):
        
        f_best = self.f_best_strat(model,output_transform,self.bounds)

        #Initialise AF for candidate selection
        AF = self.AF(model = model,
                     best_f=f_best, #TODO: curry this acqf so that cost_model and maximise are implemented beforehand
                     cost_model=self.cost_function,
                     output_transform=output_transform,
                     maximize=MAXIMIZE) #Define Cost aware and penalised EI

        ## Optimise AF and get candidates
        xn_new, _ = candidate_acq(AF,self.bounds)

        # The selected n point is rounded to the nearest integer
        xn_new[0,1] = xn_new[0,1].round(decimals=0)
        ## Update Dataset
        new_x = xn_new[0,0].reshape(1,1)
        new_n = xn_new[0,1].reshape(1,1)
        _,new_y, new_sigma2 = target_function.eval_target_noisy(new_x,new_n)

        train_x = torch.cat([train_x,new_x])
        train_n = torch.cat([train_n,new_n])
        train_y = torch.cat([train_y,new_y])
        train_sigma2 = torch.cat([train_sigma2,new_sigma2])

        ## Re-condtion model
        model,output_handle = self.model_call_func(train_x,train_n,train_y,train_sigma2)



        return model, train_x, train_n, train_y, train_sigma2,output_handle

class run_IG_exp_itr:

    def __init__(self,
                 n,
                 AF,
                 model_call_func,
                 cost_function,
                 bounds,
                 num_mv_samples = 10,
                 set_size = 10,
                 ):

        r"""Single iteration of BODES
            Args:
                n: int
                    does nothing
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
        self.bounds= bounds
        self.discrete_space = torch.linspace(bounds[0,0],bounds[1,0],set_size).unsqueeze(1)
        self.num_mv_samples = num_mv_samples
        self.cost_function = cost_function
    
    def run_iter(self,model,train_x,train_n,train_y,train_sigma2,target_function,output_transform):
        
        #Initialise AF for candidate selection
        AF = self.AF(model = model,
                     cost_model=self.cost_function,
                     output_transform= output_transform,
                     num_mv_samples = self.num_mv_samples,
                     candidate_set = self.discrete_space,
                     maximize=MAXIMIZE) #Define Cost aware and penalised EI

        ## Optimise AF and get candidates
        xn_new, acq_val = candidate_acq(AF,self.bounds)
        print(f"ACQF VAL:{acq_val.item()}")
        # The selected n point is rounded to the nearest integer
        xn_new[0,1] = xn_new[0,1].round(decimals=0)
        ## Update Dataset
        new_x = xn_new[0,0].reshape(1,1)
        new_n = xn_new[0,1].reshape(1,1)
        _,new_y, new_sigma2 = target_function.eval_target_noisy(new_x,new_n)
        
        train_x = torch.cat([train_x,new_x])
        train_n = torch.cat([train_n,new_n])
        train_y = torch.cat([train_y,new_y])
        train_sigma2 = torch.cat([train_sigma2,new_sigma2])

        ## Re-condtion model
        model,output_handle = self.model_call_func(train_x,train_n,train_y,train_sigma2)

        return model, train_x, train_n, train_y, train_sigma2,output_handle



def get_best_f_AEI(model,output_transform,bounds,maximise=MAXIMIZE):

    acq_strat_AEI = AEI_fq(model['f'],output_transform,maximize=maximise)
    _,f_best = f_best_acq(acq_strat_AEI,bounds=bounds[:,0].view(-1,1))
    print(f"f_best={f_best}\n")
    return f_best

#TODO: Modify for generality to allow HGP interface
def get_best_f_SEI(model,bounds,maximise=MAXIMIZE,output_transform=None):
    """
    Docstring for get_best_f_SEI
    
    :param model: Description
    :param bounds: Description
    :param maximise: Description
    :param output_transform: Description
    """
    #Posterior MEaximise
    acq_strat_SEI = PosteriorMean(model['f'],maximize=maximise) 
    x_best,f_best = f_best_acq(acq_strat_SEI,bounds=bounds[:,0].view(-1,1)) #f* as posteriormin

    #TODO transform output here
    f_best = f_best.reshape(1,1)
    if output_transform is not None:

        return x_best,output_transform['f'].unstandardise(f_best)
    else:
        return x_best,f_best
    

class experiment_handler:

    def __init__(self, 
                 target,
                 BO_handler):
        
        self.bounds = BO_handler.bounds
        self.BO_handler = BO_handler.run_iter
        self.target = target
    
    
    def run_T_BO_iters(self,T,
                            train_x,
                            train_n,
                            train_y,
                            train_sigma2,
                            rng_state):
        
        #Set Experiment seed
        rng_state = rng_state.byte()
        self.target.update_rng_state(rng_state)
        #Obtain model and initial evaluations

        #NOTE: Modify experiment handler to allow for model selection 
        sk_model, output_handle = get_stoch_kriging_model(train_x,train_n,train_y,train_sigma2)
        
        #Best f_acqf
        x_strs, f_strs = get_best_f_SEI(sk_model,bounds=self.bounds,output_transform=output_handle)

  
        for t in range(0,T):
            print(f'Starting iter {t} of {T}....\n',flush=True)
            #sys.stdout.write(f'Starting iter {t} of {T}....\n')
            #sys.stdout.flush()
            #Iteration Funciton
            sk_model,train_x,train_n,train_y,train_sigma2,output_handle = self.BO_handler(sk_model,
                                                                                            train_x,
                                                                                            train_n,
                                                                                            train_y,
                                                                                            train_sigma2,
                                                                                            self.target,
                                                                                            output_handle)   
            #Best f_acqf
            x_best, f_best_SEI = get_best_f_SEI(sk_model,bounds=self.bounds,output_transform=output_handle)
          
            #Append best evals to the list
            x_strs = torch.cat([x_strs,x_best])
            f_strs = torch.cat([f_strs,f_best_SEI])
            
        return train_x,train_n,train_y,train_sigma2,x_strs,f_strs
    
    def run_MT_BO_macros(self,M,
                              T,
                              train_x,
                              train_n,
                              train_y,
                              train_sigma2,
                              rngs,
                              ):

        xs = []
        ns =[]
        ys =[]
        sigma2s = []
        x_strs = []
        f_strs =[]
        #TODO seed splitter for m replications
        for m in range(0,M):
            print(f'[SIM]Starting macroreplication {m} of {M}....\n',flush=True)

            out_x,out_n,out_y,out_sigma2,x_str,f_str =self.run_T_BO_iters(T,
                                                                                  train_x[m],
                                                                                  train_n[m],
                                                                                  train_y[m],
                                                                                  train_sigma2[m],
                                                                                  rngs[m])

            #Append all data to lists
            xs.append(out_x)
            ns.append(out_n)
            ys.append(out_y)
            sigma2s.append(out_sigma2)
            x_strs.append(x_str)
            f_strs.append(f_str)
        
        xs = torch.stack(xs)
        ns = torch.stack(ns)
        ys = torch.stack(ys)
        sigma2s = torch.stack(sigma2s)
        x_strs = torch.stack(x_strs)
        f_strs = torch.stack(f_strs)

        return xs,ns,ys,sigma2s,x_strs,f_strs



#Pre-construct Experiments
from functools import partial

VANILLA = partial(run_vanilla_exp_itr,
                  AF=ExpectedImprovement,
                  f_best_strat=get_best_f_SEI,
                  model_call_func=get_stoch_kriging_model)

AEI = partial(run_DES_exp_itr,
                AF=DES_EI,
                f_best_strat=get_best_f_AEI,
                model_call_func=get_stoch_kriging_model)

IG = partial(run_IG_exp_itr,
            AF=BODES_IG,
            model_call_func=get_stoch_kriging_model)


EXPERIMENTS = {'vanilla':VANILLA,
               'AEI': AEI,
               'IG':IG}
import torch
from exp_utils import output_handler
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from gpytorch.mlls import ExactMarginalLogLikelihood


"""
This script will contain all of the optional models used in BODES. At present this is the two examples of 
heteroscedastic gaussian process

1. get_stoch_kriging_model - Stochastic Kriging Gaussian Process
2. get_VI_HGP_model - Variational Inference Heteroscedastic Gaussian Process
"""

def get_stoch_kriging_model(train_x,train_n,train_y,sigma2_hat):
    '''
    Constructs and conditions on the dataset D_t = (train_x,train_n,train_y,sigma2_hat) the 
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


#TODO Curry with partialtools to prefil with iters and n_u
"""
To make sure the model fits in with the code I already have, I will probably
have to keep get_VI_HGP_model function and its arguments exposed in a way
to interact with the rest of my code. 

"""


def get_VI_HGP_model(train_x,train_n,train_y,sigma2_hat,iters,n_u):
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
    model:dict('f' = latent model, 'eps' = noise model)
        A dictionary of the stochastic kriging Gaussian process model components
    
    '''
    # Initialise output_handler objects
    y_handler = output_handler()
    sigma2_handler = output_handler()

    # TODO Standardise Step on y input
    # train_y_std = y_handler.standardise_and_update(train_y)
    # sig_scale = 1 / y_handler.sig_std
    # s2 = (sigma2_hat / train_n).view(-1, 1) * sig_scale #Sample variance transform
    inducing_init = train_x[torch.linspace(0, train_x.size(0) - 1, steps=n_u).long()]
    #Fit main Model
    hgp_model = HeteroscedasticBOModel(train_x,
                                       train_y,
                                       inducing_init.clone())

    elbo = HeteroscedasticELBO(hgp_model.likelihood, hgp_model.model)
    hgp_model,_ = fit_vihgp_elbo(hgp_model,elbo)

    #Puts model into dictionary
    #model = {'f':gp,'eps':gp_noise}
    model = hgp_model
    #Puts transformers into dictionary
    out_transform = {'f':y_handler,'eps':sigma2_handler}
    return model, out_transform
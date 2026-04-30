from pathlib import Path
from DES_acqfs import _inverse_log_transform,_transform_GP
import torch as st
import json
import matplotlib.pyplot as plt

DPI = 500
FIGSIZE = (8,12) #Global figsize for variable
LOG_FNAME = "/Log"

TKWARGS = {
    "dtype": st.double,# Datatype used by tensors
    "device": st.device("cuda" if st.cuda.is_available() else "cpu"), # Declares the 'device' location where the Tenosrs will be stored
}
#TODO Double check if this folder is uppercase
# #TODO UTILS

def get_files(exp_name,dir_name,file_names,add=""):
	indir = Path(exp_name + f"/{dir_name}")

	data = {}
	for file_name in file_names:

		load_in = st.load(indir /  f"{add}{file_name}.pt").to(**TKWARGS)
		data[file_name] = load_in
	return data

#TODO HYPERPARAMATER HANDLING FUNCTIONS
def export_hyperparamaters(path,acqf_name,run_params,hyperparamaters):
    """
    Exports hyperparamaters in json format under the chosen name and saves in the chosen path

    paramaters:
        path: path
        name: string
        hyperparamaters: dict
            A dictionary of hyperparamaters
    
    """

    outdir = Path(path + LOG_FNAME +"/hyperparamaters")
    if run_params is not None:
        f_name = f"{acqf_name}_hyperparams_{run_params['m']}_{run_params['t']}.json"
    else:
        f_name = f"{acqf_name}__hyperparams.json"
    
    outdir = outdir / f_name
    with open(outdir,"w") as outdir:
        json.dump(hyperparamaters,outdir)
  

def get_hypers_vihgp(model):
    """
    Subprocess for get_hyperparamaters for the VI-HGP.
    This returns a dictionary including
    - The kernel function outputscales (if included)
    - The kernel function lengthscales 
    - The mean function constant
    - Inducing point means
    - Indcuing point locations
    """
    model = model.model

    #Extract Hyperparamaters
    with st.no_grad():

        try:
            taus = model.covar_module.outputscale.tolist()
        except:
            taus=[1,1]

        opt_scales_dict = {'tau_1':taus[0],'tau_2':taus[1]}
        #Extract lengthscale

        try:
            lnth_scales = model.covar_module.base_kernel.lengthscale.flatten().tolist()
        except:
            lnth_scales = model.covar_module.lengthscale.flatten().tolist()
        lnth_scales_dict = {'l_1':lnth_scales[0],'l_2':lnth_scales[1]}

        #Extract Means
        #If ZeroMean used instead

        try:
            means = model.mean_module.constant.detach().tolist()
        except:
            means = [0,0]

        means_dict = {'mu_1':means[0],'mu_2':means[1]}

        #Extract inducing Means
        u_means = model.variational_strategy.base_variational_strategy._variational_distribution.variational_mean.detach().tolist()
        u_means_dict = {'mu_u_1':u_means[0],'mu_u_2':u_means[1]}

        #Extract Inducing Points
        u_points = model.variational_strategy.base_variational_strategy.inducing_points.flatten().detach().tolist()
        u_points_dict = {'u':u_points}

    #Compile hyperparameters to dictionary
    #- Outputscales - if switched on
    #- Lengthscales 
    #- Constant Means - if switched on else 0
    #- Inducing point means 
    # Inducing point locations

    hyperparameter_dict = dict(opt_scales_dict,
                           **lnth_scales_dict,
                           **means_dict,
                           **u_means_dict,
                           **u_points_dict)
    
    return hyperparameter_dict



def _get_hypers_skhgp(model):
    """
    Subprocess for get_hyperparamaters for stochastic kriging model
    """    
    print("not implemented yet")
    return 0

# PREDICTION HANDELIING FUNCTIONS

def sausage_plot(train_x,
                 train_y,
                 grid_x,
                 pred_f,
                 pred_sigma2_f,
                 pred_sigma2_eps,
                 true_f,
                 true_sigma2,
                 path,
                 f_name,
                 plot_title="Predictions",
                 hyperparamaters =None,
                 candidates = None, #dict #TODO: Arrange canddiates 
                 ):
    
    """
    Given model predictions, training data, and the candidates just selected (if selected), this function creates a 
    'sausage plot' - including the posterior mean prediction, noise prediction, and model uncertainty, each to +/- 2\sigma.

    """
    #Declare variables

    #If not plotting within loop
    # print(f"grid_x{grid_x.shape}")
    # print(f"train_x{train_x.shape}")
    # print(f"true_f{true_f.shape}")
    # print(f"pred_f{pred_f.shape}")
    # print(f"pred_sigma2_f{pred_sigma2_f.shape}")
  
    
    if hyperparamaters is not None:
        #TODO Include hyperparamaters
        plot_title = plot_title + f"|Hyperparamaters|"
    


    fig,ax = plt.subplots(1,1,figsize=FIGSIZE)
    
    #Plot Observations
    ax.plot(train_x,train_y, "o", label="Evals", alpha=0.5)
    #Plot Predicted mean mu_f
    ax.plot(grid_x, pred_f, color="C0", label="Post Mean")
    # Plot Truth <- green
    ax.plot(grid_x,true_f,color='g',label='Truth')

    #Plot candidates if given
    if candidates is not None:
        
        new_x, new_y = candidates.values()
        ax.plot(new_x,new_y,'r*',label="Candidates")

    #Plot +/- epistemic uncretainty sigma
    ax.fill_between(
            grid_x,
            (pred_f - 2 * st.sqrt(pred_sigma2_f)),
            (pred_f + 2 * st.sqrt(pred_sigma2_f)),
            color="C0",
            alpha=0.15,
            label="±2 std $\sigma_{f}$",
    )
    #Plot +/- 2* predicted noise sigma
    ax.fill_between(
            grid_x,
            pred_f - 2 * st.sqrt(pred_sigma2_eps),
            pred_f + 2 * st.sqrt(pred_sigma2_eps),
            color="C1",
            alpha=0.15,
            label="±2 std $\sigma_{\epsilon}$",
    )
    #Plot +/- 2* true noise sigma2
    ax.fill_between(
            grid_x,
            true_f - 2 * st.sqrt(true_sigma2),
            true_f + 2 * st.sqrt(true_sigma2),
            color="C2",
            alpha=0.15,
            label="±2 std (truth)",
    )
    #Force Limits for strange behaviour in sk examples
    #ax.set_ylim(-2.5,2.5)

    #Axis labels and legends
    ax.set_xlabel("$x$")
    ax.set_ylabel("$y$")
    ax.set_title(plot_title)
    ax.legend(loc="lower left",ncol=3)

    #Savefig at subdir: preds w/name: acqf_pred_m_t
    
    plt.savefig(path / f_name, dpi=DPI, bbox_inches="tight")
    plt.close()

def input_generator(n_grid,bounds=[0,1],replications=None):
    """
    Should be able to also pass input to AF for different selections of n
    
    Optional boudns to be handled later
    """    

    #Includes n if replications is not none

    #Generate test points
    grid_x =  st.linspace(0,1,n_grid).to(**TKWARGS)

    # Generates grid of x and n values if replications given
    if replications is not None:
        grid_xn = st.stack([
            grid_x.repeat(len(replications)),          # column 0
            replications.repeat_interleave(len(grid_x))  # column 1
        ], dim=1)

        #If replications supplied
        return {"xn":grid_xn,"x":grid_x.unsqueeze(-1)}

    return grid_x.unsqueeze(-1)



def _predict_skhgp(grid_x,model,outcome_transform):

    """
    Obtains the predictions for the stochastic kriging model
    """
    #Generate posteriors for prediction grid
    with st.no_grad():
        posterior_f = model['f'].posterior(grid_x)
        posterior_g = model['eps'].posterior(grid_x)

    #Calculatte predicted variance
    sigma_2_eps = posterior_g.mean
    sigma_2_eps_var = posterior_g.variance

    ##Transform predicted variance
    sigma_2_eps, sigma_2_eps_var = _transform_GP(
                sigma_2_eps, sigma_2_eps_var, outcome_transform['eps']
            )
    sigma_2_eps = (
        _inverse_log_transform(
            sigma_2_eps, sigma_2_eps_var, outcome_transform['eps']
        )
        * outcome_transform['f'].sig_std
    )

    #Calculate the predicted mean
    mean_f = posterior_f.mean
    sigma_2_f = posterior_f.variance

    mean_f,sigma_2_f = _transform_GP(mean_f,
                                    sigma_2_f,
                                    outcome_transform['f'])

    return mean_f, sigma_2_f, sigma_2_eps
    

    return pred_f,pred_sigma2_f,pred_sigma2_eps
def _predict_vihgp(grid_x,model,outcome_transform):

    """
    Generates the predictions from the vihgp model
    """
    #Generate posteriors for prediction grid
    with st.no_grad():
        posterior_f = model.posterior(grid_x)
        posterior_g = model.noise_posterior(grid_x)

    #Calculatte predicted variance
    sigma_2_eps = posterior_g.mean
    sigma_2_eps_var = posterior_g.variance

    ##Transform predicted variance
    sigma_2_eps = (
                _inverse_log_transform(sigma_2_eps, sigma_2_eps_var, outcome_transform)
                * outcome_transform.sig_std
            )

    #Calculate the predicted mean
    mean_f = posterior_f.mean
    sigma_2_f = posterior_f.variance

    mean_f,sigma_2_f = _transform_GP(mean_f,
                                    sigma_2_f,
                                    outcome_transform)

    return mean_f, sigma_2_f,sigma_2_eps
def predictor(n_grid,model,outcome_transform):

    #Generate Grid <- by default over [0,1]
    grid_x = input_generator(n_grid)

    #Determine model to use and export prediction
    if isinstance(model,dict):
        pred_f,pred_sigma2_f,pred_sigma2_eps = _predict_skhgp(grid_x,model,outcome_transform)
        return grid_x,pred_f,pred_sigma2_f,pred_sigma2_eps
    else:
        pred_f,pred_sigma2_f,pred_sigma2_eps = _predict_vihgp(grid_x,model,outcome_transform) 
        return grid_x,pred_f,pred_sigma2_f,pred_sigma2_eps         

def prediction_plotter(train_x,
                       train_y,
                       n_grid,
                       model,
                       outcome_transform,
                       acqf_name,
                       path,
                       candidates=None, #dict of tensors {x:,y:}
                       hyperparamaters=None,
                       run_params = None,):
    
    """
    Creates a sausage plot of the supplied model and includes observations and most recent candidate point
    """

    # Generate Predictions
    grid_x,pred_f,pred_sigma2_f,pred_sigma2_eps = predictor(n_grid,model,outcome_transform)

    # TODO: Import Target from Input subdir
    test_data = get_files(path,"Input",['test_y','test_sigma2'])
    true_f = test_data['test_y']
    true_sigma2 = test_data['test_sigma2']

    """
    Comment:
    Just import the target function which is used by experiment handler.
    perhaps we can use memorisation at a later point to save wasting resources recalculating the same test grid.

    We are given the path, so we can just use that to work out where Input is and extract it
    """
    #Plot and save sausage plot figure
    plot_title = "Prediction"
    outdir = Path(path + LOG_FNAME +"/preds")
    if run_params is not None:
        plot_title = plot_title + f"|m={run_params['m']}|t={run_params['t']}|"
        f_name = f"{acqf_name}_pred_{run_params['m']}_{run_params['t']}.png"
    else:
        f_name = f"{acqf_name}_pred.png"

    if candidates is not None:
        candidates['x'] = candidates['x'].squeeze(-1)
        candidates['y'] = candidates['y'].flatten()

    sausage_plot(train_x=train_x.squeeze(-1),
                 train_y = train_y.flatten(),
                 grid_x=grid_x.squeeze(-1),
                 pred_f=pred_f.flatten(),
                 pred_sigma2_f=pred_sigma2_f.flatten(),
                 pred_sigma2_eps=pred_sigma2_eps.flatten(),
                 true_f=true_f.flatten(),
                 true_sigma2=true_sigma2.flatten(),
                 path=outdir, 
                 f_name=f_name,
                 plot_title=plot_title, 
                 candidates=candidates,
                 hyperparamaters=hyperparamaters)
    
#TODO AF TROUBLESHOOTING FUNCTIONS

def acqf_plot(grid_xn,
              acq_vals,
              path,
              f_name,
              plot_title,):
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
   

    # Initialize plot
    plt.figure(figsize=FIGSIZE)
    
    #Get grids
    grid_x = grid_xn['x']
    grid_xn = grid_xn['xn']
    n_vals = grid_xn[...,1].unique()
    with st.no_grad():
        for i, n in enumerate(n_vals):
            plt.plot(grid_x,acq_vals[i],label=f"n={n}")

    plt.xlabel('$x$')
    plt.ylabel('$AF(x)$')
    plt.title(plot_title)
    plt.legend()
    
    #Savefig at subdir: acqs w/name: acqf_acqs_m_t
    
    plt.savefig(path / f_name, dpi=DPI, bbox_inches="tight")
    plt.close()
    

def acqf_plotter(n_grid,
                 acq_func,
                 acqf_name,
                 path,
                 run_params=None,
                 replications=st.tensor([1,5,10])):

    #Generate Grid
    grid_xn = input_generator(n_grid,replications=replications)
    
    #Obtain acqf values
    acq_vals = acq_func(grid_xn['xn'].unsqueeze(1))
    plot_title = "acq vals"

    if run_params is not None:
        plot_title = plot_title + f"|m={run_params['m']}|t={run_params['t']}|"
        f_name = f"{acqf_name}_acqf_{run_params['m']}_{run_params['t']}.png"
    else:
        f_name = f"{acqf_name}_acqf.png"

    outdir = Path(path + LOG_FNAME +"/acqs")

    #Plot and save acq fig
    acqf_plot(grid_xn=grid_xn,
              acq_vals=acq_vals.reshape(replications.shape[0],n_grid),
              path=outdir,
              f_name = f_name,
              plot_title=plot_title
              )
import torch as st
import matplotlib.pyplot as plt
#TODO UTILS

def get_files(dir_name,file_names,add=""):
	data = {}
	for file_name in file_names:

		load_in = st.load(indir /  f"{add}{file_name}.pt").to(**tkwargs)
		data[file_name] = load_in
	return data

#TODO HYPERPARAMATER HANDLING FUNCTIONS
def export_hyperparamaters(path,name,hyperparamaters):
    """
    Exports hyperparamaters in x format under the chosen name and saves in the chosen path

    paramaters:
        path: path
        name: string
        hyperparamaters: dict
            A dictionary of hyperparamaters
    
    """



    print("stop")


def get_hyperparamaters(model):

    """
    Exports paramaters from the given GP model into a dictionary

    paramaters:

    returns:
        hyperparamaters: dict
            Dictionary of hyperparamaters.
    """

    #Determines which subprocess to used depending on if SK or VI supplied
    print("hodl")


def _get_hypers_vihgp(model):
    """
    Subprocess for get_hyperparamaters
    """

    #NOTE: Do with grad

def _get_hypers_skhgp(model):
    """
    Subprocess for get_hyperparamaters for stochastic kriging model
    """    

#TODO PREDICTION HANDELIING FUNCTIONS

def sausage_plot(train_x,
                 train_y,
                 grid_x,
                 pred_f,
                 pred_sigma2_f,
                 pred_sigma2_eps,
                 true_f,
                 true_sigma2,
                 name,
                 path,
                 hyperparamaters =None,
                 candidates = None, #dict #TODO: Arrange canddiates 
                 run_params = None, #dict {m:}
                 ):
    
    """
    Given model predictions, training data, and the candidates just selected (if selected), this function creates a 
    'sausage plot' - including the posterior mean prediction, noise prediction, and model uncertainty, each to +/- 2\sigma.

    """
    #Declare variables
    plot_title = f"predictions"

    if run_params is not None:
        plot_title = plot_title + f"|m={run_params['m']}|t={run_params['t']}|"
    if hyperparamaters is not None:
        #TODO Include hyperparamaters
        plot_title = plot_title + f"|Hyperparamaters|"
    


    fig,ax = plt.subplots(1,1,figsize=(8,14))
    
    #Plot Observations
    ax.plot(train_x,train_y, "o", label="Evals", alpha=0.5)
    ax.plot(grid_x, pred_f, color="C0", label="Post Mean")
    
    # Plot Truth <- green
    ax.plot(grid_x,true_f,color='g',label='Truth')

    ax.fill_between(
            grid_x,
            (pred_f - 2 * st.sqrt(pred_sigma2_f)),
            (pred_f + 2 * st.sqrt(pred_sigma2_f)),
            color="C0",
            alpha=0.15,
            label="±2 std $\sigma_{f}$",
    )
    ax.fill_between(
            grid_x,
            pred_f - 2 * st.sqrt(pred_sigma2_eps),
            pred_f + 2 * st.sqrt(pred_sigma2_eps),
            color="C1",
            alpha=0.15,
            label="±2 std $\sigma_{\epsilon}$",
    )
    ax.fill_between(
            grid_x,
            true_f - 2 * st.sqrt(true_sigma2),
            true_f + 2 * st.sqrt(true_sigma2),
            color="C2",
            alpha=0.15,
            label="±2 std (truth)",
    )
    ax.set_xlabel("$x$")
    ax.set_ylabel("$y$")

    ax.set_title(plot_title)
    
    ax.legend(loc="lower left",ncol=3)

def input_generator(n_grid,bounds,replications):
    """
    Should be able to also pass input to AF for different selections of n
    """    

    #Includes n if replications is not none
def _predict_skhgp(grid_x,model,outcome_transform):

    """
    
    """
    pred_f = 0
    return pred_f
def _predict_vihgp(grid_x,model,outcome_transform):

    """
    
    """

    return pred_f
def predictor(n_grid,model,outcome_transform):

    #Generate Grid <- by default over [0,1]
    grid_x,_ = input_generator(n_grid,bounds)

    #Determine model to use and export prediction
    if statement:
        return grid_x, _predict_skhgp(grid_x,model,outcome_transform)
    else:
        return grid_x, _predict_vihgp(grid_x,model,outcome_transform)          

def prediction_plotter(train_x,
                       train_y,
                       n_grid,
                       model,
                       outcome_transform,
                       name,
                       path,
                       candidates=None, #dict of tensors {x:,y:}
                       hyperparamaters=None):
    
    """
    Creates a sausage plot of the supplied model and includes observations and most recent candidate point
    """

    # Generate Predictions
    grid_x,pred_f,pred_sigma2_f,pred_sigma2_eps = predictor(n_grid,model,outcome_transform)

    # TODO: Generate Target
    """
    Comment:
    Just import the target function which is used by experiment handler.
    perhaps we can use memorisation at a later point to save wasting resources recalculating the same test grid.

    We are given the path, so we can just use that to work out where Input is and extract it
    """
    #Plot and save sausage plot figure
    sausage_plot(train_x,
                 train_y,
                 grid_x,
                 pred_f,
                 pred_sigma2_f,
                 pred_sigma2_eps,
                 true_f,
                 true_sigma2,
                 name, 
                 path, #TODO Supply as path + prediction folder loc
                 candidates,
                 hyperparamaters)
    
#TODO AF TROUBLESHOOTING FUNCTIONS

def acqf_plotter(n_grid,model,acq_func,name,path,replications=[1,5,10]):

    #Generate Grid
    grid_xn = input_generator(n_grid,bounds,replications)
    #Obtain acqf values

    #Plot and save acq fig
    acqf_plot(grid_xn,
              acq_vals,
              name,
              path
              )
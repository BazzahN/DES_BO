import torch
from exp_utils import output_handler
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from gpytorch.mlls import ExactMarginalLogLikelihood
import matplotlib.pyplot as plt

tkwargs = {
    "dtype": torch.double,# Datatype used by tensors
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"), # Declares the 'device' location where the Tenosrs will be stored
}


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

    #plt.show()        

def poster_plot(
    N_points,
    train_x,
    train_y,
    acqf,
    n_vals,
    model,
    test_function,
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
                     output_handler,
                     test_function,
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
    
    true_y_std = output_handler['f'].standardise(true_y)
    
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
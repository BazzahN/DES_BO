"""
Code to asertain the baseline hyperparameters for the target function in use
"""
import torch as st
import matplotlib.pyplot as plt
import yaml
# GP Fitting
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from gpytorch.mlls import ExactMarginalLogLikelihood

from gpytorch.kernels import RBFKernel,ScaleKernel
from gpytorch.means import ZeroMean

from test_utils import TEST_FUNCTION_DIAL, NOISE_FUNCTION_DIAL,Target_Function

tkwargs = {
    "dtype": st.double,# Datatype used by tensors
    "device": st.device("cuda" if st.cuda.is_available() else "cpu"), # Declares the 'device' location where the Tenosrs will be stored
}

# TODO Import config file (number of points, test functions etc)
"""
Input:
- test_function_id
- phi value
- tau value
- k
- n_grid
"""
##Import exp config file
config = "baseline/baseline_exp.yml"

with open(config) as f:
    config = yaml.safe_load(f)

test_function_id = config['test_function_index']
k = config['k']
phi = config['phi']
tau = config['tau']
n_grid = config['n_grid']
#Generate Target Function outputs 
## Initialise target funciton
test_function = TEST_FUNCTION_DIAL[test_function_id]
noise_function = NOISE_FUNCTION_DIAL[0]


target = Target_Function(test_function,
                        noise_function,
                        phi=phi,
                        tau=tau,
                        rng_state=st.Generator().manual_seed(1).get_state()
                        )


train_x,train_y,train_sigma_2 = target.eval_target_true_grid(k,0,1)

# Fit SingleTaskGP

RBF_kernel = ScaleKernel(
    RBFKernel(batch_shape=st.Size([])),
    batch_shape=st.Size([])).to(train_x)

RBF_kernel_notau = RBFKernel(batch_shape=st.Size([]))
zero_mean = ZeroMean(batch_shape=st.Size([])).to(train_x)
"""
Use either botorch's default kernel, or use a scaled RBF. Standardisation still applies as it does in the examples
"""
covar_modules = [None,RBF_kernel,RBF_kernel_notau]
mean_modules = [None,zero_mean]
model = SingleTaskGP(train_X=train_x,
                    #  train_Y=train_y,
                     train_Y=train_sigma_2,
                     train_Yvar= st.full_like(train_y,1e-16),
                     covar_module= covar_modules[0],
                     mean_module=mean_modules[0]
                     )
mll = ExactMarginalLogLikelihood(model.likelihood,model)
print("Fitting Baseline GP")
fit_gpytorch_mll(mll)

# TODO Generate predicitons and output sausage plot
print("Generating Prediction Data")
## Generate grid space
grid_x = st.linspace(0,1,n_grid).to(**tkwargs)

#Generate predictions
posterior = model.posterior(grid_x)
pred_f = posterior.mean.detach().flatten()
pred_f_var = posterior.variance.detach().flatten()

#Generate Truth
_,true_f,true_sigma_2=target.eval_target_true_grid(n_grid,0,1)

# Change to either train_y for test function and train_sigma_2 for noise
evals = train_sigma_2
trues = true_sigma_2
## Plot Predictions
plt.figure()
#Plot training data
plt.plot(train_x,evals,"o",label="Evals",alpha=0.5)
#Plot predicted mean mu_f
plt.plot(grid_x,pred_f,color="C0",label="Post Mean")
#Plot truth
plt.plot(grid_x,true_sigma_2,color="g",label="Truth")

plt.fill_between(
            grid_x,
            (pred_f - 2 * st.sqrt(pred_f_var)),
            (pred_f + 2 * st.sqrt(pred_f_var)),
            color="C0",
            alpha=0.15,
            label="±2 std $\sigma_{f}$",
    )

plt.xlabel("$x$")
plt.ylabel("$y$")
plt.title(f"Baseline GP Sausage Plot for $k=${k}")
plt.legend()

plt.savefig(f"baseline/sausage_plot_k{k}.png",dpi=200,bbox_inches="tight")
plt.close()
# TODO Output hyperparameters (print and save as yaml)

def get_hypers_baseGP(model):

    # Get constant mean values.
    try:
        means = model.mean_module.constant.detach().item()
    except:
        means = 0

    # Get outputscales
    try:
        taus = model.covar_module.outputscale.item()
    except:
        taus= 1 
    
    # Get lengthscales
    try:
        lnth_scales = model.covar_module.base_kernel.lengthscale.item()
    except:
        lnth_scales = model.covar_module.lengthscale.item()
    
    hyperparameter_dict = {"mu":means,
                           "l":lnth_scales,
                           "tau":taus}
    return hyperparameter_dict

hyper_dict = get_hypers_baseGP(model)

with open(f"baseline/hypers_k{k}.yml",'w') as f:
    yaml.safe_dump(hyper_dict,f)

print("All data exported to baseline")
print("done")    
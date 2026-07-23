from functools import partial

from exp_utils import VI_HGP,run_IG_exp_simple,get_stoch_kriging_model,get_files
from test_utils import Target_Function,NOISE_FUNCTION_DIAL,TEST_FUNCTION_DIAL,InverseLinearCostModel
from pathlib import Path
import yaml
import torch

tkwargs = {
    "dtype": torch.double,# Datatype used by tensors
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"), # Declares the 'device' location where the Tenosrs will be stored
}


#Input data
exp_fname = "configs/" + "HetGP_k10_reps.yml"

with open(exp_fname) as f:
    config = yaml.safe_load(f)

exp_name = exp_fname.split("/")[-1].removesuffix(".yml") #Creates experiment name from suffix


#Step 1: Import Arguments
#Experimental Parameters
T = config['T'] #Number of iterations


# Problem Constants
k= config["k"] #Number of points
n= config["n"] #Replications at each point

x_min = 0
x_max = 1

n_min = 1
n_max = 50 #Sample bounds

test_function_id = config["test_function_index"]
noise_function_id = config["noise_function_index"] #Function dial in test_utils

phi = config['phi']
tau = config['tau'] #Additional Noise Function Paramaters


b0 = config['b0']  #Cost Function Paramaters 1/(b0+b1x)
b1 = config['b1']
maximise = True

#Step 2: Import Data 
#Import experiment input

indir = Path(exp_name + "/Data")

import_data = partial(get_files,
                        indir = indir,
                        file_names = ['train_x','train_n','train_y','train_sigma2'])
m = 0

data_in = import_data(suffix = f"_m{m}")

#NOTE When we move to n dimensions this code will have to change
train_x = data_in['train_x']
train_n = data_in['train_n']
train_y = data_in['train_y']
train_sigma2 = data_in['train_sigma2']


##Test problem and cost function
noise_function = NOISE_FUNCTION_DIAL[noise_function_id]
test_function = TEST_FUNCTION_DIAL[test_function_id]
lin_cost_func = InverseLinearCostModel([b1,b0])

target = Target_Function(test_function,
                             noise_function,
                             phi=phi,
                             tau=tau,
                             rng_state=torch.Generator().manual_seed(1).get_state()
                            )

bounds = torch.tensor([[x_min,n_min] * 1,
                        [x_max,n_max] * 1],
                        dtype=torch.double,
                        device=torch.device("cpu")) # Bounds of combined X and N space


#FIt Model to data
VI_HGP =VI_HGP(gamma=0.4,iters=800,standardise=False,verbose=True)
model_call = VI_HGP.get_VI_HGP_model
hgp_model, out_transform,hyperparamaters = model_call(train_x.flatten().unsqueeze(-1),train_n,train_y.flatten().unsqueeze(-1),train_sigma2)



# with torch.no_grad():
#     try:
#         taus = hgp_model.model.covar_module.outputscale.tolist()
#     except:
#         taus=[0,0]

#     opt_scales_dict = {'tau_1':taus[0],'tau_2':taus[1]}
#     #Extract lengthscale
#     lnth_scales = hgp_model.model.covar_module.base_kernel.lengthscale.flatten().tolist()
#     lnth_scales_dict = {'l_1':lnth_scales[0],'l_2':lnth_scales[1]}

#     #Extract Means
#     #If ZeroMean used instead

#     try:
#         means = hgp_model.model.mean_module.constant.detach().tolist()
#     except:
#         means = [0,0]

#     means_dict = {'mu_1':means[0],'mu_2':means[1]}

#     #Extract inducing Means
#     u_means = hgp_model.model.variational_strategy.base_variational_strategy._variational_distribution.variational_mean.detach().tolist()
#     u_means_dict = {'mu_u_1':u_means[0],'mu_u_2':u_means[1]}

#     #Extract Inducing Points
#     u_points = hgp_model.model.variational_strategy.base_variational_strategy.inducing_points.flatten().detach().tolist()
#     u_points_dict = {'u':u_points}

# hyperparameter_dict = dict(opt_scales_dict,
#                            **lnth_scales_dict,
#                            **means_dict,
#                            **u_means_dict,
#                            **u_points_dict)


"""
Debugging: The costs are correct shape
"""

from DES_acqfs import BODES_IG,MUMBO_IG
from troubleshoot_utils import input_generator

AF = MUMBO_IG(model = hgp_model,
              cost_model=lin_cost_func,
              output_transform=out_transform,
              candidate_set= torch.linspace(bounds[0,0],bounds[1,0],40).unsqueeze(1))

grid_xn = input_generator(100,replications=torch.tensor([1]))

acqf_vals = AF(grid_xn['xn'].unsqueeze(1))
import matplotlib.pyplot as plt
with torch.no_grad():
    plt.plot(grid_xn['x'],acqf_vals)


# IG_exp = run_IG_exp_simple(n=2,
#                             AF=MUMBO_IG,
#                             model_call_func=model_call,
#                             cost_function=lin_cost_func,
#                             bounds=bounds[:,0].view(-1,1))



# IG_exp_sk = run_IG_exp_itr(n=1,
#                         AF=BODES_IG,
#                         model_call_func=get_stoch_kriging_model,
#                         cost_function=lin_cost_func,
#                         bounds=bounds,
#                         GP='sk')

# init_x = torch.linspace(0,1,500).reshape((500,1,1)).unsqueeze(-3)
# hgp_model,AF,train_x_v,train_n_v,train_y_v,train_sigma2_v,out_transform_v,hyperparamaters = IG_exp.run_iter(hgp_model,
#                                                                                                             train_x.flatten().unsqueeze(-1),
#                                                                                                             train_n,
#                                                                                                             train_y.flatten().unsqueeze(-1),
#                                                                                                             train_sigma2,
#                                                                                                             target,
#                                                                                                             out_transform)

# sk_model,sk_transform = get_stoch_kriging_model(train_x.mean(dim=0),train_n,train_y.mean(dim=0),train_y.var(dim=0))
# sk_model,train_x_s,train_n_s,train_y_s,train_sigma2_s,out_transform_s = IG_exp_sk.run_iter(sk_model,
#                                                                               train_x.mean(dim=0),
#                                                                               train_n,
#                                                                               train_y.mean(dim=0),
#                                                                               train_y.var(dim=0),
#                                                                               target,
#                                                                               sk_transform)
# train_x = data_in['train_x'][m]
# train_n = data_in['train_n'][m]
# train_y = data_in['train_y'][m]
# train_sigma2 = data_in['train_sigma2'][m]
# train_rngs = data_in['rngs'][m] 


init_x = torch.linspace(0,1,100).reshape((100,1))
# out_sk = sk_model['f'].posterior(init_x)
# sk_mean = sk_transform['f'].unstandardise(out_sk.mean)

true_y,true_eps = target.eval_target_true(init_x)
 


from DES_acqfs import _inverse_log_transform,_transform_GP
out_hgp = hgp_model.posterior(init_x)
hgp_mean,hgp_var = _transform_GP(out_hgp.mean,out_hgp.variance,out_transform)
hgp_2_out = hgp_model.noise_posterior(init_x)
sigma_2_eps = (
                _inverse_log_transform(hgp_2_out.mean, hgp_2_out.variance, out_transform)
                * out_transform.sig_std
            )


from troubleshoot_utils import sausage_plot
outdir = Path(exp_name + "/Plot")
outdir.mkdir(parents=True,exist_ok=True)

with torch.no_grad():
    sausage_plot(train_x=train_x.flatten(),
                train_y=train_y.flatten(),
                grid_x=init_x.flatten(),
                pred_f=hgp_mean.flatten(),
                pred_sigma2_f=hgp_var.flatten(),
                pred_sigma2_eps=sigma_2_eps.flatten(),
                true_f=true_y.flatten(),
                true_sigma2=true_eps.flatten(),
                path = outdir,
                f_name = "Good_plot")


# # plt.plot(sk_mean.detach())
plt.plot(init_x,true_y)
plt.plot(init_x,hgp_mean.detach())
plt.plot(train_x,train_y,'x')

# plt.plot(init_x,true_eps)
# plt.plot(init_x,sigma_2_eps.detach())
# from exp_utils import experiment_handler

# exp_hold = experiment_handler(target,IG_exp)
# master = torch.Generator().manual_seed(12345)

# train_x,train_n,train_y,train_sigma2,x_strs,f_strs = exp_hold.run_T_BO_iters(5,
#                                                                             train_x.flatten().unsqueeze(-1),
#                                                                             train_n,
#                                                                             train_y.flatten().unsqueeze(-1),
#                                                                             train_sigma2,
#                                                                             master.get_state()
#                                                                             )

# exp_hold = experiment_handler(target,IG_exp_sk)
# master = torch.Generator().manual_seed(12345)

# exp_hold.run_T_BO_iters(5,
#                         train_x.mean(dim=0),
#                         train_n,
#                         train_y.mean(dim=0),
#                         train_y.var(dim=0),
#                         master.get_state()
#                         )                        #

# from exp_utils import get_best_f_SEI

# x,f = get_best_f_SEI(hgp_model,bounds=bounds,output_transform=out_transform_v)
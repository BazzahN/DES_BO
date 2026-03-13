from exp_utils import VI_HGP,run_IG_exp_itr
from test_utils import Target_Function,NOISE_FUNCTION_DIAL,TEST_FUNCTION_DIAL,InverseLinearCostModel
from pathlib import Path
import yaml
import torch
tkwargs = {
    "dtype": torch.double,# Datatype used by tensors
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"), # Declares the 'device' location where the Tenosrs will be stored
}


#Input data
exp_fname = "configs/" + "pilot_01.yml"

with open(exp_fname) as f:
    config = yaml.safe_load(f)

exp_name = config["experiment_name"]
study_args = config['study']
gen_args = config["problem"]
plot_args = config["plots"]
exp_models = config["models"]

#Step 1: Import Arguments
#Experimental Parameters
T = study_args['T'] #Number of iterations
M = study_args['M'] #Number of MacroReplications


# Problem Constants
k= gen_args["k"] #Number of points
n= gen_args["n"] #Replications at each point

x_min = gen_args["x_min"] 
x_max = gen_args["x_max"] #Domain bounds

n_min = gen_args["n_min"] 
n_max = gen_args["n_max"] #Sample bounds

test_function_id = gen_args["test_function_index"]
noise_function_id = gen_args["noise_function_index"] #Function dial in test_utils

phi = gen_args['phi']
tau = gen_args['tau'] #Additional Noise Function Paramaters


b0 = gen_args['b0']  #Cost Function Paramaters 1/(b0+b1x)
b1 = gen_args['b1']
maximise = True

#Step 2: Import Data 
#Import experiment input

indir = Path(exp_name + "/Input")
data_in = {}
names = ['train_x','train_n','train_y','train_sigma2','rngs']

for name in names:

    load_in = torch.load(indir /  f"{name}.pt")
    data_in[name] = load_in.to(**tkwargs)

m = 0
train_x = data_in['train_x'][m]
train_n = data_in['train_n'][m]
train_y = data_in['train_y'][m]
train_sigma2 = data_in['train_sigma2'][m]
train_rngs = data_in['rngs'][m] 

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
VI_HGP =VI_HGP(n_u=10,iters=800,standardise=False)
model_call = VI_HGP.get_VI_HGP_model
hgp_model, out_transform = model_call(train_x,train_n,train_y,train_sigma2)


from DES_acqfs import BODES_IG
IG_exp = run_IG_exp_itr(n=1,
                        AF=BODES_IG,
                        model_call_func=model_call,
                        cost_function=lin_cost_func,
                        bounds=bounds,)



# init_x = torch.linspace(0,1,500).reshape((500,1,1)).unsqueeze(-3)
hgp_model,train_x,train_n,train_y,train_sigma2,out_transform = IG_exp.run_iter(hgp_model,
                                                                              train_x,
                                                                              train_n,
                                                                              train_y,
                                                                              train_sigma2,
                                                                              target,
                                                                              out_transform)
   
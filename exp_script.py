import torch
import argparse
import yaml
from pathlib import Path
import matplotlib.pyplot as plt
from test_utils import TEST_FUNCTION_DIAL,NOISE_FUNCTION_DIAL,InverseLinearCostModel,Target_Function
from functools import partial
from exp_utils import get_files
TKWARGS = {
    "dtype": torch.double,# Datatype used by tensors
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"), # Declares the 'device' location where the Tenosrs will be stored
}

from exp_utils import experiment_handler,EXPERIMENTS,GP_dial
import sys
def main():

    ##Import arguments from the command line
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--n_macros",type=int, required=True)
    parser.add_argument("--m_min",type=int, default=0,required=False)
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    exp_name = args.config.split("/")[-1].removesuffix(".yml") #Creates experiment name from suffix
    misc_args = config["misc"]
    #TODO Remove choice between sk and vhgp in problems
    GP_arg = "vihgp"
    #Step 1: Import Arguments
    #Experimental Parameters
    M = args.n_macros #Number of MacroReplications
    m_min = args.m_min #Minimum number of macro replications
    model = config["AF"]
    # Problem Constants
    T = config["T"]
    B = config["B"]
    n_v = config["n_v"] #Number of replications for vanilla

    x_min = 0
    x_max = 1 #Domain bounds

    n_min = 1
    n_max = 20

    test_function_id = config["test_function_index"]
    noise_function_id = config["noise_function_index"] #Function dial in test_utils

    phi = config['phi']
    tau = config['tau'] #Additional Noise Function Paramaters


    b0 = config['b0']  #Cost Function Paramaters 1/(b0+b1x)
    b1 = config['b1']
    maximise = True

    # Misc Arguments
    n_grid = misc_args['n_grid']
    troubleshoot = misc_args['troubleshoot']

    #Step 2: Import Data 
    #Import experiment input

    indir = Path(exp_name + "/Input")
    rngs = torch.load(indir /  "rngs.pt")
    
    import_data = partial(get_files,
                          indir = indir,
                          file_names = ['train_x','train_n','train_y','train_sigma2'])
    
    
    #Step 3: Initalise functions and methods
    ##Test problem and cost function
    noise_function = NOISE_FUNCTION_DIAL[noise_function_id]
    test_function = TEST_FUNCTION_DIAL[test_function_id]
    lin_cost_func = InverseLinearCostModel([b1,b0])


    ##Initalise Target Function class For Experiments
    #TODO Modify target function in utilities
    target = Target_Function(test_function,
                             noise_function,
                             phi=phi,
                             tau=tau,
                             rng_state=torch.Generator().manual_seed(1).get_state()
                            )

    if model in ["IG", "AEI","MUMBO"]:
        bounds = torch.tensor([[x_min,n_min] * 1,
                                [x_max,n_max] * 1],
                                dtype=torch.double,
                                device=torch.device("cpu")) # Bounds of combined X and N space
    else: 
        bounds = torch.tensor([[x_min] * 1,
                                [x_max] * 1],
                                dtype=torch.double,
                                device=torch.device("cpu")) # Bounds of combined X and N space
   
    #Step 4: Execute experiments

    outdir = Path(exp_name + "/Data")
    outdir.mkdir(parents=True,exist_ok=True)
    
    # Creates Log directory if troubleshoot =True
    if troubleshoot:
        subdirs = ['hyperparamaters','preds','acqs']
        for subd in subdirs:
            logdir = Path(exp_name + "/Log/" + subd)
            logdir.mkdir(parents=True,exist_ok=True)

   
    
        ##Initalise experiment handling class
    
    exp_object = EXPERIMENTS[model]
    experiment = exp_object(n=n_v, #Assigns number of replications for vanilla. If not vanilla then dummy used
                            cost_function=lin_cost_func,
                            bounds=bounds,
                            model_call_func=GP_dial(GP_arg,misc_args["vihgp"]),
                            GP=GP_arg)
    add_params = {"path":exp_name,
                  "acqf_name":model,
                  "n_grid":n_grid} #For storage of additional information
        
    #Automatically switch off troubleshoot if vanilla supplied
    if model == "vanilla":
        print("Troubleshooting tools doesn't support EI. Setting troubleshoot=False")
        run_experiment = experiment_handler(target,
                                            experiment,
                                            troubleshoot=False,
                                            additional_paramaters=add_params,
                                            )
    else:
        run_experiment = experiment_handler(target,
                                            experiment,
                                            troubleshoot,
                                            add_params
                                            )
    
    ##Run experiment
    
    for m in range(m_min,M):
        print(f'[SIM]Starting macroreplication {m} of {M}....\n',flush=True)
        data = import_data(suffix=f"_m{m}")
        #Change internal m
        data['rng_state'] = rngs[m]
        run_experiment.m = m
        
        if B is not None:
            out = run_experiment.run_B_BO_iters(B,**data)
        else:
            out = run_experiment.run_T_BO_iters(T,**data)
        run_experiment.save_output(out,outdir,m=m)    
        
    print(f'....Ending Experiment: {model}....\n')
    print(f'Results in {outdir}\n')
    #Save results as tensors
   

if __name__ == "__main__":
    main()







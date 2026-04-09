import torch
import argparse
import yaml
from pathlib import Path
import matplotlib.pyplot as plt
from test_utils import TEST_FUNCTION_DIAL,NOISE_FUNCTION_DIAL,InverseLinearCostModel,Target_Function
tkwargs = {
    "dtype": torch.double,# Datatype used by tensors
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"), # Declares the 'device' location where the Tenosrs will be stored
}

from exp_utils import experiment_handler,EXPERIMENTS,GP_dial
import sys
def main():

    ##Import arguments from the command line
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    # parser.add_argument("--outdir", required=True)
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    exp_name = config["experiment_name"]
    study_args = config['study']
    gen_args = config["problem"]
    misc_args = config["misc"]
    exp_models = config["models"]
    GP_arg = config["GP"]
    #Step 1: Import Arguments
    #Experimental Parameters
    T = study_args['T'] #Number of iterations
    M = study_args['M'] #Number of MacroReplications


    # Problem Constants
    k= gen_args["k"] #Number of points
    n= gen_args["n"] #Replications at each point
    n_v = gen_args["n_v"] #Number of replications for vanilla

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

    bounds = torch.tensor([[x_min,n_min] * 1,
                            [x_max,n_max] * 1],
                            dtype=torch.double,
                            device=torch.device("cpu")) # Bounds of combined X and N space
   
    #Step 4: Execute experiments
    names_out = ['train_x','train_n','train_y','train_sigma2','x_strs','f_strs']
    outdir = Path(exp_name + "/Data")
    outdir.mkdir(parents=True,exist_ok=True)
    #TODO create the other outdirs for predictions if predictions argument is true

    for model in exp_models: 
        ##Initalise experiment handling class
        print(f'Starting Experiment: {model}....\n')
        exp_object = EXPERIMENTS[model]
        experiment = exp_object(n=n_v, #Assigns number of replications for vanilla. If not vanilla then dummy used
                                cost_function=lin_cost_func,
                                bounds=bounds,
                                model_call_func=GP_dial(GP_arg,misc_args["vihgp"]),
                                GP=GP_arg)
        run_experiment = experiment_handler(target,experiment)
        ##Run experiment
        
        out = run_experiment.run_MT_BO_macros(M,T,**data_in)
        print(f'....Ending Experiment: {model}....\n')
        print(f'Results in {outdir}\n')
        #Save results as tensors
        for name,d in zip(names_out,out):
            torch.save(d,outdir /  f"{model}_{name}.pt")


if __name__ == "__main__":
    main()







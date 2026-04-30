'''
Purpose of this file is to take in the input data for a given iteration, or all of the data.
It should take in this data then calculate the log likelihood using the target values

- Step 1 - import file from argument 
- Step 2 - calculate true noise and true function values at target locations
- Step 3 - calculate loglikelihood values at points using targets and the true values at targets
- Step 4 - export as pandas file or csv or something

'''
from math import log, pi
from test_utils import Target_Function,TEST_FUNCTION_DIAL,NOISE_FUNCTION_DIAL
import torch as st
from pathlib import Path
import argparse
import yaml
import pandas as pd
TKWARGS = {
    "dtype": st.double,# Datatype used by tensors
    "device": st.device("cuda" if st.cuda.is_available() else "cpu"), # Declares the 'device' location where the Tenosrs will be stored
}
def calc_log_likelihoood(targets,f_preds,sigma2_preds):
    '''
    Calculates the loglikelihood for a gaussian 
    '''

    sqr_term = (targets - f_preds) ** 2
    log_likelihoood = -0.5 * (log(2.0 * pi) + st.log(sigma2_preds) + sqr_term/sigma2_preds)

    ll_sum = log_likelihoood.sum(dim=-1)

    return ll_sum

def get_files(exp_name,dir_name,file_names,macro,add=""):
	indir = Path(exp_name + f"/{dir_name}")

	data = {}
	for file_name in file_names:

		load_in = st.load(indir /  f"{add}{file_name}_m{macro}.pt").to(**TKWARGS)
		data[file_name] = load_in
	return data

def main():

    ##Import arguments from the command line
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--af_name",type=str,required=True)
    parser.add_argument("--macro",type=int,required=True)
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)
        
    #Extract relevant parameters        
    exp_name = config["experiment_name"]
    gen_args = config["problem"]
    study_args = config["study"]

    #Extract macro
    macro = args.macro

    #Extract AF name
    af_name = args.af_name 

    test_function_id = gen_args["test_function_index"]
    noise_function_id = gen_args["noise_function_index"] #Function dial in test_utils

    phi = gen_args['phi']
    tau = gen_args['tau'] #Additional Noise Function Paramaters
    T= study_args['T']
    #Define outdir - to be stored in the same loc as the data
    outdir = Path(exp_name + "/Log")

    #Import relevant data
    names = ['train_x','train_y']
    data = get_files(exp_name,"Data",names,macro,add=f"{af_name}_")

    #Initalise Target using experimental conditions
    noise_function = NOISE_FUNCTION_DIAL[noise_function_id]
    test_function = TEST_FUNCTION_DIAL[test_function_id]

    target = Target_Function(test_function,
                             noise_function,
                             phi=phi,
                             tau=tau,
                             rng_state=st.Generator().manual_seed(1).get_state()
                            )

    #For loop iterations 
    all_lhood = []
    train_x = data['train_x']
    train_y = data['train_y']
    for i in range(0,T):
        true_y,true_sigma_2 = target.eval_target_true(train_x[i])
        lhood = st.exp(calc_log_likelihoood(train_y[i],true_y,true_sigma_2))
        all_lhood.append({"iter":i,"likelihood":lhood.item()})

    df = pd.DataFrame(all_lhood)
    df.to_csv(outdir / f"lhoods_{exp_name}_m{macro}.csv")        
if __name__ == "__main__":
    main()
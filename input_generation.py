import torch
import argparse
import yaml
from pathlib import Path
from test_utils import Target_Function,get_nxk_inital_evals,NOISE_FUNCTION_DIAL,TEST_FUNCTION_DIAL
tkwargs = {
    "dtype": torch.double,# Datatype used by tensors
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"), # Declares the 'device' location where the Tenosrs will be stored
}

def spawn_generators(master_seed, M):
    master = torch.Generator().manual_seed(master_seed)
    seeds = torch.randint(
        0, 2**63 - 1, (M,), generator=master, dtype=torch.int64
    )
    return [torch.Generator().manual_seed(int(s)) for s in seeds]


def main():

    ##Import arguments from the command line
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--n_macros",type=int, required=True)
    args = parser.parse_args()

    with open(args.config,'r') as f:
        config = yaml.safe_load(f)

    exp_name = args.config.split("/")[-1].removesuffix(".yml") #Creates experiment name from suffix
    misc_args = config["misc"]
    
    #How to generate data depending on GP used
    M = args.n_macros
    ##Arguments
    k= config["k"] #Number of points
    n= config["n"] #Replications at each point
    #NOTE Always [0,1] 
    x_min = 0
    x_max = 1

    test_function_index = config["test_function_index"]
    noise_function_index = config["noise_function_index"] #Function dial in test_utils

    phi = config['phi']
    tau = config['tau'] #Additional Noise Function Paramaters

    seed = config['seed']

    # Output test function output for later comparison against prediction grid
    n_grid = misc_args['n_grid']

    
    ##Generate Dataset D=(x,n,y,sigma2)
    
    outdir = Path(exp_name + "/Input")
    outdir.mkdir(parents=True,exist_ok=True)

    Generators = spawn_generators(seed,M)

    #Generate Correct data based on GP used
    #NOTE Forced for now
    GP_arg = "vhgp"
    if GP_arg == "sk":
        moments = 1
        shape_tuple = (k,1)
    else:
        moments = 0
        shape_tuple =(n*k,1)

    train_x = torch.empty(size=shape_tuple)
    train_y = torch.empty(size=shape_tuple)
    train_n = torch.empty(size=(k,1))
    train_sig2 = torch.empty(size=(k,1))
    rng_smple = Generators[0].get_state().size()
    init_rng = torch.empty(size=(M,rng_smple[0]))

    for i,rng in enumerate(Generators):

        test_class = Target_Function(test_function=TEST_FUNCTION_DIAL[test_function_index],
                                    noise_function=NOISE_FUNCTION_DIAL[noise_function_index],
                                    phi=phi,
                                    tau=tau,
                                    rng_state=rng.get_state())


        train_x,train_n,train_y,train_sig2,test_class = get_nxk_inital_evals(k,n,test_class,x_min,x_max,moments=moments)
        init_rng[i] = test_class.get_rng_state()
        
        #TODO Generate the datasets as seperate files to be loaded individually

        torch.save(train_x, outdir / f"train_x_m{i}.pt")
        torch.save(train_n, outdir / f"train_n_m{i}.pt")
        torch.save(train_y, outdir / f"train_y_m{i}.pt")
        torch.save(train_sig2, outdir / f"train_sigma2_m{i}.pt")
    
    torch.save(init_rng,outdir / f"rngs.pt")

    test_x,test_y,test_sigma2  = test_class.eval_target_true_grid(n_grid,x_min,x_max)
    
    torch.save(test_x,outdir / f"test_x.pt")
    torch.save(test_y,outdir / f"test_y.pt")
    torch.save(test_sigma2,outdir / f"test_sigma2.pt")

    #Obtain max value
    #TODO Change this so it gives the maximiser instead
    # try:
    #     res = config['optim']
    # except:
    #     config['optim'] = test_y.max().item()
    #     with open(args.config,'w') as f:
    #         yaml.safe_dump(config,f)



if __name__ == "__main__":
    main()
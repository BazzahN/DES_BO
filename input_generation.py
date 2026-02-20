import torch
import argparse
import yaml
from pathlib import Path
from test_utils import Target_Function,get_k_inital_evals,NOISE_FUNCTION_DIAL,TEST_FUNCTION_DIAL

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
    # parser.add_argument("--outdir", required=True)
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    exp_name = config["experiment_name"]
    study_args = config['study']
    gen_args = config["problem"]
    plot_args = config["plots"]

    ##Arguments
    M = study_args['M']

    k= gen_args["k"] #Number of points
    n= gen_args["n"] #Replications at each point

    x_min = gen_args["x_min"] 
    x_max = gen_args["x_max"] #Domain bounds

    test_function_index = gen_args["test_function_index"]
    noise_function_index = gen_args["noise_function_index"] #Function dial in test_utils

    phi = gen_args['phi']
    tau = gen_args['tau'] #Additional Noise Function Paramaters

    seed = gen_args['seed']

    # Output test function output for later comparison against prediction grid
    n_grid = plot_args['n_grid']

    
    ##Generate Dataset D=(x,n,y,sigma2)
    
    outdir = Path(exp_name + "/Input")
    outdir.mkdir(parents=True,exist_ok=True)

    Generators = spawn_generators(seed,M)


    train_x = torch.empty(size=(M,k,1))
    train_n = torch.empty(size=(M,k,1))
    train_y = torch.empty(size=(M,k,1))
    train_sig2 = torch.empty(size=(M,k,1))

    rng_smple = Generators[0].get_state().size()
    init_rng = torch.empty(size=(M,rng_smple[0]))

    for i,rng in enumerate(Generators):

        test_class = Target_Function(test_function=TEST_FUNCTION_DIAL[test_function_index],
                                    noise_function=NOISE_FUNCTION_DIAL[noise_function_index],
                                    phi=phi,
                                    tau=tau,
                                    rng_state=rng.get_state())

        # train_x,train_n,train_y,train_sig2,test_class = get_k_inital_evals(k,n,test_class,x_min,x_max)
        train_x[i],train_n[i],train_y[i],train_sig2[i],test_class = get_k_inital_evals(k,n,test_class,x_min,x_max)

        init_rng[i] = test_class.get_rng_state()

        ## Ouput Dataset D=(x,n,y,sigma2)



    torch.save(train_x, outdir / f"train_x.pt")
    torch.save(train_n, outdir / f"train_n.pt")
    torch.save(train_y, outdir / f"train_y.pt")
    torch.save(train_sig2, outdir / f"train_sigma2.pt")
    torch.save(init_rng,outdir / f"rngs.pt")

    test_x,test_y,test_sigma2  = test_class.eval_target_true_grid(n_grid,x_min,x_max)
    
    torch.save(test_x,outdir / f"test_x.pt")
    torch.save(test_y,outdir / f"test_y.pt")
    torch.save(test_sigma2,outdir / f"test_sigma2.pt")


if __name__ == "__main__":
    main()
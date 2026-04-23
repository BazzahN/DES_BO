import argparse
import os
import re
import pandas as pd
import json
import yaml
from pathlib import Path
import matplotlib.pyplot as plt

def extract_hyperparameters(exp_name,af_name):
    # directory containing json files
    dir_name = "Log/hyperparamaters"
    indir = Path(exp_name + f"/{dir_name}")

    # containers
    scalar_rows = []
    vector_rows = []

    # regex to extract m and t from filename
    pattern = re.compile(
    rf"^{re.escape(af_name)}_hyperparams_(\d+)_(\d+)\.json$"
    )

    for filename in os.listdir(indir):
        if not filename.endswith(".json"):
            continue

        match = pattern.match(filename)
        if not match:
            continue

        m = int(match.group(1))
        t = int(match.group(2))

        filepath = os.path.join(indir, filename)

        with open(filepath, "r") as f:
            data = json.load(f)

        # -----------------------
        # Scalar parameters
        # -----------------------
        scalar_row = {
            "m": m,
            "t": t,
            "tau_1": data.get("tau_1"),
            "tau_2": data.get("tau_2"),
            "l_1": data.get("l_1"),
            "l_2": data.get("l_2"),
            "mu_1": data.get("mu_1"),
            "mu_2": data.get("mu_2"),
        }
        scalar_rows.append(scalar_row)

        # -----------------------
        # Vector parameters
        # -----------------------
        mu_u_1 = data.get("mu_u_1", [])
        mu_u_2 = data.get("mu_u_2", [])
        u = data.get("u", [])

        # ensure equal length
        n = min(len(mu_u_1), len(mu_u_2), len(u))

        for i in range(n):
            vector_rows.append({
                "m": m,
                "t": t,
                "index": i,
                "mu_u_1": mu_u_1[i],
                "mu_u_2": mu_u_2[i],
                "u": u[i],
            })

    # -----------------------
    # Create DataFrames
    # -----------------------
    df_GP_params = pd.DataFrame(scalar_rows)
    df_inducing_points = pd.DataFrame(vector_rows)

    # optional: sort
    df_GP_params = df_GP_params.sort_values(["m", "t"]).reset_index(drop=True)
    df_inducing_points  = df_inducing_points .sort_values(["m", "t", "index"]).reset_index(drop=True)

    return df_GP_params,df_inducing_points

def import_exp(exp_file):
	#Obtain Results Location and names
	config_loc = f"configs/{exp_file}.yml"

	with open(config_loc) as f:
		exp_params = yaml.safe_load(f)
	
	exp_name = exp_params['experiment_name']
	return exp_name,exp_params

def main():

    ##Import arguments from the command line
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--af_name",type=str,required=True)
    parser.add_argument("--macro",type=int,required=True)
    args = parser.parse_args()

    exp_name,exp_params = import_exp(args.config)
    af_name = args.af_name

    df_GP_params,df_inducing_points = extract_hyperparameters(exp_name,af_name)
    #Define outdir - to be stored in the same loc as the data
    outdir = Path(exp_name + "/Log")

    #Create plots for selected macro
    m_val = args.macro
    df_m = df_GP_params[df_GP_params["m"] == m_val].sort_values("t")
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharex=True)

    labels = ["Latent GP","Noise GP"]
    ## Outputscale, lengthscale and mean
    # -----------------------
    # Tau plot
    # -----------------------
    axes[0].plot(df_m["t"], df_m["tau_1"], marker='o', label=labels[0])
    axes[0].plot(df_m["t"], df_m["tau_2"], marker='o', label=labels[1])
    axes[0].set_title("Outputscale ($\\tau$)")
    axes[0].set_xlabel("Iteration t")
    axes[0].set_ylabel("Value")
    axes[0].legend()

    # -----------------------
    # Lengthscale plot
    # -----------------------
    axes[1].plot(df_m["t"], df_m["l_1"], marker='o', label=labels[0])
    axes[1].plot(df_m["t"], df_m["l_2"], marker='o', label=labels[1])
    axes[1].set_title("Lengthscales ($l$)")
    axes[1].set_xlabel("Iteration t")
    axes[1].legend()

    # -----------------------
    # Mu plot
    # -----------------------
    axes[2].plot(df_m["t"], df_m["mu_1"], marker='o', label=labels[0])
    axes[2].plot(df_m["t"], df_m["mu_2"], marker='o', label=labels[1])
    axes[2].set_title("Mu")
    axes[2].set_xlabel("Iteration t")
    axes[2].legend()

    plt.tight_layout()
    plt.savefig(outdir / f"hyperparameter_trace_{af_name}_m{m_val}.png",bbox_inches="tight",dpi=200)
    plt.close()



if __name__ == "__main__":
    main()

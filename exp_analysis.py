import torch
from test_utils import Target_function,test_function_2,test_function_neg,test_function,flat_noise,heteroscedastic_noise
tkwargs = {
    "dtype": torch.double,# Datatype used by tensors
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"), # Declares the 'device' location where the Tenosrs will be stored
}

PHI = 1.5
SIGMA2 = 1
direct = 'Comp_01/Data'
names = ['train_x','train_n','train_y','train_sigma2','x_strs','f_strs']
exps = ['vanilla_','IG_']
tag = '.pt'

optim_sol = torch.tensor([0.5531]).to(**tkwargs)

'''
In this script I am going to take the results from these 
small experiments and compare their relative performance.
I will do so by plotting the current best against the number of replications.

For current best I will use both:

- f_str
- x_str - This I will have to scale against the true solution <- probably using euclidian distance


'''

#Extract train_n from both experiments

def name_get(idx_1,idx_2):
    '''
    idx1
    0: BODES
    1: VANIL
    
    idx2
    0: train_x
    1: train_n
    2:train_y
    3:train_sigma2
    4:x_strs
    5:ystrs
    
    
    '''
    return direct + exps[idx_1] + names[idx_2] + tag

#Extract n Data
d_idx = 1 #n index
n_BODES = torch.load(name_get(0,d_idx))
n_AEI = torch.load(name_get(2,d_idx))
n_VANIL = torch.load(name_get(1,d_idx))

#Calculate Cumsum
n_BODES_cumsum = torch.cumsum(n_BODES,dim=1)
n_AEI_cumsum = torch.cumsum(n_AEI,dim=1)
n_VANIL_cumsum = torch.cumsum(n_VANIL,dim=1)

#Remove first 4 values - no longer needed
n_B_trim = n_BODES_cumsum[:,4:,:]
n_A_trim = n_AEI_cumsum[:,4:,:]
n_V_trim = n_VANIL_cumsum[:,4:,:]

#Extract f str Data
d_idx = 5 #f str index
fstr_BODES = torch.load(name_get(0,d_idx))
fstr_VANIL = torch.load(name_get(1,d_idx))
fstr_AEI = torch.load(name_get(2,d_idx))

#Calculate cummax
fstr_BODES_cummax,_ = torch.cummax(fstr_BODES,dim=1)
fstr_VANIL_cummax,_ = torch.cummax(fstr_VANIL,dim=1)
fstr_AEI_cummax,_ = torch.cummax(fstr_AEI,dim=1)

#Extract x str Data
d_idx = 4 #f str index
xstr_BODES = torch.load(name_get(0,d_idx))
xstr_VANIL = torch.load(name_get(1,d_idx))
xstr_AEI = torch.load(name_get(2,d_idx))

target = Target_function(test_function_neg,
                         heteroscedastic_noise,
                         phi=PHI,
                         theta=SIGMA2,)

f_xstr_B = target.eval_target_true(xstr_BODES)
f_xstr_V = target.eval_target_true(xstr_VANIL)
f_xstr_A = target.eval_target_true(xstr_AEI)
#Calculate cumx
f_xstr_B_cm,_ = torch.cummax(f_xstr_B,dim=1)
f_xstr_V_cm,_ = torch.cummax(f_xstr_V,dim=1)
f_xstr_A_cm,_ = torch.cummax(f_xstr_A,dim=1)

#Data shape
# [M,T,1] - M is the number of macros
#           T is the number of iterations

import torch
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

def interpolate_experiment(n_exp, f_exp, num_points=200):
    """
    n_exp, f_exp: tensors of shape [M, T, 1]
    Returns:
        n_grid: common x-grid
        f_mean: mean interpolated f
        f_std: std interpolated f
    """
    # convert to (M, T)
    n = n_exp.squeeze(-1).cpu().numpy()
    f = f_exp.squeeze(-1).cpu().numpy()
    M, T = n.shape

    # build common n-grid (min to max over *all* macroreplications)
    n_min = n.min()
    n_max = n.max()
    n_grid = np.linspace(n_min, n_max, num_points)

    # store all interpolated curves
    f_interp = np.zeros((M, num_points))

    for i in range(M):
        # Ensure n is strictly increasing (required for interpolation)
        order = np.argsort(n[i])
        n_i = n[i, order]
        f_i = f[i, order]

        # interpolation function (linear)
        f_fun = interp1d(n_i, f_i, kind="linear", fill_value="extrapolate")
        f_interp[i] = f_fun(n_grid)

    f_mean = f_interp.mean(axis=0)
    f_std = f_interp.std(axis=0)

    return n_grid, f_mean, f_std


# --- Align shapes: drop first 5 from n_exp so it matches f_exp ---
#Plot the ones that matter
data1 = f_xstr_B_cm
data2 = f_xstr_V_cm

n_grid1, f_mean1, f_std1 = interpolate_experiment(n_B_trim, data1)

n2_np = n_V_trim.squeeze(-1).cpu().numpy()[0]   # identical across reps
f2_np = data2.squeeze(-1).cpu().numpy()

f_mean2 = f2_np.mean(axis=0)
f_std2  = f2_np.std(axis=0,)

sns.set_theme(style="whitegrid", font_scale=1.4)
plt.figure(figsize=(12, 8))

# Experiment 1
plt.plot(n_grid1, f_mean1, label="IG - n selection", linewidth=2.5)
plt.fill_between(n_grid1, f_mean1 - f_std1, f_mean1 + f_std1, alpha=0.2)

# Experiment 2 (direct, no interpolation)
plt.plot(n2_np, f_mean2, label="EI- $n=5$", linewidth=2.5)
plt.fill_between(n2_np, f_mean2 - f_std2, f_mean2 + f_std2, alpha=0.2)

plt.xlim(25,120)
plt.xlabel("Cumulative $n$")
plt.ylabel("$f^*$")
plt.title(f"Performance Across $M=${25} Macroreplications")
plt.legend()
plt.tight_layout()
plt.savefig('IG_EI_new_std.png', dpi=500, bbox_inches="tight") #The ones that matter
plt.show()


# --- Align shapes: drop first 5 from n_exp so it matches f_exp ---

data1 = f_xstr_B_cm
data2 = f_xstr_V_cm
data3 = f_xstr_A_cm

n_grid1, f_mean1, f_std1 = interpolate_experiment(n_B_trim, data1)
n_grid3, f_mean3, f_std3 = interpolate_experiment(n_A_trim, data3)


n2_np = n_V_trim.squeeze(-1).cpu().numpy()[0]   # identical across reps
f2_np = data2.squeeze(-1).cpu().numpy()

f_mean2 = f2_np.mean(axis=0)
f_std2  = f2_np.std(axis=0,)

sns.set_theme(style="whitegrid", font_scale=1.4)
plt.figure(figsize=(12, 8))

# Experiment 1
plt.plot(n_grid1, f_mean1, label="IG - n selection", linewidth=2.5)
plt.fill_between(n_grid1, f_mean1 - f_std1, f_mean1 + f_std1, alpha=0.2)

# Experiment 2 (direct, no interpolation)
plt.plot(n2_np, f_mean2, label="EI- $n=5$", linewidth=2.5)
plt.fill_between(n2_np, f_mean2 - f_std2, f_mean2 + f_std2, alpha=0.2)

# Experiment 3 
plt.plot(n_grid3, f_mean3, label="AEI- n selection", linewidth=2.5)
plt.fill_between(n_grid3, f_mean3 - f_std3, f_mean3 + f_std3, alpha=0.2)

plt.xlim(25,100)
plt.xlabel("Cumulative $n$")
plt.ylabel("$f^*$")
plt.title(f"Performance Across $M=${25} Macroreplications")
plt.legend()
plt.tight_layout()
plt.savefig('IG_EI_AEI_new_std.png', dpi=500, bbox_inches="tight")
plt.show()

# # --- Compute mean trajectory for both experiments ---
# df1 = compute_mean_trajectory(n_B_trim, fstr_BODES_cummax, "AEI")
# df2 = compute_mean_trajectory(n_V_trim, fstr_VANIL_cummax, "EI")

# df_all = pd.concat([df1, df2], ignore_index=True)

# # --- Plot: mean trajectory n̄(t) vs f̄(t) ---
# sns.set_theme(style="whitegrid", font_scale=1.4)
# plt.figure(figsize=(10, 7))

# for label, df in df_all.groupby("experiment"):
#     plt.plot(df.mean_n, df.mean_f, label=label, linewidth=2.5)
    
#     # Uncertainty ribbon: std in both dimensions
#     plt.fill_between(
#         df.mean_n,
#         df.mean_f - df.std_f,
#         df.mean_f + df.std_f,
#         alpha=0.2
#     )

# plt.xlabel("Total $n$ expended")
# plt.ylabel("$f^*$")
# plt.title("Mean Trajectory Across ")
# plt.legend()
# plt.tight_layout()
# plt.show()

# From analysis.ipynb

## Function for plotting hyperparameter traces

# choose macroreplication
m_val = 1

df_m = df_GP_params[df_GP_params["m"] == m_val].sort_values("t")

fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharex=True)

# -----------------------
# Tau plot
# -----------------------
axes[0].plot(df_m["t"], df_m["tau_1"], marker='o', label="tau_1")
axes[0].plot(df_m["t"], df_m["tau_2"], marker='o', label="tau_2")
axes[0].set_title("Tau")
axes[0].set_xlabel("Iteration t")
axes[0].set_ylabel("Value")
axes[0].legend()

# -----------------------
# Lengthscale plot
# -----------------------
axes[1].plot(df_m["t"], df_m["l_1"], marker='o', label="l_1")
axes[1].plot(df_m["t"], df_m["l_2"], marker='o', label="l_2")
axes[1].set_title("Lengthscales (l)")
axes[1].set_xlabel("Iteration t")
axes[1].legend()

# -----------------------
# Mu plot
# -----------------------
axes[2].plot(df_m["t"], df_m["mu_1"], marker='o', label="mu_1")
axes[2].plot(df_m["t"], df_m["mu_2"], marker='o', label="mu_2")
axes[2].set_title("Mu")
axes[2].set_xlabel("Iteration t")
axes[2].legend()

plt.tight_layout()
plt.show()

## Code for plotting hyperparameter heatmap

df_m = df_inducing_points[df_inducing_points["m"] == m_val]

# define bins
u_bins = np.linspace(df_m["u"].min(), df_m["u"].max(), 50)
t_values = sorted(df_m["t"].unique())

heatmap = []

for t in t_values:
    u_vals = df_m[df_m["t"] == t]["u"]
    hist, _ = np.histogram(u_vals, bins=u_bins, density=True)
    heatmap.append(hist)

heatmap = np.array(heatmap)

plt.figure(figsize=(8, 6))

plt.imshow(
    heatmap,
    aspect='auto',
    extent=[u_bins[0], u_bins[-1], t_values[0], t_values[-1]],
    origin='lower'
)

plt.colorbar(label="Density")

plt.xlabel("u")
plt.ylabel("t")
plt.title(f"Inducing point density heatmap (m={m_val})")

plt.show()

## Code for plotting locations of inducing points

plt.figure()

# draw horizontal lines for each iteration
for t_val in df_m["t"].unique():
    plt.axhline(y=t_val, linestyle='--', linewidth=0.5, alpha=0.5)

# plot inducing points
plt.scatter(df_m["u"], df_m["t"], s=25)
plt.axvline(x=1,color='r')
plt.axvline(x=0,color='r')
plt.xlabel("u")
plt.ylabel("t")
plt.title(f"Inducing points across iterations (m={m_val})")

plt.show()

## Code For plotting inducing point means against locations

fig, axes = plt.subplots(1, len(df_m["t"].unique()), figsize=(15, 4), sharey=True)

for ax, (t_val, group) in zip(axes, df_m.groupby("t")):
    group = group.sort_values("u")
    
    ax.plot(group["u"], group["mu_u_1"], label="mu_u_1")
    ax.plot(group["u"], group["mu_u_2"], label="mu_u_2")
    
    ax.set_title(f"t={t_val}")
    ax.set_xlabel("u")

axes[0].set_ylabel("value")
axes[0].legend()

plt.tight_layout()
plt.show()
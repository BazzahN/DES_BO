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



# Formally from app.py
# Used to plot stuff as a streamlit app, mainly taken from chatgpt code

import streamlit as st
import os
import time
from PIL import Image
# import imageio

# --- CONFIG ---
IMAGE_DIR = "images"

plot_types = [
    ("GP", "GP Plot", "GP_plot_"),
    ("Uncer", "Uncertainty", "Uncer_plot_"),
    ("Noise", "Noise", "Noise_plot_"),
    ("AF", "Acquisition", "AF_plot_"),
    ("Improv", "Improvement", "improv_plot_"),
]

# Auto-discover max iteration

def discover_max_iteration():
    max_iter = 0
    files = os.listdir(IMAGE_DIR)
    for _, _, prefix in plot_types:
        for f in files:
            if f.startswith(prefix) and f.endswith(".png"):
                try:
                    it = int(f[len(prefix):-4])
                    max_iter = max(max_iter, it)
                except:
                    pass
    return max_iter

max_iter = discover_max_iteration()

# st.title("Simulation Plot Series Viewer — Full Feature Edition 🚀")

# --- SIDEBAR ---
st.sidebar.header("Controls")
series = st.sidebar.selectbox(
    "Series", [p[1] for p in plot_types], key="series_select"
)
prefix = next(p for p in plot_types if p[1] == series)[2]

iteration = st.sidebar.number_input(
    "Iteration", min_value=0, max_value=max_iter, value=0, step=1, key="iteration"
)
# Advanced Jump Controls

# Playback controls
speed = st.sidebar.slider("Speed (seconds/frame)", 0.02, 1.0, 0.15)
loop = st.sidebar.checkbox("Loop playback")
reverse_mode = st.sidebar.checkbox("Reverse playback")

play = st.sidebar.button("Play ▶️")
pause = st.sidebar.button("Stop ⏹️")

# --- DISPLAY ---
# Comparison mode
st.sidebar.header("Comparison Mode")
enable_compare = st.sidebar.checkbox("Enable comparison between series")
compare_selection = []
if enable_compare:
    for label in [p[1] for p in plot_types]:
        if st.sidebar.checkbox(label, value=(label == series)):
            compare_selection.append(label)
else:
    compare_selection = [series]

# Display comparison in 2xn grid
num = len(compare_selection)
cols_per_row = 3
rows = (num + cols_per_row - 1) // cols_per_row

row_idx = 0
col_idx = 0
for idx, ser in enumerate(compare_selection):
    if col_idx == 0:
        cols = st.columns(cols_per_row)

    pref = next(p for p in plot_types if p[1] == ser)[2]
    path = os.path.join(IMAGE_DIR, f"{pref}{iteration}.png")

    with cols[col_idx]:
        #st.subheader(f"{ser} — Iteration {iteration}")
        if os.path.exists(path):
            st.image(path)
        else:
            st.warning(f"Missing: {path}")

    col_idx += 1
    if col_idx >= cols_per_row:
        col_idx = 0
        row_idx += 1


# --- PLAYBACK FUNCTION ---

def run_playback(start_iter):
    placeholder = st.empty()

    while True:
        frame_range = range(start_iter, -1, -1) if reverse_mode else range(start_iter, max_iter + 1)

        for i in frame_range:
            if st.session_state.get("stop", False):
                st.session_state["stop"] = False
                return

            path = os.path.join(IMAGE_DIR, f"{prefix}{i}.png")
            with placeholder.container():
                st.subheader(f"{series} — Iteration {i}")
                if os.path.exists(path):
                    st.image(path)
                else:
                    st.warning(f"Missing: {path}")
            time.sleep(speed)

        if not loop:
            break

# --- PLAYBACK TRIGGER ---
if play:
    st.session_state["stop"] = False
    run_playback(iteration)

if pause:
    st.session_state["stop"] = True

# # --- EXPORT GIF/MP4 ---
# st.sidebar.header("Export Animation")
# export = st.sidebar.button("Export GIF 🎞️")

# if export:
#     frames = []
#     for i in range(0, max_iter + 1):
#         path = os.path.join(IMAGE_DIR, f"{prefix}{i}.png")
#         if os.path.exists(path):
#             frames.append(Image.open(path))
#     gif_path = f"{prefix}_animation.gif"
#     imageio.mimsave(gif_path, frames, fps=int(1/speed))
#     st.sidebar.success(f"Exported: {gif_path}")
#     with open(gif_path, "rb") as f:
#         st.sidebar.download_button("Download GIF", f, file_name=gif_path)

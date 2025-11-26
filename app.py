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

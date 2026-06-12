import numpy as np
import torch as st  
from pathlib import Path
from exp_utils import get_files
DIR = "BO_simple_k15"
indir = Path(DIR + "/Data")

m=40

data = get_files(indir=indir,file_names=['train_x','train_y'],suffix=f"_m{m}")
test_data = get_files(indir=Path(DIR + "/Input"),file_names=['test_x','test_y','test_sigma2'])

train_x = data['train_x'].numpy().flatten()
train_y = data['train_y'].numpy().flatten()



outdir=Path("/home/newtonh3/hetGPy/bo_data") 
outdir.mkdir(exist_ok=True)

np.save(outdir / f"train_x_m{m}.npy", train_x)
np.save(outdir / f"train_y_m{m}.npy", train_y)

test_x = test_data['test_x'].numpy().flatten()
test_y = test_data['test_y'].numpy().flatten()
test_sigma2 = test_data['test_sigma2'].numpy().flatten()
np.save(outdir / "test_x.npy", test_x)
np.save(outdir / "test_y.npy", test_y)
np.save(outdir / "test_sigma2.npy", test_sigma2)
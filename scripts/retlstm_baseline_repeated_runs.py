import torch
from torch import nn
import numpy as np
from tqdm import tqdm
from torch.cuda.amp import autocast, GradScaler
from pathlib import Path
from argparse import ArgumentParser
import os

from ret_lstm.models import (
    get_ConvModuleCC,
    SimpleModelCC,
    get_ConvModuleNCC,
    SimpleModelNCC,
    PatchModel,
    get_ConvModulePatch
)

from utils.misc import get_layer_mask_from_boundaries, get_loaders_retlstm, tflog2pandas
from utils.validate import validate_retlstm
import logging
import datetime

parser = ArgumentParser()
parser.add_argument("--gpu", type=int, default=0)
parser.add_argument("--cc", action="store_true")
parser.add_argument("--loss_type", choices=["mse", "mae"], default="mse")
parser.add_argument("--nl", type=int, default=4)
parser.add_argument("--ks", type=int, default=15)
parser.add_argument("--st", type=int, default=2)
parser.add_argument("--bs", type=int, default=16)
parser.add_argument("--lr", type=float, default=1e-3)
parser.add_argument("--num_epochs", type=int, default=400)
parser.add_argument("--num_runs", type=int, default=1)
parser.add_argument("--enc_type", choices=["none", "cc", "sin"], default="none")
parser.add_argument("--wlen", type=int, default=2)
parser.add_argument("--neighbors", type=int, default=0)
parser.add_argument("--fixed_seed", type=bool, default=True)
args = parser.parse_args()

if args.fixed_seed:
    torch.manual_seed(0)
CUDA_VISIBLE_DEVICES = args.gpu
checkpoints = Path(os.getenv("CKPS"))
runs = Path(os.getenv("RUNS"))
data_dir = Path(os.getenv("DATA_DIR"))
logs = Path(os.getenv("LOGS"))

logging.basicConfig(filename=logs / "retlstm-baselines.log", level=logging.INFO)

device = "cuda" if torch.cuda.is_available() else "cpu"

learning_rate = args.lr
batch_size = args.bs
kernel_size = args.ks
n_layer = args.nl
n_stride = args.st
cc = args.cc
loss_type = args.loss_type
num_epochs = args.num_epochs
num_runs = args.num_runs
neighbors = args.neighbors

model_class = SimpleModelCC if cc == True else SimpleModelNCC
model_getter = get_ConvModuleCC if cc == True else get_ConvModuleNCC

train_dl, valid_dl, mean_std = get_loaders_retlstm(data_dir, 64, batch_size, None, 2)
ces = []

for run in range(num_runs):
    ce_run = []
    conv, in_features = get_ConvModulePatch(n_layer, kernel_size, n_stride, neighbors)
    model = PatchModel(nn.Sequential(conv, nn.Linear(in_features, 8)), neighbors).to(device)
    loss_fn = nn.MSELoss() if loss_type == "mse" else nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate,)
    scaler = GradScaler()
    num_batches = len(train_dl)
    for epoch in range(num_epochs):
        train_loop = tqdm(train_dl, desc=f"Run [{run} / {num_runs}]; Epoch [{epoch}/{num_epochs}]:", leave=False)
        for batch_idx, (x, y) in enumerate(train_loop):
            x, y = x.swapaxes(0, 1).to(device), y.swapaxes(0, 1).to(device).float()
            optimizer.zero_grad()
            with autocast():
                pred = model(x)
                pred = pred[~y.isnan()]
                y = y[~y.isnan()]
                loss = loss_fn(pred, y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            loss = loss.item()
            train_loop.set_postfix(loss=loss)
        loss, dice, ce, mad = validate_retlstm(model, valid_dl, loss_fn, mean_std, 9)
        ce_run.append(ce.mean())
    ces.append(ce_run)

ces = np.array(ces)
ce_corred = np.stack([np.correlate(curr_ce, np.ones(10) / 10) for curr_ce in ces])

msg = (
    f"CE l-m: {round(ces[:, -1].mean(), 3)}, v: {round(ces[:, -1].var(), 3)}; "
    f"a-m: {round(ces.min(-1).mean(), 3)}, v: {round(ces.min(-1).var(), 3)}; "
    f"a-a-m: {round(ces.argmin(-1).mean(), 3)}, v: {round(ces.argmin(-1).var(), 3)}"
    f"s-m: {round(ce_corred.min(-1).mean(), 3)}, v: {round(ce_corred.min(-1).var(), 3)}; "
    f"s-a-m: {round(ce_corred.argmin(-1).mean(), 3)}, v: {round(ce_corred.argmin(-1).var(), 3)}"
)
log = (
    "--------------------------------------\n"
    f"filename: {__file__}  {str(datetime.datetime.now()).split('.')[0]}\n"
    f"params: lr-{learning_rate}, bs-{batch_size}, ks-{kernel_size}, nl-{n_layer}, ns-{n_stride}\n"
    f"params: cc-{cc}, loss-{loss_type}, nb-{neighbors}, nume-{num_epochs}, numr-{num_runs}\n"
    f"output: {msg}"
)
logging.info(log)
print(msg)

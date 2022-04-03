import torch
from torch import nn
import numpy as np
from tqdm import tqdm
from torch.cuda.amp import autocast, GradScaler
from pathlib import Path
import albumentations as A
import itertools
from argparse import ArgumentParser
import os

from ret_lstm.models import (
    get_ConvModuleCC,
    SimpleModelCC,
    get_ConvModuleNCC,
    SimpleModelNCC
)
from utils.data import RetLSTMDataset
from utils.misc import get_layer_mask_from_boundaries, get_loaders_retlstm, tflog2pandas
from utils.validate import validate_retlstm

parser = ArgumentParser()
parser.add_argument("--gpu", type=int, default=0)
parser.add_argument("--cc", action="store_true")
parser.add_argument("--loss", choices=["mse", "mae"], default="mse")
parser.add_argument("--num_trials", type=int, default=10)
parser.add_argument("--num_epochs", type=int, default=400)
parser.add_argument("--target", type=float, default=4.)
parser.add_argument("--nl", type=int, default=4)
parser.add_argument("--ks", type=int, default=15)
parser.add_argument("--st", type=int, default=2)
parser.add_argument("--bs", type=int, default=16)
parser.add_argument("--lr", type=float, default=1e-3)
args = parser.parse_args()

checkpoints = Path(os.getenv("CKPS"))
runs = Path(os.getenv("RUNS"))
data_dir = Path(os.getenv("DATA_DIR"))

num_epochs = 400
learning_rate = args.lr
batch_size = args.bs
kernel_size = args.ks
n_layer = args.nl
n_stride = args.st
ce_target = args.target
num_trials = args.num_trials
cc = args.cc
loss = args.loss

device = "cuda"
CUDA_VISIBLE_DEVICES = args.gpu
tag = f"pretrained/{'CC' if cc else 'NCC'}"
config = f"nl_{n_layer}_ks_{kernel_size}_st_{n_stride}_ce_{ce_target}"
model_class = SimpleModelCC if cc == True else SimpleModelNCC
model_getter = get_ConvModuleCC if cc == True else get_ConvModuleNCC
train_dl, valid_dl, mean_std = get_loaders_retlstm(data_dir, 64, batch_size, None, 2)

break_out = False
for trial in range(num_trials):
    ce_min = 1e3
    conv, in_features = model_getter(n_layer, kernel_size, n_stride)
    model = model_class(nn.Sequential(conv, nn.Linear(in_features, 8))).to(device)
    loss_fn = nn.MSELoss() if loss == "mse" else nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, )
    scaler = GradScaler()
    num_batches = len(train_dl)
    for epoch in range(num_epochs):
        train_loop = tqdm(
            train_dl, desc=f"Run [{trial} / {num_trials}]; Epoch [{epoch}/{num_epochs}]:",
            leave=False
            )
        for batch_idx, (x, y) in enumerate(train_loop):
            x, y = x.swapaxes(0, 1).to(device), y.permute((-1, 0, 1)).to(device).float()
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
        loss, ce, mad = validate_retlstm(model, valid_dl, loss_fn, mean_std, 9)
        curr_ce = ce.mean()
        ce_min = ce_min if ce_min < curr_ce else curr_ce
        if curr_ce < ce_target:
            checkpoint_folder = checkpoints / f"retlstm/{tag}/"
            checkpoint_folder.mkdir(exist_ok=True, parents=True)
            saved_obj = {
                "params": conv.state_dict(),
                "ce": curr_ce,
                "trial": trial,
                "lr": learning_rate,
                "bs:": batch_size
            }
            torch.save(saved_obj, checkpoint_folder / f"{config}.pth")
            print(f"Reached {ce.mean()} in {trial}")
            break_out = True
        if break_out:
            break
    if break_out:
        break
    print(f"Lowest in the trial: {ce_min}")

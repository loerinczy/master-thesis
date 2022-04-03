import torch
from torch import nn
import numpy as np
from tqdm import tqdm
from torch.cuda.amp import autocast, GradScaler
from pathlib import Path
from ret_lstm.models import (
    RetLSTM,
    get_ConvModuleCC,
    SimpleModelCC,
    SimpleModelNCC,
    get_ConvModuleNCC
)
from utils.misc import get_loaders_retlstm
from utils.validate import validate_retlstm
from argparse import ArgumentParser
import os
from torch.utils.tensorboard import SummaryWriter

parser = ArgumentParser()
parser.add_argument("--gpu", type=int, default=0)
parser.add_argument("--cc", action="store_true")
parser.add_argument("--loss", choices=["mse", "mae"], default="mse")
parser.add_argument("--pretrained", action="store_true")
parser.add_argument("--nl", type=int, default=4)
parser.add_argument("--ks", type=int, default=15)
parser.add_argument("--st", type=int, default=2)
parser.add_argument("--log2_bs_min", type=int, default=2)
parser.add_argument("--log2_bs_max", type=int, default=5)
parser.add_argument("--log10_lr_max", type=float, default=-2)
parser.add_argument("--log10_lr_min", type=float, default=-5)
parser.add_argument("--num_comb", type=int, default=20)
parser.add_argument("--pw", type=int, default=64)
parser.add_argument("--fc_depth_min", type=int, default=1)
parser.add_argument("--fc_depth_max", type=int, default=1)
parser.add_argument("--log2_n_hidden_min", type=int, default=5)
parser.add_argument("--log2_n_hidden_max", type=int, default=8)
parser.add_argument("--ce_target", type=float, default=4)
parser.add_argument("--num_epochs", type=int, default=400)
parser.add_argument("--shbe", type=int, default=100)
parser.add_argument("--smene", type=int, default=50)
args = parser.parse_args()

CUDA_VISIBLE_DEVICES = args.gpu
checkpoints = Path(os.getenv("CKPS"))
runs = Path(os.getenv("RUNS"))
data_dir = Path(os.getenv("DATA_DIR"))
device = "cuda" if torch.cuda.is_available() else "cpu"

batch_size = args.bs
kernel_size = args.ks
n_layer = args.nl
n_stride = args.st
cc = args.cc
loss = args.loss
pretrained = args.pretrained
if pretrained:
    ce_target = args.ce_target
fc_depth_min = args.fc_depth_min
log2_n_hidden_min = args.log2_n_hidden_min
log2_n_hidden_max = args.log2_n_hidden_max
log10_
num_epochs = args.num_epochs
num_comb = args.num_comb
store_hparams_before_end = args.shbe
save_model_every_n_epochs = args.smene
patch_width = args.pw

tag = f"lstm/{'pretrained' if pretrained else 'notpretrained'}"
model_getter = get_ConvModuleCC if cc == True else get_ConvModuleNCC

conv, in_features = model_getter(n_layer, kernel_size, n_stride)
if pretrained:
    load_config = f"nl_{n_layer}_ks_{kernel_size}_st_{n_stride}_ce_{ce_target}.pth"
    conv.load_state_dict(
        torch.load(
          checkpoints / f"retlstm/pretrained/{'CC' if cc else 'NCC'}/{load_config}"
        )["params"]
    )
conv_head = nn.Sequential(conv, nn.Linear(in_features, n_hidden))
fc = nn.Sequential(*([i for j in [(nn.Linear(n_hidden, n_hidden), nn.ReLU(True))
                        for _ in range(fc_depth - 1)] for i in j]
                    + [nn.Linear(n_hidden, 8),])
)
model = RetLSTM(conv_head, fc, n_hidden).to(device)
lr_str = np.format_float_scientific(learning_rate, 2)
train_dl, valid_dl, mean_std = get_loaders_retlstm(data_dir, patch_width, batch_size, None, 2)
loss_fn = nn.MSELoss() if loss == "mse" else nn.L1Loss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate,)
scaler = GradScaler()
config = f"lr_{lr_str}_bs_{batch_size}_nh_{n_hidden}_pw_{patch_width}_fd_{fc_depth}"
checkpoint_folder = checkpoints / f"retlstm/{tag}/{config}"
checkpoint_folder.mkdir(exist_ok=True, parents=True)
writer = SummaryWriter(runs / f"retlstm/{tag}/{config}")
num_batches = len(train_dl)
ces = []
for epoch in range(num_epochs):
    train_loop = tqdm(train_dl, desc=f"Epoch [{epoch}/{num_epochs}]:", leave=False)
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
        writer.add_scalar("Train/loss", loss, epoch * num_batches + batch_idx)
    loss, dice, ce, mad = validate_retlstm(model, valid_dl, loss_fn, mean_std, 9)
    ces.append(ce.mean())
    writer.add_scalar("Valid/loss", loss, epoch)
    writer.add_scalar("Valid/dice", dice.mean(), epoch)
    writer.add_scalar("Valid/ce", ce.mean(), epoch)
    writer.add_scalar("Valid/mad", mad.mean(), epoch)
    if num_epochs - epoch <= store_hparams_before_end:
        writer.add_hparams(
            {"lr": float(lr_str), "bs": batch_size, "ks": kernel_size, "nl": n_layer, "st": n_stride, "nh": n_hidden, "pw": patch_width},
            {"dice": dice.mean(), "ce": ce.mean(), "mad": mad.mean(), "loss": loss},
            run_name="run"
        )
        if epoch % save_model_every_n_epochs == 0 or epoch == num_epochs - 1:
            torch.save(model.state_dict(), checkpoint_folder / f"epoch_{epoch}.pth")

ces = np.array(ces)
print(f"CE min: {ces.min()} reached: {ces.argmin()}")
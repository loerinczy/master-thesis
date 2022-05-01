import torch
from torch import nn
import numpy as np
from tqdm import tqdm
from torch.cuda.amp import autocast, GradScaler
from pathlib import Path
from ret_lstm.models import (
    RetLSTM,
    get_ConvModulePatch,
    PatchModel,
    PatchModelPosEnc
)
from utils.misc import get_loaders_retlstm
from utils.validate import validate_retlstm
from argparse import ArgumentParser
import os
from torch.utils.tensorboard import SummaryWriter
import warnings
import logging
import datetime
import pickle
import shutil
from itertools import product
warnings.filterwarnings("ignore")

parser = ArgumentParser()
parser.add_argument("--loss", choices=["mae", "smooth"], default="mae")
parser.add_argument("--smooth_beta", type=float, default=1.)
parser.add_argument("--pretrained", action="store_true")
parser.add_argument("--nl", type=int, default=4)
parser.add_argument("--ks", type=int, default=15)
parser.add_argument("--st", type=int, default=2)
parser.add_argument("--bs", type=str, default="16")
parser.add_argument("--lr", type=str, default="1e-3")
parser.add_argument("--pw", type=str, default="64")
parser.add_argument("--num_workers", type=int, default=2)
parser.add_argument("--fc_depth", type=int, default=1)
parser.add_argument("--n_hidden", type=int, default=128)
parser.add_argument("--ce_target", type=float, default=4.0)
parser.add_argument("--num_epochs", type=int, default=200)
parser.add_argument("--num_runs", type=int, default=1)
parser.add_argument("--pos_enc", choices=["none", "cc", "sin"], default="none")
parser.add_argument("--sum_coord", action="store_true")
parser.add_argument("--scale", type=float, default=1.)
parser.add_argument("--wlens", type=int, default=2)
parser.add_argument("--cc_type", choices=["center", "minmax", "standardized"], default="center")
parser.add_argument("--forget_bias", type=int, default=10)
parser.add_argument("--neighbors", type=int, default=0)
parser.add_argument("--weight", type=float, default=1.)
parser.add_argument("--fixed_seed", type=bool, default=True)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--no_report", action="store_true")
parser.add_argument("--shbe", type=int, default=100)
parser.add_argument("--smene", type=int, default=50)
parser.add_argument("--comment", type=str, default="")
args = parser.parse_args()

if args.fixed_seed:
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.deterministic = True
pub = Path(os.getenv("PUB"))
checkpoints = pub / "checkpoints"
runs = pub / "runs"
data_dir = pub / "corrected_ds"
logs = Path(os.getenv("LOGS"))
device = "cuda" if torch.cuda.is_available() else "cpu"
logging.basicConfig(filename=logs / "retlstm.log", level=logging.INFO)

learning_rates = list(map(float, args.lr.split(",")))
batch_sizes = list(map(int, args.bs.split(",")))
patch_widths = list(map(int, args.pw.split(",")))
kernel_size = args.ks
n_layer = args.nl
n_stride = args.st
loss_type = args.loss
pretrained = args.pretrained
if pretrained:
    ce_target = args.ce_target
fc_depth = args.fc_depth
n_hidden = args.n_hidden
num_epochs = args.num_epochs
neighbors = args.neighbors
pos_enc = args.pos_enc
sum_coord = args.sum_coord
scale = args.scale
wlens = args.wlens
cc_type = args.cc_type
forget_bias = args.forget_bias
store_hparams_before_end = args.shbe
save_model_every_n_epochs = args.smene
weight = args.weight
report = not args.no_report
num_runs = args.num_runs
num_workers = args.num_workers
comment = args.comment

loss_factory = {
    "mae": nn.L1Loss(),
    "smooth": nn.SmoothL1Loss(beta=args.smooth_beta)
}

loss_fn = loss_factory[loss_type]
loss_weight = torch.tensor([1, weight, weight, weight, weight, 1, 1, 1]).to(device)
loss_weight = loss_weight / loss_weight.sum() * 8

parameters = product(
          learning_rates,
          batch_sizes,
          patch_widths,
)

for curr_param in parameters:
    ces = []
    learning_rate, batch_size, patch_width = curr_param
    if report:
        tag = (
            f"lstm/{'pret' if pretrained else 'notp'}_"
            f"nl{n_layer}_ks{kernel_size}_st{n_stride}"
        )
        lr_str = np.format_float_scientific(learning_rate, 2)
        config = f"lr{lr_str}_bs{batch_size}_nh{n_hidden}_pw{patch_width}_fd{fc_depth}_nb{neighbors}_{comment}"
        checkpoint_folder = checkpoints / f"{tag}/{config}"
        if checkpoint_folder.exists():
            shutil.rmtree(checkpoint_folder, ignore_errors=True)
            shutil.rmtree(runs / f"{tag}/{config}", ignore_errors=True)
        checkpoint_folder.mkdir(exist_ok=True, parents=True)
        writer = SummaryWriter(runs / f"{tag}/{config}")
    start = datetime.datetime.now()
    train_dl, valid_dl, mean_std = get_loaders_retlstm(
              data_dir, patch_width, batch_size, None,
              normalize_img=True,
              normalize_lyr=True,
              standardize=True,
              num_workers=num_workers
    )
    num_batches = len(train_dl)
    for run in range(num_runs):
        ce_run = []
        conv, in_features = get_ConvModulePatch(
                  n_layer, kernel_size, n_stride, 1 if pos_enc == "none" or sum_coord else 2,
                  neighbors
        )
        if pretrained:
            load_config = f"nl_{n_layer}_ks_{kernel_size}_st_{n_stride}_nb_{neighbors}_ce_{ce_target}"
            checkpoint_folder = checkpoints / f"retlstm/pretrained/pos_enc_{pos_enc}"
            if pos_enc == "cc":
                checkpoint_folder = checkpoint_folder / f"cc_type_{cc_type}"
            elif pos_enc == "sin":
                checkpoint_folder = checkpoint_folder / f"wlens_{wlens}"
            if pos_enc in ("cc", "sin"):
                checkpoint_folder = checkpoint_folder / f"scale_{scale}_sum_coord_{sum_coord}"
            conv.load_state_dict(
                      torch.load(
                                checkpoint_folder / f"{load_config}.pth"
                      )["params"]
            )
        if pos_enc in ("cc", "sin"):
            pm = PatchModelPosEnc(
                      conv=conv,
                      fc=nn.Linear(in_features, n_hidden),
                      neighbors=neighbors,
                      pos_enc=pos_enc,
                      cc_type=cc_type,
                      sum_coord=sum_coord,
                      scale=scale,
                      wlens=wlens
            ).to(device)
        else:
            pm = PatchModel(
                      conv=conv,
                      fc=nn.Linear(in_features, n_hidden),
                      neighbors=neighbors
            ).to(device)
        fc = nn.Sequential(
            *([i for j in [(nn.Linear(n_hidden, n_hidden), nn.ReLU(True))
                           for _ in range(fc_depth - 1)] for i in j]
              + [nn.Linear(n_hidden, 8), ])
        )
        model = RetLSTM(pm, fc, n_hidden, forget_bias).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate,)
        scaler = GradScaler()
        for epoch in range(num_epochs):
            train_loop = tqdm(
                      train_dl,
                      desc=f"R[{run + 1}/{num_runs}]; E[{epoch}/{num_epochs}]", leave=False
            )
            for batch_idx, (x, y, corner) in enumerate(train_loop):
                x, y, corner = (
                    x.swapaxes(0, 1).to(device).float(),
                    y.swapaxes(0, 1).to(device).float(),
                    corner.swapaxes(0, 1).to(device)
                )
                optimizer.zero_grad()
                with autocast():
                    x[corner != 0] = 0
                    pred = model(x)
                    pred = pred[~y.isnan()]
                    lw = loss_weight.expand_as(y)[~y.isnan()]
                    y = y[~y.isnan()]
                    loss = (loss_fn(pred, y) * lw).mean()
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                loss = loss.item()
                if report:
                    writer.add_scalar("Train/loss", loss, epoch * num_batches + batch_idx)
                train_loop.set_postfix(loss=loss)
            if epoch % 2 == 0 or num_epochs - epoch == 1:
                loss, dice, ce, mad = validate_retlstm(
                          model, valid_dl, loss_fn,
                          normalize_lyr=True,
                          standardize=True,
                          loss_weight=loss_weight,
                          mean_std=mean_std,
                          num_classes=9
                )
                curr_ce = ce.mean()
                ce_run.append(curr_ce)
                if report:
                    writer.add_scalar("Valid/loss", loss, epoch)
                    writer.add_scalar("Valid/dice", dice.mean(), epoch)
                    writer.add_scalar("Valid/ce", ce.mean(), epoch)
                    writer.add_scalar("Valid/mad", mad.mean(), epoch)
                    if num_epochs - epoch <= store_hparams_before_end:
                        writer.add_hparams(
                                  {"lr": learning_rate, "bs": batch_size, "ks": kernel_size,
                                   "nl": n_layer, "st": n_stride, "nh": n_hidden,
                                   "pw": patch_width, "fcd": fc_depth},
                                  {"dice": dice.mean(), "ce": ce.mean(), "mad": mad.mean(),
                                   "loss": loss},
                                  run_name="run"
                        )
                    if epoch != 0 and epoch % save_model_every_n_epochs == 0:
                        torch.save(model.state_dict(), checkpoint_folder / f"epoch_{epoch}.pth")
        ces.append(ce_run)
    end = datetime.datetime.now()
    ces = np.array(ces)
    ce_corred = np.stack([np.correlate(curr_ce, np.ones(10) / 10) for curr_ce in ces])
    msg = (
        f"CE l-m: {round(ces[:, -1].mean(), 3)}, v: {round(ces[:, -1].var(), 3)}; "
        f"a-m: {round(ces.min(-1).mean(), 3)}, v: {round(ces.min(-1).var(), 3)}; "
        f"a-a-m: {round(ces.argmin(-1).mean(), 3)}, v: {round(ces.argmin(-1).var(), 3)} "
        f"s-m: {round(ce_corred.min(-1).mean(), 3)}, v: {round(ce_corred.min(-1).var(), 3)}; "
        f"s-a-m: {round(ce_corred.argmin(-1).mean(), 3)}, v: {round(ce_corred.argmin(-1).var(), 3)} "
        f"total min: {ces.min()}"
    )
    log = (
        "--------------------------------------\n"
        f"filename: {__file__}  {str(datetime.datetime.now()).split('.')[0]}\n"
        f"params: lr-{learning_rate}, bs-{batch_size}, ks-{kernel_size}, nl-{n_layer}, ns-{n_stride}\n"
        f"params: enc-{pos_enc}, loss-{loss_type}, nb-{neighbors}, nume-{num_epochs}, numr-{num_runs}\n"
        f"params: cc_type-{cc_type}, sum_coord-{sum_coord}, scale-{scale}, wlens-{wlens}\n"
        f"params: pretrained-{pretrained}, n_hidden-{n_hidden}, fc_depth-{fc_depth},"
        f"params: pw-{patch_width}, forget_bias-{forget_bias}"
        f"output: {msg}"
    )
    logging.info(log)
    print(msg)

end = datetime.datetime.now()
print(f"run_time: {end-start}")

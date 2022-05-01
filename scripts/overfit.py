import warnings
warnings.filterwarnings("ignore")
import pickle
import torch
from torch import nn
import numpy as np
from tqdm import tqdm
from torch.cuda.amp import autocast, GradScaler
from pathlib import Path
from argparse import ArgumentParser
import os
from torch.utils.tensorboard import SummaryWriter
import datetime
from utils.misc import get_loaders_retlstm
from utils.validate import validate_retlstm
import shutil
from itertools import product

parser = ArgumentParser()
parser.add_argument("--loss", choices=["mae", "smooth"], default="mae")
parser.add_argument("--bs", type=int, default=16)
parser.add_argument("--lr", type=float, default=1e-3)
parser.add_argument("--pw", type=int, default=64)
parser.add_argument("--num_epochs", type=int, default=50)
parser.add_argument("--fixed_seed", type=bool, default=True)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--smooth_beta", type=float, default=1e-2)
parser.add_argument("--weight", type=float, default=1.)
parser.add_argument("--model_pkl", type=str, default="")
parser.add_argument("--n_batches", type=int, default=1)
parser.add_argument("--comment", type=str, default="")
parser.add_argument("--idp", type=float, default=0)
parser.add_argument("--odp", type=float, default=0)
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
train_models = Path("train_models")
ready_models = Path("ready_models")
device = "cuda" if torch.cuda.is_available() else "cpu"
num_workers = 2

learning_rate = args.lr
batch_size = args.bs
patch_width = args.pw
loss_type = args.loss
num_epochs = args.num_epochs
weight = args.weight
model_name = args.model_pkl
n_batches = args.n_batches
comment = args.comment
idp = args.idp
odp = args.odp

loss_factory = {
    "mae": nn.L1Loss(reduction="none"),
    "smooth": nn.SmoothL1Loss(beta=args.smooth_beta)
}


ces = []
loss_fn = loss_factory[loss_type]
loss_weight = torch.tensor([1, weight, weight, weight, weight, 1, 1, 1]).to(device)
loss_weight = loss_weight / loss_weight.sum() * 8

start = datetime.datetime.now()
train_dl, valid_dl, mean_std = get_loaders_retlstm(
          data_dir, patch_width, batch_size, None,
          normalize_img=True,
          normalize_lyr=True,
          standardize=True,
          num_workers=num_workers
)
valid_dl = [k for n, k in enumerate(valid_dl) if n < n_batches]
num_batches = len(valid_dl)
ce_run = []
try:
    with open(train_models / f"{model_name}.pkl", "rb") as model_file:
        model = pickle.load(model_file).to(device)
    # if idp != 0 or odp != 0:
    #     model.init_dropout(idp, odp)
    tag = f"overfit/{model_name}"
    lr_str = np.format_float_scientific(learning_rate, 2)
    config = f"lr{lr_str}_bs{batch_size}_pw{patch_width}_{comment}"
    checkpoint_folder = checkpoints / f"{tag}/{config}"
    if checkpoint_folder.exists():
        shutil.rmtree(checkpoint_folder, ignore_errors=True)
        shutil.rmtree(runs / f"{tag}/{config}", ignore_errors=True)
    checkpoint_folder.mkdir(exist_ok=True, parents=True)
    writer = SummaryWriter(runs / f"{tag}/{config}")
    optimizer = torch.optim.Adam(
              model.parameters(),
              lr=learning_rate,
    )
    scaler = GradScaler()
    for epoch in range(num_epochs):
        train_loop = tqdm(
                  valid_dl,
                  desc=f"E[{epoch}/{num_epochs}]", leave=False
        )
        for batch_idx, (x, y, corner, mask) in enumerate(train_loop):
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
            writer.add_scalar("Train/loss", loss, epoch * num_batches + batch_idx)
            train_loop.set_postfix(loss=loss)
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
        writer.add_scalar("Valid/loss", loss, epoch)
        writer.add_scalar("Valid/dice", dice.mean(), epoch)
        writer.add_scalar("Valid/ce", ce.mean(), epoch)
        writer.add_scalar("Valid/mad", mad.mean(), epoch)
    ces = [ce_run]
    with open(
              ready_models / f"{model_name}_overfit.pkl", "wb"
    ) as model_file:
        pickle.dump(model.cpu(), model_file)
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
    print(msg)
except Exception as e:
    print(e)
    print(f"Error with model {model_name}!")

end = datetime.datetime.now()
print(f"run_time: {end-start}")

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
import albumentations as A
from utils.misc import initialize
from ret_lstm.models import (
    RetLSTM,
    get_ConvModulePatch,
    PatchModel2d,
    PatchModelPosEnc2d,
    PatchModel,
    PatchModelPosEnc,
    Block2d,
    Block1d,
    SkipConnConv,
    RetinaConv2d
)
parser = ArgumentParser()
parser.add_argument("--loss", choices=["mae", "smooth"], default="mae")
parser.add_argument("--bs", type=str, default="16")
parser.add_argument("--lr", type=str, default="1e-3")
parser.add_argument("--pw", type=str, default="64")
parser.add_argument("--num_epochs", type=int, default=300)
parser.add_argument("--num_runs", type=int, default=1)
parser.add_argument("--fixed_seed", type=bool, default=True)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--smooth_beta", type=float, default=1e-2)
parser.add_argument("--weight", type=float, default=1.)
parser.add_argument("--model_pkl", type=str, default="")
parser.add_argument("--comment", type=str, default="")
parser.add_argument("--smene", type=int, default=50)
parser.add_argument("--weight_decay", type=float, default=0.)
parser.add_argument("--hflip", action="store_true")
parser.add_argument("--no_init", action="store_true")
parser.add_argument("--init_type", type=str, default="def")
parser.add_argument("--boundary_center", action="store_true")
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

learning_rates = list(map(float, args.lr.split(",")))
batch_sizes = list(map(int, args.bs.split(",")))
patch_widths = list(map(int, args.pw.split(",")))
loss_type = args.loss
num_epochs = args.num_epochs
num_runs = args.num_runs
weight = args.weight
model_names = args.model_pkl.split(",")
comment = f"_{args.comment}" if args.comment else args.comment
save_model_every_n_epochs = args.smene
weight_decay = args.weight_decay
boundary_center = args.boundary_center
hflip = args.hflip
idp = args.idp
odp = args.odp
init = not args.no_init
init_type = args.init_type

parameters = product(
          model_names,
          learning_rates,
          batch_sizes,
          patch_widths,
)

loss_factory = {
    "mae": nn.L1Loss(reduction="none"),
    "smooth": nn.SmoothL1Loss(beta=args.smooth_beta)
}

train_transf = A.HorizontalFlip() if hflip else None

ces = []
loss_fn = loss_factory[loss_type]
loss_weight = torch.tensor([1, weight, weight, weight, weight, 1, 1, 1]).to(device)
loss_weight = loss_weight / loss_weight.sum() * 8

start = datetime.datetime.now()
for param_idx, curr_param in enumerate(parameters):
    (
        model_name,
        learning_rate,
        batch_size,
        patch_width
    ) = curr_param
    train_dl, valid_dl, mean_std = get_loaders_retlstm(
              data_dir, patch_width, batch_size, None,
              normalize_img=True,
              normalize_lyr=True,
              standardize=True,
              boundary_center=boundary_center,
              num_workers=num_workers
    )
    num_batches = len(train_dl)
    ce_run = []
    try:
        with open(train_models / f"{model_name}.pkl", "rb") as model_file:
            model = pickle.load(model_file).to(device)
        for run in range(num_runs):
               # if idp != 0 or odp != 0:
            #     model.init_dropout(idp, odp)
            if init:
                initialize(model, init_type)
            tag = f"pkl/{model_name}"
            lr_str = np.format_float_scientific(learning_rate, 2)
            run_str = f"_{run}" if num_runs != 1 else ""
            config = f"lr{lr_str}_bs{batch_size}_pw{patch_width}{comment}{run_str}"
            checkpoint_folder = checkpoints / f"{tag}/{config}"
            if checkpoint_folder.exists():
                shutil.rmtree(checkpoint_folder, ignore_errors=True)
                shutil.rmtree(runs / f"{tag}/{config}", ignore_errors=True)
            checkpoint_folder.mkdir(exist_ok=True, parents=True)
            writer = SummaryWriter(runs / f"{tag}/{config}")
            optimizer = torch.optim.Adam(
                      model.parameters(),
                      lr=learning_rate,
                      weight_decay=weight_decay
            )
            scaler = GradScaler()
            for epoch in range(num_epochs):
                train_loop = tqdm(
                          train_dl,
                          desc=f"C{param_idx+1}; E[{epoch}/{num_epochs}]", leave=False
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
                    writer.add_scalar("Valid/loss", loss, epoch)
                    writer.add_scalar("Valid/dice", dice.mean(), epoch)
                    writer.add_scalar("Valid/ce", ce.mean(), epoch)
                    writer.add_scalar("Valid/mad", mad.mean(), epoch)
                    if epoch != 0 and epoch % save_model_every_n_epochs == 0:
                        torch.save(
                            model.state_dict(), checkpoint_folder / f"epoch_{epoch}.pth"
                        )
            ces = [ce_run]
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
        with open(ready_models / f"{model_name}.pkl", "wb") as model_file:
            pickle.dump(model.cpu(), model_file)
    except Exception as e:
        print(e)
        print(f"Error with model {model_name}!")

end = datetime.datetime.now()
print(f"run_time: {end-start}")

##
import torch
import pyro
from torch import autograd
from torch.utils.data import Dataset, DataLoader, Subset, TensorDataset

import optuna
import pytorch_lightning as pl
import torch.nn as nn
from optuna.integration import PyTorchLightningPruningCallback
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

import contextlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import numpy as p
import seaborn as sns
import datetime
from typing import Dict
from pprint import pprint
from tqdm import tqdm

##

df0 = pd.read_csv("user_profile.csv")
df0["age"] = datetime.datetime.now().year - df0.yearOfBirth
df0 = df0[["id", "gender", "age", "basalMetabolism"]]
df0.dropna(inplace=True)
users0 = df0.id.unique()

##
df1 = pd.read_csv("heart.csv")
users1 = df1.userId.unique()
users = np.intersect1d(users0, users1)

##
df0 = df0[df0.id.isin(users)]
df1 = df1[df1.userId.isin(users)]

##
def f(x):
    x = x.replace("'", '"')
    import json

    x = json.loads(x)
    s = x["time"]["hour"]
    return s


df1["time"] = df1["startTime"].apply(f)


def g(x):
    x = x.replace("'", '"')
    import json

    x = json.loads(x)
    s = f'{x["date"]["year"]}-{x["date"]["month"]:02d}-{x["date"]["day"]:02d}'
    return s


df1["day"] = df1["startTime"].apply(g)
df1 = df1[["userId", "day", "time", "value"]]

##
uid = users[1]

dfu = df1[df1.userId == uid].copy()
# dfu['dev'] = dfu.groupby(by='time')
mus = dfu.groupby(by="time").value.transform(
    "mean"
)  ##.apply(lambda x: (x - np.mean(x) / np.std(x)))
stds = dfu.groupby(by="time").value.transform(
    np.std
)  ##.apply(lambda x: (x - np.mean(x) / np.std(x)))
dfu["dev"] = (np.abs(dfu.value - mus)) / stds
p = sns.color_palette("ch:start=.2,rot=-.3_r", as_cmap=True)
sns.scatterplot(
    x="time", y="value", hue="dev", data=dfu, palette=p, size=10, linewidth=0, alpha=0.3
)
ax = plt.gca()
ax.get_legend().remove()
plt.xlabel("hour")
plt.ylabel("heart rate")
plt.show()

##
class UserProcessedHeartData:
    def __init__(
        self,
        user_tensor: np.ndarray,
        daily_tensors: Dict[str, np.ndarray],
        target_tensor: np.array,
    ):
        self.user_tensor = user_tensor
        self.daily_tensors = daily_tensors
        self.target_tensor = target_tensor


processed_data = []
dfu
## for each user I need:
for uid in tqdm(users, desc="processing users"):
    dfu = df1[df1.userId == uid].copy()
    # hourly averages
    hourly_averages = dfu.groupby(by="time").value.apply(np.mean)
    # hourly stds
    hourly_stds = dfu.groupby(by="time").value.apply(np.std)
    # hourly count
    hourly_counts = dfu.groupby(by="time").value.apply(len)
    # daily hourly averages
    daily_hourly_averages = dfu.groupby(by=["day", "time"]).value.apply(np.mean)
    # daily hourly std
    daily_hourly_stds = dfu.groupby(by=["day", "time"]).value.apply(np.std)
    # daily hourly count
    daily_hourly_counts = dfu.groupby(by=["day", "time"]).value.apply(len)

    assert len(daily_hourly_averages) == len(daily_hourly_stds)
    assert len(daily_hourly_counts) == len(daily_hourly_stds)
    user_tensor = np.zeros((24, 3), dtype=float)
    for (k0, v0), (k1, v1), (k2, v2) in zip(
        hourly_averages.iteritems(), hourly_stds.iteritems(), hourly_counts.iteritems()
    ):
        assert k0 == k1
        assert k1 == k2
        user_tensor[k0, 0] = v0
        user_tensor[k0, 1] = v1
        user_tensor[k0, 2] = v2

    for i in range(len(daily_hourly_averages)):
        daily_tensors = dict()
        for (k0, v0), (k1, v1), (k2, v2) in zip(
            daily_hourly_averages.iteritems(),
            daily_hourly_stds.iteritems(),
            daily_hourly_counts.iteritems(),
        ):
            assert k0[0] == k1[0]
            assert k1[0] == k2[0]
            assert k0[1] == k1[1]
            assert k1[1] == k2[1]
            if k0[0] not in daily_tensors:
                daily_tensors[k0[0]] = np.zeros((24, 3), dtype=float)
            daily_tensors[k0[0]][k0[1], 0] = v0
            daily_tensors[k0[0]][k0[1], 1] = v1
            daily_tensors[k0[0]][k0[1], 2] = v2

    user_tensor
    daily_tensors
    ages = set(df0[df0.id == uid].age.tolist())
    assert len(ages) == 1
    target_tensor = np.array(ages.__iter__().__next__(), dtype=float)
    d = UserProcessedHeartData(
        user_tensor=user_tensor,
        daily_tensors=daily_tensors,
        target_tensor=target_tensor,
    )
    processed_data.append(d)

##
class AgeHeartDataset(Dataset):
    def __init__(self, processed_data):
        self.data = []
        for u in processed_data:
            for k, v in u.daily_tensors.items():
                t = (u.user_tensor, v, u.target_tensor)
                self.data.append(t)
        import random

        print(
            f"len(self.data) = {len(self.data)}, be careful not to introduce data leakage here when enlarging the dataset"
        )
        random.Random(0).shuffle(self.data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        return self.data[item]


import math

ds = AgeHeartDataset(processed_data)
train_fraction = 0.8
validation_fraction = 0.1
test_fraction = 0.1
a = math.floor(len(ds) * train_fraction)
b = math.floor(len(ds) * (train_fraction + validation_fraction))
train_ds = ds[:a]
val_ds = ds[a:b]
test_ds = ds[b:]
print(f'len(train_ds) = {len(train_ds)}, len(val_ds) = {len(val_ds)}, len(test_ds) = {len(test_ds)}')

##
"""
TRAINING WITH PYTORCH LIGHTNING
"""


class Ppp:
    pass


ppp = Ppp()
ppp.MAX_EPOCHS = 12
ppp.BATCH_SIZE = 8
ppp.DETECT_ANOMALY = False
ppp.DEBUG = True
# ppp.DEBUG = False
if ppp.DEBUG:
    # ppp.NUM_WORKERS = 6
    ppp.NUM_WORKERS = 0
else:
    ppp.NUM_WORKERS = 6
    # ppp.NUM_WORKERS = 0

# sqlite-backed optuna storage does support nan https://github.com/optuna/optuna/issues/2809
def optuna_nan_workaround(loss):
    # from torch 1.9.0
    # loss = torch.nan_to_num(loss, nan=torch.finfo(loss.dtype).max)
    loss[torch.isnan(loss)] = torch.finfo(loss.dtype).max
    return loss


def get_detect_anomaly_cm():
    if ppp.DETECT_ANOMALY:
        cm = autograd.detect_anomaly()
    else:
        cm = contextlib.nullcontext()
    return cm


class NN(pl.LightningModule):
    def __init__(self, optuna_parameters, **kwargs):
        super().__init__()
        self.save_hyperparameters(kwargs)
        self.save_hyperparameters()
        self.optuna_parameters = optuna_parameters

        def _block(in_features, out_features):
            l = [
                nn.Linear(in_features, out_features),
                nn.BatchNorm1d(out_features),
                nn.ReLU(),
                nn.Dropout(p=optuna_parameters["p_dropout"]),
            ]
            return l

        l = _block(24 * 6 + 1, 20) + _block(20, 10) + _block(10, 3)
        l = l[:-3]
        self.encoder = nn.Sequential(*l)
        self.f_mu = nn.Linear(3, 3)
        self.f_log_var = nn.Linear(3, 3)
        l = _block(24 * 6 + 3, 20) + _block(20, 10) + _block(10, 3) + _block(3, 1)
        l = l[:-3] + [nn.Softmax()]
        self.decoder = nn.Sequential(*l)
        self.loss = nn.MSELoss()
        self.log_c = nn.Parameter(torch.Tensor([self.optuna_parameters["log_c"]]))

    def configure_optimizers(self):
        return torch.optim.Adam(
            self.parameters(), lr=self.optuna_parameters["learning_rate"]
        )

    def get_dist(self, a, b):
        dist = pyro.distributions.Normal(a, torch.exp(self.log_c))
        return dist

    def reconstruction_likelihood(self, a, b, x):
        dist = self.get_dist(a, b)
        zero = torch.tensor([2.0]).to(a.device)
        if torch.any(dist.log_prob(zero).isinf()):
            print("infinite value detected")
            self.trainer.should_stop = True
        if torch.any(dist.log_prob(zero).isnan()):
            print("nan value detected")
            self.trainer.should_stop = True
        log_pxz = dist.log_prob(x)
        s = log_pxz.mean(dim=-1)
        return s

    def expected_value(self, a, b=None):
        dist = self.get_dist(a, b)
        return dist.mean

    def kl_divergence(self, z, mu, std):
        # --------------------------
        # Monte carlo KL divergence
        # --------------------------
        # 1. define the first two probabilities (in this case Normal for both)
        p = torch.distributions.Normal(torch.zeros_like(mu), torch.ones_like(std))
        q = torch.distributions.Normal(mu, std)

        # 2. get the probabilities from the equation
        log_qzx = q.log_prob(z)
        log_pz = p.log_prob(z)

        # kl
        kl = log_qzx - log_pz
        kl = kl.sum(-1)
        return kl

    def loss_function(self, x, a, b, mu, std, z):
        cm = get_detect_anomaly_cm()
        with cm:
            recon_loss = self.reconstruction_likelihood(a, b, x)
            # kl
            kl = self.kl_divergence(z, mu, std)
            # elbo
            # elbo = kl - recon_loss
            elbo = self.optuna_parameters["vae_beta"] * kl - recon_loss
            elbo = elbo.mean()
            if torch.isnan(elbo).any():
                print("nan in loss detected!")
                self.trainer.should_stop = True
            if torch.isinf(elbo).any():
                print("inf in loss detected!")
                self.trainer.should_stop = True
            elbo = optuna_nan_workaround(elbo)
            return elbo, kl, recon_loss

    def forward(self, x):
        cm = get_detect_anomaly_cm()
        with cm:
            # x_encoded = self.encoder(x)
            # mu, log_var = self.fc_mu(x_encoded), self.fc_var(x_encoded)
            x = self.encoder(x)
            mu, log_var = self.f_mu(x), self.f_log_var(x)

            # sample z from q
            std = torch.exp(log_var / 2)
            q = torch.distributions.Normal(mu, std)
            z = q.rsample()

            # decoded
            a, b = self.decoder(torch.cat((x[:, :-1], z), dim=1))
            if (
                torch.isnan(a).any()
                or torch.isnan(mu).any()
                or torch.isnan(std).any()
                or torch.isnan(z).any()
            ):
                print("nan in forward detected!")
                self.trainer.should_stop = True

            # print('so far so good')
            return a, b, mu, std, z


    def process_batch(self, batch):
        x = torch.cat((batch[0].ravel(), batch[1].ravel(), batch[2]), dim=0)
        return x

    def training_step(self, batch, batch_idx):
        x = self.process_batch(batch)
        # encode x to get the mu and variance parameters
        a, b, mu, std, z = self.forward(x)
        elbo, kl, recon_loss = self.loss_function(batch[2], a, b, mu, std, z)

        self.log_dict(
            {
                "elbo": elbo,
                "kl": kl.mean(),
                "reconstruction": recon_loss.mean(),
            }
        )

        return elbo

    def validation_step(self, batch, batch_idx, dataloader_idx):
        x = self.process_batch(batch)
        a, b, mu, std, z = self.forward(x)
        elbo, kl, recon_loss = self.loss_function(batch[2], a, b, mu, std, z)

        self.logger.log_hyperparams(params={}, metrics={"hp_metric": elbo})
        d = {
            "elbo": elbo,
            "kl": kl.mean(),
            "reconstruction": recon_loss.mean(),
        }
        return d

    def validation_epoch_end(self, outputs):
        if not self.trainer.sanity_checking:
            assert type(outputs) is list
            batch_val_elbo = None
            for i, o in enumerate(outputs):
                for k in ["elbo", "kl", "reconstruction"]:
                    avg_loss = torch.stack([x[k] for x in o]).mean().cpu().detach()
                    phase = "training" if i == 0 else "validation"
                    self.logger.experiment.add_scalar(
                        f"avg_metric/{k}/{phase}", avg_loss, self.global_step
                    )
                    # self.log(f'epoch_{k} {phase}', avg_loss, on_epoch=False)
                    if phase == "validation" and k == "elbo":
                        batch_val_elbo = avg_loss
            assert batch_val_elbo is not None
            self.log("batch_val_elbo", batch_val_elbo)


def get_loaders(
    shuffle_train=False,
    val_subset=False,
):
    # train_ds = PerturbedCellDataset("train")
    # val_ds = PerturbedCellDataset("validation")

    # train_ds.perturb()
    # val_ds.perturb()
    print(f"ppp.NUM_WORKERS = {ppp.NUM_WORKERS}")

    if ppp.DEBUG:
        n = ppp.BATCH_SIZE * 1
    else:
        n = ppp.BATCH_SIZE * 1
        # incrase when downloading more data
        # n = ppp.BATCH_SIZE * 2
    indices = np.random.choice(len(train_ds), n, replace=False)
    train_subset = Subset(train_ds, indices)

    if ppp.DEBUG:
        d = train_subset
    else:
        d = train_ds
    train_loader = DataLoader(
        train_ds,
        batch_size=ppp.BATCH_SIZE,
        num_workers=ppp.NUM_WORKERS,
        pin_memory=True,
        shuffle=shuffle_train,
    )
    train_loader_batch = DataLoader(
        train_subset,
        batch_size=ppp.BATCH_SIZE,
        num_workers=ppp.NUM_WORKERS,
        pin_memory=True,
    )

    # the val set is a bit too big for training a lot of image models, we are fine with evaluating the generalization
    # on a subset of the data
    if val_subset:
        indices = np.random.choice(len(val_ds), n, replace=False)
        subset = Subset(val_ds, indices)
    else:
        subset = val_ds
    val_loader = DataLoader(
        subset,
        batch_size=ppp.BATCH_SIZE,
        num_workers=ppp.NUM_WORKERS,
        pin_memory=True,
    )
    return train_loader, val_loader, train_loader_batch


def objective(trial: optuna.trial.Trial) -> float:
    # global CHANNEL_TO_PREDICT
    logger = TensorBoardLogger(save_dir="checkpoints", name=f"nn_heart_age")
    print(f"logging in {logger.experiment.log_dir}")
    version = int(logger.experiment.log_dir.split("version_")[-1])
    trial.set_user_attr("version", version)
    checkpoint_callback = ModelCheckpoint(
        dirpath=f"{logger.experiment.log_dir}/checkpoints",
        monitor="batch_val_mse",
        # every_n_train_steps=2,
        save_last=True,
        # save_top_k=3,
    )
    early_stop_callback = EarlyStopping(
        monitor="batch_val_mse",
        min_delta=0.0005,
        patience=3,
        verbose=True,
        mode="min",
        check_finite=True,
    )
    trainer = pl.Trainer(
        gpus=0,
        max_epochs=ppp.MAX_EPOCHS,
        callbacks=[
            checkpoint_callback,
            early_stop_callback,
            PyTorchLightningPruningCallback(trial, monitor="batch_val_mse"),
        ],
        logger=logger,
        num_sanity_val_steps=0,  # track_grad_norm=2,
        log_every_n_steps=10 if not ppp.DEBUG else 1,
        val_check_interval=1 if ppp.DEBUG else 10,
    )

    train_loader, val_loader, train_loader_batch = get_loaders(
        shuffle_train=True, val_subset=True
    )

    # hyperparameters
    learning_rate = trial.suggest_float("learning_rate", 1e-8, 1, log=True)
    p_dropout = trial.suggest_float("p_dropout", 0.0, 0.5)
    log_c = trial.suggest_float("log_c", -3, 3)
    vae_beta = trial.suggest_float("vae_beta", 1e-8, 1e-3, log=True)
    optuna_parameters = dict(
        learning_rate=learning_rate,
        p_dropout=p_dropout,
        log_c=log_c,
        vae_beta=vae_beta
    )
    pprint(optuna_parameters)
    trainer.logger.log_hyperparams(optuna_parameters)

    model = NN(
        optuna_parameters=optuna_parameters,
        **ppp.__dict__,
    )
    trainer.fit(
        model,
        train_dataloaders=train_loader,
        val_dataloaders=[train_loader_batch, val_loader],
    )
    print(f"finished logging in {logger.experiment.log_dir}")

    print(trainer.callback_metrics)
    mse = trainer.callback_metrics["batch_val_mse"].item()
    return mse


# alternative: optuna.pruners.NopPruner()
pruner: optuna.pruners.BasePruner = optuna.pruners.MedianPruner()
study_name = "nn_heart_age"
storage = "sqlite:///optuna.sqlite"
# optuna.delete_study(study_name=study_name, storage=storage)
study = optuna.create_study(
    direction="minimize",
    pruner=pruner,
    storage=storage,
    load_if_exists=True,
    study_name=study_name,
)
TRAIN_SOMETHING = True
# TRAIN_SOMETHING = False
if TRAIN_SOMETHING:
    HYPERPARAMETER_OPTIMIZATION = True
    # HYPERPARAMETER_OPTIMIZATION = False
    if HYPERPARAMETER_OPTIMIZATION:
        HOURS = 60 * 60
        study.optimize(objective, n_trials=500, timeout=1 * HOURS)
        print("Number of finished trials: {}".format(len(study.trials)))
        print("Best trial:")
        trial = study.best_trial
        print("  Value: {}".format(trial.value))
        print("  Params: ")
        for key, value in trial.params.items():
            print("    {}: {}".format(key, value))
    else:
        trial = study.best_trial
        objective(trial)
else:
    trial = study.best_trial
    print(
        f"(best) trial.number = {trial.number}, (best) trial._user_attrs = {trial._user_attrs}"
    )
    import pandas as pd

    pd.set_option("expand_frame_repr", False)
    df = study.trials_dataframe()
    print(df.sort_values(by="value"))
    pd.set_option("expand_frame_repr", True)

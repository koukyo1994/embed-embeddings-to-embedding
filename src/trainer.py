from datetime import datetime as dt

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import torch.nn.functional as F

import numpy as np

from pathlib import Path

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score

from util import timer
from train_helper import seed_torch


class Trainer:
    def __init__(self, logger, n_splits=5, seed=42):
        self.n_splits = n_splits
        self.seed = seed
        self.logger = logger

        self.fold = StratifiedKFold(
            n_splits=n_splits, shuffle=True, random_state=seed)
        self.best_score = None
        self.best_threshold = None
        self.tag = dt.now().strftime("%Y-%m-%d-%H-%M-%S")

    def fit(self, train, answer, n_epochs=50):
        self.train_set = train
        self.y = answer

        self.train_preds = np.zeros((self.train_set.shape[0], 9))
        for i, (trn_index, val_index) in enumerate(
                self.fold.split(self.train_set, self.y)):
            self.fold = i
            self.logger.info(f"\nFold {i+1}")
            X_train = self.train_set[trn_index]
            X_val = self.train_set[val_index]
            y_train = self.y[trn_index]
            y_val = self.y[val_index]

            valid_preds = self._fit(
                X_train, y_train, n_epochs, eval_set=(X_val, y_val))
            self.train_preds[val_index, :] = valid_preds
        score = accuracy_score(self.y, np.argmax(self.train_preds, axis=1))
        f1 = f1_score(
            self.y, np.argmax(self.train_preds, axis=1), average="macro")
        self.logger.info(f"Accuracy: {score:.4f}")
        self.logger.info(f"F1: {f1:.4F}")


class NNTrainer(Trainer):
    def __init__(self,
                 model,
                 logger,
                 n_splits=5,
                 seed=42,
                 device="cpu",
                 train_batch=128,
                 val_batch=512,
                 kwargs={},
                 anneal=True):
        super(NNTrainer, self).__init__(
            logger=logger, n_splits=n_splits, seed=seed)
        self.model = model
        self.device = device
        self.kwargs = kwargs
        self.train_batch = train_batch
        self.val_batch = val_batch
        self.anneal = anneal
        self.loss_fn = nn.CrossEntropyLoss(reduction="mean").to(self.device)

        path = Path(f"bin/{self.tag}")
        path.mkdir(exist_ok=True, parents=True)
        self.path = path
        self.scores = {}
        self.f1s = {}

    def _fit(self, X_train, y_train, n_epochs=50, eval_set=()):
        seed_torch()
        x_train = torch.tensor(X_train, dtype=torch.long).to(self.device)
        y = torch.tensor(
            y_train[:, np.newaxis], dtype=torch.long).to(self.device)

        train = torch.utils.data.TensorDataset(x_train, y)

        train_loader = torch.utils.data.DataLoader(
            train, batch_size=self.train_batch, shuffle=True)
        if len(eval_set) == 2:
            x_val = torch.tensor(eval_set[0], dtype=torch.long).to(self.device)
            y_val = torch.tensor(
                eval_set[1][:, np.newaxis], dtype=torch.long).to(self.device)
            valid = torch.utils.data.TensorDataset(x_val, y_val)
            valid_loader = torch.utils.data.DataLoader(
                valid, batch_size=self.val_batch, shuffle=False)
        model = self.model(**self.kwargs)
        model.to(self.device)
        optimizer = optim.Adam(model.parameters())
        if self.anneal:
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=n_epochs)
        best_score = -np.inf
        epoch_score = []
        epoch_f1 = []

        for epoch in range(n_epochs):
            with timer(f"Epoch {epoch+1}/{n_epochs}", self.logger):
                model.train()
                avg_loss = 0.
                for (x_batch, y_batch) in train_loader:
                    y_pred = model(x_batch)
                    loss = self.loss_fn(y_pred, y_batch.squeeze())
                    optimizer.zero_grad()

                    loss.backward()
                    optimizer.step()
                    avg_loss += loss.item() / len(train_loader)

                valid_preds, avg_val_loss = self._val(valid_loader, model)
                score = accuracy_score(eval_set[1],
                                       np.argmax(valid_preds, axis=1))
                f1 = f1_score(
                    eval_set[1],
                    np.argmax(valid_preds, axis=1),
                    average="macro")
                epoch_score.append(score)
                epoch_f1.append(f1)
            self.logger.info(
                f"loss: {avg_loss:.4f} val_loss: {avg_val_loss:.4f}")
            self.logger.info(f"val_acc: {score} val_f1: {f1}")
            if self.anneal:
                scheduler.step()
            if f1 > best_score:
                torch.save(model.state_dict(),
                           self.path / f"best{self.fold}.pt")
                self.logger.info(f"Save model on epoch {epoch+1}")
                best_score = f1
        model.load_state_dict(torch.load(self.path / f"best{self.fold}.pt"))
        valid_preds, avg_val_loss = self._val(valid_loader, model)
        self.logger.info(f"Validation loss: {avg_val_loss}")
        self.scores[self.fold] = epoch_score
        self.f1s[self.fold] = epoch_f1
        return valid_preds

    def _val(self, loader, model):
        model.eval()
        valid_preds = np.zeros((loader.dataset.tensors[0].size(0), 9))
        avg_val_loss = 0.

        for i, (x_batch, y_batch) in enumerate(loader):
            with torch.no_grad():
                y_pred = model(x_batch).detach()
                avg_val_loss += self.loss_fn(
                    y_pred, y_batch.squeeze()).item() / len(loader)
                valid_preds[i * self.val_batch:(i + 1) * self.
                            val_batch, :] = F.softmax(y_pred).cpu().numpy()
        return valid_preds, avg_val_loss

# Authors: Alexandre Gramfort <alexandre.gramfort@inria.fr>
#          Jean-Rémi KING <jeanremi.king@gmail.com>
#
# License: BSD Style.

import copy

import numpy as np

from tqdm import tqdm

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data.dataset import ConcatDataset as _ConcatDataset  # noqa


class ConcatDataset(_ConcatDataset):
    """
    Same as torch.utils.data.dataset.ConcatDataset, but exposes an extra
    method for querying the group structure (index if dataset
    each sample comes from)
    """
    def get_groups(self):
        """Return the group index of each sample

        Returns
        -------
        groups : array of int, shape (n_samples,)
            The group indices.
        """
        groups = [k * np.ones(len(d)) for k, d in enumerate(self.datasets)]
        return np.concatenate(groups)


class EpochsDataset(Dataset):
    """Class to expose an MNE Epochs object as PyTorch dataset

    Parameters
    ----------
    epochs_data : 3d array, shape (n_epochs, n_channels, n_times)
        The epochs data.
    epochs_labels : array of int, shape (n_epochs,)
        The epochs labels.
    transform : callable | None
        The function to eventually apply to each epoch
        for preprocessing (e.g. scaling). Defaults to None.
    """
    def __init__(self, epochs_data, epochs_labels, transform=None):
        assert len(epochs_data) == len(epochs_labels)
        self.epochs_data = epochs_data
        self.epochs_labels = epochs_labels
        self.transform = transform

    def __len__(self):
        return len(self.epochs_labels)

    def __getitem__(self, idx):
        X, y = self.epochs_data[idx], self.epochs_labels[idx]
        if self.transform is not None:
            X = self.transform(X)
        X = torch.as_tensor(X[None, ...])
        return X, y


def _do_train(model, loader, optimizer, criterion):
    # training loop
    model.train()
    pbar = tqdm(loader)
    train_loss = np.zeros(len(loader))
    for idx_batch, (batch_x, batch_y) in enumerate(pbar):
        optimizer.zero_grad()
        batch_x = batch_x.float().to(model.device)
        batch_y = batch_y.long().to(model.device)

        output = model.forward(batch_x)
        loss = criterion(output, batch_y)

        loss.backward()
        optimizer.step()

        train_loss[idx_batch] = loss
        pbar.set_description(
            desc="avg train loss: {:.4f}".format(
                np.mean(train_loss[:idx_batch + 1])))


def _validate(model, loader, criterion):
    # validation loop
    pbar = tqdm(loader)
    val_loss = np.zeros(len(loader))
    accuracy = 0.
    with torch.no_grad():
        model.eval()

        for idx_batch, (batch_x, batch_y) in enumerate(pbar):
            batch_x = batch_x.float().to(model.device)
            batch_y = batch_y.long().to(model.device)
            output = model.forward(batch_x)

            loss = criterion(output, batch_y)
            val_loss[idx_batch] = loss

            _, top_class = output.topk(1, dim=1)
            accuracy += \
                torch.mean((batch_y == top_class).type(torch.FloatTensor))

            pbar.set_description(
                desc="avg val loss: {:.4f}".format(
                    np.mean(val_loss[:idx_batch + 1])))

    accuracy = accuracy / len(loader)
    print("---  Accuracy : %s" % accuracy.item(), "\n")
    return np.mean(val_loss)


def train(model, loader_train, loader_valid, optimizer, n_epochs, patience):
    """Training function

    Parameters
    ----------
    model : instance of nn.Module
        The model.
    loader_train : instance of Sampler
        The generator of EEG samples the model has to train on.
        It contains n_train samples
    loader_valid : instance of Sampler
        The generator of EEG samples the model has to validate on.
        It contains n_val samples. The validation samples are used to
        monitor the training process and to perform early stopping
    optimizer : instance of optimizer
        The optimizer to use for training.
    n_epochs : int
        The maximum of epochs to run.
    patience : int
        The patience parameter, i.e. how long to wait for the
        validation error to go down.

    Returns
    -------
    best_model : instance of nn.Module
        The model that lead to the best prediction on the validation
        dataset.
    """
    # put model on cuda if not already
    if model.device == "cuda":
        model.to(torch.device(model.device))

    # define criterion
    criterion = F.nll_loss

    best_val_loss = + np.infty
    best_model = copy.deepcopy(model)
    waiting = 0

    for epoch in range(n_epochs):
        print("\nStarting epoch {} / {}".format(epoch + 1, n_epochs))
        _do_train(model, loader_train, optimizer, criterion)
        val_loss = _validate(model, loader_valid, criterion)

        # model saving
        if np.mean(val_loss) < best_val_loss:
            print("\nbest val loss {:.4f} -> {:.4f}".format(
                best_val_loss, np.mean(val_loss)))
            best_val_loss = np.mean(val_loss)
            best_model = copy.deepcopy(model)
            waiting = 0
        else:
            print("Waiting += 1")
            waiting += 1

        # model early stopping
        if waiting >= patience:
            print("Stop training at epoch {}".format(epoch + 1))
            print("Best val loss : {:.4f}".format(best_val_loss))
            break

    return best_model

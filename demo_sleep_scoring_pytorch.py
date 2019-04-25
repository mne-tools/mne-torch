# Authors: Alexandre Gramfort <alexandre.gramfort@inria.fr>
#          Stanislas Chambon <stan.chambon@gmail.com>
#          Jean-Rémi KING <jeanremi.king@gmail.com>
#          Joan Massich <mailsik@gmail.com>
#
# License: BSD Style.

import os
import copy
import numpy as np
import mne
from mne.datasets.sleep_physionet.age import fetch_data
from joblib import Memory

##############################################################################
# Define code to get epochs for all subjects

# subjects = [0, 1]
subjects = range(20)
n_groups = 5  # keep 5 subjects out
files = fetch_data(subjects=subjects, recording=[1])

mapping = {'EOG horizontal': 'eog',
           'Resp oro-nasal': 'misc',
           'EMG submental': 'misc',
           'Temp rectal': 'misc',
           'Event marker': 'misc'}

mem = Memory(mne.get_config('MNE_DATA'))


@mem.cache
def get_epochs_data(raw_fname, annot_fname):
    print("Extracting Epochs from: %s" % os.path.basename(raw_fname))
    raw = mne.io.read_raw_edf(raw_fname)
    annot = mne.read_annotations(annot_fname)
    raw.set_annotations(annot, emit_warning=False)
    raw.set_channel_types(mapping)
    annotation_desc_2_event_id = {'Sleep stage W': 1,
                                  'Sleep stage 1': 2,
                                  'Sleep stage 2': 3,
                                  'Sleep stage 3': 4,
                                  'Sleep stage 4': 4,
                                  'Sleep stage R': 5}

    events, _ = mne.events_from_annotations(
        raw, event_id=annotation_desc_2_event_id, chunk_duration=30.)

    # create a new event_id that unifies stages 3 and 4
    event_id = {'Sleep stage W': 1,
                'Sleep stage 1': 2,
                'Sleep stage 2': 3,
                'Sleep stage 3/4': 4,
                'Sleep stage R': 5}

    tmax = 30. - 1. / raw.info['sfreq']  # tmax in included
    picks = mne.pick_types(raw.info, eeg=True, eog=True)
    epochs = mne.Epochs(raw=raw, events=events, picks=picks, preload=True,
                        event_id=event_id, tmin=0., tmax=tmax, baseline=None)
    return epochs.get_data(), epochs.events[:, 2]


epochs_data, epochs_labels = get_epochs_data(*files[0])

##############################################################################
# Define torch data manipulation objects

from tqdm import tqdm   # noqa
import torch   # noqa
from torch import nn  # noqa
import torch.nn.functional as F  # noqa
import torch.optim as optim  # noqa
from torch.utils.data import Dataset, DataLoader  # noqa
from torch.utils.data import Subset  # noqa
from torch.utils.data import RandomSampler  # noqa
from torch.utils.data import SequentialSampler  # noqa
from torch.utils.data.sampler import BatchSampler  # noqa
from torch.utils.data.dataset import ConcatDataset as _ConcatDataset  # noqa

from sklearn import model_selection  # noqa


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


def train_test_split(dataset, n_groups):
    """Split torch dataset in train and test keeping n_groups out in test

    Parameters
    ----------
    dataset : instance of Dataset
        The dataset to split
    n_groups : int
        The number of groups to leave out.

    Returns
    -------
    ds_train : instance of Dataset
        The training data.
    ds_test : instance of Dataset
        The testing data.
    """
    groups = dataset.get_groups()
    train_idx, test_idx = \
        next(model_selection.LeavePGroupsOut(n_groups).split(X=groups,
                                                             groups=groups))
    return Subset(dataset, train_idx), Subset(dataset, test_idx)


def scale(X):
    """Standard scaling of data along the last dimention.

    Parameters
    ----------
    X : array, shape (n_channels, n_times)
        The input signals.

    Returns
    -------
    X_t : array, shape (n_channels, n_times)
        The scaled signals.
    """
    X -= np.mean(X, axis=1)[:, None]
    return X / np.std(X, axis=1)[:, None]


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
        y -= 1  # make sure labels start at 0
        return X, y

all_datasets = \
    [EpochsDataset(*get_epochs_data(*file), transform=scale) for file in files]
dataset = ConcatDataset(all_datasets)
groups = dataset.get_groups()

ds_train, ds_valid = train_test_split(dataset, n_groups=n_groups)

batch_size_train = 10
batch_size_valid = 64
sampler_train = RandomSampler(ds_train)
sampler_valid = SequentialSampler(ds_valid)

# create loaders
num_workers = 0
loader_train = \
    DataLoader(ds_train, batch_size=batch_size_train,
               num_workers=num_workers, sampler=sampler_train)
loader_valid = \
    DataLoader(ds_valid, batch_size=batch_size_valid,
               num_workers=num_workers, sampler=sampler_valid)


##############################################################################
# Define the model

class SleepScoringModel(nn.Module):
    """The model implements the network for sleep scoring proposed in:

    Chambon, S., Galtier, M., Arnal, P., Wainrib, G. and Gramfort, A.
    (2018)A Deep Learning Architecture for Temporal Sleep Stage
    Classification Using Multivariate and Multimodal Time Series.
    IEEE Trans. on Neural Systems and Rehabilitation Engineering 26:
    (758-769).

    Parameters
    ----------
    spatial_dim : int
        Number of channels
    time_dim : int
        Number of time samples in one chunk of data. It should be
        less than 3840. For example 3000 for 30s at 100Hz.
        The model is optimized for signals sampled at 128Hz leading to
        3840 for 30s of data.
    device : 'cpu' | 'cuda'
        The device to use for training and inference.
    """
    def __init__(self, spatial_dim=1, time_dim=3840, device="cpu"):
        super().__init__()
        self.spatial_dim = spatial_dim
        self.time_dim = time_dim
        self.device = device

        assert time_dim <= 3840
        time_pad_size = (3840 - time_dim)  # get padding if necessary

        # define model architecture
        if self.spatial_dim != 1:
            self.spatial_filtering = nn.Conv2d(
                1, self.spatial_dim, (self.spatial_dim, 1), bias=False)

        self.features_ = nn.Sequential(
            nn.ConstantPad2d([0, time_pad_size, 0, 0], 0),
            nn.Conv2d(1, 8, (1, 64)),
            nn.ConstantPad2d([32, 31, 0, 0], 0),
            nn.ReLU(),
            nn.MaxPool2d((1, 16)),
            nn.Conv2d(8, 8, (1, 64)),
            nn.ConstantPad2d([32, 31, 0, 0], 0),
            nn.ReLU(),
            nn.MaxPool2d((1, 16)),
        )

        self.classifier = nn.Sequential(
            nn.Dropout(0.25),
            nn.Linear(self.spatial_dim * 15 * 8, 5)
        )

    def forward(self, x):
        if self.spatial_dim != 1:
            x = self.spatial_filtering(x)
            x = x.transpose(2, 1)

        x = self.features_(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        x = F.log_softmax(x, dim=1)

        return x

device = 'cuda'
model = SleepScoringModel(spatial_dim=epochs_data.shape[1],
                          time_dim=epochs_data.shape[2],
                          device=device)

# Test model works:
n_samples_test = 10
y_test = torch.randint(0, 5, (n_samples_test,))
y_pred = model.forward(torch.randn(n_samples_test, 1, 3, 3000))
output = F.nll_loss(y_pred, y_test)
_, top_class = y_pred.topk(1, dim=1)


##############################################################################
# Train

lr = 1e-3
optimizer = optim.Adam(model.parameters(), lr=lr)


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
    print("---  Accuracy : %s" % accuracy.item())
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


n_epochs = 10
patience = 5
train(model, loader_train, loader_valid, optimizer, n_epochs, patience)

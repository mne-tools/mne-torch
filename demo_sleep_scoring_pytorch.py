# Authors: Alexandre Gramfort <alexandre.gramfort@inria.fr>
#          Stanislas Chambon <stan.chambon@gmail.com>
#          Jean-Rémi KING <jeanremi.king@gmail.com>
#          Joan Massich <mailsik@gmail.com>
#
# License: BSD Style.

import os
from collections import Counter
import numpy as np
import mne
from mne.datasets.sleep_physionet.age import fetch_data
from joblib import Memory

##############################################################################
# Define code to get epochs for all subjects

subjects = [0, 1]
n_groups = 1  # keep 1 subject out
device = 'cpu'

# subjects = range(20)
# n_groups = 5  # keep 5 subjects out
# device = 'cuda'

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
    return epochs.get_data(), epochs.events[:, 2] - 1


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
# from torch.utils.data import RandomSampler  # noqa
from torch.utils.data import SequentialSampler  # noqa
from torch.utils.data import WeightedRandomSampler  # noqa
from torch.utils.data.sampler import BatchSampler  # noqa

from sklearn import model_selection  # noqa

from common import EpochsDataset, ConcatDataset  # noqa


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


all_datasets = \
    [EpochsDataset(*get_epochs_data(*file), transform=scale) for file in files]
dataset = ConcatDataset(all_datasets)
groups = dataset.get_groups()

ds_train, ds_valid = train_test_split(dataset, n_groups=n_groups)


def get_weights(ds):
    """Do one pass on dataset to get weights"""
    y = np.empty(len(ds), dtype=int)
    for idx in range(len(ds)):
        y[idx] = ds[idx][1]
    weights = np.empty(len(y))
    counts = Counter(y)
    for this_y, this_count in counts.items():
        weights[y == this_y] = 1. / this_count
    return weights


weights_train = get_weights(ds_train)

batch_size_train = 10
batch_size_valid = 64
# sampler_train = RandomSampler(ds_train)

sampler_train = WeightedRandomSampler(weights_train, len(ds_train))
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
    def __init__(self, spatial_dim=1, time_dim=3840):
        super().__init__()
        self.spatial_dim = spatial_dim
        self.time_dim = time_dim

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

model = SleepScoringModel(spatial_dim=epochs_data.shape[1],
                          time_dim=epochs_data.shape[2])

# Test model works:
n_samples_test = 10
y_test = torch.randint(0, 5, (n_samples_test,))
y_pred = model.forward(torch.randn(n_samples_test, 1, 3, 3000))
output = F.nll_loss(y_pred, y_test)
_, top_class = y_pred.topk(1, dim=1)


##############################################################################
# Train

from common import train  # noqa

lr = 1e-3
n_epochs = 10
patience = 5

optimizer = optim.Adam(model.parameters(), lr=lr)

train(model, loader_train, loader_valid, optimizer, n_epochs, patience, device)

# Authors: Alexandre Gramfort <alexandre.gramfort@inria.fr>
#          Jean-Rémi KING <jeanremi.king@gmail.com>
#
# License: BSD Style.

from sklearn.model_selection import ShuffleSplit

from mne import Epochs, pick_types, events_from_annotations
from mne.io import concatenate_raws, read_raw_edf
from mne.datasets import eegbci

# #############################################################################
# # Set parameters and read data

# avoid classification of evoked responses by using epochs that start 1s after
# cue onset.


def get_data():
    tmin, tmax = -1., 4.
    event_id = dict(hands=2, feet=3)
    subject = 1
    runs = [6, 10, 14]  # motor imagery: hands vs feet

    raw_fnames = eegbci.load_data(subject, runs)
    raw = concatenate_raws([read_raw_edf(f, preload=True) for f in raw_fnames])

    # strip channel names of "." characters
    raw.rename_channels(lambda x: x.strip('.'))

    # Apply band-pass filter
    raw.filter(7., 30., fir_design='firwin', skip_by_annotation='edge')

    events, _ = events_from_annotations(raw)

    picks = pick_types(raw.info, meg=False, eeg=True, stim=False, eog=False,
                       exclude='bads')

    # Read epochs (train will be done only between 1 and 2s)
    # Testing will be done with a running classifier
    epochs = Epochs(raw, events, event_id, tmin, tmax, proj=True, picks=picks,
                    baseline=None, preload=True)
    epochs.crop(tmin=1., tmax=None)
    labels = epochs.events[:, 2] - 2
    return epochs.get_data()[:, :, :256], labels


epochs_data, labels = get_data()

###############################################################################
# Classification with PyTorch CSP like model

import torch   # noqa
import torch.optim as optim  # noqa
from torch.utils.data import Dataset, DataLoader  # noqa
from torch.utils.data import Subset  # noqa
from torch import nn  # noqa
import torch.nn.functional as F  # noqa
from torch.utils.data import RandomSampler  # noqa
from torch.utils.data import SequentialSampler  # noqa

from common import EpochsDataset  # noqa

cv = ShuffleSplit(10, test_size=0.2, random_state=42)
cv_split = cv.split(epochs_data)
train_idx, test_idx = next(cv_split)


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
    return X / 2e-5

dataset = EpochsDataset(epochs_data, labels, transform=scale)

ds_train, ds_valid = Subset(dataset, train_idx), Subset(dataset, test_idx)

batch_size_train = len(ds_train)
batch_size_valid = len(ds_valid)
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


class CommonSpatialFilterModel(nn.Module):
    """The model implements a CSP-like network for BCI applications

    Parameters
    ----------
    spatial_dim : int
        Number of channels
    n_components : int
        The number of spatial filters.
    """
    def __init__(self, spatial_dim, n_components=5):
        super().__init__()
        self.spatial_dim = spatial_dim
        self.n_components = n_components

        # define model architecture
        self.spatial_filtering = nn.Conv2d(
            1, self.n_components, (self.spatial_dim, 1), bias=False)

        self.classifier = nn.Sequential(
            nn.Linear(n_components, 2),
        )

    def forward(self, x):
        x = self.spatial_filtering(x)
        x = torch.sum(x ** 2, dim=3)
        x = torch.log(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        x = F.log_softmax(x, dim=1)
        return x

# device = 'cuda'
device = 'cpu'
n_components = 30
model = CommonSpatialFilterModel(spatial_dim=epochs_data.shape[1],
                                 n_components=n_components)

# Test model works:
n_samples_test = 10
y_test = torch.randint(0, 2, (n_samples_test,))
y_pred = model.forward(torch.randn(n_samples_test, 1, *epochs_data.shape[1:]))
output = F.nll_loss(y_pred, y_test)
_, top_class = y_pred.topk(1, dim=1)


##############################################################################
# Train

from common import train  # noqa

lr = 1e-3
n_epochs = 300
patience = 100

model.to(device=device)  # move to device before creating the optimizer
optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9,
                      weight_decay=1e-4)

train(model, loader_train, loader_valid, optimizer, n_epochs, patience, device)

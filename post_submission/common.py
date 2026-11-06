"""Minimal copies of the experiment code in ``src/`` for the post-submission runs.

``src/`` is frozen and cannot be imported: the files are Colab notebook exports
that load data, train and run the whole experiment at import time. The
functions below are copied from ``src/pruning_experiments.py`` with the same
logic. The only changes are that the device is passed in explicitly and the
progress prints are quieter. Anything that differs from the original in a way
that could affect a number is flagged with ``# NOTE``.
"""

from __future__ import annotations

import copy
import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.prune as prune
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

REPO = Path(__file__).resolve().parent.parent
RESULTS_DIR = REPO / "results" / "post_submission"
FIGURES_DIR = REPO / "presentation_figures"
PAPER_CSV = {
    "structured": REPO / "results" / "experiment_logs" / "structured_pruning_data_3.csv",
    "unstructured": REPO / "results" / "experiment_logs" / "unstructured_pruning_data_3.csv",
}

# Same locations as the constants in src/; inside the container the host's
# data/ folders are mounted there. Outside the container, fall back to ./data.
_COLAB_ROOT = Path("/content/drive/MyDrive/Bachelor/new coding")


def default_dir(colab_name: str, local_name: str) -> Path:
    colab = _COLAB_ROOT / colab_name
    return colab if colab.is_dir() else REPO / "data" / local_name


DEFAULT_DATA_DIR = default_dir("NEW_DATA", "NEW_DATA")
DEFAULT_MODEL_DIR = default_dir("NEW_MODELS", "NEW_MODELS")

# The two models of the paper: big = FD001 (N_tw = 30), small = FD004 (N_tw = 15).
MODELS = {
    "big": {"sequence_length": 30, "checkpoint": "model0.pth", "loader_index": 0},
    "small": {"sequence_length": 15, "checkpoint": "model3.pth", "loader_index": 3},
}
FEATURES = 14
RATIOS = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90]

# Paper hyperparameters (src/pruning_experiments.py: lr, eps, batch_size).
FINE_TUNE_LR = 0.001
FINE_TUNE_EPOCHS = 20
TRAIN_BATCH = 512


def input_shape(size: str, batch: int) -> tuple[int, int, int, int]:
    return (batch, 1, MODELS[size]["sequence_length"], FEATURES)


# --------------------------------------------------------------------------
# Model and data (verbatim from src/)
# --------------------------------------------------------------------------

class CNN(nn.Module):
    def __init__(self, input_channels=1, sequence_length=30, feature_length=14):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, 10, kernel_size=(10, 1))
        self.conv2 = nn.Conv2d(10, 10, kernel_size=(10, 1))
        self.conv3 = nn.Conv2d(10, 10, kernel_size=(10, 1))
        self.conv4 = nn.Conv2d(10, 10, kernel_size=(10, 1))
        self.conv5 = nn.Conv2d(10, 1, kernel_size=(3, 1))
        self.dropout = nn.Dropout(p=0.5)
        self.fc1 = nn.Linear((sequence_length - 2) * feature_length, 100)
        self.fc2 = nn.Linear(100, 1)

    def forward(self, x):
        x = F.pad(x, (0, 0, 4, 5))
        x = torch.tanh(self.conv1(x))
        x = F.pad(x, (0, 0, 4, 5))
        x = torch.tanh(self.conv2(x))
        x = F.pad(x, (0, 0, 4, 5))
        x = torch.tanh(self.conv3(x))
        x = F.pad(x, (0, 0, 4, 5))
        x = torch.tanh(self.conv4(x))
        x = torch.tanh(self.conv5(x))
        x = torch.flatten(x, start_dim=1)
        x = torch.tanh(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class TimeWindowDataset(Dataset):
    def __init__(self, data, is_train=True):
        self.data = torch.tensor(data, dtype=torch.float32)
        self.is_train = is_train

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        features = sample[:, :-1].unsqueeze(0)
        labels = sample[-1, -1].unsqueeze(0)
        return features, labels


def load_loaders(data_dir: Path, batch_size: int = TRAIN_BATCH):
    """Return {size: (train_loader, test_loader)} for the big and small model."""
    train_timed = torch.load(Path(data_dir) / "train_timed.pt", weights_only=False)
    test_timed = torch.load(Path(data_dir) / "test_timed.pt", weights_only=False)
    loaders = {}
    for size, spec in MODELS.items():
        i = spec["loader_index"]
        loaders[size] = (
            DataLoader(TimeWindowDataset(train_timed[i], is_train=True), batch_size=batch_size, shuffle=True),
            DataLoader(TimeWindowDataset(test_timed[i], is_train=False), batch_size=batch_size, shuffle=False),
        )
    return loaders


def load_baseline(size: str, model_dir: Path) -> CNN:
    """Load the trained baseline checkpoint (model0.pth / model3.pth) on the CPU."""
    spec = MODELS[size]
    model = CNN(sequence_length=spec["sequence_length"])
    state = torch.load(Path(model_dir) / spec["checkpoint"], map_location="cpu")
    model.load_state_dict(state)
    model.eval()
    return model


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


# --------------------------------------------------------------------------
# Metrics (verbatim from src/)
# --------------------------------------------------------------------------

def rmse_loss(predictions, targets):
    criterion = nn.MSELoss()
    return torch.sqrt(criterion(predictions, targets))


def scoring_function(rul_estimated, rul_actual):
    di = rul_estimated.clone().detach() - rul_actual.clone().detach()
    si = torch.where(di < 0, torch.exp(-di / 13) - 1, torch.exp(di / 10) - 1)
    return torch.sum(si).item()


def calc_model_size(model):
    """The paper's ``size_mb``: parameters + buffers in memory, in MiB."""
    param_size = sum(p.nelement() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.nelement() * b.element_size() for b in model.buffers())
    return (param_size + buffer_size) / 1024**2


def validate_model(model, test_dl, device):
    """RMSE and S-Score on the test set, averaged over test batches as in src/."""
    model.to(device)
    model.eval()
    val_loss = 0.0
    score_value = 0.0
    total_batches = len(test_dl)
    with torch.no_grad():
        for inputs, targets in test_dl:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            outputs = (outputs + 1) / 2 * 125
            targets = (targets + 1) / 2 * 125
            val_loss += rmse_loss(outputs, targets).item()
            score_value += scoring_function(outputs, targets)
    return val_loss / total_batches, score_value / total_batches


def fine_tune_model(model, train_dl, test_dl, device, learning_rate=FINE_TUNE_LR,
                    num_epochs=FINE_TUNE_EPOCHS, log_prefix=""):
    """20 epochs of Adam at lr 1e-3, as in src/. Early stopping was disabled there
    (patience=100 > 20 epochs), so it is left out here; the last epoch's weights
    are kept, not the best ones, as in src/.

    The per-epoch validation is kept even though only every 5th result is
    printed: creating a DataLoader iterator draws a base seed from the global
    torch RNG (even with shuffle=False), so dropping it would shift the RNG
    stream relative to src/.
    """
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    for epoch in range(num_epochs):
        model.train()
        for inputs, targets in train_dl:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            loss = rmse_loss(model(inputs), targets)
            loss.backward()
            optimizer.step()
        val_loss, score = validate_model(model, test_dl, device)
        if (epoch + 1) % 5 == 0:
            print(f"{log_prefix}epoch {epoch + 1:2d}/{num_epochs}  RMSE {val_loss:7.3f}  S {score:9.2f}")
    return model


# --------------------------------------------------------------------------
# Pruning (verbatim from src/)
# --------------------------------------------------------------------------

def get_channels_to_prune(conv_layer, ratio):
    with torch.no_grad():
        importance = torch.norm(conv_layer.weight.data, p=1, dim=(1, 2, 3))
        indices = torch.argsort(importance)
    num_channels_to_prune = int(ratio * len(indices))
    return indices[:num_channels_to_prune].tolist()


def group_prune(model, conv_layer, channels_to_prune, example_input):
    import torch_pruning as tp

    model = model.eval()
    DG = tp.DependencyGraph().build_dependency(model, example_inputs=example_input)
    group = DG.get_pruning_group(conv_layer, tp.prune_conv_out_channels, idxs=channels_to_prune)
    if DG.check_pruning_group(group):
        group.prune()


def prune_structured(model: CNN, ratio: float, size: str, device) -> CNN:
    """L1 channel pruning of every conv layer's output channels (in place).

    As in src/, the dependency graph is traced with a batch-512 example input;
    the batch size has no influence on which channels are removed.
    """
    if ratio <= 0:
        return model
    example_input = torch.randn(*input_shape(size, 512)).to(device)
    model.to(device)
    for _, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            channels = get_channels_to_prune(module, ratio)
            group_prune(model, module, channels, example_input)
    return model


def prune_unstructured(model: CNN, ratio: float) -> CNN:
    """L1 unstructured pruning of the conv weights (in place), as in src/:
    a mask is attached and the weight is then re-assigned as a sparse tensor.
    The pruning forward-pre-hook recomputes a dense ``weight`` from
    ``weight_orig * weight_mask`` on every forward pass."""
    if ratio <= 0:
        return model
    for _, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            prune.l1_unstructured(module, name="weight", amount=ratio)
    for _, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            module.weight = module.weight.to_sparse()
    return model


def pruned_copy(baseline: CNN, method: str, ratio_pct: int, size: str, device) -> CNN:
    model = copy.deepcopy(baseline)
    ratio = ratio_pct / 100
    if method == "structured":
        prune_structured(model, ratio, size, device)
    elif method == "unstructured":
        model.to(device)
        prune_unstructured(model, ratio)
    else:
        raise ValueError(method)
    return model.eval()


def device_from_arg(name: str | None) -> torch.device:
    if name:
        return torch.device(name)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def env_info() -> dict:
    import platform

    info = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cpu": platform.processor() or platform.machine(),
        "hostname": os.uname().nodename if hasattr(os, "uname") else "",
    }
    if torch.cuda.is_available():
        info["gpu"] = torch.cuda.get_device_name(0)
        info["cuda"] = torch.version.cuda
        info["cudnn"] = torch.backends.cudnn.version()
    return info

!pip install torch-pruning
import torch_pruning as tp

!pip install calflops
from calflops import calculate_flops

!pip install --upgrade gspread gspread_dataframe google-auth
import gspread
from gspread_dataframe import get_as_dataframe, set_with_dataframe
from google.colab import auth
auth.authenticate_user()

import google.auth
from google.auth.transport.requests import Request
from google.auth import default
creds, _ = default()
gc = gspread.authorize(creds)

import time
import os
import warnings
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch.profiler import profile, record_function, ProfilerActivity
import random

import torch.nn.functional as F
import torch.nn.utils.prune as prune
import copy

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DRIVE_DIR = '/content/drive/MyDrive/Bachelor/new coding/CMAPSSData/'
DATA_DIR = '/content/drive/MyDrive/Bachelor/new coding/NEW_DATA/'
MODEL_DIR = '/content/drive/MyDrive/Bachelor/new coding/NEW_MODELS/'

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

def create_dataloaders(train_data, test_data, batch_size=512):
  train_loaders = [DataLoader(TimeWindowDataset(data, is_train=True), batch_size=batch_size, shuffle=True) for data in train_data]
  test_loaders = [DataLoader(TimeWindowDataset(data, is_train=False), batch_size=batch_size, shuffle=False) for data in test_data]
  return train_loaders, test_loaders

def setup_data():
  train_path = os.path.join(DATA_DIR, "train_timed.pt")
  test_path = os.path.join(DATA_DIR, "test_timed.pt")

  if os.path.exists(train_path) and os.path.exists(test_path):
    train_timed = torch.load(train_path, weights_only=False)
    test_timed = torch.load(test_path, weights_only=False)

  train_loaders, test_loaders = create_dataloaders(train_timed, test_timed)
  return train_loaders, test_loaders

train_loaders, test_loaders = setup_data()
big_tw_train_loader = train_loaders[0]
big_tw_test_loader = test_loaders[0]
small_tw_train_loader = train_loaders[3]
small_tw_test_loader = test_loaders[3]

def setup_models():
  big_tw_model = CNN(sequence_length=30)
  big_tw_model.load_state_dict(torch.load(os.path.join(MODEL_DIR, "model0.pth")))
  big_tw_model.eval()
  small_tw_model = CNN(sequence_length=15)
  small_tw_model.load_state_dict(torch.load(os.path.join(MODEL_DIR, "model3.pth")))
  small_tw_model.eval()

  return big_tw_model, small_tw_model

big_tw_model, small_tw_model = setup_models()

# Test shapes
index, sample = next(enumerate(big_tw_test_loader))
features, labels = sample
print(features.shape)
print(labels.shape)
print(labels[7])

index, sample = next(enumerate(small_tw_test_loader))
features, labels = sample
print(features.shape)
print(labels.shape)
print(labels[7])

features = features[0][0]
labels = labels[0]

plt.figure(figsize=(10, 6))
plt.title('Sensor Data for Sample')
plt.xlabel('Time')
plt.ylabel('Feature Value')
plt.grid(True)

num_features = features.shape[1]
for i in range(num_features):
  plt.plot(range(features.shape[0]), features[:, i], label=f'Feature {i+1}')

plt.ylim(-1, 1)  # adjust y-axis 
plt.legend()

def rmse_loss(predictions, targets):
  criterion = nn.MSELoss()
  return torch.sqrt(criterion(predictions, targets))

def scoring_function(rul_estimated, rul_actual):
  if not isinstance(rul_estimated, torch.Tensor):
    rul_estimated = torch.tensor(rul_estimated)
  if not isinstance(rul_actual, torch.Tensor):
    rul_actual = torch.tensor(rul_actual)

  rul_estimated = rul_estimated.clone().detach()
  rul_actual = rul_actual.clone().detach()
  di = rul_estimated - rul_actual

  si = torch.where(
    di < 0,
    torch.exp(-di / 13) - 1,
    torch.exp(di / 10) - 1
  )

  score = torch.sum(si)
  return score.item()

def calc_model_size(model):
  param_size = 0
  for param in model.parameters():
    param_size += param.nelement() * param.element_size()
  buffer_size = 0
  for buffer in model.buffers():
    buffer_size += buffer.nelement() * buffer.element_size()

  size_all_mb = (param_size + buffer_size) / 1024**2
  return size_all_mb

def validate_model(model, test_dl):
  model.to(DEVICE)
  model.eval()
  val_loss = 0.0
  score_value = 0.0
  total_batches = len(test_dl)

  with torch.no_grad():
    for val_batch_inputs, val_batch_targets in test_dl:
      val_batch_inputs, val_batch_targets = val_batch_inputs.to(DEVICE), val_batch_targets.to(DEVICE)
      val_outputs = model(val_batch_inputs)

      # de-norm
      val_outputs = (val_outputs + 1) / 2 * 125
      val_batch_targets = (val_batch_targets + 1) / 2 * 125

      batch_loss = rmse_loss(val_outputs, val_batch_targets).item()
      val_loss += batch_loss
      batch_score = scoring_function(val_outputs, val_batch_targets)
      score_value += batch_score

  val_loss /= total_batches
  score_value /= total_batches
  return val_loss, score_value

def fine_tune_model(model, train_dl, test_dl, learning_rate=0.001, num_epochs=20, patience=100, min_delta=0.25): # early stopping deactivated
  model.to(DEVICE)
  optimizer = optim.Adam(model.parameters(), lr=learning_rate)
  best_loss = float('inf')
  epochs_no_improve = 0

  print(f"Training model for up to {num_epochs} epochs...")
  for epoch in range(num_epochs):
    model.train()
    train_loss = 0.0
    for batch_inputs, batch_targets in train_dl:
      batch_inputs, batch_targets = batch_inputs.to(DEVICE), batch_targets.to(DEVICE)
      optimizer.zero_grad()
      outputs = model(batch_inputs)
      loss = rmse_loss(outputs, batch_targets)
      train_loss += loss.item() * batch_inputs.size(0)
      loss.backward()
      optimizer.step()

      # de-norma
      outputs = (outputs + 1) / 2 * 125
      batch_targets = (batch_targets + 1) / 2 * 125

      loss = rmse_loss(outputs, batch_targets)
      train_loss += loss.item() * batch_inputs.size(0)

    train_loss /= len(train_dl.dataset)

    val_loss, score = validate_model(model, test_dl)

    if best_loss - val_loss > min_delta:
      best_loss = val_loss
      epochs_no_improve = 0
    else:
      epochs_no_improve += 1

    if (epoch + 1) % 5 == 0:
      print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')

    if epochs_no_improve >= patience:
      print(f"Early stopping at epoch {epoch+1}. Best Val Loss: {best_loss:.4f}")
      break

  return model

sizes = ['big', 'small']
ratios = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90]
types = ['pruned', 'tuned']
metrics = ['rmse', 'score', 'size_mb', 'memory_mb', 'inference_ms', 'params', 'MACs', 'FLOPs']
num_runs = 10

metric_names = []

for size in sizes:
  for ratio in ratios:
    for typ in types:
      if ratio == 0 and typ == 'tuned':
        continue
      for metric in metrics:
        metric_name = f"{size}_{metric}_{typ}_{ratio}"
        metric_names.append(metric_name)

columns = ['metric'] + [f'run_{i+1}' for i in range(num_runs)]
data = [[name] + [None]*num_runs for name in metric_names]

unstructured_df = pd.DataFrame(data, columns=columns)
structured_df = pd.DataFrame(data, columns=columns)

print(unstructured_df.head())
print(structured_df.head())

def update_metric(df, size, metric, typ, ratio, run_number, value):
  metric_name = f"{size}_{metric}_{typ}_{int(ratio*100)}"
  column_name = f"run_{run_number}"

  row_index = df.index[df['metric'] == metric_name]
  if not row_index.empty:
    df.at[row_index[0], column_name] = value
  else:
    print(f"Warning: Metric name '{metric_name}' not found in DataFrame.")

def measure_inference_time(model, input, runs=100, warmup=1000):
  with torch.no_grad():
    for _ in range(warmup):
      _ = model(input)

  times = []
  with torch.no_grad():
    for _ in range(runs):
      torch.cuda.synchronize()
      start_time = time.perf_counter()
      _ = model(input)
      torch.cuda.synchronize()
      end_time = time.perf_counter()
      times.append(end_time - start_time)
  return np.median(times), np.std(times)

def model_mem_time(model, input):
  model.eval()
  model.to(DEVICE)
  input.to(DEVICE)

  median_time, std_time = measure_inference_time(model, input)

  torch.cuda.synchronize()
  with profile(activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA], profile_memory=True) as p:
    with torch.no_grad():
      model(input)
  torch.cuda.synchronize()

  result = p.key_averages().total_average().device_memory_usage / (1024 * 1024), float(median_time * 1000)

  return result

big_input = torch.randn(512, 1, 30, 14).to(DEVICE)
big_copy = copy.deepcopy(big_tw_model).to(DEVICE)

model_mem_time(big_copy, big_input)

def parse_metric(value_str):
  multipliers = {
    "K": 1e3,
    "M": 1e6,
    "G": 1e9,
  }

  parts = value_str.strip().split()
  if len(parts) != 2:
    raise ValueError(f"Invalid format: {value_str}")

  number = float(parts[0])
  unit_part = parts[1].upper()

  for prefix, multiplier in multipliers.items():
    if unit_part.startswith(prefix):
      return int(number * multiplier)

  return int(number)

parse_metric("1.36234 GFLOPS")

lr = 0.001
eps = 20

unstructured_sheet = gc.open_by_url("-").sheet1
structured_sheet = gc.open_by_url("-").sheet1

def profile_models(df, sheet, big_tw_model, small_tw_model, ratio, run, pruned=False, tuned=False):
  big_tw_model.to(DEVICE)
  small_tw_model.to(DEVICE)

  big_rmse, big_score = validate_model(big_tw_model, big_tw_test_loader)
  small_rmse, small_score = validate_model(small_tw_model, small_tw_test_loader)

  big_input = torch.randn(512, 1, 30, 14).to(DEVICE)
  small_input = torch.randn(512, 1, 15, 14).to(DEVICE)

  big_mem, big_time = model_mem_time(big_tw_model, big_input)
  small_mem, small_time = model_mem_time(small_tw_model, small_input)

  big_input_shape = (512, 1, 30, 14)
  small_input_shape = (512, 1, 15, 14)

  big_flops, big_macs, big_params = calculate_flops(model=big_tw_model,
                                        input_shape=big_input_shape,
                                        output_precision=4)

  small_flops, small_macs, small_params = calculate_flops(model=small_tw_model,
                                        input_shape=small_input_shape,
                                        output_precision=4)

  big_size = calc_model_size(big_tw_model)
  small_size = calc_model_size(small_tw_model)

  metric_values_by_size = {
    'big': {
      'rmse': big_rmse,
      'score': big_score,
      'size_mb': big_size,
      'memory_mb': big_mem,
      'inference_ms': big_time,
      'params': parse_metric(big_params),
      'MACs': parse_metric(big_macs),
      'FLOPs': parse_metric(big_flops),
    },
    'small': {
      'rmse': small_rmse,
      'score': small_score,
      'size_mb': small_size,
      'memory_mb': small_mem,
      'inference_ms': small_time,
      'params': parse_metric(small_params),
      'MACs': parse_metric(small_macs),
      'FLOPs': parse_metric(small_flops),
    }
  }

  for model_type in types:
    if (model_type == 'pruned' and pruned) or (model_type == 'tuned' and tuned):
      for size in sizes:
        for metric in metrics:
          value = metric_values_by_size[size][metric]
          update_metric(df, size, metric, model_type, ratio, run, value)

run_digits = [(i+1) for i in range(10)]
column_letters = ['B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K']
run_column_mapping = dict(zip(run_digits, column_letters))

def update_sheet(sheet, df, run):
  run_col = run_column_mapping[run]
  values = df[f'run_{run}'].tolist()
  cell_list = sheet.range(f'{run_col}2:{run_col}{len(values)+1}')

  for i, cell in enumerate(cell_list):
    if not cell.value:
      cell.value = values[i]

  sheet.update_cells(cell_list)

def set_seed(seed):
  torch.manual_seed(seed)
  np.random.seed(seed)
  random.seed(seed)

def unstructured_pruning(df, run):
  ratios = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
  set_seed(42 + run)

  for ratio in ratios:
    print(f"PRUNING FOR {ratio * 100}%")

    big_copy = copy.deepcopy(big_tw_model).to(DEVICE)
    small_copy = copy.deepcopy(small_tw_model).to(DEVICE)

    if ratio > 0:
      for name, module in big_copy.named_modules():
        if isinstance(module, torch.nn.Conv2d):
          prune.l1_unstructured(module, name='weight', amount=ratio)

      for name, module in small_copy.named_modules():
        if isinstance(module, torch.nn.Conv2d):
          prune.l1_unstructured(module, name='weight', amount=ratio)

      for name, module in big_copy.named_modules():
        if isinstance(module, torch.nn.Conv2d):
          module.weight = module.weight.to_sparse()

      for name, module in small_copy.named_modules():
        if isinstance(module, torch.nn.Conv2d):
          module.weight = module.weight.to_sparse()

    profile_models(df, unstructured_sheet, big_copy, small_copy, ratio, run, pruned=True, tuned=False)

    if ratio == 0:
      continue

    print(f"INITIALIZING FINE-TUNING FOR THE {ratio * 100}% RATIO")
    fine_tune_model(big_copy, big_tw_train_loader, big_tw_test_loader, learning_rate=lr, num_epochs=eps)
    fine_tune_model(small_copy, small_tw_train_loader, small_tw_test_loader, learning_rate=lr, num_epochs=eps)

    profile_models(df, unstructured_sheet, big_copy, small_copy, ratio, run, pruned=False, tuned=True)

  update_sheet(unstructured_sheet, df, run)

for i in range(10):
  run = i + 1
  unstructured_pruning(unstructured_df, run)

def analyze_conv_channels(conv_layer):
  with torch.no_grad():
    importance = torch.norm(conv_layer.weight.data, p=1, dim=(1, 2, 3))
    sorted_indices = torch.argsort(importance)
    return sorted_indices

def get_channels_to_prune(conv_layer, ratio):
  indices = analyze_conv_channels(conv_layer)
  num_channels_to_prune = int(ratio * len(indices))
  channels_to_prune = indices[:num_channels_to_prune].tolist()
  return channels_to_prune

def group_prune(model, conv_layer, channels_to_prune, example_input):
  model = model.eval()
  DG = tp.DependencyGraph().build_dependency(model, example_inputs=example_input)
  group = DG.get_pruning_group(conv_layer, tp.prune_conv_out_channels, idxs=channels_to_prune)
  if DG.check_pruning_group(group):
    group.prune()

def structured_pruning(df, sheet, run):
  ratios = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
  set_seed(42 + run)

  big_input = torch.randn(512, 1, 30, 14).to(DEVICE)
  small_input = torch.randn(512, 1, 15, 14).to(DEVICE)

  for ratio in ratios:
    print(f"PRUNING FOR {ratio * 100}%")

    big_copy = copy.deepcopy(big_tw_model).to(DEVICE)
    small_copy = copy.deepcopy(small_tw_model).to(DEVICE)

    if ratio > 0:
      for name, module in big_copy.named_modules():
        if isinstance(module, torch.nn.Conv2d):
          channels_to_prune = get_channels_to_prune(module, ratio)
          group_prune(big_copy, module, channels_to_prune, big_input)
      for name, module in small_copy.named_modules():
        if isinstance(module, torch.nn.Conv2d):
          channels_to_prune = get_channels_to_prune(module, ratio)
          group_prune(small_copy, module, channels_to_prune, small_input)

    profile_models(df, sheet, big_copy, small_copy, ratio, run, pruned=True, tuned=False)

    if ratio == 0:
      continue

    print(f"INITIALIZING FINE-TUNING FOR THE {ratio * 100}% RATIO")
    fine_tune_model(big_copy, big_tw_train_loader, big_tw_test_loader, learning_rate=lr, num_epochs=eps)
    fine_tune_model(small_copy, small_tw_train_loader, small_tw_test_loader, learning_rate=lr, num_epochs=eps)

    profile_models(df, sheet, big_copy, small_copy, ratio, run, pruned=False, tuned=True)

  update_sheet(sheet, df, run)

for i in range(10):
  run = i + 1
  structured_pruning(structured_df, structured_sheet, run)
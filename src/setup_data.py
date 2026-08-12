import os
import pandas as pd
import matplotlib.pyplot as plt
from google.colab import files
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import torch
from torch.utils.data import DataLoader, Dataset

DRIVE_DIR = '/content/drive/MyDrive/Bachelor/new coding/CMAPSSData/'
DATA_DIR = '/content/drive/MyDrive/Bachelor/new coding/NEW_DATA/'

def load_data_files():
  file_paths = {
    'train': [f"{DRIVE_DIR}train_FD00{i}.txt" for i in range(1, 5)],
    'test': [f"{DRIVE_DIR}test_FD00{i}.txt" for i in range(1, 5)],
    'rul': [f"{DRIVE_DIR}RUL_FD00{i}.txt" for i in range(1, 5)]
  }
  data = {key: [pd.read_csv(fp, sep=r'\s+', header=None) for fp in file_paths[key]] for key in file_paths}
  return data['train'], data['test'], data['rul']

def drop_columns(dataframes):
  drop_columns = [2, 3, 4, 5, 9, 10, 14, 20, 22, 23]

  for df in dataframes:
    df.drop(df.columns[drop_columns], axis=1, inplace=True)
  return dataframes

def scale_train_data(dataframes):
  dataframes = drop_columns(dataframes)
  scalers = []
  for df in dataframes:
    df.iloc[:, 2:] = df.iloc[:, 2:].astype(float)
    scaler = MinMaxScaler(feature_range=(-1, 1))
    df.iloc[:, 2:] = scaler.fit_transform(df.iloc[:, 2:])
    scalers.append(scaler)
  return dataframes, scalers

def scale_test_data(dataframes, scalers):
  dataframes = drop_columns(dataframes)
  for df, scaler in zip(dataframes, scalers):
    df.iloc[:, 2:] = df.iloc[:, 2:].astype(float)
    df.iloc[:, 2:] = scaler.transform(df.iloc[:, 2:])
  return dataframes

def degrade(df, rul_df=None):
  df['degradation'] = -1
  df['degradation'] = df['degradation'].astype(float)
  unit = -1
  r_now = 0
  for index, row in df[::-1].iterrows():
    engine = row.iloc[0]
    if engine != unit:
      r_now = 0
      if rul_df is not None:
        r_now = rul_df.iloc[int(engine)-1].values[0]
        r_now = min(r_now, 125) # rectified
      unit = engine
    normalized = 2 * (r_now / 125) - 1
    df.at[index, 'degradation'] = normalized
    if r_now < 125:
      r_now += 1

def time_window_transform(df, tw):
  unit = -1
  sensor_data_list = []
  for index, row in df.iterrows():
    engine = row.iloc[0]
    if engine != unit:
      unit = engine
    if index + tw >= len(df):
      continue
    if engine != df.iloc[index + tw].iloc[0]:
      continue
    sensor_data = df.iloc[index : index + tw, 2:].values
    if sensor_data.shape == (tw, 15):
      sensor_data_list.append(sensor_data)
  # print(f"Time window data length: {len(sensor_data_list)}")
  return np.array(sensor_data_list)

def test_time_window_transform(df, tw):
  unit = -1
  sensor_data_list = []
  processed_units = set()

  for index, row in df[::-1].iterrows():
    engine = row.iloc[0]
    if engine in processed_units:
      continue
    if engine != unit:
      unit = engine
    if index + tw >= len(df):
      continue
    if engine != df.iloc[index + tw].iloc[0]:
      continue
    sensor_data = df.iloc[index : index + tw, 2:].values
    if sensor_data.shape == (tw, 15):
      sensor_data_list.append(sensor_data)
      processed_units.add(engine)
  # print(f"Time window data length: {len(sensor_data_list)}")
  return np.array(sensor_data_list)

def preprocess():
  train_path = os.path.join(DATA_DIR, "train_timed.pt")
  test_path = os.path.join(DATA_DIR, "test_timed.pt")

  train_data, test_data, rul_data = load_data_files()

  # print(train_data[0].head())
  # print(test_data[0].head())
  # print(rul_data[0].iloc[0].values[0])

  train_data, scalers = scale_train_data(train_data)
  test_data = scale_test_data(test_data, scalers)

  # for train_df, test_df in zip(train_data, test_data):
  #   print(train_df.head())
  #   print(test_df.head())

  for train_df, test_df, rul_df in zip(train_data, test_data, rul_data):
    degrade(train_df)
    degrade(test_df, rul_df)

  train_timed = [time_window_transform(df, tw) for df, tw in zip(train_data, [30, 20, 30, 15])]
  test_timed = [test_time_window_transform(df, tw) for df, tw in zip(test_data, [30, 20, 30, 15])]

  # for train_df, test_df in zip(train_timed, test_timed):
  #   print(train_df.shape)
  #   print(test_df.shape)

  # print(train_timed[0][77][29][-1])
  # print(test_timed[0][77][29][-1])

  torch.save(train_timed, train_path)
  torch.save(test_timed, test_path)

  return train_data, test_data

def plot_rul(df, unit_number, filename='plot'):
  filename = filename + '.png'
  unit_df = df[df.iloc[:, 0] == unit_number]

  if unit_df.empty:
    return

  denorm_label = (unit_df.iloc[:, -1] + 1) / 2 * 125

  plt.figure(figsize=(10, 5))
  plt.plot(unit_df.iloc[:, 1], denorm_label, marker='')
  plt.title(f'RUL over Time for Engine #{unit_number}')
  plt.xlabel('Cycle')
  plt.ylabel('RUL')
  plt.grid(True)

  plt.savefig(filename)
  plt.close()

  files.download(filename)

if __name__ == "__main__":
  train_data, test_data = preprocess()
  plot_rul(train_data[0], 7, 'train1_plot_7')
  plot_rul(test_data[0], 7, 'test1_plot_7')
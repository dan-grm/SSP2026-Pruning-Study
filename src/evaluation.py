import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import re
from google.colab import files

df_structured = pd.read_csv("structured_pruning_data_3.csv")
df_unstructured = pd.read_csv("unstructured_pruning_data_3.csv")

def linearFunc(x, a, b):
  return a + b * x
def exp_decay(x, a, b, c):
  return a * np.exp(-b * x) + c

df_structured

def parse_metric_name(name):
  match = re.match(r"^(big|small)_(.+)_(pruned|tuned)_(\d+)$", name)
  if match:
    model, metric_type, method, ratio = match.groups()
    return pd.Series([model, metric_type, method, int(ratio)])
  else:
    return pd.Series([None, None, None, None])

df_structured[['model', 'metric_type', 'method', 'ratio']] = df_structured['metric'].apply(parse_metric_name)
df_unstructured[['model', 'metric_type', 'method', 'ratio']] = df_unstructured['metric'].apply(parse_metric_name)


def get_xy(
  df=df_structured,
  model='big',
  metric='rmse',
  method='pruned',
  x='ratio',
  y='avg'
):
  df_copy = df[(df.model == model) & (df.metric_type == metric) & (df.method == method)]
  x_list = [float(val) for val in df_copy[x].tolist()]
  y_list = [float(val) for val in df_copy[y].tolist()]
  return x_list, y_list

def plot_trend(x, y1, y2, std1, std2, color1, color2, label1, label2, func=linearFunc, func2=linearFunc, p0=None, p02=None, x2=None):
  # x2 defaults to x: the two series share a ratio grid unless told otherwise.
  # The tuned series does not (no tuned_0 measurement exists), so plot_metric_inter passes its own.
  x = np.asarray(x)
  x2 = x if x2 is None else np.asarray(x2)
  y1 = np.asarray(y1)
  y2 = np.asarray(y2)
  std1 = np.asarray(std1)
  std2 = np.asarray(std2)

  # add tiny std because if 0 code breaks
  std1[std1 == 0] = 1e-9
  std2[std2 == 0] = 1e-9

  a_fit1, cov1 = curve_fit(func, x, y1, sigma=std1, absolute_sigma=True, p0=p0)
  a_fit2, cov2 = curve_fit(func2, x2, y2, sigma=std2, absolute_sigma=True, p0=p02)

  yfit1 = func(x, *a_fit1)
  yfit2 = func2(x2, *a_fit2)

  plt.plot(x, yfit1, color=color1, linestyle='--', label=label1, alpha=0.3)
  plt.plot(x2, yfit2, color=color2, linestyle='--', label=label2, alpha=0.3)

  return a_fit1, a_fit2

def plot_metric_inter(
  df=df_structured,
  model='big',
  metric='rmse',
  download=False,
  trendline=False,
  func=linearFunc,
  func2=linearFunc,
  p0=None,
  p02=None,
):
  # The tuned series has its own ratios: no model is fine-tuned at ratio 0, so
  # it runs 10..90 while the pruned series runs 0..90. Reusing the pruned
  # x-values for both would mismatch the series lengths.
  x, y1 = get_xy(df=df, model=model, metric=metric, method='pruned')
  x2, y2 = get_xy(df=df, model=model, metric=metric, method='tuned')

  _, std1 = get_xy(df=df, model=model, metric=metric, method='pruned', y='std')
  _, std2 = get_xy(df=df, model=model, metric=metric, method='tuned', y='std')

  prune_method = 'structured' if df is df_structured else 'unstructured'

  fig, ax = plt.subplots()

  ax.scatter(x, y1, label='Pruned', marker='o', color='blue', alpha=0.7)
  ax.scatter(x2, y2, label='Tuned', marker='.', color='orange', alpha=0.7)

  ax.errorbar(x, y1, yerr=std1, fmt='none', ecolor='blue', elinewidth=1.5, capsize=3, alpha=0.5)
  ax.errorbar(x2, y2, yerr=std2, fmt='none', ecolor='orange', elinewidth=1.5, capsize=3, alpha=0.5)

  if trendline:
    plot_trend(x,y1,y2,std1,std2,'blue','orange','Pruned Fit','Tuned Fit',func,func2=func2,p0=p0,p02=p02,x2=x2)

  ax.set_title(f'{metric.upper()} Plot for the {model} model ({prune_method})')
  ax.set_xlabel('Ratio')
  ax.set_ylabel(metric.upper())

  ax.legend()

  filename = f'{metric}_{model}_{prune_method}.png'

  #plt.yticks(np.arange(min(y1), max(y1)+1, (max(y1)+1)*0.1))
  plt.tight_layout()

  if download:
    plt.savefig(filename, bbox_inches='tight')
    files.download(filename)
    plt.close()
  else:
    plt.show()

def plot_metric_intra(
  df=df_structured,
  df2=df_unstructured,
  model='big',
  metric='rmse',
  method='pruned',
  download=False,
  trendline=False,
  func=linearFunc,
  p0=None,
  func2=linearFunc,
  p02=None,
):
  x, y1 = get_xy(df=df, model=model, metric=metric, method=method)
  _, y2 = get_xy(df=df2, model=model, metric=metric, method=method)

  _, std1 = get_xy(df=df, model=model, metric=metric, method=method, y='std')
  _, std2 = get_xy(df=df2, model=model, metric=metric, method=method, y='std')

  prune_method = 'pruned' if method == 'pruned' else 'tuned'

  fig, ax = plt.subplots()

  ax.scatter(x, y1, label='Structured', marker='x', color='red', alpha=0.7)
  ax.scatter(x, y2, label='Unstructured', marker='+', color='green', alpha=0.7)

  ax.errorbar(x, y1, yerr=std1, fmt='none', ecolor='red', elinewidth=1.5, capsize=5, alpha=0.5)
  ax.errorbar(x, y2, yerr=std2, fmt='none', ecolor='green', elinewidth=1.5, capsize=5, alpha=0.5)

  if trendline:
    print(plot_trend(x,y1,y2,std1,std2,'red','green','Structured Fit','Unstructured Fit',func=func,func2=func2,p0=p0,p02=p02))

  ax.set_title(f'{metric.upper()} Plot for the {model} model ({prune_method})')
  ax.set_xlabel('Ratio')
  ax.set_ylabel(metric.upper())

  ax.legend()

  filename = f'{metric}_{model}_{prune_method}.png'

  #plt.yticks(np.arange(min(min(y1),min(y2)), max(max(y1),max(y2))+1, (max(max(y1),max(y2))+1)*0.1))
  plt.tight_layout()

  if download:
    plt.savefig(filename, bbox_inches='tight')
    files.download(filename)
    plt.close()
  else:
    plt.show()

prunes = [df_structured, df_unstructured]
models = ['big', 'small']
metrics = ['rmse', 'score', 'size_mb', 'memory_mb', 'inference_ms', 'params', 'MACs', 'FLOPs']
methods = ['pruned', 'tuned'] # only relevant for intra plotting, inter uses both

for df in prunes:
  for model in models:
    for metric in metrics:
      use_trendline = metric != 'rmse' and metric != 'score'
      plot_metric_inter(df=df, model=model, metric=metric, download=False, trendline=use_trendline)

for model in models:
  for method in methods:
    for metric in metrics:
      use_trendline = metric != 'rmse' and metric != 'score'
      plot_metric_intra(model=model, metric=metric, method=method, download=False, trendline=use_trendline)

# automatic generation of plots above was not sufficient for some plots, so they were done separately
plot_metric_intra(metric='params', trendline=True, func=exp_decay, p0=(1.4e9, 0.06, 0)) # p0 is an estimate for the best fit
plot_metric_intra(model='small', metric='params', trendline=True, func=exp_decay, p0=(1.4e9, 0.06, 0))
plot_metric_intra(metric='MACs', trendline=True, func=exp_decay, p0=(1.4e9, 0.06, 0))
plot_metric_intra(model='small', metric='MACs', trendline=True, func=exp_decay, p0=(1.4e9, 0.06, 0))
plot_metric_intra(metric='FLOPs', trendline=True, func=exp_decay, p0=(1.4e9, 0.06, 0))
plot_metric_intra(model='small', metric='FLOPs', trendline=True, func=exp_decay, p0=(1.4e9, 0.06, 0))
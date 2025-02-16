
import wandb
wandb.login(anonymous="allow")

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score
from torch.optim.lr_scheduler import CyclicLR
from sklearn.metrics import accuracy_score

from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression

torch.manual_seed(42)
np.random.seed(42)


def prepare_data(df_data, test_size_param):
  numpy_data = df_data.to_numpy()
  X = numpy_data[:, 3:28]
  y = numpy_data[:, -1:]
  feature_names = df_data.columns[3:28]  # Save the feature names

  #normalize
  scaler = MinMaxScaler()
  normalized_X = scaler.fit_transform(X)

  X_train_np, X_test_np, y_train_np, y_test_np = train_test_split(normalized_X, y, test_size = test_size_param, random_state = 42, shuffle = False)

  #convert from numpy to tensor
  X_train = torch.from_numpy(X_train_np.astype(np.float32))
  X_test = torch.from_numpy(X_test_np.astype(np.float32))
  y_train = torch.from_numpy(y_train_np.astype(np.float32))
  y_test = torch.from_numpy(y_test_np.astype(np.float32))

  return X_train, X_test, y_train, y_test, feature_names

def perform_rfe(X_train, X_test, y_train, feature_names, num_features_to_keep):
    estimator = LinearRegression()
    selector = RFE(estimator, n_features_to_select=num_features_to_keep)
    selector = selector.fit(X_train, y_train)


    selected_features_mask = selector.support_ #returns boolean array
    selected_feature_names = feature_names[selected_features_mask]

    #drop features
    X_train_selected = X_train[:, selected_features_mask]
    X_test_selected = X_test[:, selected_features_mask]

    return X_train_selected, X_test_selected, selected_feature_names

class MLP(nn.Module):
  def __init__(self, input_size, hidden_size, output_size):
    super(MLP, self).__init__()
    self.hidden_size = hidden_size
    self.hidden1 = nn.Linear(input_size, hidden_size)
    self.hidden2 = nn.Linear(hidden_size, hidden_size-5)
    self.hidden3 = nn.Linear(hidden_size-5, hidden_size-10)
    self.output = nn.Linear(hidden_size-10, output_size)
    self.relu = nn.ReLU()

  def forward(self, x):
    out = self.relu(self.hidden1(x))
    out = self.relu(self.hidden2(out))
    out = self.relu(self.hidden3(out))
    out = self.output(out)
    return out

def prediction_and_loss(model, X, y):
   y_predicted = model(X)

   MSE_loss_func = nn.MSELoss()
   mse_loss = MSE_loss_func(y_predicted, y)

   mse_loss_item = mse_loss.item()
   rmse_loss = np.sqrt(mse_loss_item)

   #calculate r2 score
   y_predicted_numpy = y_predicted.detach().numpy()
   y_numpy = y.detach().numpy()
   r2_value = r2_score(y_numpy, y_predicted_numpy)

   return y_predicted, mse_loss, rmse_loss, r2_value

def shuffle_tensors(tensor1, tensor2):
  indices = torch.randperm(len(tensor1))
  shuffled1 = tensor1[indices]
  shuffled2 = tensor2[indices]
  return shuffled1, shuffled2

def epoch_of_training(X_train, y_train, batch_size, optimizer, model, scheduler):
  shuffled_X_train, shuffled_y_train = shuffle_tensors(X_train, y_train)

  num_batches = len(shuffled_X_train) // batch_size
  losses = []

  for i in range(num_batches):
    X_batch = shuffled_X_train[i*batch_size: (i+1)*batch_size]
    y_batch = shuffled_y_train[i*batch_size: (i+1)*batch_size]

    optimizer.zero_grad()

    _, mse_loss_batch, _, _ = prediction_and_loss(model, X_batch, y_batch)

    mse_loss_batch.backward()
    losses.append(mse_loss_batch.item())

    optimizer.step()

  scheduler.step()
  return optimizer, model, scheduler

def splitting_data(unsplit_data, num_of_bases):

   base_size = len(unsplit_data) // num_of_bases

   split_data = []
   for i in range(num_of_bases):
      base = unsplit_data[i*base_size: (i+1)*base_size]
      split_data.append(base)

   return split_data

def create_ensemble(models, acc_list, X):
  weighted_sum = 0
  for i in range(len(models)):
    weighted_sum += acc_list[i] * (models[i])(X)

  total_acc = sum(acc_list)
  weighted_average = weighted_sum / total_acc

  return weighted_average #prediction

def train(config):
  #config = wandb.config
  torch.manual_seed(42)
  np.random.seed(42)

  #get data
  data_a = pd.read_excel(config.source_data)
  data_b = pd.read_excel(config.target_data)


  test_size_param = 0.2
  X_a_train, X_a_test, y_a_train, y_a_test, feature_names = prepare_data(data_a, test_size_param)
  X_b_train, X_b_test, y_b_train, y_b_test, feature_names = prepare_data(data_b, test_size_param)

  #drop uninmportant param.
  X_a_train_rfe, X_a_test_rfe, selected_feature_names_a = perform_rfe(X_a_train, X_a_test, y_a_train, feature_names, config.num_features_to_keep)
  X_b_train_rfe, X_b_test_rfe, selected_feature_names_b = perform_rfe(X_b_train, X_b_test, y_b_train, feature_names, config.num_features_to_keep)

  split_X_list = splitting_data(X_a_train_rfe, config.num_of_bases)
  split_y_list = splitting_data(y_a_train, config.num_of_bases)

  input_size = X_a_train_rfe.shape[1]
  output_size = 1

  #domain
  models = []
  for i in range(config.num_of_bases):
    model = MLP(input_size, config.hidden_size, output_size)
    optimizer = torch.optim.Adam(model.parameters(), lr = 0.01, weight_decay = config.weight_decay_a)
    scheduler = CyclicLR(optimizer, base_lr=config.base_lr_a, max_lr=config.max_lr_a, step_size_up=200, step_size_down=200, mode='triangular2', cycle_momentum=False)

    for epoch_a in range(config.epochs_a):
      optimizer, model, scheduler = epoch_of_training(split_X_list[i], split_y_list[i], config.batch_size_a, optimizer, model, scheduler)

    models.append(model)



  #target (training and validation)
  acc_test_size = len(X_b_test_rfe) // 2

  acc_test_X = X_b_test_rfe[:acc_test_size]
  acc_test_y = y_b_test[:acc_test_size]

  en_test_X = X_b_test_rfe[acc_test_size:]
  en_test_y = y_b_test[acc_test_size: ]

  acc_list = []

  for i in range(config.num_of_bases):
    models[i].output = nn.Linear(config.hidden_size - 10 , output_size) #change output layer

    params = [p for n,p in models[i].named_parameters()] #if "hidden1" not in n] #exclude hidden1 from optimizer
    optimizer = torch.optim.Adam(params, lr = 0.02, weight_decay = config.weight_decay_b)
    scheduler = CyclicLR(optimizer, base_lr=config.base_lr_b, max_lr=config.max_lr_b, step_size_up=200, step_size_down=200, mode='triangular2', cycle_momentum=False)

    for epoch_a in range(config.epochs_b):
       optimizer, models[i], scheduler = epoch_of_training(X_b_train_rfe, y_b_train, config.batch_size_b, optimizer, models[i], scheduler)

    _, _, _, r2_value = prediction_and_loss(models[i], acc_test_X, acc_test_y)
    acc_list.append(r2_value)

  en_prediction = create_ensemble(models, acc_list, en_test_X)

  en_predicted_numpy = en_prediction.detach().numpy()
  y_numpy = en_test_y.detach().numpy()
  r2_value = r2_score(y_numpy, en_predicted_numpy)

  MSE_loss_func = nn.MSELoss()
  mse_loss = MSE_loss_func(en_prediction, en_test_y)
  mse_loss_item = mse_loss.item()
  rmse_loss = np.sqrt(mse_loss_item)

  metrics = {"target/RMSE_loss": rmse_loss,
             "target/R2_value": r2_value}
  wandb.log(metrics)



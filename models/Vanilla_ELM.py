import wandb
wandb.login(anonymous="allow")

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score

from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression

def prepare_data(df_data, test_size_param):
  numpy_data = df_data.to_numpy()
  X = numpy_data[:, 3:28]
  y = numpy_data[:, -1:]

  #normalize
  scaler = MinMaxScaler()
  normalized_X = scaler.fit_transform(X)

  X_train_np, X_test_np, y_train_np, y_test_np = train_test_split(normalized_X, y, test_size = test_size_param, random_state = 42, shuffle = True)

  #convert from numpy to tensor
  X_train = torch.from_numpy(X_train_np.astype(np.float32))
  X_test = torch.from_numpy(X_test_np.astype(np.float32))
  y_train = torch.from_numpy(y_train_np.astype(np.float32))
  y_test = torch.from_numpy(y_test_np.astype(np.float32))

  return X_train, X_test, y_train, y_test


class ELM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(ELM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.hidden_weights = nn.Parameter(torch.randn(input_size, hidden_size))
        self.hidden_biases = nn.Parameter(torch.randn(hidden_size))
        self.output_weights = None

    def forward(self, x):
        hidden_layer = torch.sigmoid(torch.matmul(x, self.hidden_weights) + self.hidden_biases)
        output = torch.matmul(hidden_layer, self.output_weights)
        return output

def train_elm(elm_model, X, y):
   with torch.no_grad():
     H = torch.sigmoid(torch.matmul(X, elm_model.hidden_weights) + elm_model.hidden_biases)
     elm_model.output_weights = torch.matmul(torch.pinverse(H), y)
   return elm_model

def calculate_loss(y_pred, y_test):
   MSE_loss_func = nn.MSELoss()
   mse_loss = MSE_loss_func(y_pred, y_test)

   mse_loss_item = mse_loss.item()
   rmse_loss = np.sqrt(mse_loss_item)

   #calculate r2 score
   y_pred_numpy = y_pred.detach().numpy()
   y_test_numpy = y_test.detach().numpy()
   r2_value = r2_score(y_test_numpy, y_pred_numpy)
   return mse_loss_item, rmse_loss, r2_value

def train(config):
  torch.manual_seed(42)
  np.random.seed(42)

  data = pd.read_excel(config.data)

  test_size_param = 0.2
  X_train, X_test, y_train, y_test = prepare_data(data, test_size_param)

  input_size = X_train.shape[1]
  elm_model = ELM(input_size, config.hidden_size, 1)

  elm_model = train_elm(elm_model, X_train, y_train)
  with torch.no_grad():
    y_pred = elm_model(X_test)

  mse_loss, rmse_loss, r2_value = calculate_loss(y_pred, y_test)

  metrics = {"RMSE_loss": rmse_loss,
                       "R2_value": r2_value}

  wandb.log(metrics)

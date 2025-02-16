
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

def split_data(data):
  size = data.shape[0]
  ELM_size = size//4

  return  data[:ELM_size], data[ELM_size:2*ELM_size], data[2*ELM_size:3*ELM_size], data[3*ELM_size:4*ELM_size]


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

  X1, X2, X3, X4 = split_data(X_train)
  y1, y2, y3, y4 = split_data(y_train)

  X_list = [X1, X2, X3, X4]
  y_list = [y1, y2, y3, y4]

  input_size = X_train.shape[1]
  models = []
  predictions = []
  for i in range(config.model_count):
    model = ELM(input_size, config.hidden_size, 1)
    model = train_elm(model, X_list[i], y_list[i])
    pred = model(X_test)
    predictions.append(pred)
    models.append(model)

 

  y_pred_combined = torch.cat((pred[0], pred[1], pred[2], pred[3]), dim=1)

  en_y_pred = torch.mean(y_pred_combined, dim=1)
  en_y_pred = en_y_pred.view(-1, 1)  # turn to 2d

  en_mse_loss, en_rmse_loss, en_r2_value = calculate_loss(en_y_pred, y_test)

  metrics = {
    "ensemble/RMSE_loss": en_rmse_loss,
    "ensemble/R2_value": en_r2_value
  }

  wandb.log(metrics)

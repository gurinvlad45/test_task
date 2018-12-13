import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
import torch
from torch.autograd import Variable
import torch.optim as optim
from torch.autograd import Variable


# read data
data = pd.read_csv('train.csv')
data = data.values
data = data.astype('float32')

# divide features and labels
features_data = data[:, :-1]
features_data = features_data/1000
labels_data = data[:, -1]
labels_data = np.reshape(labels_data, (len(labels_data), 1))

n_features = features_data.shape[1]
n_units = features_data.shape[0]
n_labels = labels_data.shape[0]
n_classes = 1

features = Variable(torch.Tensor(features_data))
labels = Variable(torch.Tensor(labels_data))



import torch

dtype = torch.float
device = torch.device("cpu")
# device = torch.device("cuda:0")

N, D_in, H, D_out = n_units, n_features, n_features, n_classes


# fully connected neural network
class TwoLayerNet(torch.nn.Module):
    def __init__(self, D_in, H, D_out):
        """
        In the constructor we instantiate two nn.Linear modules and assign them as
        member variables.
        """
        super(TwoLayerNet, self).__init__()
        self.linear1 = torch.nn.Linear(D_in, H)
        self.relu = torch.nn.ReLU()
        self.linear2 = torch.nn.Linear(H, D_out)

    def forward(self, x):
        """
        In the forward function we accept a Tensor of input data and we must return
        a Tensor of output data. We can use Modules defined in the constructor as
        well as arbitrary (differentiable) operations on Tensors.
        """
        h_relu = self.linear1(x).clamp(min=0)
        y_pred = self.linear2(h_relu)
        return y_pred


# CONSTRUCT MODEL, COUNT LOSS, DEFINE OPTIMIZER

model = TwoLayerNet(D_in, H, D_out)
loss_fn = torch.nn.MSELoss(reduction='sum')
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)


# INITIALIZE FIRST LAYER WEIGHTS
w1 = torch.randn(D_in, device=device, dtype=dtype)

for t in range(100):

    # define output after one-to-one connection
    new_features = []
    for i in range(n_units):
        new_features.append(features[i] * w1)
    new_features = torch.stack(new_features)
    new_features_tensor = Variable(torch.Tensor(new_features))
    print('features', new_features.shape)

    # get predictions after fully-connected
    y_pred = model(new_features)
    print('Predictions', y_pred.shape)

    # count loss
    loss_1 = loss_fn(y_pred, labels)
    loss = (y_pred - labels).pow(2).sum().item()
    print('loss', t, loss)

    # Zero gradients, perform a backward pass, and update the weights.
    optimizer.zero_grad()
    loss_1.backward()
    optimizer.step()

    # count gradient for updating weights
    grad_y_pred = 2.0 * (y_pred - labels)
    grad_y_pred[y_pred < 0] = 0
    # print('gradient', grad_y_pred.shape)
    new_grad_w1 = []
    for j in range(n_units):
        print(features[j]*grad_y_pred)
        new_grad_w1.append(features[j]*grad_y_pred)
    new_grad_w1 = torch.stack(new_grad_w1)
    print(new_grad_w1.shape)

    # Update weights using gradient descent
    learning_rate = 1e-6
    w1 -= learning_rate * new_grad_w1
    print(w1.shape)

    s = torch.sum(model.fc2.weight.data)
    print(s)
    print(w1)


# TOP 10 IMPORTANT FEATURES
def get_best_features(all_features_score):
    sorted_params = sorted(all_features_score, key=abs)
    best_10 = sorted_params[-10:]
    best_feature_indexes = [all_features_score.index(i) for i in best_10]
    return best_feature_indexes


print(get_best_features(w1.tolist()))

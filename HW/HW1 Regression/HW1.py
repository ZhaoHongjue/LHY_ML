import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
from torch import nn
from torch import batch_norm
from torch.nn import functional as F
from torch.utils import data

from sklearn.preprocessing import MinMaxScaler

def load_Dataset(data_arrays, batch_size, shuffle = True):
    dataset = data.TensorDataset(*data_arrays)
    return data.DataLoader(dataset, batch_size, shuffle)

def preprocess_data(train_set, test_set):
    # Data Preprocess
    features = train_set.iloc[:, :-1].values
    labels = train_set.iloc[:, -1].values

    # Min-Max Scale
    scale = MinMaxScaler()
    features_scaled = scale.fit_transform(features)
    test_set_scaled = scale.transform(test_set.values)

    labels_scaled = labels / 100
    
    # transfer these parameters to tensor
    features_scaled_tensor = torch.FloatTensor(features_scaled)
    labels_scaled_tensor = torch.FloatTensor(labels_scaled)
    test_tensor = torch.FloatTensor(test_set_scaled)
    
    return features_scaled_tensor, labels_scaled_tensor, test_tensor
    
def train(model, loss, Optimizer, train_iter):
    loss_list = []
    for epoch in range(epochs):
        l_sum = 0
        for X, y in train_iter:
            l = loss(model(X).reshape(-1), y)
            Optimizer.zero_grad()
            l.backward()
            Optimizer.step()
            l_sum += float(l)
        loss_list.append(l_sum)
        print(f'epoch: {epoch}, loss: {l_sum:.3f}')
        
    plt.plot(loss_list)
    plt.title('MSE loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.savefig('./loss.png')
    
# hyperparameters
batch_size = 100
lr = 0.1
epochs = 20

if __name__ == '__main__':
    # Read Data
    train_set = pd.read_csv('./Data/covid.train.csv')
    test_set = pd.read_csv('./Data/covid.test.csv')
    
    #Preprocess
    features, labels, test = preprocess_data(train_set, test_set)
    
    # Load Dataset
    train_iter = load_Dataset((features, labels), batch_size)
    
    # Define Model, loss, Optimizer, and train
    model = nn.Sequential(
        nn.Linear(94, 128), nn.ReLU(),
        nn.Linear(128, 128), nn.ReLU(),
        nn.Linear(128, 1)
    )

    loss = nn.MSELoss()
    Optimizer = torch.optim.SGD(model.parameters(), lr)

    train(model, loss, Optimizer, train_iter)
    
    with torch.no_grad():
        predictions = model(features.detach()).reshape(-1)
        
        plt.clf()
        plt.plot(predictions, label = 'predictions')
        plt.plot(labels, label = 'labels')
        plt.legend()
        plt.title('train set')
        plt.savefig('./train_set.png')

    
    test_pred = model(test).reshape(-1, 1).detach().numpy()
    
    plt.clf()
    plt.plot(test_pred)
    plt.title('test set')
    plt.savefig('./test_set.png')
    
    id = np.arange(test_pred.shape[0], dtype = int).reshape(-1, 1)
    Submission = pd.DataFrame(np.concatenate((id, test_pred * 100), axis = 1), 
                            columns = ['id', 'tested_positive'])
    Submission.set_index(['id'], inplace=True)
    Submission.to_csv('./Submission.csv')
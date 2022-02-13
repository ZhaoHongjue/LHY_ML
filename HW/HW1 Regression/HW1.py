import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
from torch import nn
from torch.utils import data

from sklearn.preprocessing import MinMaxScaler

# hyperparameters
split_ratio = .2
batch_size = 150
epochs = 800
lr = 0.1

# utils
def plot(arrays, labels = None, xlabel = None, ylabel = None, title = None, grid = True, save = False):
    assert type(arrays) == tuple
    plt.clf()
    for i in range(len(arrays)):
        if labels is not None:
            plt.plot(arrays[i], label = labels[i])
        else:
            plt.plot(arrays[i])
    if labels is not None:
        plt.legend()
        
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    if grid:
        plt.grid()
    if save:
        plt.savefig('./' + title + '.png')
 
# preprocess data       
def preprocess_data(train_raw, test_raw, split_ratio = 0.2):
    # Data Preprocess
    test_raw['temp'] = np.zeros_like(test_raw.iloc[:, 0])
    train_data = train_raw.values
    test_data = test_raw.values
    
    # Min-Max Scale
    scale = MinMaxScaler()
    train_scaled = scale.fit_transform(train_data)
    test_scaled = scale.transform(test_data)[:, :-1]
    
    # Split Data to Get Valid Set
    train_num = int((1 - split_ratio) * train_scaled.shape[0])
    train = torch.FloatTensor(train_scaled[:train_num, :])
    valid = torch.FloatTensor(train_scaled[train_num:, :])
    test = torch.FloatTensor(test_scaled)
    
    return train, valid, test, scale

# load dataset
def load_Dataset(data_arrays, batch_size, shuffle = True):
    # Load Dataset
    dataset = data.TensorDataset(*data_arrays)
    return data.DataLoader(dataset, batch_size, shuffle)

def train_model(model, loss, optimizer, train_iter, valid_iter, epochs):
    train_loss_list = []
    valid_loss_list = []
    for epoch in range(epochs):
        train_loss = 0
        model.train()
        for X, y in train_iter:
            l = loss(model(X).reshape(-1), y)
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
            train_loss += float(l)
        train_loss_list.append(train_loss)
        
        valid_loss = 0
        model.eval()
        with torch.no_grad():
            for X, y in valid_iter:
                valid_loss += float(loss(model(X).reshape(-1), y))
            valid_loss_list.append(valid_loss)
        
        print(f'epoch: {epoch}, train loss: {train_loss:.3f}, test loss: {valid_loss:.3f}')
    
    plot((train_loss_list, valid_loss_list), labels = ('train loss', 'valid loss',),
         xlabel = 'epoch', ylabel = 'MSE loss', title = 'loss', save = True)

if __name__ == '__main__':
    # read data
    train_raw = pd.read_csv('./Data/covid.train.csv')
    test_raw = pd.read_csv('./Data/covid.test.csv')
    
    train, valid, test, scale = preprocess_data(train_raw, test_raw)
    train_iter = load_Dataset((train[:, :-1], train[:, -1]), batch_size)
    valid_iter = load_Dataset((valid[:, :-1], valid[:, -1]), batch_size)
    model = nn.Sequential(
        nn.Linear(94, 128), nn.ReLU(),
        nn.Dropout(0.4),
        nn.Linear(128, 128), nn.ReLU(),
        nn.Dropout(0.4),
        nn.Linear(128, 1)
    )
    loss = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr)

    train_model(model, loss, optimizer, train_iter, valid_iter, epochs)
    
    # plot curves
    model.eval()
    with torch.no_grad():
        # on train set
        train_pred = model(train[:, :-1]).reshape(-1)
        plot((train_pred, train[:, -1]), labels = ('predictions', 'labels'),
            title= 'train set', save = True)
        
        # on valid set
        valid_pred = model(valid[:, :-1]).reshape(-1)
        plot((valid_pred, valid[:, -1]), labels = ('predictions', 'labels'),
            title= 'valid set', save = True)
        
        # on test set
        test_pred = model(test).detach().numpy().reshape(-1)
        plot((test_pred,), title = 'test set', save = True)
        
    # inverse scale
    temp = np.zeros_like(test_raw)
    temp[:, -1] = test_pred
    test_positive = scale.inverse_transform(temp)[:, -1]
    
    # get submission
    id = range(len(test_pred))
    Submission = pd.DataFrame(list(zip(id, test_positive)), 
                            columns = ['id', 'tested_positive'])
    Submission.set_index(['id'], inplace = True)
    Submission.to_csv('./Submission.csv')
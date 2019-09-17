import pandas as pd
import numpy as np
import csv

from sklearn import svm
from sklearn.linear_model import LogisticRegression

import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd.variable as Variable

import TitanicDataset as tc
import nn_model

BATCH_SIZE = 3

def svm_model(train):
    X = train.drop(['Survived'], axis=1)
    y = train['Survived']
    clf = svm.LinearSVC(loss='hinge', max_iter=1000)
    clf.fit(X, y)

    return clf

def logostiC_regression_model(train):
    X = train.drop(['Survived'], axis=1)
    y = train['Survived']
    lr = LogisticRegression()
    lr.fit(X, y)

    return lr

def validation(dev, model):
    X = dev.drop(['Survived'], axis=1)
    Y = dev.filter(['Survived'], axis=1)

    acc = 0
    for (_, sample), (_, y) in zip(X.iterrows(), Y.iterrows()):
        d = np.array(sample).reshape(1, -1)
        y_hat = model.predict(d)
        if y[0] == y_hat:
            acc += 1

    return acc / X.shape[0]

def cross_validation(data, k):
    dev_size = int((len(data)) * (1 / k))

    score_sum = 0
    for i in range(k):
        train = pd.concat([data[:i * dev_size], data[i * dev_size + dev_size:]])
        dev = data[i * dev_size:i * dev_size + dev_size]

        model = svm_model(train)
        score_sum += validation(dev, model)

    return score_sum / k

def test(test_data, model):
    model.eval()
    predictions = {}
    for sample, passenger_id in test_data:
        pred = model.predict(sample).tolist()
        passenger_id = int(passenger_id.tolist()[0])
        predictions[str(passenger_id)] = str(pred)


    with open('predictions.csv', 'w') as f:
        for key in predictions.keys():
            f.write("%s,%s\n" % (key, predictions[key]))

    m=9

def dev(dev_set, dev_size, model, criterion):
    model.eval()

    acc = 0
    for sample, target in dev_set:
        X = sample
        Y = target.reshape(-1)

        # forward + backward + optimize
        outputs = model(X)
        loss = criterion(outputs, Y)

        pred = outputs.argmax(dim=1, keepdim=True)
        acc += pred.eq(Y.view_as(pred)).sum().item()

    return acc / dev_size

def train(train_set, train_size, dev_set, dev_size, model, optimizer, criterion):

    model.train()
    EPOCHS = 27
    for epoch in range(EPOCHS):

        acc = 0
        for i, (sample, target) in enumerate(train_set):
            X = sample
            Y = target.reshape(-1)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(X)
            loss = criterion(outputs, Y)
            loss.backward()
            optimizer.step()

            pred = outputs.argmax(dim=1, keepdim=True)
            acc += pred.eq(Y.view_as(pred)).sum().item()

        print("EPOCH - " + str(epoch) + ", train accuracy: " + str(acc/train_size))

        dev_acc = dev(dev_set, dev_size, model, criterion)
        print("dev acc: " + str(dev_acc))

    return model


def main():
    data_set = pd.read_csv("train.csv")
    dev_size = int((len(data_set)) * (1 / 5))
    train_set = data_set[dev_size:]
    dev_set = data_set[:dev_size]

    train_data = tc.TitanicDataset(train_set)
    dev_data = tc.TitanicDataset(dev_set)

    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=BATCH_SIZE, shuffle=True, sampler=None)

    dev_loader = torch.utils.data.DataLoader(
        dev_data, batch_size=BATCH_SIZE, shuffle=True, sampler=None)

    FEATURES = train_data[0][0].size()[1]

    criterion = nn.CrossEntropyLoss()
    model = nn_model.MLP(FEATURES, BATCH_SIZE)
    #optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=0.0001)
    optimizer = optim.SGD(model.parameters(), lr=0.01)#, weight_decay=0.01)

    model = train(train_loader, len(train_data), dev_loader, dev_size, model, optimizer, criterion)

    test_data = tc.TitanicDataset(pd.read_csv("test.csv"), is_train=False)
    test_loader = torch.utils.data.DataLoader(
        test_data, batch_size=1, shuffle=False, sampler=None)

    test(test_loader, model)


    c=8


if __name__ == '__main__':
    main()
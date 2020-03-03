import time
import os
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score
import numpy as np

from preprocess import load_cnn_trainset, load_cnn_valset, \
    load_cnn_testset, load_embeddings_as_tensor
from utils import save_result


class CNNMessageClassifier(nn.Module):

    def __init__(self, n_classes, embeddings, dropout_prob=0.5,
                 kernel_num=100, kernel_sizes=[2, 3, 4, 5]):
        super(CNNMessageClassifier, self).__init__()

        self.embedding = nn.Embedding.from_pretrained(embeddings, freeze=True)
        self.convs = nn.ModuleList(
            [nn.Conv2d(1, kernel_num, (k, embeddings.size(1))) for k in kernel_sizes])
        self.dropout = nn.Dropout(p=0.5)
        self.fc = nn.Sequential(
            nn.Linear(kernel_num * len(kernel_sizes), 200),
            nn.ReLU(),
            nn.Linear(200, n_classes)
        )

    def forward(self, x):
        # [N, W]
        x = self.embedding(x)
        # [N, W, embedding_dim]
        x = x.unsqueeze(1)
        # [N, 1, W, embedding_dim]
        x_list = [F.relu(conv(x)).squeeze(3) for conv in self.convs]
        # [N, kernel_num, W] * len(kernel_sizes)
        x_list = [F.max_pool1d(x, x.size(2)).squeeze(2) for x in x_list]
        # [N, kernel_num*len(kernel_sizes)]
        x = torch.cat(x_list, 1)
        # [N, kernel_num*len(kernel_sizes)]
        x = self.dropout(x)
        x = self.fc(x)
        # [N, n_classes]
        return x


class Text2EmojiTrainset(Dataset):
    def __init__(self):
        X, y = load_cnn_trainset()
        self.X, self.y = torch.LongTensor(X), torch.LongTensor(y)

    def __len__(self):
        return self.X.size(0)

    def __getitem__(self, index):
        return self.X[index], self.y[index]


class Text2EmojiValset(Dataset):
    def __init__(self):
        X, y = load_cnn_valset()
        self.X, self.y = torch.LongTensor(X), torch.LongTensor(y)

    def __len__(self):
        return self.X.size(0)

    def __getitem__(self, index):
        return self.X[index], self.y[index]


class Text2EmojiTestset(Dataset):
    def __init__(self):
        X = load_cnn_testset()
        self.X = torch.LongTensor(X)

    def __len__(self):
        return self.X.size(0)

    def __getitem__(self, index):
        return self.X[index]


MODELS_DIR = 'models'


def save_model(model, name):
    save_path = os.path.join(MODELS_DIR, name + '.pt')
    torch.save(model.state_dict(), save_path)


def load_model(model, name):
    load_path = os.path.join(MODELS_DIR, name + '.pt')
    model.load_state_dict(torch.load(load_path, map_location='cpu'))


def train(model, name, max_epoch=10, batch_size=100,
          lr=1e-4, weight_decay=0.0):
    trainset = Text2EmojiTrainset()
    valset = Text2EmojiValset()
    trainloader = DataLoader(trainset, batch_size=batch_size,
                             shuffle=True, num_workers=4)
    validateloader = DataLoader(valset, batch_size=10000,
                                shuffle=False, num_workers=4)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr,
                           weight_decay=weight_decay)
    scheduler = lr_scheduler.ExponentialLR(optimizer, 0.8)

    print('Begin training. batch size: %d, epoch: %d, learning rate: %f' %
          (batch_size, max_epoch, lr))

    best_score = 0.0
    model.train()
    for epoch in range(max_epoch):
        scheduler.step()

        n_sample = 0
        iter_ctr = 0
        train_loss = 0.0
        train_score = 0
        time_start = time.time()

        for X_train, y_train in trainloader:
            optimizer.zero_grad()
            outputs = model(X_train)
            loss = criterion(outputs, y_train)
            loss.backward()
            optimizer.step()

            iter_ctr += 1
            n_sample += batch_size
            train_loss += loss.item()
            train_score += eval_score(outputs, y_train)

            if iter_ctr == 10:
                # Calculate validation loss and score
                val_loss, val_score = validate(
                    model, criterion, validateloader)
                # Average training loss.
                train_loss /= iter_ctr
                train_score /= iter_ctr
                # Show results.
                print('[%d, %5d] training loss: %.3f, validation loss: %.3f, '
                      'train f1: %.3f, validation f1: %.3f, time %.2f sec' %
                      (epoch, n_sample, train_loss, val_loss,
                       train_score, val_score, time.time() - time_start))
                # Save the model if it is better than before
                if val_score >= best_score:
                    save_model(model, name)
                    best_score = val_score + 0.001
                    print('Model is saved with score: %.3f' % val_score)
                train_loss = 0.0
                train_score = 0
                iter_ctr = 0
                time_start = time.time()

    print('Training complete')


def eval_score(outputs, labels):
    pred = predict(outputs)
    pred, labels = pred.cpu(), labels.cpu()
    return f1_score(labels, pred, average='micro')


def predict(outputs):
    return torch.argmax(outputs, dim=1)


def validate(model, criterion, validateloader):
    model.eval()

    with torch.no_grad():
        outputs_list, labels_list = [], []
        for X_val, y_val in validateloader:
            outputs_list.append(model(X_val))
            labels_list.append(y_val)
        outputs = torch.cat(outputs_list, 0)
        labels = torch.cat(labels_list)
        loss = criterion(outputs, labels)
        score = eval_score(outputs, labels)

    model.train()
    return loss.item(), score


def test(model, result_file):
    print('Begin testing')
    model.eval()

    dataset = Text2EmojiTestset()
    testloader = DataLoader(dataset, batch_size=10000,
                            shuffle=False, num_workers=4)
    pred = []
    with torch.no_grad():
        for data in testloader:
            pred += list(predict(model(data)))
    save_result(result_file, pred)

    model.train()
    print('Testing complete')


def parse_args():
    parser = argparse.ArgumentParser(description='Train and test by cnn')
    parser.add_argument('cmd', choices=['train', 'test'],
                        help='sub-commands')
    parser.add_argument('-n', '--name', default='model',
                        help='the name of model')
    parser.add_argument('-d', '--dropout', type=float, default=0.5,
                        help='the dropout probability of the model')
    parser.add_argument('-k', '--kernel-num', dest='kernel_num',
                        type=float, default=200,
                        help='the kernel num of the model')
    parser.add_argument('-s', '--kernel-sizes', dest='kernel_sizes',
                        type=int, nargs='+', default=[2, 3, 4, 5],
                        help='the kernel num of the model')
    parser.add_argument('-e', '--epoch', type=int, default=5,
                        help='the max epoch for training')
    parser.add_argument('-b', '--batch', type=int, default=100,
                        help='the batch size for training')
    parser.add_argument('-l', '--lr', type=float, default=1e-4,
                        help='the learning rate for training')
    parser.add_argument('-o', '--output', default='cnn_result.csv',
                        help='the output result for testing')
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    net = CNNMessageClassifier(
        72, load_embeddings_as_tensor(), dropout_prob=args.dropout,
        kernel_num=args.kernel_num, kernel_sizes=args.kernel_sizes)
    if args.cmd == 'train':
        train(net, args.name, max_epoch=args.epoch,
              batch_size=args.batch, lr=args.lr)
    elif args.cmd == 'test':
        load_model(net, args.name)
        test(net, args.output)

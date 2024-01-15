import time
import os
import numpy as np
import random
import datetime
from eeg_dataset import *
from torch.utils.data import DataLoader
import torch.nn as nn
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score
import os.path as osp

use_cuda = torch.cuda.is_available()
print('use_cuda:', use_cuda)
device = torch.device('cuda:0' if use_cuda else 'cpu')


def seed_all(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    np.random.seed(seed)


def set_up(args):
    set_gpu(args.gpu)
    ensure_path(args.save_path)
    torch.manual_seed(args.random_seed)
    torch.backends.cudnn.deterministic = True


def set_gpu(x):
    torch.set_num_threads(1)
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ['CUDA_VISIBLE_DEVICES'] = x
    print('using gpu:', x)


def ensure_path(path):
    if os.path.exists(path):
        pass
    else:
        os.makedirs(path)


def get_dataloader(data, label, batch_size):
    # load the data  ; generator=torch.Generator(device=device),
    dataset = eegDataset(data, label)
    loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, pin_memory=False,
                        generator=torch.Generator(device=device), drop_last=True)
    return loader


def get_metrics(y_pred, y_true, classes=None):
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='macro')
    if classes is not None:
        cm = confusion_matrix(y_true, y_pred, labels=classes)
    else:
        cm = confusion_matrix(y_true, y_pred)
    return acc, f1, cm


def record_init(args):
    result_path = osp.join(args.save_path, 'result')
    ensure_path(result_path)
    text_file = osp.join(result_path,
                         "results_{}.txt".format(args.dataset))
    file = open(text_file, 'a')
    file.write("\n" + str(datetime.datetime.now()) +
               "\nTrain:Parameter setting for " + str(args.model) + ' on ' + str(args.dataset) +
               "\n1)number_class:" + str(args.num_class) +
               "\n2)random_seed:" + str(args.random_seed) +
               "\n3)learning_rate:" + str(args.learning_rate) +
               "\n4)pool:" + str(args.pool) +
               "\n5)num_epochs:" + str(args.max_epoch) +
               "\n6)batch_size:" + str(args.batch_size) +
               "\n7)dropout:" + str(args.dropout) +
               "\n8)hidden1_node:" + str(args.hidden1) + "hidden2_node:" + str(args.hidden2) +
               "\n9)input_shape:" + str(args.input_shape) +
               "\n10)train setting:" + str(args.train_session) + str(args.train_emotion) +
               "\n11)test setting:" + str(args.test_session) + str(args.test_emotion) +
               "\n12)T:" + str(args.T) + '\n')

    file.close()


def log2txt(content, args):
    result_path = osp.join(args.save_path, 'result')
    ensure_path(result_path)
    text_file = osp.join(result_path,
                         "results_{}.txt".format(args.dataset))
    file = open(text_file, 'a')
    file.write(str(content) + '\n')
    file.close()


class Averager():

    def __init__(self):
        self.n = 0
        self.v = 0

    def add(self, x):
        self.v = (self.v * self.n + x) / (self.n + 1)
        self.n += 1

    def item(self):
        return self.v


class LabelSmoothing(nn.Module):
    def __init__(self, smoothing=0.0):
        super(LabelSmoothing, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing

    def forward(self, x, target):
        logprobs = torch.nn.functional.log_softmax(x, dim=-1)
        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()


class Timer():

    def __init__(self):
        self.o = time.time()

    def measure(self, p=1):
        x = (time.time() - self.o) / p
        x = int(x)
        if x >= 3600:
            return '{:.1f}h'.format(x / 3600)
        if x >= 60:
            return '{}m'.format(round(x / 60))
        return '{}s'.format(x)


def resampling(data):
    mean = np.mean(data, axis=-1, keepdims=True)
    data = data - mean
    return data


def halved_data(train_data, train_label, test_data, test_label):
    train_data = train_data[::5]
    train_label = train_label[::5]
    test_data = test_data[::5]
    test_label = test_label[::5]
    return train_data, train_label, test_data, test_label


def normalize(input):
    # data: sample x 1 x channel x data  取各自的均值方差归一化
    for channel in range(input.shape[2]):
        input_mean = np.mean(input[:, :, channel, :])
        input_std = np.std(input[:, :, channel, :])
        input[:, :, channel, :] = (input[:, :, channel, :] - input_mean) / input_std
    return input


def normalize_v2(train, test):

    # data: sample x 1 x channel x data

    for channel in range(train.shape[2]):
        mean = np.mean(train[:, :, channel, :])
        std = np.std(train[:, :, channel, :])
        train[:, :, channel, :] = (train[:, :, channel, :] - mean)  # / std
        test[:, :, channel, :] = (test[:, :, channel, :] - mean)  # / std
    return train, test
















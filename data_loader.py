import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader



class StandardScaler():
    """
    Standard the input
    """

    def __init__(self, x):
        self.mean = x.mean()
        self.std = x.std()
        # self.mean1 = torch.from_numpy(self.mean).to('cuda:0')
        # self.std1 = torch.from_numpy(self.std).to('cuda:0')

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean


def load_data(filename,feature):

    df = pd.read_csv(filename)
    df = df.fillna(0)
    ts = df.iloc[:,1:feature+1]
    data = ts.values.astype("float32")  # (N, 1)
    print("time series shape:", data.shape)
    return data


# divide training and testing, default as 3:1
def split_data(dataset):

    num_samples=len(dataset)
    num_train = round(num_samples*0.7)
    num_test = round(num_samples*0.2)
    num_val = num_samples - num_test - num_train

    # train
    train = dataset[:num_train]
    # val
    val = dataset[num_train: num_train + num_val]
    # test
    test = dataset[-num_test:]
    return train, val, test


def create_samples(data, lookBack):

    dataX, dataY = [], []
    for i in range(len(data) - lookBack):
        sample_X = data[i:(i + lookBack), :]
        sample_Y = data[i + lookBack, :]
        dataX.append(sample_X)
        dataY.append(sample_Y)
    dataX = np.array(dataX)  # (N, lag, feature)
    dataY = np.array(dataY)  # (N, feature)

    return dataX, dataY


# 分割时间序列作为样本,支持多步预测
def create_multi_ahead_samples(data, lookBack, lookAhead=6):


    dataX, dataY = [], []
    for i in range(len(data) - lookBack - lookAhead):
        history_seq = data[i: i + lookBack]
        future_seq = data[i + lookBack: i + lookBack + lookAhead]
        dataX.append(history_seq)
        dataY.append(future_seq)
    dataX = np.array(dataX)
    dataY = np.array(dataY)
    return dataX, dataY



def plot(trainPred, trainY, testPred, testY):
    pred = np.concatenate((trainPred, testPred))
    gtruth = np.concatenate((trainY, testY))
    plt.plot(pred, 'g')
    plt.plot(gtruth, 'r')
    plt.show()


class Time_Series_Data(Dataset):
    DEFAULTS = {}
    def __init__(self, data_path, mode, feature, look_back, predict_step):

        self.mode = mode
        dataset = load_data(data_path, feature)
        train, val, test = split_data(dataset)

        self.scaler = StandardScaler(train)
        train = self.scaler.transform(train)

        x_train, y_train = create_samples(train,look_back)
        x_val, y_val = create_samples(val,look_back)
        if predict_step == 1:
            x_test, y_test = create_samples(test,look_back)
        else:
            x_test, y_test = create_multi_ahead_samples(test, look_back, predict_step)

        self.x_train = x_train
        self.y_train = y_train
        self.x_val = x_val
        self.y_val = y_val
        self.x_test = x_test
        self.y_test = y_test

    def __getitem__(self, item):
        if self.mode=='train':
            x_t = self.x_train[item]
            y_t = self.y_train[item]
        elif self.mode=='val':
            x_t = self.x_val[item]
            y_t = self.y_val[item]
        elif self.mode=='test':
            x_t = self.x_test[item]
            y_t = self.y_test[item]
        return x_t, y_t

    def __len__(self):

        if self.mode == 'train':
            return len(self.x_train)
        elif self.mode == 'val':
            return len(self.x_val)
        elif self.mode == 'test':
            return len(self.x_test)


def get_data_loader(data_path, feature, look_back, batch_size, mode, predict_step):

    if mode == 'train':
        shuffle = True
    else:
        batch_size = 1
        shuffle = False
    print('batch_size:', batch_size)
    data = Time_Series_Data(data_path, mode, feature, look_back, predict_step)
    return DataLoader(data, batch_size=batch_size, shuffle=shuffle, num_workers=4)



if __name__ == '__main__':

    data = load_data("E:/py_projects/GDN/data/CDR/result.csv",10)

    trainData, testData = split_data(data)
    lag = 24
    flag = False
    trainX, trainY = create_samples(trainData, lag, RNN=flag)
    testX, testY = create_samples(testData, lag, RNN=flag)
    print("testX shape:", testX.shape)
    print("testy shape:", testY.shape)
    print("trainX shape:", trainX.shape)
    print("trainy shape:", trainY.shape)

    dataset = Time_Series_Data(trainX, trainY)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True, sampler=None, batch_sampler=None, num_workers=4)

    for data, label in dataloader:
        print(data.shape)
        # print(data)
        print(label.shape)
        # print(label)

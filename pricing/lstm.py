from datetime import datetime
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.optim import SGD, Adam
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler


class LSTMDataSet(Dataset):
    def __init__(self, data_x, data_y):
        super(LSTMDataSet, self).__init__()
        self.data_x = data_x
        self.data_y = data_y

    def __len__(self):
        return self.data_x.shape[0]

    def __getitem__(self, item):
        return self.data_x[item], self.data_y[item]


class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_layers=12, output_size=1):
        super(LSTMModel, self).__init__()
        self.hidden_layers = hidden_layers
        self.lstm = nn.LSTM(input_size, hidden_layers, batch_first=True)
        self.fc = nn.Linear(hidden_layers, output_size)

        self.hidden_cell = (torch.zeros(output_size, 64, self.hidden_layers),
                            torch.zeros(output_size, 64, self.hidden_layers))

    def forward(self, x):
        x, _ = self.lstm(x, self.hidden_cell)
        out = self.fc(x[:, -1, :].squeeze(1))
        return out


class LSTMController:
    def create_lstm_dataset(self, data, look_back=10):
        data_x, data_y = [], []
        for i in range(len(data) - look_back):
            a = data[i: i + look_back]
            data_x.append(a)
            data_y.append(data[i + look_back])

        data_x, data_y = np.array(data_x), np.array(data_y)
        return torch.tensor(data_x).float(), torch.tensor(data_y).float()

    def train_one_epoch(self, data_train, model, criterion, optimizer):
        model.train()
        epoch_loss = 0
        num = 0
        for i, batch in enumerate(data_train):
            x, y = batch
            optimizer.zero_grad()  # set grad to 0
            model.hidden_cell = (torch.zeros(1, x.shape[0], model.hidden_layers),
                                 torch.zeros(1, x.shape[0], model.hidden_layers))
            outputs = model(x)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            num += len(y)

        return epoch_loss / num

    def val_one_epoch(self, data_val, model, criterion):
        model.eval()
        epoch_loss = 0
        num = 0
        with torch.no_grad():
            for i, batch in enumerate(data_val):
                x, y = batch
                model.hidden_cell = (torch.zeros(1, x.shape[0], model.hidden_layers),
                                     torch.zeros(1, x.shape[0], model.hidden_layers))
                outputs = model(x)
                loss = criterion(outputs, y)
                epoch_loss += loss.item()
                num += len(y)

        return epoch_loss / num

    def lstm_train(self, data_train, data_val, model, criterion, optimizer, epochs=20,
                   verbose=True, save_model=False, early_stop=True):
        best_loss = 2000

        early_stop_count = 0
        for i in range(epochs):
            epoch_train_loss = self.train_one_epoch(data_train, model, criterion, optimizer)
            epoch_val_loss = self.val_one_epoch(data_val, model, criterion)
            if verbose:
                print("epoch: {} tarin loss: {}; val loss: {}".format(i + 1, epoch_train_loss, epoch_val_loss))

            if epoch_val_loss < best_loss:
                early_stop_count = 0
                best_loss = epoch_val_loss

                early_stop_count += 1
                if early_stop_count >= 10 and early_stop:
                    break

        if save_model:
            model_name = "./output/lstm_model_{}.pth".format(int(datetime.now().timestamp()))
            torch.save(model.state_dict(), model_name)

        return model

    def rolling_predict(self, data, model, look_back, pred_preiods, scaler):
        model.eval()
        pred_data = torch.tensor(data).float()
        pred_data = pred_data[-1 * look_back:, :]
        preds = []
        for i in range(pred_preiods):
            with torch.no_grad():
                pred_x = pred_data.reshape((1, look_back, -1))
                model.hidden_cell = (torch.zeros(1, 1, model.hidden_layers),
                                     torch.zeros(1, 1, model.hidden_layers))
                pred_y = model(pred_x)
                preds.append(pred_y.numpy())
                temp_data = torch.cat((pred_data, pred_y))
                pred_data = temp_data[-1 * look_back:]

        actual_preds = scaler.inverse_transform(np.array(preds).reshape(-1, pred_y.shape[1]))
        return actual_preds

    def predict(self, ts_df, train_ratio=0.8, pred_periods=1, **kwargs):
        ts = ts_df.values
        input_data = np.array(ts)
        feature_nums = input_data.shape[1]
        scaler = MinMaxScaler(feature_range=(-1, 1))
        input_data = scaler.fit_transform(input_data.reshape(-1, feature_nums))

        train_len = int(len(input_data) * train_ratio)

        input_data_train = input_data[:train_len]
        input_data_val = input_data[train_len:]

        learning_rate = kwargs.get('learning_rate') or 0.001
        epochs = kwargs.get('epochs') or 20
        look_back = kwargs.get('look_back') or 20

        # Training & Val data
        train_data_x, train_data_y = self.create_lstm_dataset(input_data_train, look_back)
        val_data_x, val_data_y = self.create_lstm_dataset(input_data_val, look_back)
        train_set = LSTMDataSet(train_data_x, train_data_y)
        val_set = LSTMDataSet(val_data_x, val_data_y)
        pred_data_train = DataLoader(train_set, batch_size=1, shuffle=False)
        pred_data_val = DataLoader(val_set, batch_size=1, shuffle=False)

        # Initial Model, Loss & Optimizer
        model = LSTMModel(input_size=feature_nums, hidden_layers=24, output_size=feature_nums)
        criterion = nn.MSELoss()
        optimizer = SGD(model.parameters(), lr=learning_rate, momentum=0.9)

        # Training
        trained_model = self.lstm_train(pred_data_train, pred_data_val, model, criterion, optimizer, epochs=epochs)

        preds = self.rolling_predict(input_data, trained_model, look_back, pred_periods, scaler)
        preds = preds[:, :feature_nums]

        return preds


if __name__ == '__main__':
    df = pd.read_excel('../dataset/上证50ETF.xlsx')
    df = df.drop(['代码', '名称', '日期'], axis=1)
    print(df)

    c = LSTMController()
    results = c.predict(df)

    print(results)





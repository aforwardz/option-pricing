import os
import torch
import torch.nn as nn
import math
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
DATASET_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'dataset')
MODELING_DIR = os.path.join(DATASET_DIR, 'modeling')


def split_dataset_into_seq(dataset, start_index=0, end_index=None, window_size=6, step=1):
    '''split the dataset to have sequence of observations of length history size'''
    data = []
    start_index = start_index + window_size
    if end_index is None:
        end_index = len(dataset)
    for i in range(start_index, end_index):
        indices = range(i - window_size, i, step)
        data.append(dataset[indices])
    return np.array(data)


def split_dataset(data_seq, TRAIN_SPLIT=0.7, VAL_SPLIT=0.5, save_path=None):
    '''split the dataset into train, val and test splits'''

    # split between validation dataset and test set:
    train_data, val_data = train_test_split(data_seq, train_size=TRAIN_SPLIT, shuffle=True, random_state=123)
    val_data, test_data = train_test_split(val_data, train_size=VAL_SPLIT, shuffle=True, random_state=123)

    return np.float32(train_data), np.float32(val_data), np.float32(test_data)


def split_fn(chunk):
    """to split the dataset sequences into input and targets sequences"""
    inputs = torch.tensor(chunk[:, :-1, :], device=device)
    targets = torch.tensor(chunk[:, 1:, :], device=device)
    return inputs, targets


def data_to_dataset(train_data, val_data, test_data, batch_size=32, target_features=[-1]):
    '''
    split each train split into inputs and targets
    convert each train split into a tf.dataset
    '''
    x_train, y_train = split_fn(train_data)
    x_val, y_val = split_fn(val_data)
    x_test, y_test = split_fn(test_data)
    # selecting only the first 20 features for prediction:
    y_train = y_train[:, :, target_features]
    y_val = y_val[:, :, target_features]
    y_test = y_test[:, :, target_features]
    train_dataset = torch.utils.data.TensorDataset(x_train, y_train)
    val_dataset = torch.utils.data.TensorDataset(x_val, y_val)
    test_dataset = torch.utils.data.TensorDataset(x_test, y_test)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size)
    return train_loader, val_loader, test_loader


class MultiHeadAttention(nn.Module):
    '''Multi-head self-attention module'''

    def __init__(self, D, H):
        super(MultiHeadAttention, self).__init__()
        self.H = H  # number of heads
        self.D = D  # dimension

        self.wq = nn.Linear(D, D * H)
        self.wk = nn.Linear(D, D * H)
        self.wv = nn.Linear(D, D * H)

        self.dense = nn.Linear(D * H, D)

    def concat_heads(self, x):
        '''(B, H, S, D) => (B, S, D*H)'''
        B, H, S, D = x.shape
        x = x.permute((0, 2, 1, 3)).contiguous()  # (B, S, H, D)
        x = x.reshape((B, S, H * D))  # (B, S, D*H)
        return x

    def split_heads(self, x):
        '''(B, S, D*H) => (B, H, S, D)'''
        B, S, D_H = x.shape
        x = x.reshape(B, S, self.H, self.D)  # (B, S, H, D)
        x = x.permute((0, 2, 1, 3))  # (B, H, S, D)
        return x

    def forward(self, x, mask):
        q = self.wq(x)  # (B, S, D*H)
        k = self.wk(x)  # (B, S, D*H)
        v = self.wv(x)  # (B, S, D*H)

        q = self.split_heads(q)  # (B, H, S, D)
        k = self.split_heads(k)  # (B, H, S, D)
        v = self.split_heads(v)  # (B, H, S, D)

        attention_scores = torch.matmul(q, k.transpose(-1, -2))  # (B,H,S,S)
        attention_scores = attention_scores / math.sqrt(self.D)

        # add the mask to the scaled tensor.
        if mask is not None:
            attention_scores += (mask * -1e9)

        attention_weights = nn.Softmax(dim=-1)(attention_scores)
        scaled_attention = torch.matmul(attention_weights, v)  # (B, H, S, D)
        concat_attention = self.concat_heads(scaled_attention)  # (B, S, D*H)
        output = self.dense(concat_attention)  # (B, S, D)

        return output, attention_weights


# Positional encodings
def get_angles(pos, i, D):
    angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(D))
    return pos * angle_rates


def positional_encoding(D, position=20, dim=3, device=device):
    angle_rads = get_angles(np.arange(position)[:, np.newaxis],
                            np.arange(D)[np.newaxis, :],
                            D)
    # apply sin to even indices in the array; 2i
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
    # apply cos to odd indices in the array; 2i+1
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
    if dim == 3:
        pos_encoding = angle_rads[np.newaxis, ...]
    elif dim == 4:
        pos_encoding = angle_rads[np.newaxis,np.newaxis,  ...]
    return torch.tensor(pos_encoding, device=device)


# function that implement the look_ahead mask for masking future time steps.
def create_look_ahead_mask(size, device=device):
    mask = torch.ones((size, size), device=device)
    mask = torch.triu(mask, diagonal=1)
    return mask  # (size, size)


class TransformerLayer(nn.Module):
    def __init__(self, D, H, hidden_mlp_dim, dropout_rate):
        super(TransformerLayer, self).__init__()
        self.dropout_rate = dropout_rate
        self.mlp_hidden = nn.Linear(D, hidden_mlp_dim)
        self.mlp_out = nn.Linear(hidden_mlp_dim, D)
        self.layernorm1 = nn.LayerNorm(D, eps=1e-9)
        self.layernorm2 = nn.LayerNorm(D, eps=1e-9)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.dropout2 = nn.Dropout(dropout_rate)

        self.mha = MultiHeadAttention(D, H)

    def forward(self, x, look_ahead_mask):
        attn, attn_weights = self.mha(x, look_ahead_mask)  # (B, S, D)
        attn = self.dropout1(attn)  # (B,S,D)
        attn = self.layernorm1(attn + x)  # (B,S,D)

        mlp_act = torch.relu(self.mlp_hidden(attn))
        mlp_act = self.mlp_out(mlp_act)
        mlp_act = self.dropout2(mlp_act)

        output = self.layernorm2(mlp_act + attn)  # (B, S, D)

        return output, attn_weights


class Transformer(nn.Module):
    '''Transformer Decoder Implementating several Decoder Layers.
    '''

    def __init__(self, num_layers, D, H, hidden_mlp_dim, inp_features, out_features, dropout_rate):
        super(Transformer, self).__init__()
        self.sqrt_D = torch.tensor(math.sqrt(D))
        self.num_layers = num_layers
        self.input_projection = nn.Linear(inp_features, D)  # multivariate input
        self.output_projection = nn.Linear(D, out_features)  # multivariate output
        self.pos_encoding = positional_encoding(D)
        self.dec_layers = nn.ModuleList([TransformerLayer(D, H, hidden_mlp_dim,
                                                          dropout_rate=dropout_rate
                                                          ) for _ in range(num_layers)])
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x, mask):
        B, S, D = x.shape
        attention_weights = {}
        x = self.input_projection(x)
        x *= self.sqrt_D

        x += self.pos_encoding[:, :S, :]

        x = self.dropout(x)

        for i in range(self.num_layers):
            x, block = self.dec_layers[i](x=x, look_ahead_mask=mask)
            attention_weights['decoder_layer{}'.format(i + 1)] = block

        x = self.output_projection(x)

        return x, attention_weights  # (B,S,S)


def train_transformer(model, train_dataset, val_dataset):
    n_epochs = 20
    niter = len(train_dataset)
    losses, val_losses = [], []

    for e in tqdm(range(n_epochs)):

        # one epoch on train set
        model.train()
        sum_train_loss = 0.0
        for x, y in train_dataset:
            S = x.shape[1]
            mask = create_look_ahead_mask(S)
            out, _ = model(x, mask)
            loss = torch.nn.MSELoss()(out, y)
            sum_train_loss += loss.item()
            loss.backward()
            optimizer.step()
        losses.append(sum_train_loss / niter)

        # Evaluate on val set
        model.eval()
        sum_val_loss = 0.0
        for i, (x, y) in enumerate(val_dataset):
            S = x.shape[1]
            mask = create_look_ahead_mask(S)
            out, _ = model(x, mask)
            loss = torch.nn.MSELoss()(out, y)
            sum_val_loss += loss.item()
        val_losses.append(sum_val_loss / (i + 1))

    plt.plot(losses)
    plt.plot(val_losses)

    return model


def test_transformer(model, test_dataset):
    test_losses, test_preds = [], []
    model.eval()
    for (x, y) in test_dataset:
        S = x.shape[-2]
        y_pred, _ = model(x, mask=create_look_ahead_mask(S))
        loss_test = torch.nn.MSELoss()(y_pred, y)  # (B,S)
        test_losses.append(loss_test.item())
        test_preds.append(y_pred.detach().cpu().numpy())
    test_preds = np.vstack(test_preds)
    np.mean(test_losses)

    seq_len = 5
    index = 1
    feature_num = 0

    x_test, _ = test_dataset.dataset.tensors
    x_test = x_test[index, :, feature_num].cpu().numpy()
    pred = test_preds[index, :, feature_num]
    x = np.linspace(1, seq_len, seq_len)
    plt.plot(x, pred, 'red', lw=2, label='predictions for sample: {}'.format(index))
    plt.plot(x, x_test, 'cyan', lw=2, label='ground-truth for sample: {}'.format(index))
    plt.legend(fontsize=10)
    plt.show()


if __name__ == '__main__':
    labels = ['行权价', '涨跌幅', '成交额', '前结算价', '开盘价', '最高价', '最低价',
              '结算价', '成交量', '持仓量', '涨停价格', '跌停价格', 'Delta', 'Gamma',
              'Vega', 'Theta', 'Rho', 'ETF收盘价', '收盘价']

    data = np.load(os.path.join(MODELING_DIR, 'sz50etf_modeling.npy'))
    print(data.dtype)

    train_data, val_data, test_data = split_dataset(data)
    train_dataset, val_dataset, test_dataset = data_to_dataset(train_data, val_data, test_data, target_features=[-1])
    print(train_data.shape, val_data.shape, test_data.shape)
    print(len(train_dataset), len(val_dataset), len(test_dataset))

    transformer = Transformer(num_layers=1, D=32, H=4, hidden_mlp_dim=32,
                              inp_features=len(labels), out_features=1, dropout_rate=0.1).to(device)
    optimizer = torch.optim.RMSprop(transformer.parameters(), lr=0.00005)

    transformer = train_transformer(transformer, train_dataset, val_dataset)

    test_transformer(transformer, test_dataset)


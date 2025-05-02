import glob
import json
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import yfinance as yf
from torch.utils.data import Dataset, DataLoader

# ----------------------------------------
#         Temporal Convolutional Network
# ----------------------------------------
class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size
    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()

class TemporalBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size,
                               stride=stride, padding=padding, dilation=dilation)
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size,
                               stride=stride, padding=padding, dilation=dilation)
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)
        self.net = nn.Sequential(
            self.conv1, self.chomp1, self.relu1, self.dropout1,
            self.conv2, self.chomp2, self.relu2, self.dropout2
        )
        self.downsample = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)

class TCN(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        super(TCN, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_ch = num_inputs if i == 0 else num_channels[i-1]
            out_ch = num_channels[i]
            layers.append(
                TemporalBlock(
                    in_ch, out_ch, kernel_size, stride=1,
                    dilation=dilation_size,
                    padding=(kernel_size-1) * dilation_size,
                    dropout=dropout
                )
            )
        self.network = nn.Sequential(*layers)
        self.linear = nn.Linear(num_channels[-1], 1)

    def forward(self, x):
        x = x.transpose(1, 2)   # (batch, features, seq_len)
        y = self.network(x)     # (batch, channels, seq_len)
        out = y[:, :, -1]       # last time step
        return self.linear(out)

# ----------------------------------------
#            Dataset & Loader
# ----------------------------------------
class StockDataset(Dataset):
    def __init__(self, series, seq_len=30):
        self.seq_len = seq_len
        data = torch.tensor(series, dtype=torch.float32).unsqueeze(-1)
        self.X, self.y = self._create_sequences(data)

    def _create_sequences(self, data):
        xs, ys = [], []
        for i in range(len(data) - self.seq_len):
            xs.append(data[i:i+self.seq_len])
            ys.append(data[i+self.seq_len])
        return torch.stack(xs), torch.stack(ys)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# ----------------------------------------
#         Training & Evaluation
# ----------------------------------------
def evaluate(model, loader, criterion, device):
    model.eval()
    total = 0.0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            pred = model(x)
            total += criterion(pred.squeeze(), y.squeeze()).item()
    return total / len(loader)


def train(model, train_loader, val_loader, epochs=20, lr=1e-3, device='cpu'):
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    for ep in range(1, epochs+1):
        model.train()
        train_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            pred = model(x)
            loss = criterion(pred.squeeze(), y.squeeze())
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        val_loss = evaluate(model, val_loader, criterion, device)
        print(f"Epoch {ep}: Train Loss={train_loss/len(train_loader):.4f}  Val Loss={val_loss:.4f}")

# ----------------------------------------
#               Main Script
# ----------------------------------------
if __name__ == '__main__':
    # Read companies & dates from JSON
    with open('Group 1/companies.json', 'r') as f:
        cfg = json.load(f)
    symbols = cfg['companies']
    starts = cfg['start_date']
    ends = cfg['end_date']

    # Hyperparameters
    seq_len = 30
    batch_size = 16
    channels = [16, 32, 64]
    kernel = 3
    dropout = 0.1
    epochs = 20
    lr = 1e-3

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    for idx in range(len(starts)):
        start_date = starts[idx]
        end_date = ends[idx]

        for sym in symbols:
            print(f"\n--- {sym} ({start_date} to {end_date}) ---")
            # Download data
            df = yf.download(sym, start=start_date, end=end_date)
            if df.empty:
                print(f"No data for {sym}.")
                continue

            # Save formatted CSV
            out_csv = f"Group 1/Example_output/{sym}_{start_date}_{end_date}.csv"
            tmp = df.copy()
            tmp.reset_index(inplace=True)
            tmp.to_csv(out_csv, index=False)
            print(f"Saved CSV: {out_csv}")

            # Prepare series
            series = tmp['Close'].values
            mean, std = series.mean(), series.std()
            norm = (series - mean) / std

            # Split
            split = int(len(norm)*0.8)
            train_s = norm[:split]
            val_s = norm[split:]

            # Datasets & loaders
            train_ds = StockDataset(train_s, seq_len)
            val_ds = StockDataset(val_s, seq_len)
            train_ld = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
            val_ld = DataLoader(val_ds, batch_size=batch_size)

            # Model
            model = TCN(num_inputs=1, num_channels=channels, kernel_size=kernel, dropout=dropout)

            # Train
            train(model, train_ld, val_ld, epochs=epochs, lr=lr, device=device)

            # Forecast next week
            model.eval()
            inp = torch.tensor(norm[-seq_len:], dtype=torch.float32).unsqueeze(0).unsqueeze(-1).to(device)
            future_norm = []
            with torch.no_grad():
                for _ in range(7):
                    p = model(inp).item()
                    future_norm.append(p)
                    inp = torch.cat([inp[:,1:,:], torch.tensor(p).view(1,1,1).to(device)], dim=1)
            future = np.array(future_norm)*std + mean
            print(f"7-day forecast for {sym}: {future}\n")

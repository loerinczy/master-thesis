import torch
from torch import nn


class RetLSTM(nn.Module):
    def __init__(self, a_scan_length, num_classes, kernel_size, hidden_size):
        super(RetLSTM, self).__init__()
        self.a_scan_length = a_scan_length
        self.num_classes = num_classes
        self.conv = nn.Conv1d(
                  in_channels=2,
                  out_channels=1,
                  kernel_size=kernel_size
        )
        self.lstm = nn.LSTM(
                  input_size=a_scan_length-kernel_size+1,
                  hidden_size=hidden_size
        )
        self.fc = nn.Linear(
                  in_features=hidden_size,
                  out_features=num_classes
        )

    def forward(self, seq):
        coord = torch.tile(
                  torch.arange(self.a_scan_length, dtype=torch.float32),
                  dims=(*seq.shape[:2], 1)
        ).unsqueeze(2)
        seq = torch.cat([seq.unsqueeze(2), coord], dim=2)
        seq = torch.cat([self.conv(x_in).unsqueeze(0) for x_in in seq], dim=0)
        out, _ = self.lstm(seq.squeeze(2))
        prediction = torch.cat([self.fc(x_out).unsqueeze(0) for x_out in out], dim=0)
        return prediction


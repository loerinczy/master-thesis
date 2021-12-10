import torch
from torch import nn


class RetLSTM(nn.Module):
    def __init__(self, conv, fc, a_scan_length, num_classes, hidden_size, device):
        super(RetLSTM, self).__init__()
        self.a_scan_length = a_scan_length
        self.num_classes = num_classes
        self.conv = conv.to(device)
        self.lstm = nn.LSTM(
                  input_size=hidden_size,
                  hidden_size=hidden_size
        ).to(device)
        self.fc = fc.to(device)
        self.device = device

    def forward(self, seq):
        coord = torch.tile(
                  torch.arange(self.a_scan_length, dtype=torch.float32, device=self.device),
                  dims=(*seq.shape[:2], 1)
        ).unsqueeze(2)
        coord = (coord - self.a_scan_length / 2) / (self.a_scan_length / 2)
        seq = torch.cat([seq.unsqueeze(2), coord], dim=2)
        seq = torch.cat([self.conv(x_in).unsqueeze(0) for x_in in seq], dim=0)
        out, _ = self.lstm(seq.squeeze(2))
        prediction = torch.cat([self.fc(x_out).unsqueeze(0) for x_out in out], dim=0)
        return prediction


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


class OnlyConvModel(nn.Module):

  def __init__(self, conv, a_scan_length=496):
    super(OnlyConvModel, self).__init__()
    self.a_scan_length = a_scan_length
    self.conv = conv
    coord = torch.arange(self.a_scan_length)
    coord = (coord - self.a_scan_length / 2) / (self.a_scan_length / 2)
    self.register_buffer("coord", coord)

  def forward(self, seq):
      coord = torch.tile(self.coord, dims=(*seq.shape[:2], 1)).unsqueeze(2)
      seq = torch.cat([seq.unsqueeze(2), coord], dim=2)
      out = torch.cat([self.conv(x_in).unsqueeze(0) for x_in in seq], dim=0)
      return out


def get_ConvModule(n_layers, kernel_size, stride, out_features, a_scan_length=496):
  ml = []
  n_channels = 2**(n_layers)
  in_features = a_scan_length
  ml.append(
    nn.Conv1d(
        in_channels=2,
        out_channels=2,
        kernel_size=1,
    ))
  ml.append(nn.ReLU(True))
  for layer_idx in range(n_layers - 1):
    ml.append(
      nn.Conv1d(
        in_channels=2**(layer_idx+1),
        out_channels=2**(layer_idx + 2),
        kernel_size=kernel_size,
        stride=2
    ))
    ml.append(nn.ReLU(True))
    in_features = (in_features - kernel_size) // stride + 1
  ml.append(nn.Flatten(start_dim=1))
  ml.append(nn.Linear(n_channels * in_features, out_features))
  return nn.Sequential(*ml)

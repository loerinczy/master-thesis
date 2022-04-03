import torch
from torch import nn
from torch.nn import functional as F


class RetLSTM(nn.Module):
    def __init__(self, conv, fc, hidden_size, cc=False, a_scan_length=496):
        super(RetLSTM, self).__init__()
        self.a_scan_length = a_scan_length
        self.conv = conv
        self.lstm = nn.LSTM(
                  input_size=hidden_size,
                  hidden_size=hidden_size
        )
        self.fc = fc
        coord = torch.arange(self.a_scan_length)
        coord = (coord - self.a_scan_length / 2) / (self.a_scan_length / 2)
        self.register_buffer("coord", coord)
        self.cc = cc

    def forward(self, seq):
        if self.cc:
            coord = torch.tile(self.coord, dims=(*seq.shape[:2], 1)).unsqueeze(2)
            seq = torch.cat([seq.unsqueeze(2), coord], dim=2)
        else:
            seq = seq.unsqueeze(2)
        seq = torch.cat([self.conv(x_in).unsqueeze(0) for x_in in seq], dim=0)
        out, _ = self.lstm(seq.squeeze(2))
        prediction = torch.cat([self.fc(x_out).unsqueeze(0) for x_out in out], dim=0)
        return prediction


class SimpleModelCC(nn.Module):

  def __init__(self, conv, a_scan_length=496):
    super(SimpleModelCC, self).__init__()
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


class PatchModelPosEnc(nn.Module):

    def __init__(
            self, conv, neighbors=0, pos_enc="cc", scale="norm", wlen=2, a_scan_length=496
    ):
        super(PatchModelPosEnc, self).__init__()
        self.a_scan_length = a_scan_length
        self.conv = conv
        if pos_enc == "cc":
            coord = torch.arange(self.a_scan_length)
            if scale == "norm":
                coord = (coord - self.a_scan_length / 2) / (self.a_scan_length / 2)
            else:
                coord /= self.a_scan_length
        else:
            coord = torch.sin(torch.arange(self.a_scan_length) / wlen * 2 * torch.pi)
        self.register_buffer("coord", coord)
        self.neighbors = neighbors

  def forward(self, seq):
      seq = torch.cat([
          seq.unsqueeze(1),
          torch.tile(self.coord, (seq.shape[0], 1, seq.shape[1], 1)),
      ], dim=1)
      seq = F.pad(seq, (0, 0, 0, 0, self.neighbors, self.neighbors), "constant", 0)
      out = torch.cat([
               self.conv(seq[i:i+1+2*self.neighbors].swapaxes(0, 2)).unsqueeze(0)
               for i in range(seq.shape[0] - 2 * self.neighbors)
               ], dim=0
      )
      return out


class PatchModel(nn.Module):

    def __init__(self, conv, neighbors=0, a_scan_length=496):
            super(PatchModel, self).__init__()
            self.a_scan_length = a_scan_length
            self.conv = conv
            self.neighbors = neighbors

    def forward(self, seq):
        seq = F.pad(seq, (0, 0, 0, 0, self.neighbors, self.neighbors), "constant", 0)
        out = torch.cat([
               self.conv(seq[i:i+1+2*self.neighbors].swapaxes(0, 1).unsqueeze(1)).unsqueeze(0)
               for i in range(seq.shape[0] - 2 * self.neighbors)
               ], dim=0
        )
        return out


def get_ConvModulePatch(n_layers, kernel_size, stride, neighbors, a_scan_length=496):
    ml = []
    n_channels = 2**n_layers
    in_features = (a_scan_length - kernel_size) // stride + 1
    ml.append(
        nn.Conv2d(
            in_channels=1,
            out_channels=2,
            kernel_size=(2 * neighbors + 1, kernel_size),
            stride=(1, stride)
    ))
    ml.append(nn.ReLU(True))
    ml.append(nn.Flatten(1, 2))
    for layer_idx in range(n_layers - 1):
        ml.append(
          nn.Conv1d(
            in_channels=2**(layer_idx+1),
            out_channels=2**(layer_idx + 2),
            kernel_size=kernel_size,
            stride=stride
        ))
        ml.append(nn.ReLU(True))
        in_features = (in_features - kernel_size) // stride + 1
    ml.append(nn.Flatten(start_dim=1))
    return nn.Sequential(*ml), n_channels * in_features


def get_ConvModuleCC(n_layers, kernel_size, stride, a_scan_length=496):
    ml = []
    n_channels = 2**(n_layers)
    in_features = (a_scan_length - kernel_size) // stride + 1
    ml.append(
    nn.Conv1d(
        in_channels=2,
        out_channels=2,
        kernel_size=kernel_size,
        stride=stride
    ))
    ml.append(nn.ReLU(True))
    for layer_idx in range(n_layers - 1):
        ml.append(
          nn.Conv1d(
            in_channels=2**(layer_idx+1),
            out_channels=2**(layer_idx + 2),
            kernel_size=kernel_size,
            stride=stride
        ))
        ml.append(nn.ReLU(True))
        in_features = (in_features - kernel_size) // stride + 1
    ml.append(nn.Flatten(start_dim=1))
    return nn.Sequential(*ml), n_channels * in_features


class SimpleModelNCC(nn.Module):

    def __init__(self, conv, a_scan_length=496):
        super(SimpleModelNCC, self).__init__()
        self.a_scan_length = a_scan_length
        self.conv = conv

    def forward(self, seq):
      seq = seq.unsqueeze(2)
      out = torch.cat([self.conv(x_in).unsqueeze(0) for x_in in seq], dim=0)
      return out


def get_ConvModuleNCC(n_layers, kernel_size, stride, a_scan_length=496):
    ml = []
    n_channels = 2**(n_layers)
    in_features = a_scan_length
    for layer_idx in range(n_layers):
        ml.append(
          nn.Conv1d(
            in_channels=2**(layer_idx),
            out_channels=2**(layer_idx + 1),
            kernel_size=kernel_size,
            stride=stride
        ))
        ml.append(nn.ReLU(True))
        in_features = (in_features - kernel_size) // stride + 1
    ml.append(nn.Flatten(start_dim=1))
    return nn.Sequential(*ml), n_channels * in_features
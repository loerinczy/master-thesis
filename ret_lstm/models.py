import torch
from torch import nn
from torch.nn import functional as F


class RetLSTM(nn.Module):
    def __init__(
              self,
              patchmodel,
              fc,
              hidden_size,
              forget_bias=None,
              inp_dropout_p=0,
              out_dropout_p=0
    ):
        super(RetLSTM, self).__init__()
        self.patchmodel = patchmodel
        self.lstm = nn.LSTM(
                  input_size=hidden_size,
                  hidden_size=hidden_size
        )
        self.fc = fc
        # self.init_dropout(inp_dropout_p, out_dropout_p)
        if forget_bias is not None:
            self.set_forget_bias(forget_bias)

    # def init_dropout(self, idp, odp):
    #     self.inp_dropout = nn.Dropout(idp)
    #     self.out_dropout = nn.Dropout(odp)

    def set_forget_bias(self, forget_bias):
        for name, param in self.named_parameters():
            if name in ("bias_hh_l0", "bias_ih_l0"):
                param[8:16] = forget_bias


    def forward(self, seq):
        seq = self.patchmodel(seq)
        # seq = self.inp_dropout(seq)
        out, _ = self.lstm(seq.squeeze(2))
        # out = self.out_dropout(out)
        pred = self.fc(out.swapaxes(0, 1)).swapaxes(0, 1)
        # pred = torch.cat([self.fc(x_out).unsqueeze(0) for x_out in out], dim=0)
        return pred


class PatchModelPosEnc(nn.Module):

    def __init__(
            self,
            conv,
            fc,
            neighbors=0,
            pos_enc="cc",
            cc_type="center",
            wlens=2,
            sum_coord=False,
            scale=1.,
            a_scan_length=496
    ):
        super(PatchModelPosEnc, self).__init__()
        self.conv = conv
        if pos_enc == "cc":
            coord = torch.arange(a_scan_length)
            if cc_type == "center":
                coord = (coord - a_scan_length / 2) / (a_scan_length / 2)
            elif cc_type == "minmax":
                coord = coord / a_scan_length
            else:
                coord = coord / a_scan_length
                coord = (coord - coord.mean()) / coord.std()
        else:
            wlens = torch.tensor([wlens]) if type(wlens) != list else torch.tensor(wlens)
            wlens.unsqueeze_(1)
            coord = torch.tile(torch.arange(a_scan_length), (len(wlens), 1))
            coord = torch.sin(coord / wlens * 2 * torch.pi).unsqueeze(1)
        self.fc = fc
        self.register_buffer("coord", coord)
        self.neighbors = neighbors
        self.sum_coord = sum_coord
        self.scale = scale

    def forward(self, seq):
        if self.sum_coord:
            seq = (
                seq.unsqueeze(1)
                + self.scale * torch.tile(self.coord, (seq.shape[0], 1, seq.shape[1], 1))
            )
        else:
            seq = torch.cat([
              seq.unsqueeze(1),
              self.scale * torch.tile(self.coord, (seq.shape[0], 1, seq.shape[1], 1)),], dim=1
            )
        seq = F.pad(seq, (0, 0, 0, 0, 0, 0, self.neighbors, self.neighbors), "constant", 0)
        out = torch.cat([
               self.conv(seq[i:i+1+2*self.neighbors].swapaxes(0, 2)).unsqueeze(0)
               for i in range(seq.shape[0] - 2 * self.neighbors)
               ], dim=0
        )
        out = self.fc(out.swapaxes(0, 1)).swapaxes(0, 1)
        return out


class PatchModelPosEnc2d(nn.Module):

    def __init__(
            self,
            conv,
            fc,
            neighbors=0,
            pos_enc="cc",
            cc_type="center",
            wlens=2,
            sum_coord="none",
            scale=1.,
            a_scan_length=496
    ):
        super(PatchModelPosEnc2d, self).__init__()
        self.conv = conv
        if pos_enc == "cc":
            coord = torch.arange(a_scan_length)
            if cc_type == "center":
                coord = (coord - a_scan_length / 2) / (a_scan_length / 2)
            elif cc_type == "minmax":
                coord = coord / a_scan_length
            else:
                coord = coord / a_scan_length
                coord = (coord - coord.mean()) / coord.std()
        else:
            wlens = torch.tensor([wlens]) if type(wlens) != list else torch.tensor(wlens)
            wlens.unsqueeze_(1)
            coord = torch.tile(torch.arange(a_scan_length), (len(wlens), 1))
            coord = torch.sin(coord / wlens * 2 * torch.pi).unsqueeze(1)
        self.fc = fc
        self.register_buffer("coord", coord)
        self.neighbors = neighbors
        self.sum_coord = sum_coord
        self.scale = scale

    def forward(self, seq):
        if self.sum_coord == "sum":
            seq = (
                seq.unsqueeze(1)
                + self.scale * torch.tile(self.coord, (seq.shape[0], 1, seq.shape[1], 1))
            )
        elif self.sum_coord == "mul":
            seq = (
                      seq.unsqueeze(1)
                      * self.scale * torch.tile(
                self.coord, (seq.shape[0], 1, seq.shape[1], 1)
                )
            )
        else:
            seq = torch.cat([
              seq.unsqueeze(1),
              self.scale * torch.tile(self.coord, (seq.shape[0], 1, seq.shape[1], 1)),], dim=1
            )
        out = self.conv(seq.swapaxes(0, 2))
        out = self.fc(out.swapaxes(1, 2).flatten(2)).swapaxes(0, 1)
        return out


class PatchModel(nn.Module):

    def __init__(
              self,
              conv,
              fc,
              coord_length=288,
              neighbors=0,
              pos_enc="none",
              pos_op="none",
              cc_type="center",
              scale=0.1,
              wlen=2,
              a_scan_length=496
    ):
        super(PatchModel, self).__init__()
        self.a_scan_length = a_scan_length
        self.conv = conv
        self.fc = fc
        self.neighbors = neighbors
        if pos_enc == "cc":
            if cc_type == "center":
                coord = (torch.arange(coord_length) - coord_length // 2) / (
                              coord_length // 2)
                coord *= scale
            else:
                coord = torch.arange(coord_length) / (coord_length - 1) * scale
            self.register_buffer("coord", coord)
        elif pos_enc == "sin":
            coord = torch.sin(torch.arange(coord_length) / wlen * 2 * torch.pi)
            self.register_buffer("coord", coord)
        # self.pos_enc = pos_op
        self.pos_enc = pos_enc


    def forward(self, seq):
        seq = F.pad(seq, (0, 0, 0, 0, self.neighbors, self.neighbors), "constant", 0)
        out = torch.cat(
                  [
                      self.conv(
                          seq[i:i + 1 + 2 * self.neighbors].swapaxes(0, 1).unsqueeze(1)
                          ).unsqueeze(0)
                      for i in range(seq.shape[0] - 2 * self.neighbors)
                  ], dim=0
        )
        if self.pos_enc != "none":
            out = out + self.coord
        # if self.pos_enc == "add":
        #     out = out + self.coord
        # elif self.pos_enc == "mul":
        #     out = out * self.coord
        out = self.fc(out.swapaxes(0, 1)).swapaxes(0, 1)
        return out


class PatchModel2d(nn.Module):

    def __init__(
              self,
              conv,
              fc,
              a_scan_length=496
    ):
        super(PatchModel2d, self).__init__()
        self.a_scan_length = a_scan_length
        self.conv = conv
        self.fc = fc

    def forward(self, seq):
        out = self.conv(seq.swapaxes(0, 1).unsqueeze(1))
        out = self.fc(out.swapaxes(1, 2).flatten(2)).swapaxes(0, 1)
        return out


def get_ConvModulePatch(n_layers, kernel_size, stride, input_channels, neighbors, a_scan_length=496):
    ml = []
    n_channels = 2**n_layers
    in_features = (a_scan_length - kernel_size) // stride + 1
    ml.append(
        nn.Conv2d(
            in_channels=input_channels,
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


def get_ConvModule(n_layers, kernel_size, stride, input_channels, a_scan_length=496):
    ml = []
    n_channels = 2**n_layers
    in_features = (a_scan_length - kernel_size) // stride + 1
    ml.append(
    nn.Conv1d(
        in_channels=input_channels,
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


class SkipConnConv(nn.Module):
    def __init__(self, conv, skip):
        super(SkipConnConv, self).__init__()
        self.conv = conv
        self.skip = skip

    def forward(self, x):
        main = self.conv(x)
        skip = self.skip(x)
        return main + skip


class Block2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel, kernel_skip, kp=2):
        super(Block2d, self).__init__()
        self.main = nn.Sequential(
                  nn.Conv2d(
                            in_channels,
                            out_channels,
                            kernel,
                            padding="same"
                  ),
                  nn.ReLU(True),
                  nn.Conv2d(
                            out_channels,
                            out_channels,
                            kernel,
                            padding="same"
                  ),
                  nn.ReLU(True)
        )
        self.skip = nn.Sequential(
                  nn.Conv2d(in_channels, out_channels, kernel_skip, padding="same"),
                  nn.ReLU(True)
        )
        self.model = nn.Sequential(
                  SkipConnConv(self.main, self.skip),
                  nn.MaxPool2d(kp)
        )

    def forward(self, x):
        return self.model(x)


class Block1d(nn.Module):
    def __init__(self, in_channels, out_channels, kx, ks):
        super(Block1d, self).__init__()
        self.kx = kx
        self.ks = ks
        self.main = nn.Sequential(
                  nn.Conv1d(
                            in_channels,
                            out_channels,
                            kx,
                            padding="same"
                  ),
                  nn.ReLU(True),
                  nn.Conv1d(
                            out_channels,
                            out_channels,
                            kx,
                            padding="same"
                  ),
                  nn.ReLU(True)
        )
        self.skip = nn.Sequential(
                  nn.Conv1d(in_channels, out_channels, ks, padding="same"),
                  nn.ReLU(True)
        )
        self.model = nn.Sequential(
                  SkipConnConv(self.main, self.skip),
                  nn.MaxPool1d(2)
        )

    def forward(self, x):
        return self.model(x)


class RetinaConv(nn.Module):
    def __init__(self, patchmodel, conv, fc):
        super(RetinaConv, self).__init__()
        self.patchmodel = patchmodel
        self.conv = conv
        self.fc = fc

    def forward(self, seq):
        seq = self.patchmodel(seq)
        seq = seq.swapaxes(0, -1)
        seq = torch.cat([self.conv(x.unsqueeze(1)).unsqueeze(0) for x in seq], dim=0)
        seq = seq.permute((1, -1, 0))
        out = self.fc(seq).swapaxes(0, 1)
        return out


class RetinaConv2d(nn.Module):
    def __init__(self, patchmodel, conv, fc1, fc2):
        super(RetinaConv2d, self).__init__()
        self.patchmodel = patchmodel
        self.conv = conv
        self.fc1 = fc1
        self.fc2 = fc2

    def forward(self, seq):
        seq = self.patchmodel(seq)
        seq = self.conv(seq.permute(1, 2, 0).unsqueeze(1))
        out = self.fc1(seq.swapaxes(1, 2).flatten(2)).permute(2, 0, 1)
        out = self.fc2(out.swapaxes(0, 1)).swapaxes(0, 1)
        return out

import torch
from PIL import Image

def get_layer_channels(data: torch.Tensor, num_classes):
    """
    Creates the channels for each layer.
    :param data: torch.Tensor of shape N x H x W
    :return: torch.Tensor of shape N x 4 x H x W
    """

    data = data.unsqueeze(1)
    zeros = torch.zeros(data.shape[0], num_classes, *data.shape[2:])
    zeros.scatter_(1, data, 1)
    return zeros


def show_layers(data: torch.Tensor):
    """
    Shows the layers in different colors.
    :param data: torch.Tensor of shape [1 x C x] H x W
    :return: None
    """

    assert not (len(data.shape) == 4 and data.shape[0] != 1), "Only batch size of 1 can be shown!"
    data = data.cpu()
    data.squeeze_(0)
    if len(data.shape) == 3:
        data = data.max(dim=0).indices
    hue = torch.ones_like(data) * 10
    value = torch.ones_like(data) * 10
    data *= 50
    img_array = torch.stack((hue, value, data)).byte().numpy().transpose((1, 2, 0))
    img = Image.fromarray(img_array, mode="HSV")
    img.show()


@torch.no_grad()
def show_prediction(model, data, target):
    if len(data.shape) == 2:
        data.unsqueeze_(0).unsqueeze_(0)
    device = next(model.parameters()).device
    prediction = model(data.to(device).float())
    prediction = prediction.max(1).indices
    show_layers(target)
    show_layers(prediction)
import torch


class CharClassModel(torch.nn.Module):
    def __init__(self, c_in, c_hid, c_out, window_size=32):
        super(CharClassModel, self).__init__()
        self.model = torch.nn.Sequential(torch.nn.Conv2d(c_in, c_hid, 3, 1, 1),
                                         torch.nn.ReLU(),
                                         torch.nn.Conv2d(c_hid, c_hid, 3, 1, 1),
                                         torch.nn.ReLU(),
                                         torch.nn.Conv2d(c_hid, c_out, window_size))

    def forward(self, x):
        return self.model(x)

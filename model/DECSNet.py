import torch
import torch.nn as nn
from .src.encoder import Encoder
from .src.decoder import decoder

class DECSNet(nn.Module):
    def __init__(self, in_channel, out_channel, resolution, patchsz, mode, pretrained, cnn_dim, transform_dim, num_class):
        super().__init__()
        self.encoder = Encoder(in_channel, out_channel, resolution, patchsz, mode, pretrained, cnn_dim, transform_dim)
        self.decoder = decoder(num_class, transform_dim)

    def forward(self, x):
        o1, o2, o3, o4 = self.encoder(x)
        out = self.decoder(o1, o2, o3, o4)

        return out

if __name__=="__main__":
    device = torch.device("cuda:0")
    input = torch.rand(1, 3, 512, 512)
    input = input.to(device)
    net = DECSNet(3, 64, 512, 4, 'resnet34', False, [64, 128, 256, 512], [64, 128, 256, 512], 2)
    net.to(device)
    output = net(input)
    print(output.size())

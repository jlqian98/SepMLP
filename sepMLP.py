"""
SepMLP model for music source separation
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from einops.layers.torch import Rearrange

from modules.cycleMLP import cycle_blocks
from modules.utils import resize


class SepMLP(nn.Module):

    def __init__(self, input_channel=2):
        super(SepMLP, self).__init__()

        nb_filter = [64, 128, 256, 512]
        layers = [2, 2, 4, 2]
        
        self.encoder = MLPMIXER_Encoder(nb_filter=nb_filter, layers=layers, input_channel=input_channel)
        self.decoder = MLP_Decoder(nb_filter=nb_filter, num_classes=input_channel*4)


    def forward(self, x):
        """
        x: Spectrogram of input mixture (2 channels)
        return:
            Spectrograms of estimate sources (drums, bass, other and vocals, 8 channels altogether)
        """
        # masking
        encoder_output = self.encoder(x)
        decoder_output = self.decoder(encoder_output)
        decoder_output = resize(decoder_output, size=x.size()[
                                2:], mode='bilinear', align_corners=False)

        # output
        B, C, N, T = x.shape
        x = x.unsqueeze(1).expand(B, 4, C, N, T).reshape(B, 4*C, N, T)
        decoder_output = decoder_output * x

        return decoder_output


class MLPMIXER_Encoder(nn.Module):
    """
    not use
    """

    def __init__(self, nb_filter, layers, input_channel=1):
        super(MLPMIXER_Encoder, self).__init__()
        nb_filter = nb_filter
        self.layers = layers

        self.patchemb = nn.Sequential(
            nn.Conv2d(input_channel, nb_filter[0], kernel_size=7, stride=(2, 2), padding=3),
        )

        self.net = [
            nn.Sequential(
                nn.Conv2d(nb_filter[k-1], nb_filter[k], kernel_size=1) if k > 0 else nn.Identity(),
                nn.MaxPool2d(2, 2) if k > 0 else nn.Identity(),
                Rearrange('n c a b -> n a b c'),
                cycle_blocks(nb_filter[k], k, self.layers, attn_drop=0.0, drop_path_rate=0.0),
                Rearrange('n a b c -> n c a b'),
            ) for k in range(len(self.layers))
        ]       # multi-stage feature extraction

        self.net = nn.Sequential(*self.net)

    def forward(self, x):

        x = self.patchemb(x)
        out = []
        for i in range(len(self.layers)):
            x = self.net[i](x)
            out.append(x)
        return out


class MLP_Decoder(nn.Module):

    def __init__(self, nb_filter, num_classes=8):
        super().__init__()
        nb_filter = nb_filter
        embedding_dim = 256  # 256 or 768

        self.linears = [MLP_Decoder_Block(input_dim=nb_filter[k], embed_dim=embedding_dim) for k in range(len(nb_filter))]
        self.linears = nn.Sequential(*self.linears)

        self.linear_fuse = nn.Conv2d(
            embedding_dim * len(nb_filter), embedding_dim, kernel_size=1)
        self.linear_pred = nn.Conv2d(embedding_dim, num_classes, kernel_size=1)

        self.dropout = nn.Dropout(0.1)
        self.act = nn.Sigmoid()


    def forward(self, x):

        norm = x[0]
        
        _x = []
        for i in range(len(x)):
            z = self.linears[i](x[i])
            _z = resize(z, size=norm.size()[2:], mode='bilinear', align_corners=False)
            _x.append(_z)

        _c = self.linear_fuse(torch.cat(_x, dim=1))

        x = self.dropout(_c)
        x = self.linear_pred(x)
        x = self.act(x)

        return x


class MLP_Decoder_Block(nn.Module):
    def __init__(self, input_dim, embed_dim):
        super(MLP_Decoder_Block, self).__init__()
        self.fc = nn.Linear(input_dim, embed_dim)
        self.norm = nn.LayerNorm(input_dim)

    def forward(self, x):
        n, c, h, w = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.fc(x)
        x = x.transpose(1, 2).reshape(n, -1, h, w)
        return x


if __name__ == '__main__':
    
    model = SepMLP(input_channel=2)
    magnitude_input = torch.randn(1, 2, 512, 256)
    magnitude_output = model(magnitude_input)
    print(magnitude_output.shape)
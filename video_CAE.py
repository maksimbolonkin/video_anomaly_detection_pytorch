import torch.nn as nn
from convolution_lstm import ConvLSTM
from collections import OrderedDict

class VideoAutoencoderLSTM(nn.Module):
    def __init__(self, in_channels=1):
        super(VideoAutoencoderLSTM, self).__init__()
        self.in_channels = in_channels
        self.conv_encoder = nn.Sequential(OrderedDict([
              ('conv1', nn.Conv2d(in_channels=self.in_channels, out_channels=128, kernel_size=11,stride=4, padding=0)),
              ('nonl1', nn.Tanh()),
              ('conv2', nn.Conv2d(in_channels=128, out_channels=64, kernel_size=5,stride=2, padding=0)),
              ('nonl2', nn.Tanh())
            ]))
        self.rnn_enc_dec = ConvLSTM(input_channels=64,
                        hidden_channels=[64, 32, 64],
                        kernel_size=3,
                        batch_first=True,
                        input_dropout_rate=0.5,
                        reccurent_dropout_rate=0.5)
        self.conv_decoder = nn.Sequential(OrderedDict([
              ('deconv1', nn.ConvTranspose2d(in_channels=64, out_channels=128, kernel_size=5,stride=2, padding=0)),
              ('nonl1', nn.Tanh()),
              ('deconv2', nn.ConvTranspose2d(in_channels=128, out_channels=self.in_channels, kernel_size=11,stride=4, padding=0)),
              ('nonl2', nn.Tanh())
            ]))
    
    def forward(self, x):
        b, t, c, h, w = x.size()
        x = x.view(b*t,c,h,w)
        x = self.conv_encoder(x)
        x = x.view(b, t, x.size(1), x.size(2), x.size(3))
        x, _ = self.rnn_enc_dec(x)
        x = x.view(b*t, x.size(2), x.size(3), x.size(4))
        x = self.conv_decoder(x)
        x = x.view(b, t, x.size(1), x.size(2), x.size(3))
        return x

    def set_cuda(self):
        self.conv_encoder.cuda()
        self.rnn_enc_dec.cuda()
        self.conv_decoder.cuda()
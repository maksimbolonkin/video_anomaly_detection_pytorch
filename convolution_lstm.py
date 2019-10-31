import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np

# Adapted from https://github.com/automan000/Convolution_LSTM_PyTorch/blob/master/convolution_lstm.py
class ConvLSTMCell(nn.Module):
    def __init__(self,
            input_channels,
            hidden_channels,
            kernel_size,
            input_dropout_rate = 0.0,
            reccurent_drouput_rate = 0.0):
        super(ConvLSTMCell, self).__init__()

        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.num_features = 4
        self.input_dropout_rate = np.clip(input_dropout_rate, 0.0, 1.0)
        self.reccurent_drouput_rate = np.clip(reccurent_drouput_rate, 0.0, 1.0)

        self.padding = int((kernel_size - 1) / 2)

        self.input_dropout = nn.Dropout(p=self.input_dropout_rate) if 0.0 < self.input_dropout_rate < 1.0 else nn.Identity()
        self.reccurent_drouput = nn.Dropout(p=self.reccurent_drouput_rate) if 0.0 < self.input_dropout_rate < 1.0 else nn.Identity()

        self.Wxi = nn.Conv2d(self.input_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=True)
        self.Whi = nn.Conv2d(self.hidden_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=False)
        self.Wxf = nn.Conv2d(self.input_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=True)
        self.Whf = nn.Conv2d(self.hidden_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=False)
        self.Wxc = nn.Conv2d(self.input_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=True)
        self.Whc = nn.Conv2d(self.hidden_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=False)
        self.Wxo = nn.Conv2d(self.input_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=True)
        self.Who = nn.Conv2d(self.hidden_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=False)

        self.Wci = None
        self.Wcf = None
        self.Wco = None

    def forward(self, x, h, c):
        x = self.input_dropout(x)
        h = self.reccurent_drouput(h)
        ci = torch.sigmoid(self.Wxi(x) + self.Whi(h) + c * self.Wci)
        cf = torch.sigmoid(self.Wxf(x) + self.Whf(h) + c * self.Wcf)
        cc = cf * c + ci * torch.tanh(self.Wxc(x) + self.Whc(h))
        co = torch.sigmoid(self.Wxo(x) + self.Who(h) + cc * self.Wco)
        ch = co * torch.tanh(cc)
        return ch, cc

    def init_hidden(self, batch_size, hidden, shape, use_cuda=False):
        if self.Wci is None:
            self.Wci = Variable(torch.zeros(1, hidden, shape[0], shape[1]))
            self.Wcf = Variable(torch.zeros(1, hidden, shape[0], shape[1]))
            self.Wco = Variable(torch.zeros(1, hidden, shape[0], shape[1]))
        else:
            assert shape[0] == self.Wci.size()[2], 'Input Height Mismatched!'
            assert shape[1] == self.Wci.size()[3], 'Input Width Mismatched!'
        if use_cuda:
            self.Wci = self.Wci.cuda()
            self.Wcf = self.Wcf.cuda()
            self.Wco = self.Wco.cuda()
        h = Variable(torch.zeros(batch_size, hidden, shape[0], shape[1]))
        c = Variable(torch.zeros(batch_size, hidden, shape[0], shape[1]))
        if use_cuda:
            h, c = h.cuda(), c.cuda()
        return (h, c)

class ConvLSTM(nn.Module):
    # input_channels corresponds to the first input feature map
    # hidden state is a list of succeeding lstm layers.
    def __init__(self,
            input_channels,
            hidden_channels,
            kernel_size,
            batch_first=False, 
            input_dropout_rate = 0.0, 
            reccurent_dropout_rate = 0.0):
        super(ConvLSTM, self).__init__()
        self.input_channels = [input_channels] + hidden_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.num_layers = len(hidden_channels)
        self.batch_first = batch_first
        if not isinstance(input_dropout_rate, list):
            self.input_dropout_rate = [input_dropout_rate] * self.num_layers
        if not isinstance(reccurent_dropout_rate, list):
            self.reccurent_dropout_rate = [reccurent_dropout_rate] * self.num_layers

        self._all_layers = nn.ModuleList()
        for i in range(self.num_layers):
            name = 'cell{}'.format(i)
            cell = ConvLSTMCell(self.input_channels[i],
                        self.hidden_channels[i],
                        self.kernel_size, 
                        self.input_dropout_rate[i], 
                        self.reccurent_dropout_rate[i])
            setattr(self, name, cell)
            self._all_layers.append(cell)

    def forward(self, input, hidden_state=None):
        """
        Partially adapted code from
        https://github.com/ndrplz/ConvLSTM_pytorch/blob/master/convlstm.py
        Parameters
        ----------
        input_tensor: todo 
            5-D Tensor either of shape (t, b, c, h, w) or (b, t, c, h, w)
        hidden_state: todo
            None
            
        Returns
        -------
        last_state_list, layer_output
        """
        if not self.batch_first:
            # (t, b, c, h, w) -> (b, t, c, h, w)
            input = input.permute(1, 0, 2, 3, 4)

        internal_state = []
        outputs = []
        n_steps = input.size(1)
        for t in range(n_steps):
            x = input[:, t, :, :, :]
            for i in range(self.num_layers):
                # all cells are initialized in the first step
                name = 'cell{}'.format(i)
                if t == 0:
                    bsize, _, height, width = x.size()
                    (h, c) = getattr(self, name).init_hidden(batch_size=bsize, hidden=self.hidden_channels[i],
                                                             shape=(height, width), use_cuda = input.is_cuda)
                    internal_state.append((h, c))

                # do forward
                (h, c) = internal_state[i]
                x, new_c = getattr(self, name)(x, h, c)
                internal_state[i] = (x, new_c)
            outputs.append(x)
        outputs = torch.stack(outputs, dim=1)

        return outputs, (x, new_c)
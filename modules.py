import torch
from torch import nn
from torch.autograd import Variable
from torch.nn import functional as tf
from custom_types import device


def xavier(ts):
    return nn.init.xavier_normal_(ts)


def orthogonal_(ts):
    return nn.init.orthogonal_(ts)


def init_hidden(x, hidden_size: int, num_layer, bi=False):
    """
    Train the initial value of the hidden state:
    https://r2rt.com/non-zero-initial-states-for-recurrent-neural-networks.html
    """
    # if bilstm: 2 * x.size(0) * hidden_size
    if bi:
        # ini_tensor = Variable(orthogonal_(torch.zeros(2, int(x.size(0) / 2), hidden_size)))
        ini_tensor = Variable(orthogonal_(torch.zeros(2 * num_layer, x, hidden_size)))
    else:
        ini_tensor = Variable(orthogonal_(torch.zeros(1 * num_layer, x, hidden_size)))
    # if torch.cuda.is_available():
    #     ini_tensor = ini_tensor.cuda()
    return ini_tensor.to(device)


def zero_initial(batch_size, T, input_size, num_layer, bi=False):
    if bi:
        zero_tensor = Variable(orthogonal_(torch.zeros(batch_size, T, input_size * 2 * num_layer)))
    else:
        zero_tensor = Variable(orthogonal_(torch.zeros(batch_size, T, input_size * num_layer)))
    # if torch.cuda.is_available():
    #     zero_tensor = zero_tensor.cuda()
    return zero_tensor.to(device)


class Encoder(nn.Module):

    def __init__(self, input_size: int, hidden_size: int, T: int, bidirec: bool, num_layer: int):

        super(Encoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.T = T
        self.bidirec = bidirec
        self.num_layer = num_layer
        self.lstm_layer = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layer,
            bidirectional=bidirec
        )

        # self.attn_linear = nn.Linear(in_features=2 * hidden_size + T, out_features=1)
        self.attn_linear = nn.Sequential(
            nn.Linear(in_features=2 * hidden_size + T, out_features=self.T),
            nn.Linear(in_features=self.T, out_features=self.T),
            nn.Tanh(),
            nn.Linear(in_features=self.T, out_features=1)

        )

    def forward(self, input_data):
        # input_weighted = Variable(torch.zeros(input_data.size(0), self.T - 1, self.input_size))
        # input_encoded = Variable(torch.zeros(input_data.size(0), self.T - 1, self.hidden_size))
        input_weighted = Variable(torch.zeros(input_data.size(0), self.T, self.input_size)).to(device)
        input_encoded = zero_initial(input_data.size(0), self.T, self.hidden_size, self.num_layer, bi=True)
        # hidden, cell: initial states with dimension hidden_size
        hidden = init_hidden(input_data.size(0), self.hidden_size, self.num_layer, bi=True)
        cell = init_hidden(input_data.size(0), self.hidden_size, self.num_layer, bi=True)

        if self.bidirec:
            rep = int(self.input_size / 2 / self.num_layer)
        else:
            rep = int(self.input_size / self.num_layer)

        for t in range(self.T):
            # concatenate the hidden states with each predictor
            # rep 将其弄成input_size大小
            hid = hidden.repeat(rep, 1, 1).permute(1, 0, 2)
            cel = cell.repeat(rep, 1, 1).permute(1, 0, 2)
            inp = input_data.permute(0, 2, 1)

            x = torch.cat((hid, cel, inp), dim=2)
            x = x.view(-1, self.hidden_size * 2 + self.T)
            x = self.attn_linear(x)
            # encoder attention weights
            attn_weights = tf.softmax(x.view(-1, self.input_size), dim=1)
            input_weighted[:, t, :] = attn_weights
            # weighted_input <batch_size,input_size>
            weighted_input = torch.mul(attn_weights, input_data[:, t, :])
            # 保存lstm中各weight，并替换他们的指针
            self.lstm_layer.flatten_parameters()
            _, lstm_states = self.lstm_layer(weighted_input.unsqueeze(0), (hidden, cell))
            hidden = lstm_states[0]
            cell = lstm_states[1]
            # Save output
            # input_weighted[:, t, :] = weighted_input

            input_encoded[:, t, :] = hidden.view(input_data.size(0), -1)

        # return input_weighted, input_encoded.view(input_data.size(0), self.T, -1)
        return input_weighted, input_encoded


class Decoder(nn.Module):

    def __init__(self, encoder_hidden_size, decoder_hidden_size: int, T, out_feats=1, bidirec=False, num_layer=1):
        super(Decoder, self).__init__()

        self.T = T
        self.encoder_hidden_size = encoder_hidden_size
        self.decoder_hidden_size = decoder_hidden_size
        self.bidirec = bidirec
        self.num_layer = num_layer

        self.attn_layer = nn.Sequential(
            nn.Linear(2 * decoder_hidden_size + (bidirec * 2) * num_layer * encoder_hidden_size, decoder_hidden_size),
            nn.Linear(decoder_hidden_size, decoder_hidden_size),
            nn.Tanh(),
            nn.Linear(encoder_hidden_size, 1))

        self.lstm_layer = nn.LSTM(input_size=out_feats, hidden_size=decoder_hidden_size, num_layers=num_layer,
                                  bidirectional=bidirec)

        self.fc = nn.Sequential(
            nn.Linear(bidirec * 2 * num_layer * encoder_hidden_size + out_feats + 1, decoder_hidden_size),
            nn.Linear(decoder_hidden_size, out_feats)
        )
        self.fc_final = nn.Linear(decoder_hidden_size + (bidirec * 2) * num_layer * encoder_hidden_size, out_feats)

        # self.fc.weight.data.normal_()

    def forward(self, input_encoded, y_history, speed):
        # initialize hidden and cell
        hidden = init_hidden(input_encoded.size(0), self.decoder_hidden_size, self.num_layer, bi=True)
        cell = init_hidden(input_encoded.size(0), self.decoder_hidden_size, self.num_layer, bi=True)
        context = Variable(orthogonal_(torch.zeros(input_encoded.size(0), self.encoder_hidden_size)))
        weights = Variable(torch.zeros(input_encoded.size(0), self.T)).to(device)

        if self.bidirec:
            bi = 2
            rep = int(self.T / 2 / self.num_layer)
        else:
            bi = 1
            rep = int(self.T / self.num_layer)

        for t in range(self.T):
            # (batch_size, T, (2 * decoder_hidden_size + encoder_hidden_size))
            x = torch.cat((hidden.repeat(rep, 1, 1).permute(1, 0, 2),
                           cell.repeat(rep, 1, 1).permute(1, 0, 2),
                           input_encoded), dim=2)
            # decoder attention weights
            x = tf.softmax(
                self.attn_layer(
                    x.view(-1, 2 * self.decoder_hidden_size + bi * self.num_layer * self.encoder_hidden_size)
                ).view(-1, self.T),
                dim=1)

            weights += x

            # context vector
            context = torch.bmm(x.unsqueeze(1), input_encoded)[:, 0, :]

            context_tmp = torch.cat((context, y_history[:, t]), dim=1)
            y_tilde = self.fc(torch.cat((context_tmp, speed[:, t]), dim=1))

            self.lstm_layer.flatten_parameters()
            _, lstm_output = self.lstm_layer(y_tilde.unsqueeze(0), (hidden, cell))
            hidden = lstm_output[0]
            cell = lstm_output[1]

            # final output
        return weights / self.T, self.fc_final(torch.cat((hidden[0], context), dim=1))
        # return weights / self.T, self.fc_final(torch.cat((hidden[0], context), dim=1))
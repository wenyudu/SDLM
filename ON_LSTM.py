import torch.nn.functional as F
import torch.nn as nn
import torch
import numpy

from locked_dropout import LockedDropout


class LayerNorm(nn.Module):

    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones(features))
        self.beta = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.gamma * (x - mean) / (std + self.eps) + self.beta


class LinearDropConnect(nn.Linear):
    def __init__(self, in_features, out_features, bias=True, dropout=0.):
        super(LinearDropConnect, self).__init__(
            in_features=in_features,
            out_features=out_features,
            bias=bias
        )
        self.dropout = dropout

    def sample_mask(self):
        if self.dropout == 0.:
            self._weight = self.weight
        else:
            mask = self.weight.new_empty(
                self.weight.size(),
                dtype=torch.uint8
            )
            mask.bernoulli_(self.dropout)
            self._weight = self.weight.masked_fill(mask.bool(), 0.)

    def forward(self, input, sample_mask=False):
        if self.training:
            if sample_mask:
                self.sample_mask()
            return F.linear(input, self._weight, self.bias)
        else:
            return F.linear(input, self.weight * (1 - self.dropout),
                            self.bias)


def cumsoftmax(x, dim=-1):
    return torch.cumsum(F.softmax(x, dim=dim), dim=dim)


def softmax(x, dim=-1):
    return F.softmax(x, dim=dim)

def cum(x, dim=-1):
    return torch.cumsum(x, dim=dim)



class ONLSTMCell(nn.Module):

    def __init__(self, input_size, hidden_size, chunk_size, wds='no', dropconnect=0.):
        super(ONLSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.chunk_size = chunk_size
        self.n_chunk = int(hidden_size / chunk_size)

        self.ih = nn.Sequential(
            nn.Linear(input_size, 4 * hidden_size + self.n_chunk * 2, bias=True),
            # LayerNorm(3 * hidden_size)
        )
        self.hh = LinearDropConnect(hidden_size, hidden_size * 4 + self.n_chunk * 2, bias=True, dropout=dropconnect)

        # self.c_norm = LayerNorm(hidden_size)

        # self.fwh = LinearDropConnect(self.n_chunk, self.n_chunk, bias=True, dropout=dropconnect)

        # self.drop_weight_modules = [self.hh,self.fwh]

        self.wds = wds
        if self.wds != 'no':
            self.fwh = LinearDropConnect(self.n_chunk, self.n_chunk, bias=True, dropout=dropconnect)
            self.drop_weight_modules = [self.hh, self.fwh]
        else:
            self.drop_weight_modules = [self.hh]

        # self.wds = wds
        # if self.wds != 'no':
        #     self.weighted_sd_vector = nn.Parameter(torch.zeros(self.n_chunk))

    def forward(self, input, hidden,
                transformed_input=None):
        hx, cx = hidden

        if transformed_input is None:
            transformed_input = self.ih(input)

        gates = transformed_input + self.hh(hx)
        cingate_raw, cforgetgate_raw = gates[:, :self.n_chunk * 2].chunk(2, 1)
        outgate, cell, ingate, forgetgate = gates[:, self.n_chunk * 2:].view(-1, self.n_chunk * 4,
                                                                             self.chunk_size).chunk(4, 1)
        cingate = 1. - cumsoftmax(cingate_raw)
        distance_cin = cingate.sum(dim=-1) / self.n_chunk

        cforgetgate = cumsoftmax(cforgetgate_raw)
        distance_cforget = 1. - cforgetgate.sum(dim=-1) / self.n_chunk

        if self.wds != 'no':
            c_w_forgetgate = cumsoftmax(self.fwh(cforgetgate_raw))
            distance_w_cforget = 1. - c_w_forgetgate.sum(dim=-1) / self.n_chunk
        else:
            distance_w_cforget = distance_cforget

        cingate = cingate[:, :, None]
        cforgetgate = cforgetgate[:, :, None]

        ingate = torch.sigmoid(ingate)
        forgetgate = torch.sigmoid(forgetgate)
        cell = torch.tanh(cell)
        outgate = torch.sigmoid(outgate)

        # cy = cforgetgate * forgetgate * cx + cingate * ingate * cell

        overlap = cforgetgate * cingate
        forgetgate = forgetgate * overlap + (cforgetgate - overlap)
        ingate = ingate * overlap + (cingate - overlap)
        cy = forgetgate * cx + ingate * cell

        # hy = outgate * F.tanh(self.c_norm(cy))
        hy = outgate * torch.tanh(cy)

        # self.last = [transformed_input, cforgetgate, weight, distance_cforget,hy,cy]
        # if self.wds != 'no':
        #     # return hy.view(-1, self.hidden_size), cy ,(origin_distance_cforget, distance_cforget, distance_cin,self.weighted_sd_vector)
        # else:
        return hy.view(-1, self.hidden_size), cy ,(distance_cforget, distance_w_cforget, distance_cin,distance_cin)


    def init_hidden(self, bsz):
        weight = next(self.parameters()).data
        return (weight.new(bsz, self.hidden_size).zero_(),
                weight.new(bsz, self.n_chunk, self.chunk_size).zero_())

    def sample_masks(self):
        for m in self.drop_weight_modules:
            m.sample_mask()


class ONLSTMStack(nn.Module):
    def __init__(self, layer_sizes, chunk_size, l4d=0,wds='no', dropout=0., dropconnect=0.):
        super(ONLSTMStack, self).__init__()
        self.cells = nn.ModuleList([ONLSTMCell(layer_sizes[i],
                                               layer_sizes[i + 1],
                                               chunk_size,
                                               wds=wds,
                                               dropconnect=dropconnect)
                                    for i in range(len(layer_sizes) - 1)])
        self.lockdrop = LockedDropout()
        self.dropout = dropout
        self.sizes = layer_sizes
        self.l4d = l4d

    def init_hidden(self, bsz):
        return [c.init_hidden(bsz) for c in self.cells]

    def forward(self, input, hidden):
        length, batch_size, _ = input.size()

        if self.training:
            for c in self.cells:
                c.sample_masks()

        prev_state = list(hidden)
        prev_layer = input

        raw_outputs = []
        outputs = []
        distances_forget = []
        origin_distances_forget = []
        distances_in = []
        weighted_sd_vector=[]

        for l in range(len(self.cells)):
            curr_layer = [None] * length
            dist = [None] * length
            t_input = self.cells[l].ih(prev_layer)

            for t in range(length):
                hidden, cell, d = self.cells[l](
                    None, prev_state[l],
                    transformed_input=t_input[t]
                )
                prev_state[l] = hidden, cell  # overwritten every timestep
                curr_layer[t] = hidden
                dist[t] = d

            prev_layer = torch.stack(curr_layer)
            origin_dist_cforget, dist_cforget, dist_cin, wsd_vector = zip(*dist)
            origin_dist_layer_cforget = torch.stack(origin_dist_cforget)
            dist_layer_cforget = torch.stack(dist_cforget)
            dist_layer_cin = torch.stack(dist_cin)
            wsd_layer_vector = torch.stack(wsd_vector)
            raw_outputs.append(prev_layer)
            if l < len(self.cells) - 1:
                prev_layer = self.lockdrop(prev_layer, self.dropout)
            outputs.append(prev_layer)
            distances_forget.append(dist_layer_cforget)
            origin_distances_forget.append(origin_dist_layer_cforget)
            distances_in.append(dist_layer_cin)
            if l == self.l4d:
                weighted_sd_vector.append(wsd_layer_vector)
        output = prev_layer
        # print(self.cells[2].weighted_vector[0])

        return output, prev_state, raw_outputs, outputs, (torch.stack(origin_distances_forget),torch.stack(distances_forget), torch.stack(distances_in), torch.stack(weighted_sd_vector))


if __name__ == "__main__":
    x = torch.Tensor(10, 10, 10)
    x.data.normal_()
    lstm = ONLSTMStack([10, 10, 10], chunk_size=10)
    print(lstm(x, lstm.init_hidden(10))[1])

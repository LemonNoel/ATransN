import torch
import torch.nn as nn


class TransNetwork(nn.Module):
    def __init__(self, input_size, output_size, activation=None):
        super(TransNetwork, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, output_size),
            nn.BatchNorm1d(output_size),
            nn.Linear(output_size, output_size))
        self.activation = activation

        for layer in self.model:
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight)
                nn.init.constant_(layer.bias, 0.0)

    def forward(self, x):
        if x.size(0) != 1:
            trans_emb = self.model(x)
        else:
            trans_emb = x
            for layer in self.model:
                if not isinstance(layer, nn.BatchNorm1d):
                    trans_emb = layer(trans_emb)
        if self.activation is not None:
            trans_emb = self.activation(trans_emb)
        return trans_emb


class Generator(nn.Module):
    """ generate an embedding in the student KG space """

    def __init__(self, input_size, activation=None):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(2*input_size, input_size),
            nn.BatchNorm1d(input_size),
            nn.LeakyReLU(1/5.5, inplace=True),
            nn.Linear(input_size, input_size))
        self.activation = activation

        for layer in self.model:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_uniform_(layer.weight)
                nn.init.constant_(layer.bias, 0.0)

    def forward(self, x, z):
        y = torch.cat([x, z], dim=-1)
        if y.size(0) != 1:
            fake = self.model(y)
        else:
            fake = y
            for layer in self.model:
                if not isinstance(layer, nn.BatchNorm1d):
                    fake = layer(fake)
        if self.activation is not None:
            fake = self.activation(fake)
        return fake


class Discriminator(nn.Module):
    def __init__(self, input_size, activation=None):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(4*input_size, 2*input_size),
            nn.LayerNorm(2*input_size),
            nn.LeakyReLU(1/5.5, inplace=True),
            nn.Linear(2*input_size, 1))
        self.activation = activation

        for layer in self.model:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_uniform_(layer.weight)
                nn.init.constant_(layer.bias, 3.0)

    def forward(self, x, y):
        validity = self.model(torch.cat([x, y, torch.abs(x-y), x*y], dim=-1))
        if self.activation is not None:
            validity = self.activation(validity)
        return validity

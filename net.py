import torch
import torch.utils.data
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torchvision import datasets, transforms
import pickle


class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()
        self.fc1 = nn.Linear(42, 14)
        self.fc21 = nn.Linear(14, 3)
        self.fc22 = nn.Linear(14, 3)
        self.fc3 = nn.Linear(3, 14)
        self.fc4 = nn.Linear(14, 42)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.l1loss = nn.L1Loss()
        self.sigmoid = nn.Sigmoid()

    def encode(self, x):
        h1 = self.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def decode(self, z):
        h3 = self.tanh(self.fc3(z))
        return self.tanh(self.fc4(h3))

    def reparametrize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        eps = Variable(std.data.new(std.size()).normal_())
        return eps.mul(std).add_(mu)

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, 42))
        z = self.reparametrize(mu, logvar)
        return self.decode(z), mu, logvar



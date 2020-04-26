import torch

from torch import nn
from torch.nn import functional as F


class VAE(nn.Module):
    def __init__(self, input_size):
        super(VAE, self).__init__()

        self.input_size = int(input_size)

        self.fc1 = nn.Linear(self.input_size, 4096)
        self.fc2 = nn.Linear(4096, 1024)
        self.fc31 = nn.Linear(1024, 100)
        self.fc32 = nn.Linear(1024, 100)
        self.fc4 = nn.Linear(100, 1024)
        self.fc5 = nn.Linear(1024, 4096)
        self.fc6 = nn.Linear(4096, self.input_size)

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        h2 = F.relu(self.fc2(h1))
        return self.fc31(h2), self.fc32(h2)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self, z):
        h4 = F.relu(self.fc4(z))
        h5 = F.relu(self.fc5(h4))

        # rescale the sigmoid assuming encoded audio goes from -1 to 1
        # TODO: how (does) should this actually work?
        return (torch.sigmoid(self.fc6(h5)) - 1) * 2

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, self.input_size))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


    # Reconstruction + KL divergence losses summed over all elements and batch
    def loss_fn(self, recon_x, x, mu, logvar):
        BCE = F.binary_cross_entropy(recon_x, x.view(-1, self.input_size), reduction='sum')

        # see Appendix B from VAE paper:
        # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
        # https://arxiv.org/abs/1312.6114
        # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

        return BCE + KLD

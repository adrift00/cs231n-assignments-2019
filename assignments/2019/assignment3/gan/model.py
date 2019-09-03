import torch.nn as nn




class Discriminator(nn.Module):
    """
    判别器
    """

    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(1, 32, 5),
            nn.LeakyReLU(),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(32, 64, 5),
            nn.LeakyReLU(),
            nn.MaxPool2d(2, stride=2),
            Flatten(),
            nn.Linear(4 * 4 * 64, 4 * 4 * 64),
            nn.LeakyReLU(),
            nn.Linear(4 * 4 * 64, 1)
        )

    def forward(self, x):
        return self.model(x)


class Generator(nn.Module):
    """
    生成器
    """

    def __init__(self, noise_dim):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(noise_dim, 1024),
            nn.ReLU(),
            nn.BatchNorm1d(1024),
            nn.Linear(1024, 7 * 7 * 128),
            nn.ReLU(),
            nn.BatchNorm1d(7 * 7 * 128),
            Unflatten(128,7,7),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.ConvTranspose2d(64, 1, 4, stride=2, padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        return self.model(x)


class Flatten(nn.Module):

    def forward(self, x):
        N = x.size()[0]
        return x.view(N, -1)


class Unflatten(nn.Module):

    def __init__(self, C, H, W):
        super(Unflatten, self).__init__()
        self.N = -1
        self.C = C
        self.H = H
        self.W = W

    def forward(self, x):
        return x.view(self.N, self.C, self.H, self.W)

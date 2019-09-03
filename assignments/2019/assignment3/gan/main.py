import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import torchvision.datasets as dataset
import torchvision.transforms as T

from gan.model import Discriminator, Generator
from gan.utils import *

noise_dim = 96

num_epochs = 5
batch_size = 128

lr_d = 1e-3
lr_g = 1e-3

betas = (0.5, 0.999)

show_every = 250

device = torch.device('cuda')


def train():
    transform=T.Compose([
        T.ToTensor(),
        T.Normalize((0.5,),(0.5,))
    ])
    mnist = dataset.MNIST('../cs231n/datasets/MNIST_data', train=True, download=True, transform=transform)
    dataloader = DataLoader(mnist, batch_size, shuffle=True)

    D = Discriminator()
    G = Generator(noise_dim)
    D.to(device)
    G.to(device)

    optim_D = optim.Adam(D.parameters(), lr=lr_d, betas=betas)
    optim_G = optim.Adam(G.parameters(), lr=lr_g, betas=betas)

    it = 0
    for ep in range(num_epochs):
        for x, _ in dataloader:
            x = x.to(device)
            optim_D.zero_grad()
            scores_real = D(x)
            fake_seed = sample_noise(batch_size, noise_dim).to(device)
            fake_img = G(fake_seed).detach()  # 将生成的数据从计算图中分离，不计算梯度
            scores_fake = D(fake_img)
            loss_d = discriminator_loss(scores_real, scores_fake)
            loss_d.backward()
            optim_D.step()

            optim_G.zero_grad()
            fake_seed = sample_noise(batch_size, noise_dim).to(device)
            fake_img = G(fake_seed)
            scores_fake = D(fake_img)
            loss_g = generator_loss(scores_fake)
            loss_g.backward()
            optim_G.step()
            if it % show_every == 0:
                print('iter: %d, D_loss: %.4f, G_loss: %.4f' % (it, loss_d, loss_g))
                images = fake_img[:16].data.cpu().numpy()
                show_image(images)
            it += 1


if __name__ == '__main__':
    train()

import torch
import torch.nn as nn
import torch.nn.init as init
from torch.utils.data import sampler
import numpy as np
import matplotlib
# matplotlib.use("Qt5Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
plt.rcParams['image.cmap'] = 'gray'

from gan.main import device


def bce_loss(input, target):
    """
    Numerically stable version of the binary cross-entropy loss function.

    As per https://github.com/pytorch/pytorch/issues/751
    See the TensorFlow docs for a derivation of this formula:
    https://www.tensorflow.org/api_docs/python/tf/nn/sigmoid_cross_entropy_with_logits

    Inputs:
    - input: PyTorch Tensor of shape (N, ) giving scores.
    - target: PyTorch Tensor of shape (N,) containing 0 and 1 giving targets.

    Returns:
    - A PyTorch Tensor containing the mean BCE loss over the minibatch of input data.
    """
    neg_abs = - input.abs()
    loss = input.clamp(min=0) - input * target + (1 + neg_abs.exp()).log()
    return loss.mean()


def discriminator_loss(logits_real, logits_fake):
    """
    Computes the discriminator loss described above.

    Inputs:
    - logits_real: PyTorch Tensor of shape (N,) giving scores for the real data.
    - logits_fake: PyTorch Tensor of shape (N,) giving scores for the fake data.

    Returns:
    - loss: PyTorch Tensor containing (scalar) the loss for the discriminator.
    """
    real_N = logits_real.size()
    fake_N = logits_fake.size()
    labels_real = torch.ones(real_N).to(device)
    labels_fake = torch.zeros(fake_N).to(device)

    loss_real = torch.mean(bce_loss(logits_real, labels_real))
    loss_fake = torch.mean(bce_loss(logits_fake, labels_fake))

    loss = loss_real + loss_fake
    return loss


def generator_loss(logits_fake):
    """
    Computes the generator loss described above.

    Inputs:
    - logits_fake: PyTorch Tensor of shape (N,) giving scores for the fake data.

    Returns:
    - loss: PyTorch Tensor containing the (scalar) loss for the generator.
    """
    fake_N = logits_fake.size()
    labels_fake = torch.ones(fake_N).to(device)

    loss = torch.mean(bce_loss(logits_fake, labels_fake))
    return loss


def sample_noise(batch_size, dim):
    """
    Generate a PyTorch Tensor of uniform random noise.

    Input:
    - batch_size: Integer giving the batch size of noise to generate.
    - dim: Integer giving the dimension of noise to generate.

    Output:
    - A PyTorch Tensor of shape (batch_size, dim) containing uniform
      random noise in the range (-1, 1).
    """

    return 2 * torch.rand(batch_size, dim) - 1


def show_image(images):
    N,C,H,W=images.shape
    sqrtN=int(np.ceil(np.sqrt(N)))
    fig=plt.figure()
    gs=gridspec.GridSpec(sqrtN,sqrtN)
    gs.update(wspace=0.05, hspace=0.05)
    for i, img in enumerate(images):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        plt.imshow(img.reshape(H,W))
    plt.show()



def initialize_weights(m):
    if isinstance(m, nn.Linear) or isinstance(m, nn.ConvTranspose2d):
        init.xavier_uniform_(m.weight.data)
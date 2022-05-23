import itertools
import random
from typing import Any, Tuple

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data as tud
import torchvision.utils as vutils
from IPython.display import HTML
from tqdm import tqdm

import src.data as data
from src.models import gan

lr = 0.0002
beta1 = 0.5


def weights_init(m: torch.Tensor) -> None:
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


def set_requires_grad(nets: list[nn.Module], requires_grad: bool = False) -> None:
    """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
    Parameters:
        nets (network list)   -- a list of networks
        requires_grad (bool)  -- whether the networks require gradients or not
    """
    if not isinstance(nets, list):
        nets = [nets]
    for net in nets:
        if net is not None:
            for param in net.parameters():
                param.requires_grad = requires_grad


def _step(
    h: torch.Tensor,
    z: torch.Tensor,
    Gz: nn.Module,
    Dz: nn.Module,
    Gh: nn.Module,
    Dh: nn.Module,
    optimD: optim.Optimizer,
    optimG: optim.Optimizer,
):
    fake_z = Gz(h)
    fake_h = Gh(z)
    id_h = Gh(h)
    id_z = Gz(z)
    recons_z = Gz(fake_h)
    recons_h = Gh(fake_z)
    mse = nn.MSELoss()
    gan_loss = nn.MSELoss()
    cycle_loss = nn.L1Loss()
    id_loss = nn.L1Loss()
    # G loss
    set_requires_grad([Dz, Dh], False)
    optimG.zero_grad()
    square_loss_gz = mse(Dz(fake_z), torch.ones_like(Dz(fake_z)))
    cycle_consistency_loss_gz = cycle_loss(recons_z, z)
    identity_loss_gz = id_loss(id_z, z)
    square_loss_gh = mse(Dh(fake_h), torch.ones_like(Dh(fake_h)))
    cycle_consistency_loss_gh = cycle_loss(recons_h, h)
    identity_loss_gh = id_loss(id_h, h)
    G_loss = (
        square_loss_gz
        + cycle_consistency_loss_gz  # noqa
        + identity_loss_gz  # noqa
        + square_loss_gh  # noqa
        + cycle_consistency_loss_gh  # noqa
        + identity_loss_gh  # noqa  # noqa
    )
    G_loss.backward()
    optimG.step()
    # D loss
    set_requires_grad([Dz, Dh], True)
    optimD.zero_grad()
    real_loss_dz = gan_loss(Dz(h), torch.ones_like(Dz(h)))
    fake_loss_dz = gan_loss(Dz(fake_z.detach()), torch.zeros_like(Dz(fake_z)))
    real_loss_dh = gan_loss(Dh(z), torch.ones_like(Dh(z)))
    fake_loss_dh = gan_loss(Dh(fake_h.detach()), torch.zeros_like(Dh(fake_h)))
    D_loss = (real_loss_dz + fake_loss_dz + real_loss_dh + fake_loss_dh) * 0.5
    D_loss.backward()
    optimD.step()
    print(type(G_loss))
    return G_loss, D_loss


def train(
    Gz: nn.Module,
    Gh: nn.Module,
    Dz: nn.Module,
    Dh: nn.Module,
    dataloader: tud.Dataset,
    num_epochs: int = 5,
) -> Tuple[list, list, list]:
    # Lists to keep track of progress
    img_list = []
    G_losses = []
    D_losses = []
    iters = 0
    optimD = optim.Adam(
        itertools.chain(Dz.parameters(), Dh.parameters()), lr=lr, betas=(beta1, 0.999)
    )
    optimG = optim.Adam(
        itertools.chain(Gz.parameters(), Gh.parameters()), lr=lr, betas=(beta1, 0.999)
    )
    print("Starting Training Loop...")
    # For each epoch
    for epoch in range(num_epochs):
        epoch_g_loss = 0
        epoch_d_loss = 0
        # For each batch in the dataloader
        loop = tqdm(dataloader)
        for i, (h, z) in enumerate(loop):
            h, z = h.cuda(), z.cuda()
            g_loss, d_loss = _step(h, z, Gz, Dz, Gh, Dh, optimD, optimG)
            epoch_d_loss += d_loss.item()
            epoch_g_loss += g_loss.item()
            loop.set_description(f"Epoch [{epoch+1}/{num_epochs}]")
            loop.set_postfix(G_loss=g_loss.item(), D_loss=d_loss.item())
            # Check how the generator is doing by saving G's output on fixed_noise
            if (iters % 500 == 0) or (
                (epoch == num_epochs - 1) and (i == len(dataloader) - 1)
            ):
                with torch.no_grad():
                    fake = Gz(h).detach().cpu()
                img_list.append(vutils.make_grid(fake, padding=2, normalize=True))
            iters += 1
        print(
            f"Epoch {epoch+1}/{num_epochs}: G_loss: {(epoch_g_loss/i+1):.4f}, D_loss: {(epoch_d_loss/i+1):.4f}"  # noqa
        )
        G_losses.append(epoch_g_loss)
        D_losses.append(epoch_d_loss)
    return G_losses, D_losses, img_list


def plot_learning_curves(G_losses: list[int], D_losses: list[int]) -> None:
    plt.figure(figsize=(10, 5))
    plt.title("Generator and Discriminator Loss During Training")
    plt.plot(G_losses, label="G")
    plt.plot(D_losses, label="D")
    plt.xlabel("iterations")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()


def plot_images_generation(img_list: list) -> None:
    fig = plt.figure(figsize=(8, 8))
    plt.axis("off")
    ims = [[plt.imshow(np.transpose(i, (1, 2, 0)), animated=True)] for i in img_list]
    ani = animation.ArtistAnimation(
        fig, ims, interval=1000, repeat_delay=1000, blit=True
    )
    HTML(ani.to_jshtml())
    plt.show()


def fake_vs_real(dataloader: tud.Dataset, img_list: list, device: Any) -> None:
    # Grab a batch of real images from the dataloader
    real_batch = next(iter(dataloader))

    # Plot the real images
    plt.figure(figsize=(15, 15))
    plt.subplot(1, 2, 1)
    plt.axis("off")
    plt.title("Real Images")
    plt.imshow(
        np.transpose(
            vutils.make_grid(
                real_batch[0].to(device)[:64], padding=5, normalize=True
            ).cpu(),
            (1, 2, 0),
        )
    )

    # Plot the fake images from the last epoch
    plt.subplot(1, 2, 2)
    plt.axis("off")
    plt.title("Fake Images")
    plt.imshow(np.transpose(img_list[-1], (1, 2, 0)))
    plt.show()


if __name__ == "__main__":
    dataloader = data.get_datasets(
        "dataset/horse2zebra/trainA/", "dataset/horse2zebra/trainB/"
    )
    device = torch.device("cuda")
    # Create the generator
    Gz = gan.Generator().to(device)
    Gh = gan.Generator().to(device)
    Gz.apply(weights_init)
    Gh.apply(weights_init)
    # Create the Discriminator
    Dz = gan.Discriminator().to(device)
    Dh = gan.Discriminator().to(device)
    Dz.apply(weights_init)
    Dh.apply(weights_init)

    lossG, lossD, img_list = train(Gz, Gh, Dz, Dh, dataloader, 10)
    plot_images_generation(img_list)
    fake_vs_real(dataloader, img_list, device)

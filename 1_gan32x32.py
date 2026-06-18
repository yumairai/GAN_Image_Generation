import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torchvision import datasets
from torchvision import transforms
from torch.utils.data.sampler import SubsetRandomSampler


# ==========================================
# DATA LOADER
# ==========================================

def get_data_loader(batch_size, image_size, data_dir='anime/'):

    image_transforms = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
    ])

    dataset = datasets.ImageFolder(
        data_dir,
        transform=image_transforms
    )

    n_samples = min(len(dataset), 50000)

    indices = np.random.choice(
        len(dataset),
        n_samples,
        replace=False
    )

    data_loader = torch.utils.data.DataLoader(
        dataset,
        sampler=SubsetRandomSampler(indices),
        batch_size=batch_size
    )

    return data_loader


# ==========================================
# NORMALIZATION
# ==========================================

def scale(x, feature_range=(-1, 1)):
    min_val, max_val = feature_range
    x = x * (max_val - min_val) + min_val
    return x


# ==========================================
# DISCRIMINATOR
# ==========================================

class Discriminator(nn.Module):

    def __init__(self, conv_dim):

        super(Discriminator, self).__init__()

        self.conv_dim = conv_dim

        self.conv1 = nn.Conv2d(
            3,
            conv_dim,
            kernel_size=4,
            stride=2,
            padding=1,
            bias=False
        )

        self.batch_norm1 = nn.BatchNorm2d(conv_dim)

        self.conv2 = nn.Conv2d(
            conv_dim,
            conv_dim * 2,
            kernel_size=4,
            stride=2,
            padding=1,
            bias=False
        )

        self.batch_norm2 = nn.BatchNorm2d(conv_dim * 2)

        self.conv3 = nn.Conv2d(
            conv_dim * 2,
            conv_dim * 4,
            kernel_size=4,
            stride=2,
            padding=1,
            bias=False
        )

        self.batch_norm3 = nn.BatchNorm2d(conv_dim * 4)

        self.conv4 = nn.Conv2d(
            conv_dim * 4,
            conv_dim * 8,
            kernel_size=4,
            stride=2,
            padding=1,
            bias=False
        )

        self.batch_norm4 = nn.BatchNorm2d(conv_dim * 8)

        self.conv5 = nn.Conv2d(
            conv_dim * 8,
            conv_dim * 16,
            kernel_size=4,
            stride=2,
            padding=1,
            bias=False
        )

        self.fc = nn.Linear(conv_dim * 4 * 4, 1)

    def forward(self, x):

        x = F.leaky_relu(
            self.batch_norm1(self.conv1(x)),
            0.2
        )

        x = F.leaky_relu(
            self.batch_norm2(self.conv2(x)),
            0.2
        )

        x = F.leaky_relu(
            self.batch_norm3(self.conv3(x)),
            0.2
        )

        x = F.leaky_relu(
            self.batch_norm4(self.conv4(x)),
            0.2
        )

        x = self.conv5(x)

        x = x.view(-1, self.conv_dim * 4 * 4)

        x = torch.sigmoid(self.fc(x))

        return x


# ==========================================
# GENERATOR
# ==========================================

class Generator(nn.Module):

    def __init__(self, z_size, conv_dim):

        super(Generator, self).__init__()

        self.conv_dim = conv_dim

        self.fc = nn.Linear(
            z_size,
            conv_dim * 4
        )

        self.t_conv1 = nn.ConvTranspose2d(
            conv_dim,
            conv_dim * 8,
            kernel_size=4,
            stride=2,
            padding=1,
            bias=False
        )

        self.batch_norm1 = nn.BatchNorm2d(conv_dim * 8)

        self.t_conv2 = nn.ConvTranspose2d(
            conv_dim * 8,
            conv_dim * 4,
            kernel_size=4,
            stride=2,
            padding=1,
            bias=False
        )

        self.batch_norm2 = nn.BatchNorm2d(conv_dim * 4)

        self.t_conv3 = nn.ConvTranspose2d(
            conv_dim * 4,
            conv_dim * 2,
            kernel_size=4,
            stride=2,
            padding=1,
            bias=False
        )

        self.batch_norm3 = nn.BatchNorm2d(conv_dim * 2)

        self.t_conv4 = nn.ConvTranspose2d(
            conv_dim * 2,
            3,
            kernel_size=4,
            stride=2,
            padding=1,
            bias=False
        )

    def forward(self, x):

        batch_size = x.shape[0]

        x = self.fc(x)

        x = x.view(
            batch_size,
            self.conv_dim,
            2,
            2
        )

        x = F.relu(
            self.batch_norm1(
                self.t_conv1(x)
            )
        )

        x = F.relu(
            self.batch_norm2(
                self.t_conv2(x)
            )
        )

        x = F.relu(
            self.batch_norm3(
                self.t_conv3(x)
            )
        )

        x = self.t_conv4(x)

        x = torch.tanh(x)

        return x


# ==========================================
# WEIGHT INIT
# ==========================================

def init_weights_normal(m):

    classname = m.__class__.__name__

    if hasattr(m, "weight") and (
            classname.find("Conv") != -1
            or classname.find("Linear") != -1):

        nn.init.normal_(
            m.weight.data,
            0.0,
            0.02
        )

    if hasattr(m, "bias") and m.bias is not None:

        nn.init.constant_(
            m.bias.data,
            0.0
        )


def build_GAN(
        d_conv_dim,
        g_conv_dim,
        z_size):

    D = Discriminator(d_conv_dim)

    G = Generator(
        z_size=z_size,
        conv_dim=g_conv_dim
    )

    D.apply(init_weights_normal)
    G.apply(init_weights_normal)

    return D, G


# ==========================================
# LOSS FUNCTIONS
# ==========================================

def real_loss(D_out, smooth=False):

    batch_size = D_out.size(0)

    if smooth:
        labels = torch.ones(batch_size) * 0.9
    else:
        labels = torch.ones(batch_size)

    if train_on_gpu:
        labels = labels.cuda()

    criterion = nn.BCELoss()

    loss = criterion(
        D_out.squeeze(),
        labels
    )

    return loss


def fake_loss(D_out):

    batch_size = D_out.size(0)

    labels = torch.zeros(batch_size)

    if train_on_gpu:
        labels = labels.cuda()

    criterion = nn.BCELoss()

    loss = criterion(
        D_out.squeeze(),
        labels
    )

    return loss


# ==========================================
# TRAINING
# ==========================================

def train(
        D,
        G,
        train_loader,
        n_epochs,
        z_size,
        g_optimizer,
        d_optimizer):

    if train_on_gpu:
        D.cuda()
        G.cuda()

    samples = []
    losses = []

    sample_size = 36

    fixed_z = np.random.uniform(
        -1,
        1,
        size=(sample_size, z_size)
    )

    fixed_z = torch.from_numpy(
        fixed_z
    ).float()

    if train_on_gpu:
        fixed_z = fixed_z.cuda()

    for epoch in range(n_epochs):

        for batch_i, (real_images, _) in enumerate(train_loader):

            batch_size = real_images.size(0)

            real_images = scale(real_images)

            if train_on_gpu:
                real_images = real_images.cuda()

            # ======================
            # TRAIN DISCRIMINATOR
            # ======================

            d_optimizer.zero_grad()

            D_real = D(real_images)

            d_real_loss = real_loss(D_real)

            z = np.random.uniform(
                -1,
                1,
                size=(batch_size, z_size)
            )

            z = torch.from_numpy(
                z
            ).float()

            if train_on_gpu:
                z = z.cuda()

            fake_images = G(z)

            D_fake = D(fake_images)

            d_fake_loss = fake_loss(D_fake)

            d_loss = d_real_loss + d_fake_loss

            d_loss.backward()

            d_optimizer.step()

            # ======================
            # TRAIN GENERATOR
            # ======================

            g_optimizer.zero_grad()

            z = np.random.uniform(
                -1,
                1,
                size=(batch_size, z_size)
            )

            z = torch.from_numpy(
                z
            ).float()

            if train_on_gpu:
                z = z.cuda()

            fake_images = G(z)

            D_fake = D(fake_images)

            g_loss = real_loss(
                D_fake,
                smooth=True
            )

            g_loss.backward()

            g_optimizer.step()

        print(
            f"Epoch [{epoch+1}/{n_epochs}] "
            f"D Loss: {d_loss.item():.4f} "
            f"G Loss: {g_loss.item():.4f}"
        )

        losses.append(
            (d_loss.item(), g_loss.item())
        )

        G.eval()

        with torch.no_grad():
            samples_z = G(fixed_z)

        samples.append(
            samples_z.cpu()
        )

        G.train()

    return losses, samples


# ==========================================
# MAIN
# ==========================================

if __name__ == "__main__":

    batch_size = 128
    image_size = 32

    anime_train_loader = get_data_loader(
        batch_size,
        image_size
    )

    d_conv_dim = 32
    g_conv_dim = 32
    z_size = 100

    D, G = build_GAN(
        d_conv_dim,
        g_conv_dim,
        z_size
    )

    train_on_gpu = torch.cuda.is_available()

    lr = 0.0005

    g_optimizer = optim.Adam(
        G.parameters(),
        lr=lr,
        betas=(0.3, 0.999)
    )

    d_optimizer = optim.Adam(
        D.parameters(),
        lr=lr,
        betas=(0.3, 0.999)
    )

    losses, samples = train(
        D,
        G,
        anime_train_loader,
        n_epochs=10,
        z_size=z_size,
        g_optimizer=g_optimizer,
        d_optimizer=d_optimizer
    )

    torch.save(
        G.state_dict(),
        "generator.pth"
    )

    torch.save(
        D.state_dict(),
        "discriminator.pth"
    )

    fig, axes = plt.subplots(
        nrows=6,
        ncols=6,
        figsize=(8, 8)
    )

    for ax, img in zip(
            axes.flatten(),
            samples[-1]):

        img = img.permute(1, 2, 0)

        img = (img + 1) / 2

        ax.imshow(img)
        ax.axis("off")

    plt.tight_layout()
    plt.show()
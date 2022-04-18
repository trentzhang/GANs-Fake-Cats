import torch
import torch.nn as nn
from spectral_normalization import SpectralNorm


class Discriminator(torch.nn.Module):
    def __init__(self, input_channels=3):
        super(Discriminator, self).__init__()

        # Hint: Hint: Apply spectral normalization to convolutional layers. Input to SpectralNorm should be your conv nn module
        ####################################
        #          YOUR CODE HERE          #
        ####################################
        ndf = 128
        self.network = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(input_channels, ndf, 4, 2, 1),
            nn.LeakyReLU(0.2),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(ndf * 8, 1, 4, 1, 0),
            nn.Sigmoid()
        )
        ##########       END      ##########

    def forward(self, x):

        ####################################
        #          YOUR CODE HERE          #
        ####################################

        ##########       END      ##########

        return self.network(x).squeeze()


class Generator(torch.nn.Module):
    def __init__(self, noise_dim, output_channels=3):
        super(Generator, self).__init__()
        self.noise_dim = noise_dim

        ####################################
        #          YOUR CODE HERE          #
        ####################################
        # self.conv1 = nn.ConvTranspose2d(noise_dim, 1024, 4, 1, 1)
        # self.conv2 = nn.ConvTranspose2d(1024, 512, 4, 2, 1)
        # self.conv3 = nn.ConvTranspose2d(512, 256, 4, 2, 1)
        # self.conv4 = nn.ConvTranspose2d(256, 128, 4, 2, 1)
        # self.conv5 = nn.ConvTranspose2d(128, 3, 4, 2, 1)
        ngf = 128
        self.network = nn.Sequential(
            nn.ConvTranspose2d(noise_dim, ngf * 8, 4, 1, 0),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1),
            nn.BatchNorm2d(ngf),
            nn.ReLU(),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d(ngf, output_channels, 4, 2, 1),
            nn.Tanh(),
            # nn.Threshold(threshold, 0)
            # state size. (nc) x 64 x 64
        )
        ##########       END      ##########

    def forward(self, x):

        ####################################
        #          YOUR CODE HERE          #
        ####################################
        # The output should be a 3x64x64 tensor for each sample (equal dimensions to the images from the dataset).

        ##########       END      ##########

        return self.network(x)

import torch
import torch.nn as nn


class Generator(nn.Module):
    def __init__(self, noise_dim, channels_num, hidden_dim):
        super(Generator, self).__init__()
        self._net = nn.Sequential(
            #  [N, noise_dim, 1, 1]
            nn.ConvTranspose2d(noise_dim, hidden_dim*16, kernel_size=4, stride=1, padding=0),
            nn.BatchNorm2d(hidden_dim*16),
            nn.ReLU(),
            #  [N, hidden_dim*16, 4, 4]
            nn.ConvTranspose2d(hidden_dim*16, hidden_dim*8, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(hidden_dim*8),
            nn.ReLU(),
            nn.ConvTranspose2d(hidden_dim * 8, hidden_dim * 4, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(hidden_dim * 4),
            nn.ReLU(),
            nn.ConvTranspose2d(hidden_dim * 4, hidden_dim * 2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(hidden_dim * 2),
            nn.ReLU(),
            nn.ConvTranspose2d(hidden_dim * 2, channels_num, kernel_size=4, stride=2, padding=1),
            #  [N, channels_num, 64,64]
            nn.Tanh(),
        )

    def forward(self, x):
        return self._net(x)
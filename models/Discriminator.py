import torch
import torch.nn as nn


class Discriminator(nn.Module):
    def __init__(self, channels_num, hidden_dim):
        '''
        :param channels_num: 1 or 3, depending RGB or not
        :param hidden_dim: hyprer parameter of the model
        '''
        super(Discriminator, self).__init__()
        self._net = nn.Sequential(
            #  [N,channels_num,64,64]
            nn.Conv2d(channels_num, hidden_dim, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2), # helping to stablize the convergence of GAN
            #  [N,channels_num,32,32]
            nn.Conv2d(hidden_dim, hidden_dim*2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(hidden_dim*2),  # helping to stablize the convergence of GAN
            nn.LeakyReLU(0.2), # helping to stablize the convergence of GAN
            nn.Conv2d(hidden_dim*2, hidden_dim*4, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(hidden_dim * 4),  # helping to stablize the convergence of GAN
            nn.LeakyReLU(0.2),  # helping to stablize the convergence of GAN
            nn.Conv2d(hidden_dim * 4, hidden_dim * 8, kernel_size=4, stride=2, padding=1),
            #  [N,hidden_dim*8,4,4]
            nn.BatchNorm2d(hidden_dim * 8),  # helping to stablize the convergence of GAN
            nn.LeakyReLU(0.2),  # helping to stablize the convergence of GAN
            nn.Conv2d(hidden_dim * 8, 1, kernel_size=4, stride=2, padding=0),
            # [N,1,1,1]
            nn.Sigmoid()
        )

    def forward(self, x):
        return self._net(x)
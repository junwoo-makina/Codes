import torch
import torch.nn as nn

# 64*212

class encoder(nn.Module):
    def __init__(self, n_channel):
        self.n_channel = n_channel

        self.encoder = nn.Sequential(
            nn.Conv2d(n_channel, 64, (4,8), 2, 1),
            nn.LeakyReLU(0.2, True),

            nn.Conv2d(64, 128, (4, 8), 2, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, True),

            nn.Conv2d(128, 256, (4, 8), 2, 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, True),

            nn.Conv2d(256, 512, (4, 8), 2, 1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, True),

            nn.Conv2d(512, 512, (4, 9), 1, 0),
            nn.LeakyReLU(0.2, True)
        )

        # init weights
        self.weight_init()

    def encode(self, x):
        return self.encoder(x)

    def forward(self, x):
        z = self.encode(x)
        return z

    def weight_init(self):
        self.encoder.apply(weight_init)


class decoder(nn.Module):
    class decoder(nn.Module):
        def __init__(self, n_channel):
            super().__init__()
            self.n_channel = n_channel
            # 2048
            self.decoder = nn.Sequential(
                nn.ConvTranspose2d(2048, 512, (4, 9), 1, 0),
                nn.BatchNorm2d(512),
                nn.LeakyReLU(0.2, True),

                nn.ConvTranspose2d(512, 256, (4, 9), 2, 1),
                nn.BatchNorm2d(256),
                nn.LeakyReLU(0.2, True),

                nn.ConvTranspose2d(256, 128, (4, 8), 2, 1),
                nn.BatchNorm2d(128),
                nn.LeakyReLU(0.2, True),

                nn.ConvTranspose2d(128, 64, (4, 8), 2, 1),
                nn.BatchNorm2d(64),
                nn.LeakyReLU(0.2, True),

                nn.ConvTranspose2d(64, self.n_channel, (4, 8), 2, 1),
                nn.Tanh()
            )

            # init weights
            self.weight_init()

        def decode(self, z):
            return self.decoder(z)

        def forward(self, z):
            return self.decode(z)

        def weight_init(self):
            self.decoder.apply(weight_init)

    # xavier_init
def weight_init(module):
    classname = module.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.xavier_normal(module.weight.data)
        # module.weight.data.normal_(0.0, 0.01)
    elif classname.find('BatchNorm') != -1:
        module.weight.data.normal_(1.0, 0.02)
        module.bias.data.fill_(0)
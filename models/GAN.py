from torch import nn

from configs.model_config import cfg as model_cfg


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


class LayerNorm2d(nn.Module):
    def __init__(self, num_channels: int, affine: bool = True):
        super().__init__()
        self.norm = nn.GroupNorm(num_channels, num_channels, affine=affine)

    def forward(self, x):
        return self.norm(x)


class Generator(nn.Module):
    def __init__(self, cfg):
        super(Generator, self).__init__()

        self.cfg = cfg

        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(self.cfg.nz, self.cfg.ngf * 8, 4, 1, 0, bias=False),
            # nn.BatchNorm2d(self.cfg.ngf * 8),
            LayerNorm2d(self.cfg.ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(self.cfg.ngf * 8, self.cfg.ngf * 4, 4, 2, 1, bias=False),
            # nn.BatchNorm2d(self.cfg.ngf * 4),
            LayerNorm2d(self.cfg.ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(self.cfg.ngf * 4, self.cfg.ngf * 2, 4, 2, 1, bias=False),
            # nn.BatchNorm2d(self.cfg.ngf * 2),
            LayerNorm2d(self.cfg.ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d(self.cfg.ngf * 2, self.cfg.ngf, 4, 2, 1, bias=False),
            # nn.BatchNorm2d(self.cfg.ngf),
            LayerNorm2d(self.cfg.ngf),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d(self.cfg.ngf, self.cfg.nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 64 x 64
        )

    def forward(self, input):
        return self.main(input)


class Discriminator(nn.Module):
    def __init__(self, cfg):
        super(Discriminator, self).__init__()

        self.cfg = cfg

        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(self.cfg.nc, self.cfg.ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(self.cfg.ndf, self.cfg.ndf * 2, 4, 2, 1, bias=False),
            # nn.BatchNorm2d(self.cfg.ndf * 2),
            LayerNorm2d(self.cfg.ngf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(self.cfg.ndf * 2, self.cfg.ndf * 4, 4, 2, 1, bias=False),
            # nn.BatchNorm2d(self.cfg.ndf * 4),
            LayerNorm2d(self.cfg.ngf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(self.cfg.ndf * 4, self.cfg.ndf * 8, 4, 2, 1, bias=False),
            # nn.BatchNorm2d(self.cfg.ndf * 8),
            LayerNorm2d(self.cfg.ngf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(self.cfg.ndf * 8, 1, 4, 1, 0, bias=False),
            # nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input)


def get_model(cfg):
    """
    Gets model.
    """
    netG = Generator(cfg)
    if cfg.device == 'cuda':
        netG = netG.cuda()
    netG.apply(weights_init)
    print(netG)

    netD = Discriminator(cfg)
    if cfg.device == 'cuda':
        netD = netD.cuda()
    netD.apply(weights_init)
    print(netD)

    return netG, netD

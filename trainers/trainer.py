from torch import nn
import torch
import os
from torchvision import utils as vutils
import time
import numpy as np

from dataloader.dataloader import get_dataloaders
from models.GAN import get_model
from utils.logging import Logger


class Trainer(object):
    def __init__(self, cfg):
        """
        Class for initializing and performing training procedure.
        :param cfg: train config
        """
        self.cfg = cfg
        self.dl_train, self.dl_test = get_dataloaders()
        self.netG, self.netD = get_model(cfg)
        self.criterion = self.get_criterion()
        self.optimizerD, self.optimizerG = self.get_optimizer()
        self.logger = Logger(self.cfg)
        self.sigmoid = torch.nn.Sigmoid()

    @staticmethod
    def get_criterion():
        """
        Gets criterion.
        :return: criterion
        """
        criterion = nn.BCELoss()
        return criterion

    def get_optimizer(self):
        """
        Gets optimizer.
        :return: optimizer
        """
        optimizerD = torch.optim.Adam(self.netD.parameters(), lr=self.cfg.lr, betas=(self.cfg.beta1, self.cfg.beta2))
        optimizerG = torch.optim.Adam(self.netG.parameters(), lr=self.cfg.lr, betas=(self.cfg.beta1, self.cfg.beta2))
        return optimizerD, optimizerG

    def restore_model(self, model, optimizer, net='generator'):
        """
        Restores saved model.
        """
        if self.cfg.load_saved_model:
            print(f'Trying to load checkpoint for net {net} from epoch {self.cfg.epoch_to_load}...')
            try:
                checkpoint = torch.load(self.cfg.checkpoints_dir + f'/{net}_checkpoint_{self.cfg.epoch_to_load}.pth')
                load_state_dict = checkpoint['model']
                model.load_state_dict(load_state_dict)
                self.start_epoch = checkpoint['epoch'] + 1
                self.global_step = checkpoint['global_step'] + 1
                optimizer.load_state_dict(checkpoint['opt'])
                print(f'Loaded checkpoint for net {net} from epoch {self.cfg.epoch_to_load}.')
            except FileNotFoundError:
                print('Checkpoint not found')
        return model, optimizer

    def save_model(self, model, optimizer, net='generator'):
        """
        Saves model.
        """
        if self.cfg.save_model and self.epoch % self.cfg.epochs_saving_freq == 0:
            print('Saving current model...')
            state = {
                'model': model.state_dict(),
                'epoch': self.epoch,
                'global_step': self.global_step,
                'opt': optimizer.state_dict()
            }
            if not os.path.exists(self.cfg.checkpoints_dir):
                os.makedirs(self.cfg.checkpoints_dir)

            path_to_save = os.path.join(self.cfg.checkpoints_dir, f'{net}_checkpoint_{self.epoch}.pth')
            torch.save(state, path_to_save)
            print(f'Saved {net} model to {path_to_save}.')

    def make_training_step(self, batch):
        """
        Makes single training step.
        :param batch: current batch containing input vector and it`s label
        :return: loss on current batch
        """
        input_vector, label = batch[0].cuda(), batch[1].cuda()
        self.optimizer.zero_grad()
        out = self.model(input_vector)
        loss = self.criterion(out, label)
        assert not torch.isnan(loss)
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def calc_gradient_penalty(self, real_data, fake_data):
        batch_size = real_data.size(0)
        eps = torch.rand(batch_size, 1, 1, 1).to(real_data.device)
        eps = eps.expand_as(real_data)
        interpolation = eps * real_data + (1 - eps) * fake_data

        interp_logits = self.netD(interpolation)
        grad_outputs = torch.ones_like(interp_logits)

        gradients = torch.autograd.grad(
            outputs=interp_logits,
            inputs=interpolation,
            grad_outputs=grad_outputs,
            create_graph=True,
            retain_graph=True,
        )[0]

        gradients = gradients.view(batch_size, -1)
        grad_norm = gradients.norm(2, 1)
        return torch.mean((grad_norm - 1) ** 2)

    def train(self):
        """
        Runs training procedure.
        """
        total_training_start_time = time.time()
        self.start_epoch, self.epoch, self.global_step = 0, -1, 0

        # restore model if necessary
        self.netD, self.optimizerD = self.restore_model(self.netD, self.optimizerD, net='discriminator')
        self.netG, self.optimizerG = self.restore_model(self.netG, self.optimizerG, net='generator')

        # Create batch of latent vectors that we will use to visualize
        #  the progression of the generator
        fixed_noise = torch.randn(self.cfg.batch_size, self.cfg.nz, 1, 1, device=self.cfg.device)

        # Establish convention for real and fake labels during training
        real_label = torch.full((self.cfg.batch_size,), 1., dtype=torch.float, device=self.cfg.device)
        fake_label = torch.full((self.cfg.batch_size,), 0., dtype=torch.float, device=self.cfg.device)

        G_losses = []
        D_losses = []
        D_accs = []
        D_accs_fake = []
        D_accs_real = []

        # start training
        print(f'Starting training...')
        iter_num = len(self.dl_train)
        for epoch in range(self.start_epoch, self.cfg.epochs):
            epoch_start_time = time.time()
            self.epoch = epoch
            print(f'Epoch: {self.epoch}/{self.cfg.epochs}')

            for i, batch in enumerate(self.dl_train):
                # break
            # while True:

                for _ in range(5):
                    self.netG.eval()
                    self.netD.train()
                    self.optimizerD.zero_grad()

                    real = batch[0].float().to(self.cfg.device)
                    noise = torch.randn(real.size(0), self.cfg.nz, 1, 1, device=self.cfg.device)
                    fake = self.netG(noise)
                    real_logits = self.netD(real)
                    fake_logits = self.netD(fake)

                    gradient_penalty = self.cfg.w_gp * self.calc_gradient_penalty(real, fake)

                    loss_c = fake_logits.mean() - real_logits.mean()

                    fake_logits_sigmoid = self.sigmoid(fake_logits)
                    fake_acc = torch.sum(torch.round(fake_logits_sigmoid.view(-1)) == fake_label) / self.cfg.batch_size
                    real_logits_sigmoid = self.sigmoid(real_logits)
                    real_acc = torch.sum(torch.round(real_logits_sigmoid.view(-1)) == real_label) / self.cfg.batch_size
                    acc = ((fake_acc + real_acc) / 2).item()
                    D_accs.append(acc)
                    D_accs_fake.append(fake_acc.item())
                    D_accs_real.append(real_acc.item())

                    loss_d = loss_c + gradient_penalty

                    loss_d.backward()
                    self.optimizerD.step()
                    self.logger.log_metrics(['lossD', 'gradient_penalty', 'acc', 'fake_acc', 'real_acc'],
                                            [loss_d.item(),
                                             gradient_penalty.item(), acc, fake_acc.item(), real_acc.item()],
                                            self.global_step)
                    D_losses.append(loss_d.item())

                self.netG.train()
                self.netD.eval()

                self.optimizerG.zero_grad()
                noise = torch.randn(real.size(0), self.cfg.nz, 1, 1, device=self.cfg.device)
                fake = self.netG(noise)
                fake_logits = self.netD(fake)
                loss_g = -fake_logits.mean().view(-1)

                loss_g.backward()
                self.optimizerG.step()

                self.logger.log_metrics(["lossG"], [loss_g.item()], self.global_step)
                G_losses.append(loss_g.item())

                if i % 50 == 0:
                    mean_acc_fake = np.mean(D_accs_fake[-10:]) if len(D_accs_fake) > 10 else np.mean(D_accs_fake)
                    mean_acc_real = np.mean(D_accs_real[-10:]) if len(D_accs_real) > 10 else np.mean(D_accs_real)
                    mean_loss_d = np.mean(D_losses[-10:]) if len(D_losses) > 10 else np.mean(D_losses)
                    mean_loss_g = np.mean(G_losses[-10:]) if len(G_losses) > 10 else np.mean(G_losses)

                    print(f'Epoch: {epoch}/{self.cfg.epochs}, iter:{i}/{iter_num}, '
                          f'Loss_D: {mean_loss_d}, Loss_G: {mean_loss_g}, acc fake: {mean_acc_fake}, acc real: {mean_acc_real}')  # , D(x): {D_x}\tD(G(z)): {D_G_z1} /
                    # {D_G_z2}

                self.logger.log_metrics(['loss_D', 'loss_G'], [loss_d.item(), loss_g.item()], self.global_step)

                if i % 500 == 0:
                    with torch.no_grad():
                        fake = self.netG(fixed_noise).detach().cpu()
                    vutils.save_image(fake,  # torch.stack([real[0].unsqueeze((1)).cpu(), fake], 1).squeeze(0)
                                      f'../plots/epoch_{epoch}_iter_{i}.png')
                self.global_step += 1

            # save model
            self.save_model(self.netD, self.optimizerD, net='discriminator')
            self.save_model(self.netG, self.optimizerG, net='generator')

            print(f'Epoch total time: {round((time.time() - epoch_start_time) / 60, 3)} min')

        print(f'Training time: {round((time.time() - total_training_start_time) / 60, 3)} min')

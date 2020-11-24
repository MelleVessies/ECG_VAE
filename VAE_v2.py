"""
This module implements a Convolutional Neural Network in PyTorch.
You should fill in code into indicated sections.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from argparse import ArgumentParser

import pytorch_lightning as pl

import torch
import torch.nn as nn

from pytorch_lightning.callbacks import Callback

from data_util import PTBXLDataModule

import numpy as np
import wfdb
import io
import PIL.Image
from torchvision.transforms import ToTensor
import torchvision

from pytorch_lightning.loggers import TensorBoardLogger
logger = TensorBoardLogger('lightning_logs', name='TBXL_VAE_v2')


class VAEEncoder(nn.Module):
    """
        VAE Encoder module
    """
    def __init__(self, dims, in_channels:int = 12):
        super(VAEEncoder, self).__init__()
        self.layers = []

        for dim in dims:
            self.layers.append(nn.Sequential(
                nn.Conv1d(in_channels, dim, 3, stride=1, padding=1),
                nn.BatchNorm1d(dim),
                nn.ReLU(),
                nn.MaxPool1d(padding=1, dilation=2, kernel_size=3)
            ))
            in_channels = dim

        self.layers.append(nn.Flatten())

        self.layers = nn.Sequential(*self.layers)

    def forward(self, x):
        for l in self.layers:
            print(x.shape)
            x = l.forward(x)

        print(x.shape)
        exit()
        return x

        # return self.layers.forward(x)

class VAEDecoder(nn.Module):
    """
        VAE Decoder module
    """
    def __init__(self, inversed_dims, latent_dim, out_channels:int = 12):
        super(VAEDecoder, self).__init__()
        self.layers = []
        in_channels = inversed_dims[0]
        for dim in inversed_dims[1:]:
            self.layers.append(nn.Sequential(
                nn.ConvTranspose1d(in_channels, dim, 3, padding=1, stride=1),
                nn.BatchNorm1d(dim),
                nn.ReLU(),
            ))
            in_channels = dim

        self.layers.append(nn.ConvTranspose1d(inversed_dims[-1], out_channels, 3, padding=1, stride=1))

        self.layers = nn.Sequential(*self.layers)

    def forward(self, x):
        for l in self.layers:
            # print(x.shape)
            x = l.forward(x)

        # print(x.shape)
        return x

    # def forward(self, x):
    #     return self.layers.forward(x)

class VAE(pl.LightningModule):
    """
    Standard VAE with Gaussian Prior and approx posterior.
    """

    def __init__(
        self,
        kl_coeff: float = 0.1,
        latent_dim: int = 256,
        lr: float = 1e-4,
        sample_dim:int = 1000,
        **kwargs
    ):
        """
        Args:
            kl_coeff: coefficient for kl term of the loss
            latent_dim: dim of latent space
            lr: learning rate for Optimizer
            sample_dim:
        """

        super(VAE, self).__init__()

        self.save_hyperparameters()

        self.kl_coeff = kl_coeff
        self.latent_dim = latent_dim
        self.lr = lr
        self.sample_dim = sample_dim

        self.hidden_dims = [32, 64, 128, 256]

        self.enc_output_data_dim = sample_dim

        self.fc_mu = nn.Linear(self.hidden_dims[-1]*self.enc_output_data_dim, self.latent_dim)
        self.fc_var = nn.Linear(self.hidden_dims[-1]*self.enc_output_data_dim, self.latent_dim)

        self.encoder = VAEEncoder(self.hidden_dims)

        self.decoder_input = nn.Linear(latent_dim, self.hidden_dims[-1] * self.enc_output_data_dim)
        self.decoder = VAEDecoder(list(reversed(self.hidden_dims)), self.latent_dim)

    def forward(self, x):
        x = self.encoder(x)
        mu = self.fc_mu(x)
        log_var = self.fc_var(x)
        p, q, z = self.sample(mu, log_var)

        z = self.decoder_input(z)
        z = z.reshape(-1, self.hidden_dims[-1], self.sample_dim)

        return self.decoder(z)

    def _run_step(self, x):
        x = self.encoder(x)
        mu = self.fc_mu(x)
        log_var = self.fc_var(x)
        p, q, z = self.sample(mu, log_var)

        d_z = self.decoder_input(z)
        d_z = d_z.reshape(d_z.shape[0], self.hidden_dims[-1], self.enc_output_data_dim)

        return z, self.decoder(d_z), p, q

    def sample(self, mu, log_var):
        std = torch.exp(log_var / 2)
        p = torch.distributions.Normal(torch.zeros_like(mu), torch.ones_like(std))
        q = torch.distributions.Normal(mu, std)
        z = q.rsample()

        return p, q, z

    def step(self, batch, batch_idx):
        x, y = batch
        z, x_hat, p, q = self._run_step(x)

        recon_loss = torch.functional.F.mse_loss(x_hat, x, reduction='mean')

        log_qz = q.log_prob(z)
        log_pz = p.log_prob(z)

        kl = log_qz - log_pz
        kl = kl.mean()
        kl *= self.kl_coeff

        self.log("Recon loss", recon_loss, prog_bar=True)
        self.log("KL loss", kl, prog_bar=True)

        loss = kl + recon_loss

        logs = {
            "recon_loss": recon_loss,
            "kl": kl,
            "loss": loss,
        }
        return loss, logs

    def training_step(self, batch, batch_idx):
        loss, logs = self.step(batch, batch_idx)
        self.log_dict(
            {f"train_{k}": v for k, v in logs.items()}, on_step=True, on_epoch=False
        )
        return loss

    def validation_step(self, batch, batch_idx):
        loss, logs = self.step(batch, batch_idx)
        self.log_dict({f"val_{k}": v for k, v in logs.items()})
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)

        parser.add_argument("--lr", type=float, default=1e-6)

        parser.add_argument("--kl_coeff", type=float, default=0.1)
        parser.add_argument("--latent_dim", type=int, default=256)

        parser.add_argument("--batch_size", type=int, default=256)
        parser.add_argument("--num_workers", type=int, default=8)
        parser.add_argument("--data_dir", type=str, default=".")

        return parser

def figureToTensor(figure):
    buf = io.BytesIO()

    figure.savefig(buf, format='jpeg')
    buf.seek(0)

    image = PIL.Image.open(buf)

    return ToTensor()(image)


class ReconstructionPlottingCallback(Callback):
    def on_validation_end(self, trainer:pl.Trainer, model:VAE):
        test_sample = np.load("./PTB_XL/records100/21000/21477_lr.npy", allow_pickle=True)

        original = wfdb.plot.plot_items(test_sample, return_fig=True)

        test_sample = torch.from_numpy(test_sample.swapaxes(0, 1)).float().unsqueeze(dim=0).to('cuda:0')
        reconstruction_raw = model.forward(test_sample)

        reconstruction = wfdb.plot.plot_items(reconstruction_raw.squeeze(dim=0).transpose(0, 1).cpu().detach().numpy(),
                                              return_fig=True)

        grid = torchvision.utils.make_grid([figureToTensor(original), figureToTensor(reconstruction)])

        tensorboard = trainer.logger.experiment
        tensorboard.add_image("Orginal/Reconstruction", grid)


def cli_main(args=None):
    pl.seed_everything(42)

    parser = ArgumentParser()
    script_args, _ = parser.parse_known_args(args)

    dm = PTBXLDataModule()


    parser = VAE.add_model_specific_args(parser)
    parser = pl.Trainer.add_argparse_args(parser)
    args = parser.parse_args(args)

    if args.max_steps == -1:
        args.max_steps = None

    model = VAE(**vars(args))

    trainer = pl.Trainer.from_argparse_args(
        args,
        callbacks=[ReconstructionPlottingCallback()],
        logger=logger)
    trainer.fit(model, dm)
    return dm, model, trainer


if __name__ == "__main__":
    dm, model, trainer = cli_main()
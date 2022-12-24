import torch
from torch import nn
from torch.distributions import Normal
import numpy as np


class BModule(nn.Module):

    def init(self):
        super().__init__()

    def forward(self, x):
        raise NotImplementedError()

    def elbo_estimator(self, data, target, n_samples, criterion, len_dataset, batch_size, batch_idx=None, weight_type='Graves'):
        loss = 0.
        for _ in range(n_samples):
            outputs = self(data)
            loss += criterion(outputs, target)
            loss += self.kl_divergence() * self.complexity_cost_weight(len_dataset,
                                                                       batch_size,
                                                                       batch_idx,
                                                                       weight_type)
        return loss / n_samples

    def kl_divergence(self):
        kl_divergence = 0.
        for _, layer in self._modules.items():
            kl_divergence += layer.log_variational_posterior - layer.log_prior
        return kl_divergence

    def complexity_cost_weight(self, len_dataset, batch_size, batch_idx=None, weight_type='Graves'):
        M = int(len_dataset / batch_size)
        if weight_type == 'Graves':
            return 1 / (M * batch_size)
        elif weight_type == 'Blundell':
            pi_i = (2 ** (M - batch_idx)) / (2 ** M - 1)
            return pi_i / batch_size
        else:
            raise NotImplementedError()


class Scale_Mixture_Prior(nn.Module):
    def __init__(self, log_sigma_1, log_sigma_2, pi):
        super().__init__()
        self.pi = pi

        self.dist_1 = Normal(0, np.exp(log_sigma_1))
        self.dist_2 = Normal(0, np.exp(log_sigma_2))

    def log_prior(self, sample):
        log_prob_1 = torch.exp(self.dist_1.log_prob(sample))
        log_prob_2 = torch.exp(self.dist_2.log_prob(sample))

        log_mix_prior = torch.log(
            self.pi * log_prob_1 + (1 - self.pi) * log_prob_2 + 1e-6)
        return log_mix_prior.sum()

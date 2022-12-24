import torch
from torch import nn
from torch.nn import functional as F
from zhusuan.bn.bayesian_module import BModule, Scale_Mixture_Prior
from zhusuan.distributions import Normal


class BLinear(BModule):
    def __init__(self,
                 in_features,
                 out_features,
                 bias=True,
                 prior_log_sigma_1=-1,
                 prior_log_sigma_2=-7,
                 prior_pi=0.5,
                 posterior_mu_initial=0,
                 posterior_rho_initial=-3):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.bias = bias
        self.log_prior = 0.
        self.log_variational_posterior = 0.

        self.weight_prior_distribution = Scale_Mixture_Prior(
            prior_log_sigma_1, prior_log_sigma_2, prior_pi)
        self.weight_mu = nn.Parameter(torch.Tensor(
            out_features, in_features).normal_(posterior_mu_initial, 0.1))
        self.weight_rho = nn.Parameter(torch.Tensor(
            out_features, in_features).normal_(posterior_rho_initial, 0.1))

        if bias:
            self.bias_prior_distribution = Scale_Mixture_Prior(
                prior_log_sigma_1, prior_log_sigma_2, prior_pi)
            self.bias_mu = nn.Parameter(torch.Tensor(
                out_features).normal_(posterior_mu_initial, 0.1))
            self.bias_rho = nn.Parameter(torch.Tensor(
                out_features).normal_(posterior_rho_initial, 0.1))

    def forward(self, x):

        weight_distribution = Normal(mean=self.weight_mu, std=torch.log1p(
            torch.exp(self.weight_rho)), device=x.device)
        weight = weight_distribution._sample()
        weight_log_posterior = weight_distribution._log_prob().sum()
        weight_log_prior = self.weight_prior_distribution.log_prior(weight)

        if self.bias:
            bias_distribution = Normal(mean=self.bias_mu, std=torch.log1p(
                torch.exp(self.bias_rho)), device=x.device)
            bias = bias_distribution._sample()
            bias_log_posterior = bias_distribution._log_prob().sum()
            bias_log_prior = self.bias_prior_distribution.log_prior(bias)
        else:
            bias = torch.zeros((self.out_features), device=x.device)
            bias_log_posterior = 0
            bias_log_prior = 0

        # Complexity Cost
        self.log_variational_posterior = weight_log_posterior + bias_log_posterior
        self.log_prior = weight_log_prior + bias_log_prior

        return F.linear(x, weight, bias)

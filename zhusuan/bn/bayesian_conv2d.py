import torch
from torch import nn
from torch.nn import functional as F
from zhusuan.bn.bayesian_module import BModule, Scale_Mixture_Prior
from zhusuan.distributions import Normal


class BConv2d(BModule):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 bias=True,
                 prior_log_sigma_1=-1,
                 prior_log_sigma_2=-7,
                 prior_pi=0.5,
                 posterior_mu_initial=0,
                 posterior_rho_initial=-3):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.bias = bias
        self.log_prior = 0.
        self.log_variational_posterior = 0.

        self.weight_prior_distribution = Scale_Mixture_Prior(
            prior_log_sigma_1, prior_log_sigma_2, prior_pi)
        self.weight_mu = nn.Parameter(torch.Tensor(
            out_channels, in_channels // groups, *kernel_size).normal_(posterior_mu_initial, 0.1))
        self.weight_rho = nn.Parameter(torch.Tensor(
            out_channels, in_channels // groups, *kernel_size).normal_(posterior_rho_initial, 0.1))

        if bias:
            self.bias_prior_distribution = Scale_Mixture_Prior(
                prior_log_sigma_1, prior_log_sigma_2, prior_pi)
            self.bias_mu = nn.Parameter(torch.Tensor(
                out_channels).normal_(posterior_mu_initial, 0.1))
            self.bias_rho = nn.Parameter(torch.Tensor(
                out_channels).normal_(posterior_rho_initial, 0.1))

    def forward(self, x):

        weight_distribution = Normal(mean=self.weight_mu, std=torch.log1p(
            torch.exp(self.weight_rho)), device=x.device)
        weight = weight_distribution._sample()
        weight_log_prior = self.weight_prior_distribution.log_prior(weight)

        if self.bias:
            bias_distribution = Normal(mean=self.bias_mu, std=torch.log1p(
                torch.exp(self.bias_rho)), device=x.device)
            bias = bias_distribution._sample()
            bias_log_prior = self.bias_prior_distribution.log_prior(bias)
        else:
            bias = torch.zeros((self.out_channels), device=x.device)
            bias_log_prior = 0.

        self.log_prior = weight_log_prior + bias_log_prior

        if self.is_variational:
            weight_log_posterior = weight_distribution._log_prob().sum()
            if self.bias:
                bias_log_posterior = bias_distribution._log_prob().sum()
            else:
                bias_log_posterior = 0.
            self.log_variational_posterior = weight_log_posterior + bias_log_posterior

        return F.conv2d(input=x,
                        weight=weight,
                        bias=bias,
                        stride=self.stride,
                        padding=self.padding,
                        dilation=self.dilation,
                        groups=self.groups)

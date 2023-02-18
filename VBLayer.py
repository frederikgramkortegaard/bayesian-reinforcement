""" Implementation of a Bayesian layer for a VB network. """

import math
import numpy as np
import torch
import torch.nn as nn


class VBLinear(nn.Module):
    def __init__(
        self, in_features, out_features, prior_prec=10, map=True, use_cuda=False
    ):
        super(VBLinear, self).__init__()
        self.n_in = in_features
        self.n_out = out_features

        self.prior_prec = prior_prec
        self.map = map
        self.training = True
        self.device = torch.device("cpu") if not use_cuda else torch.device("cuda")

        self.bias = nn.parameter.Parameter(torch.FloatTensor(out_features)).to(
            self.device
        )
        self.mu_w = nn.parameter.Parameter(
            torch.FloatTensor(out_features, in_features)
        ).to(self.device)
        self.logsig2_w = nn.parameter.Parameter(
            torch.FloatTensor(out_features, in_features)
        ).to(self.device)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.mu_w.size(1))
        self.mu_w.data.normal_(0, stdv)
        self.logsig2_w.data.zero_().normal_(-9, 0.001)  # var init via Louizos
        self.bias.data.zero_()

    def KL(self, loguniform=False):
        if loguniform:
            k1 = 0.63576
            k2 = 1.87320
            k3 = 1.48695
            log_alpha = self.logsig2_w - 2 * torch.log(self.mu_w.abs() + 1e-8)
            kl = torch.sum(
                k1 * torch.sigmoid(k2 + k3 * log_alpha)
                - 0.5 * torch.nn.functional.softplus(-log_alpha)
                - k1
            )
        else:
            logsig2_w = self.logsig2_w.clamp(-11, 11)
            kl = (
                0.5
                * (
                    self.prior_prec * (self.mu_w.pow(2) + logsig2_w.exp())
                    - logsig2_w
                    - 1
                    - np.log(self.prior_prec)
                ).sum()
            )
        return kl

    def forward(self, input):
        # Sampling free forward pass only if MAP prediction and no training rounds
        if self.map and not self.training:
            return torch.nn.functional.linear(input, self.mu_w, self.bias)
        else:
            mu_out = torch.nn.functional.linear(input, self.mu_w, self.bias)
            logsig2_w = self.logsig2_w.clamp(-11, 11)
            s2_w = logsig2_w.exp()
            var_out = torch.nn.functional.linear(input.pow(2), s2_w) + 1e-8
            return mu_out + var_out.sqrt() * torch.randn_like(mu_out)

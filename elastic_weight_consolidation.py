# Credits: https://github.com/shivamsaboo17/Overcoming-Catastrophic-forgetting-in-Neural-Networks
# The following code is taken from the research of the paper https://arxiv.org/abs/1612.00796
# The code has also been altered to fit into our specific problem set
# As per defualt it is set up for classic image classification tasks
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch import autograd
import numpy as np
from torch.utils.data import DataLoader


class ElasticWeightConsolidation:
    def __init__(self, model, crit, kl_weight, lr=0.001, weight=1000000):
        self.model = model
        # print(model)
        self.weight = weight
        self.kl_weight = kl_weight
        self.crit = crit
        self.optimizer = optim.Adam(self.model.parameters(), lr)

    def _update_mean_params(self):
        for param_name, param in self.model.named_parameters():
            _buff_param_name = param_name.replace(".", "__")
            self.model.register_buffer(
                _buff_param_name + "_estimated_mean", param.data.clone()
            )

    def _update_fisher_params(self, current_ds, batch_size, num_batch):
        dl = DataLoader(current_ds, batch_size, shuffle=True)
        log_liklihoods = []

        for i in range(num_batch):
            data = current_ds[i]
            input = data[0]
            # print(input.shape)
            states, actions = input
            target = data[1]
            output = F.log_softmax(self.model(states, actions), dim=1)
            log_liklihoods.append(output[:, target.long()])

        log_likelihood = torch.cat(log_liklihoods).mean()
        # print(log_likelihood)
        grad_log_liklihood = autograd.grad(
            log_likelihood,
            self.model.parameters(),
            retain_graph=True,
            allow_unused=True,
        )
        _buff_param_names = [
            param[0].replace(".", "__") for param in self.model.named_parameters()
        ]
        for _buff_param_name, param in zip(_buff_param_names, grad_log_liklihood):
            # print(param.data.clone())
            if param != None:
                self.model.register_buffer(
                    _buff_param_name + "_estimated_fisher", param.data.clone() ** 2
                )

    def register_ewc_params(self, dataset, batch_size, num_batches):
        self._update_fisher_params(dataset, batch_size, num_batches)
        self._update_mean_params()

    def _compute_consolidation_loss(self, weight):
        try:
            losses = []
            for param_name, param in self.model.named_parameters():
                _buff_param_name = param_name.replace(".", "__")
                estimated_mean = getattr(
                    self.model, "{}_estimated_mean".format(_buff_param_name)
                )
                estimated_fisher = getattr(
                    self.model, "{}_estimated_fisher".format(_buff_param_name)
                )
                losses.append((estimated_fisher * (param - estimated_mean) ** 2).sum())
            return (weight / 2) * sum(losses)
        except AttributeError:
            return 0

    def forward_backward_update(self, input, target):
        states, actions = input
        output = self.model(states, actions)
        loss = self._compute_consolidation_loss(self.weight) + self.crit(output, target)
        loss += ((self.model.bayes.KL() + self.model.firsl.KL() + self.model.lin1.KL() + self.model.alin1.KL()) / 4) 
        self.optimizer.zero_grad()
        loss.backward(retain_graph=True)
        torch.nn.utils.clip_grad_value_(self.model.parameters(), 1)
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 50)
        self.optimizer.step()

    def save(self, filename):
        torch.save(self.model, filename)

    def load(self, filename):
        self.model = torch.load(filename)

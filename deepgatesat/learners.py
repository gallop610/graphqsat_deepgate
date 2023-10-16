import torch
from torch import nn
from torch.optim.lr_scheduler import StepLR
from minisat.minisat.gym.MiniSATEnv import VAR_ID_IDX


class CircuitLearner:
    def __init__(self, net, target, buffer, args):
        self.buffer = buffer
        self.net = net
        self.target = target
        self.target.eval()

        self.optimizer = torch.optim.Adam(self.net.parameters(), lr = 2e-5)

        self.loss = nn.MSELoss()

        self.bsize = 4
        self.gamma = 0.99
        self.target_update_freq = 10
        self.step_ctr = 0
        self.grad_clip = 1.0
        self.grad_clip_norm_type = 2
        self.device = args.device

    def get_qs(self, states):
        qs_value = self.net(states)

        return qs_value

    def get_target_qs(self, states):
        target_qs_value = self.target(states)

        return target_qs_value

    def step(self):
        s, a, r, s_next, nonterminals = self.buffer.sample(self.bsize)

        target_qs = self.get_target_qs(s_next)

        qs = self.qs(s)

        # self.net.train()


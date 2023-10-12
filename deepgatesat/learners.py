import torch
from torch import nn
from torch.optim.lr_scheduler import StepLR
from minisat.minisat.gym.MiniSATEnv import VAR_ID_IDX


class CircuitLearner:
    def __init__(self, net, buffer, args):
        self.buffer = buffer
        self.net = net

        self.optimizer = torch.optim.Adam(self.net.parameters(), lr = 2e-5)

        self.loss = nn.MSELoss()

        self.bsize = 4
        self.step_ctr = 0
        self.grad_clip = 1.0
        self.grad_clip_norm_type = 2
        self.device = args.device

    def step(self):
        s, a, r, s_next, nonterminals = self.buffer.sample(self.bsize)
        
        self.net.train()
        
        


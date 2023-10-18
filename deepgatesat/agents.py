import torch
import numpy as np
from minisat.minisat.gym.MiniSATEnv import VAR_ID_IDX
import torch.nn as nn


class CircuitAgent:
    def __init__(self, ckt_net, args):
        self.net = ckt_net
        self.device = args.device

    def forward(self, graph):
        return self.net(graph)

    def act(self, hist_buffer):
        graph = hist_buffer[-1]
        if np.random.random() < 1:
            acts = range(len(graph.valid_mask)*2)
            return int(np.random.choice(acts))
        else:
            qs = self.forward(graph)[graph.valid_mask, :]
            return self.choose_actions(qs)

    def choose_actions(self, qs):
        return qs.flatten().argmax().item()
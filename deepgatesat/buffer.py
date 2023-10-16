import torch
import numpy as np

class CircuitBuffer:
    def __init__(self, args, size):
        self.ctr = 0
        self.full = False
        self.size = size
        self.device = args.device
        self.dones = torch.ones(size)
        self.rewards = torch.zeros(size)
        self.actions = torch.zeros(size, dtype=torch.long)
        self.observations = np.zeros((size, 4), dtype=object)

    def add_transition(self, obs, a, r_next, done_next):
        self.dones[self.ctr] = int(done_next)
        self.rewards[self.ctr] = r_next
        self.actions[self.ctr] = a

        # should be vertex_data, edge_data, connectivity, global
        for el_idx, el in enumerate(obs):
            self.observations[self.ctr][el_idx] = el

        if (self.ctr + 1) % self.size == 0:
            self.ctr = 0
            self.full = True
        else:
            self.ctr += 1
    
    def sample(self, batch_size):
        # to be able to grab the next, we use -1
        curr_size = self.ctr - 1 if not self.full else self.size - 1
        idx = np.random.choice(range(0, curr_size), batch_size)
        return (
            self.batch(self.observations[idx]),
            self.actions[idx].to(self.device),
            self.rewards[idx].to(self.device),
            self.batch(self.observations[idx + 1]),
            1.0 - self.dones[idx].to(self.device),
        )
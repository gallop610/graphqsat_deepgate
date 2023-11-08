import torch
from torch import nn
from torch.optim.lr_scheduler import StepLR
from minisat.minisat.gym.MiniSATEnv import VAR_ID_IDX
from torch_scatter import scatter_max

class CircuitLearner:
    def __init__(self, net, target, buffer, args):
        self.net = net
        self.target = target
        self.target.eval()

        self.optimizer = torch.optim.Adam(self.net.parameters(), lr = 2e-5)
        self.lr_scheduler = StepLR(self.optimizer, step_size=args.lr_scheduler_frequency, gamma=args.lr_scheduler_gamma)

        self.loss = nn.MSELoss()

        self.batch_size = args.batch_size
        self.gamma = 0.99
        self.buffer = buffer
        self.target_update_freq = 10
        self.step_ctr = 0
        self.grad_clip = 1.0
        self.grad_clip_norm_type = 2
        self.device = args.device

    def get_qs(self, states):
        qs_value = self.net(states)
        vertex_sizes = torch.tensor([len(aig.valid_mask) for aig in states])

        return qs_value.to(self.device), vertex_sizes

    def get_target_qs(self, states):
        target_qs_value = self.target(states)
        target_vertex_sizes = torch.tensor([len(aig.valid_mask) for aig in states])

        return target_qs_value.to(self.device).detach(), target_vertex_sizes

    def step(self):
        s, a, r, s_next, nonterminals = self.buffer.sample(self.batch_size)

        with torch.no_grad():
            target_qs, target_vertex_sizes = self.get_target_qs(s_next)
            idx_for_scatter = [
                [idx] * el.item() * 2 for idx, el in enumerate(target_vertex_sizes)
            ]
            idx_for_scatter = torch.tensor(
                [el for subl in idx_for_scatter for el in subl],
                dtype=torch.long,
                device=self.device
            ).flatten()

            target_qs = scatter_max(target_qs.flatten(), idx_for_scatter, dim=0)[0]
            targets = r + nonterminals * self.gamma * target_qs
 
        self.net.train()
        qs, vertex_sizes  = self.get_qs(s)

        gather_idx = (vertex_sizes * qs.shape[1]).cumsum(0).roll(1).to(self.device)
        gather_idx[0] = 0

        qs = qs.flatten()[gather_idx + a]

        loss = self.loss(qs, targets)

        self.optimizer.zero_grad()
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.net.parameters(), self.grad_clip, norm_type=self.grad_clip_norm_type
        )
        self.optimizer.step()

        if not self.step_ctr % self.target_update_freq:
            self.target.load_state_dict(self.net.state_dict())

        self.step_ctr += 1

        lr = self.lr_scheduler.get_lr()[0]
        self.lr_scheduler.step()

        return {
            "loss": loss.item(),
            "grad_norm": grad_norm,
            "lr": lr,
            "average_q": qs.mean()
        }
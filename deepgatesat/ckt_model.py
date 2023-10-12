import torch
from torch.nn import Sequential as Seq, Linear as Lin, ReLU, LayerNorm
from torch_scatter import scatter_mean, scatter_add
from torch_geometric.nn.meta import MetaLayer
from torch import nn
import inspect
import yaml
import sys
import deepgate as dg

class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.hidden = nn.Linear(128, 16)
        self.act = nn.ReLU()
        self.output = nn.Linear(16, 1)

    def forward(self, x):
        a = self.act(self.hidden(x))
        return self.output(a)

class ckt_net(nn.Module):
    def __init__(self, graph):
        self.dg_model = dg.Model()

        self.dg_model.load_pretrained()

        self.graph = graph
        self.mlp = MLP()

    def forward(self, x, edge_index, edge_attr, u, v_indices=None, e_indices=None):
        print('[INFO] Get embeddings ...')

        # hs: structural embeddings, hf: functional embeddings
        # hs/hf: [N, D]. N: number of gates, D: embedding dimension (default: 128)
        hs, hf = self.model(self.graph)

        res = self.mlp(hf)

        return res
        
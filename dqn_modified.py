from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from collections import deque


import deepgate
import torch
import os

from deepgatesat.utils import build_argparser, make_env
from deepgatesat.agents import CircuitAgent
from deepgatesat.learners import CircuitLearner

from deepgatesat.ckt_model import ckt_net

from tensorboardX import SummaryWriter

if __name__ == '__main__':
    parser = build_argparser()
    args = parser.parse_args()
    args.device = (
        torch.device("cpu")
        if args.no_cuda or not torch.cuda.is_available()
        else torch.device("cuda")
    )

    writer = SummaryWriter()
    args.logdir = writer.logdir

    model_save_path = os.path.join(args.logdir, "model.yaml")

    env = make_env(args.train_problems_paths, args, test_mode=False)

    aig_path = './aigdata/train/uf50-01.aiger'
    print('[INFO] Parse AIG: ', aig_path)

    parser = deepgate.AigParser()   # Create AigParser
    graph = parser.read_aiger(aig_path) # Parse AIG into Graph

    net = ckt_net(graph)

    agent = CircuitAgent(net, args)
    learner = CircuitLearner(net, args)

    n_trans = 0
    ep = 0
    
    ep_step = 0

    step_ctr = 0
    batch_updates = 5000
    
    while step_ctr < batch_updates:
        ret = 0
        obs = env.reset(args.train_max_time_decisions_allowed)
        done = env.isSolved

        hist_buffer = deque(maxlen=args.history_len)
        for _ in range(args.history_len):
            hist_buffer.append(obs)

        while not done:
            action = agent.act(graph)
            next_obs, r, done, _ = env.step(action)

            obs = next_obs
            hist_buffer.append(obs)

            ret += r

            n_trans += 1
            ep_step += 1

        print(f"Episode {ep + 1}: Return {ret}.")

        ep += 1
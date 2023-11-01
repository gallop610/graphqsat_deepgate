from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from collections import deque


import deepgate
import torch
import os
import copy

from deepgatesat.utils import build_argparser, make_env, evaluate
from deepgatesat.agents import CircuitAgent
from deepgatesat.learners import CircuitLearner
from deepgatesat.buffer import CircuitBuffer

from deepgatesat.ckt_model import ckt_net
import pickle
import yaml
import numpy as np

from tensorboardX import SummaryWriter

def save_training_state(
    model,
    learner,
    episodes_done,
    transitions_seen,
    best_eval_so_far,
    args,
    model_to_best_eval,
    in_eval_mode=False,
):
    # save the model
    model_path = os.path.join(args.logdir, f"model_{learner.step_ctr}.chkp")
    torch.save(model.state_dict(), model_path)

    # save the experience replay
    buffer_path = os.path.join(args.logdir, "buffer.pkl")

    with open(buffer_path, "wb") as f:
        pickle.dump(learner.buffer, f)

    # save important parameters
    train_status = {
        "step_ctr": learner.step_ctr,
        "latest_model_name": model_path,
        "buffer_path": buffer_path,
        "args": args,
        "episodes_done": episodes_done,
        "logdir": args.logdir,
        "transitions_seen": transitions_seen,
        "optimizer_state_dict": learner.optimizer.state_dict(),
        "optimizer_class": type(learner.optimizer),
        "best_eval_so_far": best_eval_so_far,
        "scheduler_class": type(learner.lr_scheduler),
        "scheduler_state_dict": learner.lr_scheduler.state_dict(),
        "in_eval_mode": in_eval_mode,
    }
    
    
    if not learner.step_ctr % args.eval_freq: 
        model_to_best_eval[learner.step_ctr] = best_eval_so_far
        with open(args.model_to_best_eval_path, "wb") as f:
            pickle.dump(model_to_best_eval, f)
        
    status_path = os.path.join(args.logdir, "status.yaml")

    with open(status_path, "w") as f:
        yaml.dump(train_status, f, default_flow_style=False)

    return status_path

def get_annealed_eps(n_trans, args):
    if n_trans < args.init_exploration_steps:
        return args.eps_init
    if n_trans > args.eps_decay_steps:
        return args.eps_final
    else:
        assert n_trans - args.init_exploration_steps >= 0
        return (args.eps_init - args.eps_final) * (
            1 - (n_trans - args.init_exploration_steps) / args.eps_decay_steps
        ) + args.eps_final

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

    model_to_best_eval = {}

    model_save_path = os.path.join(args.logdir, "model.yaml")
    best_eval_so_far = (
        {args.eval_problems_paths: -1}
        if not args.eval_separately_on_each
        else {k: -1 for k in args.eval_problems_paths.split(":")}
    )

    env = make_env(args.train_problems_paths, args, test_mode=False)

    net = ckt_net()

    target_net = copy.deepcopy(net)

    buffer = CircuitBuffer(args, args.buffer_size)

    agent = CircuitAgent(net, args)
    learner = CircuitLearner(net, target_net, buffer, args)

    n_trans = 0
    ep = 0

    batch_updates = 1000000000
    
    while learner.step_ctr < batch_updates:
        ret = 0
        obs = env.reset(args.train_max_time_decisions_allowed)

        done = env.isSolved

        hist_buffer = deque(maxlen=args.history_len)
        for _ in range(args.history_len):
            hist_buffer.append(obs)

        ep_step = 0

        eval_resume_signal = False

        save_flag = False

        while not done:
            annealed_eps = get_annealed_eps(n_trans, args)
            action = agent.act(hist_buffer, annealed_eps)
            next_obs, r, done, _ = env.new_step(action)
            buffer.add_transition(obs, action, r, done)
            obs = next_obs

            hist_buffer.append(obs)
            ret += r

            if (not n_trans % args.step_freq) and (
                buffer.ctr > max(args.init_exploration_steps, args.batch_size + 1)
                or buffer.full
            ):
                step_info = learner.step()
                if annealed_eps is not None:
                    step_info["annealed_eps"] = annealed_eps

                if (not learner.step_ctr % args.eval_freq) or eval_resume_signal:
                    scores, _, eval_resume_signal = evaluate(agent, args, include_train_set=False)
                    for sc_key, sc_val in scores.items():
                        if len(sc_val) > 0:
                            res_vals = [el for el in sc_val.values()]
                            median_score = np.nanmedian(res_vals)
                            if best_eval_so_far[sc_key] < median_score or best_eval_so_far[sc_key] == -1:
                                best_eval_so_far[sc_key] = median_score

                if not learner.step_ctr % args.save_freq:
                    # save the exact model you evaluated and make another save after the episode ends
                    # to have proper transitions in the replay buffer to pickle
                    status_path = save_training_state(
                        net,
                        learner,
                        ep - 1,
                        n_trans,
                        best_eval_so_far,
                        args,
                        model_to_best_eval,
                        in_eval_mode=eval_resume_signal
                    )
                    save_flag = True

            n_trans += 1
            ep_step += 1

        print(f"Episode {ep + 1}: Return {ret}.")

        ep += 1
    
        if save_flag:
            status_path = save_training_state(
                net,
                learner,
                ep - 1,
                n_trans,
                best_eval_so_far,
                args,
                model_to_best_eval,
                in_eval_mode=eval_resume_signal
            )
            save_flag = False
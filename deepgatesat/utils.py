import gym, minisat

import argparse
import time
import torch
import os
import numpy as np

def build_argparser():
    parser = argparse.ArgumentParser()

    parser.add_argument("--eval_separately_on_each", dest="eval_separately_on_each", action="store_true")
    parser.add_argument("--no_eval_separately_on_each", dest="eval_separately_on_each", action="store_false")
    parser.set_defaults(eval_separately_on_each=True)

    parser.add_argument("--eval_problems_paths", default='/home/zc/projects/graphqsat_deepgate/aigdata/eval', type=str)
    parser.add_argument("--eval_freq", default=5, type=int)
    parser.add_argument("--test_time_max_decisions_allowed", default=500, type=int)

    parser.add_argument("--lr_scheduler_frequency", default=1000, type=int)
    parser.add_argument("--lr_scheduler_gamma", default=1.0, type=float)

    parser.add_argument("--step_freq", default=4, type=int)
    parser.add_argument("--init_exploration-steps", default=100, type=int)
    parser.add_argument("--batch_size", default=4, type=int)

    parser.add_argument("--penalty_size", default=0.1, type=float)

    parser.add_argument("--aig_dir", default='/home/zc/projects/graphqsat_deepgate/aigdata/train', type=str)
    parser.add_argument("--cnf_dir", default='./cnf/', type=str)
    parser.add_argument("--tmp_dir", default='./tmp', type=str)

    parser.add_argument("--with_restarts", action="store_true", dest="with_restarts")
    parser.add_argument("--no_restarts", action="store_false", dest="with_restarts")
    parser.set_defaults(with_restarts=False)

    parser.add_argument("--compare_with_restarts", action="store_true", dest="compare_with_restarts")
    parser.add_argument( "--compare_no_restarts", action="store_false", dest="compare_with_restarts")
    parser.set_defaults(compare_with_restarts=False)

    parser.add_argument("--train_max_time_decisions_allowed", type=int, default=500)

    parser.add_argument("--buffer-size", type=int, default=20000)

    parser.add_argument("--env_name", type=str, default="sat-v0")

    parser.add_argument("--max_cap_fill_buffer", dest="max_cap_fill_buffer", action="store_true")

    parser.add_argument("--no_max_cap_fill_buffer", dest="max_cap_fill_buffer",action="store_false")

    parser.set_defaults(max_cap_fill_buffer=False)

    parser.add_argument("--train-problems-paths", type=str, default="./aigdata/uf50-218-tvt/train")

    parser.add_argument("--eval-problems-paths", type=str, default="./aigdata/uf50-218-tvt/eval-problems-paths")

    parser.add_argument("--grad_clip", type=float, default=1.0)

    parser.add_argument("--grad_clip_norm_type", type=int, default=2)

    parser.add_argument("--batch_updates", type=int, default=1000000000)

    parser.add_argument("--history_len", type=int, default=1)

    parser.add_argument("--no_cuda", action="store_true", help="Use the cpu")

    parser.add_argument("--input_type", type=str, default="ckt")

    return parser

def make_env(problems_paths, args, test_mode = False):
    max_data_limit_per_set = None

    if test_mode and hasattr(args, "test_max_data_limit_per_set"):
        max_data_limit_per_set = args.test_max_data_limit_per_set
    if not test_mode and hasattr(args, "train_max_data_limit_per_set"):
        max_data_limit_per_set = args.train_max_data_limit_per_set

    return gym.make(
        args.env_name,
        problems_paths=problems_paths,
        args=args,
        test_mode=test_mode,
        max_cap_fill_buffer=False if test_mode else args.max_cap_fill_buffer,
        penalty_size=args.penalty_size if hasattr(args, "penalty_size") else None,
        with_restarts=args.with_restarts if hasattr(args, "with_restarts") else None,
        compare_with_restarts=args.compare_with_restarts
        if hasattr(args, "compare_with_restarts")
        else None,
        max_data_limit_per_set=max_data_limit_per_set,
    )

def evaluate(agent, args, include_train_set=False):
    agent.net.eval()
    problem_sets = (
        [args.eval_problems_paths]
        if not args.eval_separately_on_each
        else [k for k in args.eval_problems_paths.split(":")]
    )
    if include_train_set:
        problem_sets.extend(
            [args.train_problems_paths]
            if not args.eval_separately_on_each
            else [k for k in args.train_problems_paths.split(":")]
        )
    
    res = {}

    st_time = time.time()
    print("Starting evaluation.")

    total_iter_ours = 0
    total_iter_minisat = 0

    for pset in problem_sets:
        eval_env = make_env(pset, args, test_mode=True)
        DEBUG_ROLLOUTS = None
        pr = 0
        walltime = {}
        propagations = {}
        scores = {}
        with torch.no_grad():
            while eval_env.test_to != 0 or pr == 0:
                p_st_time = time.time()
                obs = eval_env.reset(max_decisions_cap=args.test_time_max_decisions_allowed)
                done = eval_env.isSolved

                while not done:
                    action = agent.act([obs])
                    obs, _, done, _ = eval_env.new_step(action)
                
                walltime[eval_env.curr_problem] = time.time() - p_st_time
                propagations[eval_env.curr_problem] = int(eval_env.S.getPropagations() / eval_env.step_ctr)

                sctr = 1 if eval_env.step_ctr == 0 else eval_env.step_ctr
                ns = eval_env.normalized_score(sctr, eval_env.curr_problem)
                print(f"Evaluation episode {pr+1} is over. Your score is {ns}.")
                total_iters_ours += sctr
                pdir, pname = os.path.split(eval_env.curr_problem)
                total_iters_minisat += eval_env.metadata[pdir][pname][1]
                scores[eval_env.curr_problem] = ns
                pr += 1
                if DEBUG_ROLLOUTS is not None and pr >= DEBUG_ROLLOUTS:
                    break
        print(
            f"Evaluation is done. Median relative score: {np.nanmedian([el for el in scores.values()]):.2f}, "
            f"mean relative score: {np.mean([el for el in scores.values()]):.2f}, "
            f"iters frac: {total_iters_minisat/total_iters_ours:.2f}"
        )
        res[pset] = scores

        agent.net.train()

        return (
            res,
            {
                "metadata": eval_env.metadata,
                "iters_frac": total_iters_minisat / total_iters_ours,
                "mean_score": np.mean([el for el in scores.values()]),
                "median_score": np.median([el for el in scores.values()])
            },
            False,
        )
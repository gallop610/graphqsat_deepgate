import gym, minisat

import argparse

def build_argparser():
    parser = argparse.ArgumentParser()

    parser.add_argument("--train-problems-paths", type=str, default="./aigdata/uf50-218-tvt/train")

    parser.add_argument("--eval-problems-paths", type=str, default="./aigdata/uf50-218-tvt/eval-problems-paths")

    parser.add_argument("--grad_clip", type=float, default=1.0)

    parser.add_argument("--grad_clip_norm_type", type=int, default=2)

    parser.add_argument("--batch_updates", type=int, default=1000000000)

    parser.add_argument("--history_len", type=int, default=1)

    parser.add_argument("--no_cuda", action="store_true", help="Use the cpu")

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
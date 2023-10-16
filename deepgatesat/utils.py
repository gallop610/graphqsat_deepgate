import gym, minisat

import argparse

def build_argparser():
    parser = argparse.ArgumentParser()

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

    parser.add_argument("--penalty_size", type=float, default=0.0001)

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